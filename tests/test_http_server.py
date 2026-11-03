# Copyright 2025 Pex project contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from __future__ import absolute_import, print_function

import json
import os
import socket
import subprocess
import sys
import threading
import time
from uuid import uuid4

import pytest

from pex.docs.command import HtmlDocsConfig, serve_html_docs
from pex.http import server as server_module
from pex.http.server import (
    LaunchError,
    PidStatus,
    Pidfile,
    Server,
    ServerInfo,
)
from pex.os import is_alive, process_start_time
from pex.result import Error
from pex.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator


@pytest.fixture
def doc_root(tmpdir):
    # type: (Any) -> str
    doc_root = tmpdir.join("html")
    os.mkdir(doc_root)
    with open(os.path.join(doc_root, "index.html"), "w") as fp:
        fp.write("<h1>docs</h1>")
    return doc_root


@pytest.fixture
def cache_dir(tmpdir):
    # type: (Any) -> str
    cache_dir = tmpdir.join("cache")
    os.mkdir(cache_dir)
    return cache_dir


@pytest.fixture
def server(cache_dir):
    # type: (str) -> Iterator[Server]
    test_server = Server(name="test docs HTTP server", cache_dir=cache_dir)
    yield test_server
    test_server.shutdown()


def _write_pidfile(
    cache_dir,  # type: str
    server_info,  # type: ServerInfo
):
    # type: (...) -> None
    with open(Pidfile.pidfile_path(cache_dir), "w") as fp:
        json.dump(
            dict(
                url=server_info.url,
                pid=server_info.pid,
                token=server_info.token,
                start_time=server_info.start_time,
                log=server_info.log,
            ),
            fp,
        )


def _sleeping_process():
    # type: () -> subprocess.Popen
    return subprocess.Popen(
        args=[sys.executable, "-c", "import time; time.sleep(120)"]
    )


def _wait_for(predicate, timeout=10.0, interval=0.05):
    # type: (...) -> bool
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_launch_publishes_generation_record(server, doc_root, cache_dir):
    # type: (Server, str, str) -> None
    result = server.launch(doc_root, port=0, timeout=15.0)
    assert not result.already_running
    assert result.server_info.token
    assert is_alive(result.server_info.pid)

    record = Pidfile.load(server.name, cache_dir)
    assert record is not None
    assert record.server_info == result.server_info
    assert record.is_current()

    # The start time captured at launch identifies the same operating system process.
    actual_start_time = process_start_time(result.server_info.pid)
    if actual_start_time is not None:
        assert result.server_info.start_time == actual_start_time


def test_launch_reuses_running_instance(server, doc_root):
    # type: (Server, str) -> None
    first = server.launch(doc_root, port=0, timeout=15.0)
    second = server.launch(doc_root, port=0, timeout=15.0)
    assert second.already_running
    assert second.server_info == first.server_info


def test_concurrent_launch_starts_one_instance(server, doc_root):
    # type: (Server, str) -> None
    results = []  # type: list
    errors = []  # type: list
    barrier = threading.Barrier(2)

    def launch():
        # type: () -> None
        try:
            barrier.wait()
            results.append(server.launch(doc_root, port=0, timeout=15.0))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=launch) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
        assert not thread.is_alive()

    assert not errors
    assert len(results) == 2
    assert {result.server_info.pid for result in results} == {results[0].server_info.pid}
    assert sorted(result.already_running for result in results) == [False, True]


def test_concurrent_launch_from_separate_processes(doc_root, cache_dir):
    # type: (str, str) -> None

    # Simulates running `pex3 docs --no-open` in two terminals at once: exactly one competing
    # caller creates the process, the other waits under the lock and reuses the instance.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    child_script = """
import json, os, sys, time
sys.path.insert(0, {repo_root!r})
from pex.http.server import Server
barrier = os.path.join(os.environ["CHILD_CACHE"], "barrier")
if not os.path.isdir(barrier):
    try:
        os.makedirs(barrier)
    except OSError:
        pass
with open(os.path.join(barrier, str(os.getpid())), "w"):
    pass
deadline = time.time() + 15
while len(os.listdir(barrier)) < 2 and time.time() < deadline:
    time.sleep(0.02)
result = Server(name="child docs HTTP server", cache_dir=os.environ["CHILD_CACHE"]).launch(
    os.environ["CHILD_DOC_ROOT"], port=0, timeout=15.0
)
sys.stdout.write(json.dumps({{"pid": result.server_info.pid,
                             "url": result.server_info.url,
                             "already_running": result.already_running}}))
""".format(repo_root=repo_root)

    env = dict(os.environ)
    env["CHILD_CACHE"] = cache_dir
    env["CHILD_DOC_ROOT"] = doc_root

    children = [
        subprocess.Popen(
            args=[sys.executable, "-c", child_script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    try:
        outcomes = []
        for child in children:
            stdout, stderr = child.communicate(timeout=60)
            assert child.returncode == 0, stderr.decode("utf-8")
            outcomes.append(json.loads(stdout.decode("utf-8")))

        assert outcomes[0]["pid"] == outcomes[1]["pid"]
        assert outcomes[0]["url"] == outcomes[1]["url"]
        assert sorted(outcome["already_running"] for outcome in outcomes) == [False, True]
    finally:
        Server(name="child docs HTTP server", cache_dir=cache_dir).shutdown()
        for child in children:
            if child.poll() is None:
                child.kill()


def test_running_reports_only_verified_instance(server, doc_root, cache_dir):
    # type: (Server, str, str) -> None
    assert server.running() is None

    result = server.launch(doc_root, port=0, timeout=15.0)
    running = server.running()
    assert running is not None
    assert running.server_info == result.server_info

    server.shutdown()
    assert server.running() is None

    # A corrupt record is never reported as a running instance.
    with open(Pidfile.pidfile_path(cache_dir), "w") as fp:
        fp.write("{not valid json")
    assert server.running() is None


def test_shutdown_terminates_process_and_removes_record(server, doc_root, cache_dir):
    # type: (Server, str, str) -> None
    result = server.launch(doc_root, port=0, timeout=15.0)
    pid = result.server_info.pid

    shutdown_info = server.shutdown(timeout=15.0)
    assert shutdown_info == result.server_info
    assert _wait_for(lambda: not is_alive(pid))
    assert not os.path.exists(Pidfile.pidfile_path(cache_dir))

    # Idempotent: no instance -> None.
    assert server.shutdown() is None


def test_shutdown_waits_for_exit_before_removing_record(server, doc_root, cache_dir):
    # type: (Server, str, str) -> None
    result = server.launch(doc_root, port=0, timeout=15.0)
    assert server.shutdown(timeout=15.0) is not None
    assert not is_alive(result.server_info.pid)
    assert not os.path.exists(Pidfile.pidfile_path(cache_dir))


def test_corrupt_pidfile_is_reclaimed_on_launch(server, doc_root, cache_dir):
    # type: (Server, str, str) -> None
    with open(Pidfile.pidfile_path(cache_dir), "w") as fp:
        fp.write("garbage")

    result = server.launch(doc_root, port=0, timeout=15.0)
    assert not result.already_running
    assert Pidfile.load(server.name, cache_dir).is_current()


def test_dead_pid_record_is_reclaimed(server, doc_root):
    # type: (Server, str) -> None
    dead = subprocess.Popen(args=[sys.executable, "-c", "import sys; sys.exit(0)"])
    dead.wait()

    _write_pidfile(
        server.cache_dir,
        ServerInfo(
            url="http://localhost:1", pid=dead.pid, token=uuid4().hex, start_time="1"
        ),
    )
    assert Pidfile.load(server.name, server.cache_dir).status() is PidStatus.DEAD

    result = server.launch(doc_root, port=0, timeout=15.0)
    assert not result.already_running
    assert result.server_info.pid != dead.pid


def test_foreign_pid_is_reclaimed_but_never_signaled(server, doc_root):
    # type: (Server, str) -> None
    foreign = _sleeping_process()
    try:
        _wait_for(lambda: is_alive(foreign.pid))
        _write_pidfile(
            server.cache_dir,
            ServerInfo(
                url="http://localhost:1",
                pid=foreign.pid,
                token=uuid4().hex,
                start_time="a-start-time-that-cannot-match",
            ),
        )
        record = Pidfile.load(server.name, server.cache_dir)
        assert record.status() is PidStatus.FOREIGN
        assert server.running() is None

        result = server.launch(doc_root, port=0, timeout=15.0)
        assert result.server_info.pid != foreign.pid
        assert is_alive(foreign.pid), "A process that recycled the recorded PID must not be killed"

        # Shutdown must likewise leave the foreign process alone.
        assert server.shutdown() is not None
        assert is_alive(foreign.pid)
    finally:
        if foreign.poll() is None:
            foreign.terminate()
            foreign.wait()


def test_unreachable_own_instance_is_restarted(server, doc_root):
    # type: (Server, str) -> None
    defunct = _sleeping_process()
    try:
        _wait_for(lambda: is_alive(defunct.pid))
        start_time = process_start_time(defunct.pid)
        if start_time is None:
            pytest.skip("Cannot determine process start time on this platform.")
        _write_pidfile(
            server.cache_dir,
            ServerInfo(
                url="http://localhost:1",
                pid=defunct.pid,
                token=uuid4().hex,
                start_time=start_time,
            ),
        )
        record = Pidfile.load(server.name, server.cache_dir)
        assert record.status() is PidStatus.UNREACHABLE

        result = server.launch(doc_root, port=0, timeout=15.0)
        assert not result.already_running
        assert _wait_for(lambda: not is_alive(defunct.pid)), (
            "The defunct recorded instance must be terminated before retrying"
        )
        assert server.running().server_info.pid == result.server_info.pid
    finally:
        if defunct.poll() is None:
            defunct.terminate()
            defunct.wait()


def test_unverifiable_live_pid_is_never_signaled(server, doc_root):
    # type: (Server, str) -> None
    # A legacy-style record (no start time) for a live non-serving process cannot be proven ours.
    unknown = _sleeping_process()
    try:
        _wait_for(lambda: is_alive(unknown.pid))
        _write_pidfile(
            server.cache_dir,
            ServerInfo(
                url="http://localhost:1", pid=unknown.pid, token=uuid4().hex, start_time=None
            ),
        )
        record = Pidfile.load(server.name, server.cache_dir)
        assert record.status() is PidStatus.UNKNOWN
        assert server.running() is None

        result = server.launch(doc_root, port=0, timeout=15.0)
        assert result.server_info.pid != unknown.pid
        assert is_alive(unknown.pid), "An unverifiable live process must not be terminated"
        assert server.shutdown() is not None
        assert is_alive(unknown.pid)
    finally:
        if unknown.poll() is None:
            unknown.terminate()
            unknown.wait()


def test_generation_guarded_discard(cache_dir):
    # type: (str) -> None
    info = ServerInfo(url="http://localhost:1", pid=1, token=uuid4().hex, start_time="1")
    _write_pidfile(cache_dir, info)

    # An older generation must not be able to delete a newer generation's record.
    assert os.path.exists(Pidfile.pidfile_path(cache_dir))
    assert not Pidfile.discard("test", cache_dir, "some-older-generation")
    assert os.path.exists(Pidfile.pidfile_path(cache_dir))

    # Only the current generation can remove the record.
    assert Pidfile.discard("test", cache_dir, info.token)
    assert not os.path.exists(Pidfile.pidfile_path(cache_dir))


def test_shutdown_old_generation_does_not_affect_new_generation(server, doc_root, cache_dir):
    # type: (Server, str, str) -> None
    first = server.launch(doc_root, port=0, timeout=15.0)
    old_record = Pidfile.load(server.name, cache_dir)

    # Simulate a new generation publishing while an older shutdown call is in flight: the old
    # caller's compare-and-delete must leave the new record untouched.
    server.shutdown(timeout=15.0)
    assert _wait_for(lambda: not is_alive(first.server_info.pid))

    second = server.launch(doc_root, port=0, timeout=15.0)
    assert second.server_info.pid != first.server_info.pid
    assert not Pidfile.discard(server.name, cache_dir, old_record.server_info.token)

    current = Pidfile.load(server.name, cache_dir)
    assert current.server_info.token == second.server_info.token
    assert is_alive(second.server_info.pid)


def test_busy_port_failure_leaves_no_record_but_keeps_log_and_recovers(server, doc_root, cache_dir):
    # type: (Server, str, str) -> None
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    busy_port = listener.getsockname()[1]

    try:
        with pytest.raises(LaunchError) as exc_info:
            server.launch(doc_root, port=busy_port, timeout=5.0)
        # No fake "running" record is left behind...
        assert not os.path.exists(Pidfile.pidfile_path(cache_dir))
        # ...but the generated log is retained and locatable via the error.
        assert os.path.exists(exc_info.value.log)
        assert exc_info.value.log in str(exc_info.value)
    finally:
        listener.close()

    # Recovery: the next launch succeeds and leaves a verifiable record.
    result = server.launch(doc_root, port=0, timeout=15.0)
    assert not result.already_running
    assert server.running() is not None


def test_publish_failure_aborts_server_and_leaves_no_record(
    monkeypatch, server, doc_root, cache_dir
):
    # type: (Any, Server, str, str) -> None

    published = []  # type: list

    def failing_publish(cache_dir_arg, server_info):
        # type: (str, ServerInfo) -> Pidfile
        published.append(server_info)
        raise OSError("simulated state write failure")

    monkeypatch.setattr(server_module.Pidfile, "publish", staticmethod(failing_publish))

    with pytest.raises(LaunchError):
        server.launch(doc_root, port=0, timeout=15.0)

    assert len(published) == 1
    assert not os.path.exists(Pidfile.pidfile_path(cache_dir))
    assert _wait_for(lambda: not is_alive(published[0].pid))

    # A later launch recovers once state writes succeed again.
    monkeypatch.undo()
    result = server.launch(doc_root, port=0, timeout=15.0)
    assert not result.already_running
    assert server.running() is not None


def test_serve_html_docs_without_embedded_docs(monkeypatch):
    # type: (Any) -> None
    from pex.docs import command as docs_command

    monkeypatch.setattr(docs_command.docs, "root", lambda doc_type="html": None)
    result = serve_html_docs(open_browser=False, config=HtmlDocsConfig())
    assert isinstance(result, Error)
    message = str(result)
    assert "does not include embedded docs" in message
    assert "https://docs.pex-tool.org" in message


def test_serve_html_docs_falls_back_to_ephemeral_port(
    monkeypatch, server, doc_root, cache_dir
):
    # type: (Any, Server, str, str) -> None
    from pex.docs import command as docs_command

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    busy_port = listener.getsockname()[1]

    try:
        monkeypatch.setattr(docs_command.docs, "root", lambda doc_type="html": doc_root)
        monkeypatch.setattr(docs_command, "server", server)
        monkeypatch.setattr(docs_command, "STANDARD_PORT", busy_port)

        result = serve_html_docs(open_browser=False, config=HtmlDocsConfig())
        assert not isinstance(result, Error), result
        actual_port = int(result.server_info.url.rsplit(":", 1)[1])
        assert actual_port != busy_port
        assert is_alive(result.server_info.pid)
    finally:
        listener.close()


if TYPE_CHECKING:
    from typing import Any
