# Copyright 2026 Pex project contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from __future__ import print_function

import os
import subprocess
import sys
import time

from pex.atomic_sync import PHASE_COMMITTING, SyncPublication
from pex.common import safe_mkdir
from pex.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


def _state_dir(lock_path):
    # type: (str) -> str
    return os.path.join(
        os.path.dirname(lock_path),
        ".{name}.pex-sync".format(name=os.path.basename(lock_path)),
    )


def _write(path, content):
    # type: (str, str) -> None
    safe_mkdir(os.path.dirname(path) or ".")
    with open(path, "w") as fp:
        fp.write(content)


def _read(path):
    # type: (str) -> str
    with open(path) as fp:
        return fp.read()


def _pending_paths(publication):
    # type: (SyncPublication) -> list
    state_dir = _state_dir(publication.lock_path)
    if not os.path.isdir(state_dir):
        return []
    return [
        os.path.join(state_dir, name)
        for name in os.listdir(state_dir)
        if name.startswith("pending.") and name.endswith(".json")
    ]


def _stage_lock(publication, content):
    # type: (SyncPublication, str) -> None
    _write(publication.lock_candidate_path, content)


def _stage_venv(publication, marker):
    # type: (SyncPublication, str) -> None
    os.mkdir(publication.venv_candidate_dir)
    _write(os.path.join(publication.venv_candidate_dir, "marker"), marker)


def _assert_clean_siblings(path):
    # type: (str) -> None
    parent = os.path.dirname(path)
    leftovers = [
        name
        for name in os.listdir(parent)
        if name.endswith(".pex-sync-work") or name.endswith(".pex-sync-old")
    ]
    assert [] == leftovers, leftovers


def test_lock_only_create_update(tmpdir):
    # type: (Tempdir) -> None

    lock = tmpdir.join("lock.json")

    publication = SyncPublication(lock)
    publication.acquire()
    publication.staging()
    _stage_lock(publication, "v1")
    publication.prepared()
    generation = publication.commit()
    publication.release()

    assert "v1" == _read(lock)
    assert generation.lock_sha256
    assert [] == _pending_paths(publication)
    assert generation.generation == publication.committed_generation().generation

    publication = SyncPublication(lock)
    publication.acquire()
    publication.staging()
    _stage_lock(publication, "v2")
    publication.prepared()
    publication.commit()
    publication.release()

    assert "v2" == _read(lock)
    _assert_clean_siblings(lock)


def test_lock_only_noop(tmpdir):
    # type: (Tempdir) -> None

    lock = tmpdir.join("lock.json")
    _write(lock, "v0")

    publication = SyncPublication(lock)
    publication.acquire()
    publication.staging()
    # No candidate lock is staged: nothing to publish.
    publication.abort()
    publication.release()

    assert "v0" == _read(lock)
    assert [] == _pending_paths(publication)


def test_pair_first_create(tmpdir):
    # type: (Tempdir) -> None

    lock = tmpdir.join("lock.json")
    venv = tmpdir.join("venv")

    publication = SyncPublication(lock, venv)
    publication.acquire()
    publication.staging(venv_existed=False)
    _stage_lock(publication, "v1")
    _stage_venv(publication, "venv1")
    publication.prepared()
    publication.commit()
    publication.release()

    assert "v1" == _read(lock)
    assert "venv1" == _read(os.path.join(venv, "marker"))
    _assert_clean_siblings(venv)

    generation = SyncPublication(lock, venv).committed_generation()
    assert generation is not None
    assert os.path.realpath(venv) == generation.venv_dir


def test_existing_venv_switch(tmpdir):
    # type: (Tempdir) -> None

    lock = tmpdir.join("lock.json")
    venv = tmpdir.join("venv")
    _write(lock, "v1")
    os.mkdir(venv)
    _write(os.path.join(venv, "marker"), "venv1")

    publication = SyncPublication(lock, venv)
    publication.acquire()
    publication.staging(venv_existed=True)
    _stage_lock(publication, "v2")
    _stage_venv(publication, "venv2")

    # Before the switch neither published resource has changed and a backup has not been taken.
    assert "v1" == _read(lock)
    assert "venv1" == _read(os.path.join(venv, "marker"))

    publication.prepared()
    publication.commit()
    publication.release()

    assert "v2" == _read(lock)
    assert "venv2" == _read(os.path.join(venv, "marker"))
    _assert_clean_siblings(venv)


def test_abort_keeps_resources(tmpdir):
    # type: (Tempdir) -> None

    lock = tmpdir.join("lock.json")
    venv = tmpdir.join("venv")
    _write(lock, "v1")
    os.mkdir(venv)
    _write(os.path.join(venv, "marker"), "venv1")

    publication = SyncPublication(lock, venv)
    publication.acquire()
    publication.staging(venv_existed=True)
    _stage_lock(publication, "v2-broken")
    _stage_venv(publication, "venv2-broken")
    # A resolve / install / verification failure aborts the generation.
    publication.abort()
    publication.release()

    assert "v1" == _read(lock)
    assert "venv1" == _read(os.path.join(venv, "marker"))
    assert [] == _pending_paths(publication)
    _assert_clean_siblings(venv)


def test_failed_create_clean(tmpdir):
    # type: (Tempdir) -> None

    lock = tmpdir.join("lock.json")
    venv = tmpdir.join("venv")

    publication = SyncPublication(lock, venv)
    publication.acquire()
    publication.staging(venv_existed=False)
    _stage_lock(publication, "v1")
    _stage_venv(publication, "venv1")
    publication.abort()
    publication.release()

    assert not os.path.exists(lock)
    assert not os.path.exists(venv)
    assert [] == _pending_paths(publication)
    _assert_clean_siblings(lock)
    _assert_clean_siblings(venv)


def test_staging_leftovers(tmpdir):
    # type: (Tempdir) -> None

    lock = tmpdir.join("lock.json")
    venv = tmpdir.join("venv")
    _write(lock, "v1")
    os.mkdir(venv)
    _write(os.path.join(venv, "marker"), "venv1")

    dead = SyncPublication(lock, venv)
    dead.acquire()
    dead.staging(venv_existed=True)
    _stage_lock(dead, "v2-never-finished")
    _stage_venv(dead, "venv2-never-finished")
    pending = dead._pending_path
    dead.release()  # The owner process dies; staging state is left on disk.
    del dead

    next_owner = SyncPublication(lock, venv)
    next_owner.acquire()
    next_owner.release()

    assert "v1" == _read(lock)
    assert "venv1" == _read(os.path.join(venv, "marker"))
    assert not os.path.exists(pending)
    _assert_clean_siblings(venv)


def _crash_in_committing(
    tmpdir,  # type: Any
    venv_existed,  # type: bool
    after_stash=False,  # type: bool
    after_venv_publish=False,  # type: bool
):
    lock = tmpdir.join("next-lock.json")
    venv = tmpdir.join("next-venv")
    if venv_existed:
        _write(lock, "old")
        os.mkdir(venv)
        _write(os.path.join(venv, "marker"), "old-venv")

    dead = SyncPublication(lock, venv)
    dead.acquire()
    dead.staging(venv_existed=venv_existed)
    _stage_lock(dead, "new")
    _stage_venv(dead, "new-venv")
    if after_stash:
        assert venv_existed
        os.rename(venv, dead._venv_stash)
    if after_venv_publish:
        if venv_existed:
            os.rename(venv, dead._venv_stash)
        os.rename(dead.venv_candidate_dir, venv)
    dead._phase = PHASE_COMMITTING
    dead._write_pending()
    pending = dead._pending_path
    dead.release()
    del dead

    next_owner = SyncPublication(lock, venv)
    next_owner.acquire()
    try:
        assert "new" == _read(lock)
        assert "new-venv" == _read(os.path.join(venv, "marker"))
        assert not os.path.exists(pending)
        _assert_clean_siblings(venv)
        assert os.path.isdir(venv)
    finally:
        next_owner.release()


def test_recover_after_stash(tmpdir):
    # type: (Any) -> None
    _crash_in_committing(tmpdir, venv_existed=True, after_stash=True)


def test_recover_mid_switch(tmpdir):
    # type: (Any) -> None
    _crash_in_committing(tmpdir, venv_existed=True, after_venv_publish=True)


def test_recover_pre_stash(tmpdir):
    # type: (Any) -> None
    _crash_in_committing(tmpdir, venv_existed=True)


def test_recover_first_create(tmpdir):
    # type: (Any) -> None
    _crash_in_committing(tmpdir, venv_existed=False, after_venv_publish=True)


def test_single_owner_lock(tmpdir):
    # type: (Any) -> None

    lock = tmpdir.join("contended-lock.json")
    events = tmpdir.join("events.log")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PEX_TEST_ROOT"] = repo_root

    worker = (
        "import os, sys, time\n"
        "sys.path.insert(0, os.environ['PEX_TEST_ROOT'])\n"
        "from pex.atomic_sync import SyncPublication\n"
        "lock, events, hold = sys.argv[1], sys.argv[2], float(sys.argv[3])\n"
        "pub = SyncPublication(lock)\n"
        "open(events, 'a').write('waiting ' + str(os.getpid()) + '\\n')\n"
        "pub.acquire()\n"
        "open(events, 'a').write('acquired ' + str(os.getpid()) + ' ' + str(time.time()) + '\\n')\n"
        "time.sleep(hold)\n"
        "pub.staging()\n"
        "with open(pub.lock_candidate_path, 'w') as fp:\n"
        "    fp.write(str(os.getpid()))\n"
        "pub.prepared()\n"
        "pub.commit()\n"
        "pub.release()\n"
        "open(events, 'a').write('released ' + str(os.getpid()) + ' ' + str(time.time()) + '\\n')\n"
    )

    first = subprocess.Popen([sys.executable, "-c", worker, lock, events, "2"], env=env)
    time.sleep(0.5)
    second = subprocess.Popen([sys.executable, "-c", worker, lock, events, "0"], env=env)
    assert 0 == first.wait(30)
    assert 0 == second.wait(30)

    events_by_name = {}  # type: dict
    for line in _read(events).splitlines():
        kind, rest = line.split(" ", 1)
        events_by_name.setdefault(kind, []).append(rest)

    assert 2 == len(events_by_name["acquired"])
    assert 2 == len(events_by_name["released"])

    def timestamp(entry):
        # type: (str) -> float
        return float(entry.rsplit(" ", 1)[1])

    first_pid = events_by_name["acquired"][0].split(" ")[0]
    first_release = timestamp(
        next(entry for entry in events_by_name["released"] if entry.startswith(first_pid))
    )
    second_acquire = timestamp(events_by_name["acquired"][1])
    assert second_acquire >= first_release - 0.05

    # The contender that waited publishes (it recomputed after the owner ended); either way only
    # one lock exists and it is valid.
    assert os.path.exists(lock)
    assert _read(lock) in (first_pid, events_by_name["acquired"][1].split(" ")[0])
