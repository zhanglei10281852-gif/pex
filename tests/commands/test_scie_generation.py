# Copyright 2026 Pex project contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from __future__ import absolute_import

import hashlib
import os
import shutil
import threading

import pex.cli.commands.scie as scie_cli
from pex.cli.commands.scie import (
    _BACKUP_PREFIX,
    _COMMIT_LOCK_NAME,
    _commit_generation,
    _verify_generation,
)
from pex.fs import lock as fs_lock
from pex.result import Error
from pex.scie.model import InterpreterDistribution, Provider, ScieInfo, ScieStyle
from pex.sysconfig import SysPlatform


def _write(path, content):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w") as fp:
        fp.write(content)
    return path


def _read(path):
    with open(path, "r") as fp:
        return fp.read()


def _scie_info(path, platform=None):
    return ScieInfo(
        style=ScieStyle.LAZY,
        interpreter=InterpreterDistribution(
            provider=Provider.PythonBuildStandalone,
            platform=platform or SysPlatform.CURRENT,
            version=(3, 11),
        ),
        file=path,
    )


def _make_candidate(dest_dir, files):
    candidate = os.path.join(str(dest_dir), "candidate")
    os.mkdir(candidate)
    for rel, content in files.items():
        _write(os.path.join(candidate, rel), content)
    return candidate


def _artifacts(candidate, rels):
    return tuple((os.path.join(candidate, rel), rel) for rel in rels)


def _entry_names(dest_dir):
    return set(
        name
        for name in os.listdir(str(dest_dir))
        if name != _COMMIT_LOCK_NAME and not name.startswith(_BACKUP_PREFIX)
    )


def test_commit_whole_set(tmp_path):
    dest = str(tmp_path)
    _write(os.path.join(dest, "app-a"), "old-a")
    _write(os.path.join(dest, "app-a.sha256"), "old-a-hash")
    _write(os.path.join(dest, "app-b"), "old-b")
    _write(os.path.join(dest, "keep-me"), "unrelated")

    candidate = _make_candidate(
        dest,
        {"app-a": "new-a", "app-a.sha256": "new-a-hash", "app-b": "new-b"},
    )

    result = _commit_generation(dest, _artifacts(candidate, ["app-a", "app-a.sha256", "app-b"]))
    assert result is None
    shutil.rmtree(candidate, True)

    assert "new-a" == _read(os.path.join(dest, "app-a"))
    assert "new-a-hash" == _read(os.path.join(dest, "app-a.sha256"))
    assert "new-b" == _read(os.path.join(dest, "app-b"))
    assert "unrelated" == _read(os.path.join(dest, "keep-me"))
    assert {"app-a", "app-a.sha256", "app-b", "keep-me"} == _entry_names(dest)


def test_commit_platform_subdirs(tmp_path):
    dest = str(tmp_path)
    _write(os.path.join(dest, "linux-x86_64", "app"), "old-foreign")
    _write(os.path.join(dest, "app"), "old-current")

    candidate = _make_candidate(
        dest,
        {
            "app": "new-current",
            os.path.join("linux-x86_64", "app"): "new-foreign",
            os.path.join("macos-aarch64", "app"): "new-mac",
        },
    )

    result = _commit_generation(
        dest,
        _artifacts(
            candidate,
            [
                "app",
                os.path.join("linux-x86_64", "app"),
                os.path.join("macos-aarch64", "app"),
            ],
        ),
    )
    assert result is None
    shutil.rmtree(candidate, True)

    assert "new-current" == _read(os.path.join(dest, "app"))
    assert "new-foreign" == _read(os.path.join(dest, "linux-x86_64", "app"))
    assert "new-mac" == _read(os.path.join(dest, "macos-aarch64", "app"))
    assert {"app", "linux-x86_64", "macos-aarch64"} == _entry_names(dest)


def test_commit_rollback_on_failure(tmp_path, monkeypatch):
    dest = str(tmp_path)
    _write(os.path.join(dest, "app-a"), "old-a")
    _write(os.path.join(dest, "app-b"), "old-b")
    _write(os.path.join(dest, "keep-me"), "unrelated")

    candidate = _make_candidate(dest, {"app-a": "new-a", "app-b": "new-b"})

    real_safe_rename = scie_cli.safe_rename

    def failing_safe_rename(src, dst):
        if os.path.dirname(src) == candidate and dst.endswith("app-b"):
            raise OSError("injected commit failure")
        return real_safe_rename(src, dst)

    monkeypatch.setattr(scie_cli, "safe_rename", failing_safe_rename)

    result = _commit_generation(dest, _artifacts(candidate, ["app-a", "app-b"]))
    assert isinstance(result, Error)
    message = str(result)
    assert "commit stage" in message
    assert "app-b" in message

    # The prior generation is fully visible and intact; the new generation is discarded.
    shutil.rmtree(candidate, True)
    assert "old-a" == _read(os.path.join(dest, "app-a"))
    assert "old-b" == _read(os.path.join(dest, "app-b"))
    assert "unrelated" == _read(os.path.join(dest, "keep-me"))
    assert {"app-a", "app-b", "keep-me"} == _entry_names(dest)


def test_scie_only_reclaim(tmp_path, monkeypatch):
    dest = str(tmp_path)

    # Success path: the source PEX is reclaimed together with the new generation.
    _write(os.path.join(dest, "cowsay.pex"), "the-pex")
    candidate = _make_candidate(dest, {"cowsay": "the-scie"})
    result = _commit_generation(
        dest, _artifacts(candidate, ["cowsay"]), reclaim_relpaths=("cowsay.pex",)
    )
    assert result is None
    shutil.rmtree(candidate, True)
    assert "the-scie" == _read(os.path.join(dest, "cowsay"))
    assert not os.path.exists(os.path.join(dest, "cowsay.pex"))
    assert {"cowsay"} == _entry_names(dest)

    # Failure path: the source PEX is restored; no scie is published.
    _write(os.path.join(dest, "cowsay.pex"), "the-pex-2")
    candidate = _make_candidate(dest, {"cowsay": "the-scie-2"})

    real_safe_rename = scie_cli.safe_rename

    def failing_safe_rename(src, dst):
        if os.path.dirname(src) == candidate:
            raise OSError("injected commit failure")
        return real_safe_rename(src, dst)

    monkeypatch.setattr(scie_cli, "safe_rename", failing_safe_rename)
    result = _commit_generation(
        dest, _artifacts(candidate, ["cowsay"]), reclaim_relpaths=("cowsay.pex",)
    )
    assert isinstance(result, Error)
    shutil.rmtree(candidate, True)
    # The failed switch restores the complete prior generation: the first generation's scie and
    # the in-place source PEX are both back.
    assert "the-pex-2" == _read(os.path.join(dest, "cowsay.pex"))
    assert "the-scie" == _read(os.path.join(dest, "cowsay"))
    assert {"cowsay", "cowsay.pex"} == _entry_names(dest)


def test_reclaim_collision_noop(tmp_path):
    dest = str(tmp_path)
    _write(os.path.join(dest, "cowsay"), "old-same-name")
    candidate = _make_candidate(dest, {"cowsay": "new-same-name"})

    result = _commit_generation(
        dest, _artifacts(candidate, ["cowsay"]), reclaim_relpaths=("cowsay",)
    )
    assert result is None
    shutil.rmtree(candidate, True)
    assert "new-same-name" == _read(os.path.join(dest, "cowsay"))
    assert {"cowsay"} == _entry_names(dest)


def test_verify_generation_ok(tmp_path):
    candidate = os.path.join(str(tmp_path), "candidate")
    os.mkdir(candidate)
    scie_path = _write(os.path.join(candidate, "app"), "binary")
    sha256 = hashlib.sha256(b"binary").hexdigest()
    _write(os.path.join(candidate, "app.sha256"), "{hash}  app\n".format(hash=sha256))

    assert _verify_generation((_scie_info(scie_path),), ("sha256",)) is None


def test_verify_missing_or_empty(tmp_path):
    missing = _scie_info(os.path.join(str(tmp_path), "nope"))
    result = _verify_generation((missing,), ())
    assert isinstance(result, Error)
    assert "verification stage" in str(result)
    assert missing.interpreter.render_description() in str(result)

    empty = os.path.join(str(tmp_path), "empty")
    _write(empty, "")
    result = _verify_generation((_scie_info(empty),), ())
    assert isinstance(result, Error)
    assert "0 bytes" in str(result)


def test_verify_bad_sidecars(tmp_path):
    candidate = os.path.join(str(tmp_path), "candidate")
    os.mkdir(candidate)
    scie_path = _write(os.path.join(candidate, "app"), "binary")

    result = _verify_generation((_scie_info(scie_path),), ("sha256",))
    assert isinstance(result, Error)
    assert "sha256" in str(result)
    assert "missing" in str(result)

    _write(os.path.join(candidate, "app.sha256"), "deadbeef  app\n")
    result = _verify_generation((_scie_info(scie_path),), ("sha256",))
    assert isinstance(result, Error)
    assert "does not match" in str(result)
    assert hashlib.sha256(b"binary").hexdigest() in str(result)

    sha256 = hashlib.sha256(b"binary").hexdigest()
    _write(os.path.join(candidate, "app.sha256"), "{hash} *app\n".format(hash=sha256))
    assert _verify_generation((_scie_info(scie_path),), ("sha256",)) is None


def test_concurrent_commit_lock(tmp_path):
    dest = str(tmp_path)
    _write(os.path.join(dest, "app-a"), "old-a")
    _write(os.path.join(dest, "app-b"), "old-b")

    candidate = _make_candidate(dest, {"app-a": "new-a", "app-b": "new-b"})

    # Hold the commit lock externally to stand in for a concurrent publisher mid-switch.
    held_lock = fs_lock.acquire(os.path.join(dest, _COMMIT_LOCK_NAME), exclusive=True)
    errors = []

    def commit():
        try:
            result = _commit_generation(dest, _artifacts(candidate, ["app-a", "app-b"]))
            if result is not None:
                errors.append(result)
        except Exception as e:
            errors.append(e)

    thread = threading.Thread(target=commit)
    thread.start()
    thread.join(timeout=5.0)
    assert thread.is_alive(), "Expected the second publisher to block on the commit lock."

    # While the other publisher's switch is in flight only a complete prior generation is visible.
    assert "old-a" == _read(os.path.join(dest, "app-a"))
    assert "old-b" == _read(os.path.join(dest, "app-b"))

    held_lock.release()
    thread.join(timeout=10.0)
    assert not thread.is_alive()
    assert [] == errors
    shutil.rmtree(candidate, True)

    # The waiter then performs its whole switch against the complete prior generation.
    assert "new-a" == _read(os.path.join(dest, "app-a"))
    assert "new-b" == _read(os.path.join(dest, "app-b"))
    assert {"app-a", "app-b"} == _entry_names(dest)
