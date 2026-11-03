# Copyright 2026 Pex project contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Atomic, generation-based publication of a matching lock file / venv pair.

A sync computes a lock, resolves and installs a venv and finally publishes *both* results. The
two published resources are only ever useful as a matching pair: a new lock against an old (or
half-installed) venv is a corrupt state for the next consumer (tox, nox, CI, ...).

To avoid that, a :class:`SyncPublication` drives the whole operation as a single recoverable
generation:

1. The caller acquires the *publication owner* lock for the lock file path. All concurrent syncs
   for the same lock (and thus its paired venv) block on this one lock; a process that dies
   releases it automatically.
2. Any interrupted generation from a dead owner is recovered before new work begins.
3. The new lock is written to a sibling candidate file and the new venv is built in a sibling
   candidate directory (a fresh create, or a hardlinked clone of the existing venv mutated in
   place). Neither published path is touched during this phase.
4. Once both candidates are complete and validated, a ``committing`` marker is fsynced and the
   switch is performed with only renames, in a deterministic order: first move the existing venv
   aside (if any), then publish the venv, then publish the lock. The lock file - the resource
   consumers consult first - is thus always the last thing to change and always points at a venv
   that is already in place.
5. The committed generation marker is published and all backups / candidates / pending state are
   removed.

A crash at any point leaves the previously committed resources usable and lets the next owner
either roll the interrupted generation forward (all candidates were complete when the switch
began) or discard it (the switch never began).
"""

from __future__ import absolute_import

import errno
import hashlib
import json
import os
import shutil
import threading
from uuid import uuid4

from pex import pex_warnings
from pex.common import safe_delete, safe_mkdir, safe_rmtree
from pex.fs import lock as fs_lock
from pex.fs import safe_rename
from pex.typing import TYPE_CHECKING
from pex.version import __version__

if TYPE_CHECKING:
    from typing import Dict, Optional

    import attr  # vendor:skip
else:
    from pex.third_party import attr


PHASE_STAGING = "staging"
PHASE_COMMITTING = "committing"

WORK_SUFFIX = ".pex-sync-work"
OLD_SUFFIX = ".pex-sync-old"


@attr.s(frozen=True)
class Generation(object):
    """A committed lock / venv generation."""

    @classmethod
    def load(cls, path):
        # type: (str) -> Optional[Generation]
        if not os.path.exists(path):
            return None
        with open(path, "r") as fp:
            data = json.load(fp)
        return cls(
            generation=data["generation"],
            lock_path=data["lock"],
            venv_dir=data.get("venv"),
            lock_sha256=data.get("lock_sha256"),
            pex_version=data.get("pex_version"),
        )

    generation = attr.ib()  # type: str
    lock_path = attr.ib()  # type: str
    venv_dir = attr.ib(default=None)  # type: Optional[str]
    lock_sha256 = attr.ib(default=None)  # type: Optional[str]
    pex_version = attr.ib(default=None)  # type: Optional[str]

    def dump(self, path):
        # type: (str) -> None
        _atomic_dump_json(
            path,
            {
                "generation": self.generation,
                "lock": self.lock_path,
                "venv": self.venv_dir,
                "lock_sha256": self.lock_sha256,
                "pex_version": self.pex_version,
            },
        )


@attr.s(frozen=True)
class _Pending(object):
    """The record of a generation that is being staged or switched into place."""

    @classmethod
    def load(cls, path):
        # type: (str) -> _Pending
        with open(path, "r") as fp:
            data = json.load(fp)
        return cls(
            generation=data["generation"],
            phase=data["phase"],
            lock_path=data["lock"],
            venv_dir=data.get("venv"),
            venv_existed=bool(data.get("venv_existed", False)),
            lock_candidate=data["lock_candidate"],
            venv_candidate=data.get("venv_candidate"),
            venv_stash=data.get("venv_stash"),
        )

    generation = attr.ib()  # type: str
    phase = attr.ib()  # type: str
    lock_path = attr.ib()  # type: str
    venv_dir = attr.ib(default=None)  # type: Optional[str]
    venv_existed = attr.ib(default=False)  # type: bool
    lock_candidate = attr.ib(default="")  # type: str
    venv_candidate = attr.ib(default=None)  # type: Optional[str]
    venv_stash = attr.ib(default=None)  # type: Optional[str]

    def dump(self, path):
        # type: (str) -> None
        _atomic_dump_json(
            path,
            {
                "generation": self.generation,
                "phase": self.phase,
                "lock": self.lock_path,
                "venv": self.venv_dir,
                "venv_existed": self.venv_existed,
                "lock_candidate": self.lock_candidate,
                "venv_candidate": self.venv_candidate,
                "venv_stash": self.venv_stash,
            },
        )


def _atomic_dump_json(path, data):
    # type: (str, dict) -> None
    tmp_path = "{path}.{uuid}.tmp".format(path=path, uuid=uuid4().hex)
    with open(tmp_path, "w") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    safe_rename(tmp_path, path)
    _fsync_dir(os.path.dirname(path))


def _fsync_dir(path):
    # type: (str) -> None
    if not path:
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # fsync of a directory is unsupported on some platforms / filesystems; the atomic
        # renames are still correct, durability of the directory entry is only best effort.
        pass
    finally:
        os.close(fd)


def _link_or_copy(src, dst):
    # type: (str, str) -> None
    try:
        os.link(src, dst)
        return
    except OSError as e:
        if e.errno not in (errno.EPERM, errno.EXDEV, errno.EACCES, errno.ENOSYS):
            # Retry with a copy below for any link failure rather than risking the sync; hard
            # links are only an optimization.
            pass
    shutil.copy2(src, dst)


def _clone_tree(source, destination):
    # type: (str, str) -> None
    """Recreates the `source` tree at `destination` without touching file contents.

    Directories and symlinks are recreated and regular files are hard-linked (falling back to a
    copy when the filesystem does not support links). The clone is therefore cheap and safe to
    mutate: installs replace files via rename/unlink, never by truncating a shared inode.
    """
    os.mkdir(destination)
    for root, dirnames, filenames in os.walk(source, followlinks=False):
        relative = os.path.relpath(root, source)
        destination_root = (
            destination if relative == os.curdir else os.path.join(destination, relative)
        )
        for dirname in dirnames:
            source_dir = os.path.join(root, dirname)
            destination_dir = os.path.join(destination_root, dirname)
            if os.path.islink(source_dir):
                # Symlinked directories are entries, not trees to walk.
                os.symlink(os.readlink(source_dir), destination_dir)
            else:
                os.mkdir(destination_dir)
        for filename in filenames:
            source_file = os.path.join(root, filename)
            destination_file = os.path.join(destination_root, filename)
            if os.path.islink(source_file):
                os.symlink(os.readlink(source_file), destination_file)
            else:
                _link_or_copy(source_file, destination_file)


# In-process serialization so that threads in the same process also share a single owner. The
# underlying file lock is process-scoped on POSIX (fcntl locks belong to the process).
_IN_PROCESS_LOCKS = {}  # type: Dict[str, threading.Lock]
_IN_PROCESS_LOCKS_GUARD = threading.Lock()


def _in_process_lock(path):
    # type: (str) -> threading.Lock
    with _IN_PROCESS_LOCKS_GUARD:
        lock = _IN_PROCESS_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _IN_PROCESS_LOCKS[path] = lock
        return lock


class SyncPublicationError(Exception):
    """Indicates an irrecoverable error publishing a synced lock / venv generation."""


class SyncPublication(object):
    """Owns publication of one matching lock file / venv pair for one generation."""

    def __init__(
        self,
        lock_path,  # type: str
        venv_dir=None,  # type: Optional[str]
    ):
        # type: (...) -> None
        self._lock_path = os.path.abspath(lock_path)
        self._venv_dir = os.path.realpath(venv_dir) if venv_dir else None

        lock_parent = os.path.dirname(self._lock_path) or os.curdir
        self._state_dir = os.path.join(
            lock_parent, ".{name}.pex-sync".format(name=os.path.basename(self._lock_path))
        )
        self._owner_lock_file = os.path.join(self._state_dir, "owner.lck")
        self._generation_file = os.path.join(self._state_dir, "generation.json")

        self._in_process_lock = _in_process_lock(self._owner_lock_file)
        self._file_lock = None  # type: Optional[fs_lock.FileLock]

        self._generation = None  # type: Optional[str]
        self._pending_path = None  # type: Optional[str]
        self._lock_candidate = None  # type: Optional[str]
        self._venv_candidate = None  # type: Optional[str]
        self._venv_stash = None  # type: Optional[str]
        self._venv_existed = False
        self._phase = None  # type: Optional[str]

    @property
    def lock_path(self):
        # type: () -> str
        return self._lock_path

    @property
    def venv_dir(self):
        # type: () -> Optional[str]
        return self._venv_dir

    @property
    def lock_candidate_path(self):
        # type: () -> str
        if self._lock_candidate is None:
            raise SyncPublicationError("The sync publication has not been staged.")
        return self._lock_candidate

    @property
    def venv_candidate_dir(self):
        # type: () -> str
        if self._venv_candidate is None:
            raise SyncPublicationError("The sync publication has no venv staged.")
        return self._venv_candidate

    def committed_generation(self):
        # type: () -> Optional[Generation]
        return Generation.load(self._generation_file)

    def acquire(self):
        # type: () -> None
        """Become the unique publication owner for this lock / venv pair and recover the past.

        Blocks while another in-process thread or OS process owns the publication. Any
        interrupted generation left by a dead owner is rolled forward or discarded before this
        returns, so the on-disk state is always a consistent committed generation afterwards.
        """
        self._in_process_lock.acquire()
        try:
            safe_mkdir(self._state_dir)
            self._file_lock = fs_lock.acquire(self._owner_lock_file, exclusive=True)
        except OSError:
            self._in_process_lock.release()
            raise
        # Any pending records belong to a (now dead) previous owner; the file lock is ours.
        for pending_path in self._iter_pending_paths():
            try:
                self._recover(_Pending.load(pending_path), pending_path)
            except Exception as e:
                pex_warnings.warn(
                    "Failed to recover an interrupted sync generation recorded at "
                    "{pending}: {err}".format(pending=pending_path, err=e)
                )

    def __enter__(self):
        # type: () -> SyncPublication
        self.acquire()
        return self

    def release(self):
        # type: () -> None
        if self._file_lock is not None:
            try:
                self._file_lock.release()
            finally:
                self._file_lock = None
        self._in_process_lock.release()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def _iter_pending_paths(self):
        # type: () -> list
        if not os.path.isdir(self._state_dir):
            return []
        pending_paths = [
            os.path.join(self._state_dir, name)
            for name in os.listdir(self._state_dir)
            if name.startswith("pending.") and name.endswith(".json")
        ]
        # Oldest interrupted generation first; each recovery is independent anyway.
        return sorted(pending_paths, key=lambda p: (os.path.getmtime(p), p))

    def staging(
        self,
        venv_existed=False,  # type: bool
    ):
        # type: (...) -> None
        """Open a new generation and reserve candidate paths for it.

        No committed resource is touched until :meth:`commit`; candidates abandoned before the
        switch are removed by :meth:`abort` and a dead process's staging leftovers are discarded
        by the next owner.
        """
        if self._phase is not None:
            raise SyncPublicationError("A sync generation is already in progress.")
        self._generation = uuid4().hex
        self._venv_existed = venv_existed and self._venv_dir is not None
        self._lock_candidate = "{lock}.{gen}{suffix}".format(
            lock=self._lock_path, gen=self._generation, suffix=WORK_SUFFIX
        )
        if self._venv_dir:
            self._venv_candidate = "{venv}.{gen}{suffix}".format(
                venv=self._venv_dir, gen=self._generation, suffix=WORK_SUFFIX
            )
            self._venv_stash = "{venv}.{gen}{suffix}".format(
                venv=self._venv_dir, gen=self._generation, suffix=OLD_SUFFIX
            )
        self._pending_path = os.path.join(
            self._state_dir, "pending.{gen}.json".format(gen=self._generation)
        )
        self._phase = PHASE_STAGING
        self._write_pending()

    def _write_pending(self):
        # type: () -> None
        if self._pending_path is None:
            return
        assert self._generation is not None
        assert self._lock_candidate is not None
        _Pending(
            generation=self._generation,
            phase=self._phase or PHASE_STAGING,
            lock_path=self._lock_path,
            venv_dir=self._venv_dir,
            venv_existed=self._venv_existed,
            lock_candidate=self._lock_candidate or "",
            venv_candidate=self._venv_candidate,
            venv_stash=self._venv_stash,
        ).dump(self._pending_path)

    def clone_existing_venv(self, source_venv_dir):
        # type: (str) -> None
        """Populate the venv candidate by cloning an existing venv.

        Files are hard-linked when the filesystem supports it (and copied otherwise); directories
        and symlinks are recreated. The clone is therefore cheap and safe to mutate: installs
        replace files via rename/unlink, never by truncating a shared inode.
        """
        if self._venv_candidate is None:
            raise SyncPublicationError("There is no venv to stage for this publication.")
        if os.path.exists(self._venv_candidate):
            safe_rmtree(self._venv_candidate)
        _clone_tree(os.path.realpath(source_venv_dir), self._venv_candidate)

    def lock_sha256(self):
        # type: () -> str
        if self._lock_candidate is None or not os.path.exists(self._lock_candidate):
            raise SyncPublicationError("The lock candidate has not been staged.")
        digest = hashlib.sha256()
        with open(self._lock_candidate, "rb") as fp:
            for chunk in iter(lambda: fp.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def prepared(self):
        # type: () -> None
        """Mark both candidates complete and validated; the switch may now begin.

        The phase flip is fsynced so a dead process is always observed in exactly one of two
        well-defined states: staging (nothing was renamed; discard candidates) or committing
        (both candidates are complete; roll the switch forward).
        """
        if self._phase is None:
            raise SyncPublicationError("The sync publication has not been staged.")
        assert self._lock_candidate is not None
        if not os.path.exists(self._lock_candidate):
            raise SyncPublicationError(
                "The lock candidate at {path} is missing.".format(path=self._lock_candidate)
            )
        if self._venv_dir and not (self._venv_candidate and os.path.isdir(self._venv_candidate)):
            raise SyncPublicationError(
                "The venv candidate at {path} is missing.".format(path=self._venv_candidate)
            )
        self._phase = PHASE_COMMITTING
        self._write_pending()

    def commit(self):
        # type: () -> Generation
        """Publish both candidates in the deterministic venv-then-lock order.

        Only renames are performed here: the existing venv (if any) is moved aside, the candidate
        venv takes its place and only then is the candidate lock renamed over the lock path. A
        crash leaves a ``committing`` pending record behind that the next owner rolls forward
        through the same steps.
        """
        if self._phase != PHASE_COMMITTING or self._pending_path is None:
            raise SyncPublicationError("The sync publication is not ready to commit.")

        pending = _Pending.load(self._pending_path)
        self._switch(pending)

        generation = Generation(
            generation=pending.generation,
            lock_path=pending.lock_path,
            venv_dir=pending.venv_dir,
            lock_sha256=self._hash_lock(pending.lock_path),
            pex_version=__version__,
        )
        generation.dump(self._generation_file)
        self._finish(pending, pending_path=self._pending_path)
        self._phase = None
        return generation

    def abort(self):
        # type: () -> None
        """Discard an in-progress generation that failed before (or without) switching.

        Once the switch has begun this does nothing destructive: the next owner's recovery rolls
        the generation forward.
        """
        if self._phase is None:
            return
        if self._phase == PHASE_COMMITTING:
            return
        if self._lock_candidate:
            safe_delete(self._lock_candidate)
        if self._venv_candidate:
            safe_rmtree(self._venv_candidate)
        if self._pending_path:
            safe_delete(self._pending_path)
        self._phase = None

    def _recover(self, pending, pending_path):
        # type: (_Pending, str) -> None

        if pending.phase == PHASE_STAGING:
            # The switch never started; nothing at the published paths could have been touched.
            safe_delete(pending.lock_candidate)
            if pending.venv_candidate:
                safe_rmtree(pending.venv_candidate)
            if pending.venv_stash:
                safe_rmtree(pending.venv_stash)
            safe_delete(pending_path)
            return

        # A committing generation had both candidates complete when its owner died. Roll it
        # forward through whichever renames remain; only a totally wrecked candidate set (manual
        # deletion) is rolled back.
        rolled_back = False
        venv_dir = pending.venv_dir
        if venv_dir:
            venv_stash = pending.venv_stash
            venv_candidate = pending.venv_candidate
            stashed = bool(venv_stash and os.path.exists(venv_stash))
            venv_present = os.path.exists(venv_dir)
            candidate_present = bool(venv_candidate and os.path.exists(venv_candidate))

            if stashed and not venv_present and not candidate_present:
                # Both the new candidate and the published venv are gone but the backup survives;
                # restore the known-good previous venv.
                assert venv_stash is not None
                os.rename(venv_stash, venv_dir)
                rolled_back = True
            else:
                if not stashed and pending.venv_existed and venv_present:
                    # Died before the first rename; the old venv is still live. Move it aside and
                    # then publish the complete candidate.
                    assert venv_stash is not None
                    os.rename(venv_dir, venv_stash)
                    stashed = True
                    venv_present = False
                if not venv_present and candidate_present:
                    assert venv_candidate is not None
                    os.rename(venv_candidate, venv_dir)

        if rolled_back:
            safe_delete(pending.lock_candidate)
            safe_delete(pending_path)
            return

        if pending.lock_candidate and os.path.exists(pending.lock_candidate):
            safe_rename(pending.lock_candidate, pending.lock_path)

        if pending.venv_stash and os.path.exists(pending.venv_stash):
            safe_rmtree(pending.venv_stash)

        generation = Generation(
            generation=pending.generation,
            lock_path=pending.lock_path,
            venv_dir=pending.venv_dir,
            lock_sha256=self._hash_lock(pending.lock_path),
            pex_version=__version__,
        )
        generation.dump(self._generation_file)
        self._finish(pending, pending_path=pending_path)

    def _switch(self, pending):
        # type: (_Pending) -> None
        if not pending.venv_dir:
            safe_rename(pending.lock_candidate, pending.lock_path)
            return

        if pending.venv_existed and os.path.exists(pending.venv_dir):
            assert pending.venv_stash is not None
            os.rename(pending.venv_dir, pending.venv_stash)
        if pending.venv_candidate and os.path.exists(pending.venv_candidate):
            os.rename(pending.venv_candidate, pending.venv_dir)
        elif not os.path.exists(pending.venv_dir):
            raise SyncPublicationError(
                "Cannot publish venv generation {gen}: neither the candidate at {candidate} nor a "
                "venv at {venv} exist.".format(
                    gen=pending.generation,
                    candidate=pending.venv_candidate,
                    venv=pending.venv_dir,
                )
            )
        safe_rename(pending.lock_candidate, pending.lock_path)

    @staticmethod
    def _finish(pending, pending_path):
        # type: (_Pending, str) -> None
        if pending.lock_candidate:
            safe_delete(pending.lock_candidate)
        if pending.venv_candidate:
            safe_rmtree(pending.venv_candidate)
        if pending.venv_stash:
            safe_rmtree(pending.venv_stash)
        safe_delete(pending_path)

    @staticmethod
    def _hash_lock(lock_path):
        # type: (str) -> Optional[str]
        if not os.path.exists(lock_path):
            return None
        digest = hashlib.sha256()
        with open(lock_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
