# Copyright 2024 Pex project contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from __future__ import absolute_import

import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from uuid import uuid4

from pex.common import safe_delete, safe_mkdir, safe_open, safe_sleep
from pex.enum import Enum
from pex.fs import lock as fs_lock
from pex.fs import safe_rename
from pex.os import is_alive, kill, process_start_time, terminate
from pex.subprocess import launch_python_daemon
from pex.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator, Optional

    import attr  # vendor:skip
else:
    from pex.third_party import attr

if sys.version_info[0] == 3:
    from http.client import HTTPConnection, HTTPException
    from urllib.parse import urlsplit
else:
    from httplib import HTTPConnection, HTTPException
    from urlparse import urlsplit


logger = logging.getLogger(__name__)


@attr.s(frozen=True)
class ServerInfo(object):
    url = attr.ib()  # type: str
    pid = attr.ib()  # type: int
    # A unique generation token assigned when a server process is spawned. A record only ever
    # describes the generation that published it, which allows shutdown and stale-state cleanup to
    # avoid acting on a later generation that republished the same pidfile.
    token = attr.ib(default=None)  # type: Optional[str]
    # The process start time captured at launch; used to detect a recycled PID that now belongs to
    # an unrelated process. `None` for records written by older Pex versions (or platforms where
    # the start time cannot be determined).
    start_time = attr.ib(default=None)  # type: Optional[str]
    # The log file the server process writes to.
    log = attr.ib(default=None)  # type: Optional[str]

    def __str__(self):
        # type: () -> str
        return "{url} @ {pid}".format(url=self.url, pid=self.pid)


class PidStatus(Enum["PidStatus.Value"]):
    class Value(Enum.Value):
        pass

    # The recorded process is alive, provably ours, and serving the recorded URL.
    VALID = Value("valid")
    # No process exists with the recorded PID.
    DEAD = Value("dead")
    # A process with the recorded PID exists but is a different process (the PID was recycled).
    FOREIGN = Value("foreign")
    # The recorded process is provably ours (start time matched) but no longer serves the URL.
    UNREACHABLE = Value("unreachable")
    # A process with the recorded PID is alive and not serving, but its identity could not be
    # established (no start time recorded or it could not be queried); it must never be signaled.
    UNKNOWN = Value("unknown")


PidStatus.seal()


@attr.s(frozen=True)
class Pidfile(object):
    PIDFILE_NAME = "pidfile"

    @staticmethod
    def pidfile_path(cache_dir):
        # type: (str) -> str
        return os.path.join(cache_dir, Pidfile.PIDFILE_NAME)

    @classmethod
    def load(
        cls,
        server_name,  # type: str
        cache_dir,  # type: str
    ):
        # type: (...) -> Optional[Pidfile]
        pidfile = cls.pidfile_path(cache_dir)
        try:
            with open(pidfile) as fp:
                data = json.load(fp)
            server_info = ServerInfo(
                url=data["url"],
                pid=int(data["pid"]),
                token=data.get("token"),
                start_time=(str(data["start_time"]) if data.get("start_time") is not None else None),
                log=data.get("log"),
            )
            return cls(server_info=server_info)
        except (OSError, IOError, ValueError, KeyError, TypeError) as e:
            logger.debug(
                "Failed to load {server} pid file from {path}: {err}".format(
                    server=server_name, path=pidfile, err=e
                )
            )
            return None

    @classmethod
    def publish(
        cls,
        cache_dir,  # type: str
        server_info,  # type: ServerInfo
    ):
        # type: (...) -> Pidfile
        # Atomically publish the ready record so concurrent readers and competing starters only
        # ever observe either no record or a complete committed record (never a partially written
        # "running" record).
        pidfile = cls.pidfile_path(cache_dir)
        tmp = "{pidfile}.{token}.tmp".format(pidfile=pidfile, token=server_info.token or uuid4().hex)
        try:
            with safe_open(tmp, "w") as fp:
                json.dump(
                    dict(
                        url=server_info.url,
                        pid=server_info.pid,
                        token=server_info.token,
                        start_time=server_info.start_time,
                        log=server_info.log,
                    ),
                    fp,
                    indent=2,
                    sort_keys=True,
                )
            safe_rename(tmp, pidfile)
        except (OSError, IOError, ValueError):
            safe_delete(tmp)
            raise
        return cls(server_info=server_info)

    @classmethod
    def discard(
        cls,
        server_name,  # type: str
        cache_dir,  # type: str
        token=None,  # type: Optional[str]
    ):
        # type: (...) -> bool
        # Generation guarded compare-and-delete: remove the pidfile only when the record currently
        # on disk is still the generation the caller read. A newer generation that republished in
        # the meantime is never deleted by an old caller.
        current = cls.load(server_name, cache_dir)
        if current is None:
            if not os.path.exists(cls.pidfile_path(cache_dir)):
                return False
            # A corrupt record cannot carry a generation to check; only an unguarded reclaim may
            # remove one.
            if token is not None:
                return False
        elif token is not None and current.server_info.token != token:
            return False
        safe_delete(cls.pidfile_path(cache_dir))
        return True

    server_info = attr.ib()  # type: ServerInfo

    def status(self, connect_timeout=1.0):
        # type: (float) -> PidStatus.Value

        if not is_alive(self.server_info.pid):
            return PidStatus.DEAD

        # Establish process identity positively: both a start time captured at launch and a
        # freshly queried start time that matches. This rules out a recycled PID.
        identity_verified = False
        if self.server_info.start_time:
            actual_start_time = process_start_time(self.server_info.pid)
            if actual_start_time is not None:
                if actual_start_time != self.server_info.start_time:
                    return PidStatus.FOREIGN
                identity_verified = True

        if not self.is_url_serving(self.server_info.url, timeout=connect_timeout):
            # Only terminate a non-serving process whose identity was positively verified;
            # otherwise leave the possibly-unrelated live process alone and just drop the record.
            return PidStatus.UNREACHABLE if identity_verified else PidStatus.UNKNOWN

        return PidStatus.VALID

    def is_current(self, connect_timeout=1.0):
        # type: (float) -> bool
        return self.status(connect_timeout=connect_timeout) is PidStatus.VALID

    @staticmethod
    def is_url_serving(
        url,  # type: str
        timeout=1.0,  # type: float
    ):
        # type: (...) -> bool
        split = urlsplit(url)
        hostname = split.hostname or "localhost"
        port = split.port
        if port is None:
            port = 443 if split.scheme == "https" else 80
        # The server binds all interfaces; probe IPv4 loopback directly so reachability does not
        # depend on how "localhost" resolves (127.0.0.1 vs ::1).
        probe_host = "127.0.0.1" if hostname in ("localhost",) else hostname
        connection = HTTPConnection(probe_host, port, timeout=timeout)
        try:
            try:
                connection.request("HEAD", "/")
                response = connection.getresponse()
                response.read()
            finally:
                connection.close()
            return True
        except (OSError, IOError, HTTPException) as e:
            logger.debug("The server at {url} is not serving: {err}".format(url=url, err=e))
            return False

    @staticmethod
    def _read_url(
        server_log,  # type: str
    ):
        # type: (...) -> Optional[str]

        # The permutations of Python versions, simple http server module and the output it provides:
        #  2.7: Serving HTTP on 0.0.0.0 port 46399 ... -mSimpleHTTPServer
        #  3.5: Serving HTTP on 0.0.0.0 port 45577 ... -mhttp.server
        # 3.6+: Serving HTTP on 0.0.0.0 port 33539 (http://0.0.0.0:33539/) ... -mhttp.server

        try:
            with open(server_log) as fp:
                for line in fp:
                    if line.endswith(("\r", "\n")):
                        match = re.search(r"Serving HTTP on \S+ port (?P<port>\d+)", line)
                        if match:
                            return "http://localhost:{port}".format(port=match.group("port"))
        except (OSError, IOError):
            return None
        return None


@attr.s(frozen=True)
class LaunchResult(object):
    server_info = attr.ib()  # type: ServerInfo
    already_running = attr.ib()  # type: bool


# Frozen exception types don't work under 3.11+ where the `__traceback__` attribute can be set
# after construction in some cases.
@attr.s
class LaunchError(Exception):
    """Indicates an error launching the server."""

    log = attr.ib()  # type: str
    additional_msg = attr.ib(default=None)  # type: Optional[str]
    verbose = attr.ib(default=False)  # type: bool

    def __str__(self):
        # type: () -> str
        lines = ["Error launching server."]
        if self.additional_msg:
            lines.append(self.additional_msg)
        if self.verbose:
            lines.append("Contents of the server log:")
            with open(self.log) as fp:
                lines.extend(fp.read().splitlines())
        else:
            lines.append("See the log at {log} for more details.".format(log=self.log))
        return "\n".join(lines)


def _terminate_and_wait(
    pid,  # type: int
    timeout,  # type: float
    poll_interval=0.1,  # type: float
):
    # type: (...) -> bool

    try:
        terminate(pid)
    except OSError:
        # The process may have exited after it was verified alive but before the signal was sent
        # (POSIX reports ESRCH for this and OpenProcess fails for a dead PID on Windows).
        if not is_alive(pid):
            return True
        raise

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_alive(pid):
            return True
        safe_sleep(poll_interval)

    # The process ignored / did not respond to the termination signal; escalate.
    if is_alive(pid):
        try:
            kill(pid)
        except OSError:
            return not is_alive(pid)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not is_alive(pid):
                return True
            safe_sleep(poll_interval)

    return not is_alive(pid)


@attr.s(frozen=True)
class Server(object):
    name = attr.ib()  # type: str
    cache_dir = attr.ib()  # type: str

    _LOCK_NAME = "lock"
    _LOG_DIR_NAME = "logs"
    # Bound on how many per-launch log files are retained; only older logs are swept, so the log
    # of a just-failed launch is always still locatable.
    _LOGS_TO_KEEP = 5

    def _lock_path(self):
        # type: () -> str
        return os.path.join(self.cache_dir, self._LOCK_NAME)

    @contextmanager
    def _locked(self):
        # type: () -> Iterator[None]
        # All state-changing operations are serialized so that exactly one competing starter can
        # spawn a generation, and shutdown never acts mid-publish. Both flock (POSIX) and
        # LockFileEx (Windows) are released automatically by the OS if a holder dies, so a crashed
        # holder can never cause a long-term block.
        file_lock = fs_lock.acquire(self._lock_path(), exclusive=True)
        try:
            yield
        finally:
            file_lock.release()

    def _log_dir(self):
        # type: () -> str
        return os.path.join(self.cache_dir, self._LOG_DIR_NAME)

    def pidfile(self):
        # type: () -> Optional[Pidfile]
        return Pidfile.load(self.name, self.cache_dir)

    def running(self, connect_timeout=1.0):
        # type: (float) -> Optional[Pidfile]
        # Lock-free observation: atomic publish means a loaded record is always complete, and the
        # record is only reported when the instance is verifiably alive and serving here and now.
        pidfile = Pidfile.load(self.name, self.cache_dir)
        if pidfile and pidfile.status(connect_timeout=connect_timeout) is PidStatus.VALID:
            return pidfile
        return None

    def launch(
        self,
        document_root,  # type: str
        port=0,  # type: int
        timeout=5.0,  # type: float
        verbose_error=False,  # type: bool
    ):
        # type: (...) -> LaunchResult

        with self._locked():
            pidfile = Pidfile.load(self.name, self.cache_dir)
            if pidfile is not None:
                status = pidfile.status()
                if status is PidStatus.VALID:
                    return LaunchResult(server_info=pidfile.server_info, already_running=True)
                self._reclaim(pidfile, status)
            elif os.path.exists(Pidfile.pidfile_path(self.cache_dir)):
                # Corrupt or unreadable record; identity cannot be confirmed, so just drop it.
                logger.debug("Discarding corrupt {server} pid file.".format(server=self.name))
                safe_delete(Pidfile.pidfile_path(self.cache_dir))

            return self._spawn(
                document_root, port=port, timeout=timeout, verbose_error=verbose_error
            )

    def _reclaim(
        self,
        pidfile,  # type: Pidfile
        status,  # type: PidStatus.Value
    ):
        # type: (...) -> None
        info = pidfile.server_info
        if status is PidStatus.UNREACHABLE:
            # The PID is alive and is provably our process, but it is no longer serving; terminate
            # the broken instance before retrying. A DEAD process needs no signal and neither a
            # FOREIGN process (recycled PID) nor an UNKNOWN one (unverified identity) is signaled.
            logger.debug(
                "Terminating defunct {server} {info}".format(server=self.name, info=info)
            )
            _terminate_and_wait(info.pid, timeout=5.0)
        else:
            logger.debug(
                "Discarding stale {server} record {info} ({status})".format(
                    server=self.name, info=info, status=status
                )
            )
        Pidfile.discard(self.name, self.cache_dir, info.token)

    def _spawn(
        self,
        document_root,  # type: str
        port,  # type: int
        timeout,  # type: float
        verbose_error,  # type: bool
    ):
        # type: (...) -> LaunchResult

        self._sweep_stale_logs()

        token = uuid4().hex
        log = os.path.join(self._log_dir(), "{token}.log".format(token=token))

        # Not proper daemonization, but good enough.
        http_server_module = "http.server" if sys.version_info[0] == 3 else "SimpleHTTPServer"
        env = os.environ.copy()
        # N.B.: We set up line buffering for the process pipes as well as the underlying Python
        # running the http server to ensure we can observe the `Serving HTTP on ...` line we need
        # to grab the ephemeral port chosen.
        env.update(PYTHONUNBUFFERED="1")
        try:
            with safe_open(log, "w") as fp:
                process = launch_python_daemon(
                    args=[sys.executable, "-m", http_server_module, str(port)],
                    env=env,
                    cwd=document_root,
                    # N.B.: Line buffering under Python 3 requires both bufsize=1 and
                    # universal_newlines=True.
                    bufsize=1,
                    universal_newlines=True,
                    stdout=fp.fileno(),
                    stderr=subprocess.STDOUT,
                )
        except OSError as e:
            raise LaunchError(
                log,
                additional_msg="Failed to spawn the server process: {err}".format(err=e),
                verbose=verbose_error,
            )

        try:
            url = self._wait_for_ready(process, log, timeout)
            server_info = ServerInfo(
                url=url,
                pid=process.pid,
                token=token,
                start_time=process_start_time(process.pid),
                log=log,
            )
            try:
                pidfile = Pidfile.publish(self.cache_dir, server_info)
            except (OSError, IOError, ValueError) as e:
                raise LaunchError(
                    log,
                    additional_msg=(
                        "The server started but its ready record could not be written: {err}".format(
                            err=e
                        )
                    ),
                    verbose=verbose_error,
                )

            if pidfile.status() is not PidStatus.VALID:
                # Never leave a record claiming a server that cannot be verified.
                Pidfile.discard(self.name, self.cache_dir, token)
                raise LaunchError(
                    log,
                    additional_msg="The server became unavailable before it was committed.",
                    verbose=verbose_error,
                )

            return LaunchResult(server_info=server_info, already_running=False)
        except LaunchError:
            # No verified ready record has been published (or it was just discarded above), so
            # kill the partially launched process and leave no fake "running" state. The log file
            # itself is retained and referenced by the propagated LaunchError.
            self._abort(process)
            raise

    def _wait_for_ready(
        self,
        process,  # type: subprocess.Popen
        log,  # type: str
        timeout,  # type: float
    ):
        # type: (...) -> str
        deadline = time.time() + timeout
        while True:
            returncode = process.poll()
            if returncode is not None:
                raise LaunchError(
                    log,
                    additional_msg=(
                        "The server process exited with code {code} before reporting its address.".format(
                            code=returncode
                        )
                    ),
                )

            url = Pidfile._read_url(log)
            if url is not None and Pidfile.is_url_serving(
                url, timeout=min(1.0, max(0.0, deadline - time.time()))
            ):
                return url

            if time.time() >= deadline:
                raise LaunchError(
                    log,
                    additional_msg=(
                        "Timed out after {timeout} second(s) waiting for the server to report its "
                        "address.".format(timeout=timeout)
                    ),
                )
            safe_sleep(0.05)

    @staticmethod
    def _abort(process):
        # type: (subprocess.Popen) -> None
        # Kill a partially launched server; no ready record has been (or remains) published.
        if process.poll() is None:
            try:
                kill(process.pid)
            except OSError:
                pass
        deadline = time.time() + 5.0
        while process.poll() is None and time.time() < deadline:
            safe_sleep(0.05)

    def _sweep_stale_logs(self):
        # type: () -> None
        log_dir = self._log_dir()
        safe_mkdir(log_dir)
        logs = glob.glob(os.path.join(log_dir, "*.log"))
        if len(logs) <= self._LOGS_TO_KEEP:
            return
        for stale_log in sorted(logs, key=lambda path: os.path.getmtime(path))[
            : len(logs) - self._LOGS_TO_KEEP
        ]:
            safe_delete(stale_log)

    def shutdown(self, timeout=10.0):
        # type: (float) -> Optional[ServerInfo]

        with self._locked():
            pidfile = Pidfile.load(self.name, self.cache_dir)
            if pidfile is None:
                # A corrupt record is not an instance to shut down; just clear it so the next
                # launch is not blocked.
                if os.path.exists(Pidfile.pidfile_path(self.cache_dir)):
                    safe_delete(Pidfile.pidfile_path(self.cache_dir))
                return None

            status = pidfile.status()
            info = pidfile.server_info
            if (
                status is PidStatus.DEAD
                or status is PidStatus.FOREIGN
                or status is PidStatus.UNKNOWN
            ):
                # The process is gone, the PID was recycled, or its identity cannot be proven:
                # only the stale record is removed and a live foreign/unverified process is never
                # signaled.
                logger.debug(
                    "Clearing stale {server} record {info} ({status})".format(
                        server=self.name, info=info, status=status
                    )
                )
                Pidfile.discard(self.name, self.cache_dir, info.token)
                return None

            # status() above just reconfirmed the process identity (and reachability for VALID)
            # for the exact generation we read; terminate, wait for actual exit and only then
            # remove *that* generation's record.
            logger.debug("Killing {server} {info}".format(server=self.name, info=info))
            if not _terminate_and_wait(info.pid, timeout=timeout):
                raise OSError(
                    "The {server} process with pid {pid} did not exit after being asked to "
                    "terminate.".format(server=self.name, pid=info.pid)
                )
            Pidfile.discard(self.name, self.cache_dir, info.token)
            return info
