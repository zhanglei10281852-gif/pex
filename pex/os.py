# Copyright 2025 Pex project contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from __future__ import absolute_import

import os
import sys

from pex import pex_root
from pex.enum import Enum
from pex.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, List, NoReturn, Optional, Text, Tuple, Union


class _CurrentOs(object):
    def __get__(self, obj, objtype=None):
        # type: (...) -> Os.Value
        if not hasattr(self, "_current"):
            # See: https://docs.python.org/3/library/sys.html#sys.platform

            # N.B.: Python 2.7 uses "linux2".
            if sys.platform.startswith("linux"):
                self._current = Os.LINUX
            elif sys.platform == "darwin":
                self._current = Os.MACOS
            elif sys.platform == "win32":
                self._current = Os.WINDOWS
            elif (
                sys.platform.startswith("aix")
                or sys.platform.startswith("freebsd")
                or sys.platform in ("android", "ios")
            ):
                self._current = Os.UNIX
            if not hasattr(self, "_current"):
                raise ValueError(
                    "The current operating system is not supported!: {system}".format(
                        system=sys.platform
                    )
                )
        return self._current


class Os(Enum["Os.Value"]):
    class Value(Enum.Value):
        def path_join(self, *components):
            # type: (*str) -> str
            return ("\\" if self is Os.WINDOWS else "/").join(components)

    LINUX = Value("linux")
    MACOS = Value("macos")
    WINDOWS = Value("windows")
    UNIX = Value("unix")
    CURRENT = _CurrentOs()


Os.seal()

# N.B.: Python 2.7 uses "linux2".
LINUX = Os.CURRENT is Os.LINUX
MAC = Os.CURRENT is Os.MACOS
WINDOWS = Os.CURRENT is Os.WINDOWS


HOME_ENV_VAR = "USERPROFILE" if WINDOWS else "HOME"


if WINDOWS:

    import ctypes
    from ctypes.wintypes import DWORD

    # See: https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-filetime
    class _FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", DWORD),
            ("dwHighDateTime", DWORD),
        ]

    def safe_execv(argv):
        # type: (Union[List[str], Tuple[str, ...]]) -> NoReturn

        import subprocess
        import sys

        from pex import atexit

        atexit.perform_exit()
        with pex_root.preserve_fallback():
            sys.exit(subprocess.call(args=argv))

else:

    def safe_execv(argv):
        # type: (Union[List[str], Tuple[str, ...]]) -> NoReturn

        from pex import atexit

        atexit.perform_exit()
        with pex_root.preserve_fallback() as env:
            os.execve(argv[0], argv, env)


if WINDOWS:
    _GBT = None  # type: Any

    def is_exe(path):
        # type: (Text) -> bool

        if not os.path.isfile(path):
            return False

        from pex.sysconfig import EXE_EXTENSIONS

        _, ext = os.path.splitext(path)
        if ext.lower() in EXE_EXTENSIONS:
            return True

        import ctypes
        from ctypes.wintypes import BOOL, DWORD, LPCWSTR, LPDWORD

        global _GBT
        if _GBT is None:
            # https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-getbinarytypew
            gbt = ctypes.windll.kernel32.GetBinaryTypeW  # type: ignore[attr-defined]
            gbt.argtypes = (
                # lpApplicationName
                LPCWSTR,
                # lpBinaryType
                LPDWORD,
            )
            gbt.restype = BOOL
            _GBT = gbt

        # N.B.: We don't care about the binary type, just the bool which tells us it is or is not an
        # executable.
        _binary_type = DWORD(0)
        return bool(_GBT(path, ctypes.byref(_binary_type)))

else:

    def is_exe(path):
        # type: (Text) -> bool
        """Determines if the given path is a file executable by the current user.

        :param path: The path to check.
        :return: `True if the given path is a file executable by the current user.
        """
        return os.path.isfile(path) and os.access(path, os.R_OK | os.X_OK)


if WINDOWS:

    def is_alive(pid):
        # type: (int) -> bool

        # TODO(John Sirois): This is extremely hacky, consider adding a psutil dependency for
        #  Windows. See: https://github.com/pex-tool/pex/issues/2699

        import csv
        import subprocess

        args = ["tasklist", "/FI", "PID eq {pid}".format(pid=pid), "/FO", "CSV"]
        process = subprocess.Popen(args=args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                "Failed to query status of process with pid {pid}.\n"
                "Execution of `{args}` returned exit code {returncode}.\n"
                "{stderr}".format(
                    pid=pid,
                    args=" ".join(args),
                    returncode=process.returncode,
                    stderr=stderr.decode("utf-8"),
                )
            )

        output = stdout.decode("utf-8")
        if "No tasks are running" in output:
            return False

        lines = output.splitlines()
        if len(lines) != 2:
            return False

        csv_reader = csv.DictReader(lines)
        for row in csv_reader:
            pid_value = row.get("PID", -1)
            if pid_value == -1:
                return False
            try:
                return pid == int(pid_value)
            except (ValueError, TypeError):
                return False
        return False

    # https://learn.microsoft.com/en-us/windows/win32/procthread/process-security-and-access-rights
    _PROCESS_TERMINATE = 0x1  # Required to terminate a process using TerminateProcess.

    _OP = None  # type: Any
    _TP = None  # type: Any

    def kill(pid):
        # type: (int) -> None

        import ctypes
        from ctypes.wintypes import BOOL, DWORD, HANDLE, UINT

        global _OP
        if _OP is None:
            # https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-openprocess
            op = ctypes.windll.kernel32.OpenProcess  # type: ignore[attr-defined]
            op.argtypes = (
                DWORD,  # dwDesiredAccess
                BOOL,  # bInheritHandle
                DWORD,  # dwProcessId
            )
            op.restype = HANDLE
            _OP = op

        phandle = _OP(_PROCESS_TERMINATE, False, pid)
        if not phandle:
            # TODO(John Sirois): Review literature / experiment and don't raise if this just means
            #  the process is already dead.
            #  See: https://github.com/pex-tool/pex/issues/2670
            raise ctypes.WinError()  # type: ignore[attr-defined]

        global _TP
        if _TP is None:
            # https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-terminateprocess
            tp = ctypes.windll.kernel32.TerminateProcess  # type: ignore[attr-defined]
            tp.argtypes = (
                HANDLE,  # hProcess
                UINT,  # uExitCode
            )
            tp.restype = BOOL
            _TP = tp

        if not _TP(phandle, 1):
            # TODO(John Sirois): Review literature / experiment and don't raise if this just means
            # the process is already dead (may need to consult GetLastError).
            #  See: https://github.com/pex-tool/pex/issues/2670
            raise ctypes.WinError()  # type: ignore[attr-defined]

    # https://learn.microsoft.com/en-us/windows/win32/procthread/process-security-and-access-rights
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    _GPT = None  # type: Any

    def process_start_time(pid):
        # type: (int) -> Optional[str]

        import ctypes
        from ctypes.wintypes import BOOL, DWORD, HANDLE

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        # N.B.: ctypes returns a shared, cached function object whose argtypes can be redefined by
        # other callers, so pin the prototype on each invocation.
        open_process = kernel32.OpenProcess
        open_process.argtypes = (DWORD, BOOL, DWORD)
        open_process.restype = HANDLE
        phandle = open_process(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not phandle:
            # The process is gone or cannot be queried.
            return None

        try:
            global _GPT
            if _GPT is None:
                # https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes
                _GPT = kernel32.GetProcessTimes
                _GPT.argtypes = (
                    HANDLE,  # hProcess
                    ctypes.POINTER(_FILETIME),  # lpCreationTime
                    ctypes.POINTER(_FILETIME),  # lpExitTime
                    ctypes.POINTER(_FILETIME),  # lpKernelTime
                    ctypes.POINTER(_FILETIME),  # lpUserTime
                )
                _GPT.restype = BOOL

            # The creation FILETIME is a 64-bit count of 100ns intervals since 1601-01-01 UTC.
            # A process creation time is stable for the lifetime of the process and unique across
            # recycled PIDs.
            creation_time = _FILETIME()
            if not _GPT(
                phandle,
                ctypes.byref(creation_time),
                ctypes.byref(_FILETIME()),
                ctypes.byref(_FILETIME()),
                ctypes.byref(_FILETIME()),
            ):
                return None

            return str((creation_time.dwHighDateTime << 32) | creation_time.dwLowDateTime)
        finally:
            kernel32.CloseHandle(phandle)

    def terminate(pid):
        # type: (int) -> None

        import ctypes
        from ctypes.wintypes import BOOL, DWORD, HANDLE, UINT

        # Windows has no SIGTERM equivalent; TerminateProcess is the only way to ask another
        # process to exit unconditionally and behaves like a SIGKILL for shutdown purposes.
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        # N.B.: Pin the prototypes on each invocation since the ctypes function objects are shared
        # and their argtypes can be redefined by other callers.
        open_process = kernel32.OpenProcess
        open_process.argtypes = (DWORD, BOOL, DWORD)
        open_process.restype = HANDLE
        phandle = open_process(_PROCESS_TERMINATE, False, pid)
        if not phandle:
            raise ctypes.WinError()  # type: ignore[attr-defined]
        try:
            terminate_process = kernel32.TerminateProcess
            terminate_process.argtypes = (HANDLE, UINT)
            terminate_process.restype = BOOL
            if not terminate_process(phandle, 1):
                raise ctypes.WinError()  # type: ignore[attr-defined]
        finally:
            kernel32.CloseHandle(phandle)

else:

    def is_alive(pid):
        # type: (int) -> bool

        import errno

        try:
            os.kill(pid, 0)
            return True
        except OSError as e:
            if e.errno == errno.ESRCH:  # No such process.
                return False
            raise

    def kill(pid):
        # type: (int) -> None

        import errno
        import signal

        try:
            os.kill(pid, signal.SIGKILL)  # type: ignore[attr-defined]
        except OSError as e:
            if e.errno != errno.ESRCH:  # No such process.
                raise

    if sys.platform == "darwin":

        def process_start_time(pid):
            # type: (int) -> Optional[str]

            # macOS does not expose /proc; `ps -o lstart=` renders the process start time and is
            # stable for the lifetime of the process (and distinct for a recycled PID).
            import subprocess

            try:
                process = subprocess.Popen(
                    args=["ps", "-p", str(pid), "-o", "lstart="],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, _ = process.communicate()
                if process.returncode != 0:
                    return None
                start_time = stdout.decode("utf-8").strip()
                return start_time or None
            except OSError:
                return None

    elif os.path.isdir("/proc"):

        def process_start_time(pid):
            # type: (int) -> Optional[str]

            # The 22nd field of /proc/<pid>/stat is the process start time in clock ticks since
            # boot; see https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html. This value is
            # stable for the lifetime of the process and distinct for a recycled PID.
            try:
                with open("/proc/{pid}/stat".format(pid=pid)) as fp:
                    data = fp.read()
            except (OSError, IOError):
                return None

            # The 2nd field (comm) is wrapped in parentheses and may itself contain spaces and
            # parentheses, so split on the final closing parenthesis.
            comm_end = data.rfind(")")
            if comm_end == -1:
                return None
            fields = data[comm_end + 1:].split()
            # After comm the fields resume at the 3rd field (state); starttime (22) is thus at
            # index 19.
            if len(fields) < 20:
                return None
            return fields[19]

    else:

        def process_start_time(pid):
            # type: (int) -> Optional[str]
            return None

    def terminate(pid):
        # type: (int) -> None

        import errno
        import signal

        try:
            os.kill(pid, signal.SIGTERM)  # type: ignore[attr-defined]
        except OSError as e:
            if e.errno != errno.ESRCH:  # No such process.
                raise
