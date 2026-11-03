# Copyright 2026 Pex project contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from __future__ import absolute_import

import hashlib
import os
import os.path
import re
import shutil
import subprocess
import sys
import time

import pytest

from pex.cache_helper import CacheHelper
from pex.common import safe_delete
from pex.fetcher import URLFetcher
from pex.interpreter import PythonInterpreter
from pex.layout import Layout
from pex.os import WINDOWS
from pex.pep_440 import Version
from pex.sysconfig import SysPlatform
from pex.typing import TYPE_CHECKING
from testing import (
    PY310,
    PY311,
    ensure_python_interpreter,
    installed_pex_wheel_venv_python,
    make_env,
    run_pex_command,
)
from testing.cli import run_pex3
from testing.pytest_utils.tmp import Tempdir
from testing.scie import skip_if_no_provider

if TYPE_CHECKING:
    from typing import Any, List, Optional, Text, Tuple


def test_not_a_pex(tmpdir):
    # type: (Tempdir) -> None

    run_pex3("scie", "create", "--style", "eager", tmpdir.path).assert_failure(
        expected_error_re=r"^{message}.+$".format(
            message=re.escape(
                "The path {pex_path} does not appear to be a PEX: ".format(pex_path=tmpdir.path)
            )
        )
    )


def download_pex_pex(
    tmpdir,  # type: Tempdir
    version=None,  # type: Optional[str]
):
    # type: (...) -> str

    fetcher = URLFetcher()
    if version:
        url = "https://github.com/pex-tool/pex/releases/download/v{version}/pex".format(
            version=version
        )
    else:
        url = "https://github.com/pex-tool/pex/releases/latest/download/pex"
    with fetcher.get_body_stream(url) as in_fp, open(
        tmpdir.join("pex-{version}.pex".format(version=version or "latest")), "wb"
    ) as out_fp:
        shutil.copyfileobj(in_fp, out_fp)
    return out_fp.name


def test_pex_too_old(tmpdir):
    # type: (Tempdir) -> None

    pex_2_1_24 = download_pex_pex(tmpdir, "2.1.24")
    run_pex3("scie", "create", "--style", "eager", pex_2_1_24).assert_failure(
        expected_error_re=r"^{message}$".format(
            message=re.escape(
                "Can only create scies from PEXes built by Pex 2.1.25 (which was released on "
                "January 21st, 2021) or newer.\n"
                "The PEX at {pex_file} was built by Pex 2.1.24.".format(pex_file=pex_2_1_24)
            )
        )
    )


def get_version(
    *exe,  # type: Text
    **extra_env  # type: Any
):
    # type: (...) -> Version
    return Version(
        subprocess.check_output(args=list(exe) + ["--version"], env=make_env(**extra_env))
        .decode("utf-8")
        .strip()
    )


def create_scie(
    *args,  # type: Text
    **extra_env  # type: Any
):
    # type: (...) -> Tuple[Text, Version]

    result = run_pex3("scie", "create", "--style", "eager", *args, env=make_env(**extra_env))

    match = re.search(r"^Saved PEX scie for .+ to (?P<path>.+)$", result.output)
    assert match is not None

    scie_path = match.group("path")
    assert scie_path is not None

    return scie_path, get_version(scie_path, **extra_env)


def test_pex_old(tmpdir):
    # type: (Tempdir) -> None

    pex_2_1_25 = download_pex_pex(tmpdir, "2.1.25")

    # N.B.: Pex this old does not respect XDG_CACHE_DIR, which foils CI's cache image arrangement.
    # We work around this by establishing a custom SCIE_BASE.
    scie_base = tmpdir.join("nce")

    # N.B.: The 20251031 PBS release was the last to ship Python 3.9, which is the upper-bound of
    # supported Pythons for Pex 2.1.25.
    scie_path, scie_version = create_scie(
        "--scie-pbs-release", "20251031", pex_2_1_25, SCIE_BASE=scie_base
    )
    assert scie_path != pex_2_1_25
    assert Version("2.1.25") == scie_version


# N.B.: Modern Pex PEXes ship with no ICs; so the current interpreter is used.
@skip_if_no_provider
def test_nominal(tmpdir):
    # type: (Tempdir) -> None

    pex_latest = download_pex_pex(tmpdir)
    pex_latest_version = get_version(sys.executable, pex_latest)

    scie_path, scie_version = create_scie(pex_latest)
    assert scie_path != pex_latest
    assert pex_latest_version == scie_version


@skip_if_no_provider
def test_ics(tmpdir):
    # type: (Tempdir) -> None

    major, minor = sys.version_info[:2]
    current_interpreter_constraint = "=={major}.{minor}.*".format(major=major, minor=minor)

    other_version = PY310 if (major, minor) == (3, 11) else PY311
    other_python = PythonInterpreter.from_binary(ensure_python_interpreter(other_version))
    other_interpreter_constraint = "=={major}.{minor}.*".format(
        major=other_python.version[0], minor=other_python.version[1]
    )

    cowsay_pex = tmpdir.join("cowsay.pex")

    run_pex_command(
        args=[
            "--interpreter-constraint",
            current_interpreter_constraint,
            "--interpreter-constraint",
            other_interpreter_constraint,
            "cowsay==5.0",
            "-c",
            "cowsay",
            "-o",
            cowsay_pex,
        ]
    ).assert_success()

    sorted_ics = sorted((current_interpreter_constraint, other_interpreter_constraint))
    run_pex3(
        "scie", "create", "--style", "eager", "--interpreter-constraint", "==3.9.*", cowsay_pex
    ).assert_failure(
        expected_error_re=r"^{message}$".format(
            message=re.escape(
                "The PEX has interpreter constraints of {ic1} or {ic2} and the user-supplied "
                "interpreter constraints of ==3.9.* do not form a subset of those.".format(
                    ic1=sorted_ics[0], ic2=sorted_ics[1]
                )
            )
        )
    )

    scie_path, scie_version = create_scie(
        "--interpreter-constraint", current_interpreter_constraint, cowsay_pex
    )
    assert Version("5") == scie_version
    assert b"| Moo! |" in subprocess.check_output(args=[scie_path, "Moo!"])


@skip_if_no_provider
def test_pex_file_url_in_place(tmpdir):
    # type: (Tempdir) -> None

    pex = tmpdir.join("cowsay.pex")
    run_pex_command(args=["cowsay<6", "-c", "cowsay", "-o", pex]).assert_success()
    expected_size = os.path.getsize(pex)
    expected_fingerprint = CacheHelper.hash(pex, hasher=hashlib.md5)

    scie = tmpdir.join("cowsay")

    def assert_scie(url):
        # type: (str) -> None
        safe_delete(scie)
        run_pex3("scie", "create", "--style", "eager", url, cwd=tmpdir.path).assert_success()
        assert b"| Moo! |" in subprocess.check_output(args=[scie, "Moo!"])

    assert_scie("file:{pex}".format(pex=pex))
    assert_scie("file:{pex}#size={size}".format(pex=pex, size=expected_size))
    assert_scie("file:{pex}#md5={fingerprint}".format(pex=pex, fingerprint=expected_fingerprint))
    assert_scie(
        "file:{pex}#md5={fingerprint}&size={size}".format(
            pex=pex, fingerprint=expected_fingerprint, size=expected_size
        )
    )

    def assert_reject_pex(
        url,  # type: str
        expected_error_re,  # type: str
    ):
        run_pex3("scie", "create", "--style", "eager", url, cwd=tmpdir.path).assert_failure(
            expected_error_re=expected_error_re
        )

    assert_reject_pex(
        "file:{pex}#size=1".format(pex=pex),
        expected_error_re=re.escape(
            "Rejecting file at {pex} with size {size} bytes since expected size is 1 bytes.".format(
                pex=pex, size=expected_size
            )
        ),
    )
    assert_reject_pex(
        "file:{pex}#md5=bad".format(pex=pex),
        expected_error_re=re.escape(
            "File at {pex} had unexpected fingerprint.\n"
            "Expected: bad\n"
            "Actual:   {fingerprint}".format(pex=pex, fingerprint=expected_fingerprint)
        ),
    )


@skip_if_no_provider
def test_pex_file_url_different_dest(tmpdir):
    # type: (Tempdir) -> None

    pex = tmpdir.join("cowsay.pex")
    run_pex_command(args=["cowsay<6", "-c", "cowsay", "-o", pex]).assert_success()
    expected_size = os.path.getsize(pex)
    expected_fingerprint = CacheHelper.hash(pex, hasher=hashlib.md5)

    dest = tmpdir.join("dest")
    scie = os.path.join(dest, "cowsay")

    def assert_scie(pex_url):
        # type: (str) -> None
        safe_delete(scie)
        run_pex3("scie", "create", "--style", "eager", "--dest-dir", dest, pex_url).assert_success()
        assert b"| Moo! |" in subprocess.check_output(args=[scie, "Moo!"])

    assert_scie("file:{pex}".format(pex=pex))
    assert_scie("file:{pex}#size={size}".format(pex=pex, size=expected_size))
    assert_scie("file:{pex}#md5={fingerprint}".format(pex=pex, fingerprint=expected_fingerprint))
    assert_scie(
        "file:{pex}#md5={fingerprint}&size={size}".format(
            pex=pex, fingerprint=expected_fingerprint, size=expected_size
        )
    )

    def assert_reject_pex(
        pex_url,  # type: str
        expected_error_re,  # type: str
    ):
        run_pex3(
            "scie", "create", "--style", "eager", "--dest-dir", dest, pex_url, cwd=tmpdir.path
        ).assert_failure(expected_error_re=expected_error_re)

    url = "file:{pex}#size=1".format(pex=pex)
    assert_reject_pex(
        url,
        expected_error_re=(
            "Terminating download of {url} at \\d+ bytes since expected size is 1 bytes\\.".format(
                url=re.escape(url)
            )
        ),
    )

    url = "file:{pex}#md5=bad".format(pex=pex)
    assert_reject_pex(
        url,
        expected_error_re=(
            "Download of {url} had unexpected fingerprint\\.\n"
            "Expected: bad\n"
            "Actual:   {fingerprint}".format(url=re.escape(url), fingerprint=expected_fingerprint)
        ),
    )


@skip_if_no_provider
def test_pex_url(tmpdir):
    # type: (Tempdir) -> None

    dest = tmpdir.join("dest")
    run_pex3(
        "scie",
        "create",
        "--style",
        "eager",
        "-d",
        dest,
        "https://github.com/pex-tool/pex/releases/download/v2.85.3/pex",
    ).assert_success()
    assert (
        "2.85.3"
        == subprocess.check_output(args=[os.path.join(dest, "pex"), "-V"]).decode("utf-8").strip()
    )


@skip_if_no_provider
@pytest.mark.parametrize(
    "layout", [pytest.param(layout, id=str(layout)) for layout in Layout.values()]
)
def test_pex_file_path_dest(
    tmpdir,  # type: Tempdir
    layout,  # type: Layout.Value
):
    pex = tmpdir.join("cowsay.pex")
    run_pex_command(
        args=["cowsay<6", "-c", "cowsay", "-o", pex, "--layout", str(layout)]
    ).assert_success()

    dest = tmpdir.join("dest")
    run_pex3("scie", "create", "--style", "eager", "-d", dest, pex).assert_success()

    assert b"| Moo! |" in subprocess.check_output(args=[sys.executable, pex, "Moo!"])
    assert not os.path.exists(tmpdir.join("cowsay"))
    assert b"| Foo! |" in subprocess.check_output(
        args=[sys.executable, os.path.join(dest, "cowsay.pex"), "Foo!"]
    )
    assert b"| Boo! |" in subprocess.check_output(args=[os.path.join(dest, "cowsay"), "Boo!"])

    dest2 = tmpdir.join("dest2")
    run_pex3("scie", "create", "--style", "eager", "--scie-only", "-d", dest2, pex).assert_success()

    assert b"| Zoo! |" in subprocess.check_output(args=[sys.executable, pex, "Zoo!"])
    assert not os.path.exists(tmpdir.join("cowsay"))
    assert not os.path.exists(os.path.join(dest2, "cowsay.pex"))
    assert b"| Goo! |" in subprocess.check_output(args=[os.path.join(dest2, "cowsay"), "Goo!"])


# A deterministic, offline stand-in for the `science` binary. It parses just enough of the
# `science ... lift build` command line to emit the named scie binary and any requested hash
# sidecars, and it supports test knobs via environment variables: failing a target platform,
# writing a bad checksum sidecar, tagging output with a marker and sleeping while building.
_FAKE_SCIENCE_SCRIPT = r"""
from __future__ import print_function

import hashlib
import os
import sys
import time


def main(argv):
    if len(argv) == 1 and argv[0] == "--version":
        print("0.21.0")
        return 0

    dest_dir = None
    manifest_path = None
    hash_algorithms = []
    use_platform_suffix = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--dest-dir":
            dest_dir = argv[i + 1]
            i += 2
        elif arg == "--hash":
            hash_algorithms.append(argv[i + 1])
            i += 2
        elif arg == "--cache-dir":
            i += 2
        elif arg == "--use-platform-suffix":
            use_platform_suffix = True
            i += 1
        elif arg == "--no-use-platform-suffix":
            use_platform_suffix = False
            i += 1
        elif arg.endswith(".toml"):
            manifest_path = arg
            i += 1
        else:
            i += 1

    manifest_base = os.path.basename(manifest_path)[: -len(".toml")]
    app_name, platform = manifest_base.split("-lift-", 1)
    is_windows_target = platform.startswith("windows")

    fail_platform = os.environ.get("PEX_TEST_FAIL_SCIE_PLATFORM")
    if fail_platform and platform == fail_platform:
        print(
            "fake-science: injected failure for {platform}".format(platform=platform),
            file=sys.stderr,
        )
        return 1

    if os.environ.get("PEX_TEST_SCIE_SLEEP"):
        time.sleep(float(os.environ["PEX_TEST_SCIE_SLEEP"]))

    if use_platform_suffix:
        binary = "{name}-{platform}{ext}".format(
            name=app_name,
            platform=platform,
            ext=".exe" if is_windows_target else "",
        )
    else:
        binary = "{name}{ext}".format(
            name=app_name, ext=".exe" if is_windows_target else ""
        )

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    out_path = os.path.join(dest_dir, binary)
    marker = os.environ.get("PEX_TEST_SCIE_MARKER", "")
    payload = "fake-scie-binary for {name} on {platform} marker={marker}\n".format(
        name=app_name, platform=platform, marker=marker
    ).encode("utf-8")
    with open(out_path, "wb") as fp:
        fp.write(payload)
    for algorithm in hash_algorithms:
        digest = hashlib.new(algorithm)
        digest.update(payload)
        recorded = digest.hexdigest()
        if os.environ.get("PEX_TEST_BAD_SIDECAR") == platform:
            recorded = "0" * len(recorded)
        with open("{out}.{alg}".format(out=out_path, alg=algorithm), "w") as fp:
            fp.write(
                "{digest}  {binary}\n".format(digest=recorded, binary=binary)
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
"""


def _install_fake_science(tmpdir):
    # type: (Tempdir) -> str

    script = tmpdir.join("fake_science.py")
    with open(script, "w") as fp:
        fp.write(_FAKE_SCIENCE_SCRIPT)

    if WINDOWS:
        launcher = tmpdir.join("fake-science.cmd")
        with open(launcher, "w") as fp:
            fp.write(
                '@echo off\r\n"{python}" "%~dp0fake_science.py" %*\r\n'.format(
                    python=sys.executable
                )
            )
    else:
        launcher = tmpdir.join("fake-science")
        with open(launcher, "w") as fp:
            fp.write(
                '#!/bin/sh\nexec "{python}" "$(dirname "$0")/fake_science.py" "$@"\n'.format(
                    python=sys.executable
                )
            )
        os.chmod(launcher, 0o755)
    return os.path.basename(launcher)


def _build_simple_pex(pex):
    # type: (str) -> None
    run_pex_command(args=["-m", "http.server", "-o", pex, "--no-pypi"]).assert_success()


def _staging_leftovers(dest):
    # type: (str) -> List[str]
    return [
        name
        for name in os.listdir(dest)
        if name.startswith(".~pex-scie-create-") or name.startswith(".~pex-scie-backup-")
    ]


def test_multi_platform_failure_leaves_prior_generation_intact(tmpdir):
    # type: (Tempdir) -> None

    fake_science = _install_fake_science(tmpdir)
    pex = tmpdir.join("src", "app.pex")
    os.makedirs(os.path.dirname(pex))
    _build_simple_pex(pex)

    dest = tmpdir.join("dest")
    os.makedirs(dest)
    foreign_platform = SysPlatform.MACOS_AARCH64
    current_scie = SysPlatform.CURRENT.qualified_binary_name("app")
    foreign_scie = foreign_platform.qualified_binary_name("app")
    with open(os.path.join(dest, current_scie), "w") as fp:
        fp.write("old-current-generation")
    with open(os.path.join(dest, foreign_scie), "w") as fp:
        fp.write("old-foreign-generation")

    result = run_pex3(
        "scie",
        "create",
        "--style",
        "lazy",
        "--scie-platform",
        "current",
        "--scie-platform",
        str(foreign_platform),
        "--scie-hash-alg",
        "sha256",
        "--scie-science-binary",
        fake_science,
        "-d",
        dest,
        pex,
        cwd=tmpdir.path,
        env=make_env(PEX_TEST_FAIL_SCIE_PLATFORM=str(foreign_platform)),
    )
    result.assert_failure()
    assert "science build stage" in result.error
    assert str(foreign_platform) in result.error

    # The complete prior generation is still visible and byte-for-byte intact; no platform from
    # the failed generation leaked into the destination directory and no staging directories
    # remain behind.
    with open(os.path.join(dest, current_scie)) as fp:
        assert "old-current-generation" == fp.read()
    with open(os.path.join(dest, foreign_scie)) as fp:
        assert "old-foreign-generation" == fp.read()
    # N.B.: The build stage fails before the commit lock is ever taken, so the destination only
    # contains the prior generation.
    assert sorted([current_scie, foreign_scie]) == sorted(os.listdir(dest))
    assert [] == _staging_leftovers(dest)


def test_scie_only_reclaims_in_place_source_only_after_successful_commit(tmpdir):
    # type: (Tempdir) -> None

    fake_science = _install_fake_science(tmpdir)
    dest = tmpdir.join("dest")
    os.makedirs(dest)
    # A single target uses the dynamic, unqualified scie name.
    current_scie = SysPlatform.CURRENT.binary_name("app")

    pex = os.path.join(dest, "app.pex")
    _build_simple_pex(pex)

    # A successful whole-generation switch reclaims the in-place source PEX.
    run_pex3(
        "scie",
        "create",
        "--style",
        "lazy",
        "--scie-only",
        "--scie-science-binary",
        fake_science,
        pex,
        cwd=tmpdir.path,
    ).assert_success()
    assert not os.path.exists(pex)
    assert os.path.isfile(os.path.join(dest, current_scie))

    # When the switch fails the in-place source PEX is restored and no scie is published.
    _build_simple_pex(pex)
    result = run_pex3(
        "scie",
        "create",
        "--style",
        "lazy",
        "--scie-only",
        "--scie-science-binary",
        fake_science,
        pex,
        cwd=tmpdir.path,
        env=make_env(PEX_TEST_FAIL_SCIE_PLATFORM=str(SysPlatform.CURRENT)),
    )
    result.assert_failure()
    assert os.path.isfile(pex)
    assert [] == _staging_leftovers(dest)


def test_checksum_sidecar_verification_failure_aborts_switch(tmpdir):
    # type: (Tempdir) -> None

    fake_science = _install_fake_science(tmpdir)
    pex = tmpdir.join("src", "app.pex")
    os.makedirs(os.path.dirname(pex))
    _build_simple_pex(pex)

    dest = tmpdir.join("dest")
    os.makedirs(dest)
    current_scie = SysPlatform.CURRENT.qualified_binary_name("app")
    old_sidecar = "{scie}.sha256".format(scie=current_scie)
    with open(os.path.join(dest, current_scie), "w") as fp:
        fp.write("old-current-generation")
    with open(os.path.join(dest, old_sidecar), "w") as fp:
        fp.write("old-sidecar")

    result = run_pex3(
        "scie",
        "create",
        "--style",
        "lazy",
        "--scie-hash-alg",
        "sha256",
        "--scie-science-binary",
        fake_science,
        "-d",
        dest,
        pex,
        cwd=tmpdir.path,
        env=make_env(PEX_TEST_BAD_SIDECAR=str(SysPlatform.CURRENT)),
    )
    result.assert_failure()
    assert "verification stage" in result.error
    assert str(SysPlatform.CURRENT) in result.error

    with open(os.path.join(dest, current_scie)) as fp:
        assert "old-current-generation" == fp.read()
    with open(os.path.join(dest, old_sidecar)) as fp:
        assert "old-sidecar" == fp.read()
    # Verification fails before the commit lock is taken, so only the prior generation remains.
    assert sorted([current_scie, old_sidecar]) == sorted(os.listdir(dest))
    assert [] == _staging_leftovers(dest)


def test_concurrent_publishers_commit_one_complete_generation(tmpdir):
    # type: (Tempdir) -> None

    fake_science = _install_fake_science(tmpdir)
    pex = tmpdir.join("src", "app.pex")
    os.makedirs(os.path.dirname(pex))
    _build_simple_pex(pex)

    dest = tmpdir.join("dest")
    os.makedirs(dest)

    foreign_platform = SysPlatform.MACOS_AARCH64
    current_scie = SysPlatform.CURRENT.qualified_binary_name("app")
    foreign_scie = foreign_platform.qualified_binary_name("app")

    python = installed_pex_wheel_venv_python(sys.executable)
    args = [
        "-m",
        "pex.cli",
        "scie",
        "create",
        "--style",
        "lazy",
        "--scie-platform",
        "current",
        "--scie-platform",
        str(foreign_platform),
        "--scie-science-binary",
        fake_science,
        "-d",
        dest,
        pex,
    ]

    processes = []
    for marker in ("AAA", "BBB"):
        env = make_env(PEX_TEST_SCIE_MARKER=marker, PEX_TEST_SCIE_SLEEP="1.5")
        processes.append(
            subprocess.Popen(
                args=[python] + args,
                cwd=tmpdir.path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
        time.sleep(0.5)

    for process in processes:
        stdout, stderr = process.communicate()
        assert 0 == process.returncode, stderr.decode("utf-8")

    def marker_of(name):
        # type: (str) -> object
        with open(os.path.join(dest, name)) as fp:
            match = re.search(r"marker=(\S+)", fp.read())
        assert match is not None
        return match.group(1)

    # Both processes published the same-named set concurrently; the destination must hold one
    # self-consistent generation rather than a mix of the two candidates.
    current_marker = marker_of(current_scie)
    foreign_marker = marker_of(foreign_scie)
    assert current_marker == foreign_marker
    assert current_marker in ("AAA", "BBB")
    assert [] == _staging_leftovers(dest)
    assert sorted(["app.pex", current_scie, foreign_scie, ".~pex-scie-create.lck"]) == sorted(
        os.listdir(dest)
    )


def test_platform_parent_dir_naming_commits_through_candidate(tmpdir):
    # type: (Tempdir) -> None

    fake_science = _install_fake_science(tmpdir)
    pex = tmpdir.join("src", "app.pex")
    os.makedirs(os.path.dirname(pex))
    _build_simple_pex(pex)

    dest = tmpdir.join("dest")
    os.makedirs(dest)
    foreign_platform = SysPlatform.MACOS_AARCH64

    run_pex3(
        "scie",
        "create",
        "--style",
        "lazy",
        "--scie-platform",
        "current",
        "--scie-platform",
        str(foreign_platform),
        "--scie-name-style",
        "platform-parent-dir",
        "--scie-science-binary",
        fake_science,
        "-d",
        dest,
        pex,
        cwd=tmpdir.path,
    ).assert_success()

    assert os.path.isfile(
        os.path.join(dest, str(SysPlatform.CURRENT), SysPlatform.CURRENT.binary_name("app"))
    )
    assert os.path.isfile(
        os.path.join(dest, str(foreign_platform), foreign_platform.binary_name("app"))
    )
    assert os.path.isfile(os.path.join(dest, "app.pex"))
    assert [] == _staging_leftovers(dest)
