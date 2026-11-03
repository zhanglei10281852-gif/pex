# Copyright 2026 Pex project contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from __future__ import absolute_import, print_function

import hashlib
import io
import os
import shutil
import tempfile
from argparse import ArgumentParser
from collections import OrderedDict

from pex import scie, specifier_sets
from pex.artifact_url import ArtifactURL
from pex.cache_helper import CacheHelper
from pex.cli.command import BuildTimeCommand
from pex.commands.command import OutputMixin
from pex.common import safe_mkdir
from pex.fetcher import URLFetcher
from pex.fs import lock as fs_lock
from pex.fs import safe_rename
from pex.interpreter_constraints import InterpreterConstraints
from pex.pep_440 import Version
from pex.pex_info import PexInfo
from pex.resolve import resolver_options, target_options
from pex.resolve.target_configuration import TargetConfiguration
from pex.result import Error, Ok, Result, catch, try_
from pex.scie import build as build_scies
from pex.scie.science import ScienceError
from pex.targets import Targets
from pex.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Optional, Tuple, Union

    import attr  # vendor:skip
else:
    from pex.third_party import attr


# A single `pex3 scie create` invocation publishes one indivisible generation of artifacts. All
# work happens in a per-invocation candidate directory staged inside the destination directory
# (so that final moves are same-file-system atomic renames). The candidate directory and the
# commit lock use dot-prefixed names that never collide with requested scie or PEX names and keep
# concurrent invocations isolated from each other.
_CANDIDATE_PREFIX = ".~pex-scie-create-"
_BACKUP_PREFIX = ".~pex-scie-backup-"
_COMMIT_LOCK_NAME = ".~pex-scie-create.lck"


@attr.s(frozen=True)
class _StagedInput(object):
    """The PEX staged into the candidate directory plus provenance about its source."""

    staged_pex = attr.ib()  # type: str
    pex_info = attr.ib()  # type: PexInfo
    # The absolute path of the local PEX the candidate was copied from, if any. This is only set
    # for a local path or file: URL and is the only path eligible for --scie-only reclamation.
    source = attr.ib(default=None)  # type: Optional[str]
    # Whether the source already lives directly in the destination directory and thus was used
    # in place prior to this generation.
    in_place = attr.ib(default=False)  # type: bool


@attr.s(frozen=True)
class _CommittedScie(object):
    scie_info = attr.ib()  # type: scie.ScieInfo
    final_file = attr.ib()  # type: str


def _narrow_interpreter_constraints(
    pex_interpreter_constraints,  # type: InterpreterConstraints
    user_interpreter_constraints,  # type: InterpreterConstraints
):
    # type: (...) -> Union[InterpreterConstraints, Error]

    if not pex_interpreter_constraints:
        return user_interpreter_constraints

    if not user_interpreter_constraints:
        return pex_interpreter_constraints

    for user_interpreter_constraint in user_interpreter_constraints:
        if not any(
            (
                (
                    not pex_interpreter_constraint.implementation
                    or (
                        user_interpreter_constraint.implementation
                        and pex_interpreter_constraint.implementation.includes(
                            user_interpreter_constraint.implementation,
                            user_interpreter_constraint.specifier,
                        )
                    )
                )
                and specifier_sets.includes(
                    pex_interpreter_constraint.specifier,
                    user_interpreter_constraint.specifier,
                )
            )
            for pex_interpreter_constraint in pex_interpreter_constraints
        ):
            return Error(
                "The PEX has interpreter constraints of {pex_interpreter_constraints} and the "
                "user-supplied interpreter constraints of {user_interpreter_constraints} do not "
                "form a subset of those.".format(
                    pex_interpreter_constraints=" or ".join(map(str, pex_interpreter_constraints)),
                    user_interpreter_constraints=" or ".join(
                        map(str, user_interpreter_constraints)
                    ),
                )
            )
    return user_interpreter_constraints


def _resolve_targets(
    pex_interpreter_constraints,  # type: InterpreterConstraints
    target_configuration,  # type: TargetConfiguration
):
    # type: (...) -> Union[Targets, Error]

    interpreter_constraints_or_error = _narrow_interpreter_constraints(
        pex_interpreter_constraints=pex_interpreter_constraints,
        user_interpreter_constraints=target_configuration.interpreter_constraints,
    )
    if isinstance(interpreter_constraints_or_error, Error):
        return interpreter_constraints_or_error

    if interpreter_constraints_or_error != target_configuration.interpreter_constraints:
        target_configuration = attr.evolve(
            target_configuration,
            interpreter_configuration=attr.evolve(
                target_configuration.interpreter_configuration,
                interpreter_constraints=interpreter_constraints_or_error,
            ),
        )
    return target_configuration.resolve_targets()


def _fetch(
    url,  # type: ArtifactURL
    fetcher,  # type: URLFetcher
    dest_dir=None,  # type: Optional[str]
):
    # type: (...) -> Union[str, Error]

    basename = os.path.basename(url.path)
    dest = os.path.join(safe_mkdir(dest_dir), basename) if dest_dir else basename

    digest = hashlib.new(url.fingerprint.algorithm) if url.fingerprint else None
    expected_fingerprint = url.fingerprint.hash if url.fingerprint else None

    expected_size = None  # type: Optional[int]
    expected_sizes = url.fragment_parameters.get("size")
    if expected_sizes and len(expected_sizes) == 1:
        expected_size = int(expected_sizes[0])

    if url.scheme == "file" and os.path.realpath(url.path) == os.path.realpath(dest):
        if expected_size is not None:
            size = os.path.getsize(url.path)
            if expected_size != size:
                return Error(
                    "Rejecting file at {path} with size {size} bytes since expected size is "
                    "{expected_size} bytes.".format(
                        path=url.path, size=size, expected_size=expected_size
                    )
                )

        if digest:
            fingerprint = CacheHelper.hash(path=url.path, digest=digest)
            if fingerprint != expected_fingerprint:
                return Error(
                    "File at {path} had unexpected fingerprint.\n"
                    "Expected: {expected_fingerprint}\n"
                    "Actual:   {actual_fingerprint}".format(
                        path=url.path,
                        expected_fingerprint=expected_fingerprint,
                        actual_fingerprint=fingerprint,
                    )
                )

        return dest

    size = 0
    with fetcher.get_body_stream(url.raw_url) as in_fp, open(dest, "wb") as out_fp:
        for chunk in iter(lambda: in_fp.read(io.DEFAULT_BUFFER_SIZE), b""):
            size += len(chunk)
            if expected_size is not None and size > expected_size:
                return Error(
                    "Terminating download of {url} at {size} bytes since expected size is "
                    "{expected_size} bytes.".format(
                        url=url.raw_url, size=size, expected_size=expected_size
                    )
                )

            if digest:
                digest.update(chunk)
            out_fp.write(chunk)

    if expected_size is not None and size < expected_size:
        return Error(
            "Download of {url} was too small.\n"
            "Expected: {expected_size} bytes\n"
            "Actual:   {actual_size} bytes".format(
                url=url.raw_url,
                expected_size=expected_size,
                actual_size=size,
            )
        )

    if digest:
        fingerprint = digest.hexdigest()
        if fingerprint != expected_fingerprint:
            return Error(
                "Download of {url} had unexpected fingerprint.\n"
                "Expected: {expected_fingerprint}\n"
                "Actual:   {actual_fingerprint}".format(
                    url=url.raw_url,
                    expected_fingerprint=expected_fingerprint,
                    actual_fingerprint=fingerprint,
                )
            )

    return dest


def _extract_pex_info(path):
    # type: (str) -> Union[Tuple[str, PexInfo], Error]

    pex_info_or_error = catch(PexInfo.from_pex, path)
    if isinstance(pex_info_or_error, Error):
        return Error(
            "The path {path} does not appear to be a PEX: {err}".format(
                path=path, err=pex_info_or_error
            )
        )

    raw_pex_version = pex_info_or_error.build_properties.get("pex_version")
    if raw_pex_version and Version(raw_pex_version) < Version("2.1.25"):
        return Error(
            "Can only create scies from PEXes built by Pex 2.1.25 (which was released on "
            "January 21st, 2021) or newer.\n"
            "The PEX at {path} was built by Pex {pex_version}.".format(
                path=path, pex_version=raw_pex_version
            )
        )

    return path, pex_info_or_error


def _is_in_place(path, dest_dir):
    # type: (str, str) -> bool
    return os.path.realpath(os.path.dirname(os.path.realpath(path))) == dest_dir


def _copy_into_candidate(path, candidate_dir):
    # type: (str, str) -> str
    staged = os.path.join(candidate_dir, os.path.basename(os.path.normpath(path)))
    if os.path.isfile(path):
        # N.B.: shutil.copy preserves the file's permission bits, matching the historical
        # behavior of staging a local PEX in the destination directory.
        shutil.copy(path, staged)
    else:
        shutil.copytree(path, staged)
    return staged


def _stage_input(
    candidate_dir,  # type: str
    source,  # type: str
    dest_dir,  # type: str
    url_fetcher,  # type: URLFetcher
):
    # type: (...) -> Union[_StagedInput, Error]
    """Stage the source PEX into the candidate generation directory.

    A local PEX is only ever copied, never modified, and a remote PEX is only ever downloaded into
    the candidate directory. Nothing is written to the destination directory at this stage.
    """

    if os.path.exists(source):
        extracted = _extract_pex_info(source)
        if isinstance(extracted, Error):
            return extracted
        source_path, pex_info = extracted
        staged_pex = _copy_into_candidate(source_path, candidate_dir)
        return _StagedInput(
            staged_pex=staged_pex,
            pex_info=pex_info,
            source=os.path.abspath(source_path),
            in_place=_is_in_place(source_path, dest_dir),
        )

    url = ArtifactURL.parse(source)

    # A file: URL pointing at a PEX already living directly in the destination directory is treated
    # as an in-place local input: validate any URL fragment fingerprint/size constraints against
    # the original file (without copying it) and then stage a private copy for the builds.
    if url.scheme == "file" and _is_in_place(url.path, dest_dir):
        source_path = try_(_fetch(url=url, fetcher=url_fetcher, dest_dir=dest_dir))
        extracted = _extract_pex_info(source_path)
        if isinstance(extracted, Error):
            return extracted
        source_path, pex_info = extracted
        staged_pex = _copy_into_candidate(source_path, candidate_dir)
        return _StagedInput(
            staged_pex=staged_pex,
            pex_info=pex_info,
            source=os.path.abspath(source_path),
            in_place=True,
        )

    staged_pex = try_(_fetch(url=url, fetcher=url_fetcher, dest_dir=candidate_dir))
    extracted = _extract_pex_info(staged_pex)
    if isinstance(extracted, Error):
        return Error(
            "The PEX fetched from {url} could not be used to create scies: {err}".format(
                url=source, err=extracted
            )
        )
    _, pex_info = extracted
    return _StagedInput(staged_pex=staged_pex, pex_info=pex_info)


def _verify_generation(
    scie_infos,  # type: Tuple[scie.ScieInfo, ...]
    hash_algorithms,  # type: Tuple[str, ...]
):
    # type: (...) -> Optional[Error]
    """Verify every built scie and requested checksum sidecar in the candidate directory."""

    for scie_info in scie_infos:
        target = scie_info.interpreter.render_description()
        scie_file = scie_info.file

        if not os.path.isfile(scie_file):
            return Error(
                "Failed to verify the scie for target {target} at the verification stage: the "
                "expected built scie {scie_file} is missing or is not a regular file.".format(
                    target=target, scie_file=scie_file
                )
            )
        if os.path.getsize(scie_file) == 0:
            return Error(
                "Failed to verify the scie for target {target} at the verification stage: the "
                "built scie {scie_file} is empty (0 bytes).".format(
                    target=target, scie_file=scie_file
                )
            )

        for algorithm in hash_algorithms:
            sidecar = "{scie_file}.{algorithm}".format(
                scie_file=scie_file, algorithm=algorithm
            )
            if not os.path.isfile(sidecar):
                return Error(
                    "Failed to verify the {algorithm} checksum sidecar for target {target} at the "
                    "verification stage: the expected checksum file {sidecar} is missing.".format(
                        algorithm=algorithm, target=target, sidecar=sidecar
                    )
                )
            try:
                with open(sidecar, "r") as fp:
                    recorded_fingerprint = fp.readline().strip().split()[0]
            except (IOError, OSError, IndexError) as e:
                return Error(
                    "Failed to verify the {algorithm} checksum sidecar {sidecar} for target "
                    "{target} at the verification stage: {err}".format(
                        algorithm=algorithm, sidecar=sidecar, target=target, err=e
                    )
                )
            try:
                actual_fingerprint = CacheHelper.hash(
                    scie_file, digest=hashlib.new(algorithm)
                )
            except ValueError:
                # The local hashlib cannot compute this algorithm, so only the presence of the
                # sidecar (produced by science) can be verified here.
                continue
            if recorded_fingerprint != actual_fingerprint:
                return Error(
                    "Failed to verify the {algorithm} checksum sidecar for target {target} at the "
                    "verification stage: the fingerprint recorded in {sidecar} "
                    "({recorded_fingerprint}) does not match the built scie {scie_file} "
                    "({actual_fingerprint}).".format(
                        algorithm=algorithm,
                        sidecar=sidecar,
                        target=target,
                        recorded_fingerprint=recorded_fingerprint,
                        scie_file=scie_file,
                        actual_fingerprint=actual_fingerprint,
                    )
                )
    return None


def _remove_path(path):
    # type: (str) -> None
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, True)
    elif os.path.lexists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _commit_generation(
    dest_dir,  # type: str
    artifacts,  # type: Tuple[Tuple[str, str], ...]
    reclaim_relpaths=(),  # type: Tuple[str, ...]
):
    # type: (...) -> Optional[Error]
    """Replace same-named prior outputs with the staged generation as one indivisible switch.

    An exclusive destination-directory-scoped file lock serializes concurrent publishers so the
    later finisher always swaps based on a complete prior generation. Existing outputs are moved
    aside into a backup directory on the same filesystem and are restored if any single move
    fails. The backup (the prior generation and, with `--scie-only`, the reclaimed source PEX) is
    deleted only after the whole switch has succeeded.
    """

    def error_at_commit_stage(rel, err):
        # type: (str, OSError) -> Error
        return Error(
            "Failed to commit the staged scie generation to {dest_dir} at the commit stage while "
            "installing {name}: {err}\n"
            "All previously existing files were restored; the destination directory is unchanged "
            "and the staged generation was discarded.".format(
                dest_dir=dest_dir, name=rel, err=err
            )
        )

    file_lock = fs_lock.acquire(
        os.path.join(dest_dir, _COMMIT_LOCK_NAME), exclusive=True
    )
    backup_dir = None  # type: Optional[str]
    try:
        backup_dir = tempfile.mkdtemp(dir=dest_dir, prefix=_BACKUP_PREFIX)

        displaced = OrderedDict()  # type: OrderedDict[str, str]
        created_dirs = []  # type: List[str]

        def displace(rel):
            # type: (str) -> None
            dst = os.path.join(dest_dir, rel)
            if os.path.lexists(dst):
                backup = os.path.join(backup_dir, rel)
                parent = os.path.dirname(backup)
                if parent != backup_dir and not os.path.isdir(parent):
                    safe_mkdir(parent)
                safe_rename(dst, backup)
                displaced[rel] = backup
            else:
                parent = os.path.dirname(dst)
                if parent and not os.path.isdir(parent):
                    safe_mkdir(parent)
                    created_dirs.append(parent)

        artifact_rels = frozenset(rel for _, rel in artifacts)
        reclaim_plan = tuple(rel for rel in reclaim_relpaths if rel not in artifact_rels)

        installed = []  # type: List[str]
        failure = None  # type: Optional[Tuple[str, OSError]]
        try:
            for rel in reclaim_plan:
                try:
                    displace(rel)
                except OSError as e:
                    failure = (rel, e)
                    break
            if failure is None:
                for src, rel in artifacts:
                    try:
                        displace(rel)
                        safe_rename(src, os.path.join(dest_dir, rel))
                        installed.append(rel)
                    except OSError as e:
                        failure = (rel, e)
                        break
        finally:
            if failure is not None:
                failed_rel, err = failure

                # Undo completed installs, then restore every displaced prior file.
                for rel in reversed(installed):
                    dst = os.path.join(dest_dir, rel)
                    _remove_path(dst)
                    backup = displaced.get(rel)
                    if backup and os.path.lexists(backup):
                        try:
                            safe_rename(backup, dst)
                        except OSError:
                            pass
                for rel, backup in displaced.items():
                    dst = os.path.join(dest_dir, rel)
                    if rel not in installed and not os.path.lexists(dst):
                        try:
                            safe_rename(backup, dst)
                        except OSError:
                            pass
                for directory in reversed(created_dirs):
                    try:
                        os.rmdir(directory)
                    except OSError:
                        pass

            if backup_dir and os.path.isdir(backup_dir):
                shutil.rmtree(backup_dir, True)

        if failure is not None:
            return error_at_commit_stage(failed_rel, err)
        return None
    finally:
        file_lock.release()


class Scie(OutputMixin, BuildTimeCommand):
    """Manipulate scies."""

    @classmethod
    def add_extra_arguments(cls, parser):
        subcommands = cls.create_subcommands(
            parser,
            description="Manipulate scies via the following commands.",
        )
        with subcommands.parser(
            name="create",
            help=(
                "Create one or more scies from an existing PEX file."
                "N.B.: The PEX must have been created with Pex v2.1.25 (released on January 21st, "
                "2021) or newer."
            ),
            func=cls._create,
            include_verbosity=False,
        ) as create_parser:
            cls._add_create_arguments(create_parser)

    @classmethod
    def _add_create_arguments(cls, parser):
        # type: (ArgumentParser) -> None
        parser.add_argument(
            "pex",
            nargs=1,
            help="The path or URL of a PEX to create one or more scies from.",
            metavar="PATH",
        )
        parser.add_argument("-d", "--dest-dir")
        cls.add_output_option(parser, "scie information")
        scie.register_options(
            parser.add_argument_group(title="Scie options"),
            style_option_names=("--style", "--scie", "--scie-style"),
        )
        target_options.register(
            parser.add_argument_group(title="Target interpreter options"), include_platforms=True
        )
        resolver_options.register(
            parser.add_argument_group(title="Pip options"),
        )

    def _create(self):
        # type: () -> Result

        scie_options = scie.extract_options(self.options)
        if not scie_options:
            return Error("You must specify `--style {eager,lazy}`.")

        resolver_configuration = resolver_options.configure(
            self.options,
            # N.B.: Since a scie freezes in whls via freezing in a PEX, there is no point to using
            # editable whls.
            honor_editable=False,
        )
        url_fetcher = URLFetcher(
            network_configuration=resolver_configuration.network_configuration,
            handle_file_urls=True,
            password_entries=resolver_configuration.repos_configuration.password_entries,
        )

        source = self.options.pex[0]
        if self.options.dest_dir:
            dest_dir = os.path.realpath(os.path.abspath(self.options.dest_dir))
        elif os.path.exists(source):
            # N.B.: For a local PEX path the outputs historically land next to the PEX when no
            # `--dest-dir` is given; a URL input instead lands in the current working directory.
            dest_dir = os.path.realpath(os.path.dirname(os.path.abspath(source)))
        else:
            dest_dir = os.path.realpath(os.getcwd())
        safe_mkdir(dest_dir)

        # One `pex3 scie create` invocation corresponds to one indivisible generation: the input is
        # copied or downloaded into this process's candidate directory, every target is built and
        # verified there, and only then are same-named prior outputs replaced as a whole under a
        # destination-scoped lock. Any failure leaves the pre-invocation set of files visible and
        # only ever cleans up this invocation's candidate directory.
        candidate_dir = tempfile.mkdtemp(dir=dest_dir, prefix=_CANDIDATE_PREFIX)
        try:
            # Stage 1: acquire the input into the private candidate directory.
            staged = _stage_input(
                candidate_dir=candidate_dir,
                source=source,
                dest_dir=dest_dir,
                url_fetcher=url_fetcher,
            )
            if isinstance(staged, Error):
                return staged

            targets = try_(
                _resolve_targets(
                    pex_interpreter_constraints=staged.pex_info.interpreter_constraints,
                    target_configuration=target_options.configure(
                        self.options,
                        pip_configuration=resolver_configuration.pip_configuration,
                    ),
                )
            )

            scie_configuration = scie_options.create_configuration(targets=targets)
            if not scie_configuration:
                return Error(
                    "You selected `{scie_options}`, but none of the selected targets have "
                    "compatible interpreters that can be embedded to form a scie:\n{targets}".format(
                        scie_options=scie.render_options(scie_options),
                        targets="\n".join(
                            target.render_description()
                            for target in targets.unique_targets()
                        ),
                    )
                )

            # Stage 2: build a scie for every target into the candidate directory. A failure for
            # any target aborts the whole generation; nothing has been published yet.
            try:
                scie_infos = tuple(
                    build_scies(
                        configuration=scie_configuration,
                        pex_file=staged.staged_pex,
                        url_fetcher=url_fetcher,
                        dest_dir=candidate_dir,
                    )
                )
            except ScienceError as e:
                return Error(
                    "Aborted creation of the scie generation for {source} at the science build "
                    "stage. No files in {dest_dir} were changed and the staged candidate "
                    "directory was cleaned up.\n{err}".format(
                        source=source, dest_dir=dest_dir, err=e
                    )
                )

            # Stage 3: verify every named result and requested checksum sidecar in the candidate.
            verification_error = _verify_generation(
                scie_infos=scie_infos,
                hash_algorithms=scie_configuration.options.hash_algorithms,
            )
            if verification_error:
                return Error(
                    "Aborted creation of the scie generation for {source} at the verification "
                    "stage. No files in {dest_dir} were changed and the staged candidate "
                    "directory was cleaned up.\n{err}".format(
                        source=source, dest_dir=dest_dir, err=verification_error
                    )
                )

            # Assemble the complete set of candidate artifacts keyed by destination relpath.
            artifacts = OrderedDict()  # type: OrderedDict[str, str]
            final_scies = []  # type: List[Tuple[str, scie.ScieInfo]]
            for scie_info in scie_infos:
                rel = os.path.relpath(scie_info.file, candidate_dir)
                artifacts[rel] = scie_info.file
                final_scies.append((rel, scie_info))
                for algorithm in scie_configuration.options.hash_algorithms:
                    sidecar_rel = "{rel}.{algorithm}".format(rel=rel, algorithm=algorithm)
                    artifacts[sidecar_rel] = os.path.join(candidate_dir, sidecar_rel)

            staged_pex_rel = os.path.basename(os.path.normpath(staged.staged_pex))
            reclaim_relpaths = ()  # type: Tuple[str, ...]
            if scie_configuration.options.scie_only:
                # With --scie-only the source PEX is only reclaimed when it was an in-place user
                # file and only as part of the successful whole-generation switch. When the PEX
                # name collides with a scie name, the switch itself replaces the source, so no
                # separate reclamation is needed.
                if staged.in_place and staged_pex_rel not in artifacts:
                    reclaim_relpaths = (staged_pex_rel,)
            else:
                # Publish the PEX alongside the scies, overwriting any same-named prior PEX as
                # part of the generation switch. setdefault handles a PEX whose name collides
                # with its scie's name (the scie already overwrote the staged copy).
                artifacts.setdefault(staged_pex_rel, staged.staged_pex)

            # Stage 4: switch the whole generation into place atomically with respect to other
            # processes publishing into the same directory.
            commit_error = _commit_generation(
                dest_dir=dest_dir,
                artifacts=tuple((src, rel) for rel, src in artifacts.items()),
                reclaim_relpaths=reclaim_relpaths,
            )
            if commit_error:
                return commit_error

            with self.output(self.options) as out:
                for rel, scie_info in final_scies:
                    final_scie = os.path.join(dest_dir, rel)
                    print(
                        "Saved PEX scie for {python_description} to {scie}".format(
                            python_description=scie_info.interpreter.render_description(),
                            scie=os.path.relpath(final_scie),
                        ),
                        file=out,
                    )

            return Ok()
        finally:
            shutil.rmtree(candidate_dir, True)
