# Copyright 2024 Pex project contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from __future__ import absolute_import, print_function

import itertools
import os
import re
from argparse import Action, ArgumentError, _ActionsContainer
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timedelta

from pex.cache.dirs import (
    AtomicCacheDir,
    BootstrapDir,
    BootstrapZipDir,
    BuiltWheelDir,
    CacheDir,
    DownloadDir,
    InstalledWheelDir,
    InterpreterDir,
    PackedWheelDir,
    VenvDirs,
)
from pex.cache.prunable import PexParent, Prunable
from pex.cli.command import BuildTimeCommand
from pex.cli.commands.cache.bytes import ByteAmount, ByteUnits
from pex.cli.commands.cache.du import DiskUsage
from pex.cli.commands.cache_aware import CacheAwareMixin
from pex.commands.command import OutputMixin
from pex.common import pluralize, rmtree, safe_rmtree
from pex.dist_metadata import ProjectNameAndVersion, UnrecognizedDistributionFormat
from pex.exceptions import reportable_unexpected_error_msg
from pex.jobs import (
    Retain,
    SpawnedJob,
    execute_parallel,
    iter_map_parallel,
    map_parallel,
)
from pex.orderedset import OrderedSet
from pex.pep_440 import Version
from pex.pep_503 import ProjectName
from pex.pip.installation import iter_all as iter_all_pips
from pex.pip.tool import Pip
from pex.result import Error, Ok, Result
from pex.typing import TYPE_CHECKING
from pex.variables import ENV

if TYPE_CHECKING:
    import typing
    from typing import (
        DefaultDict,
        Iterable,
        Iterator,
        List,
        Mapping,
        Optional,
        Set,
        Tuple,
        Union,
    )

    import attr  # vendor:skip
else:
    from pex.third_party import attr


class HandleAmountAction(Action):
    def __init__(self, *args, **kwargs):
        kwargs["nargs"] = 0
        super(HandleAmountAction, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, value, option_str=None):
        if option_str in ("-H", "--human-readable"):
            amount_func = ByteAmount.human_readable
        elif option_str.startswith("--"):
            amount_func = ByteAmount.for_unit(ByteUnits.for_value(option_str[2:]))
        else:
            raise ArgumentError(
                argument=self,
                message=reportable_unexpected_error_msg(
                    "The HandleAmountAction was used in an unexpected place."
                ),
            )
        setattr(namespace, self.dest, amount_func)


@attr.s(frozen=True)
class Cutoff(object):
    @classmethod
    def parse(cls, spec):
        # type: (str) -> Cutoff
        match = re.match(
            r"(?P<amount>\d+)\s+(?P<unit>second|minute|hour|day|week)s?(\s+ago)?",
            spec.strip(),
            re.IGNORECASE,
        )
        if match:
            args = {match.group("unit") + "s": int(match.group("amount"))}
            cutoff = datetime.now() - timedelta(**args)
        else:
            cutoff = datetime.strptime(spec.strip(), "%d/%m/%Y")
        return cls(spec=spec, cutoff=cutoff)

    spec = attr.ib()  # type: str
    cutoff = attr.ib()  # type: datetime


# Per-entry prune statuses. A dependency whose candidate consumers failed to be deleted is
# `skipped`; a path whose deletion was attempted but which survives is `failed`; only paths
# confirmed gone are `succeeded`.
SUCCEEDED = "succeeded"
FAILED = "failed"
SKIPPED = "skipped"


@attr.s(frozen=True)
class PrunedPath(object):
    path = attr.ib()  # type: str
    status = attr.ib()  # type: str
    disk_usage = attr.ib()  # type: DiskUsage
    error = attr.ib(default=None)  # type: Optional[str]


@attr.s(frozen=True)
class PrunedEntry(object):
    noun = attr.ib()  # type: str
    path = attr.ib()  # type: str
    paths = attr.ib()  # type: Tuple[PrunedPath, ...]
    status = attr.ib()  # type: str
    detail = attr.ib(default=None)  # type: Optional[str]

    @staticmethod
    def skipped(
        noun,  # type: str
        path,  # type: str
        reason,  # type: str
    ):
        # type: (...) -> PrunedEntry
        return PrunedEntry(noun=noun, path=path, paths=(), status=SKIPPED, detail=reason)

    @staticmethod
    def failed_external(
        noun,  # type: str
        path,  # type: str
        error,  # type: str
    ):
        # type: (...) -> PrunedEntry
        return PrunedEntry(noun=noun, path=path, paths=(), status=FAILED, detail=error)

    @property
    def disk_usage(self):
        # type: () -> DiskUsage
        return DiskUsage.aggregate(
            self.path,
            [
                pruned_path.disk_usage
                for pruned_path in self.paths
                if pruned_path.status == SUCCEEDED
            ],
        )


@attr.s(frozen=True)
class PrunePaths(object):
    """A unit of deletion: one logical entry backed by one or more on-disk paths."""

    noun = attr.ib()  # type: str
    path = attr.ib()  # type: str
    paths = attr.ib()  # type: Tuple[str, ...]


if TYPE_CHECKING:
    PipActions = Tuple[List[PrunePaths], "OrderedDict[str, Pip]"]


def _zero_disk_usage(path=""):
    # type: (str) -> DiskUsage
    return DiskUsage(path=path, subdirs=0, files=0, size=0)


def _prune_paths(prune_paths):
    # type: (PrunePaths) -> PrunedEntry
    """Delete a planned entry, verifying each backing path is actually gone.

    Errors never escape: a path that cannot be removed is reported as `failed` so unrelated
    entries can still be processed and so dependent entries can be retained.
    """
    pruned_paths = []  # type: List[PrunedPath]
    errors = []  # type: List[str]
    for path in prune_paths.paths:
        # A path already removed by a prior run (or by a companion record in this same entry) is
        # not a failure: re-running with the same cutoff simply re-converges against the disk.
        if not os.path.lexists(path):
            pruned_paths.append(
                PrunedPath(path=path, status=SUCCEEDED, disk_usage=_zero_disk_usage(path))
            )
            continue
        disk_usage = DiskUsage.collect(path)
        error = None  # type: Optional[str]
        try:
            rmtree(path)
        except OSError as e:
            error = str(e)
        if error is None and os.path.lexists(path):
            error = "The path still exists after deletion was attempted."
        if error is None:
            pruned_paths.append(
                PrunedPath(path=path, status=SUCCEEDED, disk_usage=disk_usage)
            )
        else:
            pruned_paths.append(
                PrunedPath(
                    path=path,
                    status=FAILED,
                    disk_usage=_zero_disk_usage(path),
                    error=error,
                )
            )
            errors.append("{path}: {error}".format(path=path, error=error))
    return PrunedEntry(
        noun=prune_paths.noun,
        path=prune_paths.path,
        paths=tuple(pruned_paths),
        status=FAILED if errors else SUCCEEDED,
        detail="; ".join(errors) if errors else None,
    )


def _append_unique(
    paths,  # type: List[str]
    path,  # type: Optional[str]
):
    # type: (...) -> None
    if path and path not in paths:
        paths.append(path)


def _dependency_backing_paths(
    deps,  # type: Iterable[AtomicCacheDir]
    additional_cache_dirs_by_project_name_and_version,
    # type: Mapping[Tuple[ProjectName, Version], Iterable[AtomicCacheDir]]
):
    # type: (...) -> Tuple[str, ...]
    paths = []  # type: List[str]
    for cache_dir in deps:
        if isinstance(cache_dir, InstalledWheelDir):
            # The real wheel chroot directory (the install-hash dir for a plain copy or the
            # wheel-hash chroot directory for symlink-style installs).
            _append_unique(paths, os.path.dirname(cache_dir.path))
            for packed_wheel_dir in PackedWheelDir.iter(cache_dir.install_hash):
                _append_unique(paths, packed_wheel_dir.path)
            for additional_dir in additional_cache_dirs_by_project_name_and_version.get(
                (cache_dir.project_name, cache_dir.version), ()
            ):
                if os.path.lexists(additional_dir.path):
                    _append_unique(paths, additional_dir.path)
            if cache_dir.symlink_dir:
                _append_unique(paths, cache_dir.symlink_dir)
        elif isinstance(cache_dir, BootstrapDir):
            _append_unique(paths, cache_dir.path)
            for bootstrap_zip_dir in BootstrapZipDir.iter(cache_dir.bootstrap_hash):
                _append_unique(paths, bootstrap_zip_dir.path)
        else:
            _append_unique(paths, cache_dir.path)
    return tuple(path for path in paths if os.path.lexists(path))


def _parent_backing_paths(pex_parent):
    # type: (PexParent) -> PrunePaths
    paths = [pex_parent.pex_dir.path]  # type: List[str]
    if isinstance(pex_parent.pex_dir, VenvDirs):
        _append_unique(paths, pex_parent.pex_dir.short_dir)
    return PrunePaths(
        noun="cached PEX",
        path=pex_parent.pex_dir.path,
        paths=tuple(path for path in paths if os.path.lexists(path)),
    )


class Cache(CacheAwareMixin, OutputMixin, BuildTimeCommand):
    """Interact with the Pex cache."""

    @staticmethod
    def _add_amount_argument(parser):
        # type: (_ActionsContainer) -> None

        parser.add_argument(
            "-H",
            "--human-readable",
            *["--{unit}".format(unit=unit.value) for unit in ByteUnits.values()],
            dest="amount_func",
            action=HandleAmountAction,
            default=ByteAmount.bytes,
            help=(
                "How to display disk usage amounts; defaults to bytes. The -H / --human-readable "
                "options display amounts in a human readable way and each of the other unit "
                "options displays amounts with that specified unit."
            )
        )

    @classmethod
    def _add_info_arguments(cls, parser):
        # type: (_ActionsContainer) -> None

        cls._add_amount_argument(parser)
        parser.add_argument(
            "-S",
            "--sort-size",
            dest="sort_by_size",
            action="store_true",
            help=(
                "Sort cache entry information by the total size of the cache entry. "
                "Entry information is sorted by entry name by default."
            ),
        )
        parser.add_argument(
            "-r",
            "--reverse",
            dest="reverse",
            action="store_true",
            help="Reverse the sorting of cache entry information.",
        )
        cls.add_output_option(parser, entity="Pex cache information")

    @staticmethod
    def _add_dry_run_option(parser):
        # type: (_ActionsContainer) -> None

        parser.add_argument(
            "-n",
            "--dry-run",
            dest="dry_run",
            action="store_true",
            help=(
                "Don't actually purge cache entries; instead, perform a dry run that just prints "
                "out what actions would be taken"
            ),
        )

    @classmethod
    def _add_purge_arguments(cls, parser):
        # type: (_ActionsContainer) -> None

        cls._add_amount_argument(parser)
        parser.add_argument(
            "--entries",
            action="append",
            type=CacheDir.for_value,
            choices=[cache_dir for cache_dir in CacheDir.values() if cache_dir.can_purge],
            default=[],
            help=(
                "Specific cache entries to purge. By default, all entries are purged, but by "
                "specifying one or more particular --entries to purge, only those entries (and any "
                "other cache entries dependent on those) will be purged."
            ),
        )
        cls._add_dry_run_option(parser)
        cls.add_output_option(parser, entity="Pex purge results")

    @classmethod
    def _add_prune_arguments(cls, parser):
        # type: (_ActionsContainer) -> None

        cls._add_amount_argument(parser)
        parser.add_argument(
            "--older-than",
            "--last-access",
            "--last-access-before",
            dest="cutoff",
            type=Cutoff.parse,
            default=Cutoff.parse("2 weeks ago"),
            help=(
                "Prune zipapp and venv caches (amongst others) last accessed before the specified "
                "time. If the dependencies of the selected zipapps and venvs (e.g.: installed "
                "wheels) are unused by other zipapps and venvs, those dependencies are pruned as "
                "well. The cutoff time can be specified as a date in the format "
                "`<day number>/<month number>/<4 digit year>` or as a relative time in the format "
                "`<amount> [second(s)|minute(s)|hour(s)|day(s)|week(s)]`."
            ),
        )
        cls._add_dry_run_option(parser)
        cls.add_output_option(parser, entity="Pex purge results")

    @classmethod
    def add_extra_arguments(cls, parser):
        subcommands = cls.create_subcommands(
            parser,
            description="Interact with the Pex cache via the following subcommands.",
        )

        with subcommands.parser(
            name="dir",
            help="Print the current Pex cache directory path.",
            func=cls._dir,
            include_verbosity=False,
        ) as dir_parser:
            cls.add_output_option(dir_parser, entity="Pex cache directory")

        with subcommands.parser(
            name="info",
            help="Present information about Pex cache status.",
            func=cls._info,
            include_verbosity=False,
        ) as inspect_parser:
            cls._add_info_arguments(inspect_parser)

        with subcommands.parser(
            name="purge",
            help="Purge the Pex cache safely.",
            func=cls._purge,
            include_verbosity=False,
        ) as purge_parser:
            cls._add_purge_arguments(purge_parser)

        with subcommands.parser(
            name="prune",
            help="Prune the Pex cache safely.",
            func=cls._prune,
            include_verbosity=False,
        ) as prune_parser:
            cls._add_prune_arguments(prune_parser)

    def _dir(self):
        # type: () -> Result

        with self.output(self.options) as fp:
            print(ENV.PEX_ROOT, file=fp)

        return Ok()

    @staticmethod
    def _collect_info(cache_dir):
        # type: (CacheDir.Value) -> Tuple[CacheDir.Value, DiskUsage]
        return cache_dir, DiskUsage.collect(cache_dir.path())

    def _render_usage(self, disk_usage):
        # type: (Union[DiskUsage, Iterable[DiskUsage]]) -> str

        if isinstance(disk_usage, DiskUsage):
            du = disk_usage
            prefix = ""
        else:
            du = DiskUsage.aggregate(path=ENV.PEX_ROOT, usages=disk_usage)
            prefix = "Total: "

        return "{prefix}{size} in {subdir_count} {subdirs} and {file_count} {files}.".format(
            prefix=prefix,
            size=self.options.amount_func(du.size),
            subdir_count=du.subdirs,
            subdirs=pluralize(du.subdirs, "subdirectory"),
            file_count=du.files,
            files=pluralize(du.files, "file"),
        )

    def _info(self):
        # type: () -> Result

        with self.output(self.options) as fp:
            print("Path: {path}".format(path=ENV.PEX_ROOT), file=fp)
            print(file=fp)

            disk_usages = []  # type: List[DiskUsage]
            for cache_dir, disk_usage in sorted(
                map_parallel(
                    inputs=CacheDir.values(), function=self._collect_info, noun="cache directory"
                ),
                key=lambda info: (
                    (info[1].size, info[0].name) if self.options.sort_by_size else info[0].name
                ),
                reverse=self.options.reverse,
            ):
                disk_usages.append(disk_usage)
                print(
                    "{name}: {path}".format(name=cache_dir.name, path=cache_dir.rel_path()), file=fp
                )
                print(cache_dir.description, file=fp)
                print(self._render_usage(disk_usage), file=fp)
                print(file=fp)
            print(self._render_usage(disk_usages), file=fp)
            print(file=fp)

        return Ok()

    def _purge_cache_dir(self, cache_dir):
        # type: (CacheDir.Value) -> Tuple[CacheDir.Value, DiskUsage]
        cache_dir_path = cache_dir.path()
        du = DiskUsage.collect(cache_dir_path)
        if not self.options.dry_run:
            safe_rmtree(cache_dir_path)
        return cache_dir, du

    def _purge(self):
        # type: () -> Result

        with self.output(self.options) as fp:
            cache_dirs = OrderedSet()  # type: OrderedSet[CacheDir.Value]
            if self.options.entries:
                cache_dirs.update(self.options.entries)
                print(
                    "{purging} requested entries from {pex_root}: {cache_dirs}".format(
                        purging="Would purge" if self.options.dry_run else "Purging",
                        pex_root=ENV.PEX_ROOT,
                        cache_dirs=", ".join(cache_dir.rel_path() for cache_dir in cache_dirs),
                    ),
                    file=fp,
                )

                dependents = OrderedSet()  # type: OrderedSet[CacheDir.Value]
                for cache_dir in cache_dirs:
                    dependents.update(cache_dir.iter_transitive_dependents())
                if dependents:
                    print(
                        "{purging} those entries transitive dependents in: {dependents}".format(
                            purging="Would also purge" if self.options.dry_run else "Also purging",
                            dependents=", ".join(dep.rel_path() for dep in dependents),
                        ),
                        file=fp,
                    )
                    cache_dirs.update(dependents)
            else:
                print(
                    "{purging} all cache entries from {pex_root}:".format(
                        purging="Would purge" if self.options.dry_run else "Purging",
                        pex_root=ENV.PEX_ROOT,
                    ),
                    file=fp,
                )
                cache_dirs.update(CacheDir.values())
            print(file=fp)

            if not self.options.dry_run:
                locked = self.lock_cache_for_delete(out=fp)
                print(file=fp)
                if not locked:
                    return Error("No cache entries purged.")

            disk_usages = []  # type: List[DiskUsage]
            for cache_dir, du in iter_map_parallel(
                cache_dirs, self._purge_cache_dir, noun="entry", verb="purge", verb_past="purged"
            ):
                print(
                    "{purged} cache {name} from {rel_path}".format(
                        purged="Would have purged" if self.options.dry_run else "Purged",
                        name=cache_dir.name,
                        rel_path=cache_dir.rel_path(),
                    ),
                    file=fp,
                )
                disk_usages.append(du)
                print(self._render_usage(du), file=fp)
                print(file=fp)
            if disk_usages:
                print(self._render_usage(disk_usages), file=fp)
                print(file=fp)

        return Ok()

    @staticmethod
    def _cutoff_display(cutoff):
        # type: (Cutoff) -> str
        if cutoff.spec.endswith("ago") or cutoff.spec[-1].isdigit():
            return cutoff.spec
        return "{cutoff} ago".format(cutoff=cutoff.spec)

    def _print_entry(
        self,
        fp,  # type: typing.IO[str]
        result,  # type: PrunedEntry
    ):
        # type: (...) -> None
        if result.status == SKIPPED:
            print(
                "  skipped: {noun} {path} ({detail})".format(
                    noun=result.noun, path=result.path, detail=result.detail
                ),
                file=fp,
            )
        elif result.status == FAILED:
            print(
                "  failed: {noun} {path}: {detail}".format(
                    noun=result.noun, path=result.path, detail=result.detail
                ),
                file=fp,
            )
        else:
            reclaimed = result.disk_usage.size
            reclaimed_msg = (
                " (reclaimed {size})".format(size=self.options.amount_func(reclaimed))
                if reclaimed
                else ""
            )
            print(
                "  succeeded: {noun} {path}{reclaimed}".format(
                    noun=result.noun, path=result.path, reclaimed=reclaimed_msg
                ),
                file=fp,
            )

    def _execute_entries(
        self,
        fp,  # type: typing.IO[str]
        heading,  # type: str
        entries,  # type: Tuple[PrunePaths, ...]
        noun,  # type: str
        extra_results=(),  # type: Iterable[PrunedEntry]
    ):
        # type: (...) -> List[PrunedEntry]
        if not entries and not extra_results:
            return []

        print(heading, file=fp)
        results = list(extra_results)  # type: List[PrunedEntry]
        if entries:
            # A failure in one entry never blocks the others; each result is reported individually
            # as succeeded, failed or (by the caller for blocked dependencies) skipped.
            results.extend(
                iter_map_parallel(
                    entries,
                    _prune_paths,
                    noun=noun,
                    verb="prune",
                    verb_past="pruned",
                )
            )
        results.sort(key=lambda r: (r.status, r.noun, r.path))
        for result in results:
            self._print_entry(fp, result)
        print(file=fp)
        return results

    @staticmethod
    def _pip_actions(
        plan=None,
        surviving_candidate_parents=None,
        all_pips=None,
    ):
        # type: (...) -> PipActions
        """Resolves Pip installations to prune and Pip versions retained to run cache ops against.

        When `plan` is passed the resolution reflects scan-time intent (used for the plan /
        dry-run): a Pip whose owning PEX is a prune candidate is itself a candidate. When
        `surviving_candidate_parents` is passed the resolution reflects the current disk after
        parent deletion was attempted: any candidate parent that survived (including via a failed
        deletion) retains both its Pip PEX and Pip cache.

        `all_pips` is a pre-deletion snapshot: instantiating a `Pip` ensures its venv exists and
        the Pip PEX `.deps` symlink into the shared installed wheel chroots, so re-scanning after
        wheel chroots are deleted would chase dangling symlinks. The snapshot is always taken
        before any deletion and reused.
        """
        if all_pips is None:
            all_pips = list(iter_all_pips(record_access=False))
        if surviving_candidate_parents is not None:
            # Start with every PEX dir currently on disk (recent or concurrent) and then account
            # for candidate parents that survived their deletion attempt directly, closing the
            # gap where a removed venv short-hash symlink would hide a surviving venv from a glob.
            surviving_hashes = {
                pex_dir.pex_hash for pex_dir in Prunable.iter_current_pex_dirs()
            }
            for pex_parent in surviving_candidate_parents:
                if os.path.lexists(pex_parent.pex_dir.path):
                    surviving_hashes.add(pex_parent.pex_dir.pex_hash)
        else:
            candidate_hashes = {
                pex_parent.pex_dir.pex_hash for pex_parent in (plan.parents if plan else ())
            }
            surviving_hashes = {
                pip.pex_hash for pip in all_pips if pip.pex_hash not in candidate_hashes
            }

        retained_versions = OrderedSet(
            pip.version for pip in all_pips if pip.pex_hash in surviving_hashes
        )
        retained_pips = OrderedDict()  # type: OrderedDict[str, Pip]
        for pip in all_pips:
            if pip.version in retained_versions:
                retained_pips.setdefault(str(pip.version), pip)

        entries = []  # type: List[PrunePaths]
        seen_paths = set()  # type: Set[str]
        for pip in all_pips:
            if pip.pex_hash in surviving_hashes:
                continue
            full_version_prune = pip.version not in retained_versions
            remove_path = pip.pex_dir.base_dir if full_version_prune else pip.pex_dir.path
            if remove_path in seen_paths or not os.path.lexists(remove_path):
                continue
            seen_paths.add(remove_path)
            entries.append(
                PrunePaths(
                    noun="Pip" if full_version_prune else "Pip PEX",
                    path=remove_path,
                    paths=(remove_path,),
                )
            )
        return entries, retained_pips

    def _pip_cache_results(
        self,
        fp,  # type: typing.IO[str]
        retained_pips,  # type: Mapping[str, Pip]
        removable_wheels,  # type: Set[Tuple[ProjectName, Version]]
    ):
        # type: (...) -> List[PrunedEntry]
        if not removable_wheels or not retained_pips:
            return []

        pips = list(retained_pips.values())
        results = []  # type: List[PrunedEntry]

        def spawn_list(pip):
            # type: (Pip) -> SpawnedJob[Tuple[Pip, bytes]]
            return SpawnedJob.stdout(
                job=pip.spawn_cache_list(),
                result_func=lambda stdout: (pip, stdout),
            )

        pip_removes = []  # type: List[Tuple[Pip, str, ProjectNameAndVersion]]
        for list_result in execute_parallel(
            inputs=pips, spawn_func=spawn_list, error_handler=Retain()
        ):
            pip, payload = list_result
            if isinstance(payload, Exception):
                # A `pip cache list` failure is independent of every other Pip: record it and skip
                # removes for this Pip only; completed filesystem operations are not rolled back.
                results.append(
                    PrunedEntry.failed_external(
                        noun="Pip cache list",
                        path=str(pip.version),
                        error="`pip cache list` failed: {error}".format(error=payload),
                    )
                )
                continue
            try:
                project_name_and_versions = tuple(
                    ProjectNameAndVersion.from_filename(wheel_file)
                    for wheel_file in payload.decode("utf-8").splitlines()
                    if wheel_file
                )
            except (UnicodeDecodeError, UnrecognizedDistributionFormat) as e:
                results.append(
                    PrunedEntry.failed_external(
                        noun="Pip cache list",
                        path=str(pip.version),
                        error="could not parse `pip cache list` output: {error}".format(error=e),
                    )
                )
                continue
            for pnav in project_name_and_versions:
                if (
                    pnav.canonicalized_project_name,
                    pnav.canonicalized_version,
                ) in removable_wheels:
                    pip_removes.append(
                        (
                            pip,
                            "{project_name}-{version}*".format(
                                project_name=pnav.project_name, version=pnav.version
                            ),
                            pnav,
                        )
                    )

        if not pip_removes:
            return results

        files_removed_pattern = re.compile(rb"Files removed:\s*(\d+)")

        def spawn_remove(args):
            # type: (Tuple[Pip, str, ProjectNameAndVersion]) -> SpawnedJob
            pip, wheel_name_glob, _pnav = args
            return SpawnedJob.stdout(
                job=pip.spawn_cache_remove(wheel_name_glob),
                result_func=lambda stdout: (args, stdout),
            )

        removes_by_pip = Counter()  # type: typing.Counter[str]
        for remove_result in execute_parallel(
            inputs=pip_removes, spawn_func=spawn_remove, error_handler=Retain()
        ):
            args, payload = remove_result
            pip, wheel_name_glob, _pnav = args
            if isinstance(payload, Exception):
                results.append(
                    PrunedEntry.failed_external(
                        noun="Pip cache remove",
                        path="{pip}: {glob}".format(pip=pip.version, glob=wheel_name_glob),
                        error="`pip cache remove {glob}` failed: {error}".format(
                            glob=wheel_name_glob, error=payload
                        ),
                    )
                )
                continue
            # Unlike the prior parser, unparseable output (or an empty/zero parse) is not silently
            # treated as success: only the documented "Files removed: N" line counts.
            match = files_removed_pattern.search(payload)
            if not match:
                results.append(
                    PrunedEntry.failed_external(
                        noun="Pip cache remove",
                        path="{pip}: {glob}".format(pip=pip.version, glob=wheel_name_glob),
                        error="could not parse `pip cache remove` output: {output!r}".format(
                            output=payload.decode("utf-8", errors="replace").strip()
                        ),
                    )
                )
                continue
            removes_by_pip[pip.version.value] += int(match.group(1))

        if removes_by_pip:
            total = sum(removes_by_pip.values())
            print(
                "Pruned {total} cached {wheels} from {count} Pip {version}:".format(
                    total=total,
                    wheels=pluralize(total, "wheel"),
                    count=len(removes_by_pip),
                    version=pluralize(len(removes_by_pip), "version"),
                ),
                file=fp,
            )
            for pip_version, remove_count in sorted(removes_by_pip.items()):
                print(
                    "Pip {version}: removed {remove_count} {wheels}".format(
                        version=pip_version,
                        remove_count=remove_count,
                        wheels=pluralize(remove_count, "wheel"),
                    ),
                    file=fp,
                )
            print(file=fp)
        return results

    @staticmethod
    def _interpreter_prune_paths(candidate_venv_paths=None):
        # type: (Optional[Set[str]]) -> List[PrunePaths]

        def eligible(interpreter_dir):
            # type: (InterpreterDir) -> bool
            try:
                valid = interpreter_dir.valid()
            except (OSError, ValueError):
                # An unreadable or corrupt INTERP-INFO file is as invalid as a missing
                # interpreter binary and should never abort the rest of the prune.
                valid = False
            if not valid:
                return True
            venv_dir = interpreter_dir.venv_dir()
            if venv_dir is None:
                return False
            if candidate_venv_paths is None:
                # Execution time: the interpreter cache entry is only pruned once its venv is
                # confirmed gone; a failed venv deletion retains it.
                return not os.path.lexists(venv_dir.path)
            return venv_dir.path in candidate_venv_paths

        return [
            PrunePaths(
                noun="cached interpreter",
                path=interpreter_dir.path,
                paths=(interpreter_dir.path,),
            )
            for interpreter_dir in InterpreterDir.iter_all()
            if eligible(interpreter_dir)
        ]

    def _prune(self):
        # type: () -> Result

        dry_run = self.options.dry_run
        cutoff = self.options.cutoff

        with self.output(self.options) as fp:
            if not dry_run:
                locked = self.lock_cache_for_delete(out=fp)
                print(file=fp)
                if not locked:
                    return Error("No cache entries pruned.")

            plan = Prunable.scan(cutoff.cutoff)

            # Snapshot Pip installations before anything is deleted: instantiating a Pip ensures
            # its venv and follows the Pip PEX `.deps` symlinks into shared installed wheel
            # chroots, which phase 2 may remove; a post-deletion re-scan would be invalid.
            all_pips = list(iter_all_pips(record_access=False))

            additional_cache_dirs_by_project_name_and_version = defaultdict(
                list
            )  # type: DefaultDict[Tuple[ProjectName, Version], List[AtomicCacheDir]]
            cached_artifact_dirs = itertools.chain(
                BuiltWheelDir.iter_all(), DownloadDir.iter_all()
            )  # type: Iterator[Union[BuiltWheelDir, DownloadDir]]
            for cache_dir in cached_artifact_dirs:
                additional_cache_dirs_by_project_name_and_version[
                    (cache_dir.project_name, cache_dir.version)
                ].append(cache_dir)

            def dep_prune_paths(
                noun,  # type: str
                dep_records,  # type: Iterable[AtomicCacheDir]
            ):
                # type: (...) -> PrunePaths
                records = tuple(dep_records)
                backing_paths = _dependency_backing_paths(
                    records, additional_cache_dirs_by_project_name_and_version
                )
                return PrunePaths(noun=noun, path=records[0].path, paths=backing_paths)

            parent_entries = tuple(
                _parent_backing_paths(pex_parent) for pex_parent in plan.parents
            )
            dependency_entries = tuple(
                dep_prune_paths("cached PEX dependency", dependency_use.records)
                for dependency_use in plan.dependency_uses
            )

            orphan_records_by_path = OrderedDict()  # type: OrderedDict[str, List[AtomicCacheDir]]
            for dep in plan.orphan_dependencies:
                orphan_records_by_path.setdefault(dep.path, []).append(dep)
            orphan_entries = tuple(
                dep_prune_paths("cached PEX dependency", records)
                for records in orphan_records_by_path.values()
            )

            pip_entries, retained_pips = self._pip_actions(plan=plan, all_pips=all_pips)
            candidate_venv_paths = {
                pex_parent.pex_dir.path
                for pex_parent in plan.parents
                if isinstance(pex_parent.pex_dir, VenvDirs)
            }
            interpreter_entries = tuple(
                self._interpreter_prune_paths(candidate_venv_paths=candidate_venv_paths)
            )

            # --- Reviewable plan -----------------------------------------------------------
            print(
                "{mode} Pex cache prune plan for {pex_root} (cutoff: {cutoff}):".format(
                    mode="Dry-run" if dry_run else "Executing",
                    pex_root=ENV.PEX_ROOT,
                    cutoff=self._cutoff_display(cutoff),
                ),
                file=fp,
            )
            print(file=fp)

            if not plan.parents:
                print(
                    "There are no cached PEX zipapps or venvs last accessed prior to "
                    "{cutoff}.".format(cutoff=self._cutoff_display(cutoff)),
                    file=fp,
                )
                print(file=fp)
            else:
                print(
                    "1. Cached PEX parents to remove first ({count}):".format(
                        count=len(parent_entries)
                    ),
                    file=fp,
                )
                for pex_parent in plan.parents:
                    print("     {path}".format(path=pex_parent.pex_dir.path), file=fp)
                    print("       reason: {reason}".format(reason=pex_parent.reason), file=fp)
                print(file=fp)

            print(
                "2. Shared PEX dependencies removable once their consumers are confirmed gone "
                "({count}):".format(count=len(dependency_entries)),
                file=fp,
            )
            for dependency_use in plan.dependency_uses:
                print(
                    "     {path} (consumers: {consumers})".format(
                        path=dependency_use.dep.path,
                        consumers=", ".join(dependency_use.candidate_consumers),
                    ),
                    file=fp,
                )
            print(file=fp)

            print(
                "3. Shared PEX dependencies retained by consumers not selected for pruning "
                "({count}):".format(count=len(plan.blocked_dependencies)),
                file=fp,
            )
            for dependency_use in plan.blocked_dependencies:
                print(
                    "     {path} (still used by: {consumers})".format(
                        path=dependency_use.dep.path,
                        consumers=", ".join(dependency_use.retained_consumers),
                    ),
                    file=fp,
                )
            print(file=fp)

            print(
                "4. Unreferenced dependencies with no remaining consumers ({count}):".format(
                    count=len(orphan_entries)
                ),
                file=fp,
            )
            for entry in orphan_entries:
                print("     {path}".format(path=entry.path), file=fp)
            print(file=fp)

            print(
                "5. Pip installations to prune ({count}); {retained} Pip "
                "{versions} retained:".format(
                    count=len(pip_entries),
                    retained=len(retained_pips),
                    versions=pluralize(len(retained_pips), "version"),
                ),
                file=fp,
            )
            for entry in pip_entries:
                print("     {path}".format(path=entry.path), file=fp)
            orphan_wheel_pnavs = OrderedSet()
            for records in orphan_records_by_path.values():
                for dep in records:
                    if isinstance(dep, InstalledWheelDir):
                        pnav = ProjectNameAndVersion.from_filename(dep.wheel_name)
                        orphan_wheel_pnavs.add(
                            (pnav.canonicalized_project_name, pnav.canonicalized_version)
                        )
            if orphan_wheel_pnavs and retained_pips:
                print(
                    "     Pip caches would be pruned of {count} cached {wheels}".format(
                        count=len(orphan_wheel_pnavs),
                        wheels=pluralize(len(orphan_wheel_pnavs), "wheel"),
                    ),
                    file=fp,
                )
            print(file=fp)

            print(
                "6. Cached interpreters to prune ({count}):".format(
                    count=len(interpreter_entries)
                ),
                file=fp,
            )
            for entry in interpreter_entries:
                print("     {path}".format(path=entry.path), file=fp)
            print(file=fp)

            if dry_run:
                # A dry run presents the same dependency-safe ordering and blocking reasons but
                # performs no deletion and changes no state; every amount below is potential.
                def potential_usage(label, entries):
                    # type: (str, Tuple[PrunePaths, ...]) -> None
                    if not entries:
                        return
                    disk_usages = map_parallel(
                        inputs=tuple(
                            path for entry in entries for path in entry.paths
                        ),
                        function=DiskUsage.collect,
                        noun="path",
                        verb="measure",
                        verb_past="measured",
                    )
                    print(
                        "{label} {count} {entries}: {usage}".format(
                            label=label,
                            count=len(entries),
                            entries=pluralize(entries, "entry"),
                            usage=self._render_usage(disk_usages),
                        ),
                        file=fp,
                    )

                potential_usage("Would prune cached PEX parents:", parent_entries)
                potential_usage("Would prune shared PEX dependencies:", dependency_entries)
                potential_usage("Would prune unreferenced dependencies:", orphan_entries)
                potential_usage("Would prune Pip installations:", pip_entries)
                potential_usage("Would prune cached interpreters:", interpreter_entries)
                print(file=fp)
                print(
                    "Dry run: no cache entries were pruned. Shared dependencies marked retained "
                    "in section 3 will not be removed; dependencies in section 2 are only removed "
                    "if all of their consumers in section 1 are removed successfully.",
                    file=fp,
                )
                return Ok()

            all_results = []  # type: List[PrunedEntry]

            # Phase 1: remove parents first; shared dependencies are decided afterwards based on
            # what actually survives.
            all_results.extend(
                self._execute_entries(
                    fp,
                    "Results - cached PEX parents:",
                    parent_entries,
                    noun="cached PEX",
                )
            )

            # Re-converge against the current disk rather than trusting the scan-time graph: a
            # parent that survived a failed deletion keeps every dependency it references.
            retained_dep_consumers = Prunable.iter_retained_dep_consumers(plan.parents)

            # If a surviving candidate parent can no longer enumerate its dependencies (e.g. a
            # partially deleted PEX-INFO), fall back to the scan-time graph and conservatively
            # retain everything it was known to consume.
            surviving_candidate_paths = {
                pex_parent.pex_dir.path
                for pex_parent in plan.parents
                if os.path.lexists(pex_parent.pex_dir.path)
            }
            for dependency_use in plan.dependency_uses:
                surviving = [
                    consumer
                    for consumer in dependency_use.candidate_consumers
                    if consumer in surviving_candidate_paths
                    and consumer not in retained_dep_consumers.get(dependency_use.dep.path, ())
                ]
                if surviving:
                    retained_dep_consumers.setdefault(dependency_use.dep.path, []).extend(
                        surviving
                    )

            # Phase 2: shared dependencies; skip any with a surviving consumer.
            shared_results = []  # type: List[PrunedEntry]
            removable_dependency_entries = []  # type: List[PrunePaths]
            for dependency_use, entry in zip(plan.dependency_uses, dependency_entries):
                surviving = retained_dep_consumers.get(dependency_use.dep.path)
                if surviving:
                    shared_results.append(
                        PrunedEntry.skipped(
                            noun="cached PEX dependency",
                            path=dependency_use.dep.path,
                            reason="retained by surviving consumer(s): {consumers}".format(
                                consumers=", ".join(surviving)
                            ),
                        )
                    )
                else:
                    removable_dependency_entries.append(entry)
            all_results.extend(
                self._execute_entries(
                    fp,
                    "Results - shared PEX dependencies:",
                    tuple(removable_dependency_entries),
                    noun="cached PEX dependency",
                    extra_results=shared_results,
                )
            )

            # Phase 3: unreferenced (orphan) dependencies.
            orphan_results = self._execute_entries(
                fp,
                "Results - unreferenced PEX dependencies:",
                orphan_entries,
                noun="cached PEX dependency",
            )
            all_results.extend(orphan_results)

            # Pip-cache wheel files are only removable for orphan wheels whose chroots are
            # confirmed gone (a failed chroot deletion retains its Pip-cache copies) and which no
            # surviving PEX references.
            removable_pip_wheels = set()  # type: Set[Tuple[ProjectName, Version]]
            retained_pip_wheel_entries = []  # type: List[PrunedEntry]
            for path, records in orphan_records_by_path.items():
                wheel_names = [
                    dep.wheel_name for dep in records if isinstance(dep, InstalledWheelDir)
                ]
                if not wheel_names or retained_dep_consumers.get(path):
                    continue
                if os.path.lexists(path):
                    # The wheel chroot survived; its Pip-cache copies must survive too.
                    retained_pip_wheel_entries.append(
                        PrunedEntry.skipped(
                            noun="Pip cached wheel",
                            path=path,
                            reason="retained because its installed wheel could not be removed",
                        )
                    )
                    continue
                for wheel_name in wheel_names:
                    pnav = ProjectNameAndVersion.from_filename(wheel_name)
                    removable_pip_wheels.add(
                        (pnav.canonicalized_project_name, pnav.canonicalized_version)
                    )

            # Phase 4: prune wheel files from retained Pip caches (independent failures).
            pip_cache_results = self._pip_cache_results(
                fp, retained_pips, removable_pip_wheels
            )
            pip_cache_entries = retained_pip_wheel_entries + pip_cache_results
            if pip_cache_entries:
                print("Results - Pip caches:", file=fp)
                for result in sorted(
                    pip_cache_entries, key=lambda r: (r.status, r.noun, r.path)
                ):
                    self._print_entry(fp, result)
                print(file=fp)
            all_results.extend(pip_cache_entries)

            # Phase 5: prune Pip PEXes / whole Pip versions re-resolved from the current disk.
            current_pip_entries, _retained_pips = self._pip_actions(
                surviving_candidate_parents=plan.parents, all_pips=all_pips
            )
            all_results.extend(
                self._execute_entries(
                    fp,
                    "Results - Pip installations:",
                    tuple(current_pip_entries),
                    noun="Pip",
                )
            )

            # Phase 6: interpreters whose binary is invalid or whose venv is confirmed gone.
            current_interpreter_entries = tuple(self._interpreter_prune_paths())
            all_results.extend(
                self._execute_entries(
                    fp,
                    "Results - cached interpreters:",
                    current_interpreter_entries,
                    noun="cached interpreter",
                )
            )

            counts = Counter(result.status for result in all_results)  # type: typing.Counter[str]
            succeeded = counts.get(SUCCEEDED, 0)
            failed = counts.get(FAILED, 0)
            skipped = counts.get(SKIPPED, 0)
            reclaimed = DiskUsage.aggregate(
                path=ENV.PEX_ROOT,
                usages=[result.disk_usage for result in all_results if result.status == SUCCEEDED],
            )
            print(
                "Prune results: {succeeded} succeeded, {failed} failed, {skipped} skipped.".format(
                    succeeded=succeeded, failed=failed, skipped=skipped
                ),
                file=fp,
            )
            print(
                "Disk space reclaimed (succeeded entries only): {usage}".format(
                    usage=self._render_usage(reclaimed)
                ),
                file=fp,
            )
            print(file=fp)

            if failed:
                return Error(
                    "{failed} {entries} failed to prune and {skipped} {dependencies} were skipped "
                    "because of surviving dependents; re-run after resolving the reported "
                    "failures.".format(
                        failed=failed,
                        entries=pluralize(failed, "entry"),
                        skipped=skipped,
                        dependencies=pluralize(skipped, "dependency"),
                    )
                )
            return Ok()
