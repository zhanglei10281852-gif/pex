# Copyright 2024 Pex project contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from __future__ import absolute_import

import os.path
from collections import OrderedDict, defaultdict
from datetime import datetime

from pex.cache import access
from pex.cache.dirs import (
    BootstrapDir,
    InstalledWheelDir,
    UnzipDir,
    UserCodeDir,
    VenvDirs,
)
from pex.typing import TYPE_CHECKING

if TYPE_CHECKING:
    # MyPy run for 2.7 does not recognize the DefaultDict type
    from typing import (  # type: ignore[attr-defined]
        DefaultDict,
        Dict,
        FrozenSet,
        Iterable,
        Iterator,
        List,
        Optional,
        Set,
        Tuple,
        Union,
    )

    import attr  # vendor:skip

    # A cached PEX zipapp (`UnzipDir`) or venv (`VenvDirs`); these are the "parents" that consume
    # shared dependencies.
    PexDir = Union[UnzipDir, VenvDirs]

    # A shared dependency consumed by cached PEX parents.
    Dependency = Union[BootstrapDir, UserCodeDir, InstalledWheelDir]
else:
    from pex.third_party import attr


# The intermediary unzipped instance of a --venv PEX is pruned regardless of the cutoff since it
# is only needed to perform the initial venv install.
INTERMEDIARY_UNZIP_REASON = "intermediary unzip of an installed --venv PEX"


@attr.s(frozen=True)
class PexParent(object):
    """A cached PEX zipapp or venv that is a candidate for pruning ("a parent")."""

    pex_dir = attr.ib()  # type: PexDir
    last_access = attr.ib()  # type: float
    reason = attr.ib()  # type: str


@attr.s(frozen=True)
class DependencyUse(object):
    """A shared dependency along with the cached PEX consumers known to reference it.

    `records` carries every record discovered for the dependency's real path. These only differ
    for installed wheels referenced via more than one install-hash symlink; all such symlink
    backing dirs must be removed together with the shared wheel chroot.
    """

    dep = attr.ib()  # type: Dependency
    records = attr.ib()  # type: Tuple[Dependency, ...]
    candidate_consumers = attr.ib()  # type: Tuple[str, ...]
    retained_consumers = attr.ib()  # type: Tuple[str, ...]

    @property
    def blocked(self):
        # type: () -> bool
        """`True` if at least one consumer is retained (not a prune candidate)."""
        return bool(self.retained_consumers)


def _dep_record_key(dep):
    # type: (Dependency) -> Tuple[str, Optional[str]]
    return dep.path, getattr(dep, "symlink_dir", None)


class _DepRecord(object):
    """Mutable accumulation point for a single dependency real path during a scan."""

    def __init__(self, dep):
        # type: (Dependency) -> None
        self.dep = dep
        self._records = OrderedDict()  # type: OrderedDict[Tuple[str, Optional[str]], Dependency]
        self._records[_dep_record_key(dep)] = dep
        self.candidate_consumers = []  # type: List[str]
        self.retained_consumers = []  # type: List[str]

    def add_record(self, dep):
        # type: (Dependency) -> None
        self._records.setdefault(_dep_record_key(dep), dep)

    @property
    def records(self):
        # type: () -> Tuple[Dependency, ...]
        return tuple(self._records.values())


def _format_cutoff_reason(
    last_access,  # type: float
    cutoff,  # type: datetime
):
    # type: (...) -> str
    return "last accessed {last_access} prior to the cutoff of {cutoff}".format(
        last_access=datetime.fromtimestamp(last_access).isoformat(),
        cutoff=cutoff.isoformat(),
    )


@attr.s(frozen=True)
class Prunable(object):
    """A reviewable prune plan derived from a point-in-time scan of the Pex cache.

    The plan only describes what *appears* prunable at scan time. Dependencies listed in
    `dependency_uses` are only removable once all of their candidate consumers have actually been
    deleted, which must be re-verified against the current disk state after parents are removed.
    """

    cutoff = attr.ib()  # type: datetime
    parents = attr.ib()  # type: Tuple[PexParent, ...]
    retained_pex_dirs = attr.ib()  # type: Tuple[PexDir, ...]
    dependency_uses = attr.ib()  # type: Tuple[DependencyUse, ...]
    blocked_dependencies = attr.ib()  # type: Tuple[DependencyUse, ...]
    orphan_dependencies = attr.ib()  # type: Tuple[Dependency, ...]
    venv_dir_paths = attr.ib()  # type: FrozenSet[str]

    @classmethod
    def scan(cls, cutoff):
        # type: (datetime) -> Prunable

        # First pass: classify all cached PEX parents against the cutoff.
        all_pex_dirs = []  # type: List[Tuple[PexDir, float]]
        venv_pex_hashes = set()  # type: Set[str]
        venv_dir_paths = set()  # type: Set[str]
        for pex_dir, last_access in access.iter_all_cached_pex_dirs():
            all_pex_dirs.append((pex_dir, last_access))
            if isinstance(pex_dir, VenvDirs):
                venv_pex_hashes.add(pex_dir.pex_hash)
                venv_dir_paths.add(pex_dir.path)

        parents = []  # type: List[PexParent]
        retained_pex_dirs = []  # type: List[PexDir]
        for pex_dir, last_access in all_pex_dirs:
            reasons = []  # type: List[str]
            if isinstance(pex_dir, UnzipDir) and pex_dir.pex_hash in venv_pex_hashes:
                reasons.append(INTERMEDIARY_UNZIP_REASON)
            if datetime.fromtimestamp(last_access) < cutoff:
                reasons.append(_format_cutoff_reason(last_access, cutoff))
            if reasons:
                parents.append(
                    PexParent(
                        pex_dir=pex_dir,
                        last_access=last_access,
                        reason="; ".join(reasons),
                    )
                )
            else:
                retained_pex_dirs.append(pex_dir)

        # Second pass: build the dependency -> consumer(s) reference graph. Dependencies are keyed
        # by their real on-disk path so that installed wheel symlinks to a shared chroot collapse
        # to a single dependency record.
        records_by_path = OrderedDict()  # type: OrderedDict[str, _DepRecord]

        def register(
            pex_dir,  # type: PexDir
            candidate,  # type: bool
        ):
            # type: (...) -> None
            for dep in pex_dir.iter_deps():
                record = records_by_path.get(dep.path)
                if record is None:
                    record = _DepRecord(dep)
                    records_by_path[dep.path] = record
                else:
                    record.add_record(dep)
                consumers = (
                    record.candidate_consumers if candidate else record.retained_consumers
                )
                if pex_dir.path not in consumers:
                    consumers.append(pex_dir.path)

        for pex_parent in parents:
            register(pex_parent.pex_dir, True)
        for pex_dir in retained_pex_dirs:
            register(pex_dir, False)

        dependency_uses = []  # type: List[DependencyUse]
        blocked_dependencies = []  # type: List[DependencyUse]
        referenced_dep_paths = set(records_by_path.keys())  # type: Set[str]
        for record in records_by_path.values():
            dependency_use = DependencyUse(
                dep=record.dep,
                records=record.records,
                candidate_consumers=tuple(record.candidate_consumers),
                retained_consumers=tuple(record.retained_consumers),
            )
            if dependency_use.blocked:
                blocked_dependencies.append(dependency_use)
            else:
                dependency_uses.append(dependency_use)

        # Orphans: dependencies on disk that no cached PEX parent references at all. More than one
        # orphan record may share a real path (an installed wheel chroot with multiple dangling
        # install-hash symlinks); the caller groups records by path for deletion.
        orphan_dependencies = []  # type: List[Dependency]
        for dep in _iter_all_deps():
            if dep.path not in referenced_dep_paths:
                orphan_dependencies.append(dep)

        return cls(
            cutoff=cutoff,
            parents=tuple(parents),
            retained_pex_dirs=tuple(retained_pex_dirs),
            dependency_uses=tuple(dependency_uses),
            blocked_dependencies=tuple(blocked_dependencies),
            orphan_dependencies=tuple(orphan_dependencies),
            venv_dir_paths=frozenset(venv_dir_paths),
        )

    @staticmethod
    def iter_current_pex_dirs():
        # type: () -> Iterator[PexDir]
        """Iterates the cached PEX parents present on disk right now (no cutoff)."""
        for pex_dir in UnzipDir.iter_all():
            yield pex_dir
        for pex_dir in VenvDirs.iter_all():
            yield pex_dir

    @classmethod
    def iter_retained_dep_consumers(cls, candidate_parents=()):
        # type: (Iterable[PexParent]) -> Dict[str, List[str]]
        """Re-converges against the current disk after parent deletion was attempted.

        Returns a mapping of dependency path to the paths of cached PEX consumers still present on
        disk. Candidate parents that survived a failed deletion are accounted for directly (even if
        they are no longer discoverable via a removed venv short-hash symlink), along with any PEX
        dir currently found on disk. A dependency is only safe to remove when it has no entry in
        this mapping.
        """
        consumers = defaultdict(list)  # type: DefaultDict[str, List[str]]

        accounted_parent_paths = set()  # type: Set[str]

        def add_consumer(pex_dir):
            # type: (PexDir) -> None
            try:
                deps = list(pex_dir.iter_deps())
            except (OSError, ValueError, KeyError):
                # The parent survived deletion but its metadata is corrupt or partially
                # removed; the caller conservatively retains all of its scan-time dependencies.
                return
            for dep in deps:
                if pex_dir.path not in consumers[dep.path]:
                    consumers[dep.path].append(pex_dir.path)

        for parent in candidate_parents:
            accounted_parent_paths.add(parent.pex_dir.path)
            if os.path.lexists(parent.pex_dir.path):
                # The parent survived deletion; everything it references remains in use.
                add_consumer(parent.pex_dir)

        for pex_dir in cls.iter_current_pex_dirs():
            if pex_dir.path in accounted_parent_paths:
                continue
            add_consumer(pex_dir)

        return dict(consumers)


def _iter_all_deps():
    # type: () -> Iterator[Dependency]
    for bootstrap_dir in BootstrapDir.iter_all():
        yield bootstrap_dir
    for user_code_dir in UserCodeDir.iter_all():
        yield user_code_dir
    for installed_wheel_dir in InstalledWheelDir.iter_all():
        yield installed_wheel_dir
