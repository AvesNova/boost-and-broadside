"""``bnb publish`` — render the manifest's selected views, offline and atomically.

Publication is the only writer under ``docs/``, and it is deliberately the
dullest step in the system: it plays no games, fits nothing, and reaches no
network. It verifies that each source artifact is intact, renders into a
temporary directory, and installs the result. A source produced from an unclean
checkout is noted in the provenance table and warned about, not refused.

``--check`` renders exactly the same way and installs nothing, so a stale or
hand-edited canonical output is a failure rather than a surprise.
"""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from boost_and_broadside.artifacts import (
    Artifact,
    ArtifactError,
    artifact_digest,
    load_artifact,
    require_complete,
)
from boost_and_broadside.artifacts.store import file_sha256
from boost_and_broadside.publication.manifest import (
    PublicationEntry,
    PublicationManifest,
    load_manifest,
    publication_output_path,
)
from boost_and_broadside.publication.provenance import (
    OWNERSHIP_RELATIVE_PATH,
    PROVENANCE_RELATIVE_PATH,
    load_ownership,
    render_ownership,
    render_provenance_index,
)
from boost_and_broadside.publication.renderer_api import (
    PublicationError,
    Renderer,
    RenderInputs,
)

RENDERED = "rendered"
EXTERNAL = "external"
UNCHANGED = "unchanged"
CHANGED = "changed"
MISSING = "missing"
UNSELECTED = "unselected"
# The entry's sources could not be verified, so nothing was rendered for it. A
# failure like every other, reported against the entry that owns the bad source
# rather than as the run's only output.
UNRESOLVED = "unresolved"
# Not an entry status: staleness is a property of a leftover output rather than
# of any entry, so it is reported in PublishReport.stale, not through by_status.
STALE = "stale"
# Also not an entry status, for the same reason: the generated index is one
# file pair with no owning manifest entry, so drift in it is reported in
# PublishReport.index_drift rather than through by_status.
INDEX_DRIFT = "index-drift"

_PIXEL_SUFFIXES = frozenset({".png"})


class PublicationProvenanceWarning(UserWarning):
    """A source's provenance is weaker than ideal, but not a reason to refuse it."""


@dataclass(frozen=True)
class EntryOutcome:
    """What happened to one publication."""

    name: str
    status: str
    output: str
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status in {CHANGED, MISSING, UNRESOLVED}


@dataclass
class PublishReport:
    """Everything one publish run did, for the CLI to print and tests to read."""

    checked: bool
    outcomes: list[EntryOutcome] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    index: list[str] = field(default_factory=list)
    index_drift: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        """True when ``docs/`` does not match the manifest.

        A canonical output the manifest no longer owns is as much a mismatch as
        a changed one: it is still in the tree, still reads as canonical, and
        nothing produces it any more. Only check mode records one as ``stale``;
        publish mode removes it and records it under ``removed``, which is a
        repair rather than a failure. The generated provenance index and
        ownership record are canonical outputs too, even though no manifest
        entry owns them directly; check mode records their drift the same way.
        """

        return (
            any(outcome.failed for outcome in self.outcomes)
            or bool(self.stale)
            or bool(self.index_drift)
        )

    def by_status(self, status: str) -> list[EntryOutcome]:
        return [outcome for outcome in self.outcomes if outcome.status == status]

    def render(self) -> str:
        lines = [
            f"{outcome.name}: {outcome.status}"
            + (f" — {outcome.detail}" if outcome.detail else "")
            for outcome in self.outcomes
        ]
        lines.extend(f"removed stale output {path}" for path in self.removed)
        lines.extend(
            f"stale output {path} is no longer owned by the manifest; "
            "run bnb publish to remove it"
            for path in self.stale
        )
        lines.extend(
            f"generated index {path} does not match the manifest; "
            "run bnb publish to regenerate it"
            for path in self.index_drift
        )
        counts = {}
        for outcome in self.outcomes:
            counts[outcome.status] = counts.get(outcome.status, 0) + 1
        if self.stale:
            counts[STALE] = len(self.stale)
        if self.index_drift:
            counts[INDEX_DRIFT] = len(self.index_drift)
        summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
        verb = "publish --check" if self.checked else "publish"
        lines.append(f"{verb}: {summary or 'nothing to do'}")
        return "\n".join(lines)


@contextmanager
def offline() -> Iterator[None]:
    """Refuse every network call for the duration of a render.

    Publication consumes stored evidence. A renderer that reaches for a service
    is either fetching something that belongs in an artifact or reporting
    something it cannot reproduce; both are failures, so the attempt raises here
    instead of quietly succeeding on a machine that happens to be online.
    """

    class _Blocked(socket.socket):
        def __init__(self, *args, **kwargs):
            raise PublicationError("publication is offline; a renderer attempted a network call")

    def _blocked_connection(*_args, **_kwargs):
        raise PublicationError("publication is offline; a renderer attempted a network call")

    real_socket, real_connection = socket.socket, socket.create_connection
    socket.socket, socket.create_connection = _Blocked, _blocked_connection
    try:
        yield
    finally:
        socket.socket, socket.create_connection = real_socket, real_connection


def run_publish(
    root: str | Path,
    *,
    target: str | None = None,
    check: bool = False,
) -> PublishReport:
    """Render the selected inventory, or check it against what is on disk."""

    repository = Path(root)
    manifest = load_manifest(repository)
    entries = _entries_to_process(manifest, target)

    report = PublishReport(checked=check)
    published: dict[str, str] = {}
    for entry in entries:
        if not entry.selected:
            report.outcomes.append(
                EntryOutcome(
                    entry.name,
                    UNSELECTED,
                    entry.output,
                    "no source chosen for " + ", ".join(entry.missing_sources),
                )
            )
            continue
        if entry.renderer.external:
            report.outcomes.append(_verify_external(repository, entry))
            report.index.append(_index_row(entry, {}))
            continue
        promoted = _verify_promoted(repository, entry)
        if promoted is not None:
            report.outcomes.append(promoted)
            report.index.append(_index_row(entry, _pinned_digests(entry)))
            published[entry.name] = entry.output
            continue
        try:
            sources, digests = _resolve_sources(repository, entry)
        except PublicationError as error:
            # One damaged, dirty, or unreadable source says nothing about the
            # other entries. Reporting it against its own entry lets the rest of
            # the inventory still be checked, and the run still fails: an
            # unresolved entry is a failed one, so the exit code is unchanged.
            report.outcomes.append(
                EntryOutcome(entry.name, UNRESOLVED, entry.output, _detail(entry, error))
            )
            continue
        with tempfile.TemporaryDirectory(prefix="bnb-publish-") as temporary:
            staged = Path(temporary)
            with offline():
                written = entry.renderer.render(sources, staged)
            _validate_render(entry.renderer, staged, written)
            report.outcomes.append(_install(repository, entry, staged, check=check))
        published[entry.name] = entry.output
        report.index.append(_index_row(entry, digests))

    if target is None:
        unowned = _prune_unowned(repository, manifest, check=check)
        (report.stale if check else report.removed).extend(unowned)
    # An unresolved entry contributes no provenance row and no ownership record,
    # so rewriting the index from this run would drop it from both while its
    # output stayed in `docs/` — leaving it unowned, and prunable by the next
    # run. Every other outcome still contributes its row, so only this one has
    # to hold the index back, in both modes: check verifies the same body it
    # would otherwise write, so it must be withheld under the same condition.
    index_ready = (
        target is None
        and not report.by_status(UNRESOLVED)
        and any(entry.selected and not entry.renderer.external for entry in manifest.entries)
    )
    if index_ready:
        if check:
            report.index_drift.extend(_verify_generated_index(repository, manifest, report))
        else:
            _write_generated_index(repository, manifest, report)
    return report


def _verify_external(repository: Path, entry: PublicationEntry) -> EntryOutcome:
    """An asset produced elsewhere is checked for presence, never rewritten."""

    if not (repository / entry.output).exists():
        return EntryOutcome(entry.name, MISSING, entry.output, "external asset is absent")
    return EntryOutcome(
        entry.name, EXTERNAL, entry.output, "tracked as-is; not produced in this repository"
    )


def _pinned_digests(entry: PublicationEntry) -> dict[str, str]:
    return {name: source.sha256 for name, source in entry.files.items()}


def _verify_promoted(repository: Path, entry: PublicationEntry) -> EntryOutcome | None:
    """Check a promoted output against its pinned hash when the source is gone.

    A curated clip is scratch: it is captured, reviewed, and promoted, and the
    working copy under ``out/`` is deliberately never tracked. What makes the
    published file citable is therefore not the scratch file — it is the sha256
    the manifest pins, which is committed. A checkout without ``out/`` can still
    say whether ``docs/`` holds exactly the reviewed bytes, so that is what
    happens here rather than aborting the whole inventory over a file whose
    absence is the normal state.

    Returns ``None`` when the ordinary path applies: any promoted source that is
    present is copied and compared as before.
    """

    if len(entry.files) != 1 or entry.artifacts or entry.renderer.multi_file:
        return None
    (source,) = entry.files.values()
    if (repository / source.path).is_file():
        return None

    destination = repository / entry.output
    if not destination.is_file():
        return EntryOutcome(
            entry.name,
            MISSING,
            entry.output,
            f"{source.path} is not present and the canonical output is absent",
        )
    if file_sha256(destination) == source.sha256:
        return EntryOutcome(
            entry.name,
            UNCHANGED,
            entry.output,
            f"{source.path} is not present; the output matches its pinned sha256",
        )
    return EntryOutcome(
        entry.name,
        CHANGED,
        entry.output,
        f"{source.path} is not present and the output does not match its pinned sha256",
    )


def _detail(entry: PublicationEntry, error: PublicationError) -> str:
    """The error's own words, minus the entry name the report line already carries."""

    message = str(error)
    prefix = f"publication {entry.name!r}"
    if message.startswith(prefix):
        message = message[len(prefix) :].lstrip(": ")
    return message


def _entries_to_process(
    manifest: PublicationManifest, target: str | None
) -> Sequence[PublicationEntry]:
    if target is None:
        return manifest.entries
    entry = manifest.entry(target)
    if not entry.selected:
        raise PublicationError(
            f"publication {target!r} has no source for "
            f"{', '.join(entry.missing_sources)}; it cannot be rendered yet"
        )
    return (entry,)


def _resolve_sources(
    repository: Path, entry: PublicationEntry
) -> tuple[RenderInputs, dict[str, str]]:
    renderer = entry.renderer
    artifacts: dict[str, Artifact] = {}
    digests: dict[str, str] = {}
    for name, location in entry.artifacts.items():
        path = repository / location
        try:
            artifact = load_artifact(path)
            require_complete(artifact)
        except ArtifactError as error:
            raise PublicationError(f"publication {entry.name!r}: {error}") from error
        _warn_unclean_source(entry, name, artifact)
        _require_supported_schema(entry, renderer, name, artifact)
        artifacts[name] = artifact
        digests[name] = artifact_digest(artifact)

    files: dict[str, Path] = {}
    for name, source in entry.files.items():
        path = repository / source.path
        if not path.is_file():
            raise PublicationError(
                f"publication {entry.name!r} promotes {source.path}, which is not present"
            )
        actual = file_sha256(path)
        if actual != source.sha256:
            raise PublicationError(
                f"publication {entry.name!r}: {source.path} does not match its recorded sha256"
            )
        files[name] = path
        digests[name] = actual
    return RenderInputs(artifacts=artifacts, files=files, figure=entry.figure), digests


def _warn_unclean_source(entry: PublicationEntry, name: str, artifact: Artifact) -> None:
    """Note, but do not refuse, a source produced from an unclean checkout.

    This used to raise. It was a poor trade: ``code_provenance`` shells
    ``git status --porcelain``, which counts *untracked* files, so generating one
    artifact leaves the tree dirty and blocks generating the next -- a serialised
    commit-then-generate dance for every measurement. It also only ever policed
    the input artifacts, while the render that produced the committed figures
    went unchecked, so the guarantee it appeared to give was not one it gave.

    The commit and the dirty bit are still recorded in the artifact and printed
    in the provenance table. That is what a reader needs to judge the output;
    refusing to produce it was not.
    """

    code = artifact.manifest.get("code", {})
    commit = code.get("git_commit")
    if not commit:
        warnings.warn(
            f"publication {entry.name!r} source {name!r} records no commit",
            PublicationProvenanceWarning,
            stacklevel=2,
        )
    elif code.get("git_dirty"):
        warnings.warn(
            f"publication {entry.name!r} source {name!r} was produced from a dirty "
            f"checkout at {commit[:12]}",
            PublicationProvenanceWarning,
            stacklevel=2,
        )


def _require_supported_schema(
    entry: PublicationEntry, renderer: Renderer, name: str, artifact: Artifact
) -> None:
    supported = renderer.supported_schemas.get(name)
    if supported is None:
        return
    version = artifact.manifest.get("result_schema_version")
    if version not in supported:
        raise PublicationError(
            f"publication {entry.name!r} source {name!r} carries result schema {version!r}; "
            f"renderer {renderer.name!r} reads {', '.join(str(item) for item in supported)}"
        )


def _validate_render(renderer: Renderer, staged: Path, written: Sequence[Path]) -> None:
    produced = sorted(path for path in staged.rglob("*") if path.is_file())
    if not produced:
        raise PublicationError(f"renderer {renderer.name!r} produced no output")
    if not renderer.multi_file and len(produced) != 1:
        names = ", ".join(path.name for path in produced)
        raise PublicationError(
            f"renderer {renderer.name!r} owns one file but produced {len(produced)}: {names}"
        )
    escaped = [str(path) for path in map(Path, written) if not path.is_relative_to(staged)]
    if escaped:
        raise PublicationError(
            f"renderer {renderer.name!r} wrote outside its staging directory: {escaped}"
        )


def _install(
    repository: Path, entry: PublicationEntry, staged: Path, *, check: bool
) -> EntryOutcome:
    destination = repository / entry.output
    produced = sorted(path for path in staged.rglob("*") if path.is_file())
    if not entry.renderer.multi_file:
        return _install_file(destination, produced[0], entry, check=check)

    outcomes = [
        _install_file(
            destination / path.relative_to(staged), path, entry, check=check, single=False
        )
        for path in produced
    ]
    stale = _stale_files(destination, {path.relative_to(staged) for path in produced})
    if stale and not check:
        for path in stale:
            path.unlink()
    statuses = {outcome.status for outcome in outcomes}
    if MISSING in statuses or CHANGED in statuses or (stale and check):
        detail = "; ".join(
            f"{outcome.output}: {outcome.status}" for outcome in outcomes if outcome.failed
        )
        if stale:
            listed = ", ".join(sorted(path.name for path in stale))
            detail = f"{detail}; stale: {listed}" if detail else f"stale: {listed}"
        return EntryOutcome(
            entry.name, CHANGED if MISSING not in statuses else MISSING, entry.output, detail
        )
    changed = sum(1 for outcome in outcomes if outcome.status == RENDERED)
    status = UNCHANGED if changed == 0 else RENDERED
    detail = f"{len(outcomes)} file(s)" + (f", {len(stale)} pruned" if stale else "")
    return EntryOutcome(entry.name, status, entry.output, detail)


def _install_file(
    destination: Path, produced: Path, entry: PublicationEntry, *, check: bool, single: bool = True
) -> EntryOutcome:
    relative = destination.name if not single else entry.output
    if destination.is_file():
        comparison = _compare(destination, produced)
        if comparison is not None:
            return EntryOutcome(entry.name, UNCHANGED, relative, comparison)
        if check:
            return EntryOutcome(entry.name, CHANGED, relative, "rendered output differs")
    elif check:
        return EntryOutcome(entry.name, MISSING, relative, "canonical output is absent")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(produced, temporary)
    os.replace(temporary, destination)
    return EntryOutcome(entry.name, RENDERED, relative)


def _compare(existing: Path, produced: Path) -> str | None:
    """Return how the two match, or None when they differ."""

    if file_sha256(existing) == file_sha256(produced):
        return ""
    if existing.suffix.lower() not in _PIXEL_SUFFIXES:
        return None
    import matplotlib.image as image
    import numpy as np

    try:
        same = np.array_equal(image.imread(existing), image.imread(produced))
    except (OSError, ValueError):
        return None
    return "identical pixels, different bytes" if same else None


def _stale_files(destination: Path, produced: set[Path]) -> list[Path]:
    if not destination.is_dir():
        return []
    return sorted(
        path
        for path in destination.rglob("*")
        if path.is_file() and path.relative_to(destination) not in produced
    )


def _prune_unowned(
    repository: Path, manifest: PublicationManifest, *, check: bool
) -> list[str]:
    """Find outputs a previous run owned and this manifest no longer declares.

    Publish mode removes them. Check mode changes nothing, so it only reports
    them and the caller turns that into a failure — an output nothing produces
    any more is still sitting in ``docs/`` reading as canonical.

    The ownership record is generated, but it is also tracked and hand-editable,
    so every path it names is re-validated as a location inside ``docs/`` before
    anything is deleted.
    """

    previous = load_ownership(repository)
    owned = {entry.output for entry in manifest.entries}
    found = []
    for output in sorted(set(previous) - owned):
        relative = publication_output_path(
            output, repository, described_as=f"{OWNERSHIP_RELATIVE_PATH.as_posix()} entry"
        )
        path = repository / relative
        if not path.exists():
            continue
        found.append(relative)
        if check:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return found


def _index_row(entry: PublicationEntry, digests: dict[str, str]) -> str:
    # A copied figure names the entry it copied. Without it every chart from one
    # run would read as the same source row, and the table would stop
    # distinguishing the outputs it exists to distinguish. What produced the
    # figure is recorded one hop away, in the figures artifact's own manifest.
    within = f"#{entry.figure}" if entry.figure else ""
    sources = "; ".join(
        f"{name}={location}{within} ({digests.get(name, '')[:12]})"
        for name, location in sorted(entry.artifacts.items())
    )
    promoted = "; ".join(
        f"{name}={source.path} ({source.sha256[:12]})"
        for name, source in sorted(entry.files.items())
    )
    joined = ", ".join(part for part in (sources, promoted) if part)
    return (
        f"| `{entry.output}` | {entry.renderer_name} "
        f"| {joined or 'none'} | {entry.description} |"
    )


def _write_generated_index(
    repository: Path, manifest: PublicationManifest, report: PublishReport
) -> None:
    """Write the generated provenance index and the ownership record beside it.

    Both appear only once something is actually rendered from a source: until
    the exact landmark artifacts are chosen there is nothing to trace, and an
    index listing only assets produced elsewhere would read as a claim that
    there is.
    """

    for relative, body in (
        (PROVENANCE_RELATIVE_PATH, render_provenance_index(report.index)),
        (OWNERSHIP_RELATIVE_PATH, render_ownership(manifest)),
    ):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(body)
        os.replace(temporary, destination)


def _verify_generated_index(
    repository: Path, manifest: PublicationManifest, report: PublishReport
) -> list[str]:
    """Check mode's half of ``_write_generated_index``: compare, never write.

    Nothing else in the run exercises these two files, so a hand edit, a
    renderer change that alters a digest without changing an entry's install
    status, or a manually reverted checkout would otherwise pass ``--check``
    while ``docs/results/provenance.{md,json}`` quietly disagreed with what
    ``bnb publish`` would actually produce.
    """

    drifted = []
    for relative, body in (
        (PROVENANCE_RELATIVE_PATH, render_provenance_index(report.index)),
        (OWNERSHIP_RELATIVE_PATH, render_ownership(manifest)),
    ):
        destination = repository / relative
        if not destination.is_file() or destination.read_text() != body:
            drifted.append(relative.as_posix())
    return drifted


__all__ = [
    "CHANGED",
    "EXTERNAL",
    "EntryOutcome",
    "INDEX_DRIFT",
    "MISSING",
    "PublishReport",
    "RENDERED",
    "STALE",
    "UNCHANGED",
    "UNRESOLVED",
    "UNSELECTED",
    "offline",
    "run_publish",
]
