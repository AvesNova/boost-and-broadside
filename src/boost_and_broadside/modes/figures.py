"""``bnb figures`` — render one run's whole figure set into that run's artifacts.

A run's charts belong beside the measurements they came from, for the same
reason its Elo calibration does: they are evidence about that run, and a run
directory that carries them is self-describing. Publication is then a choice of
which run to illustrate the documents with, made in ``docs/publications.toml``,
rather than a re-render.

That split is what makes a new run cheap. Producing every chart for one is a
single command against the artifacts already on disk, and it never touches
``docs/`` or disturbs the run the documents currently cite.
"""

from __future__ import annotations

from pathlib import Path

from boost_and_broadside.artifacts import (
    ArtifactRecipe,
    ArtifactStore,
    artifact_digest,
    load_artifact,
    require_complete,
)
from boost_and_broadside.evaluation.run_catalog import resolve_exact_run
from boost_and_broadside.evaluation.subjects import describe_source_artifact
from boost_and_broadside.publication import renderers  # noqa: F401  (registers them)
from boost_and_broadside.publication.figure_set import FIGURES, FigureSpec
from boost_and_broadside.publication.renderer_api import (
    PublicationError,
    RenderInputs,
    get_renderer,
)

_SCHEMA_VERSION = 1
_ARTIFACT_DIR = "artifacts"


def _latest_artifact(run_dir: Path, artifact_type: str):
    """The run's current artifact of one type, verified and complete.

    A run accumulates one directory per measurement of a kind; the newest is the
    current one, because identities sort by the instant they were taken.
    """

    candidates = sorted((run_dir / _ARTIFACT_DIR / artifact_type).glob("*/"))
    if not candidates:
        raise PublicationError(
            f"{run_dir.name} has no {artifact_type} artifact; run that measurement first"
        )
    artifact = load_artifact(candidates[-1])
    require_complete(artifact)
    return artifact


def render_run_figures(
    run_spec: str,
    *,
    checkpoint_dir: str = "checkpoints",
    only: tuple[str, ...] = (),
    store: ArtifactStore | None = None,
) -> Path:
    """Render the figure set for one finished run and return the artifact path."""

    run_dir = resolve_exact_run(run_spec, checkpoint_dir).path
    selected: tuple[FigureSpec, ...] = (
        FIGURES if not only else tuple(f for f in FIGURES if f.name in set(only))
    )
    if not selected:
        raise PublicationError(f"no figure matches {only!r}")

    # Resolve every source first: a partial figures artifact is worse than none,
    # because it looks like a complete set with charts missing.
    resolved = {}
    for figure in selected:
        for artifact_type in figure.sources.values():
            if artifact_type not in resolved:
                resolved[artifact_type] = _latest_artifact(run_dir, artifact_type)

    store = store or ArtifactStore(checkpoint_root=checkpoint_dir)
    artifact = store.create(
        ArtifactRecipe(
            artifact_type="figures",
            result_schema_version=_SCHEMA_VERSION,
            subjects={"run": run_dir.name},
            parameters={"figures": [figure.name for figure in selected]},
            # Which exact measurements these charts depict. A figure is only as
            # citable as the artifact under it, so the digests travel together.
            sources={
                artifact_type: describe_source_artifact(source.manifest, artifact_digest(source))
                for artifact_type, source in sorted(resolved.items())
            },
        ),
        store.run_owner(run_dir.name),
    )

    print(f"\n=== figures: {run_dir.name} ===")
    for figure in selected:
        renderer = get_renderer(figure.renderer)
        # Renderers are handed a directory and choose their own filenames; a
        # multi-file one prunes whatever it no longer produces, so each gets its
        # own subdirectory rather than sharing the artifact root.
        target = artifact.path / figure.name if renderer.multi_file else artifact.path
        target.mkdir(parents=True, exist_ok=True)
        inputs = RenderInputs(
            artifacts={key: resolved[kind] for key, kind in figure.sources.items()}
        )
        for written in renderer.render(inputs, target):
            artifact.attach(str(written.relative_to(artifact.path)))
        print(f"  {figure.name}")

    artifact.complete()
    print(f"\n  wrote {artifact.path}")
    return artifact.path
