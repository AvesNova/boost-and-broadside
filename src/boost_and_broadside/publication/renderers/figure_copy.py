"""Publishing a chart that has already been rendered.

Every chart in the figure set is produced once, by ``bnb figures``, into the
artifacts of the run it describes. Publication is then a choice of which run
illustrates the documents -- so it copies, rather than running the same renderer
over the same artifact a second time.

Rendering twice was not merely wasted work. Two independent renders of one
figure are only equal by convention, and nothing compared them: a renderer whose
output depended on anything outside its inputs could put one image in the run's
own evidence and a different one in ``docs/`` under the same claim.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from boost_and_broadside.publication.renderer_api import (
    PublicationError,
    Renderer,
    RenderInputs,
    register,
)


def _source(inputs: RenderInputs) -> Path:
    figures = inputs.artifact("figures")
    assert inputs.figure is not None  # the manifest requires it for these renderers
    source = figures.path / inputs.figure
    if not source.exists():
        available = ", ".join(sorted(item.name for item in figures.path.iterdir()))
        raise PublicationError(
            f"{figures.path} holds no figure named {inputs.figure!r}; it has: {available}"
        )
    return source


def _copy_figure(inputs: RenderInputs, out_dir: Path) -> list[Path]:
    source = _source(inputs)
    if not source.is_file():
        raise PublicationError(
            f"{source} is a directory; publish it with figure-tree-copy-v1"
        )
    destination = out_dir / source.name
    shutil.copyfile(source, destination)
    return [destination]


def _copy_figure_tree(inputs: RenderInputs, out_dir: Path) -> list[Path]:
    source = _source(inputs)
    if not source.is_dir():
        raise PublicationError(f"{source} is a single file; publish it with figure-copy-v1")
    written = []
    for item in sorted(source.rglob("*")):
        if item.is_dir():
            continue
        destination = out_dir / item.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, destination)
        written.append(destination)
    if not written:
        raise PublicationError(f"{source} holds no files to publish")
    return written


register(
    Renderer(
        name="figure-copy-v1",
        description="Publish one file from a run's rendered figure set.",
        render=_copy_figure,
        required_artifacts=("figures",),
        supported_schemas={"figures": (1,)},
        names_a_figure=True,
    )
)
register(
    Renderer(
        name="figure-tree-copy-v1",
        description="Publish one whole directory from a run's rendered figure set.",
        render=_copy_figure_tree,
        required_artifacts=("figures",),
        supported_schemas={"figures": (1,)},
        multi_file=True,
        names_a_figure=True,
    )
)
