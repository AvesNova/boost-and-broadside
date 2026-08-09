"""Publishing something that was not computed by a renderer.

Two cases, both deliberately narrow. A curated replay is captured as scratch and
*promoted*: the manifest pins the exact local file by hash and publication copies
it, so the tracked clip is reproducible in the only sense a rendered video can be
— it is provably the file that was reviewed. An external asset has no producer in
this repository at all; declaring it keeps it inside the inventory instead of
leaving an untracked figure nobody owns.
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


def _render_clip(inputs: RenderInputs, out_dir: Path) -> list[Path]:
    source = inputs.files["clip"]
    destination = out_dir / source.name
    shutil.copyfile(source, destination)
    return [destination]


def _render_external(inputs: RenderInputs, out_dir: Path) -> list[Path]:  # pragma: no cover
    raise PublicationError("an external asset is verified, never rendered")


register(
    Renderer(
        name="media-copy-v1",
        description="Promote a curated scratch clip, pinned by content hash.",
        render=_render_clip,
        required_files=("clip",),
    )
)
register(
    Renderer(
        name="external-asset-v1",
        description="A tracked asset with no producer in this repository.",
        render=_render_external,
        external=True,
    )
)
