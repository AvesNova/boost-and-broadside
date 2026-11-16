"""The architecture diagrams' SVGs (`docs/images/architecture/`) are rendered from their
current sources: each starts with the hash `render.py` stamps it with."""

import importlib.util

import pytest

from . import pages

FOLDER = pages.DOCS / "images" / "architecture"


def _render_module():
    spec = importlib.util.spec_from_file_location("architecture_render", FOLDER / "render.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render = _render_module()


def test_there_are_diagrams():
    assert render.diagrams(), f"no diagram sources in {FOLDER}"


@pytest.mark.parametrize("diagram", render.diagrams(), ids=lambda path: path.stem)
def test_each_svg_is_rendered_from_its_current_source(diagram):
    svg = diagram.with_suffix(".svg")
    assert svg.exists(), f"{svg.name} is missing: run docs/images/architecture/render.py"
    with svg.open() as file:
        first = file.readline()
    assert first == render.stamp(diagram), (
        f"{svg.name} is stale (its source or diagram.typ changed): "
        "run docs/images/architecture/render.py"
    )
