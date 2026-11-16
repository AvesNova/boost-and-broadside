"""Render the architecture diagrams: every `*.typ` here but `diagram.typ` (the shared helpers)
to an SVG beside it, with Typst (https://typst.app, one static binary; see
`docs/guides/figures.md#architecture-diagrams`).

Each SVG starts with a comment holding the hash of its sources (the diagram and
`diagram.typ`), so `tests/docs/test_architecture_diagrams.py` can tell a stale SVG
without Typst.

    uv run --no-sync python docs/images/architecture/render.py              # the SVGs
    uv run --no-sync python docs/images/architecture/render.py --png DIR    # and PNG previews
    uv run --no-sync python docs/images/architecture/render.py --pdf DIR    # and PDFs, for a paper
"""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPERS = HERE / "diagram.typ"
STAMP = "<!-- rendered from sources with sha256 {} by render.py -->\n"


def diagrams() -> list[Path]:
    """The diagram sources, in name order."""
    return sorted(path for path in HERE.glob("*.typ") if path != HELPERS)


def source_hash(diagram: Path) -> str:
    """The sha256 of the diagram's source and the shared helpers'."""
    digest = hashlib.sha256()
    for path in (diagram, HELPERS):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def stamp(diagram: Path) -> str:
    """The comment an up-to-date SVG of this diagram starts with."""
    return STAMP.format(source_hash(diagram))


def render(diagram: Path, png: Path | None, pdf: Path | None) -> None:
    """Compile one diagram to its stamped SVG, and to a PNG or a PDF where asked."""
    typst = shutil.which("typst")
    if typst is None:
        raise SystemExit("typst is not on PATH: see docs/guides/figures.md#architecture-diagrams")
    svg = diagram.with_suffix(".svg")
    subprocess.run([typst, "compile", "--format", "svg", diagram, svg], check=True)
    svg.write_text(stamp(diagram) + svg.read_text())
    if png is not None:
        target = png / f"{diagram.stem}.png"
        subprocess.run([typst, "compile", "--format", "png", "--ppi", "200", diagram, target], check=True)
    if pdf is not None:
        subprocess.run([typst, "compile", "--format", "pdf", diagram, pdf / f"{diagram.stem}.pdf"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--png", type=Path, help="also write PNG previews into this directory")
    parser.add_argument("--pdf", type=Path, help="also write PDFs into this directory")
    arguments = parser.parse_args()
    for directory in (arguments.png, arguments.pdf):
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
    for diagram in diagrams():
        render(diagram, arguments.png, arguments.pdf)
        print(diagram.with_suffix(".svg").relative_to(HERE.parents[2]))


if __name__ == "__main__":
    main()
