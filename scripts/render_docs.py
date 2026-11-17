"""Render the README and docs from their templates, filling in run results by name.

Every run-specific number and every chart in the README and docs lives in one place:
numbers in `docs/values.json` (built by `scripts/docs_values.py` from the reference run's
artifacts), charts in `docs/figures.yml`. The pages are written as templates next to
their output, `README.template.md` -> `README.md` and `docs/<page>.template.md` ->
`docs/<page>.md`, and refer to results by name:

    {{name}}            a value from docs/values.json, formatted plainly
    {{name:spec}}       the same through a Python format spec, e.g. {{win_rate:.1%}}
    {{figure: id}}      on a line of its own: the figure's image, then its caption in
                        italics, numbered per page ("Figure 2."), from docs/figures.yml

Fenced code blocks are copied as they are. A value that is still null (a result not yet
measured) renders as a visible marker, `[pending: name]`; a figure whose file is not in
the repository yet renders as a quoted note naming it. Both are listed on every run.

    uv run --no-sync python scripts/render_docs.py            # write every page
    uv run --no-sync python scripts/render_docs.py --check    # fail on an unknown name or
                                                              # a page out of date
    uv run --no-sync python scripts/render_docs.py --check --strict   # also fail on pending

The rendered pages are committed so that GitHub shows real numbers. Edit the template,
never the rendered page: `--check` fails when a rendered page differs from its template.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VALUES = ROOT / "docs" / "values.json"
FIGURES = ROOT / "docs" / "figures.yml"
TEMPLATE_SUFFIX = ".template.md"

PLACEHOLDER = re.compile(r"\{\{\s*([^{}\n]+?)\s*\}\}")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class RenderError(ValueError):
    """A placeholder that cannot be rendered: an unknown name or a malformed form."""


def templates() -> list[Path]:
    """Every template: README.template.md at the root and docs/**/*.template.md."""
    found = sorted(ROOT.glob(f"*{TEMPLATE_SUFFIX}")) + sorted(
        (ROOT / "docs").rglob(f"*{TEMPLATE_SUFFIX}")
    )
    return found


def output_for(template: Path) -> Path:
    return template.with_name(template.name[: -len(TEMPLATE_SUFFIX)] + ".md")


def load_values() -> dict:
    values = json.loads(VALUES.read_text())
    for name, value in values.items():
        if not NAME.match(name):
            raise RenderError(f"{VALUES.name}: {name!r} is not a value name (a-z, 0-9, _)")
        if value is not None and not isinstance(value, (int, float, str)):
            raise RenderError(f"{VALUES.name}: {name} must be a number, a string or null")
    return values


def load_figures() -> dict:
    figures = yaml.safe_load(FIGURES.read_text()) or {}
    for figure_id, entry in figures.items():
        missing = {"path", "title"} - set(entry or {})
        if missing:
            raise RenderError(f"{FIGURES.name}: {figure_id} lacks {sorted(missing)}")
    return figures


def format_value(value, spec: str) -> str:
    if spec:
        return format(value, spec)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


class Page:
    def __init__(self, template: Path, values: dict, figures: dict):
        self.template = template
        self.output = output_for(template)
        self.values = values
        self.figures = figures
        self.figure_count = 0
        self.pending: list[str] = []
        self.errors: list[str] = []

    def where(self, line: int) -> str:
        return f"{self.template.relative_to(ROOT)}:{line}"

    def link(self, path: str) -> str:
        """A repository-root path, relative to this page's directory."""
        here = self.output.parent.relative_to(ROOT).as_posix()
        return posixpath.relpath(path, here if here != "." else ".")

    def value(self, body: str, line: int) -> str:
        name, _, spec = body.partition(":")
        name = name.strip()
        if not NAME.match(name):
            self.errors.append(f"{self.where(line)}: malformed placeholder {{{{{body}}}}}")
            return body
        if name not in self.values:
            self.errors.append(f"{self.where(line)}: unknown value {name!r}")
            return f"{{{{{body}}}}}"
        value = self.values[name]
        if value is None:
            self.pending.append(f"{self.where(line)}: value {name}")
            return f"[pending: {name}]"
        try:
            return format_value(value, spec.strip())
        except (ValueError, TypeError) as error:
            self.errors.append(f"{self.where(line)}: {name} with format {spec!r}: {error}")
            return str(value)

    def figure(self, figure_id: str, line: int) -> str:
        entry = self.figures.get(figure_id)
        if entry is None:
            self.errors.append(f"{self.where(line)}: unknown figure {figure_id!r}")
            return f"{{{{figure: {figure_id}}}}}"
        path = entry["path"]
        if not (ROOT / path).is_file():
            self.pending.append(f"{self.where(line)}: figure {figure_id} ({path})")
            return f"> **Figure pending:** {entry['title']} (`{path}`)."
        self.figure_count += 1
        alt = entry.get("alt") or entry["title"]
        image = f"![{alt}]({self.link(path)})"
        caption = entry.get("caption")
        if not caption:
            return image
        caption = PLACEHOLDER.sub(lambda match: self.value(match.group(1), line), caption)
        caption = " ".join(caption.split())
        lead = f"Figure {self.figure_count}."
        return f"{image}\n\n*{lead} {caption}*"

    def render(self) -> str:
        text = self.template.read_text()
        out = []
        fence = None
        for number, line in enumerate(text.splitlines(keepends=True), start=1):
            marker = FENCE.match(line)
            if fence is not None:
                if marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence):
                    fence = None
                out.append(line)
                continue
            if marker:
                fence = marker.group(1)
                out.append(line)
                continue
            stripped = line.strip()
            whole = PLACEHOLDER.fullmatch(stripped)
            if whole and whole.group(1).startswith("figure:"):
                ending = "\n" if line.endswith("\n") else ""
                out.append(self.figure(whole.group(1)[len("figure:") :].strip(), number) + ending)
                continue
            if "figure:" in line and PLACEHOLDER.search(line):
                for match in PLACEHOLDER.finditer(line):
                    if match.group(1).startswith("figure:"):
                        self.errors.append(f"{self.where(number)}: a figure goes on its own line")
            out.append(PLACEHOLDER.sub(lambda match: self.value(match.group(1), number), line))
        if fence is not None:
            self.errors.append(f"{self.where(len(out))}: unclosed code fence")
        return "".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="write nothing; fail on errors or stale pages"
    )
    parser.add_argument(
        "--strict", action="store_true", help="also fail while any value or figure is pending"
    )
    args = parser.parse_args(argv)

    values = load_values()
    figures = load_figures()
    errors, pending, stale = [], [], []
    for template in templates():
        page = Page(template, values, figures)
        rendered = page.render()
        errors += page.errors
        pending += page.pending
        current = page.output.read_text() if page.output.exists() else None
        if current != rendered:
            if args.check:
                stale.append(str(page.output.relative_to(ROOT)))
            elif not page.errors:
                page.output.write_text(rendered)
                print(f"wrote {page.output.relative_to(ROOT)}")

    for line in pending:
        print(f"pending  {line}")
    for line in errors:
        print(f"error    {line}", file=sys.stderr)
    for name in stale:
        print(f"stale    {name} (run scripts/render_docs.py)", file=sys.stderr)
    failed = bool(errors or stale or (args.strict and pending))
    print(
        f"{len(templates())} templates, {len(pending)} pending, {len(errors)} errors"
        + (f", {len(stale)} stale" if args.check else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
