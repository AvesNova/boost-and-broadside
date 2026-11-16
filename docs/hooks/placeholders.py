"""Placeholders: results that drop into the pages and the README as they arrive.

Three forms, anywhere in a page's Markdown outside fenced blocks and code spans:

    {{name}}  or  {{name:.0%}}
        A value from the values file, `docs/results/values.json`: one flat JSON object,
        {name: number, string or null}. After the colon, a Python format spec. A name the
        file does not hold fails in every mode (a typo). A null is a value not yet measured.
    {{pending: key | one-line description}}
        Text that can only be written after seeing the results. The key names it; the
        description says what will go there.
    {{figure: docs/results/win_rate_fan.png | alt text}}
        An image, on a line of its own, the path from the repository root (the light
        variant). Shown when the file exists; otherwise a slot that names it. When the dark
        variant exists beside it (win_rate_fan_dark.png), both are shown, each in its
        colour scheme: Material's #only-light and #only-dark on the site, a <picture> with
        a prefers-color-scheme source in the README.

The values file can be swapped with the environment variable BNB_RESULTS_VALUES (a path,
from the repository root or absolute), to preview the pages with example numbers without
touching the real file.

The mode is the environment variable BNB_RESULTS: `draft` (the default) renders every
outstanding placeholder visibly (a null value as its name in angle brackets, a pending
marker or a missing figure as a quoted note) and reports them; `publish` refuses to render
a page that still has one.

The MkDocs build renders the site's pages through `on_page_markdown`. GitHub shows
README.md as it is, so README.md is rendered from README.template.md by this file:

    uv run --no-sync python docs/hooks/placeholders.py readme        # write README.md
    uv run --no-sync python docs/hooks/placeholders.py check         # list what is open
    BNB_RESULTS=publish uv run --no-sync python docs/hooks/placeholders.py check
    # a preview with example numbers, the real README.md and values file untouched:
    BNB_RESULTS_VALUES=example.json uv run --no-sync python docs/hooks/placeholders.py readme --out preview/README.md
"""

import dataclasses
import json
import logging
import os
import pathlib
import posixpath
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
VALUES = ROOT / "docs" / "results" / "values.json"
README_TEMPLATE = ROOT / "README.template.md"
README = ROOT / "README.md"
MODES = ("draft", "publish")
README_NOTE = ("<!-- Generated from README.template.md by docs/hooks/placeholders.py; edit the "
               "template, then run: uv run --no-sync python docs/hooks/placeholders.py readme -->")

PLACEHOLDER = re.compile(r"\{\{([^{}\n]+)\}\}")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
CODE_SPAN = re.compile(r"(`+)(?:.+?)\1")
NAME = re.compile(r"^[a-z][a-z0-9_]*$")

log = logging.getLogger("mkdocs.hooks.placeholders")


class PlaceholderError(ValueError):
    """A placeholder that cannot be rendered: an unknown name, a bad form, or (in publish
    mode) anything still outstanding."""


@dataclasses.dataclass(frozen=True)
class Outstanding:
    """A placeholder still open: a value not measured, a pending text or a missing figure."""

    page: str  # from the repository root
    line: int
    kind: str  # "value", "pending" or "figure"
    key: str  # the value's name, the pending key or the figure's path
    description: str = ""

    def __str__(self) -> str:
        detail = f": {self.description}" if self.description else ""
        return f"{self.page}:{self.line}: {self.kind} {self.key}{detail}"


def dark_variant(path: str) -> str:
    """A figure's dark theme variant: the same name with "_dark" before the extension
    (`bnb.figures.theme.suffix`)."""
    stem, dot, extension = path.rpartition(".")
    return f"{stem}_dark.{extension}" if dot else f"{path}_dark"


def mode() -> str:
    """The mode from BNB_RESULTS: draft (the default) or publish."""
    chosen = os.environ.get("BNB_RESULTS", "draft").strip() or "draft"
    if chosen not in MODES:
        raise PlaceholderError(f"BNB_RESULTS={chosen!r}: draft or publish")
    return chosen


def values_path() -> pathlib.Path:
    """The values file: BNB_RESULTS_VALUES when set (from the repository root or absolute),
    otherwise docs/results/values.json."""
    chosen = os.environ.get("BNB_RESULTS_VALUES", "").strip()
    return ROOT / chosen if chosen else VALUES


def load_values(path: pathlib.Path | None = None) -> dict:
    """The values file's {name: number, string or null}; empty when there is no file."""
    path = values_path() if path is None else path
    if not path.exists():
        return {}
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise PlaceholderError(f"{path}: one flat JSON object of named values")
    for name, value in values.items():
        if not NAME.match(name):
            raise PlaceholderError(f"{path}: {name!r} is not a value name (a-z, 0-9, _)")
        if value is not None and not isinstance(value, (int, float, str)):
            raise PlaceholderError(f"{path}: {name} is a {type(value).__name__}; a value is "
                                   "a number, a string or null")
    return values


def _format(value, spec: str) -> str:
    if spec:
        return format(value, spec)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _parts(body: str) -> tuple[str, str]:
    """`key | description` split at the first bar, both stripped."""
    key, _, description = body.partition("|")
    return key.strip(), description.strip()


@dataclasses.dataclass
class _Page:
    name: str  # from the repository root
    values: dict
    outstanding: list[Outstanding]

    def link(self, path: str) -> str:
        return posixpath.relpath(path, posixpath.dirname(self.name) or ".")

    def figure(self, path: str, alt: str) -> str:
        """An existing figure, in both theme variants when its dark one (`dark_variant`)
        exists: on the site, Material's #only-light and #only-dark images; in the README,
        GitHub's <picture> with the light image as the fallback."""
        dark = dark_variant(path)
        if not (ROOT / dark).is_file():
            return f"![{alt}]({self.link(path)})"
        if self.name == str(README_TEMPLATE.relative_to(ROOT)):
            quoted = alt.replace('"', "&quot;")
            return (f'<picture><source media="(prefers-color-scheme: dark)" '
                    f'srcset="{self.link(dark)}"><img alt="{quoted}" '
                    f'src="{self.link(path)}"></picture>')
        return (f"![{alt}]({self.link(path)}#only-light)"
                f"![{alt}]({self.link(dark)}#only-dark)")

    def placeholder(self, match: re.Match, number: int, whole_line: bool) -> str:
        body = match.group(1).strip()
        kind, colon, rest = body.partition(":")
        kind = kind.strip()
        if colon and kind == "pending":
            key, description = _parts(rest)
            if not NAME.match(key):
                raise PlaceholderError(f"{self.name}:{number}: pending key {key!r} "
                                       "(a-z, 0-9, _)")
            self.outstanding.append(Outstanding(self.name, number, "pending", key, description))
            if whole_line:
                return f"> **Pending ({key}).** {description}"
            return f"**⟨pending: {description or key}⟩**"
        if colon and kind == "figure":
            path, alt = _parts(rest)
            if not whole_line:
                raise PlaceholderError(f"{self.name}:{number}: a figure goes on a line of "
                                       "its own")
            if (ROOT / path).is_file():
                return self.figure(path, alt)
            self.outstanding.append(Outstanding(self.name, number, "figure", path, alt))
            return f"> **Figure to come** ({path}): {alt}"
        name, spec = kind, rest.strip() if colon else ""
        if not NAME.match(name):
            raise PlaceholderError(f"{self.name}:{number}: {{{{{body}}}}} is not a value, "
                                   "pending marker or figure")
        if name not in self.values:
            raise PlaceholderError(f"{self.name}:{number}: no value {name!r} in "
                                   f"{values_path()}")
        value = self.values[name]
        if value is None:
            self.outstanding.append(Outstanding(self.name, number, "value", name))
            return f"**⟨{name}⟩**"
        try:
            return _format(value, spec)
        except ValueError as error:
            raise PlaceholderError(f"{self.name}:{number}: {name}: {error}") from error

    def line(self, text: str, number: int) -> str:
        whole = PLACEHOLDER.fullmatch(text.strip())
        if whole is not None:
            indent = text[: len(text) - len(text.lstrip())]
            return indent + self.placeholder(whole, number, True)
        out, start = [], 0
        for span in CODE_SPAN.finditer(text):  # code spans keep their braces
            out.append(PLACEHOLDER.sub(lambda m: self.placeholder(m, number, False),
                                       text[start:span.start()]))
            out.append(span.group(0))
            start = span.end()
        out.append(PLACEHOLDER.sub(lambda m: self.placeholder(m, number, False), text[start:]))
        return "".join(out)


def render(markdown: str, page: str, values: dict | None = None,
           chosen_mode: str | None = None) -> tuple[str, list[Outstanding]]:
    """The page (`page`: its path from the repository root, for the figures' links) with
    every placeholder outside fenced blocks and code spans rendered, and what is still
    outstanding. Raises PlaceholderError on an unknown name or a bad form, and in publish
    mode when anything is outstanding."""
    state = _Page(page, load_values() if values is None else values, [])
    out, fence = [], None
    for number, text in enumerate(markdown.split("\n"), 1):
        opening = FENCE.match(text)
        if fence is None and opening:
            fence = opening.group(1)
        elif fence is not None:
            if text.strip().startswith(fence):
                fence = None
        else:
            text = state.line(text, number)
        out.append(text)
    if (chosen_mode or mode()) == "publish" and state.outstanding:
        listed = "\n  ".join(map(str, state.outstanding))
        raise PlaceholderError(f"publish mode: {len(state.outstanding)} placeholder(s) still "
                               f"open:\n  {listed}")
    return "\n".join(out), state.outstanding


def render_readme(chosen_mode: str | None = None) -> tuple[str, list[Outstanding]]:
    """README.md's text, from README.template.md and the values file."""
    text, outstanding = render(README_TEMPLATE.read_text(), "README.template.md",
                               chosen_mode=chosen_mode)
    return f"{README_NOTE}\n\n{text}", outstanding


def site_pages() -> list[pathlib.Path]:
    """The site's Markdown pages (docs/, without the internal design records)."""
    internal = ROOT / "docs" / "rewrite"
    return sorted(path for path in (ROOT / "docs").rglob("*.md") if internal not in path.parents)


def check(chosen_mode: str | None = None) -> list[Outstanding]:
    """Render every site page and the README template; what is outstanding. Raises on an
    unknown name or a bad form, and in publish mode when anything is outstanding, listing
    every open placeholder of every page."""
    outstanding = []
    for path in [*site_pages(), README_TEMPLATE]:
        page = str(path.relative_to(ROOT))
        outstanding += render(path.read_text(), page, chosen_mode="draft")[1]
    if (chosen_mode or mode()) == "publish" and outstanding:
        listed = "\n  ".join(map(str, outstanding))
        raise PlaceholderError(f"publish mode: {len(outstanding)} placeholder(s) still "
                               f"open:\n  {listed}")
    return outstanding


# ------------------------------------------------------------------ the MkDocs hook

def on_page_markdown(markdown: str, page, config, files) -> str:
    text, outstanding = render(markdown, f"docs/{page.file.src_path}")
    if outstanding:
        log.info("%s: %d placeholder(s) open (draft)", page.file.src_path, len(outstanding))
    return text


# ------------------------------------------------------------------ the command

def main(arguments: list[str]) -> int:
    usage = "usage: placeholders.py {readme,check} [--publish] [--out PATH]"
    if not arguments or arguments[0] not in ("readme", "check"):
        print(usage, file=sys.stderr)
        return 2
    chosen_mode = "publish" if "--publish" in arguments[1:] else None
    out = README
    if "--out" in arguments[1:]:
        at = arguments.index("--out")
        if at + 1 >= len(arguments):
            print(usage, file=sys.stderr)
            return 2
        out = pathlib.Path(arguments[at + 1])
    try:
        if arguments[0] == "readme":
            text, outstanding = render_readme(chosen_mode)
            out.write_text(text)
            print(f"wrote {out} ({len(outstanding)} placeholder(s) open)")
            return 0
        outstanding = check(chosen_mode)
    except PlaceholderError as error:
        print(error, file=sys.stderr)
        return 1
    for item in outstanding:
        print(item)
    print(f"{len(outstanding)} placeholder(s) open ({chosen_mode or mode()} mode)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
