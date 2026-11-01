"""The documentation pages as the checks read them: front matter, code spans, fenced blocks.

Every location is a (page, line) pair, with the page relative to the repository root, so a
failure names where to look.
"""

import dataclasses
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
INTERNAL = DOCS / "rewrite"  # the design records: not part of the site, not checked
RUNNABLE = "<!-- runnable -->"
NOT_RUNNABLE = re.compile(r"^<!-- not-runnable: .+ -->$")
# Fences in these languages must say whether they run (see contributing/writing-docs.md).
COMMAND_LANGUAGES = ("bash", "sh", "shell", "console")
CHECKED_LANGUAGES = ("python", *COMMAND_LANGUAGES)

FENCE = re.compile(r"^(?P<indent>\s*)(?P<fence>`{3,}|~{3,})\s*(?P<language>[\w+-]*)")
CODE_SPAN = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
AUTOREF = re.compile(r"\]\[((?:bnb|bench)(?:\.\w+)+)\]")
DIRECTIVE = re.compile(r"^:::\s+((?:bnb|bench)(?:\.\w+)*)\s*$")
DOTTED = re.compile(r"^(?:bnb|bench)(?:\.\w+)+$")
REPOSITORY_PATH = re.compile(r"^(?:src|tests|bench|docs)/[\w./-]*(?:::\w+)?$")


@dataclasses.dataclass(frozen=True)
class Fence:
    page: str
    line: int  # the opening fence's line
    language: str
    marker: str  # the line before the fence (stripped), "" if none
    code: str

    @property
    def runnable(self) -> bool:
        return self.marker == RUNNABLE

    @property
    def location(self) -> str:
        return f"{self.page}:{self.line}"


@dataclasses.dataclass(frozen=True)
class Page:
    path: pathlib.Path

    @property
    def name(self) -> str:
        return str(self.path.relative_to(ROOT))

    @property
    def lines(self) -> list[str]:
        return self.path.read_text().splitlines()

    def front_matter(self) -> dict[str, list[str]]:
        """The page's front matter: each key to its list of values. Supports the forms
        these pages use: `key: [a, b]`, `key: []` and a block list (`key:` then `- a`)."""
        lines = self.lines
        if not lines or lines[0].strip() != "---":
            return {}
        out: dict[str, list[str]] = {}
        key = None
        for line in lines[1:]:
            if line.strip() == "---":
                return out
            item = re.match(r"^\s+-\s+(.+?)\s*$", line)
            if item and key is not None:
                out[key].append(item.group(1).strip("'\""))
                continue
            pair = re.match(r"^(\w+):\s*(.*?)\s*$", line)
            if not pair:
                raise ValueError(f"{self.name}: cannot read front matter line {line!r}")
            key, value = pair.groups()
            if value.startswith("[") and value.endswith("]"):
                inner = value[1:-1].strip()
                out[key] = [v.strip().strip("'\"") for v in inner.split(",")] if inner else []
            elif value:
                out[key] = [value.strip("'\"")]
            else:
                out[key] = []
        raise ValueError(f"{self.name}: front matter is not closed")

    def fences(self) -> list[Fence]:
        out = []
        lines = self.lines
        index = 0
        while index < len(lines):
            opening = FENCE.match(lines[index])
            if not opening:
                index += 1
                continue
            fence, indent = opening.group("fence"), len(opening.group("indent"))
            body = []
            end = index + 1
            while end < len(lines) and not lines[end].strip().startswith(fence):
                body.append(lines[end][indent:])
                end += 1
            before = index - 1
            while before >= 0 and not lines[before].strip():
                before -= 1
            marker = lines[before].strip() if before >= 0 else ""
            out.append(Fence(self.name, index + 1, opening.group("language").lower(), marker,
                             "\n".join(body) + "\n"))
            index = end + 1
        return out

    def prose(self) -> list[tuple[int, str]]:
        """(line number, text) of every line outside fenced blocks and front matter."""
        lines = self.lines
        inside = set()
        for fence in self.fences():
            end = fence.line + len(fence.code.splitlines())
            inside.update(range(fence.line, end + 2))
        if lines and lines[0].strip() == "---":
            closing = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
            inside.update(range(1, closing + 2))
        return [(number, line) for number, line in enumerate(lines, 1) if number not in inside]

    def code_references(self) -> list[tuple[int, str]]:
        """(line, reference): dotted names (in backticks, mkdocstrings links or `:::`
        directives) and repository paths (in backticks), outside fenced blocks."""
        out = []
        for number, line in self.prose():
            directive = DIRECTIVE.match(line)
            if directive:
                out.append((number, directive.group(1)))
            out.extend((number, name) for name in AUTOREF.findall(line))
            for span in CODE_SPAN.findall(line):
                span = span.strip()
                if DOTTED.match(span) or REPOSITORY_PATH.match(span):
                    out.append((number, span))
        return out


def pages() -> list[Page]:
    """Every Markdown page of the site (docs/, without the internal design records)."""
    return [Page(path) for path in sorted(DOCS.rglob("*.md"))
            if INTERNAL not in path.parents]


def fences(languages: tuple[str, ...] = CHECKED_LANGUAGES) -> list[Fence]:
    return [fence for page in pages() for fence in page.fences() if fence.language in languages]
