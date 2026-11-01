"""Every code reference in the docs resolves, and every page says what it covers and
whether its examples run. A failure names the page and the line."""

import importlib.util

import pytest

from . import pages, resolve

PAGES = pages.pages()


def _load_hook():
    spec = importlib.util.spec_from_file_location(
        "docs_generated_hook", pages.DOCS / "hooks" / "generated.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HOOK = _load_hook()


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_every_code_reference_resolves(page):
    broken = [f"{page.name}:{line}: `{text}`: {why}"
              for line, text in page.code_references()
              if (why := resolve.reference(text)) is not None]
    assert not broken, "\n".join(broken)


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_every_python_block_parses_and_its_imports_resolve(page):
    broken = []
    for fence in page.fences():
        if fence.language != "python":
            continue
        try:
            imports = resolve.python_imports(fence.code)
        except SyntaxError as error:
            broken.append(f"{fence.location}: does not parse: {error}")
            continue
        broken.extend(f"{page.name}:{fence.line + line}: import {name}: {why}"
                      for line, name in imports
                      if (why := resolve.dotted_name(name)) is not None)
    assert not broken, "\n".join(broken)


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_every_command_and_python_block_says_whether_it_runs(page):
    unmarked = [f"{fence.location}: a {fence.language} block needs `{pages.RUNNABLE}` or "
                f"`<!-- not-runnable: reason -->` on the line before it"
                for fence in page.fences() if fence.language in pages.CHECKED_LANGUAGES
                and not (fence.runnable or pages.NOT_RUNNABLE.match(fence.marker))]
    assert not unmarked, "\n".join(unmarked)


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_every_page_declares_what_it_covers_and_the_paths_exist(page):
    front = page.front_matter()
    assert "covers" in front, f"{page.name}:1: front matter needs `covers: [...]`"
    missing = [f"{page.name}:1: covers {path}, which does not exist"
               for path in front["covers"] if not (pages.ROOT / path).exists()]
    assert not missing, "\n".join(missing)


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_every_generated_table_is_known(page):
    text = page.path.read_text()
    unknown = [f"{page.name}:{text[:match.start()].count(chr(10)) + 1}: unknown table "
               f"{match.group(1)!r}" for match in HOOK.MARKER.finditer(text)
               if match.group(1) not in HOOK.TABLES]
    assert not unknown, "\n".join(unknown)


def test_every_marker_like_comment_is_a_valid_marker():
    """A misspelt marker would silently render nothing."""
    bad = []
    for page in PAGES:
        for number, line in enumerate(page.lines, 1):
            stripped = line.strip()
            if not stripped.startswith("<!--"):
                continue
            if not (HOOK.MARKER.match(stripped) or stripped == pages.RUNNABLE
                    or pages.NOT_RUNNABLE.match(stripped)):
                bad.append(f"{page.name}:{number}: {stripped}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("name", sorted(set(HOOK.TABLES) - {"roadmap"}))
def test_every_generated_table_renders_from_the_code(name):
    table = HOOK.render(name)
    assert table.startswith("| "), table


def test_the_roadmap_data_is_valid():
    pytest.importorskip("yaml", reason="the roadmap needs PyYAML: uv sync --extra docs")
    entries = HOOK.roadmap_entries()
    assert [entry["section"] for entry in entries], "the roadmap is empty"


def test_the_roadmap_design_records_exist():
    """Without PyYAML: every path the roadmap names exists."""
    text = (pages.DOCS / "roadmap.yml").read_text()
    paths = [line.split(":", 1)[1].strip() for line in text.splitlines()
             if line.strip().startswith("spec:")]
    missing = [path for path in paths if path != "null" and not (pages.ROOT / path).exists()]
    assert not missing, missing
