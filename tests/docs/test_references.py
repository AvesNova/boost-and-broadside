"""Every code reference in the docs resolves, every page says what it covers, and every
fence says what it is and whether it runs. A failure names the page and the line.

The pages are the site's and README.md (which GitHub renders without the hook).
"""

import importlib.util

import pytest

from . import pages, resolve

PAGES = pages.pages()
SITE = [page for page in PAGES if page.on_site]


def _load_hook():
    spec = importlib.util.spec_from_file_location(
        "docs_generated_hook", pages.DOCS / "hooks" / "generated.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HOOK = _load_hook()


def _ids(page) -> str:
    return page.name


@pytest.mark.parametrize("page", PAGES, ids=_ids)
def test_every_full_name_and_path_resolves(page):
    broken = [f"{page.name}:{line}: `{text}`: {why}"
              for line, text in page.code_references()
              if (why := resolve.reference(text)) is not None]
    assert not broken, "\n".join(broken)


@pytest.mark.parametrize("page", PAGES, ids=_ids)
def test_every_short_name_names_one_thing_in_the_code(page):
    broken = [f"{page.name}:{line}: {why}" for line, span in page.code_spans()
              if (why := resolve.short_name(span)) is not None]
    assert not broken, "\n".join(broken)


@pytest.mark.parametrize("page", PAGES, ids=_ids)
def test_every_bnb_command_uses_real_options(page):
    broken = [f"{page.name}:{line}: bnb{arguments}: {why}"
              for line, arguments in page.commands()
              if (why := resolve.command(arguments)) is not None]
    assert not broken, "\n".join(broken)


def test_the_command_check_takes_positional_arguments_but_not_as_options():
    assert resolve.command("runs show 2026-11-02-mixed-purple-pumpkin") is None
    assert resolve.command("runs upload RUN --at 268435456 --league") is None
    assert "has no option --action" in resolve.command("runs --action show")
    assert "has no option --run" in resolve.command("runs show --run RUN")


@pytest.mark.parametrize("page", PAGES, ids=_ids)
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


@pytest.mark.parametrize("page", PAGES, ids=_ids)
def test_every_fence_names_its_language(page):
    unnamed = [f"{fence.location}: a fence needs a language (bash, python, markdown, text, ...)"
               for fence in page.fences() if not fence.language]
    assert not unnamed, "\n".join(unnamed)


@pytest.mark.parametrize("page", PAGES, ids=_ids)
def test_every_command_and_python_block_says_whether_it_runs(page):
    unmarked = [f"{fence.location}: a {fence.language or 'plain'} block needs "
                f"`{pages.RUNNABLE}` or `<!-- not-runnable: reason -->` on the line before it"
                for fence in page.fences() if fence.needs_marker
                and not (fence.runnable or pages.NOT_RUNNABLE.match(fence.marker))]
    assert not unmarked, "\n".join(unmarked)


@pytest.mark.parametrize("page", SITE, ids=_ids)
def test_every_page_declares_what_it_covers_and_the_paths_exist(page):
    front = page.front_matter()
    assert "covers" in front, f"{page.name}:1: front matter needs `covers: [...]`"
    missing = [f"{page.name}:1: covers {path}, which does not exist"
               for path in front["covers"] if not (pages.ROOT / path).exists()]
    assert not missing, "\n".join(missing)


@pytest.mark.parametrize("page", PAGES, ids=_ids)
def test_every_generated_block_is_known_and_only_on_the_site(page):
    text = page.path.read_text()
    bad = []
    for match in HOOK.MARKER.finditer(text):
        line = text[:match.start()].count("\n") + 1
        if not page.on_site:
            bad.append(f"{page.name}:{line}: generated blocks render only on the site")
        elif match.group(1) not in HOOK.BLOCKS:
            bad.append(f"{page.name}:{line}: unknown generated block {match.group(1)!r}")
    assert not bad, "\n".join(bad)


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


def test_every_readme_link_resolves():
    """The site's links are checked by the strict build; README.md's here."""
    readme = pages.Page(pages.README)
    broken = [f"README.md:{line}: {target}" for line, target in readme.links()
              if not (pages.ROOT / target.partition("#")[0]).exists()]
    assert not broken, "\n".join(broken)


@pytest.mark.parametrize("name", sorted(HOOK.BLOCKS))
def test_every_generated_block_renders(name):
    if name in ("roadmap", "roadmap-summary", "roadmap-in-progress", "packages",
                "section-status"):
        pytest.importorskip("yaml", reason="the data files need PyYAML: the docs dependency group")
    page = "explanation/in-progress/league.md" if name == "section-status" \
        else "explanation/index.md"
    assert HOOK.render(name, page).strip()


def test_the_data_files_name_pages_and_records_that_exist():
    """Without PyYAML: every path the roadmap and the package list name exists."""
    missing = []
    for data in ("roadmap.yml", "packages.yml"):
        for line in (pages.DOCS / data).read_text().splitlines():
            key, _, value = line.strip().removeprefix("- ").partition(":")
            value = value.strip()
            if key == "spec" and value != "null" and not (pages.ROOT / value).exists():
                missing.append(f"docs/{data}: {value}")
            if key in ("page", "reference") and not (pages.DOCS / value).exists():
                missing.append(f"docs/{data}: {value}")
    assert not missing, missing
