"""The results placeholders (`docs/hooks/placeholders.py`): every value a page quotes is in
the values file, every placeholder is well formed, and README.md is the rendered
README.template.md. In publish mode (BNB_RESULTS=publish) nothing may be open; in draft
mode, the default, what is open is listed as a warning and the tests pass."""

import importlib.util
import re
import warnings

import pytest

from . import pages


def _load():
    spec = importlib.util.spec_from_file_location(
        "docs_placeholders_hook", pages.DOCS / "hooks" / "placeholders.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLACEHOLDERS = _load()
VALUES = {"win_rate": 0.634, "ratio": 1.5, "steps": 100000000.0, "run": "r0-rl",
          "missing": None}


def _render(text: str, mode: str = "draft", page: str = "docs/results/page.md"):
    return PLACEHOLDERS.render(text, page, VALUES, mode)


def test_a_value_renders_with_its_format():
    text, open_ = _render("{{win_rate:.0%}} at {{steps:,.0f}} by {{run}}, {{ratio}}")
    assert text == "63% at 100,000,000 by r0-rl, 1.5"
    assert open_ == []


def test_a_null_value_is_open_and_shows_its_name():
    text, open_ = _render("ratio {{missing}}")
    assert "missing" in text and "{{" not in text
    assert [(item.kind, item.key, item.line) for item in open_] == [("value", "missing", 1)]


def test_an_unknown_name_fails_in_every_mode():
    for mode in PLACEHOLDERS.MODES:
        with pytest.raises(PLACEHOLDERS.PlaceholderError, match="no value 'typo'"):
            _render("{{typo}}", mode)


def test_pending_on_its_own_line_and_inline():
    text, open_ = _render("{{pending: headline | The headline.}}\nA {{pending: x | one word}}.")
    block, inline = text.split("\n")
    assert block.startswith("> ") and "The headline." in block
    assert "one word" in inline and "{{" not in inline
    assert [(item.kind, item.key) for item in open_] == [("pending", "headline"), ("pending", "x")]


def test_a_figure_links_when_the_file_exists_and_is_open_when_not():
    text, open_ = _render("{{figure: docs/index.md | alt}}", page="docs/results/page.md")
    assert text == "![alt](../index.md)" and open_ == []
    text, open_ = _render("{{figure: docs/results/none.png | alt}}", page="README.md")
    assert "docs/results/none.png" in text and not text.startswith("!")
    assert [(item.kind, item.key) for item in open_] == [("figure", "docs/results/none.png")]


def test_a_figure_inside_a_sentence_fails():
    with pytest.raises(PLACEHOLDERS.PlaceholderError, match="line of its own"):
        _render("see {{figure: docs/index.md | alt}} here")


def test_fences_and_code_spans_are_left_alone():
    text = "```markdown\n{{typo}}\n```\nWrite `{{typo}}` or ``{{pending: a | b}}``."
    assert _render(text) == (text, [])


def test_publish_mode_fails_on_anything_open():
    for text in ("{{missing}}", "{{pending: a | b}}", "{{figure: docs/results/none.png | a}}"):
        with pytest.raises(PLACEHOLDERS.PlaceholderError, match="publish mode"):
            _render(text, "publish")
    assert _render("{{ratio}}", "publish") == ("1.5", [])


def test_the_mode_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("BNB_RESULTS", raising=False)
    assert PLACEHOLDERS.mode() == "draft"
    monkeypatch.setenv("BNB_RESULTS", "publish")
    assert PLACEHOLDERS.mode() == "publish"
    monkeypatch.setenv("BNB_RESULTS", "final")
    with pytest.raises(PLACEHOLDERS.PlaceholderError):
        PLACEHOLDERS.mode()


def test_the_values_file_is_well_formed():
    PLACEHOLDERS.load_values()


def test_every_page_renders():
    """Every value resolves and every placeholder is well formed; in publish mode nothing
    is open. In draft mode the open ones are listed."""
    open_ = PLACEHOLDERS.check()
    if open_:
        warnings.warn(f"{len(open_)} results placeholder(s) open (draft mode):\n  "
                      + "\n  ".join(map(str, open_)), stacklevel=1)


def test_the_readme_is_the_rendered_template():
    text, _ = PLACEHOLDERS.render_readme()
    assert pages.README.read_text() == text, (
        "README.md is not README.template.md rendered: run "
        "uv run --no-sync python docs/hooks/placeholders.py readme")


def test_the_values_file_comes_from_the_environment(monkeypatch, tmp_path):
    example = tmp_path / "example.json"
    example.write_text('{"elo_example": 1234.5}')
    monkeypatch.setenv("BNB_RESULTS_VALUES", str(example))
    assert PLACEHOLDERS.values_path() == example
    assert PLACEHOLDERS.render("{{elo_example:.0f}}", "page.md")[0] == "1234"
    monkeypatch.delenv("BNB_RESULTS_VALUES")
    assert PLACEHOLDERS.values_path() == PLACEHOLDERS.VALUES


def test_publish_check_lists_every_open_placeholder(monkeypatch, tmp_path):
    """In publish mode `check` names every open placeholder of every page, not the first."""
    first, second = tmp_path / "a.md", tmp_path / "b.md"
    first.write_text("{{pending: one | the first}}\n")
    second.write_text("{{pending: two | the second}}\n")
    monkeypatch.setattr(PLACEHOLDERS, "ROOT", tmp_path)
    monkeypatch.setattr(PLACEHOLDERS, "site_pages", lambda: [first])
    monkeypatch.setattr(PLACEHOLDERS, "README_TEMPLATE", second)
    with pytest.raises(PLACEHOLDERS.PlaceholderError) as raised:
        PLACEHOLDERS.check("publish")
    assert "2 placeholder(s)" in str(raised.value)
    assert "pending one" in str(raised.value) and "pending two" in str(raised.value)
    assert len(PLACEHOLDERS.check("draft")) == 2


def test_a_figure_with_a_dark_variant_follows_the_colour_scheme(monkeypatch, tmp_path):
    (tmp_path / "docs" / "results").mkdir(parents=True)
    for name in ("chart.png", "chart_dark.png"):
        (tmp_path / "docs" / "results" / name).write_bytes(b"png")
    monkeypatch.setattr(PLACEHOLDERS, "ROOT", tmp_path)
    monkeypatch.setattr(PLACEHOLDERS, "README_TEMPLATE", tmp_path / "README.template.md")
    line = "{{figure: docs/results/chart.png | a chart}}"
    site, _ = PLACEHOLDERS.render(line, "docs/results/page.md", VALUES, "publish")
    assert site == "![a chart](chart.png#only-light)![a chart](chart_dark.png#only-dark)"
    readme, _ = PLACEHOLDERS.render(line, "README.template.md", VALUES, "publish")
    assert 'media="(prefers-color-scheme: dark)" srcset="docs/results/chart_dark.png"' in readme
    assert '<img alt="a chart" src="docs/results/chart.png">' in readme


def _captioned(monkeypatch, tmp_path, **entry):
    """A chart.png (and its dark twin) under tmp_path, held by the captions file."""
    (tmp_path / "docs" / "results").mkdir(parents=True)
    for name in ("chart.png", "chart_dark.png"):
        (tmp_path / "docs" / "results" / name).write_bytes(b"png")
    monkeypatch.setattr(PLACEHOLDERS, "ROOT", tmp_path)
    monkeypatch.setattr(PLACEHOLDERS, "README_TEMPLATE", tmp_path / "README.template.md")
    held = PLACEHOLDERS.captions.parse({"notes": {"caveat": "A caveat."}, "figures": {
        "chart": dict(title="Short title", caption="At {{steps:,.0f}} steps.",
                      notes=["caveat"], **entry)}})
    monkeypatch.setattr(PLACEHOLDERS.captions, "load", lambda: held)


def test_a_captioned_figure_is_a_figure_with_its_caption(monkeypatch, tmp_path):
    """On the site: a <figure> of both variants, each telling the lightbox its title and
    the caption to show, and a numbered <figcaption> with the values and the notes."""
    _captioned(monkeypatch, tmp_path)
    line = "{{figure: docs/results/chart.png | a chart}}"
    site, _ = PLACEHOLDERS.render(f"{line}\n\n{line}", "docs/results/page.md", VALUES, "publish")
    assert site.count('<figure class="captioned" markdown="1">') == 2
    assert ('![a chart](chart.png#only-light){ data-title="Short title" '
            'data-description=".figure-caption-1" }') in site
    assert "![a chart](chart_dark.png#only-dark)" in site
    assert ('<figcaption class="figure-caption-1" markdown="span">**Figure 1.** At '
            "100,000,000 steps. A caveat.</figcaption>") in site
    assert "**Figure 2.**" in site and "figure-caption-2" in site


def test_a_captioned_figure_in_the_readme_is_a_small_line(monkeypatch, tmp_path):
    _captioned(monkeypatch, tmp_path)
    readme, _ = PLACEHOLDERS.render("{{figure: docs/results/chart.png}}", "README.template.md",
                                    VALUES, "publish")
    picture, caption = readme.split("\n\n")
    assert picture.startswith("<picture>") and 'alt="Short title"' in picture
    assert caption == "<sub>**Figure 1.** At 100,000,000 steps. A caveat.</sub>"


def test_a_figure_without_a_drawn_title_leads_its_caption_with_it(monkeypatch, tmp_path):
    _captioned(monkeypatch, tmp_path, title_in_image=False)
    site, _ = PLACEHOLDERS.render("{{figure: docs/results/chart.png | a}}",
                                  "docs/results/page.md", VALUES, "publish")
    assert "**Figure 1. Short title.** At 100,000,000 steps." in site


def test_every_figure_on_the_pages_has_a_title_and_a_caption():
    """Every {{figure: ...}} placeholder (outside code spans) names a figure the captions
    file holds."""
    figure = re.compile(r"\{\{\s*figure:\s*([^|}]+)")
    held = PLACEHOLDERS.captions.load()
    missing = []
    for path in [*PLACEHOLDERS.site_pages(), PLACEHOLDERS.README_TEMPLATE]:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            for found in figure.finditer(PLACEHOLDERS.CODE_SPAN.sub("", line)):
                name = PLACEHOLDERS.captions.name_of(found.group(1).strip())
                if name not in held:
                    missing.append(f"{path.relative_to(pages.ROOT)}:{number}: {name}")
    assert not missing, "add these to docs/figures.yml: " + ", ".join(missing)


def test_every_caption_renders():
    """Every caption's values are in the values file."""
    for entry in PLACEHOLDERS.captions.load().values():
        PLACEHOLDERS.render(entry.caption, "docs/figures.yml")
