"""The results placeholders (`docs/hooks/placeholders.py`): every value a page quotes is in
the values file, every placeholder is well formed, and README.md is the rendered
README.template.md. In publish mode (BNB_RESULTS=publish) nothing may be open; in draft
mode, the default, what is open is listed as a warning and the tests pass."""

import importlib.util
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
