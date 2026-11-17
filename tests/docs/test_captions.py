"""The captions file (`bnb.figures.captions`, docs/figures.yml): short titles, a caption
for each figure, notes that exist."""

import pytest

from bnb.figures import captions


def test_the_captions_file_is_well_formed():
    figures = captions.load()
    assert "win_rate_fan" in figures
    for entry in figures.values():
        assert len(entry.title.split()) <= captions.TITLE_WORDS
        assert entry.caption


def test_names_drop_the_dark_suffix():
    assert captions.name_of("docs/results/win_rate_fan_dark.png") == "win_rate_fan"
    assert captions.find("docs/results/win_rate_fan.svg").name == "win_rate_fan"


def test_notes_are_appended_and_must_exist():
    held = {"notes": {"n": "A note."}, "figures": {"f": dict(title="T", caption="C.", notes=["n"])}}
    assert captions.parse(held)["f"].caption == "C. A note."
    held["figures"]["f"]["notes"] = ["other"]
    with pytest.raises(captions.CaptionError, match="no note"):
        captions.parse(held)


@pytest.mark.parametrize(
    "entry, message",
    [
        (dict(title="One two three four five six seven", caption="C."), "more than"),
        (dict(title="T {{value}}", caption="C."), "quotes no values"),
        (dict(title="T"), "needs a title and a caption"),
        (dict(title="T", caption="C.", colour="red"), "unknown keys"),
    ],
)
def test_a_bad_entry_fails(entry, message):
    with pytest.raises(captions.CaptionError, match=message):
        captions.parse({"figures": {"f": entry}})


def test_a_chart_without_an_entry_fails():
    with pytest.raises(captions.CaptionError, match="no figure"):
        captions.title("not_a_figure")
