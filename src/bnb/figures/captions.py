"""Each figure's short title and its caption, from one file, `docs/figures.yml`, that the
charts, the site, the README and the paper all read, so they never drift.

The file holds two maps:

    notes:     shared caveats by key, {key: text}, appended to every caption that names them
    figures:   per figure, by its file's stem (win_rate_fan for docs/results/win_rate_fan.png;
               the dark variant, win_rate_fan_dark.png, shares its entry):
                 title:     a few words, drawn above the chart (`title_in_image`) and the
                            lightbox's title
                 caption:   what a reader needs to read the figure correctly: what the
                            marks are, the sample, the conditions, which weights, the caveats;
                            may quote the values file as {{name}} or {{name:format}}
                 notes:     keys of `notes` to append to the caption (optional)
                 title_in_image: false for a figure that draws no title of its own (a still,
                            a clip, a diagram): the caption then leads with the title
                            (optional, default true)

The web PNGs draw the title (`bnb.figures.results.common.title`); the paper's PDF and SVG
draw none, the paper's figure environment carrying the title and the caption. The site and
the README put the caption under the figure (`docs/hooks/placeholders.py`).
"""

import dataclasses
import functools
import pathlib
import re

FIGURES = pathlib.Path(__file__).resolve().parents[3] / "docs" / "figures.yml"
DARK = "_dark"
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
TITLE_WORDS = 6  # a title is a few words


class CaptionError(ValueError):
    """A malformed captions file, or a figure it does not hold."""


@dataclasses.dataclass(frozen=True)
class Figure:
    """One figure's title and caption (the caption's notes appended)."""

    name: str
    title: str
    caption: str
    title_in_image: bool = True


def name_of(path: str | pathlib.Path) -> str:
    """A figure's name: its file's stem without the dark variant's suffix."""
    stem = pathlib.PurePosixPath(str(path)).stem
    return stem.removesuffix(DARK)


def parse(held: dict, source: str = "figures.yml") -> dict[str, Figure]:
    """The figures of a captions file's parsed content, by name; raises CaptionError on a
    malformed entry or an unknown note."""
    if not isinstance(held, dict) or not isinstance(held.get("figures"), dict):
        raise CaptionError(f"{source}: a map with a 'figures' map (and an optional 'notes')")
    notes = held.get("notes") or {}
    figures = {}
    for name, entry in held["figures"].items():
        if not NAME.match(str(name)):
            raise CaptionError(f"{source}: {name!r} is not a figure name (a file's stem)")
        if not isinstance(entry, dict) or not entry.get("title") or not entry.get("caption"):
            raise CaptionError(f"{source}: {name} needs a title and a caption")
        unknown = set(entry) - {"title", "caption", "notes", "title_in_image"}
        if unknown:
            raise CaptionError(f"{source}: {name}: unknown keys {sorted(unknown)}")
        title = " ".join(str(entry["title"]).split())
        if "{{" in title:
            raise CaptionError(f"{source}: {name}: a title quotes no values")
        if len(title.split()) > TITLE_WORDS:
            raise CaptionError(
                f"{source}: {name}: the title {title!r} is more than "
                f"{TITLE_WORDS} words; the details go in the caption"
            )
        parts = [" ".join(str(entry["caption"]).split())]
        for key in entry.get("notes") or []:
            if key not in notes:
                raise CaptionError(f"{source}: {name}: no note {key!r}")
            parts.append(" ".join(str(notes[key]).split()))
        figures[name] = Figure(
            name, title, " ".join(parts), bool(entry.get("title_in_image", True))
        )
    return figures


def load(path: pathlib.Path = FIGURES) -> dict[str, Figure]:
    """Every figure of the captions file, by name (empty when there is no file); read
    again when the file changes (a docs server keeps running)."""
    if not path.exists():
        return {}
    return _parsed(path, path.stat().st_mtime_ns)


@functools.cache
def _parsed(path: pathlib.Path, modified: int) -> dict[str, Figure]:
    import yaml

    return parse(yaml.safe_load(path.read_text()), str(path))


def find(path_or_name: str | pathlib.Path) -> Figure | None:
    """A figure's entry by its path or name, or None."""
    return load().get(name_of(path_or_name))


def title(name: str) -> str:
    """A figure's short title; raises CaptionError when the captions file has none."""
    entry = find(name)
    if entry is None:
        raise CaptionError(f"{FIGURES}: no figure {name!r}; add its title and caption there")
    return entry.title
