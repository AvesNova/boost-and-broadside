"""Render the review trail stills from the recorded matches (scratch_records/*.pt).

    python trail_stills.py <records dir> <review dir>

Part 1: review/stills/new-imugi/ (the house style). Part 2: review/style-options/ (one duel
window and one 5v5 window in styles a-d, patched at run time; the defaults are unchanged).
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections  # noqa: E402
import numpy as np  # noqa: E402

from bnb.figures import capture, trails  # noqa: E402

ORIGINAL = dict(
    ship_colours=trails.ship_colours,
    OLDEST_ALPHA=trails.OLDEST_ALPHA,
    TEAM_NAMES=trails.TEAM_NAMES,
    LineCollection=matplotlib.collections.LineCollection,
    SIZES=dict(trails.SIZES),
)


def tick_length(artifact) -> float:
    return float(artifact["match"]["about"]["tick_length"])


def game_seconds(artifact) -> float:
    return len(artifact["match"]["clock"]) * tick_length(artifact)


def busiest(artifact, seconds: float):
    return trails.busiest_window(artifact["match"], tick_length(artifact), seconds)


def counts(artifact, start, end):
    """Deaths per team and in total in a window."""
    match = artifact["match"]
    step = tick_length(artifact)
    first, final = round((start or 0) / step), min(len(match["clock"]) - 1, round((end or 1e9) / step))
    died = match["died"].numpy()[first : final + 1]
    team = match["team"].numpy()
    return [int(died[:, team == side].sum()) for side in (0, 1)]


# ---- part 2 styles -------------------------------------------------------------------------

OKABE = {0: "#0072b2", 1: "#d55e00"}  # Okabe-Ito blue and vermillion
DASHES = [None, (5.0, 1.6), (1.2, 1.4), (6.0, 1.6, 1.2, 1.6), (2.6, 1.6)]


def _unique(hex_colour: str, index: int) -> str:
    """The colour nudged by `index` in its blue channel's last bit (a key, invisible)."""
    value = int(hex_colour[1:], 16)
    return f"#{(value & ~0x7) | index:06x}"


def _dashed_collection(dash_of: dict):
    base = ORIGINAL["LineCollection"]

    class Dashed(base):
        """Trail segments regrouped into polylines (so a dash pattern runs along them), each
        chunk at its segments' mean colour, dashed by its ship (keyed by colour)."""

        def __init__(self, segments, colors=None, **options):
            segments = np.asarray(segments)
            rgba = np.asarray(matplotlib.colors.to_rgba_array(colors)) if colors is not None else None
            if rgba is None or len(segments) < 2 or len(rgba) != len(segments) or segments.shape[1] != 2:
                super().__init__(segments, colors=colors, **options)
                return
            key = matplotlib.colors.to_hex(rgba[0][:3])
            chunk = max(8, len(segments) // 40)
            lines, shades = [], []
            for start in range(0, len(segments), chunk):
                part = segments[start : start + chunk]
                lines.append(np.concatenate([part[:, 0], part[-1:, 1]]))
                shades.append(rgba[start : start + chunk].mean(0))
            dash = dash_of.get(key)
            if dash is not None:
                options["linestyles"] = [(0, dash)] * len(lines)
            options["capstyle"] = "butt"
            super().__init__(lines, colors=shades, **options)

    return Dashed


def reset() -> None:
    trails.ship_colours = ORIGINAL["ship_colours"]
    trails.OLDEST_ALPHA = ORIGINAL["OLDEST_ALPHA"]
    trails.TEAM_NAMES = ORIGINAL["TEAM_NAMES"]
    trails.SIZES.clear()
    trails.SIZES.update(ORIGINAL["SIZES"])
    matplotlib.collections.LineCollection = ORIGINAL["LineCollection"]
    trails.THEMES.pop("night", None)
    trails.THEMES.pop("paper", None)


def style_b() -> str:
    """Two colour-blind-safe hues, one per team; each ship of a team its own dash."""
    dash_of = {}

    def colours(team, theme):
        out = []
        for ship, side in enumerate(team.tolist()):
            slot = int((team[:ship] == side).sum())
            colour = _unique(OKABE[int(side)], slot)
            dash_of[colour] = DASHES[slot % len(DASHES)]
            out.append(colour)
        return out

    trails.ship_colours = colours
    trails.TEAM_NAMES = ("blue team", "vermillion team")
    trails.SIZES["web"] = (6.4, 160, 1.15, 8.0)
    matplotlib.collections.LineCollection = _dashed_collection(dash_of)
    return "light"


def style_c() -> str:
    """Night: a near-black blue surface, the dark ramps, thicker lines, a longer fade."""
    trails.THEMES["night"] = dict(
        surface="#0d1117", ink="#e6edf3", muted="#8b949e", grid="#30363d", field="#6e7681"
    )
    base = ORIGINAL["ship_colours"]
    trails.ship_colours = lambda team, theme: base(team, "dark")
    trails.OLDEST_ALPHA = 0.25
    trails.SIZES["web"] = (6.4, 160, 1.35, 8.0)
    return "night"


def style_d() -> str:
    """Paper: white, team 0 in greys to black, team 1 one accent (vermillion) in two
    lightnesses; thin lines, a short fade."""
    trails.THEMES["paper"] = dict(
        surface="#ffffff", ink="#111111", muted="#666666", grid="#dddddd", field="#9a9a9a"
    )
    greys = ("#9a9a9a", "#7a7a7a", "#555555", "#333333", "#111111")
    accents = ("#f08a5d", "#e66a2c", "#d55e00", "#b34d00", "#8f3d00")

    def colours(team, theme):
        out = []
        for ship, side in enumerate(team.tolist()):
            ramp = greys if side == 0 else accents
            slots = int((team == side).sum())
            slot = int((team[:ship] == side).sum())
            place = 0.6 if slots == 1 else 0.2 + 0.8 * slot / (slots - 1)
            out.append(ramp[round(place * (len(ramp) - 1))])
        return out

    trails.ship_colours = colours
    trails.TEAM_NAMES = ("grey team", "accent team")
    trails.OLDEST_ALPHA = 0.35
    trails.SIZES["web"] = (6.4, 160, 0.75, 8.0)
    return "paper"


STYLES = {"a": lambda: "light", "b": style_b, "c": style_c, "d": style_d}


def shrink(path: pathlib.Path, limit: int = 400_000) -> None:
    """Quantise a PNG over `limit` bytes to 256 colours, keeping its text chunks."""
    if path.stat().st_size <= limit:
        return
    from PIL import Image, PngImagePlugin

    image = Image.open(path)
    info = PngImagePlugin.PngInfo()
    for key, value in image.text.items():
        info.add_text(key, value)
    image.convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(path, pnginfo=info, optimize=True)


def still(artifact, directory, name, **config):
    paths, numbers = trails.still(artifact, directory, name, trails.StillConfig(**config))
    for path in paths:
        shrink(path)
    return numbers


def main() -> None:
    records, review = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    only = sys.argv[3] if len(sys.argv) > 3 else "all"
    load = {path.stem: capture.load(path) for path in sorted(records.glob("*.pt"))}
    listing = []
    new = review / "stills" / "new-imugi"

    def add(key, name, start, end, what):
        artifact = load[key]
        numbers = still(artifact, new, name, start=start, end=end)
        shown = numbers["window_seconds"]
        listing.append(dict(file=f"{name}.png", record=key, window=shown, deaths=numbers["deaths"], what=what))
        print(name, shown, numbers["deaths"], flush=True)

    if only in ("all", "part1"):
        for key in ("dm1v1_s0", "dm1v1_s1"):
            if key in load:
                add(key, f"{key}_game", None, None, "whole game")
                start, end = busiest(load[key], 35.0)
                add(key, f"{key}_window", start, end, "busiest 35 s")
        if "dm2v2_s0" in load:
            start, end = busiest(load["dm2v2_s0"], 50.0)
            add("dm2v2_s0", "dm2v2_s0_window", start, end, "busiest 50 s")
        if "dm5v5_s0" in load:
            add("dm5v5_s0", "dm5v5_s0_opening", None, 60.0, "opening 60 s")
            start, end = busiest(load["dm5v5_s0"], 60.0)
            add("dm5v5_s0", "dm5v5_s0_busiest", start, end, "busiest 60 s")
        if "elim5v5_s0" in load:
            start, end = busiest(load["elim5v5_s0"], 50.0)
            add("elim5v5_s0", "elim5v5_s0_window", start, end, "busiest 50 s, fog on, trails only")
        (new / "listing.json").write_text(json.dumps(listing, indent=1) + "\n")

    if only in ("all", "part2"):
        options = review / "style-options"
        windows = {}
        if "dm1v1_s0" in load:
            windows["duel"] = ("dm1v1_s0", *busiest(load["dm1v1_s0"], 35.0))
        if "dm5v5_s0" in load:
            windows["5v5"] = ("dm5v5_s0", *busiest(load["dm5v5_s0"], 40.0))
        for letter, apply in STYLES.items():
            reset()
            theme = apply()
            for label, (key, start, end) in windows.items():
                still(load[key], options, f"style-{letter}_{label}", start=start, end=end, theme=theme)
                print(f"style-{letter}_{label}", flush=True)
        reset()
        (options / "trail-windows.json").write_text(json.dumps(windows, indent=1) + "\n")


if __name__ == "__main__":
    main()
