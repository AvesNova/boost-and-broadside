"""Render the results charts (win-rate fan, run health) in a few style options, for review.

Run with PYTHONPATH=<feat/results-charts worktree>/src: it patches that code's style
constants at run time and changes none of its defaults.

    python chart_styles.py <preview data dir> <out dir>
"""

import pathlib
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bnb.figures import output  # noqa: E402
from bnb.figures.results import common, fan, health  # noqa: E402

LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", muted="#52514e", grid="#e4e3df")
DARK = dict(surface="#1a1a19", ink="#ecebe8", muted="#a3a29e", grid="#383835")
MUTED_SET = ["#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#956cb4", "#8c613c"]  # seaborn "muted"

# letter: palette, font, grid, theme, size (web: as is at 150 dpi; page: 7 in at 300 dpi;
# column: 3.4 in at 300 dpi, panels stacked)
STYLES = {
    "a": dict(palette="viridis", font="DejaVu Sans", grid="full", theme=LIGHT, size="web"),
    "b": dict(palette="cividis", font="Nimbus Roman", grid="y", theme=LIGHT, size="page"),
    "c": dict(palette="muted", font="Lato", grid="none", theme=DARK, size="web"),
    "d": dict(palette="viridis", font="Liberation Sans", grid="y", theme=LIGHT, size="column"),
}


def apply(style: dict) -> None:
    theme = style["theme"]
    for module in (output, common, fan, health):
        for name, key in (("SURFACE", "surface"), ("INK", "ink"), ("MUTED", "muted"), ("GRID", "grid")):
            if hasattr(module, name):
                setattr(module, name, theme[key])
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams["font.family"] = style["font"]
    matplotlib.rcParams["text.color"] = theme["ink"]
    matplotlib.rcParams["axes.labelcolor"] = theme["ink"]
    matplotlib.rcParams["legend.labelcolor"] = theme["ink"]

    if style["palette"] == "muted":
        common.ordered_colours = lambda count: MUTED_SET[:count]
    else:
        common.ordered_colours = _ramp(style["palette"])

    base_style = _original["style"]

    def styled(axis):
        base_style(axis)
        if style["grid"] == "none":
            axis.grid(False)
        elif style["grid"] == "y":
            axis.grid(False)
            axis.yaxis.grid(True, color=theme["grid"], linewidth=0.5)

    output.style = styled

    size = style["size"]

    def new_figure(rows=1, columns=1, size_each=(6.0, 4.5), **options):
        size_each = options.pop("size", size_each)
        if size == "column":
            # one column of panels (the fan's three modes stacked), 3.4 in wide
            total = rows * columns
            columns_now = 1 if total <= 3 else 2
            rows_now = -(-total // columns_now)
            width = 3.4
            height_each = width / columns_now * size_each[1] / size_each[0] * (1.15 if columns_now == 1 else 1.3)
            if "sharey" in options and columns_now == 1:
                options = dict(options, sharey=False)
            return _make(rows_now, columns_now, (width, height_each * rows_now), **options)
        if size == "page":
            width = 7.0
            height = width * (size_each[1] * rows) / (size_each[0] * columns)
            return _make(rows, columns, (width, max(height, 2.2 * rows)), **options)
        return _make(rows, columns, (size_each[0] * columns, size_each[1] * rows), **options)

    common.new_figure = new_figure

    def title(figure, text):
        width = figure.get_size_inches()[0]
        figure.suptitle("\n".join(textwrap.wrap(text, int(width * 72 / 5.5))), fontsize=10 if size == "web" else 8, color=theme["ink"])

    common.title = title

    def save(figure, directory, name, sources, numbers=None, formats=common.FORMATS):
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if size != "web":
            for axis in figure.axes:
                axis.tick_params(labelsize=6)
                axis.title.set_fontsize(7.5)
                axis.xaxis.label.set_fontsize(6.5)
                axis.yaxis.label.set_fontsize(6.5)
                for text in axis.texts:
                    text.set_fontsize(min(text.get_fontsize(), 6))
            figure.tight_layout()
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=150 if size == "web" else 300, facecolor=theme["surface"])
        plt.close(figure)
        if path.stat().st_size > 400_000:  # 256 colours, under the review's size budget
            from PIL import Image

            Image.open(path).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
        return [path]

    common.save = save


def _ramp(name):
    def ordered(count):
        ramp = matplotlib.colormaps[name]
        low, high = (0.0, 0.82) if name == "viridis" else (0.0, 0.85)
        if count == 1:
            return [ramp(low)]
        return [ramp(low + (high - low) * index / (count - 1)) for index in range(count)]

    return ordered


def _make(rows, columns, figsize, **options):
    figure, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False, facecolor=output.SURFACE, **options)
    for axis in axes.flat:
        output.style(axis)
    return figure, axes


_original = dict(style=output.style)


def _register_fonts() -> None:
    """Fonts installed after matplotlib's cache was built (Lato) by their files."""
    import subprocess

    from matplotlib import font_manager

    for family in ("Lato", "Nimbus Roman", "Liberation Sans"):
        found = subprocess.run(["fc-list", f":family={family}", "file"], capture_output=True, text=True)
        for line in found.stdout.splitlines():
            path = line.split(":")[0].strip()
            if path.endswith((".ttf", ".otf")):
                font_manager.fontManager.addfont(path)


def main() -> None:
    _register_fonts()
    data, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    for letter, style in STYLES.items():
        apply(style)
        fan.draw(data / "break-even", out, name=f"chart-{letter}_fan", formats=("png",))
        columns = 2 if style["size"] == "column" else 4
        health.draw([str(data / "r-bc"), str(data / "r-rl")], out, columns=columns, name=f"chart-{letter}_health", formats=("png",))
        print(f"style {letter} done", flush=True)


if __name__ == "__main__":
    main()
