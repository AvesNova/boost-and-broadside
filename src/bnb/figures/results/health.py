"""Run health and the auxiliary heads over training: small multiples against environment
steps, one panel per quantity (`PANELS`: PPO's clip fraction, approximate KL, explained
variance per reward part, entropy per action factor, value loss per reward part, BC's KL to
its teacher) and one per auxiliary head's loss (every `<head>/loss` and `<head>/loss/<part>`
key). A panel whose keys a run never logged is left out.

**Sources** (`read_source`): a run directory or its `metrics.jsonl` (the local sink,
`bnb.runs.sinks`), or a W&B run path `wandb:<entity>/<project>/<id>`, read only through
`run.history(samples=N, keys=[...])` (one sampled request per key, never scan_history).
Several sources are a chain (BC then RL): each one's steps start after the previous one's
last, the hand-off marked. Each line is drawn raw (faint) and as a centred moving mean.
"""

import pathlib
import re

from bnb.runs.sinks import METRICS, read_metrics

from .. import theme
from . import common

STEPS = "counters/environment_steps"
WANDB = "wandb:"
# (title, the pattern of its keys); the label of a line is its key's last part.
PANELS = (
    ("PPO clip fraction", r"^policy/clip_fraction$"),
    ("approximate KL (policy update)", r"^policy/kl$"),
    ("explained variance", r"^critic/explained_variance/[^/]+$"),
    ("entropy (nats)", r"^policy/entropy/[^/]+$"),
    ("value loss", r"^critic/value_loss/[^/]+$"),
    ("BC: KL to the teacher", r"^bc/kl$"),
)
HEAD_LOSS = re.compile(r"^(?!critic/|policy/|bc/|episode/)([a-z_]+)/loss(/[^/]+)?$")


def panels_for(keys) -> list[tuple[str, list[str]]]:
    """The panels the keys fill, in order: `PANELS`, then one per auxiliary head."""
    keys = sorted(keys)
    found = []
    for heading, pattern in PANELS:
        mine = [key for key in keys if re.match(pattern, key)]
        if mine:
            found.append((heading, mine))
    heads: dict[str, list[str]] = {}
    for key in keys:
        matched = HEAD_LOSS.match(key)
        if matched:
            heads.setdefault(matched.group(1), []).append(key)
    found += [(f"{head.replace('_', ' ')} head loss", mine) for head, mine in heads.items()]
    return found


def _numeric(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value == value


def read_source(source: str, samples: int = 1000) -> dict[str, list[tuple[float, float]]]:
    """{key: [(steps, value), ...] in order of steps} for every key a panel draws."""
    if source.startswith(WANDB):
        import wandb

        run = wandb.Api().run(source.removeprefix(WANDB))
        wanted = [key for _, keys in panels_for(run.summary.keys()) for key in keys]
        found = {}
        for key in wanted:
            rows = run.history(samples=samples, keys=[STEPS, key], pandas=False)
            found[key] = sorted(
                (float(row[STEPS]), float(row[key]))
                for row in rows
                if _numeric(row.get(STEPS)) and _numeric(row.get(key))
            )
        return found
    path = pathlib.Path(source)
    rows = read_metrics(path / METRICS if path.is_dir() else path)
    keys = {key for row in rows for key, value in row.items() if _numeric(value)}
    wanted = [key for _, mine in panels_for(keys) for key in mine]
    return {
        key: sorted(
            (float(row[STEPS]), float(row[key]))
            for row in rows
            if _numeric(row.get(STEPS)) and _numeric(row.get(key))
        )
        for key in wanted
    }


def moving_mean(values: list[float], window: int) -> list[float]:
    """The centred moving mean over `window` points (shorter at the ends)."""
    half = window // 2
    out = []
    for index in range(len(values)):
        part = values[max(0, index - half) : index + half + 1]
        out.append(sum(part) / len(part))
    return out


def draw(
    sources: list[str],
    out: str | pathlib.Path,
    samples: int = 1000,
    columns: int = 4,
    formats: tuple[str, ...] = common.FORMATS,
    name: str = "run_health",
) -> tuple[list[pathlib.Path], dict]:
    """Draw the small multiples of `sources` (a chain, in order) into `out`; (the files
    written, the values: each key's mean over the last twentieth of its last source)."""
    read = [read_source(source, samples) for source in sources]
    offsets, offset = [], 0.0
    for found in read:
        offsets.append(offset)
        offset += max((points[-1][0] for points in found.values() if points), default=0.0)
    keys = {key for found in read for key, points in found.items() if points}
    panels = panels_for(keys)
    if not panels:
        raise SystemExit(f"no health or head-loss keys in {sources}")
    rows = -(-len(panels) // columns)
    figure, axes = common.new_figure(rows, columns, size=(3.3, 2.5))
    values: dict[str, float] = {}
    for axis, (heading, mine) in zip(axes.flat, panels, strict=False):
        axis.set_title(heading, fontsize=9)
        colours = (
            [common.ordered_colours(1)[0]] if len(mine) == 1 else common.ordered_colours(len(mine))
        )
        labelled = []
        for key, colour in zip(mine, colours, strict=True):
            label = key.rsplit("/", 1)[-1] if len(mine) > 1 else None
            xs_all, ys_all = [], []
            for found, start in zip(read, offsets, strict=True):
                points = found.get(key, [])
                if not points:
                    continue
                xs = [(start + x) / 1e6 for x, _ in points]
                ys = [y for _, y in points]
                axis.plot(xs, ys, color=colour, linewidth=0.6, alpha=0.3)
                axis.plot(
                    xs,
                    moving_mean(ys, max(1, len(ys) // 30)),
                    color=colour,
                    linewidth=1.4,
                    label=label if not xs_all else None,
                )
                xs_all += xs
                ys_all += ys
                tail = ys[-max(1, len(ys) // 20) :]
                values[f"health_{common.slug(key)}_final"] = sum(tail) / len(tail)
            if label and xs_all:
                labelled.append(
                    (label, xs_all, moving_mean(ys_all, max(1, len(ys_all) // 30)), colour)
                )
        for start in offsets[1:]:
            axis.axvline(start / 1e6, color=theme.MUTED, linewidth=0.8, linestyle="--")
        common.label_ends(axis, labelled)
    drawn = list(axes.flat[: len(panels)])
    right = max(axis.get_xlim()[1] for axis in drawn)  # one x range: the labelled panels'
    for axis in drawn:
        axis.set_xlim(min(axis.get_xlim()[0] for axis in drawn), right)
    for axis in axes.flat[len(panels) :]:
        axis.set_visible(False)
    for index in range(len(panels)):
        if index >= len(panels) - columns:
            axes.flat[index].set_xlabel("environment steps (M)", fontsize=8)
    common.title(figure, "Run health and auxiliary heads over training")
    figure.tight_layout(rect=(0, 0, 1, 1))
    written = common.save(figure, out, name, list(sources), values, formats)
    return written, values
