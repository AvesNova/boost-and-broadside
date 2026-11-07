"""Merge the baselines bench's per-segment results (`bench/baselines_measure.py`) and print
the markdown tables of docs/rewrite/sections/baselines-bench.md (`--part main`) or the full
tables (`--part full`, docs/rewrite/sections/baselines-bench-tables.md).

Pooling: every player and preset at equal weight (each group's histograms normalised by
its pairs). Quantiles are read from the signed log histograms (100 bins per decade, the
exact zeros and the max kept exactly; interpolated geometrically within a bin). The
entropy is that of the mean code: one symlog HL-Gauss code per series (quantity /
baseline / component; `bnb.models.yemong.codes.symlog_encode`), knee at the pooled median
|residual| (the median of the nonzero part where the median is 0), range at the pooled max,
51 real bins for position and velocity components and 31 otherwise, applied to every
group and subset alike.

Run: uv run --no-sync python -m bench.baselines_report --part main
"""

import argparse
import json
import math
import pathlib

import numpy as np
import torch

from bnb.models.yemong import codes
from bnb.models.yemong.delta import COMPONENTS as DELTA_COMPONENTS

from .baselines_measure import BINS, LOWEST, OUT, PER_DECADE, PLAYERS, PRESETS

GROUPS = tuple((player, preset) for player in PLAYERS for preset in PRESETS)
CENTRES = 10.0 ** (LOWEST + (np.arange(BINS) + 0.5) / PER_DECADE)
SIGNED_CENTRES = np.concatenate([[0.0], CENTRES, -CENTRES])
EDGES = 10.0 ** (LOWEST + np.arange(BINS + 1) / PER_DECADE)
DELTA = {c.name: c for c in DELTA_COMPONENTS}
EVENT_THRESHOLD = dict(shield_recharge_wait=0.5 / 30.0, gun_cooldown=0.5)  # s; ticks
NOISE = 1e-6  # |residual| at or below: float rounding, for the knee where the median is 0
QUANTITIES = (
    "position",
    "velocity",
    "heading",
    "nose_rate",
    "angle_of_attack",
    "energy",
    "shield",
    "shield_recharge_wait",
    "gun_cooldown",
    "log_refractive_index",
)
UNITS = dict(
    position="m", velocity="m/s", heading="rad", nose_rate="rad/s", angle_of_attack="rad",
    energy="of max", shield="of max", shield_recharge_wait="s", gun_cooldown="ticks",
    log_refractive_index="ln n",
)  # fmt: skip


def g(value) -> str:
    """A number in three figures (scientific when tiny or large)."""
    if value is None or value != value:
        return "-"
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3g}"


def pct(value) -> str:
    """A share as a percentage."""
    if value != value:
        return "-"
    return f"{100 * value:.1f}%" if value >= 0.001 or value == 0 else f"{100 * value:.2g}%"


class Merged:
    """The merged results of every group."""

    def __init__(self, root: pathlib.Path) -> None:
        self.groups = {}
        for group in GROUPS:
            directory = root / group[0] / group[1]
            paths = sorted(directory.glob("segment_*.npz"))
            if not paths:
                continue
            entry = dict(samples={})
            for path in paths:
                data = np.load(path)
                index = json.loads(path.with_suffix(".json").read_text())
                for key in ("histograms", "wins", "regression", "moments"):
                    entry[key] = entry.get(key, 0) + data[key]
                entry["maxima"] = np.maximum(entry.get("maxima", 0), data["maxima"])
                counts = entry.setdefault("counts", {})
                for name, value in index["counts"].items():
                    counts[name] = counts.get(name, 0) + value
                for key in data.files:
                    if key.startswith("sample:"):
                        entry["samples"].setdefault(key[7:], []).append(data[key])
            entry["samples"] = {k: np.concatenate(v) for k, v in entry["samples"].items()}
            self.groups[group] = entry
        self.index = index
        self.names = index["names"]
        self.subsets = index["subsets"]
        self.series = {name: i for i, name in enumerate(self.names)}
        self.subset = {name: i for i, name in enumerate(self.subsets)}
        self.comparisons = {tuple(c): i for i, c in enumerate(index["comparisons"])}
        self.codes = {}
        for name in self.names:
            self.codes[name] = self._code(name)

    # ---------------------------------------------------------------- histograms

    def histogram(self, name, subset="all", group=None):
        """(counts (2 BINS + 1,) float, max): one group's, or pooled at equal weight."""
        i, j = self.series[name], self.subset[subset]
        if group is not None:
            entry = self.groups[group]
            return entry["histograms"][i, j].astype(float), float(entry["maxima"][i, j])
        total, top = np.zeros(2 * BINS + 1), 0.0
        for entry in self.groups.values():
            total += entry["histograms"][i, j] / max(entry["counts"]["pairs"], 1)
            top = max(top, float(entry["maxima"][i, j]))
        return total, top

    @staticmethod
    def quantile(counts, top, q):
        """The q-quantile (0-1) of |residual|."""
        absolute = np.concatenate([[counts[0]], counts[1 : BINS + 1] + counts[BINS + 1 :]])
        total = absolute.sum()
        if total == 0:
            return float("nan")
        target = q * total
        cumulative = np.cumsum(absolute)
        k = int(np.searchsorted(cumulative, target, side="left"))
        if k == 0:
            return 0.0
        below = cumulative[k - 1]
        fraction = (target - below) / max(absolute[k], 1e-300)
        low, high = EDGES[k - 1], EDGES[k]
        return float(min(low * (high / low) ** fraction, top))

    @staticmethod
    def nonzero_median(counts, top):
        """The median of |residual| above NOISE (float32 rounding of the recorded state)."""
        rest = counts.copy()
        rest[0] = 0
        noise = np.concatenate([EDGES[1:], EDGES[1:]]) <= NOISE
        rest[1:][noise] = 0
        return Merged.quantile(rest, top, 0.5)

    def _code(self, name):
        """(knee, bound, bins) of a series' code, from the pooled distribution."""
        counts, top = self.histogram(name)
        knee = self.quantile(counts, top, 0.5)
        if not knee > 0:
            knee = self.nonzero_median(counts, top)
        quantity = name.split("/")[0]
        bins = 51 if quantity in ("position", "velocity") else 31
        if not knee > 0 or not top > 0:
            return None
        return knee, max(top, knee * 1.000001), bins

    def entropy(self, name, subset="all", group=None, code=None):
        """(H, H_self) in nats: the entropy of the mean code of the series in a subset, and
        the mean entropy of one sample's code (the HL-Gauss smoothing's own floor). code:
        (knee, bound, bins), the series' own by default."""
        code = code or self.codes[name]
        if name.endswith("/norm"):
            return float("nan"), float("nan")
        counts, _ = self.histogram(name, subset, group)
        if counts.sum() == 0:
            return float("nan"), float("nan")
        if code is None:
            return 0.0, 0.0
        knee, bound, bins = code
        used = counts > 0
        values = torch.tensor(SIGNED_CENTRES[used], dtype=torch.float64)
        encoded, _ = codes.symlog_encode(values, knee, bound, bins)
        weights = torch.tensor(counts[used] / counts.sum(), dtype=torch.float64)
        p = (weights[:, None] * encoded).sum(0)
        p = p[p > 0]
        own = -(encoded * torch.log(encoded.clamp_min(1e-300))).sum(-1)
        return float(-(p * p.log()).sum()), float((weights * own).sum())

    def event_entropy(self, name, threshold, subset="all", group=None):
        """The binary entropy (nats) of |residual| > threshold: the two-bin event code."""
        counts, top = self.histogram(name, subset, group)
        absolute = np.concatenate([[counts[0]], counts[1 : BINS + 1] + counts[BINS + 1 :]])
        total = absolute.sum()
        if total == 0:
            return float("nan")
        above = absolute[1:][CENTRES > threshold].sum() / total
        if above <= 0 or above >= 1:
            return 0.0
        return float(-(above * math.log(above) + (1 - above) * math.log(1 - above)))

    def delta_entropy(self, name, subset="all", group=None):
        """(H, H_self) under today's delta code (`delta.COMPONENTS`; events: two bins)."""
        quantity, _, component = name.split("/")
        if component == "norm":
            return float("nan"), float("nan")
        if quantity in EVENT_THRESHOLD:
            return self.event_entropy(name, EVENT_THRESHOLD[quantity], subset, group), 0.0
        key = (
            quantity
            if quantity not in ("position", "velocity")
            else (
                f"{quantity}_{'along' if component.endswith('along') or component == 'x' else 'across'}"  # noqa: E501
            )
        )
        c = DELTA[key]
        return self.entropy(name, subset, group, (c.knee, c.bound, c.bins))

    def stats(self, name, subset="all", group=None):
        """The series' statistics in a subset."""
        counts, top = self.histogram(name, subset, group)
        total = counts.sum()
        entropy = self.entropy(name, subset, group)
        delta = self.delta_entropy(name, subset, group)
        return dict(
            n=total,
            p50=self.quantile(counts, top, 0.5),
            p90=self.quantile(counts, top, 0.9),
            p99=self.quantile(counts, top, 0.99),
            p999=self.quantile(counts, top, 0.999),
            max=top,
            zero=counts[0] / total if total else float("nan"),
            small=self.below(counts, 1e-6),
            H=entropy[0],
            excess=entropy[0] - entropy[1],
            H_delta=delta[0],
            excess_delta=delta[0] - delta[1],
        )

    @staticmethod
    def below(counts, threshold):
        """The share with |residual| < threshold (exact zeros included)."""
        absolute = counts[1 : BINS + 1] + counts[BINS + 1 :]
        return float((counts[0] + absolute[EDGES[1:] <= threshold].sum()) / counts.sum())

    # ---------------------------------------------------------------- other merges

    def wins(self, quantity, component, first, second, subset="all"):
        """(share |first| < |second|, share |first| > |second|), pooled at equal weight."""
        i, j = self.comparisons[(quantity, component, first, second)], self.subset[subset]
        better = worse = 0.0
        for entry in self.groups.values():
            n = entry["counts"][f"subset_{subset}"]
            if n:
                better += entry["wins"][i, j, 0] / n / len(self.groups)
                worse += entry["wins"][i, j, 1] / n / len(self.groups)
        return better, worse

    def regression(self, quantity, subset="all", group=None):
        """(beta, the change's R^2 under the best beta, under beta = 1 (E)) of the change on
        t against last tick's: E assumes beta = 1, P beta = 0."""
        i, j = self.index["regressed"].index(quantity), self.subset[subset]
        groups = [self.groups[group]] if group else list(self.groups.values())
        cross = square = total = 0.0
        for entry in groups:
            a, b, c, n = entry["regression"][i, j]
            if n:
                cross, square, total = cross + a / n, square + b / n, total + c / n
        if square == 0 or total == 0:
            return float("nan"), float("nan"), float("nan")
        beta = cross / square
        best = 1.0 - (total - 2 * beta * cross + beta * beta * square) / total
        unit = 1.0 - (total - 2 * cross + square) / total
        return beta, best, unit

    def ellipse(self, k, forecast, quantity, frame, group=None):
        """Pearson, axis ratio sqrt(l1 / l2) and the major axis angle (deg) from the
        streaming moments (equal weight per group), and from the subsample: Spearman, and
        the same trimmed at the p99 of the norm; the median |component|."""
        h = self.index["horizons"].index(k)
        f = self.index["forecasts"].index(forecast)
        q = ("position", "velocity").index(quantity)
        r = self.index["frames"].index(frame)
        groups = [self.groups[group]] if group else list(self.groups.values())
        mean = np.zeros(5)
        for entry in groups:
            n, *sums = entry["moments"][h, f, q, r]
            mean += np.asarray(sums) / n / len(groups)
        ma, mb, maa, mbb, mab = mean
        covariance = np.array([[maa - ma * ma, mab - ma * mb], [mab - ma * mb, mbb - mb * mb]])
        sample = np.concatenate(
            [entry["samples"][f"{k}/{forecast}/{quantity}/{frame}"] for entry in groups]
        ).astype(np.float64)
        norm = np.hypot(sample[:, 0], sample[:, 1])
        trimmed = sample[norm <= np.percentile(norm, 99)]
        return dict(
            pearson=_pearson(covariance),
            **_axes(covariance),
            spearman=_spearman(sample),
            trimmed_pearson=_pearson(np.cov(trimmed.T)),
            **{f"trimmed_{k_}": v for k_, v in _axes(np.cov(trimmed.T)).items()},
            median_a=float(np.median(np.abs(sample[:, 0]))),
            median_b=float(np.median(np.abs(sample[:, 1]))),
            p99_norm=float(np.percentile(norm, 99)),
        )


def _pearson(covariance):
    return float(covariance[0, 1] / math.sqrt(covariance[0, 0] * covariance[1, 1]))


def _axes(covariance):
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, 1]
    angle = math.degrees(math.atan2(major[1], major[0]))
    angle = (angle + 90.0) % 180.0 - 90.0
    return dict(ratio=float(math.sqrt(values[1] / max(values[0], 1e-300))), angle=angle)


def _spearman(sample):
    ranks = np.argsort(np.argsort(sample, axis=0), axis=0).astype(np.float64)
    return float(np.corrcoef(ranks.T)[0, 1])


# -------------------------------------------------------------------- tables

COMPONENTS = dict(
    position=("x", "y", "nose_along", "nose_across", "track_along", "track_across", "norm"),
    velocity=("x", "y", "nose_along", "nose_across", "track_along", "track_across", "norm"),
)
BASELINES = dict(position=("P", "R", "E", "E-arc"))


def baselines_of(quantity):
    return BASELINES.get(quantity, ("P", "R", "E"))


def components_of(quantity):
    return COMPONENTS.get(quantity, ("value",))


def table_headline(m: Merged, components=None) -> list[str]:
    lines = [
        "H: the entropy of the mean code (own code: knee at this baseline's median, range at "
        "its max); H - H_self: less the code's own smoothing floor. Today's code: "
        "`delta.COMPONENTS` (two-bin events for the wait and the cooldown).",
        "",
        "| quantity | component | baseline | median | p90 | p99 | p99.9 | max | exactly 0 | below 1e-6 | H own | H - H_self own | H today's code | H - H_self today's | own code: knee, range, bins |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for quantity in QUANTITIES:
        for component in components or components_of(quantity):
            if component not in components_of(quantity):
                continue
            for baseline in baselines_of(quantity):
                name = f"{quantity}/{baseline}/{component}"
                s = m.stats(name)
                code = m.codes[name]
                code_text = (
                    "-"
                    if code is None or component == "norm"
                    else (f"{g(code[0])}, {g(code[1])}, {code[2]}")
                )
                lines.append(
                    f"| {quantity} ({UNITS[quantity]}) | {component} | {baseline} | {g(s['p50'])} "
                    f"| {g(s['p90'])} | {g(s['p99'])} | {g(s['p999'])} | {g(s['max'])} "
                    f"| {pct(s['zero'])} | {pct(s['small'])} | {g(s['H'])} | {g(s['excess'])} "
                    f"| {g(s['H_delta'])} | {g(s['excess_delta'])} | {code_text} |"
                )
    return lines


def table_groups(m: Merged, picks) -> list[str]:
    header = "| series | " + " | ".join(f"{p} {r}" for p, r in m.groups) + " |"
    lines = [
        "Each cell: median / p99.9 / H - H_self (nats, the pooled own code).",
        "",
        header,
        "|---|" + "---|" * len(m.groups),
    ]
    for name in picks:
        cells = []
        for group in m.groups:
            s = m.stats(name, group=group)
            cells.append(f"{g(s['p50'])} / {g(s['p999'])} / {g(s['excess'])}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def table_splits(m: Merged, picks, subsets) -> list[str]:
    lines = [
        "Each cell: median / p99.9 / H - H_self (nats, the pooled own code). Share: the "
        "subset's share of the pairs (pooled).",
        "",
        "| series | " + " | ".join(subsets) + " |",
        "|---|" + "---|" * len(subsets),
    ]
    shares = []
    for subset in subsets:
        share = np.mean([e["counts"][f"subset_{subset}"] / e["counts"]["pairs"]
                         for e in m.groups.values()])  # fmt: skip
        shares.append(pct(share))
    lines.append("| share | " + " | ".join(shares) + " |")
    for name in picks:
        cells = []
        for subset in subsets:
            s = m.stats(name, subset)
            cells.append(f"{g(s['p50'])} / {g(s['p999'])} / {g(s['excess'])}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def table_wins(m: Merged, picks, subsets) -> list[str]:
    lines = [
        "Each cell: share of pairs where the first leaves a strictly smaller |residual| / "
        "strictly larger (the rest are ties).",
        "",
        "| quantity | component | first vs second | " + " | ".join(subsets) + " |",
        "|---|---|---|" + "---|" * len(subsets),
    ]
    for quantity, component, first, second in picks:
        cells = []
        for subset in subsets:
            better, worse = m.wins(quantity, component, first, second, subset)
            cells.append(f"{pct(better)} / {pct(worse)}")
        lines.append(
            f"| {quantity} | {component} | {first} vs {second} | " + " | ".join(cells) + " |"
        )
    return lines


def table_regression(m: Merged, subsets) -> list[str]:
    lines = [
        "beta: least squares of the change on t on last tick's change (E assumes 1, P 0); "
        "then the share of the change's mean square removed by the best beta and by beta = 1 "
        "(E; negative: E adds error).",
        "",
        "| quantity | " + " | ".join(subsets) + " |",
        "|---|" + "---|" * len(subsets),
    ]
    for quantity in m.index["regressed"]:
        cells = []
        for subset in subsets:
            beta, best, unit = m.regression(quantity, subset)
            cells.append(f"{g(beta)}; {g(best)}, {g(unit)}")
        lines.append(f"| {quantity} | " + " | ".join(cells) + " |")
    return lines


def table_counts(m: Merged) -> list[str]:
    lines = [
        "| player | preset | one-tick pairs | E falls back (t - 1 missing) | velocity frame falls back (< 1 m/s) | held | changed | power changed | stick changed | in a field band | hit |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (player, preset), e in m.groups.items():
        c = e["counts"]
        n = c["pairs"]
        lines.append(
            f"| {player} | {preset} | {n:,} | {c['fallback']:,} ({pct(c['fallback'] / n)}) "
            f"| {c['track_fallback']:,} | {pct(c['subset_held'] / n)} | {pct(c['subset_changed'] / n)} "  # noqa: E501
            f"| {pct(c['subset_power_changed'] / n)} | {pct(c['subset_stick_changed'] / n)} "
            f"| {pct(c['subset_field'] / n)} | {pct(c['subset_hit'] / n)} |"
        )
    return lines


def table_ellipse(m: Merged, horizons, forecasts, quantities=("position", "velocity")) -> list[str]:
    lines = [
        "Pearson and the axis ratio / major-axis angle (deg from the frame's first axis: x, "
        "along) from the full moments; Spearman from the subsample; trimmed: the subsample "
        "without its largest 1% by norm. Median |first| / |second| component.",
        "",
        "| quantity | k (ticks) | forecast | frame | Pearson | Spearman | trimmed Pearson | ratio | angle | trimmed ratio | trimmed angle | median first / second | p99 norm |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for quantity in quantities:
        for k in horizons:
            for forecast in forecasts:
                if forecast == "P" and k > 1:
                    continue
                if forecast == "CV" and k == 1 and quantity == "velocity":
                    continue  # CV velocity is P
                for frame in m.index["frames"]:
                    e = m.ellipse(k, forecast, quantity, frame)
                    lines.append(
                        f"| {quantity} | {k} | {forecast} | {frame} | {e['pearson']:+.3f} "
                        f"| {e['spearman']:+.3f} | {e['trimmed_pearson']:+.3f} | {e['ratio']:.2f} "
                        f"| {e['angle']:+.0f} | {e['trimmed_ratio']:.2f} | {e['trimmed_angle']:+.0f} "  # noqa: E501
                        f"| {g(e['median_a'])} / {g(e['median_b'])} | {g(e['p99_norm'])} |"
                    )
    return lines


def table_least_correlated(m: Merged, horizons, forecasts) -> list[str]:
    lines = [
        "|Spearman| per frame (world / nose / track) and the frame with the smallest; same for "
        "|Pearson|.",
        "",
        "| quantity | k | forecast | abs Spearman w / n / t | least | abs Pearson w / n / t | least |",  # noqa: E501
        "|---|---|---|---|---|---|---|",
    ]
    for quantity in ("position", "velocity"):
        for k in horizons:
            for forecast in forecasts:
                if (forecast == "P" and k > 1) or (forecast == "CV" and k == 1):
                    continue
                values = [m.ellipse(k, forecast, quantity, f) for f in m.index["frames"]]
                sp = [abs(v["spearman"]) for v in values]
                pe = [abs(v["pearson"]) for v in values]
                frames = m.index["frames"]
                lines.append(
                    f"| {quantity} | {k} | {forecast} | {' / '.join(f'{v:.3f}' for v in sp)} "
                    f"| {frames[int(np.argmin(sp))]} | {' / '.join(f'{v:.3f}' for v in pe)} "
                    f"| {frames[int(np.argmin(pe))]} |"
                )
    return lines


def table_ellipse_groups(m: Merged, horizons, forecast) -> list[str]:
    lines = [
        f"Spearman per player and preset, forecast {forecast}, frames world / nose / track.",
        "",
        "| quantity | k | " + " | ".join(f"{p} {r}" for p, r in m.groups) + " |",
        "|---|---|" + "---|" * len(m.groups),
    ]
    for quantity in ("position", "velocity"):
        for k in horizons:
            cells = []
            for group in m.groups:
                values = [m.ellipse(k, forecast, quantity, f, group)["spearman"]
                          for f in m.index["frames"]]  # fmt: skip
                cells.append(" / ".join(f"{v:+.2f}" for v in values))
            lines.append(f"| {quantity} | {k} | " + " | ".join(cells) + " |")
    return lines


MAIN_SPLIT_PICKS = [
    "position/R/nose_along", "position/E/nose_along", "position/E-arc/nose_along",
    "position/R/nose_across", "position/E/nose_across", "position/E-arc/nose_across",
    "velocity/P/track_along", "velocity/R/track_along", "velocity/E/track_along",
    "velocity/P/track_across", "velocity/R/track_across", "velocity/E/track_across",
    "heading/R/value", "heading/E/value",
    "nose_rate/P/value", "nose_rate/E/value",
    "angle_of_attack/P/value", "angle_of_attack/E/value",
    "energy/P/value", "energy/E/value",
    "shield/R/value", "shield/E/value",
    "log_refractive_index/P/value", "log_refractive_index/E/value",
]  # fmt: skip
WIN_PICKS = [
    ("position", "norm", "R", "P"), ("position", "norm", "E", "R"),
    ("position", "norm", "E-arc", "R"), ("position", "norm", "E-arc", "E"),
    ("velocity", "norm", "R", "P"), ("velocity", "norm", "E", "P"), ("velocity", "norm", "E", "R"),
    ("heading", "value", "R", "P"), ("heading", "value", "E", "R"),
    ("nose_rate", "value", "E", "P"), ("angle_of_attack", "value", "E", "P"),
    ("energy", "value", "E", "P"), ("shield", "value", "E", "R"),
    ("shield_recharge_wait", "value", "E", "R"), ("gun_cooldown", "value", "E", "R"),
    ("log_refractive_index", "value", "E", "P"),
]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("main", "full"), default="main")
    parser.add_argument("--root", default=str(OUT))
    args = parser.parse_args()
    m = Merged(pathlib.Path(args.root))
    out = []
    held = ["all", "held", "changed", "power_changed", "stick_changed", "held_quiet",
            "changed_quiet", "fallback"]  # fmt: skip
    situations = ["field", "no_field", "hit", "no_hit", "quiet"]
    if args.part == "main":
        out += ["### Counts", "", *table_counts(m), ""]
        out += ["### One tick, pooled", "", *table_headline(m), ""]
        out += ["### Held and changed control", "", *table_splits(m, MAIN_SPLIT_PICKS, held), ""]
        out += [
            "### Wins",
            "",
            *table_wins(m, WIN_PICKS, ["all", "held_quiet", "changed", "field", "hit"]),
            "",
        ]
        out += ["### Last tick's change as a predictor", "",
                *table_regression(m, ["all", "held_quiet", "changed_quiet", "field", "hit"]), ""]  # fmt: skip  # noqa: E501
        out += ["### Axis correlation, least-correlated frame", "",
                *table_least_correlated(m, m.index["horizons"], ("P", "CV", "R", "E")), ""]  # fmt: skip  # noqa: E501
        out += ["### Axis correlation and ellipse, E forecast", "",
                *table_ellipse(m, m.index["horizons"], ("E",)), ""]  # fmt: skip
    else:
        picks = [n for n in m.names if not n.endswith(("/x", "/y"))]
        out += ["### Per player and preset", "", *table_groups(m, picks), ""]
        out += ["### Situations", "", *table_splits(m, MAIN_SPLIT_PICKS, situations), ""]
        out += ["### Held and changed, every series", "", *table_splits(m, picks, held), ""]
        out += ["### Last tick's change as a predictor, more subsets", "",
                *table_regression(m, held + situations), ""]  # fmt: skip
        out += ["### Axis correlation and ellipse, every forecast", "",
                *table_ellipse(m, m.index["horizons"], ("P", "CV", "R", "E")), ""]  # fmt: skip
        for forecast in ("CV", "E"):
            out += [f"### Spearman per player and preset ({forecast})", "",
                    *table_ellipse_groups(m, m.index["horizons"], forecast), ""]  # fmt: skip
    print("\n".join(out))


if __name__ == "__main__":
    main()
