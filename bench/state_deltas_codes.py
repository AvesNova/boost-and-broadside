"""Compare target codes for each ship-state change (docs/rewrite/sections/state-deltas-brief.md,
part 2), on the subsamples bench/state_deltas_measure.py saved: both players and every
preset pooled with equal weight (the same number of samples from each).

Every code here is a two-hot code: a value between two neighbouring centres puts weight
1 - f on the lower and f on the upper, so it decodes exactly inside the code's range. Per
candidate:

    bins        units of the code (the sum over its groups)
    coverage    the share of samples inside the range (no clipping), and whether the
                pooled 99.9% |change| is inside
    width@p50   the centre spacing at the median |change| (m, rad, ...), and as a fraction
                of the median; width@p99.9 the same at the 99.9% quantile
    nearest     the mean |error| of decoding to the nearest centre (a one-hot read-out)
    clip        the mean |error| of the two-hot decode (only clipping)
    H           the entropy of the code's marginal (the mean target over samples), nats: the
                cross-entropy of a predictor that knows only the distribution of the change;
                equal-mass bins maximise it
    H_self      the mean entropy of one target (<= ln 2 per group): the cross-entropy floor
                of a predictor that knows each change exactly
    tail        samples clipped

Writes <scratch>/codes.json and prints markdown tables.
"""

import json
import math
import pathlib

import numpy as np

SCRATCH = pathlib.Path("/tmp/bnb-scratch-deltas")
PLAYERS = ("scripted", "calm-fern")
PRESETS = ("frontline", "deathmatch", "deathmatch_wide")
BUDGET_1D = 21
PER_AXIS = 27  # 2-D budget: 54 units (per axis 2 x 27, 3 x 19, 6 levels x 9, 27 + 27)
DIRECTIONS = 19
LEVELS_2D = 6
COLOURS = 3
SQRT3 = math.sqrt(3.0)


def nice_ceiling(value: float) -> float:
    """The next of 1, 2, 5 x 10^k at or above value."""
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    for mantissa in (1.0, 2.0, 5.0, 10.0):
        if mantissa * 10**exponent >= value * (1 - 1e-12):
            return mantissa * 10**exponent
    return 10.0 ** (exponent + 1)


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def self_entropy(fraction: np.ndarray) -> float:
    f = np.clip(fraction, 1e-12, 1 - 1e-12)
    h = -(f * np.log(f) + (1 - f) * np.log(1 - f))
    return float(np.where((fraction <= 0) | (fraction >= 1), 0.0, h).mean())


def two_hot(centres: np.ndarray, z: np.ndarray):
    """(lower index, fraction, clipped z) on a monotone grid."""
    clipped = np.clip(z, centres[0], centres[-1])
    lower = np.clip(np.searchsorted(centres, clipped, "right") - 1, 0, len(centres) - 2)
    width = centres[lower + 1] - centres[lower]
    fraction = np.clip((clipped - centres[lower]) / width, 0.0, 1.0)
    return lower, fraction, clipped


def grid_report(centres: np.ndarray, z: np.ndarray) -> dict:
    """The metrics of one two-hot group over monotone centres."""
    lower, fraction, clipped = two_hot(centres, z)
    K = len(centres)
    marginal = np.bincount(lower, 1 - fraction, K) + np.bincount(lower + 1, fraction, K)
    marginal /= marginal.sum()
    nearest = np.where(fraction < 0.5, centres[lower], centres[lower + 1])
    return dict(
        lower=lower,
        fraction=fraction,
        nearest=nearest,
        decoded=clipped,
        H=entropy(marginal),
        H_self=self_entropy(fraction),
        tail=int(((z < centres[0]) | (z > centres[-1])).sum()),
    )


def local_width(centres: np.ndarray, value: float) -> float:
    """The spacing of the interval holding value (the mean of the +- value intervals)."""
    widths = []
    for v in (value, -value):
        if centres[0] <= v <= centres[-1]:
            i = int(np.clip(np.searchsorted(centres, v, "right") - 1, 0, len(centres) - 2))
            widths.append(centres[i + 1] - centres[i])
    return float(np.mean(widths)) if widths else float("nan")


# ------------------------------------------------------------------- 1-D grids


def uniform_grid(low, high, bins):
    return np.linspace(low, high, bins)


def symlog_grid(low, high, bins, knee):
    def forward(v):
        return np.sign(v) * np.log1p(np.abs(v) / knee)

    u = np.linspace(forward(low), forward(high), bins)
    grid = np.sign(u) * knee * np.expm1(np.abs(u))
    if low < 0 < high and bins % 2 == 1:
        grid[bins // 2] = 0.0
    return grid


def quantile_grid(z, bins):
    """Equal-mass centres with an exact-zero centre: the quantiles of the nonzero changes at
    bins - 1 evenly spaced probabilities from 0 to 1, and 0."""
    nonzero = z[z != 0]
    if nonzero.size == 0:
        return np.array([-1.0, 0.0, 1.0])
    q = np.quantile(nonzero, np.linspace(0, 1, bins - 1))
    return np.unique(np.concatenate([q, [0.0]]))


def hierarchy_1d(z, period, colours, levels):
    """The multi-level circular colour code of z (period = range): per level, two-hot
    over `colours` circular colours of (z mod p_l) / p_l * colours. Decodes exactly; the
    resolution is the finest level's spacing."""
    shifted = np.remainder(z + 0.5 * period, period)
    H, H_self = 0.0, 0.0
    for level in range(levels):
        p = period / colours**level
        coordinate = np.remainder(shifted, p) / p * colours
        lower = np.floor(coordinate).astype(int) % colours
        fraction = coordinate - np.floor(coordinate)
        marginal = np.bincount(lower, 1 - fraction, colours) + np.bincount(
            (lower + 1) % colours, fraction, colours
        )
        H += entropy(marginal / marginal.sum())
        H_self += self_entropy(fraction)
    finest = period / colours**levels
    nearest = np.round(shifted / finest) * finest - 0.5 * period
    return dict(H=H, H_self=H_self, nearest=nearest, finest=finest)


def wrapped_grid_report(z, bins):
    """Uniform bins over (-pi, pi] with wrap (the last neighbours the first)."""
    spacing = 2 * math.pi / bins
    coordinate = np.remainder(z + math.pi, 2 * math.pi) / spacing
    lower = np.floor(coordinate).astype(int) % bins
    fraction = coordinate - np.floor(coordinate)
    marginal = np.bincount(lower, 1 - fraction, bins) + np.bincount(
        (lower + 1) % bins, fraction, bins
    )
    nearest = (np.round(coordinate) * spacing) - math.pi
    error = np.abs(np.remainder(z - nearest + math.pi, 2 * math.pi) - math.pi)
    return dict(H=entropy(marginal / marginal.sum()), H_self=self_entropy(fraction),
                nearest_error=float(error.mean()), spacing=spacing)  # fmt: skip


def evaluate_1d(z: np.ndarray, circular: bool = False, budget: int = BUDGET_1D) -> dict:
    absolute = np.abs(z)
    median, p999, top = (float(np.percentile(absolute, q)) for q in (50, 99.9, 100))
    nonzero_median = float(np.median(absolute[absolute > 0])) if (absolute > 0).any() else 1.0
    low, high = float(z.min()), float(z.max())
    R = nice_ceiling(max(-low, high))
    knee = nonzero_median
    results = {}

    def add(name, centres):
        r = grid_report(centres, z)
        results[name] = dict(
            bins=len(centres),
            range=[float(centres[0]), float(centres[-1])],
            coverage=1 - r["tail"] / z.size,
            covers_p999=bool(centres[0] <= -p999 or z.min() >= -p999) and centres[-1] >= p999,
            width_p50=local_width(centres, median),
            width_p999=local_width(centres, p999),
            nearest=float(np.abs(z - r["nearest"]).mean()),
            clip=float(np.abs(z - r["decoded"]).mean()),
            H=r["H"],
            H_self=r["H_self"],
            tail=r["tail"],
        )

    add("uniform (max range)", uniform_grid(-R, R, budget))
    R999 = nice_ceiling(p999)
    add("uniform (99.9% range)", uniform_grid(-R999, R999, budget))
    add(f"symlog (knee {knee:.3g})", symlog_grid(-R, R, budget, knee))
    add("quantile (+ zero)", quantile_grid(z, budget))
    levels = budget // COLOURS
    hierarchy = hierarchy_1d(z, 2 * R, COLOURS, levels)
    results[f"hierarchy {levels}x{COLOURS} (period {2 * R:g})"] = dict(
        bins=levels * COLOURS,
        range=[-R, R],
        coverage=float((absolute <= R).mean()),
        covers_p999=bool(p999 <= R),
        width_p50=hierarchy["finest"],
        width_p999=hierarchy["finest"],
        nearest=float(np.abs(z - hierarchy["nearest"]).mean()),
        clip=0.0,
        H=hierarchy["H"],
        H_self=hierarchy["H_self"],
        tail=int((absolute > R).sum()),
    )
    if circular:
        circle = hierarchy_1d(z, 2 * math.pi, COLOURS, levels)
        results[f"circular colours {levels}x{COLOURS} (period 2pi)"] = dict(
            bins=levels * COLOURS, range=[-math.pi, math.pi], coverage=1.0, covers_p999=True,
            width_p50=circle["finest"], width_p999=circle["finest"],
            nearest=float(np.abs(z - circle["nearest"]).mean()), clip=0.0,
            H=circle["H"], H_self=circle["H_self"], tail=0,
        )  # fmt: skip
        wrapped = wrapped_grid_report(z, budget)
        results[f"wrapped uniform {budget}"] = dict(
            bins=budget, range=[-math.pi, math.pi], coverage=1.0, covers_p999=True,
            width_p50=wrapped["spacing"], width_p999=wrapped["spacing"],
            nearest=wrapped["nearest_error"], clip=0.0,
            H=wrapped["H"], H_self=wrapped["H_self"], tail=0,
        )  # fmt: skip
    for entry in results.values():
        entry["width_p50_fraction"] = entry["width_p50"] / median if median > 0 else float("nan")
    return dict(
        n=int(z.size),
        zero=float((z == 0).mean()),
        median=median,
        p999=p999,
        max=top,
        knee=knee,
        candidates=results,
    )


# ------------------------------------------------------------------- 2-D codes


def project(x, y):
    return np.stack([x, -0.5 * x + 0.5 * SQRT3 * y, -0.5 * x - 0.5 * SQRT3 * y], -1)


def axis_grid(z: np.ndarray, bins: int) -> np.ndarray:
    """A symlog grid for one axis: its own knee (the median nonzero |z|) and range (the
    nice ceiling of max |z|), symmetric with a zero centre."""
    absolute = np.abs(z)
    knee = float(np.median(absolute[absolute > 0])) if (absolute > 0).any() else 1.0
    R = nice_ceiling(float(absolute.max()))
    return symlog_grid(-R, R, bins, knee)


def evaluate_2d(frames: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """The 2-D candidates. frames: {"world": (x, y), "nose": (along, across), "track":
    (along, across)}; NaN rows (the track frame below its speed threshold) are dropped
    within their frame. The norm is frame-free, so medians are the world frame's."""
    x, y = frames["world"]
    norm = np.hypot(x, y)
    median, p999, top = (float(np.percentile(norm, q)) for q in (50, 99.9, 100))
    R = nice_ceiling(max(np.abs(x).max(), np.abs(y).max()))
    results = {}

    def per_axis(name, a, b, grid_a, grid_b):
        ra, rb = grid_report(grid_a, a), grid_report(grid_b, b)
        error = np.hypot(a - ra["nearest"], b - rb["nearest"])
        tail = int(((a < grid_a[0]) | (a > grid_a[-1]) | (b < grid_b[0]) | (b > grid_b[-1])).sum())
        results[name] = dict(
            bins=len(grid_a) + len(grid_b),
            samples=int(a.size),
            coverage=1 - tail / a.size,
            covers_p999=bool(min(grid_a[-1], grid_b[-1]) >= p999 / math.sqrt(2)),
            width_p50=0.5
            * (
                local_width(grid_a, median / math.sqrt(2))
                + local_width(grid_b, median / math.sqrt(2))
            ),
            width_p999=0.5
            * (local_width(grid_a, p999 / math.sqrt(2)) + local_width(grid_b, p999 / math.sqrt(2))),
            nearest=float(error.mean()),
            clip=float(np.hypot(a - ra["decoded"], b - rb["decoded"]).mean()),
            H=ra["H"] + rb["H"],
            H_self=ra["H_self"] + rb["H_self"],
            H_axes=[ra["H"], rb["H"]],
            tail=tail,
        )

    def polar(name, a, b):
        magnitude, angle = np.hypot(a, b), np.arctan2(b, a)
        knee_m = float(np.median(magnitude[magnitude > 0]))
        grid_m = symlog_grid(0.0, nice_ceiling(magnitude.max()), PER_AXIS, knee_m)
        rm = grid_report(grid_m, magnitude)
        wrapped = wrapped_grid_report(angle, PER_AXIS)
        spacing = wrapped["spacing"]
        nearest_angle = np.round((angle + math.pi) / spacing) * spacing - math.pi
        na, nb = rm["nearest"] * np.cos(nearest_angle), rm["nearest"] * np.sin(nearest_angle)
        results[name] = dict(
            bins=2 * PER_AXIS,
            samples=int(a.size),
            coverage=1 - rm["tail"] / a.size,
            covers_p999=bool(grid_m[-1] >= p999),
            width_p50=float(np.hypot(local_width(grid_m, median), spacing * median)),
            width_p999=float(np.hypot(local_width(grid_m, p999), spacing * p999)),
            nearest=float(np.hypot(a - na, b - nb).mean()),
            clip=float(np.abs(magnitude - rm["decoded"]).mean()),
            H=rm["H"] + wrapped["H"],
            H_self=rm["H_self"] + wrapped["H_self"],
            H_axes=[rm["H"], wrapped["H"]],
            tail=rm["tail"],
        )

    uniform = uniform_grid(-R, R, PER_AXIS)
    per_axis(f"world x, y uniform 2x{PER_AXIS}", x, y, uniform, uniform)
    for frame, (a, b) in frames.items():
        keep = np.isfinite(a) & np.isfinite(b)
        a, b = a[keep], b[keep]
        per_axis(f"{frame} per axis symlog 2x{PER_AXIS}", a, b,
                 axis_grid(a, PER_AXIS), axis_grid(b, PER_AXIS))  # fmt: skip
        polar(f"{frame} polar symlog |d| {PER_AXIS} + angle {PER_AXIS}", a, b)
    # three directions 120 degrees apart (world), least-squares decode
    projections = project(x, y)
    knee = float(np.median(np.abs(np.concatenate([x, y]))[np.concatenate([x, y]) != 0]))
    grid = symlog_grid(-R, R, DIRECTIONS, knee)
    reports = [grid_report(grid, projections[:, i]) for i in range(3)]

    def decode(p):
        return (2 / 3) * (p[:, 0] - 0.5 * (p[:, 1] + p[:, 2])), (2 / 3) * (
            0.5 * SQRT3 * (p[:, 1] - p[:, 2])
        )

    nx, ny = decode(np.stack([r["nearest"] for r in reports], -1))
    cx, cy = decode(np.stack([r["decoded"] for r in reports], -1))
    tail = int(np.any(np.abs(projections) > R, -1).sum())
    results[f"world three directions symlog 3x{DIRECTIONS}"] = dict(
        bins=3 * DIRECTIONS,
        samples=int(x.size),
        coverage=1 - tail / x.size,
        covers_p999=bool(R >= p999),
        width_p50=local_width(grid, median / math.sqrt(2)),
        width_p999=local_width(grid, p999 / math.sqrt(2)),
        nearest=float(np.hypot(x - nx, y - ny).mean()),
        clip=float(np.hypot(x - cx, y - cy).mean()),
        H=sum(r["H"] for r in reports),
        H_self=sum(r["H_self"] for r in reports),
        tail=tail,
    )
    # joint 2-D hierarchy, 3 x 3 colours per level, period = range (world)
    period = 2 * R
    sx, sy = np.remainder(x + R, period), np.remainder(y + R, period)
    H, H_self = 0.0, 0.0
    for level in range(LEVELS_2D):
        p = period / COLOURS**level
        cxl, cyl = np.remainder(sx, p) / p * COLOURS, np.remainder(sy, p) / p * COLOURS
        lx, ly = np.floor(cxl).astype(int) % COLOURS, np.floor(cyl).astype(int) % COLOURS
        fx, fy = cxl - np.floor(cxl), cyl - np.floor(cyl)
        joint = np.zeros(9)
        for dx_, wx in ((0, 1 - fx), (1, fx)):
            for dy_, wy in ((0, 1 - fy), (1, fy)):
                joint += np.bincount(
                    ((lx + dx_) % COLOURS) * COLOURS + (ly + dy_) % COLOURS, wx * wy, 9
                )
        H += entropy(joint / joint.sum())
        H_self += self_entropy(fx) + self_entropy(fy)
    finest = period / COLOURS**LEVELS_2D
    nearest_x = np.round(sx / finest) * finest - R
    nearest_y = np.round(sy / finest) * finest - R
    results[f"world joint 2-D hierarchy {LEVELS_2D}x9 (period {period:g})"] = dict(
        bins=9 * LEVELS_2D,
        samples=int(x.size),
        coverage=float(((np.abs(x) <= R) & (np.abs(y) <= R)).mean()),
        covers_p999=bool(R >= p999),
        width_p50=finest,
        width_p999=finest,
        nearest=float(np.hypot(x - nearest_x, y - nearest_y).mean()),
        clip=0.0,
        H=H,
        H_self=H_self,
        tail=0,
    )
    for entry in results.values():
        entry["width_p50_fraction"] = entry["width_p50"] / median if median > 0 else float("nan")
    return dict(n=int(x.size), median=median, p999=p999, max=top, range=R, candidates=results)


def budget_sweep(z: np.ndarray, counts=(11, 15, 21, 31, 41, 63)) -> dict:
    """The symlog per-axis code of one component at several bin counts: width at the
    median |z| (and as a fraction of it), width at p99.9, H and the nearest-centre error."""
    z = z[np.isfinite(z)]
    absolute = np.abs(z)
    median, p999 = float(np.median(absolute)), float(np.percentile(absolute, 99.9))
    out = {}
    for bins in counts:
        grid = axis_grid(z, bins)
        r = grid_report(grid, z)
        width = local_width(grid, median)
        out[str(bins)] = dict(
            width_p50=width,
            width_p50_fraction=width / median if median > 0 else float("nan"),
            width_p999=local_width(grid, p999),
            nearest=float(np.abs(z - r["nearest"]).mean()),
            H=r["H"],
        )
    return dict(median=median, p999=p999, sweep=out)


# ------------------------------------------------------------------- driver


def pooled(names: list[str]) -> dict[str, np.ndarray]:
    """The named sample arrays, pooled over every player and preset at equal weight."""
    parts: dict[str, list] = {name: [] for name in names}
    counts = []
    files = []
    for player in PLAYERS:
        for preset in PRESETS:
            path = SCRATCH / "samples" / player / f"{preset}.npz"
            if path.exists():
                files.append(np.load(path))
    for name in names:
        counts.append(min(len(f[name]) for f in files))
    for f in files:
        for name, count in zip(names, counts):
            parts[name].append(f[name][:count].astype(np.float64))
    return {name: np.concatenate(v) for name, v in parts.items()}


ONE_D = (
    ("heading", "persistence", 1, True),
    ("heading", "dead_reckoning", 1, True),
    ("heading", "persistence", 8, True),
    ("heading", "dead_reckoning", 8, True),
    ("heading", "persistence", 32, True),
    ("heading", "dead_reckoning", 32, True),
    ("nose_rate", "persistence", 1, False),
    ("angle_of_attack", "persistence", 1, False),
    ("shield_recharge_wait", "countdown", 1, False),
    ("shield", "persistence", 1, False),
    ("shield", "recharge", 1, False),
    ("energy", "persistence", 1, False),
    ("gun_cooldown", "countdown", 1, False),
    ("log_refractive_index", "persistence", 1, False),
    ("log_refractive_index", "dead_reckoning", 1, False),
)
TWO_D = (
    ("position", "persistence", 1),
    ("position", "dead_reckoning", 1),
    ("position", "arc", 1),
    ("position", "arc", 8),
    ("position", "arc", 32),
    ("velocity", "persistence", 1),
    ("velocity", "rotated", 1),
    ("velocity", "rotated", 8),
    ("velocity", "rotated", 32),
)


def main() -> None:
    report = {}
    for quantity, base, horizon, circular in ONE_D:
        component = "angle" if quantity == "heading" else "value"
        name = f"{quantity}.{base}.{horizon}.{component}"
        z = pooled([name])[name]
        report[name] = evaluate_1d(z, circular)
        print(name, "done", flush=True)
    for quantity, base, horizon in TWO_D:
        prefix = f"{quantity}.{base}.{horizon}"
        components = ("x", "y", "nose_along", "nose_across", "track_along", "track_across")
        data = pooled([f"{prefix}.{c}" for c in components])
        get = {c: data[f"{prefix}.{c}"] for c in components}
        report[prefix] = evaluate_2d(
            dict(
                world=(get["x"], get["y"]),
                nose=(get["nose_along"], get["nose_across"]),
                track=(get["track_along"], get["track_across"]),
            )
        )
        report[prefix]["track_skipped"] = float(np.mean(~np.isfinite(get["track_along"])))
        report[prefix]["budget"] = {c: budget_sweep(get[c]) for c in components}
        print(quantity, base, horizon, "done", flush=True)
    for quantity, base, horizon, _ in ONE_D:
        component = "angle" if quantity == "heading" else "value"
        name = f"{quantity}.{base}.{horizon}.{component}"
        report[name]["budget"] = budget_sweep(pooled([name])[name])
    # gun cooldown: a discrete residual; its distinct values
    z = pooled(["gun_cooldown.countdown.1.value"])["gun_cooldown.countdown.1.value"]
    values, counts = np.unique(np.round(z, 3), return_counts=True)
    report["gun_cooldown.values"] = dict(zip(map(str, values.tolist()), (counts / z.size).tolist()))
    (SCRATCH / "codes.json").write_text(json.dumps(report, indent=1, default=lambda o: o.item()))


if __name__ == "__main__":
    main()
