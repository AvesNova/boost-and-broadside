"""Aliasing of a rotary position ladder in 2D (docs/rewrite/sections/position-codes/).

A candidate code is a list of bands, each a period P_k (m) and an axis angle theta_k: band k
reads the relative position s through two phases, 2 pi (s . u) / P_k for u its x axis
(cos theta_k, sin theta_k) and its y axis (-sin theta_k, cos theta_k). The heading pairs are
left out. The code's similarity at a shift s is

    S(s) = sum_t w_t cos(phase_t(s)) / sum_t w_t      (t over bands x axes)

the normalised dot product of the codes of x and x + s (uniform w: every pair weighed alike,
as an untrained query would). Two summaries per candidate and range R:

- confusion C(d, R) = max S over d <= |s| <= R: the best match among shifts at least d away.
  A far ghost shows as C staying high as d grows; an exact alias as C = 1.
- phase margin M(d, R) = min over d <= |s| <= R of sqrt(sum_t (delta_t / 2 sigma_t)^2 / T):
  delta_t the wrapped phase difference of s from 0, so phase errors of delta_t / 2 on each
  pair make s tie with the truth. In units of sigma_t: the RMS phase error (degrees, with
  every sigma equal) that makes a shift at least d away read as the truth. The
  "coarse noisier" profile reads band k with sigma growing x sqrt(2) per band toward the
  coarse end (the coarse bands read less precisely), so their mismatches count less.

Both extrema are exact to a tolerance, by branch and bound over square cells: each pair's
phase over a cell lies in an interval of half-width 2 pi h (|u_x| + |u_y|) / P, which bounds
its cosine (and its wrapped phase) over the cell. S(s) = S(-s), so the half plane y >= 0
suffices.

Run (CPU, a few minutes):
    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 nice -n 15 \
        env CUDA_VISIBLE_DEVICES="" PYTHONPATH=src .venv/bin/python bench/rotary_aliasing.py \
        --out docs/rewrite/sections/position-codes
"""

import argparse
import dataclasses
import json
import math
import pathlib
import time

import numpy as np

GOLDEN_ANGLE = 180.0 * (3.0 - math.sqrt(5.0))  # 137.508 degrees
GOLDEN_FRACTION = (math.sqrt(5.0) - 1.0) / 2.0  # 0.618


@dataclasses.dataclass(frozen=True)
class Code:
    """A rotary code: its bands' periods and axis angles."""

    name: str
    periods: tuple[float, ...]  # m, finest first
    angles: tuple[float, ...]  # degrees, one per band

    def terms(self, noise: str = "equal") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per pair: the axis (T, 2), the period (T,) and the noise sigma (T,) in units of the
        finest band's."""
        axes, periods, sigmas = [], [], []
        for index, (period, angle) in enumerate(zip(self.periods, self.angles)):
            theta = math.radians(angle)
            sigma = 1.0 if noise == "equal" else math.sqrt(2.0) ** index
            for axis in ((math.cos(theta), math.sin(theta)), (-math.sin(theta), math.cos(theta))):
                axes.append(axis)
                periods.append(period)
                sigmas.append(sigma)
        return np.array(axes), np.array(periods), np.array(sigmas)


def ladder(start: float, ratio: float, bands: int = 8) -> tuple[float, ...]:
    """The periods start x ratio^k, finest first."""
    return tuple(start * ratio**k for k in range(bands))


def wrapped(phase: np.ndarray) -> np.ndarray:
    """|phase| wrapped to [0, pi]."""
    return np.abs(np.remainder(phase + np.pi, 2.0 * np.pi) - np.pi)


def objective(kind: str, distance: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """What the search minimises from each pair's wrapped phase distance (..., T):
    "similarity" 1 - S (so the minimum is the max S), "margin" the mean of
    (delta / 2 sigma)^2 (its root is the margin)."""
    if kind == "similarity":
        return 1.0 - np.cos(distance).mean(-1)
    return ((distance / (2.0 * sigmas)) ** 2).mean(-1)


def search(
    code: Code,
    inner: float,
    outer: float,
    kind: str = "similarity",
    noise: str = "equal",
    tolerance: float | None = None,
    chunk: int = 2_000_000,
) -> tuple[float, tuple[float, float]]:
    """The minimum of `objective` over inner <= |s| <= outer, y >= 0, to within tolerance
    (absolute on the objective), and where it is."""
    if tolerance is None:  # 1e-4 on S; on the margin, under 0.1 degree near 0
        tolerance = 1e-4 if kind == "similarity" else 1e-6
    axes, periods, sigmas = code.terms(noise)
    spread = 2.0 * math.pi * (np.abs(axes).sum(-1)) / periods  # phase half-width per metre
    frequency = 2.0 * math.pi * axes / periods[:, None]  # (T, 2)
    h = max(code.periods[0], outer / 2000.0)  # starting half-width
    n = int(math.ceil(outer / (2 * h)))
    xs = (np.arange(-n, n) + 0.5) * 2 * h
    ys = (np.arange(0, n) + 0.5) * 2 * h
    cx, cy = (grid.ravel() for grid in np.meshgrid(xs, ys))
    best, where = np.inf, (math.nan, math.nan)
    while cx.size:
        keep_x, keep_y, bounds = [], [], []
        for start in range(0, cx.size, chunk):
            x, y = cx[start : start + chunk], cy[start : start + chunk]
            radius = np.hypot(x, y)
            reach = h * math.sqrt(2.0)
            inside = (radius + reach >= inner) & (radius - reach <= outer)
            x, y, radius = x[inside], y[inside], radius[inside]
            phases = x[:, None] * frequency[:, 0] + y[:, None] * frequency[:, 1]
            distance = wrapped(phases)
            # the centre, if it lies in the annulus, is a candidate
            value = objective(kind, distance, sigmas)
            valid = (radius >= inner) & (radius <= outer)
            if valid.any():
                index = np.argmin(np.where(valid, value, np.inf))
                if value[index] < best:
                    best, where = float(value[index]), (float(x[index]), float(y[index]))
            low = objective(kind, np.maximum(distance - h * spread, 0.0), sigmas)
            keep_x.append(x)
            keep_y.append(y)
            bounds.append(low)
        x, y, low = (np.concatenate(parts) for parts in (keep_x, keep_y, bounds))
        alive = low < best - tolerance
        x, y = x[alive], y[alive]
        if h < 1e-3 * code.periods[0]:
            break
        h /= 2
        cx = np.concatenate([x - h, x + h, x - h, x + h])
        cy = np.concatenate([y - h, y - h, y + h, y + h])
    return best, where


def candidates() -> list[Code]:
    """The named candidates of the table (progress notes, step A)."""
    flat = (0.0,) * 8
    golden = tuple((k * GOLDEN_ANGLE) % 360.0 for k in range(8))
    golden90 = tuple((k * 90.0 * GOLDEN_FRACTION) % 90.0 for k in range(8))
    even45 = tuple(k * 45.0 for k in range(8))
    even1125 = tuple(k * 11.25 for k in range(8))
    rng = np.random.default_rng(20261112)
    random = tuple(float(a) for a in rng.uniform(0.0, 90.0, 8))
    proposed = ladder(48.0, 3.405)
    codes = [
        Code("pow2 8192->64 (today)", tuple(8192.0 / 2**k for k in range(8))[::-1], flat),
        Code("r3.405 flat", proposed, flat),
        Code("r3.405 golden 137.5", proposed, golden),
        Code("r3.405 golden-of-90 55.6", proposed, golden90),
        Code("r3.405 45 steps", proposed, even45),
        Code("r3.405 11.25 steps", proposed, even1125),
        Code("r3.405 random", proposed, random),
    ]
    for ratio in (3.3, 3.35, 3.45, 3.5):
        codes.append(Code(f"r{ratio} golden 137.5", ladder(48.0, ratio), golden))
        codes.append(Code(f"r{ratio} golden-of-90 55.6", ladder(48.0, ratio), golden90))
    return codes


def search_candidates(randoms: int = 16) -> list[Code]:
    """The wider search: ratios 3.3-3.5 crossed with the structured rotation sets and
    `randoms` random fixed sets (uniform in [0, 90): the two axes repeat every 90 degrees)."""
    rng = np.random.default_rng(7)
    sets = {
        "flat": (0.0,) * 8,
        "golden 137.5": tuple((k * GOLDEN_ANGLE) % 360.0 for k in range(8)),
        "golden-of-90 55.6": tuple((k * 90.0 * GOLDEN_FRACTION) % 90.0 for k in range(8)),
        "11.25 steps": tuple(k * 11.25 for k in range(8)),
        "33.75 steps": tuple((k * 33.75) % 90.0 for k in range(8)),
        "45 steps": tuple(k * 45.0 for k in range(8)),
    }
    for index in range(randoms):
        sets[f"random {index}"] = tuple(float(a) for a in rng.uniform(0.0, 90.0, 8))
    return [
        Code(f"r{ratio} {name}", ladder(48.0, ratio), angles)
        for ratio in (3.3, 3.35, 3.405, 3.45, 3.5)
        for name, angles in sets.items()
    ]


def turn_candidates() -> list[Code]:
    """The one-parameter family `YemongConfig.rotary_turn` builds: band k turned k x turn.
    Turns t and t + 90 are the same code (the two axes repeat every 90 degrees), and t and
    -t mirror images (the same summaries), so 0..45 degrees covers it."""
    return [
        Code(f"r{ratio} turn {turn:g}", ladder(48.0, ratio), tuple(k * turn for k in range(8)))
        for ratio in (3.35, 3.405, 3.45)
        for turn in np.arange(0.0, 45.01, 1.25)
    ]


# the wider search's summaries: (kind, noise, inner, outer)
SEARCH = (
    ("margin", "equal", 1000.0, 130000.0),
    ("margin", "coarse", 1000.0, 20000.0),
    ("margin", "coarse", 1000.0, 130000.0),
    ("similarity", "equal", 1000.0, 130000.0),
)
INNER = (200.0, 1000.0, 5000.0)
OUTER = (2000.0, 20000.0, 130000.0, 500000.0)


def profile(out: pathlib.Path) -> None:
    """The coarse-noise margin M(d, 130 km) and the confusion C(d, 130 km) against d for the
    record ladder, the decided ratio unturned, at the golden angle and at the chosen turn;
    a PNG and the numbers (rotary_aliasing_profile.json)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proposed = ladder(48.0, 3.405)
    codes = [
        Code(
            "record: 8192 m / 2^k, world axes",
            tuple(8192.0 / 2**k for k in range(8))[::-1],
            (0.0,) * 8,
        ),
        Code("r 3.405, world axes", proposed, (0.0,) * 8),
        Code("r 3.405, golden angle 137.5", proposed, tuple(k * GOLDEN_ANGLE for k in range(8))),
        Code("r 3.405, turn 35 (chosen)", proposed, tuple(k * 35.0 for k in range(8))),
    ]
    inner = np.geomspace(100.0, 100_000.0, 25)
    data = {}
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for code in codes:
        margins = [
            math.degrees(math.sqrt(search(code, d, 130_000.0, "margin", "coarse")[0]))
            for d in inner
        ]
        confusion = [1.0 - search(code, d, 130_000.0, "similarity")[0] for d in inner]
        data[code.name] = dict(inner=inner.tolist(), margin=margins, confusion=confusion)
        left.plot(inner / 1e3, margins, label=code.name)
        right.plot(inner / 1e3, confusion, label=code.name)
        print(code.name, "done", flush=True)
    for axis, title in (
        (left, "coarse-noise phase margin M(d, 130 km)"),
        (right, "confusion C(d, 130 km) = max S"),
    ):
        axis.set_xscale("log")
        axis.set_xlabel("shift at least d (km)")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3)
    left.set_ylabel("finest-band sigmas (degrees)")
    right.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "rotary_aliasing_profile.png", dpi=120)
    (out / "rotary_aliasing_profile.json").write_text(json.dumps(data, indent=1))


def main() -> None:
    """The sweep's command line (the module docstring)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--only", default=None, help="substring of the candidates to run")
    parser.add_argument("--search", action="store_true", help="the wider ratio x rotation search")
    parser.add_argument("--turns", action="store_true", help="the k x turn family's sweep")
    parser.add_argument("--profile", action="store_true", help="the margin against d, as a PNG")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    if arguments.profile:
        profile(arguments.out)
        return
    name = (
        "rotary_aliasing_turns.jsonl"
        if arguments.turns
        else "rotary_aliasing_search.jsonl"
        if arguments.search
        else "rotary_aliasing.jsonl"
    )
    records = arguments.out / name
    done = set()
    if records.exists():
        for line in records.read_text().splitlines():
            row = json.loads(line)
            done.add((row["code"], row["kind"], row["noise"], row["inner"], row["outer"]))
    pool = (
        turn_candidates()
        if arguments.turns
        else search_candidates()
        if arguments.search
        else candidates()
    )
    codes = [c for c in pool if arguments.only is None or arguments.only in c.name]
    if arguments.search or arguments.turns:
        jobs = [(code, *summary) for code in codes for summary in SEARCH]
    else:
        jobs = [
            (code, kind, noise, inner, outer)
            for code in codes
            for kind, noise in (("similarity", "equal"), ("margin", "equal"), ("margin", "coarse"))
            for outer in OUTER
            for inner in INNER
            if inner < outer
        ]
    began = time.time()
    for index, (code, kind, noise, inner, outer) in enumerate(jobs):
        if (code.name, kind, noise, inner, outer) in done:
            continue
        value, where = search(code, inner, outer, kind, noise)
        row = dict(
            code=code.name,
            kind=kind,
            noise=noise,
            inner=inner,
            outer=outer,
            value=1.0 - value if kind == "similarity" else math.degrees(math.sqrt(value)),
            at=[round(where[0], 1), round(where[1], 1)],
            at_distance=round(math.hypot(*where), 1),
            periods=[round(p, 1) for p in code.periods],
            angles=[round(a, 2) for a in code.angles],
        )
        with records.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        elapsed = time.time() - began
        eta = time.strftime("%H:%M", time.localtime(began + elapsed * len(jobs) / (index + 1)))
        print(
            f"[{index + 1}/{len(jobs)} eta {eta}] {code.name:28s} {kind:10s} {noise:6s} "
            f"[{inner:>6.0f}, {outer:>7.0f}] {row['value']:.3f} at {row['at_distance']:.0f} m",
            flush=True,
        )


if __name__ == "__main__":
    main()
