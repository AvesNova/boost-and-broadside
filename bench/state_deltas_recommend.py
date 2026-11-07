"""The recommended delta-head codes (docs/rewrite/sections/state-deltas.md, section 4),
evaluated on the pooled one-tick samples: a symlog two-hot code per component whose range
is the hard bound (bench/state_deltas_bounds.py, the physical box) and whose knee is the
median nonzero |change|, with K = 1 + 4 ln(1 + range / knee) bins rounded up to odd (a
symlog step of 0.5 per bin, so the bin at the knee is about one knee wide). Prints a table."""

import math

import numpy as np

from bench.state_deltas_codes import entropy, grid_report, local_width, pooled, symlog_grid

# component sample, hard bound (range), unit
RECOMMENDED = (
    ("position.arc.1.nose_along", 10.0, "m"),
    ("position.arc.1.nose_across", 10.0, "m"),
    ("velocity.rotated.1.nose_along", 400.0, "m/s"),
    ("velocity.rotated.1.nose_across", 400.0, "m/s"),
    ("heading.dead_reckoning.1.angle", 1.0, "rad"),
    ("nose_rate.persistence.1.value", 25.0, "rad/s"),
    ("angle_of_attack.persistence.1.value", 0.5, "rad"),
    ("energy.persistence.1.value", 0.07, "fraction"),
    ("log_refractive_index.dead_reckoning.1.value", 0.35, "ln n"),
    ("shield.recharge.1.value", 1.0, "fraction"),
)
# Knees set from the rules where the median nonzero change is float rounding: the shield's
# recharge residual is 0 up to ~1e-8 except for damage, whose smallest step is the heavy's
# boundary damage at the arena edge, 5 dt / 160 = 0.001 per tick.
KNEES = {"shield.recharge.1.value": 0.001}


def bins_for(bound: float, knee: float) -> int:
    """The odd bin count with a symlog step of 0.5 per bin from 0 to bound."""
    k = math.ceil(1 + 4 * math.log1p(bound / knee))
    return k if k % 2 else k + 1


def main() -> None:
    """Print the recommended codes' table."""
    print(
        "| component | knee | range | bins | share at 0 | width at p50 / p99 / p99.9 / max "
        "| beyond the range | H (nats) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for name, bound, unit in RECOMMENDED:
        z = pooled([name])[name]
        z = z[np.isfinite(z)]
        absolute = np.abs(z)
        knee = KNEES.get(name) or float(f"{float(np.median(absolute[absolute > 0])):.1g}")
        bins = bins_for(bound, knee)
        grid = symlog_grid(-bound, bound, bins, knee)
        report = grid_report(grid, z)
        widths = [local_width(grid, float(np.percentile(absolute, q))) for q in (50, 99, 99.9)]
        widths.append(local_width(grid, float(absolute.max())))
        print(
            f"| {name} ({unit}) | {knee:g} | +-{bound:g} | {bins} | "
            f"{100 * (z == 0).mean():.1f}% | " + " / ".join(f"{w:.3g}" for w in widths)
            + f" | {report['tail']} | {report['H']:.2f} |"
        )  # fmt: skip
    # the two event quantities: their residual is one of two values
    for name in ("gun_cooldown.countdown.1.value", "shield_recharge_wait.countdown.1.value"):
        z = pooled([name])[name]
        # a reset leaves at least one tick's countdown (dt) of residual; a shot one tick
        event = np.abs(z) > (0.5 if name.startswith("gun") else 0.01)
        p = float(event.mean())
        h = entropy(np.array([p, 1 - p]))
        print(f"| {name} | event share {100 * p:.2f}% | 2 bins | H {h:.3f} |")


if __name__ == "__main__":
    main()
