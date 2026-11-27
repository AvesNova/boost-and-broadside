"""Round 2, steps 3-4: the start point of the search from the per-size quadratic fits.

    surrogate.py ANALYSIS-2.json CROWD.json OUT_PREFIX [--measure near|known|own] [--half 1]

Each size's fit (analyse2.py: Elo = a + sum_i b_i x_i + c_i (x_i^2 - 1/12), x = u - 1/2) is
read as a surrogate of that size's Elo. A crowded row at duel coordinate u and crowd value v
(box units: the crowd value is the duel value x e^(2 width v)) sits at u + v w_s at size s,
w_s the mean crowd weight of the measure at that size (crowd.py, the yardstick mirror). The
surrogate's mean over 1v1, 4v4 and 16v16 is maximised over u in [U_LOW_FIT, U_HIGH_FIT] (the
fits are not trusted far outside the box) and v, by coordinate (the fits are additive, so per
row). Writes OUT_PREFIX-space.json (tune.py's --space, with x0) and OUT_PREFIX-plain.json (no
crowd rows: one value per row).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lean.scripted.parameters import CROWDED, SAMPLED

SIZES = ("1v1", "4v4", "16v16")
MEASURES = {"near": 0.0, "known": 1.0, "own": 2.0}
U_LOW_FIT, U_HIGH_FIT = -0.25, 1.25
INERT = ("boundary_weight", "seek_weight")
# A crowded row is kept only if the crowd improves the surrogate's mean by this much (Elo).
CROWD_GAIN = 8.0


def surrogate(fit, k, u):
    x = np.asarray(u) - 0.5
    return fit["quadratic_slope"][k] * x + fit["curvature"][k] * (x * x - 1.0 / 12.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis")
    parser.add_argument("crowd")
    parser.add_argument("prefix")
    parser.add_argument("--measure", choices=tuple(MEASURES), default="own")
    parser.add_argument("--half", type=float, default=1.0)
    arguments = parser.parse_args()
    fits = json.loads(Path(arguments.analysis).read_text())["fits"]
    crowd = json.loads(Path(arguments.crowd).read_text())
    weights = []
    for size in SIZES:
        # The mean crowd weight at this half, from the measure's histogram (crowd.py).
        centres, counts = np.array(crowd[size][arguments.measure]["histogram"]).T
        excess = np.clip(centres - 1.0, 0.0, None)
        weights.append(float((counts * excess / (excess + arguments.half)).sum() / counts.sum()))
    weights = np.array(weights)
    grid = np.linspace(U_LOW_FIT, U_HIGH_FIT, 61)
    v_grid = np.linspace(-1.5, 1.5, 61)
    rows = [name for name in SAMPLED if name not in INERT]
    plain, x0, crowded, report = {}, {}, [], {}
    for name in rows:
        k = SAMPLED.index(name)
        each = np.stack([surrogate(fits[s], k, grid) for s in SIZES])  # (sizes, grid)
        best = int(np.argmax(each.mean(0)))
        plain[name] = float(grid[best])
        entry = {
            "plain_u": float(grid[best]),
            "plain_gain": float(each.mean(0)[best] - each.mean(0)[np.argmin(abs(grid - 0.5))]),
            "best_u_per_size": {s: float(grid[int(np.argmax(each[i]))]) for i, s in enumerate(SIZES)},
        }
        if name in CROWDED:
            # u + v w_s per size, on the grid of (u, v).
            uu, vv = np.meshgrid(grid, v_grid, indexing="ij")
            total = np.zeros_like(uu)
            for i, s in enumerate(SIZES):
                position = np.clip(uu + vv * weights[i], U_LOW_FIT, U_HIGH_FIT)
                total += surrogate(fits[s], k, position)
            total /= len(SIZES)
            index = np.unravel_index(int(np.argmax(total)), total.shape)
            gain = float(total[index] - each.mean(0)[best])
            entry.update({"crowd_u": float(uu[index]), "crowd_v": float(vv[index]), "crowd_gain": gain})
            if gain >= CROWD_GAIN:
                crowded.append(name)
                x0[name] = (float(uu[index]), float(vv[index]))
        report[name] = entry
    space_rows = rows
    z = [x0[name][0] if name in x0 else plain[name] for name in space_rows]
    z += [x0[name][1] for name in crowded]
    predicted = {}
    for i, s in enumerate(SIZES):
        total = fits[s]["intercept"]
        for name in space_rows:
            k = SAMPLED.index(name)
            u = x0[name][0] + x0[name][1] * weights[i] if name in x0 else plain[name]
            total += float(surrogate(fits[s], k, np.clip(u, U_LOW_FIT, U_HIGH_FIT)))
        predicted[s] = total
    predicted_plain = {
        s: fits[s]["intercept"]
        + sum(float(surrogate(fits[s], SAMPLED.index(n), plain[n])) for n in space_rows)
        for s in SIZES
    }
    space = {
        "rows": space_rows,
        "crowded": crowded,
        "measure": MEASURES[arguments.measure],
        "half": arguments.half,
        "x0": z,
        "weights": dict(zip(SIZES, weights.tolist(), strict=True)),
        "predicted": predicted,
        "report": report,
    }
    Path(f"{arguments.prefix}-space.json").write_text(json.dumps(space, indent=1))
    plain_space = {
        "rows": space_rows,
        "crowded": [],
        "measure": 0.0,
        "half": 1.0,
        "x0": [plain[name] for name in space_rows],
        "predicted": predicted_plain,
    }
    Path(f"{arguments.prefix}-plain.json").write_text(json.dumps(plain_space, indent=1))
    print("crowd weights", dict(zip(SIZES, np.round(weights, 3).tolist(), strict=True)))
    print("crowded", crowded)
    print("predicted (crowd)", {s: round(v) for s, v in predicted.items()})
    print("predicted (plain)", {s: round(v) for s, v in predicted_plain.items()})


if __name__ == "__main__":
    main()
