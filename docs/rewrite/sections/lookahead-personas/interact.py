"""Round 2, redirected study: skill couplings from the Elo data (CPU only).

    interact.py SKILL.json [--max-terms 6] > OUT.md     # skill-2.json or around-2.json

Per size: a weighted least-squares fit of Elo on the unit coordinates (x = u - 1/2), linear +
square per row, then pairwise products x_i x_j added by forward selection (the one that most
lowers the weighted residual), kept while the leave-one-out R^2 rises, at most `--max-terms`.
From the fitted surface, Monte Carlo Sobol indices of skill over the spread-1 box (uniform u):
first order S1 (the row's main effect), total ST (with its interactions); and the iso-Elo
trade-offs: pairs of strong rows whose linear effects are opposite at the centre (moving both
up together keeps Elo: the trade ratio is the slope ratio).
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse2  # noqa: E402

from lean.scripted.parameters import SAMPLED  # noqa: E402


def design(x, pairs):
    columns = [np.ones(len(x)), *x.T, *(x.T**2 - 1.0 / 12.0)]
    columns += [x[:, i] * x[:, j] for i, j in pairs]
    return np.column_stack(columns)


def fit(x, y, e, pairs):
    return analyse2.weighted_fit(design(x, pairs), y, e)


def predict(coefficients, x, pairs):
    return design(x, pairs) @ coefficients


def sobol(coefficients, pairs, dimension, samples=200000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random((samples, dimension)) - 0.5
    b = rng.random((samples, dimension)) - 0.5
    fa, fb = predict(coefficients, a, pairs), predict(coefficients, b, pairs)
    variance = np.var(np.concatenate([fa, fb]))
    first, total = [], []
    for i in range(dimension):
        ab = a.copy()
        ab[:, i] = b[:, i]
        fab = predict(coefficients, ab, pairs)
        first.append(float(np.mean(fb * (fab - fa)) / variance))
        total.append(float(0.5 * np.mean((fa - fab) ** 2) / variance))
    return first, total, float(np.sqrt(variance))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skill")
    parser.add_argument("--max-terms", type=int, default=6)
    parser.add_argument("--out-json")
    arguments = parser.parse_args()
    record = json.loads(Path(arguments.skill).read_text())
    for agent in record["agents"]:  # around.py labels its centre "centre"
        if agent["label"] == "centre":
            agent["label"] = "yardstick"
    for entries in record["setups"].values():
        if "centre" in entries:
            entries["yardstick"] = entries.pop("centre")
    d = len(SAMPLED)
    result = {}
    for setup in record["setups"]:
        x, y, e, _ = analyse2.points(record, setup)
        pairs = []
        _, _, r2, q2, _ = fit(x, y, e, pairs)
        history = [(None, r2, q2)]
        candidates = list(itertools.combinations(range(d), 2))
        while len(pairs) < arguments.max_terms:
            best = None
            for pair in candidates:
                if pair in pairs:
                    continue
                c, _, r2_, q2_, _ = fit(x, y, e, pairs + [pair])
                if best is None or r2_ > best[1]:
                    best = (pair, r2_, q2_)
            if best is None or best[2] <= history[-1][2]:
                break
            pairs.append(best[0])
            history.append(best)
        c, s, r2, q2, _ = fit(x, y, e, pairs)
        first, total, sd = sobol(c, pairs, d)
        slopes = c[1 : d + 1]
        slope_errors = s[1 : d + 1]
        strong = [i for i in range(d) if abs(slopes[i]) > 3 * slope_errors[i]]
        trades = []
        for i, j in itertools.combinations(strong, 2):
            if slopes[i] * slopes[j] < 0:
                trades.append((SAMPLED[i], SAMPLED[j], float(-slopes[i] / slopes[j])))
        trades.sort(key=lambda t: -abs(slopes[SAMPLED.index(t[0])] * slopes[SAMPLED.index(t[1])]))
        result[setup] = {
            "pairs": [
                {
                    "rows": [SAMPLED[i], SAMPLED[j]],
                    "coefficient": float(c[1 + 2 * d + k]),
                    "error": float(s[1 + 2 * d + k]),
                }
                for k, (i, j) in enumerate(pairs)
            ],
            "r2": r2,
            "q2": q2,
            "r2_additive": history[0][1],
            "q2_additive": history[0][2],
            "sobol_first": dict(zip(SAMPLED, first, strict=True)),
            "sobol_total": dict(zip(SAMPLED, total, strict=True)),
            "model_sd": sd,
            "trades": trades[:8],
        }
        print(f"## {setup}: skill surface (points {len(y)})\n")
        print(
            f"Additive (linear + square): R^2 {history[0][1]:.2f}, "
            f"leave-one-out {history[0][2]:.2f}; "
            f"with {len(pairs)} interactions: R^2 {r2:.2f}, leave-one-out {q2:.2f}; "
            f"model Elo sd over the box {sd:.0f}.\n"
        )
        rows = [
            [" x ".join(p["rows"]), f"{p['coefficient']:+.0f} +- {p['error']:.0f}"]
            for p in result[setup]["pairs"]
        ]
        if rows:
            print(analyse2.table(["interaction (Elo per box^2)", "coefficient"], rows) + "\n")
        order = sorted(range(d), key=lambda i: -total[i])
        rows = [
            [SAMPLED[i], f"{first[i]:.3f}", f"{total[i]:.3f}", f"{total[i] - first[i]:+.3f}"]
            for i in order
        ]
        print(analyse2.table(["row", "S1", "ST", "ST - S1"], rows) + "\n")
        if trades:
            rows = [[a, b, f"{ratio:+.2f}"] for a, b, ratio in trades[:8]]
            print(
                analyse2.table(
                    ["row a", "row b", "equal Elo: du_b per du_a (linear, at the centre)"], rows
                )
                + "\n"
            )
    if arguments.out_json:
        Path(arguments.out_json).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
