"""Round 2 tables from skill2.py's report (CPU only).

    analyse2.py SKILL2.json > tables-2-skill.md   # also writes analysis-2.json beside this file

Per size: the box edges' Elo, the population's Elo spread, weighted least-squares fits of Elo
on the unit coordinates (linear; and quadratic: each row's slope and curvature at the centre,
no interactions), the rows whose effect depends on size (the linear slopes' differences
between sizes in standard errors), the style counters with their correlation to Elo, and the
cross-size correlation of the personas' Elo.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

from lean.scripted.parameters import SAMPLED

HERE = Path(__file__).resolve().parent
STYLE = ("turning", "two_circle", "store", "speed", "shots", "hit_range", "climb", "dive")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
    return "\n".join(out)


def points(record, setup):
    """(unit coordinates centred at 0, Elo, Elo error, labels) of every non-yardstick agent."""
    units = {agent["label"]: agent["unit"] for agent in record["agents"]}
    xs, ys, es, labels = [], [], [], []
    for label, entry in record["setups"][setup].items():
        if label == "yardstick" or entry["elo"] is None:
            continue
        xs.append(np.array(units[label]) - 0.5)
        ys.append(entry["elo"])
        es.append(max(entry["elo_error"], 1.0))
        labels.append(label)
    return np.array(xs), np.array(ys), np.array(es), labels


def weighted_fit(design, y, errors):
    w = 1.0 / errors**2
    sw = np.sqrt(w)
    coefficients, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
    residual = y - design @ coefficients
    dof = max(len(y) - design.shape[1], 1)
    scale = float((w * residual**2).sum() / dof)
    covariance = np.linalg.inv((design * w[:, None]).T @ design) * max(scale, 1.0)
    r2 = 1.0 - float((w * residual**2).sum() / (w * (y - np.average(y, weights=w)) ** 2).sum())
    # Leave-one-out R^2 (the hat matrix): how well the fit predicts an agent it has not seen.
    weighted = design * sw[:, None]
    hat = np.einsum("ij,jk,ik->i", weighted, np.linalg.pinv(weighted.T @ weighted), weighted)
    loo = residual / np.clip(1.0 - hat, 1e-6, None)
    q2 = 1.0 - float((w * loo**2).sum() / (w * (y - np.average(y, weights=w)) ** 2).sum())
    return coefficients, np.sqrt(np.diag(covariance)), r2, q2, covariance


def fits(record):
    out = {}
    for setup in record["setups"]:
        x, y, e, _ = points(record, setup)
        if len(y) < 2 * len(SAMPLED) + 2:
            continue
        linear = np.column_stack([np.ones(len(y)), x])
        c1, s1, r2_1, q2_1, _ = weighted_fit(linear, y, e)
        # Quadratic in x = u - 0.5 per row: Elo = a + sum b_i x_i + c_i (x_i^2 - 1/12).
        quadratic = np.column_stack([linear, x**2 - 1.0 / 12.0])
        c2, s2, r2_2, q2_2, _ = weighted_fit(quadratic, y, e)
        d = len(SAMPLED)
        b, c = c2[1 : d + 1], c2[d + 1 :]
        # The best u per row on [0, 1] from the quadratic (others at the centre).
        best = []
        for slope, curvature in zip(b, c, strict=True):
            candidates = [-0.5, 0.5]
            if curvature < 0:
                candidates.append(float(np.clip(-slope / (2 * curvature), -0.5, 0.5)))
            values = [slope * t + curvature * t * t for t in candidates]
            best.append(0.5 + candidates[int(np.argmax(values))])
        out[setup] = {
            "intercept": float(c1[0]),
            "slope": c1[1:].tolist(),
            "slope_error": s1[1:].tolist(),
            "r2": r2_1,
            "q2": q2_1,
            "quadratic_slope": b.tolist(),
            "quadratic_slope_error": s2[1 : d + 1].tolist(),
            "curvature": c.tolist(),
            "curvature_error": s2[d + 1 :].tolist(),
            "quadratic_r2": r2_2,
            "quadratic_q2": q2_2,
            "best_u": best,
            "points": int(len(y)),
        }
    return out


def population(record):
    rows, record_out = [], {}
    personas = [a["label"] for a in record["agents"] if a["label"].startswith("persona")]
    for setup, entries in record["setups"].items():
        elo = np.array([entries[p]["elo"] for p in personas if entries[p]["elo"] is not None])
        err = np.array([entries[p]["elo_error"] for p in personas if entries[p]["elo"] is not None])
        if len(elo) < 4:
            continue
        games = sum(entries[label].get("games", 0) for label in entries if label != "yardstick")
        sd = float(elo.std(ddof=1))
        true_sd = math.sqrt(max(sd**2 - float((err**2).mean()), 0.0))
        quantiles = np.quantile(elo, [0, 0.1, 0.5, 0.9, 1.0])
        best = max(personas, key=lambda p: entries[p]["elo"] or -1e9)
        draws = [entries[p]["draws"] / entries[p]["games"] for p in personas if entries[p]["games"]]
        clock = [entries[p]["at_clock"] for p in personas if entries[p]["at_clock"] is not None]
        record_out[setup] = {
            "mean": float(elo.mean()),
            "sd": sd,
            "sd_true": true_sd,
            "mean_error": float(err.mean()),
            "quantiles": quantiles.tolist(),
            "games": games,
            "above_1000": int((elo > 1000).sum()),
            "best": best,
            "draws_mean": float(np.mean(draws)),
            "draws_max": float(np.max(draws)),
            "at_clock_mean": float(np.mean(clock)) if clock else None,
        }
        rows.append(
            [
                setup,
                len(elo),
                games,
                f"{elo.mean():.0f}",
                f"{sd:.0f}",
                f"{true_sd:.0f}",
                f"{err.mean():.0f}",
                " / ".join(f"{q:.0f}" for q in quantiles),
                int((elo > 1000).sum()),
                best,
                f"{np.mean(draws):.1%} / {np.max(draws):.1%}",
                f"{np.mean(clock):.1%}" if clock else "-",
            ]
        )
    header = [
        "size",
        "personas",
        "games (all agents)",
        "mean Elo",
        "sd observed",
        "sd without error",
        "mean error",
        "min / 10% / median / 90% / max",
        "above 1000",
        "best",
        "draws mean / max",
        "games at the clock",
    ]
    return table(header, rows), record_out


def correlations(record):
    personas = [a["label"] for a in record["agents"] if a["label"].startswith("persona")]
    setups = list(record["setups"])
    out = {}
    rows = []
    for i, first in enumerate(setups):
        for second in setups[i + 1 :]:
            a = [record["setups"][first][p]["elo"] for p in personas]
            b = [record["setups"][second][p]["elo"] for p in personas]
            r = float(np.corrcoef(a, b)[0, 1])
            out[f"{first}-{second}"] = r
            rows.append([f"{first} against {second}", f"{r:+.2f}"])
    return table(["personas' Elo", "r"], rows), out


def main() -> None:
    record = json.loads(Path(sys.argv[1]).read_text())
    setups = list(record["setups"])
    result = {}
    print("## Box edges (one row at u = 0 or 1, the rest at the default)\n")
    rows = []
    for name in SAMPLED:
        cells = [name]
        for setup in setups:
            for side in ("low", "high"):
                entry = record["setups"][setup].get(f"{name}@{side}")
                cells.append(
                    f"{entry['elo']:.0f} +- {entry['elo_error']:.0f}"
                    if entry and entry["elo"] is not None
                    else "-"
                )
        rows.append(cells)
    header = ["row"] + sum([[f"{s} u=0", f"{s} u=1"] for s in setups], [])
    print(table(header, rows) + "\n")
    print("## The population (64 personas, spread 1)\n")
    text, result["population"] = population(record)
    print(text + "\n")
    text, result["correlations"] = correlations(record)
    print(text + "\n")
    fitted = fits(record)
    result["fits"] = fitted
    print("## Linear fit: Elo change from u = 0 to u = 1 per row (weighted least squares)\n")
    rows = []
    for k, name in enumerate(SAMPLED):
        rows.append(
            [name]
            + [
                f"{fitted[s]['slope'][k]:+.0f} +- {fitted[s]['slope_error'][k]:.0f}"
                for s in setups
                if s in fitted
            ]
        )
    rows.append(
        ["R^2 / leave-one-out R^2 / points"]
        + [
            f"{fitted[s]['r2']:.2f} / {fitted[s]['q2']:.2f} / {fitted[s]['points']}"
            for s in setups
            if s in fitted
        ]
    )
    print(table(["row"] + [s for s in setups if s in fitted], rows) + "\n")
    print(
        "## Quadratic fit: slope at the centre (Elo per box) / curvature (Elo per box^2) / "
        "best u on [0, 1]\n"
    )
    rows = []
    for k, name in enumerate(SAMPLED):
        rows.append(
            [name]
            + [
                f"{fitted[s]['quadratic_slope'][k]:+.0f} / {fitted[s]['curvature'][k]:+.0f} "
                f"+- {fitted[s]['curvature_error'][k]:.0f} / {fitted[s]['best_u'][k]:.2f}"
                for s in setups
                if s in fitted
            ]
        )
    rows.append(
        ["R^2 / leave-one-out R^2"]
        + [
            f"{fitted[s]['quadratic_r2']:.2f} / {fitted[s]['quadratic_q2']:.2f}"
            for s in setups
            if s in fitted
        ]
    )
    print(table(["row"] + [s for s in setups if s in fitted], rows) + "\n")
    # Size dependence: slope differences between sizes, in standard errors.
    print("## Size dependence: linear slope differences (Elo per box), z = difference / SE\n")
    pairs = [(a, b) for i, a in enumerate(setups) for b in setups[i + 1 :] if a in fitted]
    rows, dependence = [], {}
    for k, name in enumerate(SAMPLED):
        cells, entry = [name], {}
        for a, b in pairs:
            difference = fitted[b]["slope"][k] - fitted[a]["slope"][k]
            error = math.hypot(fitted[a]["slope_error"][k], fitted[b]["slope_error"][k])
            entry[f"{a}->{b}"] = (difference, difference / error)
            cells.append(f"{difference:+.0f} (z {difference / error:+.1f})")
        signs = {np.sign(fitted[s]["slope"][k]) for s in setups if s in fitted}
        strong = [
            s for s in setups if abs(fitted[s]["slope"][k]) > 2.5 * fitted[s]["slope_error"][k]
        ]
        flip = (
            len(signs) > 1
            and any(fitted[s]["slope"][k] > 2.5 * fitted[s]["slope_error"][k] for s in setups)
            and any(fitted[s]["slope"][k] < -2.5 * fitted[s]["slope_error"][k] for s in setups)
        )
        entry["flip"] = bool(flip)
        entry["strong_in"] = strong
        dependence[name] = entry
        cells.append("yes" if flip else "")
        rows.append(cells)
    result["dependence"] = dependence
    print(
        table(["row"] + [f"{b} minus {a}" for a, b in pairs] + ["sign flips (2.5 SE)"], rows) + "\n"
    )
    # Style per size.
    result["style"] = {}
    for setup in setups:
        entries = record["setups"][setup]
        personas = [p for p in entries if p.startswith("persona") and entries[p]["style"]]
        yard = entries["yardstick"]["style"] or {}
        elo = np.array([entries[p]["elo"] for p in personas])
        rows, record_style = [], {}
        for key in STYLE + ("hits_per_minute",):

            def get(style):
                if key == "hits_per_minute":
                    return style["hits"] / max(style["minutes"], 1e-9)
                return style[key]

            values = np.array([get(entries[p]["style"]) for p in personas], dtype=float)
            ok = np.isfinite(values)
            r = float(np.corrcoef(values[ok], elo[ok])[0, 1]) if ok.sum() > 3 else math.nan
            q = np.quantile(values[ok], [0, 0.5, 1.0])
            record_style[key] = {
                "yardstick": get(yard) if yard else None,
                "min": q[0],
                "median": q[1],
                "max": q[2],
                "sd": float(values[ok].std(ddof=1)),
                "r": r,
            }
            rows.append(
                [
                    key,
                    f"{get(yard):.3f}" if yard else "-",
                    f"{q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}",
                    f"{values[ok].std(ddof=1):.3f}",
                    f"{r:+.2f}",
                ]
            )
        result["style"][setup] = record_style
        print(f"## Style, {setup} (64 personas against the yardstick)\n")
        print(table([setup, "yardstick", "min / median / max", "sd", "r with Elo"], rows) + "\n")
    (HERE / "analysis-2.json").write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
