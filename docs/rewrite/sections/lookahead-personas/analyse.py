"""Tables from divergence.py's and skill.py's outputs (CPU only).

    analyse.py DATA SKILL.json > tables.md      # also writes analysis.json beside this file

Divergence: the one-at-a-time JS table, Sobol indices of JS from the yardstick (SALib, first
order and total with their 95% intervals), the mean one-row redraw JS(A, AB_i) of the Saltelli
sample (the behaviour change of redrawing that row alone, spread 1), the population's label
blur per spread. Skill: Elo of the edge agents and the population, the spread of the
population's Elo (observed and with the measurement error removed), a least-squares fit of
Elo on the unit coordinates (standardised effect per row, its standard error), the style
view per agent, and the degenerate-persona checks.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import divergence  # noqa: E402

from lean.scripted.parameters import PARAMETERS, SAMPLED  # noqa: E402

HERE = Path(__file__).resolve().parent
SETUPS = tuple(divergence.SETUPS)
STYLE = ("turning", "two_circle", "store", "speed", "shots", "hit_range", "climb", "dive")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
    return "\n".join(out)


def f(value, digits=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def oat(data: Path, result: dict) -> str:
    d = np.load(data / "oat.npz")
    labels = d["labels"]
    rows, record = [], {}
    for column, name in enumerate(SAMPLED):
        m = labels[:, 0] == column
        row = PARAMETERS[name]
        low, high = (
            (row.default * math.exp(-row.width), row.default * math.exp(row.width))
            if row.law == "log"
            else (row.default - row.width, row.default + row.width)
        )
        entry = {}
        cells = [name, f"{row.default:g} [{low:.3g}, {high:.3g}]"]
        for setup in SETUPS:
            j = d[f"{setup}_joint"][m]
            entry[setup] = {
                key: d[f"{setup}_{key}"][m].tolist()
                for key in (
                    "joint",
                    "stick",
                    "power",
                    "fire",
                    "fire_share",
                    "straight",
                    "dive",
                    "climb",
                    "entropy",
                )
            }
            cells += [f"{j[0]:.4f} / {j[1]:.4f} / {j[2]:.4f} / {j[3]:.4f}"]
            heads = [d[f"{setup}_{key}"][m] for key in ("stick", "power", "fire")]
            share = [100.0 * h[[0, 3]].max() / max(j[[0, 3]].max(), 1e-12) for h in heads]
            cells += ["/".join(f"{s:.0f}" for s in share)]
        record[name] = entry
        rows.append(cells)
    result["oat"] = record
    header = ["row", "default [spread-1 box]"]
    for setup in SETUPS:
        header += [f"{setup} JS u=0 / .25 / .75 / 1", f"{setup} stick/power/fire % of joint"]
    return table(header, rows)


def sobol(data: Path, base: int, result: dict) -> str:
    from SALib.analyze import sobol as sobol_analyze

    path = data / f"saltelli{base}.npz"
    if not path.exists():
        return "(Saltelli sample not complete)"
    d = np.load(path)
    problem = {
        "num_vars": len(SAMPLED),
        "names": list(SAMPLED),
        "bounds": [[0.0, 1.0]] * len(SAMPLED),
    }
    record, columns = {}, {}
    for setup in SETUPS:
        y = d[f"{setup}_yard_joint"].reshape(-1)  # (base x (D + 2)) in the sample's order
        si = sobol_analyze.analyze(problem, y, calc_second_order=False, seed=1)
        redraw = d[f"{setup}_redraw_joint"]  # (base, D + 1): AB_1 .. AB_D, B against A
        mean, error = redraw.mean(0), redraw.std(0, ddof=1) / math.sqrt(redraw.shape[0])
        heads = {key: d[f"{setup}_redraw_{key}"].mean(0) for key in ("stick", "power", "fire")}
        record[setup] = {
            "S1": si["S1"].tolist(),
            "S1_conf": si["S1_conf"].tolist(),
            "ST": si["ST"].tolist(),
            "ST_conf": si["ST_conf"].tolist(),
            "redraw": mean[:-1].tolist(),
            "redraw_error": error[:-1].tolist(),
            "redraw_all": float(mean[-1]),
            "redraw_all_error": float(error[-1]),
            "redraw_heads": {key: value[:-1].tolist() for key, value in heads.items()},
            "from_yardstick_mean": float(y.mean()),
            "base": int(redraw.shape[0]),
        }
        columns[setup] = record[setup]
    result["sobol"] = record
    order = np.argsort(-np.array(columns[SETUPS[-1]]["redraw"]))
    rows = []
    for k in order:
        cells = [SAMPLED[k]]
        for setup in SETUPS:
            r = columns[setup]
            share = r["redraw"][k] / r["redraw_all"]
            cells += [
                f"{r['redraw'][k]:.4f} +- {r['redraw_error'][k]:.4f} ({100 * share:.0f}%)",
                f"{r['ST'][k]:.3f} +- {r['ST_conf'][k]:.3f}",
                f"{r['S1'][k]:.3f} +- {r['S1_conf'][k]:.3f}",
            ]
        rows.append(cells)
    rows.append(
        ["every row redrawn (A against B)"]
        + sum(
            [
                [f"{columns[s]['redraw_all']:.4f} +- {columns[s]['redraw_all_error']:.4f}", "", ""]
                for s in SETUPS
            ],
            [],
        )
    )
    header = ["row"]
    for setup in SETUPS:
        header += [f"{setup} redraw JS (share of all)", f"{setup} ST", f"{setup} S1"]
    return table(header, rows)


def population(data: Path, result: dict) -> str:
    rows, record = [], {}
    for spread in divergence.SPREADS:
        for setup in SETUPS:
            p = np.load(data / f"population_{setup}_{spread}.npz")
            pairs = p["pairwise"][np.triu_indices(divergence.POPULATION, 1)]
            mixture, members = float(p["mixture_entropy"]), float(p["member_entropy"].mean())
            yard = float(p["yardstick_entropy"])
            entry = {
                "yardstick_entropy": yard,
                "member_entropy": members,
                "mixture_entropy": mixture,
                "generalized_js": mixture - members,
                "member_from_yardstick": float(p["from_yardstick"].mean()),
                "member_from_yardstick_max": float(p["from_yardstick"].max()),
                "pairwise_mean": float(pairs.mean()),
                "pairwise_max": float(pairs.max()),
                "mixture_from_yardstick": float(p["mixture_from_yardstick"]),
            }
            record[f"{setup}_{spread}"] = entry
            rows.append(
                [
                    spread,
                    setup,
                    f(yard, 3),
                    f(members, 3),
                    f(mixture, 3),
                    f"{mixture - members:.4f} ({100 * (mixture - members) / members:.1f}%)",
                    f(entry["member_from_yardstick"]),
                    f(entry["member_from_yardstick_max"]),
                    f(entry["pairwise_mean"]),
                    f(entry["mixture_from_yardstick"], 5),
                ]
            )
    result["population"] = record
    return table(
        [
            "spread",
            "setup",
            "yardstick H",
            "members' mean H",
            "mixture H",
            "blur: mixture H - mean H",
            "JS member-yardstick mean",
            "max",
            "JS pairwise mean",
            "JS mixture-yardstick",
        ],
        rows,
    )


def skill(path: Path, result: dict) -> list[str]:
    record = json.loads(path.read_text())
    labels = [agent["label"] for agent in record["agents"]]
    units = {agent["label"]: agent["unit"] for agent in record["agents"]}
    out = []
    # Edge agents.
    rows = []
    edge = {}
    for name in SAMPLED:
        cells = [name]
        for setup in SETUPS:
            entries = record["setups"][setup]
            low, high = entries.get(f"{name}@low"), entries.get(f"{name}@high")
            cells += [
                f"{low['elo']:.0f} +- {low['elo_error']:.0f}" if low and low["elo"] else "-",
                f"{high['elo']:.0f} +- {high['elo_error']:.0f}" if high and high["elo"] else "-",
            ]
            edge.setdefault(setup, {})[name] = (low and low["elo"], high and high["elo"])
        rows.append(cells)
    header = ["row"] + sum([[f"{s} Elo at u=0", f"{s} Elo at u=1"] for s in SETUPS], [])
    out.append(table(header, rows))
    # Population.
    summary_rows, fits = [], {}
    personas = [label for label in labels if label.startswith("persona")]
    for setup in SETUPS:
        entries = record["setups"][setup]
        elo = np.array(
            [entries[label]["elo"] for label in personas if entries[label]["elo"] is not None]
        )
        err = np.array(
            [entries[label]["elo_error"] for label in personas if entries[label]["elo"] is not None]
        )
        games = sum(entries[label].get("games", 0) for label in labels[1:])
        if len(elo) < 4:
            continue
        observed_sd = float(elo.std(ddof=1))
        true_sd = float(math.sqrt(max(observed_sd**2 - float((err**2).mean()), 0.0)))
        quantiles = np.quantile(elo, [0.0, 0.1, 0.5, 0.9, 1.0])
        best = personas[int(np.argmax([entries[label]["elo"] for label in personas]))]
        summary_rows.append(
            [
                setup,
                len(elo),
                games,
                f"{elo.mean():.0f}",
                f"{observed_sd:.0f}",
                f"{true_sd:.0f}",
                f"{err.mean():.0f}",
                " / ".join(f"{q:.0f}" for q in quantiles),
                int((elo > 1000).sum()),
                best,
            ]
        )
        # Least squares of Elo on the unit coordinates (centred at 0.5), personas and edges.
        xs, ys, ws = [], [], []
        for label in labels[1:]:
            entry = entries[label]
            if entry["elo"] is None:
                continue
            xs.append(np.array(units[label]) - 0.5)
            ys.append(entry["elo"])
            ws.append(1.0 / max(entry["elo_error"], 1.0) ** 2)
        x = np.column_stack([np.ones(len(xs)), np.array(xs)])
        y, w = np.array(ys), np.array(ws)
        sw = np.sqrt(w)
        coefficients, *_ = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)
        residual = y - x @ coefficients
        dof = max(len(y) - x.shape[1], 1)
        scale = float((w * residual**2).sum() / dof)
        covariance = np.linalg.inv((x * w[:, None]).T @ x) * scale
        errors = np.sqrt(np.diag(covariance))
        r2 = 1.0 - float((w * residual**2).sum() / (w * (y - np.average(y, weights=w)) ** 2).sum())
        fits[setup] = {
            "intercept": float(coefficients[0]),
            "slope": coefficients[1:].tolist(),
            "slope_error": errors[1:].tolist(),
            "r2": r2,
            "points": len(y),
            "population_mean": float(elo.mean()),
            "population_sd": observed_sd,
            "population_sd_true": true_sd,
            "mean_error": float(err.mean()),
            "quantiles": quantiles.tolist(),
            "games": games,
        }
    out.append(
        table(
            [
                "setup",
                "personas",
                "games (all agents)",
                "mean Elo",
                "sd observed",
                "sd without error",
                "mean error",
                "min / 10% / median / 90% / max",
                "above 1000",
                "best",
            ],
            summary_rows,
        )
    )
    rows = []
    for k, name in enumerate(SAMPLED):
        cells = [name]
        for setup in SETUPS:
            if setup in fits:
                cells.append(
                    f"{fits[setup]['slope'][k]:+.0f} +- {fits[setup]['slope_error'][k]:.0f}"
                )
            else:
                cells.append("-")
        rows.append(cells)
    rows.append(
        ["R^2 / points"]
        + [f"{fits[s]['r2']:.2f} / {fits[s]['points']}" if s in fits else "-" for s in SETUPS]
    )
    out.append(table(["row (Elo change from u=0 to u=1, linear fit)"] + list(SETUPS), rows))
    # Style and degenerate checks.
    style_rows, degenerate = [], []
    for setup in SETUPS:
        entries = record["setups"][setup]
        yard_style = entries["yardstick"]["style"] or {}
        for label in labels:
            entry = entries[label]
            s = entry["style"]
            if s is None:
                continue
            flags = []
            if yard_style:
                if s["shots"] is not None and s["shots"] < 0.5 * yard_style["shots"]:
                    flags.append("fires under half the yardstick's shots")
                if s["speed"] is not None and s["speed"] < 0.8 * yard_style["speed"]:
                    flags.append(
                        f"slow ({s['speed']:.2f} corner, yardstick {yard_style['speed']:.2f})"
                    )
                if s["turning"] is not None and s["turning"] > 1.5 * yard_style["turning"]:
                    flags.append("turning fight 1.5x the yardstick's")
                if s["store"] is not None and s["store"] < 0.15:
                    flags.append("store nearly empty")
                if (
                    s["hit_range"] is not None
                    and yard_style["hit_range"]
                    and s["hit_range"] > 2.0 * yard_style["hit_range"]
                ):
                    flags.append("hits from over twice the yardstick's range")
            if entry.get("games") and entry["draws"] / entry["games"] > 0.2:
                flags.append(f"draws {entry['draws'] / entry['games']:.0%}")
            if entry.get("elo") is not None and entry["elo"] < 800:
                flags.append(f"Elo {entry['elo']:.0f}")
            if flags:
                degenerate.append([setup, label, "; ".join(flags)])
            style_rows.append(
                [setup, label, f"{entry['elo']:.0f}" if entry.get("elo") else "-", s["games"]]
                + [f(s[key], 3) if s[key] is not None else "-" for key in STYLE]
                + [f(s["hits"] / max(s["minutes"], 1e-9), 2)]
            )
    out.append(
        table(["setup", "agent", "Elo", "games"] + list(STYLE) + ["hits/ship-min"], style_rows)
    )
    out.append(
        table(["setup", "agent", "flag"], degenerate) if degenerate else "(no agent flagged)"
    )
    result["skill"] = {"fits": fits, "edge": edge, "degenerate": degenerate}
    return out


def main() -> None:
    data = Path(sys.argv[1]).expanduser()
    result = {}
    print("## One at a time\n")
    print(oat(data, result))
    print("\n## Saltelli\n")
    print(sobol(data, 256, result))
    print("\n## Population label blur\n")
    print(population(data, result))
    if len(sys.argv) > 2 and Path(sys.argv[2]).exists():
        for part in skill(Path(sys.argv[2]), result):
            print("\n## Skill\n")
            print(part)
    (HERE / "analysis.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
