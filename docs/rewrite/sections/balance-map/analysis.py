"""The balance map's analysis: the map's heat map, a model from the handles to each searched
condition's kill share (scikit-learn gradient boosting, cross-validated), its partial
dependence on every handle, and proposals for the map's empty cells from the model.

    PYTHONPATH=$PWD/src:$PWD uv run --no-sync python \
        docs/rewrite/sections/balance-map/analysis.py RUN_DIR OUT_DIR [--proposals 16]

RUN_DIR: a `bench.balance_map search` run (candidates.jsonl, evaluations.jsonl). Writes into
OUT_DIR: map.png, partial_dependence.png, model.pkl, model.json (cross-validated accuracy,
the handles' importance), map_table.md and proposals.json (sweep pairs for the model's best
card in each empty cell, to be played).
"""

import argparse
import json
import math
import pathlib
import pickle

import numpy

from bench import balance_map

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4, fixed order
SEQUENTIAL = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
SURFACE, INK, MUTED = "#fcfcfb", "#1f1f1e", "#6b6b68"
UNITS = {
    "sustained_turn_rate": "rad/s",
    "nose_reversal_time": "s",
    "low_speed_acceleration": "m/s²",
    "climb_power": "s (at the v23 store)",
    "cruise_speed": "m/s",
    "stall_speed": "m/s",
    "g_limit": "g",
    "max_shield": "",
    "hit_damage": "",
    "gun_range": "m",
    "store": "J",
    "shot_cost": "J",
    "dive_acceleration": "m/s²",
}


def load(run: pathlib.Path) -> balance_map.Records:
    return balance_map.Records(run)


def dataset(records: balance_map.Records):
    """(coordinates (N, 13) of the handles as solved, kill shares (N, C), weights (N, C),
    ids): every played card with all searched conditions."""
    rows, shares, weights, ids = [], [], [], []
    for candidate in records.candidates:
        if "card" not in candidate:
            continue
        pooled = balance_map.pooled(records.results_of.get(candidate["id"], []))
        if any(c not in pooled for c in balance_map.SEARCHED):
            continue
        rows.append(balance_map.encode(candidate["handles"]))
        shares.append([pooled[c][0] for c in balance_map.SEARCHED])
        weights.append([1.0 / max(pooled[c][1], 1e-3) ** 2 for c in balance_map.SEARCHED])
        ids.append(candidate["id"])
    return numpy.array(rows), numpy.array(shares), numpy.array(weights), ids


def logit(p):
    p = numpy.clip(p, 1e-4, 1 - 1e-4)
    return numpy.log(p / (1 - p))


def fit(x, shares, weights, folds: int = 5):
    """One gradient-boosting model a condition on logit(kill share), sample-weighted by the
    inverse variance (normalised); its cross-validated R² and the noise's own share of the
    variance (the ceiling R² can reach)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    models, report = [], {}
    for column, condition in enumerate(balance_map.SEARCHED):
        y = logit(shares[:, column])
        # the logit's variance from the share's: se^2 / (p (1 - p))^2
        p = numpy.clip(shares[:, column], 1e-3, 1 - 1e-3)
        variance = (1.0 / weights[:, column]) / (p * (1 - p)) ** 2
        w = 1.0 / variance
        w = w / w.mean()

        def model():
            return HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=0,
            )

        predicted = numpy.zeros_like(y)
        for train, test in KFold(folds, shuffle=True, random_state=0).split(x):
            predicted[test] = (
                model().fit(x[train], y[train], sample_weight=w[train]).predict(x[test])
            )
        mean = numpy.average(y, weights=w)
        total = numpy.average((y - mean) ** 2, weights=w)
        r2 = 1.0 - numpy.average((y - predicted) ** 2, weights=w) / total
        noise = numpy.average(variance, weights=w) / total
        models.append(model().fit(x, y, sample_weight=w))
        win = numpy.array([balance_map.win_equivalent(s, condition) for s in shares[:, column]])
        win_hat = 100.0 / (1.0 + numpy.exp(-balance_map.SLOPES[condition] * predicted))
        report[condition] = dict(
            cards=len(y),
            r2_cross_validated=round(float(r2), 3),
            noise_share_of_variance=round(float(noise), 3),
            win_equivalent_rmse=round(float(numpy.sqrt(numpy.mean((win - win_hat) ** 2))), 2),
        )
    return models, report


def objective_from_logits(logits: numpy.ndarray) -> numpy.ndarray:
    """The objective from predicted logits (N, C)."""
    slopes = numpy.array([balance_map.SLOPES[c] for c in balance_map.SEARCHED])
    win = 100.0 / (1.0 + numpy.exp(-slopes * logits))
    return 50.0 - numpy.abs(win - 50.0).max(-1)


def partial_dependence_plot(models, x, path: pathlib.Path) -> dict:
    """Each handle's partial dependence (the model's average logit over the data as the
    handle sweeps its bounds), as win-rate-equivalent points, one panel a handle, one line a
    condition; returns each handle's swing (max - min) by condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import partial_dependence

    grid = numpy.linspace(0.0, 1.0, 21)
    columns = 4
    rows = math.ceil(len(balance_map.SPACE) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(13, 2.6 * rows), sharey=True)
    figure.patch.set_facecolor(SURFACE)
    swings: dict[str, dict] = {}
    for number, axis in enumerate(balance_map.SPACE):
        panel = axes.flat[number]
        panel.set_facecolor(SURFACE)
        values = [axis.value(g) for g in grid]
        swings[axis.handle] = {}
        for column, (condition, model) in enumerate(zip(balance_map.SEARCHED, models, strict=True)):
            result = partial_dependence(
                model,
                x,
                [number],
                grid_resolution=21,
                custom_values={number: grid},
                kind="average",
                method="brute",
            )
            average = result["average"][0]
            win = 100.0 / (1.0 + numpy.exp(-balance_map.SLOPES[condition] * average))
            swings[axis.handle][condition] = round(float(win.max() - win.min()), 1)
            panel.plot(values, win, color=SERIES[column], linewidth=2, label=condition)
        panel.axhline(50.0, color=MUTED, linewidth=0.8, linestyle=":")
        unit = UNITS.get(axis.handle, "")
        panel.set_title(axis.handle.replace("_", " "), fontsize=9, color=INK)
        panel.set_xlabel(unit, fontsize=8, color=MUTED)
        panel.tick_params(labelsize=7, colors=MUTED)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
        panel.grid(alpha=0.25, linewidth=0.5)
    for panel in list(axes.flat)[len(balance_map.SPACE) :]:
        panel.set_visible(False)
    for row in range(rows):
        axes[row, 0].set_ylabel("win-rate equivalent (%)", fontsize=8, color=MUTED)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right", ncol=1, frameon=False, fontsize=9)
    figure.suptitle(
        "Partial dependence: each handle's effect on the candidate team's balance against v23 "
        "Jujak (model average over the evaluated cards)",
        fontsize=11,
        color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(figure)
    return swings


def map_plot(cells: dict, path: pathlib.Path, dims=(10, 10)) -> None:
    """The map: each cell's best card's objective lower bound (sequential blue; darker more
    balanced), its games annotated; empty cells blank."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    grid = numpy.full(dims, numpy.nan)
    for key, cell in cells.items():
        i, j = map(int, key.split(","))
        grid[j, i] = max(cell["lower"], 0.0)
    colours = LinearSegmentedColormap.from_list("blue", SEQUENTIAL)
    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    image = axis.imshow(
        grid, origin="lower", cmap=colours, vmin=20, vmax=50, extent=(0, 1, 0, 1), aspect="auto"
    )
    for key, cell in cells.items():
        i, j = map(int, key.split(","))
        dark = cell["lower"] > 33
        axis.text(
            (i + 0.5) / dims[0],
            (j + 0.5) / dims[1],
            f"{cell['objective']:.0f}\n{cell['games']}",
            ha="center",
            va="center",
            fontsize=6.5,
            color="#ffffff" if dark else INK,
        )
    axis.set_xlabel("agility (0.5 turn + 0.25 low-speed acceleration + 0.25 nose quickness)")
    axis.set_ylabel("toughness x reach (0.5 shield + 0.5 gun range)")
    axis.set_xticks(numpy.linspace(0, 1, dims[0] + 1))
    axis.set_yticks(numpy.linspace(0, 1, dims[1] + 1))
    axis.tick_params(labelsize=7, colors=MUTED)
    axis.grid(color=SURFACE, linewidth=1.5)
    bar = figure.colorbar(image, ax=axis)
    bar.set_label("objective's lower bound (50 = even everywhere; 45 = 45-55 worst)")
    axis.set_title(
        "Balance map: best card a cell against v23 Jujak\n(cell text: objective, "
        "games a condition)",
        fontsize=10,
        color=INK,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(figure)


def proposals(models, records: balance_map.Records, cells: dict, count: int, samples: int = 200000):
    """The model's best card for each empty cell (sampled coordinates in the cell, the
    highest predicted objective), up to `count` cells, nearest the filled ones first."""
    rng = numpy.random.default_rng(0)
    x = rng.random((samples, len(balance_map.SPACE)))
    base = dict(balance_map.reference_ships()[balance_map.REFERENCE].handles)
    # the handles as decoded (damage repair; the low-speed repair is the solver's, later)
    decoded = numpy.array([balance_map.encode(balance_map.decode(row, base)) for row in x])
    found = numpy.array([balance_map.measures(row) for row in decoded])
    logits = numpy.stack([m.predict(decoded) for m in models], -1)
    objective = objective_from_logits(logits)
    out = []
    for i in range(10):
        for j in range(10):
            if f"{i},{j}" in cells:
                continue
            inside = (numpy.minimum((found[:, 0] * 10).astype(int), 9) == i) & (
                numpy.minimum((found[:, 1] * 10).astype(int), 9) == j
            )
            if not inside.any():
                continue
            best = numpy.flatnonzero(inside)[objective[inside].argmax()]
            out.append(
                dict(
                    cell=f"{i},{j}",
                    predicted=round(float(objective[best]), 2),
                    coordinates=decoded[best].round(6).tolist(),
                )
            )
    out.sort(key=lambda r: -r["predicted"])
    return out[:count]


def conversion_refit(records: balance_map.Records) -> dict:
    """balance-eval's conversion refit on this run's own evaluations (fog on, fields off):
    logit(score) = a + b logit(kill share) by condition, weighted least squares (the score's
    logit variance), every evaluation a point; with the kill shares at 45 / 50 / 55%."""
    out = {}
    conditions = sorted({e["condition"] for e in records.evaluations})
    for condition in conditions:
        rows = [e for e in records.evaluations if e["condition"] == condition]
        k = numpy.array([r["kill_share"][0] for r in rows])
        score = numpy.array([r["score"][0] for r in rows])
        error = numpy.array([max(r["score"][1], 1e-3) for r in rows])
        keep = (k > 0.02) & (k < 0.98) & (score > 0.02) & (score < 0.98)
        if keep.sum() < 8:
            continue
        x, y = logit(k[keep]), logit(score[keep])
        w = (score[keep] * (1 - score[keep]) / error[keep]) ** 2
        design = numpy.stack([numpy.ones_like(x), x], -1)
        a, b = numpy.linalg.solve(design.T @ (design * w[:, None]), design.T @ (w * y))
        residual = y - (a + b * x)
        chi2 = float((w * residual**2).sum() / max(len(x) - 2, 1))
        band = [float(1 / (1 + numpy.exp(-(logit(t) - a) / b))) for t in (0.45, 0.5, 0.55)]
        out[condition] = dict(
            points=int(keep.sum()),
            intercept=round(float(a), 3),
            slope=round(float(b), 3),
            chi2_per_dof=round(chi2, 2),
            kill_share_at_45_50_55=[round(v, 4) for v in band],
            slope_used=balance_map.SLOPES.get(condition),
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--proposals", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = load(args.run)
    cells = records.map()
    map_plot(cells, args.out / "map.png")
    x, shares, weights, ids = dataset(records)
    models, report = fit(x, shares, weights)
    swings = partial_dependence_plot(models, x, args.out / "partial_dependence.png")
    with (args.out / "model.pkl").open("wb") as handle:
        pickle.dump(
            dict(models=models, conditions=balance_map.SEARCHED, features=balance_map.NAMES), handle
        )
    proposed = proposals(models, records, cells, args.proposals)
    refit = conversion_refit(records)
    (args.out / "model.json").write_text(
        json.dumps(dict(accuracy=report, swings=swings, conversion_refit=refit), indent=1)
    )
    print(json.dumps(refit, indent=1))
    (args.out / "proposals.json").write_text(json.dumps(proposed, indent=1))
    lines = [
        "| cell | id | objective | lower | games | worst | win-equivalent by condition |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in sorted(cells, key=lambda k: tuple(map(int, k.split(",")))):
        c = cells[key]
        lines.append(
            f"| {key} | {c['id']} | {c['objective']:.1f} | {c['lower']:.1f} | "
            f"{c['games']} | {c['worst']} | {c['win_equivalent']} |"
        )
    (args.out / "map_table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=1))
    print(f"{len(ids)} cards modelled; {len(cells)} cells; {len(proposed)} proposals")


if __name__ == "__main__":
    main()
