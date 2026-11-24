"""The divergence study's analysis (docs/rewrite/scripted-divergence.md): reads divergence.py's
results in DATA, writes divergence.json beside this file and the embedding figure.

    divergence_analyse.py DATA [--base 512] [--count 300]

Sobol indices of the JS divergence from the yardstick two ways: direct (Saltelli sample, SALib)
and from a surrogate fitted to the 4,000 screening configurations (the screening's recipe).
Families: classical MDS of sqrt(JS) (a metric) and average-linkage clusters, the number by
silhouette.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from lean.scripted.parameters import SAMPLED

HERE = Path(__file__).resolve().parent
LN2 = math.log(2.0)
REFERENCES = 9  # yardstick, cma, bfm, persona1..6 lead the screen and pairwise records
FLAT = ("weakest_weight", "turning_floor", "engage_range", "defend_range", "cash_angle", "relax_angle",
        "scissors_range", "switch_gain", "overshoot_range", "overshoot_rate", "overshoot_crossing",
        "yo_yo_scale", "merge_speed", "speed_floor", "spend_share", "spend_need", "spend_range",
        "speed_band", "power_scale", "turn", "level_three", "boundary_margin")
SHARES = ("straight", "full_stick", "dive", "climb", "fire")


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {text}", file=sys.stderr, flush=True)


def load(path: Path) -> dict:
    with np.load(path) as record:
        return {key: record[key] for key in record.files}


def problem() -> dict:
    return {"num_vars": len(SAMPLED), "names": list(SAMPLED), "bounds": [[0.0, 1.0]] * len(SAMPLED)}


def direct_indices(record: dict) -> dict:
    from SALib.analyze import sobol as sobol_analyze
    out = {}
    for key in ("joint", "stick", "power", "fire"):
        indices = sobol_analyze.analyze(problem(), record[key].astype(np.float64), calc_second_order=False, seed=0)
        out[key] = {name: {"S1": float(indices["S1"][i]), "S1_conf": float(indices["S1_conf"][i]),
                           "ST": float(indices["ST"][i]), "ST_conf": float(indices["ST_conf"][i])}
                    for i, name in enumerate(SAMPLED)}
    return out


def r2(target, prediction) -> float:
    return float(1.0 - np.sum((target - prediction) ** 2) / np.sum((target - target.mean()) ** 2))


def surrogate_indices(x, y) -> dict:
    """Gradient-boosted trees, settings by 5-fold CV on 80%, held-out R^2 on 20%, Sobol
    indices of the refitted surrogate (8,192 base points)."""
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_predict, train_test_split

    grid = [dict(n_estimators=n, learning_rate=rate, max_depth=depth, subsample=0.8, min_samples_leaf=leaf)
            for n, rate in ((800, 0.025),) for depth in (3, 4) for leaf in (10,)]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    folds = KFold(5, shuffle=True, random_state=0)
    scores = []
    for settings in grid:
        prediction = cross_val_predict(GradientBoostingRegressor(random_state=0, **settings), x_train, y_train, cv=folds)
        scores.append((r2(y_train, prediction), settings))
    scores.sort(key=lambda item: -item[0])
    settings = scores[0][1]
    held_out = r2(y_test, GradientBoostingRegressor(random_state=0, **settings).fit(x_train, y_train).predict(x_test))
    model = GradientBoostingRegressor(random_state=0, **settings).fit(x, y)
    points = sobol_sample.sample(problem(), 8192, calc_second_order=False, seed=0)
    indices = sobol_analyze.analyze(problem(), model.predict(points), calc_second_order=False, seed=0)
    return {"r2_held_out": held_out, "settings": settings, "sum_S1": float(np.sum(indices["S1"])),
            "rows": {name: {"S1": float(indices["S1"][i]), "ST": float(indices["ST"][i])}
                     for i, name in enumerate(SAMPLED)}}


def spearman(a, b) -> float:
    ranks = [np.argsort(np.argsort(item)) for item in (a, b)]
    return float(np.corrcoef(*ranks)[0, 1])


def families(matrix: np.ndarray) -> dict:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score
    distance = np.sqrt(np.clip(matrix, 0.0, None))
    np.fill_diagonal(distance, 0.0)
    n = len(distance)
    centring = np.eye(n) - 1.0 / n
    gram = -0.5 * centring @ (distance ** 2) @ centring
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    positive = eigenvalues.clip(min=0.0)
    coordinates = eigenvectors[:, :2] * np.sqrt(positive[:2])
    tree = linkage(squareform(distance, checks=False), "average")
    silhouettes, best = {}, None
    for k in range(2, 9):
        labels = fcluster(tree, k, "maxclust")
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(distance, labels, metric="precomputed"))
        silhouettes[k] = score
        if best is None or score > best[0]:
            best = (score, k, labels)
    return {"coordinates": coordinates, "explained": (positive[:2] / positive.sum()).tolist(),
            "silhouettes": silhouettes, "k": best[1], "labels": best[2], "silhouette": best[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("--base", type=int, default=512)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--figure", default=None, help="write the embedding figure here (after an analysis)")
    parser.add_argument("--share", default="full_stick")
    arguments = parser.parse_args()
    data = Path(arguments.data).expanduser()
    if arguments.figure:
        figure(data, arguments.count, Path(arguments.figure), arguments.share)
        return
    out = {"ln2": LN2}
    screen = load(data / "screen.npz")
    names = [str(name) for name in screen["names"]]
    keys = ("joint", "joint_error", "stick", "power", "fire", "straight", "full_stick", "dive", "climb", "fire_share")
    out["references"] = {names[i]: {key: float(screen[key][i]) for key in keys} for i in range(REFERENCES)}
    js = screen["joint"][REFERENCES:]
    units = screen["units"]
    skill = np.array([entry["skill"] for entry in json.loads((HERE / "skills.json").read_text())["screen"]])
    out["screen"] = {
        "count": int(len(js)), "mean": float(js.mean()), "std": float(js.std()),
        "quantiles": {str(q): float(np.quantile(js, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "max": float(js.max()), "median_error": float(np.median(screen["joint_error"][REFERENCES:])),
        "heads_mean": {key: float(screen[key][REFERENCES:].mean()) for key in ("stick", "power", "fire")},
        "skill_pearson": float(np.corrcoef(js, skill)[0, 1]), "skill_spearman": spearman(js, skill),
        "shares_mean": {key: float(screen[key][REFERENCES:].mean()) for key in keys[5:]},
    }
    log("surrogate")
    out["surrogate"] = surrogate_indices(units, js)
    path = data / f"saltelli{arguments.base}.npz"
    if path.exists():
        record = load(path)
        out["direct"] = direct_indices(record)
        out["direct_configurations"] = int(len(record["joint"]))
        out["direct_base"] = arguments.base
    path = data / "elementary1024.npz"
    if path.exists():
        record = load(path)
        out["elementary_direct"] = direct_indices({key: record[f"yardstick_{key}"] for key in ("joint", "stick", "power", "fire")})
        one = {key: record[f"one_row_{key}"] for key in ("joint", "stick", "power", "fire")}  # (base, D)
        every = record["every_row_joint"]
        rng = np.random.default_rng(0)
        draws = rng.integers(0, len(every), (1000, len(every)))
        rows = {}
        for column, name in enumerate(SAMPLED):
            values = one["joint"][:, column]
            boot = values[draws].mean(1)
            rows[name] = {"mean": float(values.mean()), "low": float(np.quantile(boot, 0.025)),
                          "high": float(np.quantile(boot, 0.975)), "max": float(values.max()),
                          "share_above_0.01": float((values > 0.01).mean()),
                          **{key: float(one[key][:, column].mean()) for key in ("stick", "power", "fire")}}
        out["one_row"] = {"base": int(len(every)), "every_row_mean": float(every.mean()),
                          "every_row_quantiles": {str(q): float(np.quantile(every, q)) for q in (0.05, 0.5, 0.95)},
                          "rows": rows}
    path = data / "sweep.npz"
    if path.exists():
        record = load(path)
        labels = record["labels"]
        out["sweep"] = {name: [[float(labels[i][1]), float(record["joint"][i]), float(record["stick"][i]),
                                float(record["power"][i]), float(record["fire"][i])]
                               for i in range(len(labels)) if int(labels[i][0]) == column]
                        for column, name in enumerate(SAMPLED)}
    path = data / f"pairwise{arguments.count}.npz"
    if path.exists():
        pair = load(path)
        pair_names = [str(name) for name in pair["names"]]
        matrix = pair["matrix"]
        by_name = {f"screen{i}": i for i in range(len(skill))}
        pair_skill = np.array([skill[by_name[name]] if name in by_name else np.nan for name in pair_names])
        found = families(matrix)
        upper = matrix[np.triu_indices(len(pair_names), 1)]
        out["pairwise"] = {"count": len(pair_names), "explained": found["explained"], "silhouettes": found["silhouettes"],
                           "k": found["k"], "silhouette": found["silhouette"], "mean_js": float(upper.mean()),
                           "quantiles": {str(q): float(np.quantile(upper, q)) for q in (0.05, 0.5, 0.95)},
                           "reference_pairs": {f"{a}-{b}": float(matrix[pair_names.index(a), pair_names.index(b)])
                                               for a, b in (("yardstick", "cma"), ("yardstick", "bfm"), ("cma", "bfm"))}}
        # Correlation of the embedding's axes with skill and the shares.
        coordinates = found["coordinates"]
        finite = np.isfinite(pair_skill)
        out["pairwise"]["axes"] = {
            f"axis{axis + 1}": {"skill": float(np.corrcoef(coordinates[finite, axis], pair_skill[finite])[0, 1]),
                                **{key: float(np.corrcoef(coordinates[:, axis], pair[key])[0, 1]) for key in SHARES},
                                **{row: float(np.corrcoef(coordinates[finite, axis],
                                                          units[[by_name[n] for n in np.array(pair_names)[finite]], column])[0, 1])
                                   for column, row in enumerate(SAMPLED)}}
            for axis in range(2)}
        labels = found["labels"]
        clusters = []
        for label in sorted(set(labels)):
            members = [i for i in range(len(labels)) if labels[i] == label]
            others = [i for i in range(len(labels)) if labels[i] != label]
            member_units = np.array([units[by_name[pair_names[i]]] for i in members if pair_names[i] in by_name])
            inside = matrix[np.ix_(members, members)]
            entry = {"label": int(label), "size": len(members),
                     "references": [pair_names[i] for i in members if pair_names[i] not in by_name],
                     "skill_mean": float(np.nanmean(pair_skill[members])) if finite[members].any() else None,
                     "within_js": float(inside[np.triu_indices(len(members), 1)].mean()) if len(members) > 1 else 0.0,
                     "to_others_js": float(matrix[np.ix_(members, others)].mean()) if others else None,
                     **{key: float(pair[key][members].mean()) for key in SHARES}}
            if len(member_units):
                means = member_units.mean(0)
                top = np.argsort(-np.abs(means - 0.5))[:5]
                entry["distinct_rows"] = {SAMPLED[i]: float(means[i]) for i in top}
            clusters.append(entry)
        out["pairwise"]["families"] = clusters
        np.savez(data / f"embedding{arguments.count}.npz", coordinates=coordinates, labels=labels, skill=pair_skill,
                 names=np.array(pair_names), **{key: pair[key] for key in SHARES})
    path = HERE / "divergence.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    log(f"wrote {path}")


def figure(data: Path, count: int, out_path: Path, share: str) -> None:
    """The MDS embedding of the pairwise configurations, three panels stacked (for a phone):
    coloured by screening skill, by mean P(dive) and by mean P(full stick) on the states."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    record = load(data / f"embedding{count}.npz")
    xy, skill = record["coordinates"], record["skill"]
    names = [str(name) for name in record["names"]]
    panels = (("screening skill", skill), ("mean P(dive) on the states", record["dive"]),
              ("mean P(full stick) on the states", record["full_stick"]))
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 13.0), facecolor="#fcfcfb")
    for axis, (title, colour) in zip(axes, panels, strict=True):
        axis.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#8a8984")
        axis.tick_params(colors="#52514e", labelsize=8)
        axis.set_xlabel("MDS 1", color="#52514e", fontsize=9)
        axis.set_ylabel("MDS 2", color="#52514e", fontsize=9)
        finite = np.isfinite(colour)
        points = axis.scatter(xy[finite, 0], xy[finite, 1], c=colour[finite], cmap="Blues", s=20,
                              edgecolor="#8a8984", linewidth=0.3)
        fig.colorbar(points, ax=axis, shrink=0.85)
        axis.set_title(title, fontsize=10, color="#0b0b0b")
        for name in ("yardstick", "cma", "bfm"):
            i = names.index(name)
            axis.scatter(xy[i, 0], xy[i, 1], s=80, facecolor="#eb6834", edgecolor="#0b0b0b", linewidth=1.0, marker="D")
            axis.annotate(name, xy[i], xytext=(6, 4), textcoords="offset points", fontsize=8, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor="#fcfcfb")
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
