"""The screening's analysis: Morris indices, the surrogate (gradient-boosted trees) with its
held-out R^2, Sobol indices of the surrogate (SALib), SHAP importances and interactions, and
the robustness measures. Reads skills.json (screen.py skills); writes analysis.json and prints
the tables of docs/rewrite/scripted-screening.md.

    analyse.py [--quick]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from lean.scripted.parameters import PARAMETERS, SAMPLED, UPWARD

HERE = Path(__file__).resolve().parent
LEVELS = 4
DELTA = LEVELS / (2 * (LEVELS - 1))
NAMES = list(SAMPLED)
DEFAULT_UNIT = np.array([0.0 if PARAMETERS[name].law == UPWARD else 0.5 for name in NAMES])


def r2(y, prediction) -> float:
    return float(1.0 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))


def morris(skills: dict) -> dict:
    entries = skills["morris"]
    sample = np.load(HERE / "morris_sample.npy")
    if len(entries) != len(sample):
        raise SystemExit(f"morris: {len(entries)} of {len(sample)} configurations complete")
    y = np.array([entry["skill"] for entry in entries])
    error = float(np.median([entry["error"] for entry in entries]))
    problem = {"num_vars": len(NAMES), "names": NAMES, "bounds": [[0.0, 1.0]] * len(NAMES)}
    result = morris_analyze.analyze(problem, sample, y, num_levels=LEVELS, num_resamples=1000, seed=1)
    # Noise alone: an elementary effect is a difference of two skills over DELTA, so its
    # noise is sqrt(2) error / DELTA, and E|N(0, s)| = s sqrt(2 / pi).
    effect_noise = math.sqrt(2.0) * error / DELTA
    floor = effect_noise * math.sqrt(2.0 / math.pi)
    rows = []
    for index, name in enumerate(NAMES):
        rows.append({"name": name, "mu_star": float(result["mu_star"][index]), "mu": float(result["mu"][index]),
                     "sigma": float(result["sigma"][index]),
                     "mu_star_conf": float(result["mu_star_conf"][index])})
    rows.sort(key=lambda row: -row["mu_star"])
    return {"rows": rows, "skill_error": error, "effect_noise": effect_noise, "noise_floor": floor,
            "skill_mean": float(y.mean()), "skill_std": float(y.std()), "configurations": len(y)}


def fit(x, y, quick: bool):
    """The surrogate: gradient-boosted trees, settings chosen by 5-fold CV on the training
    split; returns (model, settings, CV R^2 table)."""
    grid = [dict(n_estimators=n, learning_rate=rate, max_depth=depth, subsample=0.8, min_samples_leaf=leaf)
            for n, rate in ((400, 0.05), (800, 0.025)) for depth in (2, 3, 4) for leaf in (10, 30)]
    if quick:
        grid = grid[:2]
    scores = []
    folds = KFold(5, shuffle=True, random_state=0)
    for settings in grid:
        prediction = cross_val_predict(GradientBoostingRegressor(random_state=0, **settings), x, y, cv=folds)
        scores.append((r2(y, prediction), settings))
    scores.sort(key=lambda item: -item[0])
    best = scores[0][1]
    return GradientBoostingRegressor(random_state=0, **best).fit(x, y), best, scores


def surrogate(skills: dict, quick: bool) -> dict:
    entries = skills["screen"]
    x = np.array([entry["unit"] for entry in entries])
    y = np.array([entry["skill"] for entry in entries])
    errors = np.array([entry["error"] for entry in entries])
    noise = float(np.mean(errors ** 2))
    # The best R^2 any model can reach: the share of the skills' variance that is not noise.
    ceiling = 1.0 - noise / float(y.var())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    model, settings, table = fit(x_train, y_train, quick)
    held_out = r2(y_test, model.predict(x_test))
    model = GradientBoostingRegressor(random_state=0, **settings).fit(x, y)

    # Sobol indices of the surrogate (Saltelli's scheme, SALib).
    problem = {"num_vars": len(NAMES), "names": NAMES, "bounds": [[0.0, 1.0]] * len(NAMES)}
    base = 1024 if quick else 8192
    points = sobol_sample.sample(problem, base, calc_second_order=False, seed=0)
    indices = sobol_analyze.analyze(problem, model.predict(points), calc_second_order=False, seed=0)

    # SHAP (TreeExplainer): importances on every configuration, interactions on 1,000.
    import shap
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x)
    importance = np.abs(values).mean(0)
    rng = np.random.default_rng(0)
    subset = x[rng.choice(len(x), size=min(1000, len(x)), replace=False)]
    interaction = np.abs(explainer.shap_interaction_values(subset)).mean(0)
    pairs = []
    for i in range(len(NAMES)):
        for j in range(i + 1, len(NAMES)):
            # The matrix splits a pair's interaction evenly between (i, j) and (j, i).
            pairs.append((float(interaction[i, j] + interaction[j, i]), NAMES[i], NAMES[j]))
    pairs.sort(reverse=True)

    # One-at-a-time profiles of the surrogate through the defaults (the yardstick's point),
    # and the average profile over the sample (partial dependence).
    grid = np.linspace(0.0, 1.0, 21)
    profiles = {}
    for index, name in enumerate(NAMES):
        through_default = np.repeat(DEFAULT_UNIT[None], len(grid), 0)
        through_default[:, index] = grid
        averaged = []
        for value in grid:
            moved = x.copy()
            moved[:, index] = value
            averaged.append(float(model.predict(moved).mean()))
        profiles[name] = {"through_default": model.predict(through_default).tolist(), "partial": averaged}

    rows = []
    for index, name in enumerate(NAMES):
        rows.append({"name": name, "S1": float(indices["S1"][index]), "S1_conf": float(indices["S1_conf"][index]),
                     "ST": float(indices["ST"][index]), "ST_conf": float(indices["ST_conf"][index]),
                     "shap": float(importance[index])})
    rows.sort(key=lambda row: -row["ST"])
    top = np.argsort(-y)[:40]
    return {
        "configurations": len(y), "skill_mean": float(y.mean()), "skill_std": float(y.std()),
        "skill_quantiles": {str(q): float(np.quantile(y, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
        "skill_max": float(y.max()), "noise_variance": noise, "r2_ceiling": ceiling, "r2_held_out": held_out,
        "cv_table": [(score, settings) for score, settings in table], "settings": settings,
        "sobol_rows": rows, "sobol_sum_S1": float(np.sum(indices["S1"])), "pairs": pairs[:15],
        "default_prediction": float(model.predict(DEFAULT_UNIT[None])[0]),
        "profiles": profiles, "grid": grid.tolist(),
        "top_mean_unit": dict(zip(NAMES, x[top].mean(0).tolist(), strict=True)),
        "top_skills": y[top].tolist(),
        "condition_factors": conditions(skills, entries),
    }


def conditions(skills: dict, entries) -> dict:
    """Skill by condition factor (arena radius, team size, spawn) per configuration: the
    factor levels' means and the correlation of skill between levels across configurations
    (raw, and corrected for the noise in each: Spearman's disattenuation)."""
    labels = skills["conditions"]
    by = np.array([entry["by_condition"] for entry in entries])  # (C, 18)
    out = {}
    for factor, key in (("radius", lambda c: c["radius"]), ("sizes", lambda c: c["sizes"][0]),
                        ("spawn", lambda c: c["spawn"])):
        levels = sorted({key(label) for label in labels}, key=str)
        means = {str(level): by[:, [i for i, c in enumerate(labels) if key(c) == level]].mean(1) for level in levels}
        out[factor] = {"means": {level: float(value.mean()) for level, value in means.items()}}
        names = list(means)
        # Noise of a level's mean: the skill's noise variance times (cells / cells in the level).
        share = len(labels) / (len(labels) / len(names))
        noise = float(np.mean(np.array([entry["error"] for entry in entries]) ** 2)) * share
        correlations = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = means[names[i]], means[names[j]]
                raw = float(np.corrcoef(a, b)[0, 1])
                reliability = [max(1e-9, 1.0 - noise / float(v.var())) for v in (a, b)]
                correlations[f"{names[i]}-{names[j]}"] = {"raw": raw, "corrected": raw / math.sqrt(np.prod(reliability))}
        out[factor]["correlations"] = correlations
    return out


def local(skills: dict) -> dict:
    out = {}
    for name in ("local25", "local50"):
        entries = skills.get(name) or []
        if not entries:
            continue
        y = np.array([entry["skill"] for entry in entries])
        out[name] = {"configurations": len(y), "mean": float(y.mean()), "std": float(y.std()),
                     "quantiles": {str(q): float(np.quantile(y, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                     "error": float(np.median([entry["error"] for entry in entries]))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    skills = json.loads((HERE / "skills.json").read_text())
    out = {"reference": skills["reference"]}
    if skills.get("morris"):
        out["morris"] = morris(skills)
    if skills.get("screen"):
        out["surrogate"] = surrogate(skills, arguments.quick)
    out["local"] = local(skills)
    (HERE / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    report(out)


def report(out: dict) -> None:
    for entry in out["reference"]:
        print(f"reference {entry['id']}: {entry['skill']:.3f} +- {entry['error']:.3f} ({entry['games']} games, "
              f"length {entry['mean_length']:.0f})")
    if "morris" in out:
        m = out["morris"]
        print(f"\nMorris: {m['configurations']} configs, skill {m['skill_mean']:.3f} sd {m['skill_std']:.3f}, "
              f"skill error {m['skill_error']:.4f}, noise floor mu* {m['noise_floor']:.3f}")
        for row in m["rows"]:
            print(f"  {row['name']:20s} mu* {row['mu_star']:.3f} +- {row['mu_star_conf']:.3f}  mu {row['mu']:+.3f}  "
                  f"sigma {row['sigma']:.3f}")
    if "surrogate" in out:
        s = out["surrogate"]
        print(f"\nSurrogate: {s['configurations']} configs, skill {s['skill_mean']:.3f} sd {s['skill_std']:.3f}; "
              f"R^2 held out {s['r2_held_out']:.3f}, ceiling {s['r2_ceiling']:.3f}; sum S1 {s['sobol_sum_S1']:.3f}; "
              f"settings {s['settings']}")
        print(f"  quantiles {s['skill_quantiles']}, max {s['skill_max']:.3f}; default predicted "
              f"{s['default_prediction']:.3f}")
        for row in s["sobol_rows"]:
            print(f"  {row['name']:20s} S1 {row['S1']:.3f}  ST {row['ST']:.3f} +- {row['ST_conf']:.3f}  "
                  f"shap {row['shap']:.4f}")
        print("  pairs:")
        for value, a, b in s["pairs"]:
            print(f"    {a} x {b}: {value:.4f}")
        print(f"  conditions: {json.dumps(s['condition_factors'], indent=1)}")
    for name, entry in out["local"].items():
        print(f"{name}: {entry}")


if __name__ == "__main__":
    main()
