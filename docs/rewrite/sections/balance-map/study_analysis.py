"""The balance study's analysis: per-condition ability models, their accuracy, each handle's
effect, the active subspace, the cycle check, and predicted-balanced pairs.

    PYTHONPATH=$PWD/src:$PWD uv run --no-sync python \
        docs/rewrite/sections/balance-map/study_analysis.py RUN OUT [--pairs 6]

RUN: a `bench.balance_study` run (cards.jsonl, evaluations.jsonl, references.json). The model
per condition c is Bradley-Terry-like: logit(kill share of card a against card b) =
f_c(a) - f_c(b), with f_c a gradient-boosting model of the 13 search coordinates and each
reference's ability its own free offset r_c(reference) (backfitting: f on the targets plus
the offsets, the offsets as the weighted mean residuals, three rounds). Weighted by the
inverse variance of each logit.

Writes into OUT: models.pkl, models.json (cross-validated accuracy, the reference offsets
against the model's own value at each reference's coordinates, the cycle check, the active
subspaces), partial_dependence.png, pairs.json (predicted-balanced, far-apart pairs, ready
for `bench.balance_study confirm`).
"""

import argparse
import json
import math
import pathlib
import pickle

import numpy

from bench import balance_map

CONDITIONS = balance_map.EVERY
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE, INK, MUTED = "#fcfcfb", "#1f1f1e", "#6b6b68"


def read_lines(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def logit(p):
    p = numpy.clip(p, 1e-3, 1 - 1e-3)
    return numpy.log(p / (1 - p))


def win_equivalent(logit_value, condition):
    return 100.0 / (1.0 + numpy.exp(-balance_map.SLOPES[condition] * logit_value))


def load(run: pathlib.Path):
    """(cards by id with "x" the coordinates of the solved handles, reference coordinates,
    evaluation rows)."""
    cards = {}
    for row in read_lines(run / "cards.jsonl"):
        if "card" in row:
            cards[row["id"]] = dict(row, x=balance_map.encode(row["handles"]))
    references = {}
    ships = balance_map.reference_ships()
    for name in ("jujak", "yongma"):
        references[name] = balance_map.encode(dict(ships[name].handles))
    base = dict(ships[balance_map.REFERENCE].handles)
    g = balance_map.decode(balance_map.encode(dict(balance_map.seeds())["G-yongma"]), base)
    references["g-yongma"] = balance_map.encode(g)
    return cards, references, read_lines(run / "evaluations.jsonl")


def rows_of(evaluations, cards, condition, kind):
    """(card ids a, opponents b, y logits, w weights) for one condition and kind."""
    a, b, y, w = [], [], [], []
    for e in evaluations:
        if e["condition"] != condition or e["kind"] != kind or e["a"] not in cards:
            continue
        if kind == "cross" and e["b"] not in cards:
            continue
        share, error = e["kill_share"]
        p = min(max(share, 1e-3), 1 - 1e-3)
        variance = max(error, 0.01) ** 2 / (p * (1 - p)) ** 2
        a.append(e["a"])
        b.append(e["b"])
        y.append(logit(share))
        w.append(1.0 / variance)
    return a, b, numpy.array(y), numpy.array(w)


def model():
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=0,
    )


def fit_bt(x, references, y, w, rounds: int = 3):
    """f (a fitted model) and the reference offsets {name: r} for y = f(x) - r[reference]
    (backfitting; each round's offsets from 3-fold cross-fitted predictions, so f cannot
    absorb a card's reference into its own value)."""
    from sklearn.model_selection import KFold

    names = sorted(set(references))
    masks = {n: numpy.array([r == n for r in references]) for n in names}
    offsets = {n: 0.0 for n in names}
    for _ in range(rounds):
        target = y + numpy.array([offsets[r] for r in references])
        predicted = numpy.zeros_like(y)
        for train, test in KFold(3, shuffle=True, random_state=0).split(x):
            part = model().fit(x[train], target[train], sample_weight=w[train] / w.mean())
            predicted[test] = part.predict(x[test])
        for n in names:
            m = masks[n]
            offsets[n] = float(numpy.average(predicted[m] - y[m], weights=w[m]))
    target = y + numpy.array([offsets[r] for r in references])
    f = model().fit(x, target, sample_weight=w / w.mean())
    return f, offsets


def cross_validate(x, references, y, w, groups, folds: int = 3):
    """Out-of-fold predictions of y (folds by card)."""
    from sklearn.model_selection import GroupKFold

    predicted = numpy.zeros_like(y)
    for train, test in GroupKFold(folds).split(x, y, groups):
        f, offsets = fit_bt(x[train], [references[i] for i in train], y[train], w[train])
        predicted[test] = f.predict(x[test]) - numpy.array([offsets[references[i]] for i in test])
    return predicted


def weighted_r2(y, predicted, w):
    mean = numpy.average(y, weights=w)
    return 1.0 - numpy.average((y - predicted) ** 2, weights=w) / numpy.average(
        (y - mean) ** 2, weights=w
    )


def active_subspace(x, f) -> dict:
    """A smooth quadratic (ridge) fitted to the model's predictions, its gradients' outer
    product averaged over the data: eigenvalues (normalised) and the leading direction."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(2, include_bias=False)
    smooth = Ridge(alpha=1e-2).fit(poly.fit_transform(x), f.predict(x))
    epsilon = 1e-3
    sample = x[numpy.random.default_rng(0).choice(len(x), min(len(x), 3000), replace=False)]
    gradients = []
    for k in range(x.shape[1]):
        up, down = sample.copy(), sample.copy()
        up[:, k] += epsilon
        down[:, k] -= epsilon
        gradients.append(
            (smooth.predict(poly.transform(up)) - smooth.predict(poly.transform(down)))
            / (2 * epsilon)
        )
    g = numpy.stack(gradients, -1)
    c = g.T @ g / len(g)
    values, vectors = numpy.linalg.eigh(c)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    return dict(
        eigenvalues=[round(float(v / values.sum()), 4) for v in values],
        leading=dict(
            zip(balance_map.NAMES, [round(float(v), 3) for v in vectors[:, 0]], strict=True)
        ),
        second=dict(
            zip(balance_map.NAMES, [round(float(v), 3) for v in vectors[:, 1]], strict=True)
        ),
    )


def partial_dependence_plot(models, x_all, path: pathlib.Path, jujak_ability) -> dict:
    """Each handle's partial dependence of f_c (minus Jujak's ability), as win-equivalent
    points against Jujak; one panel a handle, one line a condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import partial_dependence

    grid = numpy.linspace(0.0, 1.0, 21)
    columns = 4
    rows = math.ceil(len(balance_map.SPACE) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(13, 2.6 * rows), sharey=True)
    figure.patch.set_facecolor(SURFACE)
    swings = {}
    for number, axis in enumerate(balance_map.SPACE):
        panel = axes.flat[number]
        panel.set_facecolor(SURFACE)
        values = [axis.value(g) for g in grid]
        swings[axis.handle] = {}
        for column, condition in enumerate(CONDITIONS):
            f = models[condition]["f"]
            result = partial_dependence(
                f, x_all, [number], custom_values={number: grid}, kind="average", method="brute"
            )
            win = win_equivalent(result["average"][0] - jujak_ability[condition], condition)
            swings[axis.handle][condition] = round(float(win.max() - win.min()), 1)
            panel.plot(values, win, color=SERIES[column], linewidth=1.6, label=condition)
        panel.axhline(50.0, color=MUTED, linewidth=0.8, linestyle=":")
        panel.set_title(axis.handle.replace("_", " "), fontsize=9, color=INK)
        panel.tick_params(labelsize=7, colors=MUTED)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
        panel.grid(alpha=0.25, linewidth=0.5)
    for panel in list(axes.flat)[len(balance_map.SPACE) :]:
        panel.set_visible(False)
    for row in range(rows):
        axes[row, 0].set_ylabel("win-equivalent v Jujak (%)", fontsize=8, color=MUTED)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right", ncol=2, frameon=False, fontsize=8)
    figure.suptitle(
        "Partial dependence of each condition's ability on each handle (Sobol "
        "study; win-rate equivalent against v23 Jujak)",
        fontsize=11,
        color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(figure)
    return swings


def predicted_pairs(models, cards, jujak_ability, count: int, pool_size: int = 4000):
    """Far-apart pairs of solved cards whose predicted abilities agree in all 8 conditions:
    each card's ability vector (win-equivalent against Jujak); a pair's imbalance the worst
    condition's win-equivalent gap between the two (from the logit difference); among pairs
    with predicted imbalance under 3 points and the time to kill in the band both ways, the
    farthest apart in coordinates, greedily (each card in at most one pair)."""
    ids = list(cards)
    x = numpy.array([cards[i]["x"] for i in ids])
    ability = numpy.stack([models[c]["f"].predict(x) - jujak_ability[c] for c in CONDITIONS], -1)
    slopes = numpy.array([balance_map.SLOPES[c] for c in CONDITIONS])
    # cards near some common level first: the closest to Jujak's level, and spread
    rng = numpy.random.default_rng(0)
    pool = rng.choice(len(ids), min(pool_size, len(ids)), replace=False)
    found = []
    for n, i in enumerate(pool):
        gaps = numpy.abs(ability[pool[n + 1 :]] - ability[i]) * slopes[None, :] * 25.0
        worst = gaps.max(-1)
        for offset in numpy.flatnonzero(worst < 3.0):
            j = pool[n + 1 + offset]
            a, b = cards[ids[i]]["handles"], cards[ids[j]]["handles"]
            times = balance_map.mutual_time_to_kill(a, b)
            if not all(
                balance_map.TIME_TO_KILL_BAND[0] <= t <= balance_map.TIME_TO_KILL_BAND[1]
                for t in times
            ):
                continue
            distance = float(numpy.linalg.norm(x[i] - x[j]))
            found.append((distance, float(worst[offset]), ids[i], ids[j]))
    found.sort(reverse=True)
    chosen, used = [], set()
    for distance, worst, i, j in found:
        if i in used or j in used:
            continue
        chosen.append(
            dict(
                a=i,
                b=j,
                distance=round(distance, 3),
                predicted_worst_gap=round(worst, 2),
                level_against_jujak={
                    c: round(float(win_equivalent(ability[ids.index(i)][k], c)), 1)
                    for k, c in enumerate(CONDITIONS)
                },
            )
        )
        used.update((i, j))
        if len(chosen) >= count:
            break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--pairs", type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cards, reference_x, evaluations = load(args.run)
    models, report = {}, {}
    for condition in CONDITIONS:
        a, b, y, w = rows_of(evaluations, cards, condition, "reference")
        if len(y) < 50:
            continue
        x = numpy.array([cards[i]["x"] for i in a])
        oof = cross_validate(x, b, y, w, a)
        f, offsets = fit_bt(x, b, y, w)
        jujak = float(f.predict(reference_x["jujak"][None])[0])
        models[condition] = dict(f=f, offsets=offsets)
        # the cycle check: card against card, predicted by the transitive model
        ca, cb, cy, cw = rows_of(evaluations, cards, condition, "cross")
        cross = {}
        if len(cy):
            xa = numpy.array([cards[i]["x"] for i in ca])
            xb = numpy.array([cards[i]["x"] for i in cb])
            predicted = f.predict(xa) - f.predict(xb)
            z_cross = (cy - predicted) * numpy.sqrt(cw)
            z_reference = (y - oof) * numpy.sqrt(w)
            cross = dict(
                games=len(cy),
                r2=round(float(weighted_r2(cy, predicted, cw)), 3),
                rms_z_cross=round(float(numpy.sqrt(numpy.mean(z_cross**2))), 2),
                rms_z_reference_out_of_fold=round(float(numpy.sqrt(numpy.mean(z_reference**2))), 2),
                mean_residual_cross=round(float(numpy.mean(cy - predicted)), 3),
            )
        report[condition] = dict(
            rows=len(y),
            r2_cross_validated=round(float(weighted_r2(y, oof, w)), 3),
            rmse_logit=round(float(numpy.sqrt(numpy.average((y - oof) ** 2, weights=w))), 3),
            noise_rms_logit=round(float(numpy.sqrt(numpy.mean(1.0 / w))), 3),
            offsets={k: round(v, 3) for k, v in offsets.items()},
            model_at_reference={
                k: round(float(f.predict(v[None])[0]), 3) for k, v in reference_x.items()
            },
            jujak_ability=round(jujak, 3),
            cycle_check=cross,
            active_subspace=active_subspace(x, f),
        )
        models[condition]["jujak"] = jujak
        print(
            condition,
            json.dumps({k: v for k, v in report[condition].items() if k != "active_subspace"}),
            flush=True,
        )
    jujak_ability = {c: models[c]["jujak"] for c in models}
    x_all = numpy.array([c["x"] for c in cards.values()])
    swings = {}
    if len(models) == len(CONDITIONS):
        swings = partial_dependence_plot(
            models, x_all, args.out / "partial_dependence.png", jujak_ability
        )
        pairs = predicted_pairs(models, cards, jujak_ability, args.pairs)
        (args.out / "pairs.json").write_text(json.dumps(pairs, indent=1))
        print(json.dumps(pairs, indent=1))
    with (args.out / "models.pkl").open("wb") as handle:
        pickle.dump(dict(models=models, features=balance_map.NAMES), handle)
    (args.out / "models.json").write_text(json.dumps(dict(report=report, swings=swings), indent=1))


if __name__ == "__main__":
    main()
