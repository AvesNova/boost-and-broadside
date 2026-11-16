"""Win rate against kill share, per condition, from the balance-final records (2026-11-16):
every card's "energy v turn" job, its `score` (the Yongma team's wins and half the draws) and
`kill_score` (its per-game mean share of the points, a side's points the other side's deaths
of any cause). Those records ran with fog and fields on, each preset's own map, deathmatch
300 s. Card candidates only (a doctrine candidate, with "energy" or "turn" fields, is left
out: a doctrine change moves the wins and the kills differently).

    PYTHONPATH=src:. python docs/rewrite/sections/balance-eval/conversion_fit.py \
        /home/vizia/avesnova/bnb-balance-final/docs/rewrite/sections/balance-final/data

The fit per condition: logit(score) = a + b logit(kill share), weighted least squares with
each point's variance the score's (delta method) plus b^2 times the kill share's, iterated.
Prints per condition the points, a, b, the reduced chi^2, the weighted R^2, the residual
spread in score points, and the kill-share band that maps to a 45-55% score.
"""

import json
import math
import pathlib
import sys


def logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def records(folders: list[pathlib.Path]) -> dict[str, list[tuple]]:
    """(kill, kill se, score, score se, games) per condition, duplicates dropped."""
    seen, out = set(), {}
    for folder in folders:
        for path in sorted(folder.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    job = row["jobs"]["energy v turn"]
                    point = (*job["kill_score"], *job["score"], job["games"])
                except KeyError, ValueError, TypeError:
                    continue
                candidate = row.get("candidate", {})
                if candidate.get("energy") or candidate.get("turn"):
                    continue  # a doctrine candidate: the fit is of card changes
                key = (row.get("name"), row.get("mode"), row.get("ships"), point)
                if key in seen:
                    continue
                seen.add(key)
                condition = f"{row['mode']}-{row['ships']}v{row['ships']}"
                out.setdefault(condition, []).append(point)
    return out


def fit(points: list[tuple]) -> dict:
    """The weighted logit-logit fit and its quality."""
    b = 4.0
    for _ in range(20):
        rows = []
        for kill, kill_se, score, score_se, _ in points:
            p = min(max(score, 0.02), 0.98)
            k = min(max(kill, 0.02), 0.98)
            var_y = (max(score_se, 1e-3) / (p * (1 - p))) ** 2
            var_x = (max(kill_se, 1e-4) / (k * (1 - k))) ** 2
            rows.append((logit(k), logit(p), 1.0 / (var_y + b * b * var_x)))
        sw = sum(w for _, _, w in rows)
        mx = sum(w * x for x, _, w in rows) / sw
        my = sum(w * y for _, y, w in rows) / sw
        sxx = sum(w * (x - mx) ** 2 for x, _, w in rows)
        sxy = sum(w * (x - mx) * (y - my) for x, y, w in rows)
        new_b = sxy / sxx
        if abs(new_b - b) < 1e-6:
            break
        b = new_b
    a = my - b * mx
    chi2 = sum(w * (y - a - b * x) ** 2 for x, y, w in rows)
    syy = sum(w * (y - my) ** 2 for _, y, w in rows)
    residual = [score - sigmoid(a + b * logit(kill)) for kill, _, score, _, _ in points]
    band = [sigmoid((logit(target) - a) / b) for target in (0.45, 0.5, 0.55)]
    return dict(
        points=len(points),
        a=a,
        b=b,
        reduced_chi2=chi2 / max(len(points) - 2, 1),
        r2=1 - chi2 / syy,
        residual_sd=math.sqrt(sum(r * r for r in residual) / len(residual)),
        kill_band=band,
        kill_range=(min(p[0] for p in points), max(p[0] for p in points)),
    )


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    by_condition = records([root, root / "box-out"])
    print(
        "| condition | cards | slope b | intercept a | R^2 | chi^2/dof | residual (score) "
        "| kill share for 45 / 50 / 55% | kill shares measured |"
    )
    print("|---" * 9 + "|")

    def size(condition: str) -> tuple[str, int]:
        mode, ships = condition.split("-")
        return mode, int(ships.split("v")[0])

    order = sorted(by_condition, key=size)
    for condition in order:
        f = fit(by_condition[condition])
        band = " / ".join(f"{100 * k:.1f}" for k in f["kill_band"])
        print(
            f"| {condition} | {f['points']} | {f['b']:.2f} | {f['a']:+.3f} | {f['r2']:.3f} "
            f"| {f['reduced_chi2']:.2f} | {100 * f['residual_sd']:.1f} pts | {band} "
            f"| {100 * f['kill_range'][0]:.0f}-{100 * f['kill_range'][1]:.0f} |"
        )


if __name__ == "__main__":
    main()
