"""Pick the champion of the averaging sweep (box/eval_queue.sh, step 1) from its rating pool.

    python box/eval_champion.py POOL OUT.json --presets frontline_5v5 deathmatch_5v5 ...

Each candidate's score is its mean Elo over the presets (one mode each, equal weights, as
the evaluation suite weighs the modes). The rule, fixed before the games: the incumbent
(the run's own exponential average, rebuilt: `training_exponential`, the product today)
stays the champion unless another average's mean is above it by more than `--margin`
standard errors of the difference; then the highest mean wins. The difference's standard
error treats the two ratings as independent, which overstates it (both share the anchor),
so the rule leans toward the incumbent: picking the best of 14 noisy means alone would
carry the winner's curse into every later step. A candidate needs a finite rating in every
preset. Writes OUT.json (the champion's name and policy path, every candidate's means, the
rule and the reason) and prints the reason.
"""

import argparse
import json
import math
import pathlib
import sys

INCUMBENT = "training_exponential"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pool")
    parser.add_argument("out")
    parser.add_argument("--presets", nargs="+", required=True)
    parser.add_argument("--margin", type=float, default=2.0)
    options = parser.parse_args()

    from bnb.cli import _pool_scoring
    from bnb.league import rating_pool

    opened = rating_pool.RatingPool(options.pool)
    players, _ = opened.players()
    ratings: dict[str, dict[str, tuple[float, float]]] = {}
    for preset in options.presets:
        rows, _, _ = rating_pool.table(opened, preset, _pool_scoring(opened, preset, None))
        for row in rows:
            if row["player"] == rating_pool.ANCHOR or row["status"] != "rated":
                continue
            ratings.setdefault(row["player"], {})[preset] = (row["calibrated_elo"], row["stderr"])
    candidates = {}
    for name, held in ratings.items():
        if len(held) != len(options.presets):
            continue
        mean = sum(elo for elo, _ in held.values()) / len(held)
        stderr = math.sqrt(sum(se * se for _, se in held.values())) / len(held)
        candidates[name] = dict(mean=mean, stderr=stderr, presets={p: held[p] for p in held})
    if not candidates:
        print("no candidate has a finite rating in every preset", file=sys.stderr)
        return 1
    ranked = sorted(candidates, key=lambda name: -candidates[name]["mean"])
    best = ranked[0]
    incumbent = next((name for name in ranked if f"/average/{INCUMBENT}_" in name), None)
    if incumbent is None or incumbent == best:
        champion = best
        reason = (
            f"{best} has the highest mean Elo ({candidates[best]['mean']:.0f} +- "
            f"{candidates[best]['stderr']:.0f})"
            + ("" if incumbent else "; the incumbent has no finite rating in every preset")
        )
        gap = gap_stderr = None
    else:
        gap = candidates[best]["mean"] - candidates[incumbent]["mean"]
        gap_stderr = math.hypot(candidates[best]["stderr"], candidates[incumbent]["stderr"])
        if gap > options.margin * gap_stderr:
            champion = best
            reason = (
                f"{best} beats the incumbent {incumbent} by {gap:.0f} +- {gap_stderr:.0f} Elo "
                f"(mean over the presets), more than {options.margin:g} standard errors"
            )
        else:
            champion = incumbent
            reason = (
                f"the incumbent {incumbent} stays: the best mean, {best}, is {gap:.0f} +- "
                f"{gap_stderr:.0f} Elo above it, within {options.margin:g} standard errors"
            )
    path = next(iter(players[champion].paths), None)
    found = dict(
        champion=champion,
        path=path,
        reason=reason,
        rule=(
            f"the incumbent ({INCUMBENT}) unless another mean beats it by more than "
            f"{options.margin:g} standard errors of the difference; mean Elo over "
            f"{list(options.presets)}"
        ),
        best=best,
        incumbent=incumbent,
        gap=gap,
        gap_stderr=gap_stderr,
        ranking=[dict(name=name, **candidates[name]) for name in ranked],
    )
    pathlib.Path(options.out).write_text(json.dumps(found, indent=1) + "\n")
    print(reason)
    for name in ranked:
        held = candidates[name]
        print(f"  {held['mean']:7.1f} +- {held['stderr']:4.1f}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
