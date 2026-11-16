"""Pick the champion of the averaging sweep (box/eval_queue.sh, step 1) from its rating pool.

    python box/eval_champion.py POOL OUT.json --presets frontline_5v5 deathmatch_5v5 ...

The incumbent is the run's own exponential average, rebuilt (`training_exponential`, the
product today). Each other candidate's score is its Elo difference from the incumbent,
averaged over the presets (one mode each, equal weights, as the evaluation suite weighs
the modes), with the difference's standard error from the fit's covariance: the two share
the anchor, so a difference is measured far better than either rating, and it stays
measured when both sit beyond Imugi's reach (bounds against the anchor). The rule, fixed
before the games: the incumbent stays unless the best candidate's mean difference is above
it by more than `--margin` standard errors; picking the best of 13 noisy means alone would
carry the winner's curse into every later step. A candidate counts only where its
difference is measured in every preset (the two in one connected component of the fit);
with none, the incumbent stays by default (with no incumbent among the players, the first
player), so the later steps still run. Writes OUT.json (the champion's name and policy
path, every candidate's differences, each player's Elo and status per preset, the rule and
the reason) and prints the reason.
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
    from bnb.league.design import RATED
    from bnb.league.tournament import rating_columns

    opened = rating_pool.RatingPool(options.pool)
    players, _ = opened.players()
    names = [name for name in players if name != rating_pool.ANCHOR]
    if not names:
        print("no players in the pool", file=sys.stderr)
        return 1
    incumbent = next((name for name in names if f"/average/{INCUMBENT}_" in name), None)
    differences: dict[str, dict[str, tuple[float, float]]] = {}
    elo: dict[str, dict[str, dict]] = {}
    for preset in options.presets:
        counted, posterior, _, _ = rating_pool.fit(
            opened, preset, _pool_scoring(opened, preset, None)
        )
        index = {name: number for number, name in enumerate(counted.names)}
        for name in names:
            if name not in index:
                continue
            number = index[name]
            elo.setdefault(name, {})[preset] = rating_columns(posterior, number)
            if incumbent not in index or name == incumbent:
                continue
            other = index[incumbent]
            if int(posterior.component[number]) != int(posterior.component[other]):
                continue
            covariance = posterior.covariance
            gap = float(posterior.relative[number] - posterior.relative[other])
            variance = float(
                covariance[number, number]
                + covariance[other, other]
                - 2 * covariance[number, other]
            )
            if math.isfinite(gap) and math.isfinite(variance) and variance > 0:
                differences.setdefault(name, {})[preset] = (gap, math.sqrt(variance))
    candidates = {}
    for name, held in differences.items():
        if len(held) != len(options.presets):
            continue
        mean = sum(gap for gap, _ in held.values()) / len(held)
        stderr = math.sqrt(sum(se * se for _, se in held.values())) / len(held)
        candidates[name] = dict(gap=mean, stderr=stderr, presets=held)
    ranked = sorted(candidates, key=lambda name: -candidates[name]["gap"])
    best = ranked[0] if ranked else None
    gap = gap_stderr = None
    if incumbent is None:
        champion = names[0]
        reason = f"no incumbent ({INCUMBENT}) among the players: {champion} by default"
    elif best is None:
        champion = incumbent
        reason = f"no candidate measured against the incumbent in every preset: {incumbent}"
    else:
        gap, gap_stderr = candidates[best]["gap"], candidates[best]["stderr"]
        if gap > options.margin * gap_stderr:
            champion = best
            reason = (
                f"{best} beats the incumbent {incumbent} by {gap:.0f} +- {gap_stderr:.0f} Elo "
                f"(mean over the presets), more than {options.margin:g} standard errors"
            )
        else:
            champion = incumbent
            reason = (
                f"the incumbent {incumbent} stays: the best, {best}, is {gap:+.0f} +- "
                f"{gap_stderr:.0f} Elo from it, within {options.margin:g} standard errors"
            )
    if not any(row["status"] == RATED for held in elo.values() for row in held.values()):
        reason += " (no player has a finite Elo against Imugi)"
    path = next(iter(players[champion].paths), None)
    found = dict(
        champion=champion,
        path=path,
        reason=reason,
        rule=(
            f"the incumbent ({INCUMBENT}) unless another average's mean Elo difference from it "
            f"over {list(options.presets)} exceeds {options.margin:g} standard errors"
        ),
        best=best,
        incumbent=incumbent,
        gap=gap,
        gap_stderr=gap_stderr,
        ranking=[dict(name=name, **candidates[name]) for name in ranked],
        elo=elo,
    )
    pathlib.Path(options.out).write_text(json.dumps(found, indent=1) + "\n")
    print(reason)
    for name in ranked:
        held = candidates[name]
        print(f"  {held['gap']:+7.1f} +- {held['stderr']:4.1f}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
