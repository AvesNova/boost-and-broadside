"""Survivor distribution + candidate ranking from hero.py batch logs.

analyze.py games.jsonl [--k 2] [--top 12] [--survivors-out survivors.json]
"""

import argparse
import collections
import json

EVERY = 8  # hero.TIMELINE_EVERY


def features(r):
    ours, theirs = r["ours"], r["theirs"]
    T = len(ours)
    # Chase tail: samples from the moment red is down to its last ship until the end.
    last_red = next((t for t in range(T) if theirs[t] <= 1), T)
    tail = (T - last_red) * EVERY
    # First blood: when the first ship of either side dies.
    first = next((t for t in range(T) if ours[t] < r["n0"] or theirs[t] < r["n1"]), T) * EVERY
    # Deepest deficit (red minus blue) after first blood, and whether blue led late.
    deficit = max(b - a for a, b in zip(ours, theirs))
    # Kill pace: how much of the game sees deaths (fraction of samples with a change).
    changes = sum(1 for t in range(1, T) if (ours[t], theirs[t]) != (ours[t - 1], theirs[t - 1]))
    # Red ships still alive when blue first drops to its final count: how close it got.
    k = r["blue_left"]
    at_final = next((theirs[t] for t in range(T) if ours[t] <= k), 0)
    return dict(
        tail=tail,
        first_blood=first,
        max_deficit=deficit,
        change_samples=changes,
        red_when_final=at_final,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--survivors-out")
    a = p.parse_args()
    rows = [json.loads(line) for f in a.logs for line in open(f)]
    n0, n1 = rows[0]["n0"], rows[0]["n1"]
    wins = [r for r in rows if r["winner"] == "blue"]
    losses = [r for r in rows if r["winner"] == "red"]
    ties = [r for r in rows if r["winner"] == "tie"]
    dist = collections.Counter(r["blue_left"] for r in wins)
    loss_dist = collections.Counter(r["red_left"] for r in losses)
    summary = dict(
        matchup=f"{n0}v{n1}",
        scenario=rows[0].get("scenario", "vs_scripted"),
        checkpoint="wild-universe-731/avg_step_000671846400.pt",
        games=len(rows),
        blue_wins=len(wins),
        red_wins=len(losses),
        ties=len(ties),
        win_rate=len(wins) / len(rows),
        wins_by_blue_survivors={str(k): dist.get(k, 0) for k in range(1, n0 + 1)},
        losses_by_red_survivors={str(k): loss_dist.get(k, 0) for k in range(1, n1 + 1)},
        max_steps=1024,
        note="Each game is one env of a seeded batch; finished envs are not reset. "
        "Survivors = ships of the winning side alive at the end.",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "note"}, indent=1))
    if a.survivors_out:
        json.dump(summary, open(a.survivors_out, "w"), indent=1)
    cands = [r for r in wins if r["blue_left"] == a.k]
    for r in cands:
        r.update(features(r))
    # A real fight: short chase tail, the fight starts early, blue fell well behind.
    # Close: red still had ships when blue fell to its final count. Watchable: >= 4 s.
    cands.sort(key=lambda r: (r["tail"] > 40, r["steps"] < 230, -r["red_when_final"], -r["steps"]))
    print(f"\n{len(cands)} wins with exactly {a.k} left; best {a.top}:")
    for r in cands[: a.top]:
        print(
            f"seed {r['seed']} env {r['env']:3d} steps {r['steps']:4d} tail {r['tail']:4d} "
            f"first {r['first_blood']:4d} deficit {r['max_deficit']} "
            f"redAtFinal {r['red_when_final']} "
            f"ours {r['ours'][::8]} theirs {r['theirs'][::8]}"
        )


if __name__ == "__main__":
    main()
