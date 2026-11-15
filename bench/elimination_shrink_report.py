"""Elimination shrink sweep: per (start, duration), game length and how much the boundary bears."""

import collections
import json
import statistics
import sys

games, meta = collections.defaultdict(list), {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    games[r["label"]] += r["games"]
    meta[r["label"]] = r


def q(xs, p):
    xs = sorted(xs)
    return xs[min(int(p * len(xs)), len(xs) - 1)]


print(
    "start dur clock |  mean median  p80  p90 | ended before start | radius<75% at end | <50% | boundary deaths | games w/ one | clock ends"
)
for label, g in games.items():
    hold, dur = (int(x[1:]) for x in label.split("_"))
    clock = hold + dur + 8
    t = [x["end_tick"] / 30 for x in g]
    frac = [1 - min(max((s - hold) / dur, 0), 1) for s in t]
    deaths = sum(x["deaths"] for x in g)
    bd = sum(x["boundary_deaths"] for x in g)
    n = len(g)
    print(
        f"{hold:5d} {dur:3d} {clock:5d} | {statistics.mean(t):5.0f} {statistics.median(t):6.0f} {q(t, 0.8):4.0f} {q(t, 0.9):4.0f} |"
        f" {sum(s < hold for s in t) / n:18.2f} | {sum(f < 0.75 for f in frac) / n:17.2f} | {sum(f < 0.5 for f in frac) / n:4.2f} |"
        f" {bd / deaths:15.3f} | {sum(x['boundary_deaths'] > 0 for x in g) / n:12.2f} | {sum(not x['by_rule'] for x in g) / n:10.2f}"
    )
