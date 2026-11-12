"""Round 0: the baselines on the base card (the ship files, hit damage x 0.65): doctrines off
and the hand doctrines in every lineup, the league's Elo (mixed lineups), the hand
doctrines' distinctness. Writes round0.json and prints the tables.

    PYTHONPATH=src:. python docs/rewrite/sections/game-loop/round0.py
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from league import load  # noqa: E402

from bench.game_loop import Group, Job, Loop, distinctness, job_summary, markdown, ratings, run  # noqa: E402

HERE = pathlib.Path(__file__).parent
SECONDS = 120.0
B = 8192
started = time.time()
loop = Loop(B, 1, SECONDS)
players = load()
by = {p.name: p for p in players}
jobs = []
for name in ("off", "hand"):
    for lineup in ("energy v turn", "turn mirror", "energy mirror", "mixed", "mixed reversed"):
        jobs.append(Job(0, lineup, by[name], by[name], 2048))
names = list(by)
for i, a in enumerate(names):
    for b in names[i + 1 :]:
        for lineup in ("mixed", "mixed reversed"):
            jobs.append(Job(0, lineup, by[a], by[b], 512))
outs = run(loop, jobs)
rows = [job_summary(job, out, SECONDS) for job, out in zip(jobs, outs)]
elo = ratings([r for r in rows if r["a"] != r["b"]])
hand = by["hand"]
distinct = distinctness(loop, [Group(0, "mixed", hand, hand, B)], hand.energy, hand.turn)
(HERE / "round0.json").write_text(json.dumps(dict(rows=rows, elo=elo, distinct=distinct), indent=1))
print(markdown(rows))
print("Elo", json.dumps(elo))
print("distinct", json.dumps(distinct))
print(f"{time.time() - started:.0f} s")
