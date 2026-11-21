"""CPU simulation of the unified engine's assignment (planner, lanes, residents, swaps) on a
box-like field: 108 players (scripted, imugi_v24, two chains of 52 snapshots + champion),
the box's targets in bucket 5, static mode slices (frontline 36%, deathmatch 45%,
elimination 18%). Game lengths from the laptop's games; outcomes from Bradley-Terry truth.

usage: PYTHONPATH=src python bench/unified_park_sim.py ENVIRONMENTS GAME_SECONDS [--label X]
Runs against whichever bnb tree is on PYTHONPATH (engine methods used as they are there).
"""

import argparse
import collections
import heapq
import json
import math
import pathlib
import random
import time

import numpy

from bnb.league.unified import engine as E
from bnb.league.unified.cells import Cell, Condition, Outcome
from bnb.league.unified.planner import Planner
from bnb.league.unified.run import load_prior, load_spec, player_shares
from bnb.league.unified.targets import BreakEvenTarget, EloTarget, WinRateTarget, planning_ratings

R = pathlib.Path("/home/vizia/avesnova/boost-and-broadside/data/ueval-runs")
MODES = {
    "frontline": "real_scale_frontline",
    "deathmatch": "real_scale_deathmatch",
    "elimination": "real_scale_elimination",
}
SLICES = {
    "real_scale_frontline": 0.36,
    "real_scale_deathmatch": 0.46,
    "real_scale_elimination": 0.18,
}
BLOCK = 32 / 30.0  # game seconds a block


def base_field():
    spec = load_spec(R / "curves-v24.toml", R)
    planner = Planner(spec.targets, (5,))
    load_prior(planner, [R / "prior-v24/pool"], spec.players, lambda s: None)
    laptop = [(c, o) for c, o in E.GameLog(R / "curves-v24").replay()]
    return spec, laptop


def build(seed=0):
    spec, laptop = base_field()
    old = {t.condition.preset: t for t in spec.targets}
    names = [p.name for p in spec.players if p.name != "scripted"]
    chains = ["v26", "v24"]
    field = ["scripted", "imugi_v24"]
    for chain in chains:
        field += [f"{chain}/{n}" for n in names] + [f"{chain}/champion"]
    neighbours = []
    for chain in chains:
        seq = [f"{chain}/{n}" for n in names]
        neighbours += list(zip(seq, seq[1:]))
    neighbours += [(f"v26/{n}", f"v24/{n}") for n in names]
    rng = numpy.random.default_rng(seed)
    targets, truth = [], {}
    for mode, preset in MODES.items():
        o = old[preset]
        cond = Condition(preset, 5, 5)
        t = EloTarget(f"curves/{mode}", cond, field, 60.0, budget=100_000, neighbours=neighbours)
        theta_old = planning_ratings(o.wins, o.neighbours)
        th = {}
        th["scripted"] = 0.0
        th["imugi_v24"] = -0.3  # v24's Imugi a little weaker (natural units)
        for i, n in enumerate(o.players):
            if n == "scripted":
                continue
            th[f"v24/{n}"] = theta_old[i] + th["imugi_v24"]
            th[f"v26/{n}"] = theta_old[i] + rng.normal(0, 0.3)
        th["v24/champion"] = th["v24/rl_98566144"]
        th["v26/champion"] = th["v26/rl_98566144"] + 0.2
        truth[preset] = th
        # prior: the v24 pool's games (its scripted = imugi_v24) for the v24 chain
        for a in range(len(o.players)):
            for b in range(a + 1, len(o.players)):
                g = o.games[a, b]
                if not g:
                    continue
                na = "imugi_v24" if o.players[a] == "scripted" else f"v24/{o.players[a]}"
                nb = "imugi_v24" if o.players[b] == "scripted" else f"v24/{o.players[b]}"
                ia, ib = t.index[na], t.index[nb]
                t.wins[ia, ib] += o.wins[a, b]
                t.wins[ib, ia] += o.wins[b, a]
                t.games[ia, ib] += g
                t.games[ib, ia] += g
                t.finished += int(g)
        targets.append(t)
    # v26's prior: a unified run's games (the laptop's, renamed, its scripted = the anchor)
    by = {t.condition.preset: t for t in targets}
    for cell, outcome in laptop:
        t = by[cell.condition.preset]
        a = "scripted" if cell.first == "scripted" else f"v26/{cell.first}"
        b = "scripted" if cell.second == "scripted" else f"v26/{cell.second}"
        t.record(Cell(cell.condition, a, b), outcome, pending=False)
    for t in targets:
        t._fit()
    for chain in chains:
        champion = f"{chain}/champion"
        for mode, preset in MODES.items():
            sizes = [[5, 5], [1, 1], [2, 2]] if mode != "frontline" else [[5, 5]]
            for f, s in sizes:
                c = Cell(Condition(preset, f, s), champion, "scripted")
                targets.append(WinRateTarget(f"suite/{c.key}", c, 0.02, budget=512))
            c = Cell(Condition(preset, 5, 5), champion, "imugi_v24")
            targets.append(WinRateTarget(f"suite_v24/{c.key}", c, 0.02, budget=512))
            targets.append(
                BreakEvenTarget(
                    f"be/{preset}/{champion}",
                    preset,
                    champion,
                    "scripted",
                    0.1,
                    20,
                    2048,
                    policy_count=5,
                )
            )
            team = []
            for n in range(1, 11):
                team.append([n, n])
                if n % 2 == 0:
                    team.append([n, 3 * n // 2])
                team.append([n, 2 * n])
            if mode == "frontline":
                team = [x for x in team if min(x) >= 3]
            for f, s in team:
                if max(f, s) > 5:
                    continue
                c = Cell(Condition(preset, f, s), champion, "scripted")
                targets.append(WinRateTarget(f"teamsize/{c.key}", c, 0.05, budget=128))
    planner = Planner(targets, (5, 10, 20))
    planner._touched_all()
    lengths = collections.defaultdict(list)
    for cell, outcome in laptop:
        lengths[cell.condition.preset].append(outcome.seconds)
    return planner, field, truth, lengths


class Seat:
    def __init__(self, runs):
        self.lanes = [list(m) for _, m, _ in runs]
        self.swaps = []

    def swap(self, run, lane, index):
        self.lanes[run][lane] = index


def make_engine(planner, field, environments, residents=E.RESIDENTS):
    engine = E.BucketEngine.__new__(E.BucketEngine)
    engine.bucket, engine.planner = 5, planner
    engine.index = {n: i for i, n in enumerate(field)}
    groups = [None if n in ("scripted", "imugi_v24") else 0 for n in field]
    engine.network = [g is not None for g in groups]
    engine.group_of = groups
    counts = E._split(environments, list(SLICES.values()))
    engine.presets = list(SLICES)
    engine.preset_of = [p for p, c in zip(SLICES, counts) for _ in range(c)]
    engine.environments = environments
    rows = 2 * environments
    shares = player_shares(planner)
    shares_of = [shares.get(n, 0.0) for n in field]
    engine.resident_set = True
    runs = E.resident_runs(groups, shares_of, rows, residents)
    engine.runs = runs
    engine.capacity_of = {i: c for _, m, c in runs for i in m if i is not None}
    engine.leaving = set()
    engine.swaps = 0
    engine.seat = Seat(runs)
    engine.in_use = [0] * len(field)
    engine.long_held = [0] * len(field)
    engine.current = [None] * environments
    engine.next = [None] * environments
    engine.scripted = 0
    engine.blocks = 0
    engine.parked = 0
    engine.games = 0
    if hasattr(E, "LANE_STATE"):  # the fixed tree's extra state
        E.prepare_assignment(engine)
    return engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("environments", type=int)
    parser.add_argument("seconds", type=float)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    random.seed(1)
    started = time.time()
    planner, field, truth, lengths = build()
    print(
        f"built {time.time() - started:.0f}s; unresolved per target:",
        {t.name: int(t.unresolved.sum()) for t in planner.targets if isinstance(t, EloTarget)},
        flush=True,
    )
    engine = make_engine(planner, field, args.environments)
    resident_names = [field[i] for i in engine.seat.lanes[0] if i is not None]
    print("residents:", resident_names, "capacity", engine.runs[0][2], flush=True)
    elo = {t.condition.preset: t for t in planner.targets if isinstance(t, EloTarget)}
    parked_len = {
        "real_scale_frontline": 120.0,
        "real_scale_deathmatch": 90.0,
        "real_scale_elimination": 30.0,
    }

    def duration(env, slot):
        preset = engine.preset_of[env]
        if slot.cell is None:
            return parked_len[preset] * random.uniform(0.6, 1.4)
        c = slot.cell.condition
        base = random.choice(lengths[preset]) if lengths[preset] else random.uniform(30.0, 90.0)
        return max(5.0, base * (max(c.first, c.second) / 5.0) ** 0.5)

    def outcome(slot):
        c = slot.cell
        th = truth[c.condition.preset]
        d = th.get(c.first, 0.0) - th.get(c.second, 0.0)
        d += 0.4 * math.log((c.condition.first + 0.5) / (c.condition.second + 0.5)) * 3
        p = 1.0 / (1.0 + math.exp(-d))
        return Outcome(1.0 if random.random() < p else 0.0, 0, 0, 0.0)

    # start: as BucketEngine.start
    for env in range(engine.environments):
        engine._set_next(env, engine._choose(env))
    for env in range(engine.environments):
        engine._count(env, -1)
        engine.current[env], engine.next[env] = engine.next[env], None
        engine._count(env, 1)
    for env in range(engine.environments):
        engine._set_next(env, engine._choose(env))
    print(f"started {time.time() - started:.0f}s", flush=True)
    res = [i for i in engine.seat.lanes[0] if i is not None]
    print(
        "in_use",
        [
            (
                field[i],
                engine.in_use[i],
                int(
                    elo["real_scale_deathmatch"].unresolved[
                        elo["real_scale_deathmatch"].index[field[i]]
                    ]
                ),
            )
            for i in res
        ],
        "leaving",
        engine.leaving,
        flush=True,
    )
    print(
        "parked by slice",
        collections.Counter(
            engine.preset_of[e]
            for e in range(engine.environments)
            if engine.current[e].cell is None
        ),
        flush=True,
    )
    now = 0.0
    ends = []
    for env in range(engine.environments):
        d = duration(env, engine.current[env])
        engine.current[env].sim_seconds = d
        heapq.heappush(ends, (d, env))
    real_seconds = parked_seconds = 0.0
    real_games = parked_games = 0
    samples = []
    next_rebalance = 4 * BLOCK
    next_sample = 0.0
    while now < args.seconds:
        t, env = heapq.heappop(ends)
        while next_rebalance <= t:
            engine.blocks += 4
            engine._rebalance()
            next_rebalance += 4 * BLOCK
        while next_sample <= t:
            parked_now = sum(1 for s in engine.current if s.cell is None)
            samples.append((next_sample, parked_now, engine.swaps, real_games))
            print(
                f"t={next_sample:6.0f}s parked now {parked_now / engine.environments:.1%}, "
                f"real games {real_games}, parked games {parked_games}, swaps {engine.swaps}, "
                f"leaving {len(engine.leaving)}, wall {time.time() - started:.0f}s",
                flush=True,
            )
            next_sample += 60.0
        now = t
        slot = engine.current[env]
        spent = slot.sim_seconds
        if slot.cell is not None:
            planner.finished(slot.cell, outcome(slot))
            real_games += 1
            real_seconds += spent
        else:
            parked_games += 1
            parked_seconds += spent
        # as BucketEngine._tally's per-environment step
        if hasattr(engine, "_advance"):
            engine._advance(env)
        else:
            engine._count(env, -1)
            engine.current[env], engine.next[env] = engine.next[env], None
            engine._count(env, 1)
            engine._set_next(env, engine._choose(env))
        d = duration(env, engine.current[env])
        engine.current[env].sim_seconds = d
        heapq.heappush(ends, (now + d, env))
    total = real_seconds + parked_seconds
    result = dict(
        label=args.label,
        environments=engine.environments,
        game_seconds=args.seconds,
        real_games=real_games,
        parked_games=parked_games,
        parked_share_env_time=parked_seconds / total,
        real_games_per_env_second=real_games / (engine.environments * args.seconds),
        swaps=engine.swaps,
        wall=time.time() - started,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
