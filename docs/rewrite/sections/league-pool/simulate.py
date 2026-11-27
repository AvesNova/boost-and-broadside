"""Share tracking of the two-level league (lean.learn.pool) against core's capacity rule
(2770a376), simulated on the CPU with the real League: synthetic win rates and game lengths,
the real snapshot schedule at parity (1,048,576 steps an update), L = 1,024 league games.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/league-pool/simulate.py \
        --updates 300 --out docs/rewrite/sections/league-pool/simulation.json

Per update: the target shares (for the two-level league PFSP at the update's start, the
shares level 1 allocated with; for core the PFSP probabilities of its draws, weighted by the
games each tick dealt) and the games dealt
per member. Over windows of 1, 10 and 100 updates: the realised share of games dealt against
the deal-weighted mean target, as total variation (half the sum of absolute differences)
and the largest member error; the scripted share of the games dealt early.

The synthetic learner: win rate against the yardstick 0.15 rising to 0.75 by update 150;
each persona 0.05 x (its index + 1) below that; the EMA 0.5, the uniform average 0.55; a
snapshot 0.5 rising 0.004 an update after it is taken, to 0.95. Game lengths uniform on
[300, 1,928] ticks, ticks simulated 8 at a time (64 steps an update).
"""

import argparse
import dataclasses
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import torch
from torch import nn

from lean.learn import LearnConfig, WeightAverages
from lean.learn.pool import (
    EMA,
    PERSONA_KIND,
    SCRIPTED,
    SCRIPTED_KIND,
    SNAPSHOT_KIND,
    UNIFORM,
    League,
    pfsp,
)

PER_UPDATE = 2048 * 4 * 128
LEAGUE_ENVS = 1024
TICKS = 512
GRAIN = 8
SHORTEST, LONGEST = 300, 1928


class Tiny(nn.Module):
    """A stand-in policy: the league only copies its weights."""

    def __init__(self):
        super().__init__()
        self.config = "league-simulation"
        self.weight = nn.Parameter(torch.zeros(4))


def core_pool():
    """core 2770a376's lean.learn.pool (the capacity rule), loaded from git."""
    source = subprocess.run(
        ["git", "show", "2770a376:lean/learn/pool.py"], capture_output=True, text=True, check=True
    ).stdout
    path = Path(tempfile.mkdtemp()) / "core_pool.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("core_pool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def win_rate(kind: str, index: int, update: int, taken_at: int | None) -> float:
    yardstick = 0.15 + 0.6 * min(1.0, update / 150)
    if kind == SCRIPTED_KIND:
        return yardstick
    if kind == PERSONA_KIND:
        return min(0.98, max(0.02, yardstick - 0.05 * (index + 1)))
    if kind == "ema":
        return 0.5
    if kind == "uniform":
        return 0.55
    if taken_at is None:  # a snapshot member not taken yet (never dealt)
        return 0.5
    return min(0.95, 0.5 + 0.004 * (update - taken_at))


def lengths(count: int, generator) -> torch.Tensor:
    return torch.randint(SHORTEST, LONGEST + 1, (count,), generator=generator)


def two_level(updates: int, seed: int, config: LearnConfig) -> dict:
    generator = torch.Generator().manual_seed(seed)
    policy = Tiny()
    averages = WeightAverages(policy, config)
    league = League(config)
    league.open(policy, averages)
    league.prepare(LEAGUE_ENVS, torch.full((LEAGUE_ENVS,), -1))
    member, slot = league.draw(
        torch.zeros(LEAGUE_ENVS, dtype=torch.long),
        torch.full((LEAGUE_ENVS,), -1),
        torch.ones(LEAGUE_ENVS, dtype=torch.bool),
        generator,
    )
    remaining = lengths(LEAGUE_ENVS, generator)
    taken_at: dict[int, int] = {}
    persona_index = {
        m: m - league.first_persona for m in range(league.first_persona, league.members)
    }
    targets, deals, calls, scripted_slots = [], [], [], []
    for update in range(updates):
        targets.append(league.probabilities().tolist())
        before = league.dealt.clone()
        rates = torch.tensor(
            [
                win_rate(kind, persona_index.get(m, 0), update, taken_at.get(m))
                for m, kind in enumerate(league.kinds)
            ]
        )
        network_calls = 0
        for step in range(TICKS // GRAIN):
            if step % (TICKS // GRAIN // 4) == 0:  # a segment's start
                league.prepare(LEAGUE_ENVS, slot)
                network_calls += 0 if league.table is None else len(league.table.slots)
            remaining = remaining - GRAIN
            done = remaining <= 0
            won = torch.rand(LEAGUE_ENVS, generator=generator) < rates[member.clamp(min=0)]
            league.record(member, done, won, torch.zeros_like(done))
            new_member, new_slot = league.draw(member, slot, done, generator)
            assert torch.equal(new_slot[~done], slot[~done])  # no game changes slot mid-game
            assert bool(league.slot_loaded[new_slot[done]].all())  # deals go to loaded slots
            member, slot = new_member, new_slot
            remaining = torch.where(done, lengths(LEAGUE_ENVS, generator), remaining)
        deals.append((league.dealt - before).tolist())
        calls.append(network_calls / 4)
        scripted_slots.append(league.summary()["scripted_slots"])
        with torch.no_grad():
            policy.weight.add_(0.01)
        averages.update(policy, (update + 1) * 128)
        taken = league.taken
        member = league.after_update(
            policy, averages, update * PER_UPDATE, (update + 1) * PER_UPDATE, member, slot
        )
        if league.taken > taken:
            newest = max(league.ordinal, key=lambda m: league.ordinal[m])
            taken_at[newest] = update + 1
    return {
        "kinds": league.kinds,
        "targets": targets,
        "deals": deals,
        "network_calls": calls,
        "scripted_slots": scripted_slots,
        "overflow_games": float(league.overflow_games),
        "loads": league.loads,
    }


def capacity_rule(updates: int, seed: int, config: LearnConfig, core) -> dict:
    """core's rule on core's membership: the scripted agent (uncapped), the EMA from the
    first snapshot, the uniform average once it holds an update, snapshots; each network
    member C = ceil(1.25 L / 15)."""
    generator = torch.Generator().manual_seed(seed)
    members = 3 + config.snapshot_members
    kinds = [SCRIPTED_KIND, "ema", "uniform"] + [SNAPSHOT_KIND] * config.snapshot_members
    tallies = types.SimpleNamespace(
        games=torch.zeros(members),
        wins=torch.zeros(members),
        draws=torch.zeros(members),
        decay=config.league_decay,
        members=members,
    )
    present = torch.zeros(members, dtype=torch.bool)
    present[SCRIPTED] = True
    capacity = torch.full((members,), math.ceil(1.25 * LEAGUE_ENVS / members))
    capacity[SCRIPTED] = LEAGUE_ENVS
    schedule = core.SnapshotSchedule(config.league_snapshot_steps, config.league_snapshot_period)
    ordinal: dict[int, int] = {}
    taken_at: dict[int, int] = {}
    taken = 0
    member = torch.zeros(LEAGUE_ENVS, dtype=torch.long)
    remaining = lengths(LEAGUE_ENVS, generator)
    targets, deals = [], []
    adam = 0
    for update in range(updates):
        rates = torch.tensor(
            [win_rate(kind, 0, update, taken_at.get(m)) for m, kind in enumerate(kinds)]
        )
        dealt = torch.zeros(members)
        weighted = torch.zeros(members)  # each tick's draw probabilities by its games
        for _ in range(TICKS // GRAIN):
            remaining = remaining - GRAIN
            done = remaining <= 0
            won = torch.rand(LEAGUE_ENVS, generator=generator) < rates[member.clamp(min=0)]
            League.record(tallies, member, done, won, torch.zeros_like(done))
            probability = pfsp(League.win_rates(tallies), present, config.league_floor)
            drawn = core.capacity_draws(
                probability, present, capacity, member, done, generator, members
            )
            dealt += torch.bincount(drawn[done], minlength=members).float()
            weighted += probability * done.sum()
            member = torch.where(done, drawn, member)
            remaining = torch.where(done, lengths(LEAGUE_ENVS, generator), remaining)
        deals.append(dealt.tolist())
        targets.append((weighted / weighted.sum().clamp(min=1e-12)).tolist())
        adam += 128
        if schedule.due(update * PER_UPDATE, (update + 1) * PER_UPDATE):
            kept = core.retained(
                [*ordinal.values(), taken],
                config.recent_snapshots,
                config.snapshot_history_from,
                config.snapshot_members,
            )
            for dropped in [m for m, o in ordinal.items() if o not in kept]:
                present[dropped] = False
                del ordinal[dropped]
                member = torch.where(member == dropped, torch.full_like(member, -1), member)
            index = next(m for m in range(3, members) if m not in ordinal)
            ordinal[index] = taken
            taken_at[index] = update + 1
            taken += 1
            for fresh in (index, EMA):
                if fresh == index or not present[EMA]:
                    present[fresh] = True
                    tallies.games[fresh] = tallies.wins[fresh] = tallies.draws[fresh] = 0.0
        if taken and adam >= config.warmup_steps and not present[UNIFORM]:
            present[UNIFORM] = True
    return {"kinds": kinds, "targets": targets, "deals": deals}


def window_errors(targets, deals, window: int) -> dict:
    """Mean over non-overlapping windows of the total variation and the largest member error
    between the realised share of games dealt and the deal-weighted mean target."""
    target, dealt = torch.tensor(targets), torch.tensor(deals)
    variation, largest = [], []
    for start in range(0, target.shape[0] - window + 1, window):
        games = dealt[start : start + window]
        weight = games.sum(1, keepdim=True)
        if weight.sum() == 0:
            continue
        mean_target = (target[start : start + window] * weight).sum(0) / weight.sum()
        realised = games.sum(0) / games.sum()
        difference = (realised - mean_target).abs()
        variation.append(0.5 * float(difference.sum()))
        largest.append(float(difference.max()))
    if not variation:
        return {"windows": 0}
    return {
        "windows": len(variation),
        "total_variation": sum(variation) / len(variation),
        "largest_member_error": sum(largest) / len(largest),
    }


def scripted_share(result: dict, first: int, last: int) -> float:
    dealt = torch.tensor(result["deals"][first:last]).sum(0)
    scripted = [
        m for m, kind in enumerate(result["kinds"]) if kind in (SCRIPTED_KIND, PERSONA_KIND)
    ]
    yardstick = float(dealt[SCRIPTED] / dealt.sum())
    return float(dealt[scripted].sum() / dealt.sum()), yardstick


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slots", type=int, default=None, help="K (default LearnConfig's)")
    parser.add_argument("--spare", type=int, default=None, help="R (default LearnConfig's)")
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    config = LearnConfig()
    if arguments.slots is not None:
        config = dataclasses.replace(config, league_slots=arguments.slots)
    if arguments.spare is not None:
        config = dataclasses.replace(config, league_spare_slots=arguments.spare)
    began = time.perf_counter()
    results = {
        "two_level": two_level(arguments.updates, arguments.seed, config),
        "capacity_rule": capacity_rule(arguments.updates, arguments.seed, config, core_pool()),
    }
    print(f"simulated {arguments.updates} updates each in {time.perf_counter() - began:.0f} s")
    summary = {}
    for name, result in results.items():
        summary[name] = {
            f"window_{window}": window_errors(result["targets"], result["deals"], window)
            for window in (1, 10, 100)
        }
        for label, first, last in (("updates_0_5", 0, 5), ("updates_0_96", 0, 96)):
            everything, yardstick = scripted_share(result, first, last)
            summary[name][f"scripted_share_{label}"] = everything
            summary[name][f"yardstick_share_{label}"] = yardstick
        if "network_calls" in result:
            calls = result["network_calls"]
            summary[name]["network_calls_mean_0_96"] = sum(calls[:96]) / len(calls[:96])
            summary[name]["network_calls_max"] = max(calls)
            summary[name]["overflow_games"] = result["overflow_games"]
            summary[name]["loads"] = result["loads"]
    print(json.dumps(summary, indent=1))
    if arguments.out is not None:
        arguments.out.write_text(json.dumps({"summary": summary, "results": results}))


if __name__ == "__main__":
    sys.exit(main())
