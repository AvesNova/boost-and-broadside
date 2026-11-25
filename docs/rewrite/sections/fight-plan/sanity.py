"""Fight plan: a sanity measurement (docs/rewrite/sections/fight-plan/progress.md).

    PYTHONPATH=<tree> uv run --no-sync python docs/rewrite/sections/fight-plan/sanity.py OUT.json
        [--sizes 1 4] [--envs 256] [--seed 0] [--threads 4]

The yardstick on both teams (ScriptedConfig(), one `act` call for both), CPU, the game as
evaluated (GameConfig(): fog, the standard spawn, the 1,800-tick clock), without fields and with
them; every game runs to its end (no re-deal). Runs against whichever tree is on PYTHONPATH.

A head-on pass is a swept closest approach between two living ships inside HEAD_ON_PASS metres
with the noses at least HEAD_ON_NOSES apart (as nose-cone/sanity.py). The plan is read from the
memory: the sign of "plan" (this branch) or of "last_pass" (core before it), +1 two-circle, -1
one-circle, 0 none. An engagement starts when a living ship's plan goes from 0 to a sign; a
switch is a sign going to the other sign; an engagement ends when the plan returns to 0 (the
target changed or was lost) or the ship dies or the game ends. The energy advantage at each
engagement's start is recorded as a histogram. The geometry actually flown: of ship-ticks in the
turning tactic (weight over a half), the share with opposite rotations (flow < -0.5: one-circle)
and with the same rotation (flow > 0.5: two-circle).
"""

import argparse
import json
import math
import sys
import time

import torch

from lean import game, scripted
from lean.scripted.opponent import explain

HEAD_ON_PASS = 100.0  # m
HEAD_ON_NOSES = math.radians(150.0)
ADVANTAGE_EDGES = [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]


def plan_of(memory: dict) -> torch.Tensor:
    return torch.sign(memory["plan"] if "plan" in memory else memory["last_pass"])


@torch.no_grad()
def measure(size: int, fields: bool, envs: int, seed: int) -> dict:
    base = game.GameConfig()
    config = base if fields else game.GameConfig(field_count=0)
    generator = torch.Generator().manual_seed(seed)
    sizes = torch.tensor([[size, size]]).expand(envs, 2)
    state = game.reset(config, envs, sizes, generator, "cpu")
    slots = state.exists.shape[1]
    agent = scripted.ScriptedConfig()
    memory = scripted.fresh_memory(envs, slots, generator, "cpu")
    h = config.tick_length
    upper = torch.triu(torch.ones(slots, slots, dtype=torch.bool), 1)
    over = torch.zeros(envs, dtype=torch.bool)
    length = torch.zeros(envs, dtype=torch.long)
    result = torch.full((envs,), -1, dtype=torch.long)
    tally = dict(
        deaths=0,
        head_on_passes=0,
        engagements=0,
        engagements_two_circle=0,
        engagements_one_circle=0,
        switches=0,
        commits=0,
        ship_ticks_one_circle=0,
        ship_ticks_two_circle=0,
        turning_ticks=0,
        turning_ticks_opposite_rotation=0,
        turning_ticks_same_rotation=0,
    )
    advantage_counts = [0] * (len(ADVANTAGE_EDGES) + 1)
    started = time.time()
    for tick in range(config.clock_ticks):
        uniforms = scripted.draw_uniforms(envs, slots, generator, "cpu")
        before_plan = plan_of(memory)
        before_sense = memory["sense"]
        reasons = explain(agent, state, None, None, memory)
        action, memory = scripted.act(agent, state, None, None, uniforms, memory)
        playing = ~over
        alive0 = state.alive & state.exists & playing[:, None]
        after_plan = plan_of(memory)
        started_now = alive0 & (before_plan == 0) & (after_plan != 0)
        tally["engagements"] += int(started_now.sum())
        tally["engagements_two_circle"] += int((started_now & (after_plan > 0)).sum())
        tally["engagements_one_circle"] += int((started_now & (after_plan < 0)).sum())
        tally["switches"] += int((alive0 & (before_plan * after_plan < 0)).sum())
        tally["commits"] += int((alive0 & (before_sense == 0) & (memory["sense"] != 0)).sum())
        tally["ship_ticks_two_circle"] += int((alive0 & (after_plan > 0)).sum())
        tally["ship_ticks_one_circle"] += int((alive0 & (after_plan < 0)).sum())
        turning = alive0 & (reasons.weights["turning"] > 0.5)
        flow = reasons.situation.flow
        tally["turning_ticks"] += int(turning.sum())
        tally["turning_ticks_opposite_rotation"] += int((turning & (flow < -0.5)).sum())
        tally["turning_ticks_same_rotation"] += int((turning & (flow > 0.5)).sum())
        if bool(started_now.any()):
            advantage = reasons.situation.energy_advantage[started_now]
            bucket = torch.bucketize(advantage, torch.tensor(ADVANTAGE_EDGES))
            for index, count in enumerate(torch.bincount(bucket, minlength=len(advantage_counts))):
                advantage_counts[index] += int(count)
        new, out = game.tick(config, state, action, generator)
        gap_x = state.x[:, None, :] - state.x[:, :, None]
        gap_y = state.y[:, None, :] - state.y[:, :, None]
        move_x = (new.x - state.x)[:, None, :] - (new.x - state.x)[:, :, None]
        move_y = (new.y - state.y)[:, None, :] - (new.y - state.y)[:, :, None]
        move_squared = (move_x * move_x + move_y * move_y).clamp_min(1e-12)
        tau = -(gap_x * move_x + gap_y * move_y) / move_squared
        clamped = tau.clamp(0.0, 1.0)
        miss = torch.hypot(gap_x + clamped * move_x, gap_y + clamped * move_y)
        in_tick = (tau >= 0) & (tau < 1) & alive0[:, :, None] & alive0[:, None, :] & upper
        noses = (
            new.heading_cos[:, :, None] * new.heading_cos[:, None, :]
            + new.heading_sin[:, :, None] * new.heading_sin[:, None, :]
        )
        head_on = in_tick & (miss < HEAD_ON_PASS) & (noses < math.cos(HEAD_ON_NOSES))
        tally["head_on_passes"] += int(head_on.sum())
        tally["deaths"] += int((out.died & alive0).sum())
        finished = out.done & playing
        length = torch.where(finished, tick + 1, length)
        result = torch.where(finished, out.result, result)
        over = over | out.done
        state = new
        if bool(over.all()):
            break
    games = envs
    minutes = float(length.sum()) * h / 60.0
    engagements = max(tally["engagements"], 1)
    record = dict(
        size=size,
        fields=fields,
        games=games,
        seconds=round(time.time() - started, 1),
        draws=int((result == 2).sum()),
        team_zero_wins=int((result == 0).sum()),
        mean_length_s=float(length.float().mean()) * h,
        **tally,
        head_on_passes_per_kill=tally["head_on_passes"] / max(tally["deaths"], 1),
        kills_per_minute=tally["deaths"] / max(minutes, 1e-9),
        share_two_circle=tally["engagements_two_circle"] / engagements,
        share_one_circle=tally["engagements_one_circle"] / engagements,
        switches_per_engagement=tally["switches"] / engagements,
        commits_per_engagement=tally["commits"] / engagements,
        turning_share_one_circle_geometry=tally["turning_ticks_opposite_rotation"]
        / max(tally["turning_ticks"], 1),
        turning_share_two_circle_geometry=tally["turning_ticks_same_rotation"]
        / max(tally["turning_ticks"], 1),
        advantage_edges=ADVANTAGE_EDGES,
        advantage_at_start=advantage_counts,
    )
    print(json.dumps(record), file=sys.stderr, flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    records = []
    for size in arguments.sizes:
        for fields in (False, True):
            records.append(measure(size, fields, arguments.envs, arguments.seed))
            with open(arguments.out, "w") as handle:  # each run persisted as it lands
                json.dump(records, handle, indent=1)
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
