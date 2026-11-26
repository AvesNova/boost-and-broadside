"""The deflector in play: god-mode passes and kills per minute, off / symmetric / frontal.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/bullet-deflector/godmode.py OUT.json
        [--presets off symmetric frontal] [--envs 128] [--ticks 1800] [--seed 0] [--threads 4]

The rule-based scripted agent (lean.scripted, ScriptedConfig(), one `act` for both teams; the
look-ahead agent is not on this branch) on both sides, CPU.

- God mode: 1v1, fog off, no fields, the standard spawn, `--ticks` ticks; every ship's shield is
  refilled before each tick, so nobody dies (a ship dies only when hurt at 0 shield). Per real
  bullet that damaged its target (the step's own per-bullet damage: core._damage wrapped): the
  tick, the shooter, the hit angle (the bullet's path off the target's nose). A pass is the
  tick holding the closest approach of the two ships' straight relative move (0 <= tau < 1)
  within the gun range (closest approaches within PASS_MERGE of the last pass are the same
  pass); each damaging hit goes to the pass nearest in time. Control: the
  shooter within the gun range, its nose within 30 deg of the line to the target, behind the
  target's beam (the target's nose pointing away from it). Frontal: a hit within 30 deg of the
  target's nose; contacts are bullets that touched (shot_outcome 1), damaging or not.
- Kills per minute: 1v1 and 4v4 with normal damage, GameConfig() otherwise (fog, fields, the
  clock), each environment re-dealt when its game ends, `--ticks` ticks.
Writes one record per preset to OUT (and prints it).
"""

import argparse
import json
import math
import statistics
import time

import torch

from lean import game, scripted
from lean.game import core

CONTROL_NOSE = math.radians(30.0)
FRONTAL = math.radians(30.0)
PASS_MERGE = 1.0  # s: closest approaches this close in time are one pass

_LAST = {}
_original_damage = core._damage


def _recording_damage(*args, **kwargs):
    damage = _original_damage(*args, **kwargs)
    _LAST["damage"] = damage
    return damage


core._damage = _recording_damage


def value(state, name):
    return float(torch.as_tensor(state.constants[name]).flatten()[0])


@torch.no_grad()
def god_mode(preset: str, envs: int, ticks: int, seed: int) -> dict:
    config = game.GameConfig(field_count=0, fog=False, deflector=preset, clock_ticks=10**9)
    generator = torch.Generator().manual_seed(seed)
    sizes = torch.tensor([[1, 1]]).expand(envs, 2)
    state = game.reset(config, envs, sizes, generator, "cpu")
    agent = scripted.ScriptedConfig()
    memory = scripted.fresh_memory(envs, 2, generator, "cpu")
    gun_range, max_shield = value(state, "gun_range"), value(state, "max_shield")
    K = state.bullets.shape[-1]
    h = config.tick_length
    hits = []  # (env, tick, shooter, hit_cos, damage, control)
    contacts = frontal_contacts = 0
    passes = [[] for _ in range(envs)]
    control_ticks = torch.zeros(envs, 2, dtype=torch.long)
    streaks, run = [], torch.zeros(envs, 2, dtype=torch.long)
    damage_total = 0.0
    started = time.time()
    for tick in range(ticks):
        state = state.replace(shield=torch.full_like(state.shield, max_shield))
        uniforms = scripted.draw_uniforms(envs, 2, generator, "cpu")
        action, memory = scripted.act(agent, state, None, None, uniforms, memory)
        new, out = game.tick(config, state, action, generator)
        # Control at tick start (shooter s = 0, 1 against the other).
        gap_x, gap_y = state.x[:, 1] - state.x[:, 0], state.y[:, 1] - state.y[:, 0]
        distance = torch.hypot(gap_x, gap_y).clamp_min(1e-6)
        control = []
        for shooter in (0, 1):
            sign = 1.0 if shooter == 0 else -1.0
            line_x, line_y = sign * gap_x / distance, sign * gap_y / distance  # to the target
            other = 1 - shooter
            aim = state.heading_cos[:, shooter] * line_x + state.heading_sin[:, shooter] * line_y
            away = state.heading_cos[:, other] * line_x + state.heading_sin[:, other] * line_y
            control.append((distance < gun_range) & (aim > math.cos(CONTROL_NOSE)) & (away > 0.0))
        control = torch.stack(control, 1)
        control_ticks += control.long()
        ended = (run > 0) & ~control
        streaks.extend((run[ended].float() * h).tolist())
        run = torch.where(control, run + 1, 0)
        # Passes.
        move_x = (new.x[:, 1] - state.x[:, 1]) - (new.x[:, 0] - state.x[:, 0])
        move_y = (new.y[:, 1] - state.y[:, 1]) - (new.y[:, 0] - state.y[:, 0])
        squared = (move_x * move_x + move_y * move_y).clamp_min(1e-12)
        tau = -(gap_x * move_x + gap_y * move_y) / squared
        miss = torch.hypot(gap_x + tau * move_x, gap_y + tau * move_y)
        passing = (tau >= 0) & (tau < 1) & (miss < gun_range)
        for e in passing.nonzero()[:, 0].tolist():
            if not passes[e] or tick - passes[e][-1] >= PASS_MERGE / h:
                passes[e].append(tick)
        # Bullets.
        damage = _LAST["damage"].reshape(envs, 2, K)
        damage_total += float(damage.sum())
        # A bullet ending this tick kept its slot (the fire stage writes before the move), so
        # the new state's real flag is its own.
        real_touch = (out.shot_outcome == 1) & new.bullet_real
        direction_x, direction_y = new.bullet_dir_x, new.bullet_dir_y
        nose_x = new.heading_cos.flip(1)[..., None]  # the target's (the other ship's) nose
        nose_y = new.heading_sin.flip(1)[..., None]
        hit_cos = -(direction_x * nose_x + direction_y * nose_y)
        contacts += int(real_touch.sum())
        frontal_contacts += int((real_touch & (hit_cos > math.cos(FRONTAL))).sum())
        for e, shooter, k in (damage > 0.0).nonzero().tolist():
            hits.append(
                (
                    e,
                    tick,
                    shooter,
                    float(hit_cos[e, shooter, k]),
                    float(damage[e, shooter, k]),
                    bool(control[e, shooter]),
                )
            )
        state = new
    streaks.extend((run[run > 0].float() * h).tolist())
    # Hits per pass.
    per_pass = []
    for e in range(envs):
        times = passes[e]
        if not times:
            continue
        counts = [0] * len(times)
        for he, tick, *_ in hits:
            if he == e:
                nearest = min(range(len(times)), key=lambda i: abs(times[i] - tick))
                counts[nearest] += 1
        per_pass.extend(counts)
    gaps = [(b - a) * h for e in range(envs) for a, b in zip(passes[e], passes[e][1:])]
    frontal_hits = sum(1 for hit in hits if hit[3] > math.cos(FRONTAL))
    in_control = sum(1 for hit in hits if hit[5])
    seconds = envs * ticks * h
    return dict(
        preset=preset,
        games=envs,
        seconds_each=ticks * h,
        wall_s=round(time.time() - started, 1),
        damaging_hits=len(hits),
        damaging_hits_per_minute_per_game=60.0 * len(hits) / seconds,
        bullet_damage_per_minute_per_game=60.0 * damage_total / seconds,
        contacts=contacts,
        frontal_contacts=frontal_contacts,
        frontal_damaging_hits=frontal_hits,
        frontal_damage=sum(hit[4] for hit in hits if hit[3] > math.cos(FRONTAL)),
        passes=len(per_pass),
        pass_interval_median_s=statistics.median(gaps) if gaps else None,
        hits_per_pass_mean=statistics.mean(per_pass) if per_pass else None,
        passes_with_no_hit=sum(c == 0 for c in per_pass) / max(len(per_pass), 1),
        passes_with_11_or_more=sum(c >= 11 for c in per_pass) / max(len(per_pass), 1),
        hits_in_control_share=in_control / max(len(hits), 1),
        control_time_share=float(control_ticks.sum()) / (2 * envs * ticks),
        control_streak_median_s=statistics.median(streaks) if streaks else None,
        control_streak_longest_s=max(streaks) if streaks else None,
    )


@torch.no_grad()
def kills_per_minute(preset: str, size: int, envs: int, ticks: int, seed: int) -> dict:
    config = game.GameConfig(deflector=preset)
    generator = torch.Generator().manual_seed(seed)
    sizes = torch.tensor([[size, size]]).expand(envs, 2)
    state = game.reset(config, envs, sizes, generator, "cpu")
    slots = state.exists.shape[1]
    agent = scripted.ScriptedConfig()
    memory = scripted.fresh_memory(envs, slots, generator, "cpu")
    deaths = games = draws = 0
    started = time.time()
    for _ in range(ticks):
        uniforms = scripted.draw_uniforms(envs, slots, generator, "cpu")
        action, memory = scripted.act(agent, state, None, None, uniforms, memory)
        state, out = game.tick(config, state, action, generator)
        deaths += int(out.died.sum())
        games += int(out.done.sum())
        draws += int((out.done & (out.result == 2)).sum())
        if bool(out.done.any()):
            fresh = game.deal(config, envs, slots, sizes, generator, "cpu")
            state = game.take_deal(state, out.done, fresh)
            new_memory = scripted.fresh_memory(envs, slots, generator, "cpu")
            memory = {
                name: torch.where(out.done[:, None], new_memory[name], memory[name])
                for name in memory
            }
    minutes = envs * ticks * config.tick_length / 60.0
    return dict(
        preset=preset,
        size=size,
        envs=envs,
        wall_s=round(time.time() - started, 1),
        kills_per_minute=deaths / minutes,
        games_finished=games,
        draws=draws,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--presets", nargs="+", default=["off", "symmetric", "frontal"])
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--ticks", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    records = []
    for preset in args.presets:
        record = dict(god_mode=god_mode(preset, args.envs, args.ticks, args.seed))
        print(json.dumps(record), flush=True)
        for size in (1, 4):
            record[f"kills_{size}v{size}"] = kills_per_minute(
                preset, size, args.envs, args.ticks, args.seed
            )
            print(json.dumps(record[f"kills_{size}v{size}"]), flush=True)
        records.append(record)
        with open(args.out, "w") as handle:
            json.dump(records, handle, indent=1)


if __name__ == "__main__":
    main()
