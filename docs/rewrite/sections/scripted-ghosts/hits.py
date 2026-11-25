"""Where scripted hits strike and how much of their nominal damage they deal
(docs/rewrite/sections/scripted-ghosts/notes.md; the method of shield-target/notes.md).

    hits.py OUT [--envs 256] [--ticks 1800] [--no-fields] [--seed 0]

Scripted yardstick on both teams, CPU, GameConfig(training_spawn=True) (fields and fog on
unless --no-fields), the training team sizes, finished games re-dealt with a fresh memory.
Every real hit is recorded through lean.game.core._damage: its damage, its nominal damage
(the same call with the head-on fraction set to 1, so the aspect factor is 1) and the hit
angle's cosine (1 at the target's nose). Reports the share of hits within 25 degrees of the
nose, the share of nominal damage dealt, kills, shots fired. Writes OUT (json).
"""

import argparse
import dataclasses
import json
import math
import sys
import time

import torch

import lean
from lean import game, scripted
from lean.contract import Constants
from lean.game import core
from lean.learn.config import LearnConfig
from lean.learn.rollout import draw_team_sizes, scripted_actions

RECORD = []
ORIGINAL = core._damage


def traced(state, flown, bullets, real, contact):
    out = ORIGINAL(state, flown, bullets, real, contact)
    c = state.constants
    assert "head_on" in c.uniform  # one ship type: a number
    nominal_constants = Constants({**c.uniform, "head_on": 1.0}, c.columns, c.table)
    nominal = ORIGINAL(
        dataclasses.replace(state, constants=nominal_constants), flown, bullets, real, contact
    )
    E, _, N, K = bullets.shape
    direction_x = bullets[:, core.BULLET_DIR_X].reshape(E, N * K)
    direction_y = bullets[:, core.BULLET_DIR_X + 1].reshape(E, N * K)
    nose_cos = core._on_target(flown.heading_cos, contact.target)
    nose_sin = core._on_target(flown.heading_sin, contact.target)
    hit_cos = -(direction_x * nose_cos + direction_y * nose_sin)
    mask = contact.hit & real
    RECORD.append(torch.stack([out[mask], nominal[mask], hit_cos[mask]], -1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--ticks", type=int, default=1800)
    parser.add_argument("--no-fields", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    core._damage = traced
    torch.manual_seed(arguments.seed)
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    learn = LearnConfig()
    config = game.GameConfig(training_spawn=True)
    if arguments.no_fields:
        config = dataclasses.replace(config, field_count=0)
    envs = arguments.envs
    state = game.reset(config, envs, draw_team_sizes(learn, envs, generator), generator, device)
    slots = state.exists.shape[1]
    agent = scripted.ScriptedConfig()
    persona = scripted.personas(agent, envs, slots, generator, device, learn.persona_spread)
    memory = scripted.fresh_memory(envs, slots, generator, device)
    shots = kills = games = 0
    started = time.time()
    with torch.no_grad():
        for tick in range(arguments.ticks):
            move = scripted_actions(agent, state, persona, memory, generator)
            shots += int((move.action[..., 2] == 1).logical_and(state.alive).sum())
            after, out = game.tick(config, state, move.action, generator)
            kills += int((state.alive & ~after.alive & state.exists).sum())
            done = out.done
            games += int(done.sum())
            fresh = game.deal(
                config, envs, slots, draw_team_sizes(learn, envs, generator), generator, device
            )
            state = game.take_deal(after, done, fresh)
            renewed = scripted.fresh_memory(envs, slots, generator, device)
            memory = {
                name: torch.where(done.view(-1, *[1] * (value.dim() - 1)), renewed[name], value)
                for name, value in move.memory.items()
            }
            if tick % 300 == 299:
                rate = (tick + 1) / (time.time() - started)
                eta = time.strftime("%H:%M", time.localtime(started + arguments.ticks / rate))
                print(f"{time.strftime('%H:%M')} tick {tick + 1}, ETA {eta}", file=sys.stderr)
    rows = torch.cat(RECORD)
    damage, nominal, hit_cos = rows.unbind(-1)
    near_nose = hit_cos > math.cos(math.radians(25.0))
    result = {
        "lean": lean.__file__,
        "envs": envs,
        "ticks": arguments.ticks,
        "fields": not arguments.no_fields,
        "hits": int(len(damage)),
        "hits_within_25_deg_of_nose": float(near_nose.float().mean()),
        "damage": float(damage.sum()),
        "nominal_damage": float(nominal.sum()),
        "share_of_nominal_dealt": float(damage.sum() / nominal.sum()),
        "damage_per_hit": float(damage.mean()),
        "shots": shots,
        "hits_per_shot": len(damage) / max(shots, 1),
        "kills": kills,
        "games_ended": games,
        "kills_per_env_minute": kills / (envs * arguments.ticks * config.tick_length / 60.0),
    }
    print(json.dumps(result, indent=1))
    with open(arguments.out, "w") as handle:
        json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
