"""What becomes of spawns near the edge: in a preset's matches (the scripted agent on both
sides, batched on the CPU, uncompiled), the share of spawns that die to the boundary (a
death with no shooter) within `window` ticks, the share that leave the arena in that time,
and the boundary damage per spawn. Only respawns with a whole window ahead count (a match's
first spawns are its reset, not a respawn).

hold: ticks a spawned ship flies straight (coasting, no shot) before the agent takes over,
a slow pilot; 0 the agent itself.

Usage: `python bench/spawn_boundary_deaths.py [preset=arena] [environments=128]
[ticks=6000] [hold=0] [window=150] [seed=7]`
Prints progress lines, then one JSON line.
"""

import dataclasses
import json
import sys
import time

import torch

from bnb.contracts import Action, Obs
from bnb.controllers import scripted
from bnb.game import api
from bnb.game.handling.characteristics import characteristics


def measure(preset: str, environments: int, ticks: int, hold: int, window: int, seed: int) -> dict:
    """The module's measurement of one preset's spawns (the module docstring): a dict."""
    torch.manual_seed(seed)  # the agent's draws
    config = api.PRESETS[preset]
    types = api.ship_types(ship_set=config.ship_set)
    rules = api.rules_of(config, types=types)
    lineup = api.preset_lineup(preset)
    if lineup is not None:
        rules = dataclasses.replace(rules, team_types=tuple(lineup))
    B, S, N = environments, rules.ships_per_team, rules.num_ships
    draws = torch.Generator().manual_seed(seed + 1)
    fogged = torch.ones(B, dtype=torch.bool)
    state = api.reset(rules, types, api.uniforms(rules, B, generator=draws), fogged)
    state, observation, _ = api.perceive(rules, state)
    agent = scripted.Scripted(characteristics(types, rules), S)
    carry = agent.init_carry(2 * B, torch.device("cpu"))
    age = torch.full((B, N), -1)  # ticks since the followed spawn; -1 none followed
    outside = torch.zeros((B, N), dtype=torch.bool)
    spawns = boundary_deaths = shot_deaths = left = 0
    damage = 0.0
    started = time.time()
    with torch.no_grad():
        for tick in range(ticks):
            view = Obs(
                **{
                    name: getattr(observation, name).flatten(0, 1)
                    for name in ("ships", "fields", "globals", "zones")
                }
            )
            first = torch.full((2 * B,), tick == 0)
            action, carry, _ = agent.step(carry, view, first)
            packed = action.packed().reshape(B, N, 3)
            holding = ((age >= 0) & (age < hold))[..., None]
            packed = torch.where(holding, 0, packed).to(torch.int32)  # coast, straight
            moved, sweep = api.move(
                rules, types, state, Action.unpack(packed), api.spread(rules, B, generator=draws)
            )
            state, events = api.resolve(
                rules,
                types,
                moved,
                sweep,
                Action.unpack(packed),
                api.uniforms(rules, B, generator=draws),
            )
            state, observation, _ = api.perceive(rules, state)
            following = age >= 0
            age = torch.where(following, age + 1, age)
            outside |= following & (events.boundary_damage > 0)
            damage += float(torch.where(following, events.boundary_damage, 0.0).sum())
            dying = following & events.died
            boundary_deaths += int((dying & (events.killer < 0)).sum())
            shot_deaths += int((dying & (events.killer >= 0)).sum())
            closing = dying | (age >= window)
            left += int((closing & outside).sum())
            age = torch.where(closing, -1, age)
            outside &= ~closing
            if tick < ticks - window:
                new = events.spawned & ~events.done[:, None]
                spawns += int(new.sum())
                age = torch.where(new, 0, age)
                outside &= ~new
            if tick % 600 == 0:
                print(
                    f"tick {tick}/{ticks}: {spawns} spawns, {time.time() - started:.0f} s",
                    file=sys.stderr,
                    flush=True,
                )
    share = max(spawns, 1)
    return dict(
        preset=preset,
        environments=B,
        ticks=ticks,
        hold=hold,
        window=window,
        spawns=spawns,
        boundary_deaths=boundary_deaths,
        boundary_death_share=boundary_deaths / share,
        shot_deaths=shot_deaths,
        left_arena=left,
        left_arena_share=left / share,
        boundary_damage_per_spawn=damage / share,
    )


if __name__ == "__main__":
    arguments = dict(argument.split("=", 1) for argument in sys.argv[1:])
    print(
        json.dumps(
            measure(
                arguments.get("preset", "arena"),
                int(arguments.get("environments", 128)),
                int(arguments.get("ticks", 6000)),
                int(arguments.get("hold", 0)),
                int(arguments.get("window", 150)),
                int(arguments.get("seed", 7)),
            )
        )
    )
