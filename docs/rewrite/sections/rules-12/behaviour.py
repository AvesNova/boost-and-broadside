"""Item 12: game statistics of scripted mirror games under one rule set (the tree it runs in).

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/rules-12/behaviour.py \
        --agent centre --size 4 --envs 512 --out DIR

Both teams the same agent (the round-3 robust centre, or the yardstick ScriptedConfig()), the
game's defaults (fields and fog on), E environments dealt once and played to their end (at most
the clock). Reads only lean.game (deal, tick, StepOut, train_angle) and lean.scripted (act), so
the same script measures core's rules and the new ones. Per tick, for ticks before a game's end:

- deaths by cause: a ship that died with collision damage that tick (a collision death), else
  bullet damage (a bullet death), else the boundary;
- collision contacts: ship pairs with collision damage either way that tick; how many kill
  one ship, and how many kill both;
- bullet hits (shooter, target, tick with real-bullet damage on an enemy): the tail-chase share
  (the target's train angle to the shooter beyond 120 deg, lean.eval.style's rule) and the mean
  range (shooter to target at the start of the hit's tick);
- shots: real bullets fired (living ships whose fire action this tick applied: the pending fire);
- game length and the share of games that end by a wipe before the clock.

Writes DIR/{agent}_{size}v{size}.json (one file per condition, written at the end; a run is
bounded by the clock: at most clock_ticks ticks)."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from lean import game, scripted
from lean.game import train_angle

HERE = Path(__file__).resolve().parent
CENTRE = HERE.parent / "lookahead-personas" / "candidates" / "robust-centre-3.json"
TAIL = math.radians(120.0)


def agent_config(name: str):
    if name == "yardstick":
        return scripted.ScriptedConfig()
    return scripted.ScriptedConfig().replace(**json.loads(CENTRE.read_text()))


@torch.no_grad()
def play(agent: str, size: int, envs: int, seed: int, device: str) -> dict:
    config = game.GameConfig()
    generator = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    sizes = torch.tensor([[size, size]] * envs)
    state = game.reset(config, envs, sizes, generator, device)
    who = agent_config(agent)
    slots = 2 * size
    running = torch.ones(envs, dtype=torch.bool, device=device)
    length = torch.full((envs,), config.clock_ticks, device=device)
    wiped = torch.zeros(envs, dtype=torch.bool, device=device)
    counts = {
        k: torch.zeros((), device=device, dtype=torch.float64)
        for k in (
            "deaths",
            "collision_deaths",
            "bullet_deaths",
            "boundary_deaths",
            "contacts",
            "fatal_contacts",
            "mutual_contacts",
            "bullet_hits",
            "tail_hits",
            "hit_range_sum",
            "shots",
            "ship_ticks",
        )
    }
    enemy = state.team[:, :, None] != state.team[:, None, :]
    upper = torch.ones(slots, slots, dtype=torch.bool, device=device).triu(1)
    started = time.perf_counter()
    for t in range(config.clock_ticks):
        uniforms = scripted.draw_uniforms(envs, slots, generator, device)
        action = scripted.act(who, state, None, None, uniforms)
        before = state
        state, out = game.tick(config, state, action, generator)
        live = running[:, None]
        alive = before.alive & before.exists
        counts["ship_ticks"] += (alive & live).sum()
        counts["shots"] += ((before.pending[..., 2] == 1) & alive & live).sum()
        died = out.died & live
        crash_taken = out.collision_damage.sum(1) > 0  # [attacker, target]: by target
        shot_taken = out.bullet_damage.sum(1) > 0
        counts["deaths"] += died.sum()
        counts["collision_deaths"] += (died & crash_taken).sum()
        counts["bullet_deaths"] += (died & ~crash_taken & shot_taken).sum()
        counts["boundary_deaths"] += (died & ~crash_taken & ~shot_taken).sum()
        pair = ((out.collision_damage + out.collision_damage.transpose(1, 2)) > 0) & upper
        pair = pair & live[..., None]
        both = died[:, :, None] & died[:, None, :]
        one = died[:, :, None] | died[:, None, :]
        counts["contacts"] += pair.sum()
        counts["fatal_contacts"] += (pair & one).sum()
        counts["mutual_contacts"] += (pair & both).sum()
        hits = (out.bullet_damage > 0) & enemy & live[..., None]  # [shooter, target]
        line_x = before.x[:, :, None] - before.x[:, None, :]  # shooter minus target
        line_y = before.y[:, :, None] - before.y[:, None, :]
        angle = train_angle(
            before.heading_cos[:, None, :], before.heading_sin[:, None, :], line_x, line_y
        )
        distance = torch.sqrt(line_x * line_x + line_y * line_y)
        counts["bullet_hits"] += hits.sum()
        counts["tail_hits"] += (hits & (angle.abs() > TAIL)).sum()
        counts["hit_range_sum"] += torch.where(hits, distance, 0.0).sum(dtype=torch.float64)
        ended = out.done & running
        length = torch.where(ended, t + 1, length)
        left = torch.stack([(state.alive & (state.team == k)).sum(-1) for k in (0, 1)], -1)
        wiped = wiped | (ended & (left.min(-1).values == 0))
        running = running & ~out.done
        if t % 300 == 299:
            print(
                f"{time.strftime('%H:%M:%S')} {agent} {size}v{size} tick {t + 1}: "
                f"{int(running.sum())} running, {time.perf_counter() - started:.0f} s",
                file=sys.stderr,
                flush=True,
            )
        if not bool(running.any()):
            break
    c = {k: float(v) for k, v in counts.items()}
    minutes = float(length.sum()) / 30.0 / 60.0
    return {
        "agent": agent,
        "size": size,
        "games": envs,
        "seed": seed,
        "seconds": time.perf_counter() - started,
        "counts": c,
        "game_minutes": minutes,
        "kills_per_minute": c["deaths"] / minutes,
        "collision_death_share": c["collision_deaths"] / max(c["deaths"], 1),
        "bullet_death_share": c["bullet_deaths"] / max(c["deaths"], 1),
        "boundary_death_share": c["boundary_deaths"] / max(c["deaths"], 1),
        "contacts_per_game": c["contacts"] / envs,
        "mutual_share_of_contacts": c["mutual_contacts"] / max(c["contacts"], 1),
        "mutual_share_of_fatal_contacts": c["mutual_contacts"] / max(c["fatal_contacts"], 1),
        "tail_share_of_bullet_hits": c["tail_hits"] / max(c["bullet_hits"], 1),
        "mean_hit_range": c["hit_range_sum"] / max(c["bullet_hits"], 1),
        "bullet_hits_per_ship_minute": c["bullet_hits"] / (c["ship_ticks"] / 1800.0),
        "shots_per_ship_minute": c["shots"] / (c["ship_ticks"] / 1800.0),
        "mean_length_seconds": float(length.float().mean()) / 30.0,
        "wipe_share": float(wiped.float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("centre", "yardstick"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    result = play(arguments.agent, arguments.size, arguments.envs, arguments.seed, arguments.device)
    result["label"] = arguments.label
    path = arguments.out / f"{arguments.agent}_{arguments.size}v{arguments.size}.json"
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "counts"}), flush=True)


if __name__ == "__main__":
    main()
