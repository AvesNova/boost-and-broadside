"""Item 20: game statistics of scripted mirror games of one trigger variant (zone-13's
behaviour.py with the agent read from a parameter file).

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/trigger-20/behaviour.py \
        --agent docs/rewrite/sections/trigger-20/variants/new-k1.json --size 4 --envs 1024 --out DIR

Both teams the same agent (ScriptedConfig() with the file's rows), the
game's defaults (fields and fog on), E environments dealt once and played to their end (at most
the derived longest game, core.longest_game_ticks: no clock). Reads only lean.game (deal,
tick, StepOut, train_angle) and lean.scripted (act), so
the same script measures core's rules and the new ones. Per tick, for ticks before a game's end:

- deaths by cause: a ship that died with collision damage that tick (a collision death), else
  bullet damage (a bullet death), else the boundary;
- collision contacts: ship pairs with collision damage either way that tick; how many kill
  one ship, and how many kill both;
- bullet hits (shooter, target, tick with real-bullet damage on an enemy): the tail-chase share
  (the target's train angle to the shooter beyond 120 deg, lean.eval.style's rule) and the mean
  range (shooter to target at the start of the hit's tick);
- shots: real bullets fired (living ships whose fire action this tick applied: the pending fire);
- game length (mean, median, p90, max) and the share of games that end with one team wiped
  (a win) or both (a draw);
- the map-edge target (core.geometry_record's edge: signed distance beyond the zone's edge) of
  living ships every 15th tick: its quantiles, for the head's range.

Writes DIR/{agent}_{size}v{size}.json (one file per condition, written at the end; a run is
bounded by core.longest_game_ticks)."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from lean import game, scripted
from lean.game import core, train_angle

TAIL = math.radians(120.0)


def agent_config(path: str):
    return scripted.ScriptedConfig().replace(**json.loads(Path(path).read_text()))


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
    bound = core.longest_game_ticks(config)
    length = torch.full((envs,), bound + 1, device=device)  # bound + 1: did not end
    wiped = torch.zeros(envs, dtype=torch.bool, device=device)
    both_wiped = torch.zeros(envs, dtype=torch.bool, device=device)
    edges = []
    edge_column = core.GEOMETRY_COLUMNS.index("edge")
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
    for t in range(bound):
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
        both_wiped = both_wiped | (ended & (left.max(-1).values == 0))
        if t % 15 == 0:
            living = (state.alive & state.exists) & running[:, None]
            edge = core.geometry_record(config, state)[..., edge_column]
            edges.append(edge[living].cpu())
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
    lengths = length.float().cpu() / 30.0
    edge = torch.cat(edges)
    quantiles = torch.tensor([0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0])
    edge_quantiles = torch.quantile(edge[torch.randperm(len(edge))[:5_000_000]], quantiles)
    minutes = float(length.clamp_max(bound).sum()) / 30.0 / 60.0
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
        "mean_length_seconds": float(lengths.mean()),
        "median_length_seconds": float(lengths.median()),
        "p90_length_seconds": float(torch.quantile(lengths, 0.9)),
        "max_length_seconds": float(lengths.max()),
        "bound_seconds": bound / 30.0,
        "unfinished": int((length > bound).sum()),
        "wipe_share": float(wiped.float().mean()),
        "one_team_wiped_share": float((wiped & ~both_wiped).float().mean()),
        "both_wiped_share": float(both_wiped.float().mean()),
        "edge_quantiles": dict(
            zip([f"{q:g}" for q in quantiles.tolist()], edge_quantiles.tolist())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, help="a JSON file of parameter rows")
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
    result["agent"] = Path(arguments.agent).stem
    path = arguments.out / f"{Path(arguments.agent).stem}_{arguments.size}v{arguments.size}.json"
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "counts"}), flush=True)


if __name__ == "__main__":
    main()
