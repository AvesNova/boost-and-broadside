"""Constant-ship-density Frontline map scaling, and what it does to match pace.

The 5v5 map is the reference: 10 ships inside a 2600 px playable disk, five
zones on a 1200 px ring, ten fields drawn from [30, 750] px. Every other fleet
size is that map rescaled by a single linear factor

    s = sqrt(num_ships / 10)

applied to every Frontline length. Areas go as ``s**2 = num_ships / 10``, so
ships per unit area, zone area per ship, and field area per ship are all held at
their 5v5 values. Zone and field *counts* never change, and the zone ring keeps
a constant fraction of the playable radius, so the map is a literal zoom of the
5v5 layout rather than a differently shaped one.

The world period is 65536 px, four times the historical Frontline contract.
2600 * sqrt(10) = 8222 px exceeds the 16384 toroid's 8192 px half-period, so
exact density preservation at 50v50 is unreachable there; at 65536 the
half-period is 32768 px and the constraint does not bind until ~1590 ships.
Because 65536 is a power of two, ``position_fourier_frequencies`` grows 8 -> 10
by adding two *coarse* harmonics and the finest period stays exactly 128 px.

Ship physics is deliberately not scaled: hull size, speed, and weapon range are
what make a 5v5 engagement feel the way it does, and holding areal density fixed
while they stay fixed is the whole point. The consequence is that traversal
takes ``s`` times longer, which is what this benchmark measures.

Usage:
    uv run --no-sync python benchmarks/frontline_density_scaling.py \
        --scenario 50v50 --envs 64 --out artifacts/benchmarks/density.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config.core import EnvConfig, ShipConfig
from boost_and_broadside.env import env as env_module
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import frontline_ship_config
from boost_and_broadside.env.perception import team_visibility_from_state
from boost_and_broadside.profiles import PROFILES

# --- The scaling contract -------------------------------------------------

#: Fleet the shipped Frontline geometry was tuned for; the density reference.
BASELINE_SHIPS = 10

#: Four times the historical 16384 px contract. A power of two, so the base-2
#: position basis extends by two coarse harmonics with the finest period fixed.
SCALED_WORLD = 65536.0

#: Ships may drift past the soft boundary before dying; a playable disk is only
#: well defined while its diameter stays inside the toroid's period.
_HALF_PERIOD_MARGIN = 0.95


def frontline_scale(num_ships: int) -> float:
    """Linear factor holding ship areal density at its ``BASELINE_SHIPS`` value."""

    if num_ships < 2:
        raise ValueError(f"num_ships must be at least 2, got {num_ships}")
    return math.sqrt(num_ships / BASELINE_SHIPS)


def max_density_preserving_ships(world: float = SCALED_WORLD) -> int:
    """Largest fleet whose density-preserving playable disk fits the toroid."""

    baseline = PROFILES["rl"].frontline.playable_radius
    limit = _HALF_PERIOD_MARGIN * 0.5 * world
    return int(BASELINE_SHIPS * (limit / baseline) ** 2)


def scaled_configs(
    num_ships: int, *, scale_vision: bool, max_episode_steps: int
) -> tuple[ShipConfig, EnvConfig]:
    """The (ship, env) pair for ``num_ships`` at constant 5v5 ship density."""

    profile = PROFILES["rl"]
    s = frontline_scale(num_ships)
    base_ship = frontline_ship_config(profile.ship_config)
    base_front = profile.frontline

    playable = base_front.playable_radius * s
    if playable >= _HALF_PERIOD_MARGIN * 0.5 * SCALED_WORLD:
        raise ValueError(
            f"{num_ships} ships need a {playable:.0f} px playable radius, which does not fit "
            f"the {SCALED_WORLD:.0f} px toroid (max {max_density_preserving_ships()} ships)"
        )

    ship_config = replace(
        base_ship,
        world_size=(SCALED_WORLD, SCALED_WORLD),
        field_radius_min=base_ship.field_radius_min * s,
        field_radius_max=base_ship.field_radius_max * s,
        field_transition_width_min=base_ship.field_transition_width_min * s,
        field_transition_width_max=base_ship.field_transition_width_max * s,
    )
    frontline = replace(
        base_front,
        zone_radius=base_front.zone_radius * s,
        zone_ring_radius=base_front.zone_ring_radius * s,
        playable_radius=playable,
    )
    env_config = EnvConfig(
        num_ships=num_ships,
        num_fields=profile.num_fields,
        max_bullets=profile.max_bullets,
        max_episode_steps=max_episode_steps,
        action_repeat=profile.action_repeat,
        spawn_resource_spread=profile.spawn_resource_spread,
        # Sight is a weapon-scale length, not a map-scale one. Left unscaled,
        # constant areal density times a constant sight area keeps the number of
        # ships inside a ship's sight at its 5v5 value -- which is the invariant
        # that makes a local engagement feel the same. Scaling it instead holds
        # "fraction of the map visible" constant and multiplies ships-in-sight
        # by the fleet ratio.
        vision_range=profile.vision_range * (s if scale_vision else 1.0),
        zones_occlude=profile.zones_occlude,
        frontline=frontline,
    )
    return ship_config, env_config


def scaled_agent_config(num_ships: int) -> StochasticAgentConfig:
    """Scripted tunables, with the map-scale length scaled and the rest fixed.

    ``frontline_zone_radius`` is an objective-geometry length and has to track
    the zone it stands for. ``frontline_combat_radius`` and
    ``frontline_separation_radius`` are weapon- and hull-scale and must not move,
    for the same reason ship physics does not.
    """

    s = frontline_scale(num_ships)
    base = StochasticAgentConfig()
    zone_radius = None if base.frontline_zone_radius is None else base.frontline_zone_radius * s
    return replace(base, frontline_zone_radius=zone_radius)


# --- Measurement ----------------------------------------------------------


@dataclass
class MatchStats:
    """Per-environment capture and match timing, in decisions."""

    first_done: torch.Tensor  # (B,) step index of match end, or -1
    capture_steps: list[list[int]]  # per env, step indices of completed captures
    results: torch.Tensor  # (B,) MatchResult at first done


def run_scenario(
    num_ships: int,
    num_envs: int,
    max_steps: int,
    device: torch.device,
    *,
    scale_vision: bool,
    seed: int = 0,
    progress_every: int = 0,
) -> tuple[MatchStats, EnvConfig, ShipConfig, float]:
    ship_config, env_config = scaled_configs(
        num_ships, scale_vision=scale_vision, max_episode_steps=max_steps
    )
    # The frontline mode asserts the historical contract world; this experiment
    # is the proposal to change it, so the assertion is relaxed for the run.
    env_module.FRONTLINE_WORLD_SIZE = (SCALED_WORLD, SCALED_WORLD)

    torch.manual_seed(seed)
    env = TensorEnv(num_envs, ship_config, env_config, device)
    env.reset(seed=seed)
    agent = StochasticScriptedAgent(ship_config, scaled_agent_config(num_ships))

    first_done = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    results = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    capture_steps: list[list[int]] = [[] for _ in range(num_envs)]

    start = time.perf_counter()
    with torch.no_grad():
        for step in range(max_steps):
            visibility = team_visibility_from_state(
                env.state, ship_config, env_config, perceive_bullets=False
            )
            action = agent.get_actions(env.state, visibility.ship)
            dones, _ = env.step(action.int())

            live = first_done < 0
            captured = (env.state.front_delta != 0) & live
            if bool(captured.any()):
                for idx in captured.nonzero(as_tuple=True)[0].tolist():
                    capture_steps[idx].append(step)

            if progress_every and step % progress_every == 0:
                done_n = int((first_done >= 0).sum())
                rate = (step + 1) / max(1e-9, time.perf_counter() - start)
                remaining = (max_steps - step) / max(1e-9, rate)
                print(
                    f"[{num_ships // 2}v{num_ships // 2}] step {step}/{max_steps} "
                    f"finished={done_n}/{num_envs} {rate:.1f} steps/s "
                    f"worst-case {remaining / 60:.1f} min left",
                    flush=True,
                )

            newly = dones & live
            if bool(newly.any()):
                first_done = torch.where(newly, step, first_done)
                results = torch.where(newly, env.state.match_result.long(), results)
                if bool((first_done >= 0).all()):
                    break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (
        MatchStats(first_done.cpu(), capture_steps, results.cpu()),
        env_config,
        ship_config,
        elapsed,
    )


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p10": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "min": ordered[0],
        "max": ordered[-1],
    }


def report(
    name: str,
    stats: MatchStats,
    env_config: EnvConfig,
    ship_config: ShipConfig,
    elapsed: float,
    max_steps: int,
) -> dict:
    hz = 1.0 / (ship_config.dt * env_config.action_repeat)
    num_envs = stats.first_done.numel()
    finished = stats.first_done >= 0
    decided = finished & (stats.first_done < max_steps - 1)

    game_seconds = [float(v) / hz for v in stats.first_done[decided].tolist()]
    first_capture: list[float] = []
    intervals: list[float] = []
    capture_counts: list[int] = []
    for idx in range(num_envs):
        end = int(stats.first_done[idx]) if finished[idx] else max_steps
        events = [c for c in stats.capture_steps[idx] if c <= end]
        capture_counts.append(len(events))
        if events:
            first_capture.append(events[0] / hz)
            intervals.extend((b - a) / hz for a, b in zip(events, events[1:]))

    front = env_config.frontline
    s = frontline_scale(env_config.num_ships)
    ships_per_megapixel = env_config.num_ships / (math.pi * front.playable_radius**2) * 1e6
    return {
        "scenario": name,
        "num_ships": env_config.num_ships,
        "num_envs": num_envs,
        "scale": round(s, 4),
        "world_size": ship_config.world_size[0],
        "playable_radius": round(front.playable_radius, 1),
        "zone_ring_radius": round(front.zone_ring_radius, 1),
        "zone_radius": round(front.zone_radius, 1),
        "field_radius_range": [
            round(ship_config.field_radius_min, 1),
            round(ship_config.field_radius_max, 1),
        ],
        "vision_range": env_config.vision_range,
        "ring_fraction_of_playable": round(front.zone_ring_radius / front.playable_radius, 4),
        "ships_per_megapixel": round(ships_per_megapixel, 3),
        "decision_hz": hz,
        "max_steps": max_steps,
        "decided": int(decided.sum()),
        "timed_out": int((finished & ~decided).sum()) + int((~finished).sum()),
        "game_seconds": _summary(game_seconds),
        "seconds_to_first_capture": _summary(first_capture),
        "seconds_between_captures": _summary(intervals),
        "captures_per_match": _summary([float(c) for c in capture_counts]),
        "results": {
            "team0": int((stats.results == 0).sum()),
            "team1": int((stats.results == 1).sum()),
            "draw": int((stats.results == 2).sum()),
            "unfinished": int((stats.results < 0).sum()),
        },
        "wall_seconds": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", default=None,
                        help="ships per team, e.g. 5 or 50 (repeatable)")
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=54_000,
                        help="decision cap per match (54000 = 30 min at 30 Hz)")
    parser.add_argument("--scale-vision", action="store_true",
                        help="scale vision_range with the map instead of holding it fixed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500,
                        help="log progress every N decisions (0 disables)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    per_team = [int(v) for v in (args.scenario or ["5", "50"])]
    device = torch.device(args.device)
    reports = []
    for team in per_team:
        ships = 2 * team
        name = f"{team}v{team}"
        stats, env_config, ship_config, elapsed = run_scenario(
            ships, args.envs, args.max_steps, device,
            scale_vision=args.scale_vision, seed=args.seed,
            progress_every=args.progress_every,
        )
        entry = report(name, stats, env_config, ship_config, elapsed, args.max_steps)
        reports.append(entry)
        print(json.dumps(entry, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"scenarios": reports}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
