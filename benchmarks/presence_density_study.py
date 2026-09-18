"""Choose the presence kernel's radius and compression from measured behaviour.

The ally/enemy presence scalars have to say something useful at 5v5 and still
say something useful at 50v50 -- both on a map scaled for the larger fleet and on
the unscaled map the zero-shot crossover sweep actually uses, where local
crowding really does grow by an order of magnitude.

Two things are being chosen:

* **radius** -- the Gaussian kernel's width in world pixels;
* **compression** -- ``log1p(s)`` against a bounded ``s / (s + k)``.

The criterion is discrimination, not magnitude: within one fleet size the
feature must separate a lone ship from a swarmed one, and across fleet sizes it
must not collapse. A compression that maps 10 and 30 neighbours to the same
number has thrown away the thing it was added to supply.

Realistic samples come from resetting the actual Frontline environment and
stepping it with the scripted controller, so the spatial distribution is one the
policy will really see. Synthetic uniform layouts bracket it.

Usage:
    uv run --no-sync python benchmarks/presence_density_study.py --out study.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config.core import EnvConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import perceived_observation_from_state
from boost_and_broadside.env.perception import team_visibility_from_state
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.features import PRESENCE_RADIUS, local_presence

PROFILE = PROFILES["rl"]
RADII = (250.0, 500.0, 1000.0)
LARGE_SCALE = 7000.0 / 2600.0


def raw_presence(
    position: torch.Tensor,
    team_id: torch.Tensor,
    source: torch.Tensor,
    world_size: tuple[float, float],
    radius: float,
) -> torch.Tensor:
    """Uncompressed Gaussian aggregate, so each compression can be applied after."""

    width, height = world_size
    dx = position[..., :, None, 0] - position[..., None, :, 0]
    dy = position[..., :, None, 1] - position[..., None, :, 1]
    dx = (dx + width / 2.0) % width - width / 2.0
    dy = (dy + height / 2.0) % height - height / 2.0
    weight = torch.exp(-(dx * dx + dy * dy) / (2.0 * radius * radius))
    contributes = source.unsqueeze(-2)
    same = team_id.unsqueeze(-1) == team_id.unsqueeze(-2)
    identity = torch.eye(weight.shape[-1], dtype=torch.bool, device=weight.device)
    ally = (weight * (contributes & same & ~identity)).sum(-1)
    enemy = (weight * (contributes & ~same)).sum(-1)
    return torch.stack((ally, enemy), dim=-1)


def scenario_env(num_ships: int, num_fields: int, scale: float, num_envs: int, device: str):
    frontline = replace(
        PROFILE.frontline,
        zone_radius=PROFILE.frontline.zone_radius * scale,
        zone_ring_radius=PROFILE.frontline.zone_ring_radius * scale,
        playable_radius=PROFILE.frontline.playable_radius * scale,
    )
    env_config = EnvConfig(
        num_ships=num_ships,
        num_fields=num_fields,
        max_bullets=PROFILE.max_bullets,
        max_episode_steps=PROFILE.max_episode_steps,
        frontline=frontline,
        vision_range=PROFILE.vision_range * scale,
    )
    env = TensorEnv(num_envs, PROFILE.ship_config, env_config, device)
    env.reset(seed=17)
    return env, env_config


@torch.inference_mode()
def sample_scene(
    num_ships: int, num_fields: int, scale: float, num_envs: int, steps: int, device: str
):
    """Step a real Frontline match and return the observed ship geometry."""

    env, env_config = scenario_env(num_ships, num_fields, scale, num_envs, device)
    agent = StochasticScriptedAgent(PROFILE.ship_config, StochasticAgentConfig())
    for _ in range(steps):
        # Bullet perception off, matching the shipped profile
        # (``n_bullet_cross_per_block=0``): computing ship-to-bullet line of
        # sight for a 100-ship fleet is a 1000-projectile cross product this
        # study has no use for, and it alone exhausts an 8 GB card.
        sight = team_visibility_from_state(
            env.state, PROFILE.ship_config, env_config, perceive_bullets=False
        )
        env.step(agent.get_actions(env.state, sight.ship))
    observation, _ = perceived_observation_from_state(env.state, PROFILE.ship_config, env_config)
    view = observation.for_team(0)
    team_id = view["team_id"]
    source = view["belief_valid"].bool() & (team_id < 2)
    return (
        view["pos"].float().cpu(),
        team_id.cpu(),
        source.cpu(),
        (team_id < 2).cpu(),
    )


def summarize(values: torch.Tensor, ship_rows: torch.Tensor) -> dict:
    """Distribution of one presence channel over live ship tokens."""

    live = values[ship_rows]
    quantiles = torch.quantile(live, torch.tensor([0.1, 0.5, 0.9, 0.99]))
    return {
        "mean": live.mean().item(),
        "p10": quantiles[0].item(),
        "p50": quantiles[1].item(),
        "p90": quantiles[2].item(),
        "p99": quantiles[3].item(),
        # Spread across the bulk of the distribution: how much of the feature's
        # range actually separates a sparse neighbourhood from a crowded one.
        "p10_p90_spread": (quantiles[2] - quantiles[0]).item(),
        "max": live.max().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--envs", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    world = tuple(float(s) for s in PROFILE.ship_config.world_size)
    scenarios = [
        ("5v5", 10, 10, 1.0),
        ("50v50_scaled_map", 100, 72, LARGE_SCALE),
        # The crossover sweep's shape: a big fleet on the *training* map, where
        # local crowding really does grow with the fleet.
        ("50v50_same_map", 100, 10, 1.0),
        ("32v32_same_map", 64, 10, 1.0),
    ]

    rows = []
    for name, ships, fields, scale in scenarios:
        position, team_id, source, ship_rows = sample_scene(
            ships, fields, scale, args.envs, args.steps, args.device
        )
        for radius in RADII:
            raw = raw_presence(position, team_id, source, world, radius)
            for channel, index in (("ally", 0), ("enemy", 1)):
                entry = {
                    "scenario": name,
                    "num_ships": ships,
                    "map_scale": round(scale, 3),
                    "radius": radius,
                    "channel": channel,
                    "raw": summarize(raw[..., index], ship_rows),
                    "log1p": summarize(torch.log1p(raw[..., index]), ship_rows),
                    "saturating_k2": summarize(
                        raw[..., index] / (raw[..., index] + 2.0), ship_rows
                    ),
                }
                rows.append(entry)

    header = f"{'scenario':<18s}{'ch':<6s}{'r':>6s}{'raw p50':>9s}{'raw p99':>9s}"
    header += f"{'log1p p50':>11s}{'log1p spread':>14s}{'sat p50':>9s}{'sat spread':>12s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['scenario']:<18s}{row['channel']:<6s}{row['radius']:>6.0f}"
            f"{row['raw']['p50']:>9.2f}{row['raw']['p99']:>9.2f}"
            f"{row['log1p']['p50']:>11.2f}{row['log1p']['p10_p90_spread']:>14.2f}"
            f"{row['saturating_k2']['p50']:>9.2f}"
            f"{row['saturating_k2']['p10_p90_spread']:>12.2f}"
        )

    print("\nacross-scenario separation (log1p vs saturating), enemy channel:")
    for radius in RADII:
        picks = {
            row["scenario"]: row
            for row in rows
            if row["radius"] == radius and row["channel"] == "enemy"
        }
        small = picks["5v5"]
        big = picks["50v50_same_map"]
        log_ratio = big["log1p"]["p50"] / max(small["log1p"]["p50"], 1e-9)
        sat_ratio = big["saturating_k2"]["p50"] / max(small["saturating_k2"]["p50"], 1e-9)
        print(
            f"  r={radius:>6.0f}  log1p p50 {small['log1p']['p50']:.2f} -> "
            f"{big['log1p']['p50']:.2f} (x{log_ratio:.2f})   "
            f"saturating {small['saturating_k2']['p50']:.2f} -> "
            f"{big['saturating_k2']['p50']:.2f} (x{sat_ratio:.2f})"
        )

    # Sanity: the shipped feature must equal log1p of the raw aggregate.
    position, team_id, source, ship_rows = sample_scene(10, 10, 1.0, 4, 50, args.device)
    shipped = local_presence(position, team_id, source, world)
    reference = torch.log1p(raw_presence(position, team_id, source, world, PRESENCE_RADIUS))
    reference = reference * (team_id < 2).unsqueeze(-1)
    assert torch.allclose(shipped, reference, atol=1e-6), "shipped feature drifted from the study"
    print("\nshipped local_presence matches log1p(raw, r=500) exactly")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
