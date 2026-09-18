"""Frontline inference throughput at the training scale and at fleet scale.

Two regimes, one code path, so an architecture change can be attributed rather
than guessed at:

``5v5``    the shipped RL/BC environment -- 10 ships, 10 fields, 26 entity tokens.
``50v50``  the zero-shot target -- 100 ships on a geometrically scaled map with a
           proportionally larger field count, 178 entity tokens.

The 50v50 map keeps ``world_size`` at the Frontline contract's 16384 px toroid.
That is not a convenience: the position Fourier basis (and the RoPE frequencies
derived from it) is a function of the world period, so a policy trained on the
16384 toroid can only be *run* on the 16384 toroid. What scales instead is the
battlefield drawn inside it -- ``playable_radius``, the zone ring, the zone
radius, the sight range, and the field count. See ``LARGE_SCALE`` below for the
chosen factor and why it is capped where it is.

The policy is built from the profile's own ``ModelConfig`` with random weights:
this measures architecture cost, not skill, and an untrained policy executes
exactly the same kernels as a trained one.

Reported per regime:

* end-to-end decisions/second and environment steps/second, including
  perception, observation assembly, belief composition and physics;
* policy-only forward latency and ship-decisions/second, timed separately with
  CUDA events so the attention cost is not hidden behind the simulator;
* peak allocated/reserved device memory;
* which SDPA backend the spatial attention actually ran on.

Usage:
    uv run --no-sync python benchmarks/frontline_inference_scaling.py \
        --out artifacts/benchmarks/inference.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from boost_and_broadside.config.core import EnvConfig, entity_token_count
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import perceived_observation_from_state
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.belief import BeliefTracker
from boost_and_broadside.train.rl.policy_io import build_policy, compile_policy

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

# Linear factor applied to every Frontline length for the 50v50 map.
#
# Ten times the ships want sqrt(10) = 3.162 in length to hold areal density
# fixed, which puts ``playable_radius`` at 8221 px. The toroid's half-period is
# 8192 px, and a minimum-image displacement stops being well defined at exactly
# that distance -- so the density-preserving factor is not reachable on the
# contract world at all. 7000 px leaves a ~1200 px margin for ships that stray
# past the soft boundary and lands at 2.69, which is 1.4x the 5v5 areal ship
# density. A denser 50v50 is the conservative direction for an attention
# benchmark: more ships inside each other's sight radius, not fewer.
LARGE_SCALE = 7000.0 / 2600.0


@dataclass(frozen=True)
class Scenario:
    """One benchmark regime: fleet size, map geometry, and batch width."""

    name: str
    num_ships: int
    num_fields: int
    scale: float
    num_envs: int

    @property
    def tokens(self) -> int:
        return entity_token_count(self.num_ships, self.num_fields, _FRONTLINE)


_PROFILE = PROFILES["rl"]
_FRONTLINE = _PROFILE.frontline


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("5v5", num_ships=10, num_fields=10, scale=1.0, num_envs=512),
    # 72 fields is 10 x the 5v5 count scaled by map area (2.69^2 = 7.25).
    # Overlapping fields are legal, so no placement search bounds the count.
    Scenario("50v50", num_ships=100, num_fields=72, scale=LARGE_SCALE, num_envs=64),
)


def scenario_configs(scenario: Scenario):
    """Return the (ship_config, env_config) pair a scenario runs under."""

    ship_config = _PROFILE.ship_config  # world_size stays at the 16384 contract
    frontline = replace(
        _FRONTLINE,
        zone_radius=_FRONTLINE.zone_radius * scenario.scale,
        zone_ring_radius=_FRONTLINE.zone_ring_radius * scenario.scale,
        playable_radius=_FRONTLINE.playable_radius * scenario.scale,
    )
    env_config = EnvConfig(
        num_ships=scenario.num_ships,
        num_fields=scenario.num_fields,
        max_bullets=_PROFILE.max_bullets,
        max_episode_steps=_PROFILE.max_episode_steps,
        action_repeat=_PROFILE.action_repeat,
        spawn_resource_spread=_PROFILE.spawn_resource_spread,
        frontline=frontline,
        # Sight scales with the map so the fraction of the battlefield a ship can
        # see -- and therefore the live token count attention actually mixes over
        # -- is comparable between the two regimes.
        vision_range=_PROFILE.vision_range * scenario.scale,
        zones_occlude=_PROFILE.zones_occlude,
    )
    return ship_config, env_config


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _observed_backend(policy, obs, hidden) -> str:
    """Name the SDPA backend the spatial attention dispatched to."""

    if not torch.cuda.is_available():
        return "cpu"
    from torch.nn.attention import SDPBackend, sdpa_kernel

    for backend, name in (
        (SDPBackend.FLASH_ATTENTION, "flash"),
        (SDPBackend.EFFICIENT_ATTENTION, "mem_efficient"),
        (SDPBackend.CUDNN_ATTENTION, "cudnn"),
    ):
        try:
            with sdpa_kernel(backend), torch.inference_mode():
                policy.get_action_and_value(obs, hidden)
            return name
        except RuntimeError:
            continue
    return "math"


@torch.inference_mode()
def run_scenario(
    scenario: Scenario,
    *,
    steps: int,
    warmup: int,
    compile_mode: str | None,
    device: torch.device,
    seed: int,
) -> dict:
    """Time one regime end to end and the policy forward on its own."""

    ship_config, env_config = scenario_configs(scenario)
    torch.manual_seed(seed)

    env = TensorEnv(scenario.num_envs, ship_config, env_config, device)
    env.reset(seed=seed)

    policy = build_policy(
        _PROFILE.model_config,
        ship_config,
        num_value_components=12,
        num_ships=scenario.num_ships,
        team_pma_k=(0, 1),
    ).to(device)
    policy.eval()
    policy.requires_grad_(False)
    policy = compile_policy(policy, compile_mode)

    belief = BeliefTracker(
        scenario.num_envs,
        scenario.num_ships,
        ship_config.dt * env_config.action_repeat,
        policy.coordinator,
        device,
    )
    hidden = policy.initial_hidden(scenario.num_envs, scenario.num_ships, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def one_step(state_hidden: torch.Tensor) -> torch.Tensor:
        observation, _ = perceived_observation_from_state(env.state, ship_config, env_config)
        view = belief.compose(observation.for_team(0))
        action, _, _, prediction, new_hidden = policy.get_action_and_value(view, state_hidden)
        belief.advance(view, prediction)
        dones, _ = env.step(action.int())
        if bool(dones.any()):
            env.reset_envs(dones)
            belief.reset(dones)
            new_hidden = policy.reset_hidden_for_envs(new_hidden, dones, scenario.num_ships)
        return new_hidden

    for _ in range(warmup):
        hidden = one_step(hidden)
    _sync()

    start = time.perf_counter()
    for _ in range(steps):
        hidden = one_step(hidden)
    _sync()
    elapsed = time.perf_counter() - start

    # Policy forward alone, on a fixed observation so no simulator work is timed.
    observation, _ = perceived_observation_from_state(env.state, ship_config, env_config)
    view = belief.compose(observation.for_team(0))
    for _ in range(5):
        policy.get_action_and_value(view, hidden)
    _sync()
    samples: list[float] = []
    for _ in range(max(steps // 2, 10)):
        begin = time.perf_counter()
        policy.get_action_and_value(view, hidden)
        _sync()
        samples.append(1000.0 * (time.perf_counter() - begin))

    decisions = steps * scenario.num_envs * scenario.num_ships
    result = {
        "scenario": scenario.name,
        "num_ships": scenario.num_ships,
        "num_fields": scenario.num_fields,
        "entity_tokens": scenario.tokens,
        "num_envs": scenario.num_envs,
        "map_scale": round(scenario.scale, 4),
        "playable_radius": round(env_config.frontline.playable_radius, 1),
        "vision_range": round(env_config.vision_range, 1),
        "steps": steps,
        "compile_mode": compile_mode,
        "end_to_end_env_steps_per_s": steps * scenario.num_envs / elapsed,
        "end_to_end_decisions_per_s": decisions / elapsed,
        "end_to_end_ms_per_batched_step": 1000.0 * elapsed / steps,
        "policy_forward_ms_median": statistics.median(samples),
        "policy_forward_ms_p90": sorted(samples)[int(0.9 * (len(samples) - 1))],
        "policy_decisions_per_s": (
            scenario.num_envs * scenario.num_ships / (statistics.median(samples) / 1000.0)
        ),
        "sdpa_backend": _observed_backend(policy, view, hidden),
    }
    if device.type == "cuda":
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--compile-mode", default="default", help="'none' leaves the policy eager")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--scenario", action="append", choices=[s.name for s in SCENARIOS])
    parser.add_argument("--label", default="", help="architecture stage this run measures")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    compile_mode = None if args.compile_mode in ("none", "None") else args.compile_mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wanted = set(args.scenario) if args.scenario else {s.name for s in SCENARIOS}

    rows = []
    for scenario in SCENARIOS:
        if scenario.name not in wanted:
            continue
        row = run_scenario(
            scenario,
            steps=args.steps,
            warmup=args.warmup,
            compile_mode=compile_mode,
            device=device,
            seed=args.seed,
        )
        row["label"] = args.label
        rows.append(row)
        print(
            f"[{args.label or 'run'}] {row['scenario']:>6s}  "
            f"tokens={row['entity_tokens']:>4d}  "
            f"env_sps={row['end_to_end_env_steps_per_s']:>10,.0f}  "
            f"decisions/s={row['end_to_end_decisions_per_s']:>12,.0f}  "
            f"fwd={row['policy_forward_ms_median']:>7.3f} ms  "
            f"peak={row.get('peak_allocated_mib', 0):>7.1f} MiB  "
            f"backend={row['sdpa_backend']}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
