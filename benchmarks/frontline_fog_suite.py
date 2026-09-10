"""GPU-batched Frontline fog-of-war distribution diagnostic.

The scripted fleets act with the production vision range. Additional ranges are
counterfactual perception probes over that same trajectory, avoiding several
expensive duplicate physics runs while still measuring the geometry contract.

Example:
    .venv/bin/python benchmarks/frontline_fog_suite.py --device cuda --games 256
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import frontline_ship_config
from boost_and_broadside.env.observation import (
    ObservationBuffers,
    perceived_observation_from_state,
)
from boost_and_broadside.env.perception import TeamVisibility, team_visibility_from_state
from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": values.mean().item(),
        "std": values.std(correction=0).item(),
        "p10": torch.quantile(values, 0.10).item(),
        "median": torch.quantile(values, 0.50).item(),
        "p90": torch.quantile(values, 0.90).item(),
    }


def _timed_cuda(device: torch.device, fn, iterations: int) -> float:
    for _ in range(10):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - started


def _profile_perception(
    env: TensorEnv,
    ship_config: ShipConfig,
    env_config,
    iterations: int,
) -> dict[str, float]:
    buffers = ObservationBuffers.allocate(
        env.num_envs,
        env_config.num_ships,
        env_config.num_fields,
        5 if env_config.frontline is not None else 0,
        ship_config,
        env.device,
    )
    visibility_seconds = _timed_cuda(
        env.device,
        lambda: team_visibility_from_state(env.state, ship_config, env_config),
        iterations,
    )
    observation_seconds = _timed_cuda(
        env.device,
        lambda: perceived_observation_from_state(
            env.state,
            ship_config,
            env_config,
            buffers,
            include_bullets=False,
        ),
        iterations,
    )
    return {
        "iterations": iterations,
        "batch_size": env.num_envs,
        "visibility_ms_per_batch": visibility_seconds * 1000.0 / iterations,
        "team_pair_observation_ms_per_batch": observation_seconds * 1000.0 / iterations,
        "visibility_envs_per_second": env.num_envs * iterations / visibility_seconds,
        "team_pair_observation_envs_per_second": (
            env.num_envs * iterations / observation_seconds
        ),
    }


@dataclass
class FogAccumulator:
    """On-device counters for one counterfactual vision-range probe."""

    visible: torch.Tensor
    range_visible: torch.Tensor
    los_visible: torch.Tensor
    enemy_slots: torch.Tensor
    observer_visible: torch.Tensor
    observer_pairs: torch.Tensor
    hidden_age_sum: torch.Tensor
    hidden_samples: torch.Tensor
    ever_seen: torch.Tensor
    enemy_mask: torch.Tensor
    previous_visible: torch.Tensor
    hidden_age: torch.Tensor
    reacquisitions: torch.Tensor
    hidden_runs: torch.Tensor
    duration_histogram: torch.Tensor
    bins_steps: torch.Tensor

    @classmethod
    def create(
        cls,
        games: int,
        num_ships: int,
        device: torch.device,
        bins_steps: torch.Tensor,
    ) -> FogAccumulator:
        zeros_per_game = torch.zeros(games, dtype=torch.float64, device=device)
        shape = (games, 2, num_ships)
        return cls(
            visible=zeros_per_game.clone(),
            range_visible=zeros_per_game.clone(),
            los_visible=zeros_per_game.clone(),
            enemy_slots=zeros_per_game.clone(),
            observer_visible=zeros_per_game.clone(),
            observer_pairs=zeros_per_game.clone(),
            hidden_age_sum=zeros_per_game.clone(),
            hidden_samples=zeros_per_game.clone(),
            ever_seen=torch.zeros(shape, dtype=torch.bool, device=device),
            enemy_mask=torch.zeros(shape, dtype=torch.bool, device=device),
            previous_visible=torch.zeros(shape, dtype=torch.bool, device=device),
            hidden_age=torch.zeros(shape, dtype=torch.int32, device=device),
            reacquisitions=zeros_per_game.clone(),
            hidden_runs=zeros_per_game.clone(),
            duration_histogram=torch.zeros(
                bins_steps.numel() + 1, dtype=torch.int64, device=device
            ),
            bins_steps=bins_steps,
        )

    def update(self, sight: TeamVisibility, state) -> None:
        perspective = torch.arange(2, device=state.device).view(1, 2, 1)
        enemy = state.ship_team_id[:, None, :] != perspective
        enemy_alive = enemy & state.ship_alive[:, None, :]
        visible = sight.ship & enemy_alive
        range_visible = sight.range_only_ship & enemy_alive
        los_visible = sight.los_ship & enemy_alive
        self.enemy_mask |= enemy
        ever_before = self.ever_seen
        reacquired = visible & ~self.previous_visible & ever_before
        hidden_started = ~visible & self.previous_visible & enemy_alive

        for index, upper in enumerate(self.bins_steps):
            lower = 0 if index == 0 else self.bins_steps[index - 1]
            self.duration_histogram[index] += (
                reacquired & (self.hidden_age > lower) & (self.hidden_age <= upper)
            ).sum()
        self.duration_histogram[-1] += (
            reacquired & (self.hidden_age > self.bins_steps[-1])
        ).sum()

        ever_after = ever_before | visible
        hidden = enemy_alive & ever_after & ~visible
        self.hidden_age = torch.where(hidden, self.hidden_age + 1, 0)
        self.ever_seen = ever_after
        self.previous_visible = visible

        observer_team = state.ship_team_id[:, :, None]
        observer_enemy = (
            (observer_team != state.ship_team_id[:, None, :])
            & state.ship_alive[:, :, None]
            & state.ship_alive[:, None, :]
        )
        observer_visible = sight.observer_ship & observer_enemy

        self.visible += visible.sum(dim=(1, 2))
        self.range_visible += range_visible.sum(dim=(1, 2))
        self.los_visible += los_visible.sum(dim=(1, 2))
        self.enemy_slots += enemy_alive.sum(dim=(1, 2))
        self.observer_visible += observer_visible.sum(dim=(1, 2))
        self.observer_pairs += observer_enemy.sum(dim=(1, 2))
        self.hidden_age_sum += (self.hidden_age * hidden).sum(dim=(1, 2))
        self.hidden_samples += hidden.sum(dim=(1, 2))
        self.reacquisitions += reacquired.sum(dim=(1, 2))
        self.hidden_runs += hidden_started.sum(dim=(1, 2))

    def result(self, decision_seconds: float) -> dict[str, object]:
        visible_fraction = self.visible / self.enemy_slots.clamp(min=1)
        range_fraction = self.range_visible / self.enemy_slots.clamp(min=1)
        individual_fraction = self.observer_visible / self.observer_pairs.clamp(min=1)
        field_occluded = (
            self.range_visible - self.los_visible
        ) / self.range_visible.clamp(min=1)
        never_seen = (self.enemy_mask & ~self.ever_seen).sum(dim=(1, 2)).double()
        never_seen /= self.enemy_mask.sum(dim=(1, 2)).clamp(min=1)
        mean_hidden_seconds = (
            self.hidden_age_sum / self.hidden_samples.clamp(min=1) * decision_seconds
        )
        completed = self.reacquisitions.sum()
        hidden_runs = self.hidden_runs.sum()
        bins_seconds = self.bins_steps.double() * decision_seconds
        return {
            "visible_fraction": _summary(visible_fraction.cpu()),
            "range_only_visible_fraction": _summary(range_fraction.cpu()),
            "field_occluded_fraction_of_in_range": _summary(field_occluded.cpu()),
            "individual_visible_fraction": _summary(individual_fraction.cpu()),
            "team_shared_gain": _summary((visible_fraction - individual_fraction).cpu()),
            "never_seen_fraction_at_horizon": _summary(never_seen.cpu()),
            "mean_hidden_age_seconds": _summary(mean_hidden_seconds.cpu()),
            "reacquisitions_per_game": _summary(self.reacquisitions.cpu()),
            "hidden_runs_per_game": _summary(self.hidden_runs.cpu()),
            "completed_hidden_run_fraction": (completed / hidden_runs.clamp(min=1)).item(),
            "censored_hidden_slots_at_horizon": int(
                ((self.hidden_age > 0) & self.ever_seen).sum().item()
            ),
            "occlusion_duration_histogram": {
                "upper_bounds_seconds": bins_seconds.cpu().tolist(),
                "counts": self.duration_histogram.cpu().tolist(),
            },
        }


@torch.inference_mode()
def run_suite(
    *,
    games: int,
    seed: int,
    device: torch.device,
    seconds: float,
    probe_ranges: tuple[float, ...],
    profile_iterations: int = 0,
) -> dict[str, object]:
    if games < 1:
        raise ValueError("games must be positive")
    if seconds <= 0.0:
        raise ValueError("seconds must be positive")
    if not probe_ranges or any(value <= 0.0 for value in probe_ranges):
        raise ValueError("probe ranges must be positive")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    torch.manual_seed(seed)
    ship_config = frontline_ship_config(ShipConfig())
    ticks = round(seconds / ship_config.dt)
    # Keep all maps alive for the same diagnostic horizon. Captures and strategy
    # still operate normally; only early threshold termination is disabled.
    frontline = replace(PLAY_ENV_CONFIG.frontline, front_win_threshold=1_000_000)
    env_config = replace(
        PLAY_ENV_CONFIG,
        max_episode_steps=ticks,
        frontline=frontline,
    )
    env = TensorEnv(games, ship_config, env_config, device)
    env.reset(options={"team_sizes": (4, 4)}, seed=seed)
    throughput = (
        _profile_perception(env, ship_config, env_config, profile_iterations)
        if profile_iterations > 0
        else None
    )
    agent = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
    decision_seconds = ship_config.dt * env_config.action_repeat
    bins_steps = torch.tensor(
        [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0], device=device
    ).div(decision_seconds).round().to(torch.int32)
    probes = {
        vision_range: FogAccumulator.create(
            games, env_config.num_ships, device, bins_steps
        )
        for vision_range in probe_ranges
    }
    probe_configs = {
        vision_range: replace(env_config, vision_range=vision_range)
        for vision_range in probe_ranges
    }
    action = torch.zeros((games, env_config.num_ships, 3), dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    for tick in range(ticks):
        sights = {
            vision_range: team_visibility_from_state(
                env.state, ship_config, probe_configs[vision_range]
            )
            for vision_range in probe_ranges
        }
        for vision_range, sight in sights.items():
            probes[vision_range].update(sight, env.state)
        if tick % env_config.action_repeat == 0:
            action = agent.get_actions(env.state, sights[env_config.vision_range].ship)
        env.tick(action)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "games": games,
        "seed": seed,
        "horizon_seconds": seconds,
        "ticks": ticks,
        "production_vision_range": env_config.vision_range,
        "probe_note": (
            "All ranges are counterfactual probes over trajectories controlled with "
            "the production vision range."
        ),
        "wall_seconds": elapsed,
        "simulated_game_seconds_per_wall_second": games * seconds / elapsed,
        "peak_torch_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None
        ),
        "perception_throughput": throughput,
        "ranges": {
            str(int(vision_range)): accumulator.result(decision_seconds)
            for vision_range, accumulator in probes.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--ranges", type=float, nargs="+", default=(1200.0, 1600.0, 2000.0))
    parser.add_argument("--profile-iterations", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_suite(
        games=args.games,
        seed=args.seed,
        device=torch.device(args.device),
        seconds=args.seconds,
        probe_ranges=tuple(dict.fromkeys((*args.ranges, PLAY_ENV_CONFIG.vision_range))),
        profile_iterations=args.profile_iterations,
    )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
