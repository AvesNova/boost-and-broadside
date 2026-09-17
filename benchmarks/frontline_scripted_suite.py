"""Batched Frontline scripted-v-scripted statistical playtest.

Example:
    .venv/bin/python benchmarks/frontline_scripted_suite.py \
        --device cuda --games 256 --output /tmp/frontline-scripted-suite.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import MatchResult
from boost_and_broadside.config.defaults import SHIP_CONFIG
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import (
    frontline_ship_config,
)
from boost_and_broadside.env.perception import team_visibility_from_state
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


@torch.inference_mode()
def run_suite(
    *,
    games: int,
    seed: int,
    device: torch.device,
    max_ticks: int,
    capture_seconds: float | None = None,
    team_size: int = 5,
) -> dict:
    """Run independent matches in one tensor batch and retain per-game samples."""

    if team_size < 1:
        raise ValueError("team_size must be positive")
    if games < 1:
        raise ValueError("games must be positive")
    if max_ticks < 1:
        raise ValueError("max_ticks must be positive")
    if capture_seconds is not None and capture_seconds <= 0.0:
        raise ValueError("capture_seconds must be positive")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    torch.manual_seed(seed)
    ship_config = frontline_ship_config(SHIP_CONFIG)
    frontline = PLAY_ENV_CONFIG.frontline
    if capture_seconds is not None:
        frontline = replace(frontline, capture_seconds=capture_seconds)
    env_config = replace(
        PLAY_ENV_CONFIG,
        max_episode_steps=max_ticks,
        frontline=frontline,
        num_ships=2 * team_size,
    )
    env = TensorEnv(games, ship_config, env_config, device)
    env.reset(options={"team_sizes": (team_size, team_size)}, seed=seed)
    agent = StochasticScriptedAgent(ship_config, StochasticAgentConfig())

    running = torch.ones(games, dtype=torch.bool, device=device)
    result = torch.full((games,), int(MatchResult.ONGOING), dtype=torch.int8, device=device)
    duration = torch.zeros(games, dtype=torch.int32, device=device)
    first_capture = torch.full((games,), -1, dtype=torch.int32, device=device)
    second_capture = torch.full((games,), -1, dtype=torch.int32, device=device)
    team0_captures = torch.zeros_like(duration)
    team1_captures = torch.zeros_like(duration)
    simultaneous = torch.zeros_like(duration)
    respawns = torch.zeros_like(duration)
    combat_deaths = torch.zeros_like(duration)
    boundary_deaths = torch.zeros_like(duration)
    healing = torch.zeros(games, dtype=torch.float32, device=device)
    front_min = torch.zeros(games, dtype=torch.long, device=device)
    front_max = torch.zeros(games, dtype=torch.long, device=device)
    action = torch.zeros((games, 2 * team_size, 3), dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    ticks_run = 0
    for tick in range(max_ticks):
        if tick % env_config.action_repeat == 0:
            visibility = team_visibility_from_state(env.state, ship_config, env_config, False)
            action = agent.get_actions(env.state, visibility.ship)
        dones, truncated = env.tick(action)
        ticks_run = tick + 1

        active = running
        capture_count = (
            env.state.team0_captured.to(torch.int32) + env.state.team1_captured.to(torch.int32)
        ) * active
        captures_before = team0_captures + team1_captures
        captured = (capture_count > 0) & active
        first_capture = torch.where(
            captured & (first_capture < 0),
            torch.full_like(first_capture, tick + 1),
            first_capture,
        )
        reached_second = ((captures_before >= 1) & (capture_count >= 1)) | (
            (captures_before == 0) & (capture_count >= 2)
        )
        second_capture = torch.where(
            reached_second & (second_capture < 0),
            torch.full_like(second_capture, tick + 1),
            second_capture,
        )
        team0_captures += env.state.team0_captured.to(torch.int32) * active
        team1_captures += env.state.team1_captured.to(torch.int32) * active
        simultaneous += env.state.simultaneous_capture.to(torch.int32) * active
        respawns += env.state.ship_respawned.sum(dim=1).to(torch.int32) * active
        combat_deaths += env.state.ship_combat_death.sum(dim=1).to(torch.int32) * active
        boundary_deaths += env.state.ship_boundary_death.sum(dim=1).to(torch.int32) * active
        healing += env.state.ship_shield_recharge.sum(dim=1) * active
        front_min = torch.where(
            active, torch.minimum(front_min, env.state.front_position), front_min
        )
        front_max = torch.where(
            active, torch.maximum(front_max, env.state.front_position), front_max
        )

        newly_finished = (dones | truncated) & active
        result = torch.where(newly_finished, env.state.match_result, result)
        duration = torch.where(newly_finished, tick + 1, duration)
        running &= ~newly_finished
        if (tick + 1) % 60 == 0 and not running.any().item():
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if running.any().item():
        raise RuntimeError(f"{int(running.sum().item())} games did not finish by {max_ticks} ticks")

    result_cpu = result.cpu()
    duration_seconds = duration.cpu().float() * ship_config.dt
    first_capture_seconds = first_capture.cpu().float() * ship_config.dt
    second_capture_seconds = second_capture.cpu().float() * ship_config.dt
    first_capture_mask = first_capture.cpu() >= 0
    second_capture_mask = second_capture.cpu() >= 0
    per_game = {
        "result": result_cpu.tolist(),
        "duration_seconds": duration_seconds.tolist(),
        "first_capture_seconds": first_capture_seconds.tolist(),
        "second_capture_seconds": second_capture_seconds.tolist(),
        "team0_captures": team0_captures.cpu().tolist(),
        "team1_captures": team1_captures.cpu().tolist(),
        "simultaneous_captures": simultaneous.cpu().tolist(),
        "respawns": respawns.cpu().tolist(),
        "combat_deaths": combat_deaths.cpu().tolist(),
        "boundary_deaths": boundary_deaths.cpu().tolist(),
        "shield_recharge": healing.cpu().tolist(),
        "front_min": front_min.cpu().tolist(),
        "front_max": front_max.cpu().tolist(),
    }
    final_front = (team0_captures - team1_captures).cpu()
    per_game["final_front"] = final_front.tolist()
    summaries = {
        "duration_seconds": _summary(duration_seconds),
        "total_captures": _summary((team0_captures + team1_captures).cpu()),
        "respawns": _summary(respawns.cpu()),
        "combat_deaths": _summary(combat_deaths.cpu()),
        "boundary_deaths": _summary(boundary_deaths.cpu()),
        "shield_recharge": _summary(healing.cpu()),
    }
    summaries["first_capture_seconds"] = (
        _summary(first_capture_seconds[first_capture_mask]) if first_capture_mask.any() else None
    )
    summaries["second_capture_seconds"] = (
        _summary(second_capture_seconds[second_capture_mask]) if second_capture_mask.any() else None
    )

    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "games": games,
        "seed": seed,
        "ticks_run": ticks_run,
        "wall_seconds": elapsed,
        "simulated_game_seconds_per_wall_second": (duration_seconds.sum().item() / elapsed),
        "peak_torch_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "outcomes": {
            "team0_wins": int((result_cpu == int(MatchResult.TEAM0_WIN)).sum().item()),
            "team1_wins": int((result_cpu == int(MatchResult.TEAM1_WIN)).sum().item()),
            "draws": int((result_cpu == int(MatchResult.DRAW)).sum().item()),
        },
        "capture_reached_fraction": first_capture_mask.float().mean().item(),
        "second_capture_reached_fraction": second_capture_mask.float().mean().item(),
        "termination_causes": {
            "front_threshold": int(
                (final_front.abs() >= env_config.frontline.front_win_threshold).sum().item()
            ),
            "timeout": int(
                (final_front.abs() < env_config.frontline.front_win_threshold).sum().item()
            ),
        },
        "summaries": summaries,
        "ship_config": asdict(ship_config),
        "env_config": asdict(env_config),
        "scripted_config": asdict(agent.config),
        "per_game": per_game,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--team-size", type=int, default=5)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--max-ticks", type=int, default=PLAY_ENV_CONFIG.max_episode_steps)
    parser.add_argument("--capture-seconds", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_suite(
        games=args.games,
        team_size=args.team_size,
        seed=args.seed,
        device=torch.device(args.device),
        max_ticks=args.max_ticks,
        capture_seconds=args.capture_seconds,
    )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    printable = {key: value for key, value in result.items() if key != "per_game"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
