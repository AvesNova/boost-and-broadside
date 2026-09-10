"""Natural-occlusion point-estimate baseline and belief-cache throughput.

The accuracy probe uses a deliberately simple no-learning predictor: retain the
last observed physical state and advance position at constant velocity.  It is a
useful floor for deciding whether hidden intervals are benign enough for one
point estimate, while production policies use their learned next-state head.

Example:
    .venv/bin/python benchmarks/frontline_belief_suite.py --device cuda --games 128
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import frontline_ship_config
from boost_and_broadside.env.observation import perceived_observation_from_state
from boost_and_broadside.env.perception import team_visibility_from_state
from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG
from boost_and_broadside.train.rl.belief import DualBeliefTracker
from boost_and_broadside.train.rl.features import build_standard_coordinator
from boost_and_broadside.train.rl.policy_io import load_policy_bundle

_AGE_EDGES = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
_VARIABLES = ("position_px", "velocity_px_s", "health", "power", "cooldown_s")


def _empty_stats(device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    names = ["visible", "hidden"]
    lower = 0.0
    for upper in _AGE_EDGES:
        names.append(f"hidden_age_{lower:g}_{upper:g}s")
        lower = upper
    names.append("hidden_age_30_inf_s")
    return {
        name: {
            "count": torch.zeros((), dtype=torch.float64, device=device),
            **{
                variable: torch.zeros((), dtype=torch.float64, device=device)
                for variable in _VARIABLES
            },
        }
        for name in names
    }


def _add(
    stats: dict[str, dict[str, torch.Tensor]],
    name: str,
    mask: torch.Tensor,
    errors: dict[str, torch.Tensor],
) -> None:
    stats[name]["count"] += mask.sum()
    for variable, values in errors.items():
        stats[name][variable] += (values * mask).sum()


def _finish_stats(
    stats: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for name, values in stats.items():
        count = values["count"].item()
        results[name] = {
            "samples": int(count),
            **{
                variable: (values[variable].item() / count if count else 0.0)
                for variable in _VARIABLES
            },
        }
    return results


@torch.inference_mode()
def _run_learned_accuracy(
    *,
    checkpoint: Path,
    games: int,
    decisions: int,
    seed: int,
    device: torch.device,
    ship_config,
    env_config,
) -> dict[str, object]:
    """Evaluate one checkpoint's recursively fed team-0 beliefs."""

    torch.manual_seed(seed)
    env = TensorEnv(games, ship_config, env_config, device)
    env.reset(options={"team_sizes": (4, 4)}, seed=seed)
    scripted = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
    bundle = load_policy_bundle(
        str(checkpoint),
        device=device,
        num_ships=env_config.num_ships,
        ship_config=ship_config,
        allow_config_drift=True,
    )
    policy = bundle.policy
    hidden = policy.initial_hidden(games, env_config.num_ships, device)
    tracker = DualBeliefTracker(
        games,
        env_config.num_ships,
        ship_config.dt * env_config.action_repeat,
        policy.coordinator,
        device,
    ).team0
    stats = _empty_stats(device)
    baseline_stats = _empty_stats(device)
    world_w, world_h = ship_config.world_size
    baseline_valid = torch.zeros((games, env_config.num_ships), dtype=torch.bool, device=device)
    baseline_age = torch.zeros((games, env_config.num_ships), dtype=torch.int32, device=device)
    baseline_pos = torch.zeros((games, env_config.num_ships), dtype=torch.complex64, device=device)
    baseline_vel = torch.zeros_like(baseline_pos)
    baseline_health = torch.zeros_like(baseline_pos.real)
    baseline_power = torch.zeros_like(baseline_pos.real)
    baseline_cooldown = torch.zeros_like(baseline_pos.real)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(decisions):
        state = env.state
        perceived, sight = perceived_observation_from_state(
            state,
            ship_config,
            env_config,
            include_bullets=bundle.reads_bullets,
        )
        view = tracker.compose(perceived.for_team(0))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            policy_action, _, _, prediction, hidden = policy.get_action_and_value(view, hidden)
        tracker.advance(view, prediction)

        visible = view["visible"][:, : env_config.num_ships] & (
            view["team_id"][:, : env_config.num_ships] == 1
        )
        hidden_enemy = (
            ~view["visible"][:, : env_config.num_ships]
            & view["belief_valid"][:, : env_config.num_ships]
            & (view["team_id"][:, : env_config.num_ships] == 1)
        )
        age_seconds = view["time_since_observation"][:, : env_config.num_ships, 0]

        baseline_hidden = baseline_valid & ~visible
        baseline_pos = torch.where(visible, state.ship_pos, baseline_pos)
        baseline_vel = torch.where(visible, state.ship_vel, baseline_vel)
        baseline_health = torch.where(visible, state.ship_health, baseline_health)
        baseline_power = torch.where(visible, state.ship_power, baseline_power)
        baseline_cooldown = torch.where(visible, state.ship_cooldown, baseline_cooldown)
        baseline_age = torch.where(
            visible,
            torch.zeros_like(baseline_age),
            torch.where(baseline_hidden, baseline_age + 1, torch.zeros_like(baseline_age)),
        )
        baseline_valid |= visible
        baseline_forecast_pos = baseline_pos + baseline_vel * (
            ship_config.dt * env_config.action_repeat
        )
        baseline_forecast_pos = torch.complex(
            baseline_forecast_pos.real.remainder(world_w),
            baseline_forecast_pos.imag.remainder(world_h),
        )

        scripted_action = scripted.get_actions(state, sight.ship)
        action = torch.where(
            (state.ship_team_id == 0).unsqueeze(-1), policy_action, scripted_action
        )
        for _ in range(env_config.action_repeat):
            env.tick(action)

        forecast = policy.coordinator.decode_targets(tracker.predicted_targets)
        forecast_pos = torch.complex(
            forecast["position_x"].squeeze(-1), forecast["position_y"].squeeze(-1)
        )
        next_state = env.state
        transition_contiguous = ~next_state.ship_respawned
        dx = forecast_pos.real - next_state.ship_pos.real + world_w / 2.0
        dy = forecast_pos.imag - next_state.ship_pos.imag + world_h / 2.0
        dx = dx.remainder(world_w) - world_w / 2.0
        dy = dy.remainder(world_h) - world_h / 2.0
        forecast_vel = torch.complex(forecast["velocity"][..., 0], forecast["velocity"][..., 1])
        errors = {
            "position_px": torch.sqrt(dx.square() + dy.square()),
            "velocity_px_s": (forecast_vel - next_state.ship_vel).abs(),
            "health": (forecast["health"].squeeze(-1) - next_state.ship_health).abs(),
            "power": (forecast["power"].squeeze(-1) - next_state.ship_power).abs(),
            "cooldown_s": (forecast["cooldown"].squeeze(-1) - next_state.ship_cooldown).abs(),
        }
        visible &= transition_contiguous
        hidden_enemy &= transition_contiguous
        _add(stats, "visible", visible, errors)
        _add(stats, "hidden", hidden_enemy, errors)
        lower = 0.0
        for upper in _AGE_EDGES:
            _add(
                stats,
                f"hidden_age_{lower:g}_{upper:g}s",
                hidden_enemy & (age_seconds > lower) & (age_seconds <= upper),
                errors,
            )
            lower = upper
        _add(
            stats,
            "hidden_age_30_inf_s",
            hidden_enemy & (age_seconds > 30.0),
            errors,
        )

        baseline_dx = baseline_forecast_pos.real - next_state.ship_pos.real + world_w / 2.0
        baseline_dy = baseline_forecast_pos.imag - next_state.ship_pos.imag + world_h / 2.0
        baseline_dx = baseline_dx.remainder(world_w) - world_w / 2.0
        baseline_dy = baseline_dy.remainder(world_h) - world_h / 2.0
        baseline_errors = {
            "position_px": torch.sqrt(baseline_dx.square() + baseline_dy.square()),
            "velocity_px_s": (baseline_vel - next_state.ship_vel).abs(),
            "health": (baseline_health - next_state.ship_health).abs(),
            "power": (baseline_power - next_state.ship_power).abs(),
            "cooldown_s": (baseline_cooldown - next_state.ship_cooldown).abs(),
        }
        baseline_visible = visible
        baseline_hidden &= transition_contiguous
        _add(baseline_stats, "visible", baseline_visible, baseline_errors)
        _add(baseline_stats, "hidden", baseline_hidden, baseline_errors)
        baseline_age_seconds = baseline_age.float() * (ship_config.dt * env_config.action_repeat)
        lower = 0.0
        for upper in _AGE_EDGES:
            _add(
                baseline_stats,
                f"hidden_age_{lower:g}_{upper:g}s",
                baseline_hidden & (baseline_age_seconds > lower) & (baseline_age_seconds <= upper),
                baseline_errors,
            )
            lower = upper
        _add(
            baseline_stats,
            "hidden_age_30_inf_s",
            baseline_hidden & (baseline_age_seconds > 30.0),
            baseline_errors,
        )
        baseline_pos = torch.where(baseline_valid, baseline_forecast_pos, baseline_pos)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "checkpoint": str(checkpoint),
        "wall_seconds": time.perf_counter() - started,
        "prediction_error_mean": _finish_stats(stats),
        "same_trajectory_baseline_error_mean": _finish_stats(baseline_stats),
    }


@torch.inference_mode()
def run_suite(
    *,
    games: int,
    seconds: float,
    seed: int,
    device: torch.device,
    profile_iterations: int,
    checkpoint: Path | None = None,
) -> dict[str, object]:
    torch.manual_seed(seed)
    ship_config = frontline_ship_config(ShipConfig())
    frontline = replace(PLAY_ENV_CONFIG.frontline, front_win_threshold=1_000_000)
    decisions = round(seconds / (ship_config.dt * PLAY_ENV_CONFIG.action_repeat))
    env_config = replace(
        PLAY_ENV_CONFIG,
        max_episode_steps=decisions * PLAY_ENV_CONFIG.action_repeat + 1,
        frontline=frontline,
    )
    env = TensorEnv(games, ship_config, env_config, device)
    env.reset(options={"team_sizes": (4, 4)}, seed=seed)
    agent = StochasticScriptedAgent(ship_config, StochasticAgentConfig())

    b, n = games, env_config.num_ships
    shape = (b, 2, n)
    valid = torch.zeros(shape, dtype=torch.bool, device=device)
    age = torch.zeros(shape, dtype=torch.int32, device=device)
    cached_pos = torch.zeros(shape, dtype=torch.complex64, device=device)
    cached_vel = torch.zeros(shape, dtype=torch.complex64, device=device)
    cached_health = torch.zeros(shape, device=device)
    cached_power = torch.zeros(shape, device=device)
    cached_cooldown = torch.zeros(shape, device=device)
    stats = _empty_stats(device)
    decision_dt = ship_config.dt * env_config.action_repeat

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(decisions):
        state = env.state
        sight = team_visibility_from_state(state, ship_config, env_config)
        perspective = torch.arange(2, device=device).view(1, 2, 1)
        enemy = state.ship_team_id[:, None, :] != perspective
        visible = sight.ship & enemy & state.ship_alive[:, None, :]
        hidden = enemy & valid & ~visible

        cached_pos = torch.where(visible, state.ship_pos[:, None, :], cached_pos)
        cached_vel = torch.where(visible, state.ship_vel[:, None, :], cached_vel)
        cached_health = torch.where(visible, state.ship_health[:, None, :], cached_health)
        cached_power = torch.where(visible, state.ship_power[:, None, :], cached_power)
        cached_cooldown = torch.where(visible, state.ship_cooldown[:, None, :], cached_cooldown)
        age = torch.where(
            visible, torch.zeros_like(age), torch.where(hidden, age + 1, torch.zeros_like(age))
        )
        valid |= visible

        forecast_pos = cached_pos + cached_vel * decision_dt
        world_w, world_h = ship_config.world_size
        forecast_pos = torch.complex(
            forecast_pos.real.remainder(world_w), forecast_pos.imag.remainder(world_h)
        )

        action = agent.get_actions(state, sight.ship)
        for _ in range(env_config.action_repeat):
            env.tick(action)

        next_state = env.state
        transition_contiguous = ~next_state.ship_respawned[:, None, :]
        dx = forecast_pos.real - next_state.ship_pos[:, None, :].real + world_w / 2.0
        dy = forecast_pos.imag - next_state.ship_pos[:, None, :].imag + world_h / 2.0
        dx = dx.remainder(world_w) - world_w / 2.0
        dy = dy.remainder(world_h) - world_h / 2.0
        errors = {
            "position_px": torch.sqrt(dx.square() + dy.square()),
            "velocity_px_s": (cached_vel - next_state.ship_vel[:, None, :]).abs(),
            "health": (cached_health - next_state.ship_health[:, None, :]).abs(),
            "power": (cached_power - next_state.ship_power[:, None, :]).abs(),
            "cooldown_s": (cached_cooldown - next_state.ship_cooldown[:, None, :]).abs(),
        }
        visible &= transition_contiguous
        hidden &= transition_contiguous
        _add(stats, "visible", visible, errors)
        _add(stats, "hidden", hidden, errors)
        age_seconds = age.float() * decision_dt
        lower = 0.0
        for upper in _AGE_EDGES:
            _add(
                stats,
                f"hidden_age_{lower:g}_{upper:g}s",
                hidden & (age_seconds > lower) & (age_seconds <= upper),
                errors,
            )
            lower = upper
        _add(stats, "hidden_age_30_inf_s", hidden & (age_seconds > 30.0), errors)

        cached_pos = torch.where(valid, forecast_pos, cached_pos)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    # Isolate the fixed-shape production cache overhead from physics/scripted work.
    perceived, _ = perceived_observation_from_state(
        env.state, ship_config, env_config, include_bullets=False
    )
    coordinator = build_standard_coordinator(ship_config)
    tracker = DualBeliefTracker(b, n, decision_dt, coordinator, device)
    composed = tracker.compose(perceived)
    prediction = torch.zeros((b, n, coordinator.total_prediction_dimension), device=device)

    def belief_iteration() -> None:
        nonlocal composed
        tracker.advance(composed, prediction, prediction)
        composed = tracker.compose(perceived)

    for _ in range(5):
        belief_iteration()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    profile_started = time.perf_counter()
    for _ in range(profile_iterations):
        belief_iteration()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    profile_seconds = time.perf_counter() - profile_started

    result = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "games": games,
        "seconds_per_game": seconds,
        "seed": seed,
        "wall_seconds": elapsed,
        "simulated_game_seconds_per_wall_second": games * seconds / elapsed,
        "peak_torch_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "predictor": "last_observation_plus_constant_velocity_position",
        "prediction_error_mean": _finish_stats(stats),
        "belief_cache_profile": {
            "iterations": profile_iterations,
            "ms_per_team_pair_batch": profile_seconds * 1000.0 / profile_iterations,
            "envs_per_second": games * profile_iterations / profile_seconds,
        },
    }
    if checkpoint is not None:
        result["learned_policy"] = _run_learned_accuracy(
            checkpoint=checkpoint,
            games=games,
            decisions=decisions,
            seed=seed,
            device=device,
            ship_config=ship_config,
            env_config=env_config,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--profile-iterations", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_suite(
        games=args.games,
        seconds=args.seconds,
        seed=args.seed,
        device=torch.device(args.device),
        profile_iterations=args.profile_iterations,
        checkpoint=args.checkpoint,
    )
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
