"""Scripted Frontline play statistics: how a scripted controller flies and fights.

Runs a scripted controller against itself on the ``rl`` profile and reports what
the flight model changes for it: hit quality, kill rate, speed, time above the
corner, and how often the turn command flips. The script uses only interfaces
that exist on both sides of the slip-model change, so it can be run from an
older checkout for a before/after comparison.

``--candidate`` instead plays side-balanced full Frontline matches of a
candidate controller configuration (``StochasticAgentConfig`` overrides as
JSON) against the default one and reports the candidate's results.

Examples:
    uv run --no-sync python benchmarks/scripted_duels.py --out duels.json
    uv run --no-sync python benchmarks/scripted_duels.py \
        --candidate '{"turn_angle_ramp": [0.05, 0.15]}'
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EnvConfig, MatchResult
from boost_and_broadside.constants import TurnActions
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.perception import team_visibility_from_state
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.runtime.actions import PendingActionState, advance_autonomous_decision

CORNER_SPEED = 100.0
_LEFT = {int(TurnActions.TURN_LEFT), int(TurnActions.SHARP_LEFT)}
_RIGHT = {int(TurnActions.TURN_RIGHT), int(TurnActions.SHARP_RIGHT)}


def _side(turn: torch.Tensor) -> torch.Tensor:
    """-1 left, 0 straight, +1 right, for a turn-action tensor."""
    left = torch.zeros_like(turn, dtype=torch.bool)
    right = torch.zeros_like(turn, dtype=torch.bool)
    for code in _LEFT:
        left |= turn == code
    for code in _RIGHT:
        right |= turn == code
    return right.long() - left.long()


def _env_config(profile) -> EnvConfig:
    return EnvConfig(
        num_ships=profile.num_ships,
        num_fields=profile.num_fields,
        max_bullets=profile.max_bullets,
        max_episode_steps=profile.max_episode_steps,
        action_repeat=profile.action_repeat,
        spawn_resource_spread=profile.spawn_resource_spread,
        frontline=profile.frontline,
        vision_range=profile.vision_range,
        zones_occlude=profile.zones_occlude,
        spawn_reveal=profile.spawn_reveal,
    )


@torch.inference_mode()
def head_to_head(
    device: torch.device,
    candidate: StochasticAgentConfig,
    baseline: StochasticAgentConfig,
    *,
    envs: int,
    seed: int,
) -> dict:
    """One full match per environment; the candidate plays team 0 in the first half."""
    profile = PROFILES["rl"]
    config = profile.ship_config
    env_config = _env_config(profile)
    torch.manual_seed(seed)
    env = TensorEnv(envs, config, env_config, device)
    env.reset(seed=seed)
    agents = (
        StochasticScriptedAgent(config, candidate),
        StochasticScriptedAgent(config, baseline),
    )
    candidate_team = (torch.arange(envs, device=device) >= envs // 2).int()  # (B,)
    pending = PendingActionState.allocate(envs, profile.num_ships, device)
    finished = torch.zeros(envs, dtype=torch.bool, device=device)
    result = torch.full((envs,), int(MatchResult.DRAW), dtype=torch.int8, device=device)
    while not bool(finished.all()):
        sight = team_visibility_from_state(env.state, config, env_config, perceive_bullets=False)
        mine = env.state.ship_team_id == candidate_team[:, None]  # (B, N)
        chosen = agents[0].get_actions(env.state, sight.ship)
        other = agents[1].get_actions(env.state, sight.ship)
        selected = torch.where(mine.unsqueeze(-1), chosen, other)
        done, truncated, _ = advance_autonomous_decision(env, pending, selected)
        ended = (done | truncated) & ~finished
        result = torch.where(ended, env.state.match_result, result)
        finished |= done | truncated
        if bool((done | truncated).any()):
            env.reset_envs(done | truncated)
            pending.reset(done | truncated)
    won = torch.where(
        candidate_team == 0,
        result == int(MatchResult.TEAM0_WIN),
        result == int(MatchResult.TEAM1_WIN),
    )
    lost = torch.where(
        candidate_team == 0,
        result == int(MatchResult.TEAM1_WIN),
        result == int(MatchResult.TEAM0_WIN),
    )
    wins, losses = int(won.sum()), int(lost.sum())
    return {
        "games": envs,
        "seed": seed,
        "wins": wins,
        "losses": losses,
        "draws": envs - wins - losses,
        "score": round((wins + 0.5 * (envs - wins - losses)) / envs, 4),
    }


@torch.inference_mode()
def run(
    device: torch.device,
    *,
    envs: int,
    decisions: int,
    seed: int,
    agent_config: StochasticAgentConfig | None = None,
) -> dict:
    profile = PROFILES["rl"]
    config = profile.ship_config
    env_config = _env_config(profile)
    torch.manual_seed(seed)
    env = TensorEnv(envs, config, env_config, device)
    env.reset(seed=seed)
    scripted = StochasticScriptedAgent(config, agent_config or StochasticAgentConfig())
    pending = PendingActionState.allocate(envs, profile.num_ships, device)
    zero = torch.zeros((), device=device)
    totals = {
        "shots": zero.clone(),
        "combat_damage": zero.clone(),
        "kills": zero.clone(),
        "live_ship_decisions": zero.clone(),
        "speed_sum": zero.clone(),
        "above_corner": zero.clone(),
        "turn_changes": zero.clone(),
        "side_flips": zero.clone(),
        "aim_error_sum": zero.clone(),
        "aimed_shots": zero.clone(),
    }
    previous_turn = None
    for _ in range(decisions):
        sight = team_visibility_from_state(env.state, config, env_config, perceive_bullets=False)
        selected = scripted.get_actions(env.state, sight.ship)
        applied = pending.applied_action()  # (B, N, 3)
        shooters_before = env.state.ship_pos
        done, truncated, _ = advance_autonomous_decision(env, pending, selected)
        state = env.state
        live = state.ship_alive & ~state.ship_respawned  # (B, N)
        turn = applied[..., 1].long()  # (B, N)
        speed = state.ship_local_index * state.ship_vel.abs()  # (B, N)
        totals["live_ship_decisions"] += live.sum()
        totals["speed_sum"] += (speed * live).sum()
        totals["above_corner"] += ((speed > CORNER_SPEED) & live).sum()
        totals["shots"] += state.ship_is_shooting.sum()
        totals["combat_damage"] += state.ship_combat_damage.sum()
        totals["kills"] += state.ship_combat_death.sum()
        if previous_turn is not None:
            totals["turn_changes"] += ((turn != previous_turn) & live).sum()
            flips = (_side(turn) * _side(previous_turn)) < 0
            totals["side_flips"] += (flips & live).sum()
        previous_turn = turn

        # Aim error at the moment of firing: the gun's angle to the nearest
        # enemy, the quantity a lagging nose would degrade.
        enemy = state.ship_team_id[:, :, None] != state.ship_team_id[:, None, :]
        offset = state.ship_pos[:, None, :] - shooters_before[:, :, None]  # (B, N, N)
        world_w, world_h = config.world_size
        offset = torch.complex(
            (offset.real + world_w / 2) % world_w - world_w / 2,
            (offset.imag + world_h / 2) % world_h - world_h / 2,
        )
        distance = torch.where(enemy & state.ship_alive[:, None, :], offset.abs(), math.inf)
        nearest = distance.argmin(-1, keepdim=True)  # (B, N, 1)
        bearing = offset.gather(-1, nearest).squeeze(-1)  # (B, N)
        error = torch.angle(bearing * torch.conj(state.ship_attitude)).abs()
        fired = state.ship_is_shooting & live
        totals["aim_error_sum"] += (error * fired).sum()
        totals["aimed_shots"] += fired.sum()

        finished = done | truncated
        if bool(finished.any()):
            env.reset_envs(finished)
            pending.reset(finished)
            previous_turn = None

    t = {k: float(v) for k, v in totals.items()}
    decision_seconds = config.dt * profile.action_repeat
    ship_minutes = t["live_ship_decisions"] * decision_seconds / 60.0
    return {
        "envs": envs,
        "decisions": decisions,
        "seed": seed,
        "ship_minutes": round(ship_minutes, 1),
        "damage_per_shot": round(t["combat_damage"] / max(t["shots"], 1.0), 4),
        "shots_per_ship_minute": round(t["shots"] / ship_minutes, 2),
        "kills_per_ship_minute": round(t["kills"] / ship_minutes, 4),
        "mean_proper_speed": round(t["speed_sum"] / t["live_ship_decisions"], 2),
        "fraction_above_corner": round(t["above_corner"] / t["live_ship_decisions"], 4),
        "turn_change_rate": round(t["turn_changes"] / t["live_ship_decisions"], 4),
        "turn_side_flip_rate": round(t["side_flips"] / t["live_ship_decisions"], 4),
        "mean_aim_error_at_shot_deg": round(
            math.degrees(t["aim_error_sum"] / max(t["aimed_shots"], 1.0)), 3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--decisions", type=int, default=3000)
    parser.add_argument("--seeds", default="271828,314159")
    parser.add_argument("--candidate", default=None, help="config overrides as JSON")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    started = time.perf_counter()
    if args.candidate is None:
        rows = [
            run(device, envs=args.envs, decisions=args.decisions, seed=int(seed))
            for seed in args.seeds.split(",")
        ]
    else:
        overrides = {
            k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(args.candidate).items()
        }
        candidate = replace(StochasticAgentConfig(), **overrides)
        rows = [
            {
                "candidate": overrides,
                **head_to_head(
                    device, candidate, StochasticAgentConfig(), envs=args.envs, seed=int(seed)
                ),
                "stats": run(
                    device,
                    envs=args.envs,
                    decisions=args.decisions,
                    seed=int(seed),
                    agent_config=candidate,
                ),
            }
            for seed in args.seeds.split(",")
        ]
    for row in rows:
        print(json.dumps(row))
    if args.out is not None:
        args.out.write_text(
            json.dumps(
                {"runs": rows, "elapsed_seconds": round(time.perf_counter() - started, 1)},
                indent=1,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
