"""Side-balanced Frontline matches for scripted strategy tuning.

The legacy controller is loaded from the immutable integration-base commit so
this benchmark can compare it with a dirty working tree without copying the old
implementation into the product package.

Examples:
    .venv/bin/python benchmarks/frontline_agent_head_to_head.py pair \
        --candidate default --opponent legacy --games 8 --seconds 120
    .venv/bin/python benchmarks/frontline_agent_head_to_head.py pair \
        --candidate default --opponent checkpoint --games 4 --seconds 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import types
from dataclasses import asdict, replace
from pathlib import Path

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import MatchResult
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.evaluation.environment import create_evaluation_env
from boost_and_broadside.evaluation.match import MatchRunner
from boost_and_broadside.evaluation.tournament import load_run_config
from boost_and_broadside.train.rl.policy_io import load_policy_bundle

LEGACY_COMMIT = "d21ff5262b1862c9de564ca1916f9059f78982cb"
LEGACY_SOURCE = "src/boost_and_broadside/agents/stochastic_scripted.py"
LEGACY_SHA256 = "75c70e2fd2cdac29116efb9b052e3368885d7d0fae25e22cc6eaa6eea4145d68"
DEFAULT_CHECKPOINT = Path("checkpoints/misty-energy-739/step_000079626240.pt")


class LegacyAgentAdapter(StochasticScriptedAgent):
    """Expose the pinned pre-redesign agent through the current match interface."""

    def __init__(self, legacy, ship_config, config) -> None:
        self.legacy = legacy(ship_config, config)

    def get_actions(self, state, team_visibility=None):
        return self.legacy.get_actions(state, team_visibility)


def _legacy_agent(ship_config) -> ResolvedAgent:
    source = subprocess.run(
        ["git", "show", f"{LEGACY_COMMIT}:{LEGACY_SOURCE}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != LEGACY_SHA256:
        raise RuntimeError(f"legacy source digest changed: expected {LEGACY_SHA256}, got {digest}")
    module = types.ModuleType("_frontline_legacy_scripted")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{LEGACY_COMMIT}:{LEGACY_SOURCE}", "exec"), module.__dict__)
    legacy_config = StochasticAgentConfig()
    # These fields belonged only to the deleted role controller. The legacy
    # source reads them dynamically, so attach its base-commit defaults here.
    legacy_config.frontline_heal_health_fraction = 0.3
    legacy_config.frontline_enemy_engage_distance = 500.0
    adapter = LegacyAgentAdapter(module.StochasticScriptedAgent, ship_config, legacy_config)
    return ResolvedAgent("scripted", adapter)


def _parse_candidate(spec: str, ship_config) -> tuple[ResolvedAgent, dict]:
    if spec == "legacy":
        return _legacy_agent(ship_config), {"legacy_commit": LEGACY_COMMIT}
    if spec == "default":
        values = {}
    else:
        values = json.loads(spec)
        if not isinstance(values, dict):
            raise ValueError("candidate must be 'default' or a JSON object")
    config = StochasticAgentConfig(**values)
    return ResolvedAgent("scripted", StochasticScriptedAgent(ship_config, config)), asdict(config)


def _checkpoint_agent(path: Path, ship_config, model_config, num_ships: int) -> ResolvedAgent:
    bundle = load_policy_bundle(
        str(path),
        device="cpu",
        num_ships=num_ships,
        ship_config=ship_config,
        model_config=model_config,
    )
    return ResolvedAgent("policy", bundle.policy, bundle=bundle)


@torch.inference_mode()
def play_pair(
    candidate: ResolvedAgent,
    opponent: ResolvedAgent,
    *,
    games: int,
    team_size: int,
    seed: int,
    seconds: float,
    ship_config,
    env_config,
) -> dict:
    """Play one side-balanced batch and report results from candidate's view."""
    if games < 2 or games % 2:
        raise ValueError("games must be positive and even for exact side balance")
    num_ships = 2 * team_size
    max_steps = round(seconds / ship_config.dt)
    runtime_env = replace(env_config, num_ships=num_ships, max_episode_steps=max_steps)
    env = create_evaluation_env(games, ship_config, runtime_env, "cpu")
    half = games // 2
    candidate_team0 = torch.arange(games) < half
    team0_index = torch.where(candidate_team0, 0, 1).long()
    team1_index = 1 - team0_index
    runner = MatchRunner(
        env,
        [candidate, opponent],
        team0_index,
        team1_index,
        ship_config,
        num_ships,
    )
    runner.init_hidden()
    reset_options = {"team_sizes": (team_size, team_size)}
    env.reset(options=reset_options, seed=seed)
    finished = torch.zeros(games, dtype=torch.bool)
    candidate_captures = torch.zeros(games, dtype=torch.int32)
    opponent_captures = torch.zeros_like(candidate_captures)
    candidate_deaths = torch.zeros_like(candidate_captures)
    opponent_deaths = torch.zeros_like(candidate_captures)
    lengths = torch.zeros_like(candidate_captures)
    results = torch.full((games,), int(MatchResult.DRAW), dtype=torch.int8)
    final_front = torch.zeros(games, dtype=torch.float32)
    started = time.perf_counter()

    while not finished.all():
        dones, truncated = runner.step()
        active = ~finished
        team0_capture = env.state.team0_captured.int()
        team1_capture = env.state.team1_captured.int()
        candidate_captures += torch.where(candidate_team0, team0_capture, team1_capture) * active
        opponent_captures += torch.where(candidate_team0, team1_capture, team0_capture) * active
        team0_deaths = (env.state.ship_combat_death & (env.state.ship_team_id == 0)).sum(1).int()
        team1_deaths = (env.state.ship_combat_death & (env.state.ship_team_id == 1)).sum(1).int()
        candidate_deaths += torch.where(candidate_team0, team0_deaths, team1_deaths) * active
        opponent_deaths += torch.where(candidate_team0, team1_deaths, team0_deaths) * active
        done_any = dones | truncated
        newly_done = done_any & active
        if newly_done.any():
            lengths[newly_done] = env.state.step_count[newly_done]
            candidate_front = torch.where(
                candidate_team0, env.state.front_position, -env.state.front_position
            ).float()
            final_front[newly_done] = candidate_front[newly_done]
            match_result = env.state.match_result
            candidate_won = torch.where(
                candidate_team0,
                match_result == int(MatchResult.TEAM0_WIN),
                match_result == int(MatchResult.TEAM1_WIN),
            )
            opponent_won = torch.where(
                candidate_team0,
                match_result == int(MatchResult.TEAM1_WIN),
                match_result == int(MatchResult.TEAM0_WIN),
            )
            results[newly_done & candidate_won] = 0
            results[newly_done & opponent_won] = 1
            finished |= newly_done
        runner.reset_finished(done_any, options=reset_options)

    wins = int((results == 0).sum())
    losses = int((results == 1).sum())
    draws = games - wins - losses
    team0_results = results[candidate_team0]
    team1_results = results[~candidate_team0]

    def side_summary(side_results: torch.Tensor) -> dict:
        side_wins = int((side_results == 0).sum())
        side_losses = int((side_results == 1).sum())
        side_draws = int((side_results == 2).sum())
        return {
            "wins": side_wins,
            "losses": side_losses,
            "draws": side_draws,
            "expected_score": (side_wins + 0.5 * side_draws) / int(side_results.numel()),
        }

    return {
        "games": games,
        "team_size": team_size,
        "seed": seed,
        "seconds": seconds,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "expected_score": (wins + 0.5 * draws) / games,
        "candidate_captures": int(candidate_captures.sum()),
        "opponent_captures": int(opponent_captures.sum()),
        "candidate_combat_deaths": int(candidate_deaths.sum()),
        "opponent_combat_deaths": int(opponent_deaths.sum()),
        "mean_signed_front": float(final_front.mean()),
        "mean_steps": float(lengths.float().mean()),
        "wall_seconds": time.perf_counter() - started,
        "candidate_as_team0": side_summary(team0_results),
        "candidate_as_team1": side_summary(team1_results),
        "per_game": {
            "candidate_team": torch.where(candidate_team0, 0, 1).tolist(),
            "result": ["win" if x == 0 else "loss" if x == 1 else "draw" for x in results],
            "final_front": final_front.tolist(),
            "candidate_captures": candidate_captures.tolist(),
            "opponent_captures": opponent_captures.tolist(),
            "candidate_combat_deaths": candidate_deaths.tolist(),
            "opponent_combat_deaths": opponent_deaths.tolist(),
            "steps": lengths.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pair",))
    parser.add_argument("--candidate", default="default")
    parser.add_argument("--opponent", choices=("legacy", "checkpoint"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--team-size", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=905_001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.checkpoint.parent
    env_config, model_config, ship_config, _ = load_run_config(run_dir)
    candidate, candidate_config = _parse_candidate(args.candidate, ship_config)
    if args.opponent == "legacy":
        opponent = _legacy_agent(ship_config)
    else:
        opponent = _checkpoint_agent(args.checkpoint, ship_config, model_config, 2 * args.team_size)
    torch.set_num_threads(1)
    result = play_pair(
        candidate,
        opponent,
        games=args.games,
        team_size=args.team_size,
        seed=args.seed,
        seconds=args.seconds,
        ship_config=ship_config,
        env_config=env_config,
    )
    result.update(
        {
            "candidate": candidate_config,
            "opponent": args.opponent,
            "checkpoint": str(args.checkpoint) if args.opponent == "checkpoint" else None,
            "legacy_commit": LEGACY_COMMIT,
        }
    )
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
