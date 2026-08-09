"""``collect-stats`` mode: run games between two specified agents and report stats."""

import time

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.evaluation.agents import resolve_agent_spec
from boost_and_broadside.evaluation.match import evaluate_matchup
from boost_and_broadside.evaluation.sizes import MatchupParseError, parse_matchup


def run_collect_stats_mode(
    team0_spec: str,
    team1_spec: str,
    num_envs: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    model_config: ModelConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    matchups: list[str] | None = None,
) -> None:
    """Run num_envs parallel games between team0 and team1 agents and print stats.

    Args:
        team0_spec:     Exact agent name or checkpoint path for team 0.
        team1_spec:     Agent spec for team 1.
        num_envs:       Number of games to run in parallel.
        ship_config:    Physics constants.
        env_config:     Environment sizing (num_ships will be overridden).
        model_config:   Policy architecture (needed if either spec is a checkpoint).
        device:         Torch device string.
        checkpoint_dir: Checkpoint root supplied by the CLI adapter.
        matchups:       List of matchup sizes like "1v1", "2v3". Defaults to ["2v2"].
    """
    if team0_spec == "null" or team1_spec == "null":
        raise ValueError("collect-stats does not support the 'null' agent spec")

    if not matchups:
        matchups = ["2v2"]

    B = num_envs

    for matchup in matchups:
        try:
            parsed = parse_matchup(matchup)
        except MatchupParseError:
            print(f"Skipping invalid matchup: {matchup}")
            continue
        n0, n1 = parsed
        N = parsed.num_ships

        agent0 = resolve_agent_spec(
            team0_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N
        )
        agent1 = resolve_agent_spec(
            team1_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N
        )

        t0 = time.perf_counter()
        num_0, num_1, n_tie, avg_len = evaluate_matchup(
            agent0, agent1, n0, n1, B, ship_config, env_config, device
        )
        elapsed = time.perf_counter() - t0
        sim_fps = 1.0 / ship_config.dt

        w = 56
        print(f"\n{'─' * w}")
        print(f"  collect-stats: {matchup} ({B} games)  ({device})")
        print(f"  Team 0: {team0_spec:<18}  Team 1: {team1_spec}")
        print(f"{'─' * w}")
        print(f"  Team 0 wins : {num_0:6d}  ({100 * num_0 / B:5.1f}%)")
        print(f"  Team 1 wins : {num_1:6d}  ({100 * num_1 / B:5.1f}%)")
        print(f"  Ties        : {n_tie:6d}  ({100 * n_tie / B:5.1f}%)")
        print(f"{'─' * w}")
        print(f"  Avg episode : {avg_len:7.1f} steps  ({avg_len / sim_fps:.1f}s sim)")
        print(f"  Wall time   : {elapsed:.2f}s")
        print(f"{'─' * w}\n")
