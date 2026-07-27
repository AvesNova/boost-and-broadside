"""collect_stats mode: run parallel games between two specified agents and report stats."""

import time
from dataclasses import replace

import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import MVPObservation, ObsKey, observation_from_state
from boost_and_broadside.modes.agent_factory import (
    ResolvedAgent,
    get_actions,
    init_hidden,
    reset_done_envs,
    resolve_agent_spec,
)


def evaluate_matchup(
    agent0: ResolvedAgent,
    agent1: ResolvedAgent,
    n0: int,
    n1: int,
    num_envs: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    device: str,
) -> tuple[int, int, int, float]:
    """Run ``num_envs`` parallel games of n0 (team 0) vs n1 (team 1) to completion.

    Both agents already carry the right ``num_ships`` (= n0 + n1); team 1 sees a
    team-flipped observation so a policy on that side plays from its own ego view.
    Returns ``(team0_wins, team1_wins, ties, mean_episode_steps)``.
    """
    N = n0 + n1
    num_tokens = N + env_config.num_obstacles
    dev = torch.device(device)
    env = TensorEnv(num_envs, ship_config, replace(env_config, num_ships=N), dev)

    results = torch.zeros(num_envs, dtype=torch.int32, device=dev)  # 0=t0, 1=t1, 2=tie
    ep_lengths = torch.zeros(num_envs, dtype=torch.int64, device=dev)
    finished = torch.zeros(num_envs, dtype=torch.bool, device=dev)

    init_hidden(agent0, num_envs, num_tokens, dev)
    init_hidden(agent1, num_envs, num_tokens, dev)
    env.reset(options={"team_sizes": (n0, n1)})

    while not finished.all():
        state = env.state
        obs = observation_from_state(state, ship_config)

        action0 = get_actions(agent0, obs, state, num_envs, N, dev)
        # Team 1 agent sees itself as team 0 (flipped team IDs).
        obs_t1_data = {k: v for k, v in obs.data.items()}
        obs_t1_data[ObsKey.TEAM_ID] = obs_t1_data[ObsKey.TEAM_ID].clone()
        obs_t1_data[ObsKey.TEAM_ID][..., :N] = 1 - obs_t1_data[ObsKey.TEAM_ID][..., :N]
        action1 = get_actions(agent1, MVPObservation(data=obs_t1_data), state, num_envs, N, dev)

        team_id = state.ship_team_id
        action = torch.where((team_id == 0).unsqueeze(-1), action0, action1)

        dones, truncated = env.step(action)
        done_any = dones | truncated
        new_done = done_any & ~finished
        if new_done.any():
            ep_lengths[new_done] = env.state.step_count[new_done].long()
            alive = env.state.ship_alive
            team = env.state.ship_team_id
            team0_alive = (alive & (team == 0)).any(dim=1)
            team1_alive = (alive & (team == 1)).any(dim=1)
            team0_won = new_done & team0_alive & ~team1_alive
            team1_won = new_done & team1_alive & ~team0_alive
            results[team0_won] = 0
            results[team1_won] = 1
            results[new_done & ~team0_won & ~team1_won] = 2
            finished |= new_done
        if done_any.any():
            env.reset_envs(done_any, options={"team_sizes": (n0, n1)})
            reset_done_envs(agent0, done_any, num_tokens)
            reset_done_envs(agent1, done_any, num_tokens)

    results_cpu = results.cpu()
    return (
        int((results_cpu == 0).sum()),
        int((results_cpu == 1).sum()),
        int((results_cpu == 2).sum()),
        float(ep_lengths.cpu().float().mean()),
    )


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
        team0_spec:     Agent spec for team 0 (random, scripted, latest, or path.pt).
        team1_spec:     Agent spec for team 1.
        num_envs:       Number of games to run in parallel.
        ship_config:    Physics constants.
        env_config:     Environment sizing (num_ships will be overridden).
        model_config:   Policy architecture (needed if either spec is a checkpoint).
        device:         Torch device string.
        checkpoint_dir: Root directory searched when a spec is "latest".
        matchups:       List of matchup sizes like "1v1", "2v3". Defaults to ["2v2"].
    """
    if team0_spec == "null" or team1_spec == "null":
        raise ValueError("collect_stats does not support the 'null' agent spec")

    if not matchups:
        matchups = ["2v2"]

    B = num_envs

    for matchup in matchups:
        parts = matchup.split("v")
        if len(parts) != 2:
            print(f"Skipping invalid matchup: {matchup}")
            continue
        n0, n1 = int(parts[0]), int(parts[1])
        N = n0 + n1

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
        print(f"  collect_stats: {matchup} ({B} games)  ({device})")
        print(f"  Team 0: {team0_spec:<18}  Team 1: {team1_spec}")
        print(f"{'─' * w}")
        print(f"  Team 0 wins : {num_0:6d}  ({100 * num_0 / B:5.1f}%)")
        print(f"  Team 1 wins : {num_1:6d}  ({100 * num_1 / B:5.1f}%)")
        print(f"  Ties        : {n_tie:6d}  ({100 * n_tie / B:5.1f}%)")
        print(f"{'─' * w}")
        print(f"  Avg episode : {avg_len:7.1f} steps  ({avg_len / sim_fps:.1f}s sim)")
        print(f"  Wall time   : {elapsed:.2f}s")
        print(f"{'─' * w}\n")
