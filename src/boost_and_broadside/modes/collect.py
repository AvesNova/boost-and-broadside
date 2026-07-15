"""collect_stats mode: run parallel games between two specified agents and report stats.

Also exports ``_run_matchup`` — a low-level helper used by the PPO trainer to compute
ELO win rates without printing any output.
"""

import time
from dataclasses import replace

import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.constants import EPS
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import MVPObservation, ObsKey
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.modes.agent_factory import (
    ResolvedAgent,
    get_actions,
    init_hidden,
    reset_done_envs,
    resolve_agent_spec,
)


# _run_matchup was removed since it is no longer required centrally.


def _obs_from_state(
    state: TensorState, ship_config: ShipConfig
) -> MVPObservation:
    """Build a policy-ready MVPObservation from TensorState.

    Mirrors MVPEnvWrapper._get_obs() exactly — all values are RAW (no
    normalization). Includes obstacle tokens (team_id=2) when num_obstacles > 0.
    """
    B = state.num_envs
    M = state.num_obstacles
    dev = state.ship_pos.device

    ship_pos = torch.stack([state.ship_pos.real, state.ship_pos.imag], dim=-1)
    ship_vel = torch.stack([state.ship_vel.real, state.ship_vel.imag], dim=-1)
    ship_att = torch.stack([state.ship_attitude.real, state.ship_attitude.imag], dim=-1)
    ship_ang = state.ship_ang_vel.unsqueeze(-1)
    ship_health   = state.ship_health.unsqueeze(-1)
    ship_power    = state.ship_power.unsqueeze(-1)
    ship_cooldown = state.ship_cooldown.unsqueeze(-1)
    ship_prev_action = state.prev_action.long()                           # (B, N, 3)
    ship_radius = torch.full(
        (B, state.max_ships, 1),
        ship_config.collision_radius,
        device=dev,
        dtype=torch.float32,
    )

    if M > 0:
        obs_pos = torch.stack(
            [state.obstacle_pos.real, state.obstacle_pos.imag], dim=-1
        )
        obs_vel = torch.stack([state.obstacle_vel.real, state.obstacle_vel.imag], dim=-1)
        obs_speed = torch.norm(obs_vel, dim=-1, keepdim=True).clamp(min=EPS)
        obs_att = obs_vel / obs_speed
        obs_ang_vel    = torch.zeros(B, M, 1, device=dev)
        obs_health     = torch.full((B, M, 1), ship_config.max_health, device=dev)
        obs_power      = torch.zeros(B, M, 1, device=dev)
        obs_cooldown   = torch.zeros(B, M, 1, device=dev)
        obs_team_id    = torch.full((B, M), 2, device=dev, dtype=torch.int32)
        obs_alive      = torch.ones(B, M, device=dev, dtype=torch.bool)
        obs_prev_action = torch.zeros(B, M, 3, device=dev, dtype=torch.long)
        obs_radius     = state.obstacle_radius.unsqueeze(-1)
        return MVPObservation(data={
            ObsKey.POS:             torch.cat([ship_pos,         obs_pos],          dim=1),
            ObsKey.VEL:             torch.cat([ship_vel,         obs_vel],          dim=1),
            ObsKey.ATT:             torch.cat([ship_att,         obs_att],          dim=1),
            ObsKey.ANG_VEL:         torch.cat([ship_ang,         obs_ang_vel],      dim=1),
            ObsKey.HEALTH:          torch.cat([ship_health,      obs_health],       dim=1),
            ObsKey.POWER:           torch.cat([ship_power,       obs_power],        dim=1),
            ObsKey.COOLDOWN:        torch.cat([ship_cooldown,    obs_cooldown],     dim=1),
            ObsKey.TEAM_ID:         torch.cat([state.ship_team_id, obs_team_id],    dim=1),
            ObsKey.ALIVE:           torch.cat([state.ship_alive, obs_alive],        dim=1),
            ObsKey.PREVIOUS_ACTION: torch.cat([ship_prev_action, obs_prev_action],  dim=1),
            ObsKey.RADIUS:          torch.cat([ship_radius,      obs_radius],       dim=1),
        })

    return MVPObservation(data={
        ObsKey.POS:             ship_pos,
        ObsKey.VEL:             ship_vel,
        ObsKey.ATT:             ship_att,
        ObsKey.ANG_VEL:         ship_ang,
        ObsKey.HEALTH:          ship_health,
        ObsKey.POWER:           ship_power,
        ObsKey.COOLDOWN:        ship_cooldown,
        ObsKey.TEAM_ID:         state.ship_team_id,
        ObsKey.ALIVE:           state.ship_alive,
        ObsKey.PREVIOUS_ACTION: ship_prev_action,
        ObsKey.RADIUS:          ship_radius,
    })


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
    dev = torch.device(device)

    for matchup in matchups:
        parts = matchup.split('v')
        if len(parts) != 2:
            print(f"Skipping invalid matchup: {matchup}")
            continue
        n0, n1 = int(parts[0]), int(parts[1])
        N = n0 + n1
        M = env_config.num_obstacles
        num_tokens = N + M

        curr_env_config = replace(env_config, num_ships=N)

        agent0 = resolve_agent_spec(
            team0_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N
        )
        agent1 = resolve_agent_spec(
            team1_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N
        )

        env = TensorEnv(B, ship_config, curr_env_config, device)

        # Per-game outcome tracking: 0 = team0 wins, 1 = team1 wins, 2 = tie
        results = torch.zeros(B, dtype=torch.int32, device=dev)
        ep_lengths = torch.zeros(B, dtype=torch.int64, device=dev)
        finished = torch.zeros(B, dtype=torch.bool, device=dev)

        init_hidden(agent0, B, num_tokens, dev)
        init_hidden(agent1, B, num_tokens, dev)

        env.reset(options={"team_sizes": (n0, n1)})
        total_steps = 0
        t0 = time.perf_counter()

        while not finished.all():
            state = env.state
            obs = _obs_from_state(state, ship_config)

            action0 = get_actions(agent0, obs, state, B, N, dev)
            # Team 1 agent sees itself as team 0 (flipped team IDs)
            obs_t1_data = {k: v for k, v in obs.data.items()}
            obs_t1_data[ObsKey.TEAM_ID] = obs_t1_data[ObsKey.TEAM_ID].clone()
            obs_t1_data[ObsKey.TEAM_ID][..., :N] = 1 - obs_t1_data[ObsKey.TEAM_ID][..., :N]
            obs_t1 = MVPObservation(data=obs_t1_data)
            action1 = get_actions(agent1, obs_t1, state, B, N, dev)

            # Each agent generates actions for all ships; select by team ownership
            team_id = state.ship_team_id  # (B, N)
            action = torch.where((team_id == 0).unsqueeze(-1), action0, action1)

            dones, truncated = env.step(action)
            done_any = dones | truncated
            total_steps += B

            new_done = done_any & ~finished
            if new_done.any():
                ep_lengths[new_done] = env.state.step_count[new_done].long()

                alive = env.state.ship_alive  # (B, N) — post-step terminal state
                team = env.state.ship_team_id  # (B, N)

                team0_alive = (alive & (team == 0)).any(dim=1)  # (B,)
                team1_alive = (alive & (team == 1)).any(dim=1)  # (B,)

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

        elapsed = time.perf_counter() - t0

        # ---- Print results -------------------------------------------------------
        results_cpu = results.cpu()
        ep_lengths_cpu = ep_lengths.cpu()

        num_0 = int((results_cpu == 0).sum())
        num_1 = int((results_cpu == 1).sum())
        n_tie = int((results_cpu == 2).sum())

        avg_len = float(ep_lengths_cpu.float().mean())
        min_len = int(ep_lengths_cpu.min())
        max_len = int(ep_lengths_cpu.max())
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
        print(f"  Min / Max   : {min_len} / {max_len} steps")
        print(f"  Wall time   : {elapsed:.2f}s  ({total_steps / elapsed:,.0f} steps/s)")
        print(f"{'─' * w}\n")
