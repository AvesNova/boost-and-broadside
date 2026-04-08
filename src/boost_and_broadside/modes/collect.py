"""collect_stats mode: run parallel games between two specified agents and report stats.

Also exports ``_run_matchup`` — a low-level helper used by the PPO trainer to compute
ELO win rates without printing any output.
"""

import time

import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
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
) -> dict[str, torch.Tensor]:
    """Build a policy-ready obs dict from TensorState.

    Mirrors MVPEnvWrapper._get_obs() exactly so policy agents see the same
    observations here as they do during training.
    """
    world_w, world_h = ship_config.world_size
    return {
        "pos": torch.stack(
            [state.ship_pos.real / world_w, state.ship_pos.imag / world_h], dim=-1
        ),
        "vel": torch.stack([state.ship_vel.real, state.ship_vel.imag], dim=-1),
        "att": torch.stack(
            [state.ship_attitude.real, state.ship_attitude.imag], dim=-1
        ),
        "ang_vel": state.ship_ang_vel.unsqueeze(-1),
        "scalars": torch.stack(
            [
                state.ship_health / ship_config.max_health,
                state.ship_power / ship_config.max_power,
                (state.ship_cooldown / ship_config.firing_cooldown).clamp(0.0, 1.0),
            ],
            dim=-1,
        ),
        "team_id": state.ship_team_id,
        "alive": state.ship_alive,
        "prev_action": state.prev_action.long(),
    }


def run_collect_stats_mode(
    team0_spec: str,
    team1_spec: str,
    num_envs: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    model_config: ModelConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """Run num_envs parallel games between team0 and team1 agents and print stats.

    Args:
        team0_spec:     Agent spec for team 0 (random, scripted, latest, or path.pt).
        team1_spec:     Agent spec for team 1.
        num_envs:       Number of games to run in parallel.
        ship_config:    Physics constants.
        env_config:     Environment sizing.
        model_config:   Policy architecture (needed if either spec is a checkpoint).
        device:         Torch device string.
        checkpoint_dir: Root directory searched when a spec is "latest".
    """
    if team0_spec == "null" or team1_spec == "null":
        raise ValueError("collect_stats does not support the 'null' agent spec")

    B = num_envs
    N = env_config.num_ships
    dev = torch.device(device)

    agent0 = resolve_agent_spec(
        team0_spec, ship_config, model_config, device, checkpoint_dir
    )
    agent1 = resolve_agent_spec(
        team1_spec, ship_config, model_config, device, checkpoint_dir
    )

    env = TensorEnv(B, ship_config, env_config, device)

    # Per-game outcome tracking: 0 = team0 wins, 1 = team1 wins, 2 = tie
    results = torch.zeros(B, dtype=torch.int32, device=dev)
    ep_lengths = torch.zeros(B, dtype=torch.int64, device=dev)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)

    init_hidden(agent0, B, N, dev)
    init_hidden(agent1, B, N, dev)

    env.reset()
    total_steps = 0
    t0 = time.perf_counter()

    while not finished.all():
        state = env.state
        obs = _obs_from_state(state, ship_config)

        action0 = get_actions(agent0, obs, state, B, N, dev)
        action1 = get_actions(agent1, obs, state, B, N, dev)

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
            env.reset_envs(done_any)
            reset_done_envs(agent0, done_any, N)
            reset_done_envs(agent1, done_any, N)

    elapsed = time.perf_counter() - t0

    # ---- Print results -------------------------------------------------------
    results_cpu = results.cpu()
    ep_lengths_cpu = ep_lengths.cpu()

    n0 = int((results_cpu == 0).sum())
    n1 = int((results_cpu == 1).sum())
    n_tie = int((results_cpu == 2).sum())

    avg_len = float(ep_lengths_cpu.float().mean())
    min_len = int(ep_lengths_cpu.min())
    max_len = int(ep_lengths_cpu.max())
    sim_fps = 1.0 / ship_config.dt

    w = 56
    print(f"\n{'─' * w}")
    print(f"  collect_stats: {B} games  ({device})")
    print(f"  Team 0: {team0_spec:<18}  Team 1: {team1_spec}")
    print(f"{'─' * w}")
    print(f"  Team 0 wins : {n0:6d}  ({100 * n0 / B:5.1f}%)")
    print(f"  Team 1 wins : {n1:6d}  ({100 * n1 / B:5.1f}%)")
    print(f"  Ties        : {n_tie:6d}  ({100 * n_tie / B:5.1f}%)")
    print(f"{'─' * w}")
    print(f"  Avg episode : {avg_len:7.1f} steps  ({avg_len / sim_fps:.1f}s sim)")
    print(f"  Min / Max   : {min_len} / {max_len} steps")
    print(f"  Wall time   : {elapsed:.2f}s  ({total_steps / elapsed:,.0f} steps/s)")
    print(f"{'─' * w}\n")
