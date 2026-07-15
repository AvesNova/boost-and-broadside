import matplotlib.pyplot as plt
import numpy as np
import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, RewardConfig, ShipConfig
from boost_and_broadside.env.observation import MVPObservation
from boost_and_broadside.env.wrapper import MVPEnvWrapper
from boost_and_broadside.modes.agent_factory import (
    _decode_targets_to_obs,
    get_actions,
    init_hidden,
    resolve_agent_spec,
)


def run_ar_report_mode(
    team0_spec: str,
    team1_spec: str,
    num_steps: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    rewards: RewardConfig,
    model_config: ModelConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    out_dir: str = "docs/ar_report",
):
    print("Initializing agents...")
    agent0 = resolve_agent_spec(
        team0_spec,
        ship_config,
        model_config,
        device,
        checkpoint_dir,
        num_ships=env_config.num_ships,
    )
    agent1 = resolve_agent_spec(
        team1_spec,
        ship_config,
        model_config,
        device,
        checkpoint_dir,
        num_ships=env_config.num_ships,
    )

    wrapper = MVPEnvWrapper(
        num_envs=1,
        ship_config=ship_config,
        env_config=env_config,
        rewards=rewards,
        device=device,
        obstacle_cache=None,
    )

    N = wrapper.num_ships
    num_tokens = N + env_config.num_obstacles

    print("Running ground truth simulation...")
    obs = wrapper.reset()
    init_hidden(agent0, 1, num_tokens, device)
    init_hidden(agent1, 1, num_tokens, device)

    # Save initial state for AR
    init_obs = MVPObservation(data={k: v.clone() for k, v in obs.items()})
    init_hidden0 = agent0.hidden.clone() if agent0.hidden is not None else None
    init_hidden1 = agent1.hidden.clone() if agent1.hidden is not None else None

    history_sim = []
    actions_sim = []

    for _ in range(num_steps):
        state = wrapper.state
        action0 = get_actions(agent0, obs, state, 1, N, device, return_pred_next=False)
        action1 = get_actions(agent1, obs, state, 1, N, device, return_pred_next=False)

        team_id = obs["team_id"][:, :N]
        action = torch.where((team_id == 0).unsqueeze(-1), action0, action1)
        actions_sim.append(action.clone())

        history_sim.append(
            {
                "pos": obs["pos"][:, :N].clone(),
                "vel": obs["vel"][:, :N].clone(),
                "att": obs["att"][:, :N].clone(),
                "ang_vel": obs["ang_vel"][:, :N].clone(),
                "health": obs["health"][:, :N].clone(),
                "power": obs["power"][:, :N].clone(),
                "cooldown": obs["cooldown"][:, :N].clone(),
                "alive": obs["alive"][:, :N].clone(),
                "alive_prob": torch.ones_like(obs["alive"][:, :N], dtype=torch.float32),
            }
        )

        obs, _, terminated, truncated, _ = wrapper.step(action)

        if terminated or truncated:
            print(f"Episode finished early at step {_}. Truncating rollout.")
            break

    actual_steps = len(history_sim)

    print("Running AR Rollout (Closed Loop)...")
    history_closed = _run_ar(
        agent0,
        agent1,
        init_obs,
        init_hidden0,
        init_hidden1,
        actual_steps,
        N,
        ship_config,
        actions_sim,
        True,
    )

    print("Running AR Rollout (Open Loop)...")
    history_open = _run_ar(
        agent0,
        agent1,
        init_obs,
        init_hidden0,
        init_hidden1,
        actual_steps,
        N,
        ship_config,
        None,
        False,
    )

    print("Generating plots and report...")
    _generate_report(history_sim, history_closed, history_open, ship_config, actual_steps, out_dir)
    print(f"Done! Wrote report to {out_dir}.")


def _run_ar(
    agent0,
    agent1,
    init_obs,
    init_hidden0,
    init_hidden1,
    num_steps,
    N,
    ship_config,
    forced_actions,
    is_closed_loop,
):
    obs = MVPObservation(data={k: v.clone() for k, v in init_obs.items()})
    if agent0.hidden is not None:
        agent0.hidden = init_hidden0.clone()
    if agent1.hidden is not None:
        agent1.hidden = init_hidden1.clone()

    # Get coordinator from whichever agent is a policy (prefer agent0)
    coordinator = None
    if agent0.kind == "policy":
        coordinator = agent0.agent.coordinator
    elif agent1.kind == "policy":
        coordinator = agent1.agent.coordinator

    history = []
    curr_ship_targets = coordinator.get_target_vector(obs)[:, :N] if coordinator else None

    for step in range(num_steps):
        action0, pred_next0 = get_actions(
            agent0, obs, None, 1, N, obs["pos"].device, return_pred_next=True
        )
        action1, pred_next1 = get_actions(
            agent1, obs, None, 1, N, obs["pos"].device, return_pred_next=True
        )

        team_id = obs["team_id"][:, :N]
        mask = (team_id == 0).unsqueeze(-1)

        # Merge imagined actions and predictions
        imag_action = torch.where(mask, action0, action1)
        pred_next = None
        if pred_next0 is not None and pred_next1 is not None:
            pred_next = torch.where(mask, pred_next0, pred_next1)
        elif pred_next0 is not None:
            pred_next = torch.where(mask, pred_next0, torch.zeros_like(pred_next0))
        elif pred_next1 is not None:
            pred_next = torch.where(mask, torch.zeros_like(pred_next1), pred_next1)

        # Use forced action if closed loop
        action_to_apply = (
            forced_actions[step] if is_closed_loop and forced_actions is not None else imag_action
        )

        # Record state before applying delta
        history.append(
            {
                "pos": obs["pos"][:, :N].clone(),
                "vel": obs["vel"][:, :N].clone(),
                "att": obs["att"][:, :N].clone(),
                "ang_vel": obs["ang_vel"][:, :N].clone(),
                "health": obs["health"][:, :N].clone(),
                "power": obs["power"][:, :N].clone(),
                "cooldown": obs["cooldown"][:, :N].clone(),
                "alive": obs["alive"][:, :N].clone(),
                "alive_prob": obs["alive"][:, :N].float().clone(),
            }
        )

        if pred_next is not None and coordinator is not None:
            next_ship_targets = coordinator.apply_scaled_predictions(curr_ship_targets, pred_next)
            obs = _decode_targets_to_obs(
                next_ship_targets, obs, action_to_apply, N, ship_config, coordinator
            )
            curr_ship_targets = coordinator.get_target_vector(obs)[:, :N]

    return history


def _generate_report(history_sim, history_closed, history_open, ship_config, num_steps, out_dir):
    import os

    os.makedirs(out_dir, exist_ok=True)

    num_ships = history_sim[0]["pos"].shape[1]  # shape is (1, N, 2) — dim 1 is ship count
    plot_N = 2 if num_ships == 2 else 1

    # Extract arrays for ALL ships — squeeze the leading env/batch dim (dim 1 of stacked)
    def extract_feat(hist, key):
        return np.array([h[key].squeeze(0).cpu().numpy() for h in hist])

    def get_ship_feat(hist, key):
        # returns shape: [steps, plot_N, ...]
        return extract_feat(hist, key)[:, :plot_N].astype(float)

    # Sim (All Ships)
    sim_pos_all = extract_feat(history_sim, "pos")
    sim_alive_all = extract_feat(history_sim, "alive")

    # Closed Loop (All Ships)
    cl_pos_all = extract_feat(history_closed, "pos")
    cl_alive_all = extract_feat(history_closed, "alive")

    # Open Loop (All Ships)
    ol_pos_all = extract_feat(history_open, "pos")
    ol_alive_all = extract_feat(history_open, "alive")

    # Sim (Selected Ships)
    sim_pos = get_ship_feat(history_sim, "pos")
    sim_vel = get_ship_feat(history_sim, "vel")
    sim_att = get_ship_feat(history_sim, "att")
    sim_ang_vel = get_ship_feat(history_sim, "ang_vel")
    sim_health = get_ship_feat(history_sim, "health")
    sim_power = get_ship_feat(history_sim, "power")
    sim_cooldown = get_ship_feat(history_sim, "cooldown")
    sim_alive = extract_feat(history_sim, "alive")[:, :plot_N]

    # Closed Loop (Selected Ships)
    cl_pos = get_ship_feat(history_closed, "pos")
    cl_vel = get_ship_feat(history_closed, "vel")
    cl_att = get_ship_feat(history_closed, "att")
    cl_ang_vel = get_ship_feat(history_closed, "ang_vel")
    cl_health = get_ship_feat(history_closed, "health")
    cl_power = get_ship_feat(history_closed, "power")
    cl_cooldown = get_ship_feat(history_closed, "cooldown")
    cl_alive = extract_feat(history_closed, "alive")[:, :plot_N]
    cl_alive_prob = get_ship_feat(history_closed, "alive_prob")

    # Open Loop (Selected Ships)
    ol_pos = get_ship_feat(history_open, "pos")
    ol_vel = get_ship_feat(history_open, "vel")
    ol_att = get_ship_feat(history_open, "att")
    ol_ang_vel = get_ship_feat(history_open, "ang_vel")
    ol_health = get_ship_feat(history_open, "health")
    ol_power = get_ship_feat(history_open, "power")
    ol_cooldown = get_ship_feat(history_open, "cooldown")
    ol_alive = extract_feat(history_open, "alive")[:, :plot_N]
    ol_alive_prob = get_ship_feat(history_open, "alive_prob")

    def clamp_alive_prob(alive_prob_arr, alive_arr):
        for s in range(plot_N):
            d = np.where(~alive_arr[:, s])[0]
            if len(d) > 0:
                alive_prob_arr[d[0] :, s] = 0.0
        return alive_prob_arr

    cl_alive_prob = clamp_alive_prob(cl_alive_prob, cl_alive)
    ol_alive_prob = clamp_alive_prob(ol_alive_prob, ol_alive)

    # Unwrap position arrays per-ship
    def unwrap_1d(arr_1d, W):
        unwrapped = arr_1d.copy()
        for i in range(1, len(unwrapped)):
            diff = unwrapped[i] - unwrapped[i - 1]
            if diff > W / 2:
                unwrapped[i:] -= W
            elif diff < -W / 2:
                unwrapped[i:] += W
        return unwrapped

    def unwrap_pos(pos, W_x, W_y):
        # pos shape: (steps, plot_N, 2)
        uw = pos.copy()
        for s in range(pos.shape[1]):
            uw[:, s, 0] = unwrap_1d(pos[:, s, 0], W_x)
            uw[:, s, 1] = unwrap_1d(pos[:, s, 1], W_y)
        return uw

    W_x, W_y = ship_config.world_size
    sim_pos_uw = unwrap_pos(sim_pos, W_x, W_y)
    cl_pos_uw = unwrap_pos(cl_pos, W_x, W_y)
    ol_pos_uw = unwrap_pos(ol_pos, W_x, W_y)

    TEAM_DOT_COLORS = ["blue", "red"]  # team0=blue, team1=red
    METHOD_COLORS = {"gt": "black", "cl": "orange", "ol": "green"}

    def toroidal_center_of_mass(positions, W_x, W_y):
        """Compute the toroidal center of mass across a flat array of (x,y) points."""
        angles_x = 2 * np.pi * positions[:, 0] / W_x
        angles_y = 2 * np.pi * positions[:, 1] / W_y
        mean_x = np.arctan2(np.nanmean(np.sin(angles_x)), np.nanmean(np.cos(angles_x)))
        mean_y = np.arctan2(np.nanmean(np.sin(angles_y)), np.nanmean(np.cos(angles_y)))
        return mean_x * W_x / (2 * np.pi), mean_y * W_y / (2 * np.pi)

    # Collect all pos points for ALL displayed ships (plot_N) from unwrapped arrays,
    # then compute a toroidal CoM for the centered/cropped map
    def center_pos_uw(pos_uw_arr, com_x, com_y):
        """Shift all unwrapped positions so that CoM is at (0,0)."""
        centered = pos_uw_arr.copy()
        centered[:, :, 0] -= com_x
        centered[:, :, 1] -= com_y
        return centered

    # --- 1. Full-world 2D Game Map (ALL Ships) --- only for 2v2
    if plot_N > 1 or num_ships > 2:
        # 2v2 case: always draw the full-world map
        pass

    # Always draw full-world map for non-1v1
    def plot_trajectory_on_ax(ax_target, pos_arr_all, alive_arr_all, label, method_color, W_x, W_y):
        n_ships = pos_arr_all.shape[1]
        for s in range(n_ships):
            pos_arr = pos_arr_all[:, s]
            alive_arr = alive_arr_all[:, s]
            dot_color = TEAM_DOT_COLORS[1] if s >= n_ships // 2 else TEAM_DOT_COLORS[0]

            dead_idx = np.where(~alive_arr)[0]
            if len(dead_idx) > 0:
                d_idx = dead_idx[0]
                ax_target.plot(
                    pos_arr[d_idx, 0],
                    pos_arr[d_idx, 1],
                    marker="x",
                    color="red",
                    markersize=5,
                    mew=1.5,
                    zorder=5,
                )

            # Draw dots first (lower z-order)
            for i in range(len(pos_arr)):
                if i % 10 == 0:
                    ax_target.plot(
                        pos_arr[i, 0],
                        pos_arr[i, 1],
                        marker="o",
                        color=dot_color,
                        markersize=2,
                        zorder=2,
                    )

            # Draw lines on top (higher z-order)
            for i in range(len(pos_arr) - 1):
                dist_x = abs(pos_arr[i + 1, 0] - pos_arr[i, 0])
                dist_y = abs(pos_arr[i + 1, 1] - pos_arr[i, 1])
                if dist_x < W_x / 2 and dist_y < W_y / 2:
                    ax_target.plot(
                        [pos_arr[i, 0], pos_arr[i + 1, 0]],
                        [pos_arr[i, 1], pos_arr[i + 1, 1]],
                        color=method_color,
                        alpha=0.6,
                        lw=1.0,
                        zorder=3,
                    )
        ax_target.plot([], [], color=method_color, label=label, lw=1)

    if num_ships > 2:  # 2v2+ only
        fig_map = plt.figure(figsize=(12, 12))
        ax_map = fig_map.add_subplot(1, 1, 1)
        ax_map.set_title("2D Trajectory Map (All Ships)")
        ax_map.set_xlim(0, W_x)
        ax_map.set_ylim(0, W_y)
        ax_map.set_aspect("equal")
        ax_map.grid(True, linestyle="--", alpha=0.5)
        plot_trajectory_on_ax(
            ax_map, sim_pos_all, sim_alive_all, "Ground Truth", METHOD_COLORS["gt"], W_x, W_y
        )
        plot_trajectory_on_ax(
            ax_map, cl_pos_all, cl_alive_all, "AR Closed Loop", METHOD_COLORS["cl"], W_x, W_y
        )
        plot_trajectory_on_ax(
            ax_map, ol_pos_all, ol_alive_all, "AR Open Loop", METHOD_COLORS["ol"], W_x, W_y
        )
        ax_map.legend()
        fig_map.tight_layout()
        fig_map.savefig(os.path.join(out_dir, "2d_map.png"))
        plt.close(fig_map)

    # Use toroidal CoM on ONLY the ground truth positions — anchors center to actual game
    gt_pts_raw = sim_pos[:, :plot_N].reshape(-1, 2)
    com_x, com_y = toroidal_center_of_mass(gt_pts_raw, W_x, W_y)

    sim_pos_c = center_pos_uw(sim_pos_uw, com_x, com_y)
    cl_pos_c = center_pos_uw(cl_pos_uw, com_x, com_y)
    ol_pos_c = center_pos_uw(ol_pos_uw, com_x, com_y)

    map_title = "2D Trajectory Map" if num_ships == 2 else "2D Trajectory Map (Featured Ships)"
    fig_map2 = plt.figure(figsize=(8, 8))
    ax_map2 = fig_map2.add_subplot(1, 1, 1)
    ax_map2.set_title(map_title)
    ax_map2.set_aspect("equal")
    ax_map2.grid(True, linestyle="--", alpha=0.5)

    def plot_centered_trajectory(ax_target, pos_c, alive_arr_all, label, method_color):
        for s in range(plot_N):
            pos_s = pos_c[:, s]
            alive_arr = alive_arr_all[:, s]
            dot_color = TEAM_DOT_COLORS[1] if s >= num_ships // 2 else TEAM_DOT_COLORS[0]

            dead_idx = np.where(~alive_arr)[0]
            if len(dead_idx) > 0:
                d_idx = dead_idx[0]
                ax_target.plot(
                    pos_s[d_idx, 0],
                    pos_s[d_idx, 1],
                    marker="x",
                    color="red",
                    markersize=5,
                    mew=1.5,
                    zorder=5,
                )

            # Draw dots first (lower z-order)
            for i in range(len(pos_s)):
                if i % 10 == 0:
                    ax_target.plot(
                        pos_s[i, 0],
                        pos_s[i, 1],
                        marker="o",
                        color=dot_color,
                        markersize=2,
                        zorder=2,
                    )

            # Draw lines on top (higher z-order)
            for i in range(len(pos_s) - 1):
                ax_target.plot(
                    [pos_s[i, 0], pos_s[i + 1, 0]],
                    [pos_s[i, 1], pos_s[i + 1, 1]],
                    color=method_color,
                    alpha=0.6,
                    lw=1.0,
                    zorder=3,
                )
        ax_target.plot([], [], color=method_color, label=label, lw=1)

    alive_for_plot = sim_alive[:, :plot_N]
    cl_alive_for_plot = cl_alive[:, :plot_N]
    ol_alive_for_plot = ol_alive[:, :plot_N]

    plot_centered_trajectory(
        ax_map2, sim_pos_c, alive_for_plot, "Ground Truth", METHOD_COLORS["gt"]
    )
    plot_centered_trajectory(
        ax_map2, cl_pos_c, cl_alive_for_plot, "AR Closed Loop", METHOD_COLORS["cl"]
    )
    plot_centered_trajectory(
        ax_map2, ol_pos_c, ol_alive_for_plot, "AR Open Loop", METHOD_COLORS["ol"]
    )
    ax_map2.legend()

    # Crop to GT trajectory bounds — AR may go off-screen if it diverges
    gt_cx = sim_pos_c[:, :, 0].ravel()
    gt_cy = sim_pos_c[:, :, 1].ravel()
    if len(gt_cx) > 0:
        px = max((np.nanmax(gt_cx) - np.nanmin(gt_cx)) * 0.15, W_x * 0.05)
        py = max((np.nanmax(gt_cy) - np.nanmin(gt_cy)) * 0.15, W_y * 0.05)
        ax_map2.set_xlim(np.nanmin(gt_cx) - px, np.nanmax(gt_cx) + px)
        ax_map2.set_ylim(np.nanmin(gt_cy) - py, np.nanmax(gt_cy) + py)

    fig_map2.tight_layout()
    # For 1v1, this IS the primary map; for 2v2+, it's a supplemental ship0 map
    map2_filename = "2d_map.png" if num_ships == 2 else "2d_map_ship0.png"
    fig_map2.savefig(os.path.join(out_dir, map2_filename))
    plt.close(fig_map2)

    # --- 1.75. 2D Velocity-Space Map ---
    fig_vel_map = plt.figure(figsize=(8, 8))
    ax_vel_map = fig_vel_map.add_subplot(1, 1, 1)
    ax_vel_map.set_title("Velocity Space (Vx, Vy)")
    ax_vel_map.set_xlabel("Vx")
    ax_vel_map.set_ylabel("Vy")
    ax_vel_map.set_aspect("equal")
    ax_vel_map.grid(True, linestyle="--", alpha=0.5)

    def plot_vel_trajectory(ax_target, vel_arr, alive_arr_all, label, method_color):
        for s in range(plot_N):
            vel_s = vel_arr[:, s]  # shape (steps, 2)
            alive_arr = alive_arr_all[:, s]
            dot_color = TEAM_DOT_COLORS[1] if s >= num_ships // 2 else TEAM_DOT_COLORS[0]

            dead_idx = np.where(~alive_arr)[0]
            if len(dead_idx) > 0:
                d_idx = dead_idx[0]
                ax_target.plot(
                    vel_s[d_idx, 0],
                    vel_s[d_idx, 1],
                    marker="x",
                    color="red",
                    markersize=5,
                    mew=1.5,
                    zorder=5,
                )

            # Dots first
            for i in range(len(vel_s)):
                if i % 10 == 0:
                    ax_target.plot(
                        vel_s[i, 0],
                        vel_s[i, 1],
                        marker="o",
                        color=dot_color,
                        markersize=2,
                        zorder=2,
                    )

            # Lines on top
            for i in range(len(vel_s) - 1):
                ax_target.plot(
                    [vel_s[i, 0], vel_s[i + 1, 0]],
                    [vel_s[i, 1], vel_s[i + 1, 1]],
                    color=method_color,
                    alpha=0.6,
                    lw=1.0,
                    zorder=3,
                )
        ax_target.plot([], [], color=method_color, label=label, lw=1)

    plot_vel_trajectory(ax_vel_map, sim_vel, alive_for_plot, "Ground Truth", METHOD_COLORS["gt"])
    plot_vel_trajectory(
        ax_vel_map, cl_vel, cl_alive_for_plot, "AR Closed Loop", METHOD_COLORS["cl"]
    )
    plot_vel_trajectory(ax_vel_map, ol_vel, ol_alive_for_plot, "AR Open Loop", METHOD_COLORS["ol"])
    ax_vel_map.legend()
    fig_vel_map.tight_layout()
    fig_vel_map.savefig(os.path.join(out_dir, "2d_vel_map.png"))
    plt.close(fig_vel_map)

    # --- 2. Divergence Stats (MAE & L2) Line Charts ---
    def calc_toroidal_euclidean(pos1, pos2, W_x, W_y, alive1, alive2):
        dx = np.abs(pos1[..., 0] - pos2[..., 0])
        dy = np.abs(pos1[..., 1] - pos2[..., 1])
        dx = np.minimum(dx, W_x - dx)
        dy = np.minimum(dy, W_y - dy)
        dist = np.sqrt(dx**2 + dy**2)
        dist[~(alive1 & alive2)] = np.nan
        return dist

    def calc_euclidean(arr1, arr2, alive1, alive2):
        dist = np.linalg.norm(arr1 - arr2, axis=-1)
        dist[~(alive1 & alive2)] = np.nan
        return dist

    def calc_4d_euclidean(pos1, vel1, pos2, vel2, W_x, W_y, alive1, alive2):
        dx = np.abs(pos1[..., 0] - pos2[..., 0])
        dy = np.abs(pos1[..., 1] - pos2[..., 1])
        dx = np.minimum(dx, W_x - dx)
        dy = np.minimum(dy, W_y - dy)

        dvx = vel1[..., 0] - vel2[..., 0]
        dvy = vel1[..., 1] - vel2[..., 1]

        dist = np.sqrt(dx**2 + dy**2 + dvx**2 + dvy**2)
        dist[~(alive1 & alive2)] = np.nan
        return dist

    def calc_mae(arr1, arr2, alive1, alive2):
        err = np.mean(np.abs(arr1 - arr2), axis=-1)
        err[~(alive1 & alive2)] = np.nan
        return err

    steps = np.arange(num_steps)

    err_pos_cl = calc_toroidal_euclidean(cl_pos, sim_pos, W_x, W_y, cl_alive, sim_alive)
    err_pos_ol = calc_toroidal_euclidean(ol_pos, sim_pos, W_x, W_y, ol_alive, sim_alive)

    err_vel_cl = calc_euclidean(cl_vel, sim_vel, cl_alive, sim_alive)
    err_vel_ol = calc_euclidean(ol_vel, sim_vel, ol_alive, sim_alive)

    err_4d_cl = calc_4d_euclidean(cl_pos, cl_vel, sim_pos, sim_vel, W_x, W_y, cl_alive, sim_alive)
    err_4d_ol = calc_4d_euclidean(ol_pos, ol_vel, sim_pos, sim_vel, W_x, W_y, ol_alive, sim_alive)

    mae_features = [
        ("position", "Position Error (Toroidal L2)", err_pos_cl, err_pos_ol),
        ("velocity", "Velocity Error (L2)", err_vel_cl, err_vel_ol),
        ("pos_vel_4d", "Pos+Vel 4D Error (L2)", err_4d_cl, err_4d_ol),
        (
            "attitude",
            "Attitude Error (MAE)",
            calc_mae(cl_att, sim_att, cl_alive, sim_alive),
            calc_mae(ol_att, sim_att, ol_alive, sim_alive),
        ),
        (
            "health",
            "Health Error (MAE)",
            calc_mae(cl_health, sim_health, cl_alive, sim_alive),
            calc_mae(ol_health, sim_health, ol_alive, sim_alive),
        ),
        (
            "power",
            "Power Error (MAE)",
            calc_mae(cl_power, sim_power, cl_alive, sim_alive),
            calc_mae(ol_power, sim_power, ol_alive, sim_alive),
        ),
    ]

    for file_key, name, err_cl, err_ol in mae_features:
        fig_err = plt.figure(figsize=(8, 4))
        ax_err = fig_err.add_subplot(1, 1, 1)
        ax_err.set_title(name)
        for s in range(plot_N):
            l_style = "--" if s >= num_ships // 2 else "-"
            ax_err.plot(
                steps,
                err_cl[:, s],
                label=f"CL (S{s})",
                color=METHOD_COLORS["cl"],
                linestyle=l_style,
            )
            ax_err.plot(
                steps,
                err_ol[:, s],
                label=f"OL (S{s})",
                color=METHOD_COLORS["ol"],
                linestyle=l_style,
                alpha=0.8,
            )
        ax_err.set_xlabel("Steps")
        ax_err.set_ylabel("Error")
        ax_err.legend()
        ax_err.grid(True, alpha=0.3)
        fig_err.tight_layout()
        fig_err.savefig(os.path.join(out_dir, f"mae_{file_key}.png"))
        plt.close(fig_err)

    # --- 3. Feature Divergence Line Plots ---
    features = [
        ("position_x", "Position X", sim_pos_uw[..., 0], cl_pos_uw[..., 0], ol_pos_uw[..., 0]),
        ("velocity_x", "Velocity X", sim_vel[..., 0], cl_vel[..., 0], ol_vel[..., 0]),
        ("angle_cos", "Angle (cos)", sim_att[..., 0], cl_att[..., 0], ol_att[..., 0]),
        ("angular_vel", "Angular Vel", sim_ang_vel[..., 0], cl_ang_vel[..., 0], ol_ang_vel[..., 0]),
        (
            "angular_vel_scaled",
            "Angular Vel (Scaled to GT)",
            sim_ang_vel[..., 0],
            cl_ang_vel[..., 0],
            ol_ang_vel[..., 0],
        ),
        ("health", "Health", sim_health[..., 0], cl_health[..., 0], ol_health[..., 0]),
        ("power", "Power", sim_power[..., 0], cl_power[..., 0], ol_power[..., 0]),
        ("cooldown", "Cooldown", sim_cooldown[..., 0], cl_cooldown[..., 0], ol_cooldown[..., 0]),
        ("alive", "Alive Prob", sim_alive, cl_alive_prob, ol_alive_prob),
    ]

    for file_key, name, sim_f, cl_f, ol_f in features:
        fig_feat = plt.figure(figsize=(8, 4))
        ax = fig_feat.add_subplot(1, 1, 1)
        ax.set_title(name)

        for s in range(plot_N):
            l_style = "--" if s >= num_ships // 2 else "-"
            if name == "Alive Prob":
                ax.plot(
                    steps,
                    sim_f[:, s].astype(float),
                    label=f"GT (S{s} Alive)",
                    color=METHOD_COLORS["gt"],
                    linestyle=l_style,
                )
            else:
                ax.plot(
                    steps,
                    sim_f[:, s],
                    label=f"GT S{s}",
                    color=METHOD_COLORS["gt"],
                    linestyle=l_style,
                )
            ax.plot(
                steps,
                cl_f[:, s],
                label=f"CL S{s}",
                color=METHOD_COLORS["cl"],
                alpha=0.9,
                linestyle=l_style,
            )
            ax.plot(
                steps,
                ol_f[:, s],
                label=f"OL S{s}",
                color=METHOD_COLORS["ol"],
                alpha=0.9,
                linestyle=l_style,
            )

        ax.grid(True, alpha=0.3)
        ax.legend()

        if name == "Angular Vel (Scaled to GT)":
            gt_min = np.min(sim_f)
            gt_max = np.max(sim_f)
            pad = (gt_max - gt_min) * 0.1
            if pad == 0:
                pad = 0.1
            ax.set_ylim(gt_min - pad, gt_max + pad)

        fig_feat.tight_layout()
        fig_feat.savefig(os.path.join(out_dir, f"feature_{file_key}.png"))
        plt.close(fig_feat)

    # --- Markdown Report ---
    with open(os.path.join(out_dir, "ar_report.md"), "w") as f:
        f.write("# Autoregressive Rollout Report\n\n")
        f.write(
            "This report compares the ground truth simulation with closed-loop "
            "(forced actions) and open-loop (imagined actions) autoregressive rollouts.\n\n"
        )

        if num_ships > 2:
            f.write("## 2D Trajectory Map (All Ships)\n")
            f.write("![Trajectory Map](2d_map.png)\n\n")
            f.write("## 2D Trajectory Map (Featured Ships — Centered)\n")
            f.write("![Trajectory Map Ship 0](2d_map_ship0.png)\n\n")
        else:
            f.write("## 2D Trajectory Map\n")
            f.write("![Trajectory Map](2d_map.png)\n\n")

        f.write("## 2D Velocity Space Map\n")
        f.write("![Velocity Space](2d_vel_map.png)\n\n")

        f.write("## Error Metrics Over Time\n")
        f.write("Calculated only while both the ground truth and rollout ships are alive.\n\n")
        for file_key, name, _, _ in mae_features:
            f.write(f"### {name}\n")
            f.write(f"![{name}](mae_{file_key}.png)\n\n")

        f.write("## Feature Divergence\n")
        for file_key, name, _, _, _ in features:
            f.write(f"### {name}\n")
            f.write(f"![{name}](feature_{file_key}.png)\n\n")
