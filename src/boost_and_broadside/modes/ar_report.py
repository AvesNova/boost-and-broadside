"""Autoregressive-rollout diagnostic report.

`--mode ar_report` runs one ground-truth episode, then replays it two ways through the
policy's learned next-state predictor: a *closed-loop* rollout (the recorded actions are
forced, so only the imagined dynamics drift) and an *open-loop* rollout (the policy also
imagines its own actions). It writes a set of PNG plots and a markdown report under
``out_dir`` comparing how far each imagined rollout diverges from ground truth over time.

The plotting/markdown layer is a public output contract: filenames, headings, metric
definitions, ship ordering, death markers, the every-10-steps trajectory dot sampling, and
toroidal-wrap handling are all preserved exactly. Pure calculations (position unwrapping,
toroidal center of mass, error metrics) live as module-level helpers so they can be tested
independently of matplotlib.
"""

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, RewardConfig, ShipConfig
from boost_and_broadside.env.observation import YemongObservation
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.modes.agent_factory import (
    ResolvedAgent,
    _decode_targets_to_obs,
    get_actions,
    init_hidden,
    resolve_agent_spec,
)

# Trajectory dots are drawn every N steps (part of the report's visual contract).
_DOT_INTERVAL = 10
TEAM_DOT_COLORS = ["blue", "red"]  # team0=blue, team1=red
METHOD_COLORS = {"gt": "black", "cl": "orange", "ol": "green"}

History = list[dict[str, torch.Tensor]]


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
) -> None:
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

    wrapper = YemongEnvWrapper(
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
    init_obs = YemongObservation(data={k: v.clone() for k, v in obs.items()})
    init_hidden0 = agent0.hidden.clone() if agent0.hidden is not None else None
    init_hidden1 = agent1.hidden.clone() if agent1.hidden is not None else None

    history_sim: History = []
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
        None,
        False,
    )

    print("Generating plots and report...")
    _generate_report(history_sim, history_closed, history_open, ship_config, actual_steps, out_dir)
    print(f"Done! Wrote report to {out_dir}.")


def _run_ar(
    agent0: ResolvedAgent,
    agent1: ResolvedAgent,
    init_obs: YemongObservation,
    init_hidden0: torch.Tensor | None,
    init_hidden1: torch.Tensor | None,
    num_steps: int,
    N: int,
    forced_actions: list[torch.Tensor] | None,
    is_closed_loop: bool,
) -> History:
    obs = YemongObservation(data={k: v.clone() for k, v in init_obs.items()})
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

    history: History = []
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
            obs = _decode_targets_to_obs(next_ship_targets, obs, action_to_apply, N, coordinator)
            curr_ship_targets = coordinator.get_target_vector(obs)[:, :N]

    return history


# --- Pure calculations (no matplotlib) ------------------------------------------------


def _extract_feat(hist: History, key: str) -> np.ndarray:
    """Stack one feature across a rollout, squeezing the leading env/batch dim."""
    return np.array([h[key].squeeze(0).cpu().numpy() for h in hist])


def _clamp_alive_prob(alive_prob: np.ndarray, alive: np.ndarray, plot_N: int) -> np.ndarray:
    """Zero each ship's alive-probability from its first death step onward (in place)."""
    for s in range(plot_N):
        d = np.where(~alive[:, s])[0]
        if len(d) > 0:
            alive_prob[d[0] :, s] = 0.0
    return alive_prob


def _unwrap_1d(arr_1d: np.ndarray, W: float) -> np.ndarray:
    """Undo toroidal wrapping of a 1D position sequence spanning world extent W."""
    unwrapped = arr_1d.copy()
    for i in range(1, len(unwrapped)):
        diff = unwrapped[i] - unwrapped[i - 1]
        if diff > W / 2:
            unwrapped[i:] -= W
        elif diff < -W / 2:
            unwrapped[i:] += W
    return unwrapped


def _unwrap_pos(pos: np.ndarray, W_x: float, W_y: float) -> np.ndarray:
    """Unwrap a (steps, ships, 2) position array along each world axis independently."""
    uw = pos.copy()
    for s in range(pos.shape[1]):
        uw[:, s, 0] = _unwrap_1d(pos[:, s, 0], W_x)
        uw[:, s, 1] = _unwrap_1d(pos[:, s, 1], W_y)
    return uw


def _toroidal_center_of_mass(positions: np.ndarray, W_x: float, W_y: float) -> tuple[float, float]:
    """Compute the toroidal center of mass across a flat array of (x,y) points."""
    angles_x = 2 * np.pi * positions[:, 0] / W_x
    angles_y = 2 * np.pi * positions[:, 1] / W_y
    mean_x = np.arctan2(np.nanmean(np.sin(angles_x)), np.nanmean(np.cos(angles_x)))
    mean_y = np.arctan2(np.nanmean(np.sin(angles_y)), np.nanmean(np.cos(angles_y)))
    return mean_x * W_x / (2 * np.pi), mean_y * W_y / (2 * np.pi)


def _center_pos_uw(pos_uw: np.ndarray, com_x: float, com_y: float) -> np.ndarray:
    """Shift all unwrapped positions so that the CoM is at (0,0)."""
    centered = pos_uw.copy()
    centered[:, :, 0] -= com_x
    centered[:, :, 1] -= com_y
    return centered


def _calc_toroidal_euclidean(
    pos1: np.ndarray,
    pos2: np.ndarray,
    W_x: float,
    W_y: float,
    alive1: np.ndarray,
    alive2: np.ndarray,
) -> np.ndarray:
    dx = np.abs(pos1[..., 0] - pos2[..., 0])
    dy = np.abs(pos1[..., 1] - pos2[..., 1])
    dx = np.minimum(dx, W_x - dx)
    dy = np.minimum(dy, W_y - dy)
    dist = np.sqrt(dx**2 + dy**2)
    dist[~(alive1 & alive2)] = np.nan
    return dist


def _calc_euclidean(
    arr1: np.ndarray, arr2: np.ndarray, alive1: np.ndarray, alive2: np.ndarray
) -> np.ndarray:
    dist = np.linalg.norm(arr1 - arr2, axis=-1)
    dist[~(alive1 & alive2)] = np.nan
    return dist


def _calc_4d_euclidean(
    pos1: np.ndarray,
    vel1: np.ndarray,
    pos2: np.ndarray,
    vel2: np.ndarray,
    W_x: float,
    W_y: float,
    alive1: np.ndarray,
    alive2: np.ndarray,
) -> np.ndarray:
    dx = np.abs(pos1[..., 0] - pos2[..., 0])
    dy = np.abs(pos1[..., 1] - pos2[..., 1])
    dx = np.minimum(dx, W_x - dx)
    dy = np.minimum(dy, W_y - dy)

    dvx = vel1[..., 0] - vel2[..., 0]
    dvy = vel1[..., 1] - vel2[..., 1]

    dist = np.sqrt(dx**2 + dy**2 + dvx**2 + dvy**2)
    dist[~(alive1 & alive2)] = np.nan
    return dist


def _calc_mae(
    arr1: np.ndarray, arr2: np.ndarray, alive1: np.ndarray, alive2: np.ndarray
) -> np.ndarray:
    err = np.mean(np.abs(arr1 - arr2), axis=-1)
    err[~(alive1 & alive2)] = np.nan
    return err


@dataclass
class _RolloutArrays:
    """Numpy trajectory arrays for one rollout method (ground-truth / closed / open loop).

    All arrays index ``[step, ship, ...]``. The ``*_all`` arrays keep every ship (used for
    the full-world map); the rest keep only the featured ships (``plot_N``). ``pos_uw`` is
    the toroidally-unwrapped position; ``pos_c`` is the unwrapped position re-centered on
    the ground-truth center of mass, filled in after all three methods are extracted.
    """

    pos_all: np.ndarray
    alive_all: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    att: np.ndarray
    ang_vel: np.ndarray
    health: np.ndarray
    power: np.ndarray
    cooldown: np.ndarray
    alive: np.ndarray
    alive_prob: np.ndarray
    pos_uw: np.ndarray
    pos_c: np.ndarray | None = None


def _extract_rollout_arrays(hist: History, plot_N: int, W_x: float, W_y: float) -> _RolloutArrays:
    """Pull every plotted feature out of a rollout history into a `_RolloutArrays`."""

    def featured(key: str) -> np.ndarray:
        return _extract_feat(hist, key)[:, :plot_N].astype(float)

    pos = featured("pos")
    return _RolloutArrays(
        pos_all=_extract_feat(hist, "pos"),
        alive_all=_extract_feat(hist, "alive"),
        pos=pos,
        vel=featured("vel"),
        att=featured("att"),
        ang_vel=featured("ang_vel"),
        health=featured("health"),
        power=featured("power"),
        cooldown=featured("cooldown"),
        alive=_extract_feat(hist, "alive")[:, :plot_N],
        alive_prob=featured("alive_prob"),
        pos_uw=_unwrap_pos(pos, W_x, W_y),
    )


# --- Plotting -------------------------------------------------------------------------


def _plot_ship_trajectories(
    ax: plt.Axes,
    positions: np.ndarray,
    alive: np.ndarray,
    label: str,
    method_color: str,
    num_ships: int,
    wrap_size: tuple[float, float] | None = None,
) -> None:
    """Draw one method's per-ship trajectory onto ``ax``.

    Args:
        positions:   (steps, ships, 2) point array (world, centered, or velocity space).
        alive:       (steps, ships) alive mask; the first death step gets an "x" marker.
        method_color: Line color identifying the rollout method.
        num_ships:   Total ship count — the first half are team0 (blue dots), rest team1.
        wrap_size:   World (W_x, W_y). When set, connecting lines spanning more than half
            the world are skipped (toroidal-wrap gaps); when None, all lines are drawn.
    """
    for s in range(positions.shape[1]):
        pos_s = positions[:, s]
        alive_s = alive[:, s]
        dot_color = TEAM_DOT_COLORS[1] if s >= num_ships // 2 else TEAM_DOT_COLORS[0]

        dead_idx = np.where(~alive_s)[0]
        if len(dead_idx) > 0:
            d_idx = dead_idx[0]
            ax.plot(
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
            if i % _DOT_INTERVAL == 0:
                ax.plot(
                    pos_s[i, 0],
                    pos_s[i, 1],
                    marker="o",
                    color=dot_color,
                    markersize=2,
                    zorder=2,
                )

        # Draw lines on top (higher z-order)
        for i in range(len(pos_s) - 1):
            if wrap_size is not None:
                W_x, W_y = wrap_size
                dist_x = abs(pos_s[i + 1, 0] - pos_s[i, 0])
                dist_y = abs(pos_s[i + 1, 1] - pos_s[i, 1])
                if not (dist_x < W_x / 2 and dist_y < W_y / 2):
                    continue
            ax.plot(
                [pos_s[i, 0], pos_s[i + 1, 0]],
                [pos_s[i, 1], pos_s[i + 1, 1]],
                color=method_color,
                alpha=0.6,
                lw=1.0,
                zorder=3,
            )
    ax.plot([], [], color=method_color, label=label, lw=1)


def _plot_full_world_map(
    out_dir: str,
    gt: _RolloutArrays,
    cl: _RolloutArrays,
    ol: _RolloutArrays,
    num_ships: int,
    W_x: float,
    W_y: float,
) -> None:
    """Full-world 2D map of every ship — 2v2+ only."""
    fig_map = plt.figure(figsize=(12, 12))
    ax_map = fig_map.add_subplot(1, 1, 1)
    ax_map.set_title("2D Trajectory Map (All Ships)")
    ax_map.set_xlim(0, W_x)
    ax_map.set_ylim(0, W_y)
    ax_map.set_aspect("equal")
    ax_map.grid(True, linestyle="--", alpha=0.5)
    wrap = (W_x, W_y)
    _plot_ship_trajectories(
        ax_map, gt.pos_all, gt.alive_all, "Ground Truth", METHOD_COLORS["gt"], num_ships, wrap
    )
    _plot_ship_trajectories(
        ax_map, cl.pos_all, cl.alive_all, "AR Closed Loop", METHOD_COLORS["cl"], num_ships, wrap
    )
    _plot_ship_trajectories(
        ax_map, ol.pos_all, ol.alive_all, "AR Open Loop", METHOD_COLORS["ol"], num_ships, wrap
    )
    ax_map.legend()
    fig_map.tight_layout()
    fig_map.savefig(os.path.join(out_dir, "2d_map.png"))
    plt.close(fig_map)


def _plot_centered_map(
    out_dir: str,
    gt: _RolloutArrays,
    cl: _RolloutArrays,
    ol: _RolloutArrays,
    num_ships: int,
    W_x: float,
    W_y: float,
) -> None:
    """Featured-ship 2D map, centered on the ground-truth CoM and cropped to its bounds."""
    map_title = "2D Trajectory Map" if num_ships == 2 else "2D Trajectory Map (Featured Ships)"
    fig_map2 = plt.figure(figsize=(8, 8))
    ax_map2 = fig_map2.add_subplot(1, 1, 1)
    ax_map2.set_title(map_title)
    ax_map2.set_aspect("equal")
    ax_map2.grid(True, linestyle="--", alpha=0.5)

    _plot_ship_trajectories(
        ax_map2, gt.pos_c, gt.alive, "Ground Truth", METHOD_COLORS["gt"], num_ships
    )
    _plot_ship_trajectories(
        ax_map2, cl.pos_c, cl.alive, "AR Closed Loop", METHOD_COLORS["cl"], num_ships
    )
    _plot_ship_trajectories(
        ax_map2, ol.pos_c, ol.alive, "AR Open Loop", METHOD_COLORS["ol"], num_ships
    )
    ax_map2.legend()

    # Crop to GT trajectory bounds — AR may go off-screen if it diverges
    gt_cx = gt.pos_c[:, :, 0].ravel()
    gt_cy = gt.pos_c[:, :, 1].ravel()
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


def _plot_velocity_map(
    out_dir: str,
    gt: _RolloutArrays,
    cl: _RolloutArrays,
    ol: _RolloutArrays,
    num_ships: int,
) -> None:
    """Featured-ship trajectory in velocity space (Vx, Vy)."""
    fig_vel_map = plt.figure(figsize=(8, 8))
    ax_vel_map = fig_vel_map.add_subplot(1, 1, 1)
    ax_vel_map.set_title("Velocity Space (Vx, Vy)")
    ax_vel_map.set_xlabel("Vx")
    ax_vel_map.set_ylabel("Vy")
    ax_vel_map.set_aspect("equal")
    ax_vel_map.grid(True, linestyle="--", alpha=0.5)

    _plot_ship_trajectories(
        ax_vel_map, gt.vel, gt.alive, "Ground Truth", METHOD_COLORS["gt"], num_ships
    )
    _plot_ship_trajectories(
        ax_vel_map, cl.vel, cl.alive, "AR Closed Loop", METHOD_COLORS["cl"], num_ships
    )
    _plot_ship_trajectories(
        ax_vel_map, ol.vel, ol.alive, "AR Open Loop", METHOD_COLORS["ol"], num_ships
    )
    ax_vel_map.legend()
    fig_vel_map.tight_layout()
    fig_vel_map.savefig(os.path.join(out_dir, "2d_vel_map.png"))
    plt.close(fig_vel_map)


def _compute_error_metrics(
    gt: _RolloutArrays, cl: _RolloutArrays, ol: _RolloutArrays, W_x: float, W_y: float
) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Per-step CL/OL-vs-GT divergence metrics, masked to steps where both are alive."""
    err_pos_cl = _calc_toroidal_euclidean(cl.pos, gt.pos, W_x, W_y, cl.alive, gt.alive)
    err_pos_ol = _calc_toroidal_euclidean(ol.pos, gt.pos, W_x, W_y, ol.alive, gt.alive)

    err_vel_cl = _calc_euclidean(cl.vel, gt.vel, cl.alive, gt.alive)
    err_vel_ol = _calc_euclidean(ol.vel, gt.vel, ol.alive, gt.alive)

    err_4d_cl = _calc_4d_euclidean(cl.pos, cl.vel, gt.pos, gt.vel, W_x, W_y, cl.alive, gt.alive)
    err_4d_ol = _calc_4d_euclidean(ol.pos, ol.vel, gt.pos, gt.vel, W_x, W_y, ol.alive, gt.alive)

    return [
        ("position", "Position Error (Toroidal L2)", err_pos_cl, err_pos_ol),
        ("velocity", "Velocity Error (L2)", err_vel_cl, err_vel_ol),
        ("pos_vel_4d", "Pos+Vel 4D Error (L2)", err_4d_cl, err_4d_ol),
        (
            "attitude",
            "Attitude Error (MAE)",
            _calc_mae(cl.att, gt.att, cl.alive, gt.alive),
            _calc_mae(ol.att, gt.att, ol.alive, gt.alive),
        ),
        (
            "health",
            "Health Error (MAE)",
            _calc_mae(cl.health, gt.health, cl.alive, gt.alive),
            _calc_mae(ol.health, gt.health, ol.alive, gt.alive),
        ),
        (
            "power",
            "Power Error (MAE)",
            _calc_mae(cl.power, gt.power, cl.alive, gt.alive),
            _calc_mae(ol.power, gt.power, ol.alive, gt.alive),
        ),
    ]


def _plot_error_metrics(
    out_dir: str,
    mae_features: list[tuple[str, str, np.ndarray, np.ndarray]],
    plot_N: int,
    num_ships: int,
    steps: np.ndarray,
) -> None:
    """Line charts of each error metric over time, per featured ship."""
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


def _build_feature_series(
    gt: _RolloutArrays, cl: _RolloutArrays, ol: _RolloutArrays
) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    """Per-feature (GT, CL, OL) series driving the feature-divergence line plots."""
    return [
        ("position_x", "Position X", gt.pos_uw[..., 0], cl.pos_uw[..., 0], ol.pos_uw[..., 0]),
        ("velocity_x", "Velocity X", gt.vel[..., 0], cl.vel[..., 0], ol.vel[..., 0]),
        ("angle_cos", "Angle (cos)", gt.att[..., 0], cl.att[..., 0], ol.att[..., 0]),
        ("angular_vel", "Angular Vel", gt.ang_vel[..., 0], cl.ang_vel[..., 0], ol.ang_vel[..., 0]),
        (
            "angular_vel_scaled",
            "Angular Vel (Scaled to GT)",
            gt.ang_vel[..., 0],
            cl.ang_vel[..., 0],
            ol.ang_vel[..., 0],
        ),
        ("health", "Health", gt.health[..., 0], cl.health[..., 0], ol.health[..., 0]),
        ("power", "Power", gt.power[..., 0], cl.power[..., 0], ol.power[..., 0]),
        ("cooldown", "Cooldown", gt.cooldown[..., 0], cl.cooldown[..., 0], ol.cooldown[..., 0]),
        ("alive", "Alive Prob", gt.alive, cl.alive_prob, ol.alive_prob),
    ]


def _plot_feature_divergence(
    out_dir: str,
    features: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]],
    plot_N: int,
    num_ships: int,
    steps: np.ndarray,
) -> None:
    """Line charts overlaying GT/CL/OL for each individual feature."""
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


def _write_markdown_report(
    out_dir: str,
    num_ships: int,
    mae_features: list[tuple[str, str, np.ndarray, np.ndarray]],
    features: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Write ar_report.md linking every generated plot."""
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


def _generate_report(
    history_sim: History,
    history_closed: History,
    history_open: History,
    ship_config: ShipConfig,
    num_steps: int,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    num_ships = history_sim[0]["pos"].shape[1]  # shape is (1, N, 2) — dim 1 is ship count
    plot_N = 2 if num_ships == 2 else 1
    W_x, W_y = ship_config.world_size

    gt = _extract_rollout_arrays(history_sim, plot_N, W_x, W_y)
    cl = _extract_rollout_arrays(history_closed, plot_N, W_x, W_y)
    ol = _extract_rollout_arrays(history_open, plot_N, W_x, W_y)

    # GT alive-prob is constant 1; only the AR rollouts can "revive" a dead ship's prob.
    cl.alive_prob = _clamp_alive_prob(cl.alive_prob, cl.alive, plot_N)
    ol.alive_prob = _clamp_alive_prob(ol.alive_prob, ol.alive, plot_N)

    # Center on the toroidal CoM of the ground-truth featured ships (raw, wrapped coords),
    # anchoring the centered/cropped map to where the actual game happened.
    com_x, com_y = _toroidal_center_of_mass(gt.pos.reshape(-1, 2), W_x, W_y)
    gt.pos_c = _center_pos_uw(gt.pos_uw, com_x, com_y)
    cl.pos_c = _center_pos_uw(cl.pos_uw, com_x, com_y)
    ol.pos_c = _center_pos_uw(ol.pos_uw, com_x, com_y)

    if num_ships > 2:  # 2v2+ only
        _plot_full_world_map(out_dir, gt, cl, ol, num_ships, W_x, W_y)
    _plot_centered_map(out_dir, gt, cl, ol, num_ships, W_x, W_y)
    _plot_velocity_map(out_dir, gt, cl, ol, num_ships)

    steps = np.arange(num_steps)
    mae_features = _compute_error_metrics(gt, cl, ol, W_x, W_y)
    _plot_error_metrics(out_dir, mae_features, plot_N, num_ships, steps)

    features = _build_feature_series(gt, cl, ol)
    _plot_feature_divergence(out_dir, features, plot_N, num_ships, steps)

    _write_markdown_report(out_dir, num_ships, mae_features, features)
