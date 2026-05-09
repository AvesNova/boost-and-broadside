"""noise_calibration mode: measure NextStateHead prediction error statistics.

Runs two phases:
  Phase 1 — 512 envs × 512 steps, collecting per-feature single-step prediction
             errors to estimate sigma (noise std) and rho (lag-1 autocorrelation).
  Phase 2 — 256 envs × 20 windows, running short closed-loop AR rollouts (teacher-
             forced with real sim actions) to measure how RMSE grows with rollout depth.

AUX continuous state layout (16 dims):
  [0:2]  pos_x (sin, cos)     [2:4]  pos_y (sin, cos)
  [4:6]  vel_dir (cos, sin)   [6]    symlog_speed
  [7:9]  att (cos, sin)       [9]    ang_vel (symlog, absolute)
  [10:12] health (sin, cos)   [12:14] power (sin, cos)
  [14:16] cooldown (sin, cos)

Outputs written to docs/noise_calibration/:
  noise_params.json, error_distributions.png, autocorrelation.png,
  ar_growth.png, team_symmetry.png
"""

import datetime
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.modes.agent_factory import (
    ALIVE_HEALTH_EPS,
    ResolvedAgent,
    _apply_aux_deltas_local,
    _extract_aux_cont,
    _imagined_obs_from_aux,
    get_actions,
    init_hidden,
    reset_done_envs,
    resolve_agent_spec,
)
from boost_and_broadside.modes.collect import _obs_from_state

_N_AUX = 16
_AR_WINDOW = 20
_WARMUP_STEPS = 50
_TEAM_SYMMETRY_REL_TOL = 0.15

# Maps logical feature name → AUX dim indices
_FEAT_GROUPS: dict[str, tuple[list[int], str]] = {
    "pos_x":       ([0, 1],    "toroidal phase x"),
    "pos_y":       ([2, 3],    "toroidal phase y"),
    "vel_dir":     ([4, 5],    "velocity direction"),
    "symlog_speed":([6],       "symlog speed"),
    "att":         ([7, 8],    "heading angle"),
    "ang_vel":     ([9],       "angular velocity"),
    "health":      ([10, 11],  "health quarter-wave"),
    "power":       ([12, 13],  "power quarter-wave"),
    "cooldown":    ([14, 15],  "cooldown quarter-wave"),
}

_DIM_NAMES = (
    "pos_sin_x", "pos_cos_x", "pos_sin_y", "pos_cos_y",
    "vel_dir_cos", "vel_dir_sin",
    "symlog_speed",
    "att_cos", "att_sin",
    "ang_vel_abs",
    "health_sin", "health_cos",
    "power_sin", "power_cos",
    "cooldown_sin", "cooldown_cos",
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_noise_calibration_mode(
    team0_spec: str,
    team1_spec: str,
    num_envs: int,
    num_steps: int,
    num_ar_envs: int,
    num_ar_windows: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    model_config: ModelConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "docs/noise_calibration",
) -> None:
    dev = torch.device(device)
    N = env_config.num_ships
    num_tokens = N + env_config.num_obstacles

    print("Resolving agents...")
    agent0 = resolve_agent_spec(team0_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N)
    agent1 = resolve_agent_spec(team1_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N)

    if agent0.kind != "policy":
        raise ValueError(
            f"noise_calibration requires team0 to be a policy checkpoint, got kind={agent0.kind!r}. "
            f"Pass a .pt path via --team0."
        )

    scripted_for_warmup = resolve_agent_spec("scripted", ship_config, model_config, device, num_ships=N)

    print(f"\n{'='*60}")
    print(f"Phase 1: single-step error collection")
    print(f"  {num_envs} envs × {num_steps} steps")
    print(f"{'='*60}")
    phase1 = _run_phase1(agent0, agent1, num_envs, N, num_tokens, num_steps,
                         ship_config, env_config, dev)

    print(f"\n{'='*60}")
    print(f"Phase 2: short closed-loop AR rollouts")
    print(f"  {num_ar_envs} envs × {num_ar_windows} windows × {_AR_WINDOW}-step chains")
    print(f"{'='*60}")
    phase2 = _run_phase2(agent0, scripted_for_warmup, num_ar_envs, N, num_tokens,
                         num_ar_windows, ship_config, env_config, dev)

    print("\nBuilding output...")
    output = _build_output(phase1, phase2, team0_spec, num_envs, num_steps,
                           num_ar_envs, num_ar_windows)
    _write_outputs(output, output_dir)

    _print_summary(output)
    print(f"\nWrote results to {output_dir}/")


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def _run_phase1(
    agent0: ResolvedAgent,
    agent1: ResolvedAgent,
    B: int,
    N: int,
    num_tokens: int,
    num_steps: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    dev: torch.device,
) -> dict:
    env = TensorEnv(B, ship_config, env_config, dev)
    init_hidden(agent0, B, num_tokens, dev)
    init_hidden(agent1, B, num_tokens, dev)
    env.reset()

    err_sum           = torch.zeros(_N_AUX, device=dev)
    err_sq_sum        = torch.zeros(_N_AUX, device=dev)
    err_count         = torch.zeros(1, device=dev)
    lag1_cross_sum    = torch.zeros(_N_AUX, device=dev)
    lag1_sq_sum       = torch.zeros(_N_AUX, device=dev)
    lag1_count        = torch.zeros(1, device=dev)
    team_err_sq_sum   = torch.zeros(2, _N_AUX, device=dev)
    team_count        = torch.zeros(2, device=dev)
    combat_err_sq_sum = torch.zeros(2, _N_AUX, device=dev)
    combat_count      = torch.zeros(2, device=dev)

    prev_err   = torch.zeros(B, N, _N_AUX, device=dev)
    prev_valid = torch.zeros(B, N, dtype=torch.bool, device=dev)

    t0 = time.perf_counter()
    print(f"Collecting {num_steps} steps across {B} envs...")

    for step in range(num_steps):
        obs = _obs_from_state(env.state, ship_config)

        # Capture combat flag before step (prev_action is the action that produced current state)
        combat = (env.state.prev_action[:, :N, 2] > 0.5).any(dim=1)  # (B,)

        action0, pred_next = get_actions(agent0, obs, env.state, B, N, dev, return_pred_next=True)
        action1 = get_actions(agent1, obs, env.state, B, N, dev)

        team_id = env.state.ship_team_id  # (B, N) int32
        action = torch.where((team_id == 0).unsqueeze(-1), action0, action1)

        curr_alive = obs["alive"][:, :N].clone()  # (B, N) bool, before step
        curr_aux = _extract_aux_cont(obs, N, ship_config)  # (B, N, 16)

        dones, truncated = env.step(action)
        done_any = dones | truncated  # (B,)

        next_obs = _obs_from_state(env.state, ship_config)
        next_alive = env.state.ship_alive  # (B, N) bool, after step

        if pred_next is not None:
            pred_aux = _apply_aux_deltas_local(curr_aux, pred_next)          # (B, N, 16)
            true_aux = _extract_aux_cont(next_obs, N, ship_config)           # (B, N, 16)
            err = pred_aux - true_aux                                         # (B, N, 16)

            # valid: alive at both ends, no episode boundary
            episode_end = done_any.unsqueeze(-1)                              # (B, 1)
            valid = curr_alive & next_alive & ~episode_end                    # (B, N)

            if valid.any():
                v_err = err[valid]                                            # (K, 16)
                err_sum    += v_err.sum(0)
                err_sq_sum += v_err.pow(2).sum(0)
                err_count  += valid.sum().float()

                # Lag-1 autocorrelation
                lag_valid = valid & prev_valid                                # (B, N)
                if lag_valid.any():
                    lv_curr = err[lag_valid]                                  # (M, 16)
                    lv_prev = prev_err[lag_valid]                             # (M, 16)
                    lag1_cross_sum += (lv_prev * lv_curr).sum(0)
                    lag1_sq_sum    += lv_prev.pow(2).sum(0)
                    lag1_count     += lag_valid.sum().float()

                # Team-stratified
                for t in range(2):
                    tm = valid & (env.state.ship_team_id == t)
                    if tm.any():
                        te = err[tm]
                        team_err_sq_sum[t] += te.pow(2).sum(0)
                        team_count[t]      += tm.sum().float()

                # Combat-stratified
                c_expand = combat.unsqueeze(-1).expand_as(valid)              # (B, N)
                for c_idx, c_cond in enumerate([~c_expand, c_expand]):
                    m = valid & c_cond
                    if m.any():
                        combat_err_sq_sum[c_idx] += err[m].pow(2).sum(0)
                        combat_count[c_idx]      += m.sum().float()

            # Lag-1 buffer update (valid already masks done envs via ~episode_end)
            prev_valid = valid.clone()
            prev_err   = err.detach().clone()

        if done_any.any():
            env.reset_envs(done_any)
            reset_done_envs(agent0, done_any, num_tokens)
            reset_done_envs(agent1, done_any, num_tokens)

        if (step + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  step {step+1}/{num_steps}  "
                  f"valid samples: {int(err_count.item()):,}  "
                  f"elapsed: {elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    print(f"Phase 1 done in {elapsed:.1f}s — {int(err_count.item()):,} valid (ship, step) pairs.")

    return {
        "err_sum":           err_sum.cpu().numpy(),
        "err_sq_sum":        err_sq_sum.cpu().numpy(),
        "err_count":         float(err_count.cpu().item()),
        "lag1_cross_sum":    lag1_cross_sum.cpu().numpy(),
        "lag1_sq_sum":       lag1_sq_sum.cpu().numpy(),
        "lag1_count":        float(lag1_count.cpu().item()),
        "team_err_sq_sum":   team_err_sq_sum.cpu().numpy(),
        "team_count":        team_count.cpu().numpy(),
        "combat_err_sq_sum": combat_err_sq_sum.cpu().numpy(),
        "combat_count":      combat_count.cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def _run_phase2(
    agent0: ResolvedAgent,
    warmup_agent1: ResolvedAgent,
    B: int,
    N: int,
    num_tokens: int,
    num_windows: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    dev: torch.device,
) -> dict:
    env = TensorEnv(B, ship_config, env_config, dev)
    init_hidden(agent0, B, num_tokens, dev)
    init_hidden(warmup_agent1, B, num_tokens, dev)
    env.reset()

    ar_sq_sum = torch.zeros(_AR_WINDOW, _N_AUX, device=dev)
    ar_count  = torch.zeros(_AR_WINDOW, device=dev)

    t0 = time.perf_counter()

    for window in range(num_windows):
        # --- Warmup ---
        for _ in range(_WARMUP_STEPS):
            obs = _obs_from_state(env.state, ship_config)
            action0 = get_actions(agent0, obs, env.state, B, N, dev)
            action1 = get_actions(warmup_agent1, obs, env.state, B, N, dev)
            team_id = env.state.ship_team_id
            action = torch.where((team_id == 0).unsqueeze(-1), action0, action1)
            dones, truncated = env.step(action)
            done_any = dones | truncated
            if done_any.any():
                env.reset_envs(done_any)
                reset_done_envs(agent0, done_any, num_tokens)
                reset_done_envs(warmup_agent1, done_any, num_tokens)

        # --- Snapshot after warmup ---
        ar_start_obs    = {k: v.clone() for k, v in _obs_from_state(env.state, ship_config).items()}
        ar_start_hidden = agent0.hidden.clone()
        ar_start_aux    = _extract_aux_cont(ar_start_obs, N, ship_config)  # (B, N, 16)

        # --- Real-sim recording ---
        stored_actions  = []  # list of (B, N, 3) int tensors
        stored_true_aux = []  # list of (B, N, 16) float tensors
        stored_alive    = []  # list of (B, N) bool tensors
        window_valid    = torch.ones(B, dtype=torch.bool, device=dev)

        for k in range(_AR_WINDOW):
            obs = _obs_from_state(env.state, ship_config)
            action0 = get_actions(agent0, obs, env.state, B, N, dev)
            action1 = get_actions(warmup_agent1, obs, env.state, B, N, dev)
            team_id = env.state.ship_team_id
            action = torch.where((team_id == 0).unsqueeze(-1), action0, action1)
            stored_actions.append(action.clone())

            dones, truncated = env.step(action)
            done_any = dones | truncated
            window_valid &= ~done_any

            next_obs = _obs_from_state(env.state, ship_config)
            stored_true_aux.append(_extract_aux_cont(next_obs, N, ship_config).clone())
            stored_alive.append(env.state.ship_alive.clone())

            if done_any.any():
                env.reset_envs(done_any)
                reset_done_envs(agent0, done_any, num_tokens)
                reset_done_envs(warmup_agent1, done_any, num_tokens)

        # --- AR replay from snapshot ---
        curr_obs    = {k: v.clone() for k, v in ar_start_obs.items()}
        curr_hidden = ar_start_hidden.clone()
        curr_aux    = ar_start_aux.clone()

        with torch.no_grad():
            for k in range(_AR_WINDOW):
                _, _, _, pred_next, curr_hidden = agent0.agent.get_action_and_value(
                    curr_obs, curr_hidden
                )
                if pred_next is None:
                    break

                next_aux_ar = _apply_aux_deltas_local(curr_aux, pred_next)       # (B, N, 16)
                err_k = (next_aux_ar - stored_true_aux[k]).pow(2)                # (B, N, 16)

                # valid: window not terminated + ship alive in ground truth
                valid_k = window_valid.unsqueeze(-1) & stored_alive[k]           # (B, N)

                if valid_k.any():
                    mask = valid_k.unsqueeze(-1).float()                          # (B, N, 1)
                    ar_sq_sum[k] += (err_k * mask).sum(dim=(0, 1))
                    ar_count[k]  += valid_k.sum().float()

                curr_obs = _imagined_obs_from_aux(
                    next_aux_ar, curr_obs, stored_actions[k], N, ship_config
                )
                curr_aux = next_aux_ar

        elapsed = time.perf_counter() - t0
        if (window + 1) % 5 == 0 or window == 0:
            print(f"  window {window+1}/{num_windows}  elapsed: {elapsed:.1f}s  "
                  f"valid at depth-1: {int(ar_count[0].item()):,}")

    elapsed = time.perf_counter() - t0
    print(f"Phase 2 done in {elapsed:.1f}s.")

    return {
        "ar_sq_sum": ar_sq_sum.cpu().numpy(),   # (20, 16)
        "ar_count":  ar_count.cpu().numpy(),    # (20,)
    }


# ---------------------------------------------------------------------------
# Output building
# ---------------------------------------------------------------------------

def _build_output(
    phase1: dict,
    phase2: dict,
    checkpoint_path: str,
    num_envs: int,
    num_steps: int,
    num_ar_envs: int,
    num_ar_windows: int,
) -> dict:
    n = max(phase1["err_count"], 1.0)
    sigma_per_dim = np.sqrt(phase1["err_sq_sum"] / n)           # (16,)
    bias_per_dim  = phase1["err_sum"] / n                        # (16,)

    lag_denom = phase1["lag1_sq_sum"]
    rho_per_dim = np.where(
        lag_denom > 1e-9,
        phase1["lag1_cross_sum"] / lag_denom,
        0.0,
    ).clip(-1.0, 1.0)                                            # (16,)

    team_sigma = np.sqrt(
        phase1["team_err_sq_sum"] / np.maximum(phase1["team_count"][:, None], 1.0)
    )                                                             # (2, 16)

    combat_sigma = np.sqrt(
        phase1["combat_err_sq_sum"] / np.maximum(phase1["combat_count"][:, None], 1.0)
    )                                                             # (2, 16)

    ar_rmse = np.sqrt(
        phase2["ar_sq_sum"] / np.maximum(phase2["ar_count"][:, None], 1.0)
    )                                                             # (20, 16)

    features_json: dict = {}
    for name, (dims, _desc) in _FEAT_GROUPS.items():
        sig = float(np.mean(sigma_per_dim[dims]))
        s0  = float(np.mean(team_sigma[0, dims]))
        s1  = float(np.mean(team_sigma[1, dims]))
        features_json[name] = {
            "aux_dims":        dims,
            "sigma":           sig,
            "bias":            float(np.mean(bias_per_dim[dims])),
            "rho_lag1":        float(np.mean(rho_per_dim[dims])),
            "sigma_team0":     s0,
            "sigma_team1":     s1,
            "team_symmetry_ok": bool(abs(s0 - s1) / max(sig, 1e-9) < _TEAM_SYMMETRY_REL_TOL),
            "sigma_combat":    float(np.mean(combat_sigma[1, dims])),
            "sigma_noncombat": float(np.mean(combat_sigma[0, dims])),
        }

    ar_growth_json = {
        "depth": list(range(1, _AR_WINDOW + 1)),
        "rmse_per_feature": {
            name: [float(np.mean(ar_rmse[k, dims])) for k in range(_AR_WINDOW)]
            for name, (dims, _) in _FEAT_GROUPS.items()
        },
    }

    recommended_noise = {
        name: {
            "sigma": features_json[name]["sigma"],
            "rho":   features_json[name]["rho_lag1"],
        }
        for name in _FEAT_GROUPS
    }

    return {
        "metadata": {
            "checkpoint":      checkpoint_path,
            "num_envs":        num_envs,
            "num_steps":       num_steps,
            "num_ar_envs":     num_ar_envs,
            "num_ar_windows":  num_ar_windows,
            "ar_window_len":   _AR_WINDOW,
            "warmup_steps":    _WARMUP_STEPS,
            "timestamp":       datetime.datetime.utcnow().isoformat() + "Z",
        },
        "features":          features_json,
        "ar_growth":         ar_growth_json,
        "recommended_noise": recommended_noise,
    }


# ---------------------------------------------------------------------------
# Output writing (plots + JSON)
# ---------------------------------------------------------------------------

def _write_outputs(data: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    with open(os.path.join(output_dir, "noise_params.json"), "w") as f:
        json.dump(data, f, indent=2)

    feats      = data["features"]
    feat_names = list(_FEAT_GROUPS.keys())

    # --- error_distributions.png: sigma bar chart per AUX dim with bias overlay ---
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    fig.suptitle("Per-dim sigma (bar) and bias (line marker)", fontsize=11)

    # Rebuild per-dim sigma and bias from feature groups
    sigma_arr = np.zeros(_N_AUX)
    bias_arr  = np.zeros(_N_AUX)
    for name, (dims, _) in _FEAT_GROUPS.items():
        sigma_arr[dims] = feats[name]["sigma"]
        bias_arr[dims]  = feats[name]["bias"]

    for i, ax in enumerate(axes.flat):
        ax.bar([0], [sigma_arr[i]], color="steelblue", alpha=0.8, label="sigma")
        ax.axhline(bias_arr[i], color="red", linewidth=1.5, linestyle="--", label="bias")
        ax.set_title(_DIM_NAMES[i], fontsize=8)
        ax.set_xticks([])
        ax.set_ylim(0, max(sigma_arr[i] * 1.5, 1e-6))
        if i == 0:
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "error_distributions.png"), dpi=120)
    plt.close(fig)

    # --- autocorrelation.png: rho_lag1 per feature group ---
    rho_vals = [feats[n]["rho_lag1"] for n in feat_names]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(feat_names, rho_vals, color="darkorange", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Lag-1 autocorrelation of prediction error (rho) per feature")
    ax.set_ylabel("rho")
    ax.set_ylim(-1.0, 1.0)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "autocorrelation.png"), dpi=120)
    plt.close(fig)

    # --- ar_growth.png: RMSE vs depth per feature group ---
    depths = data["ar_growth"]["depth"]
    rmse_by_feat = data["ar_growth"]["rmse_per_feature"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in feat_names:
        ax.plot(depths, rmse_by_feat[name], label=name, marker="o", markersize=3)
    ax.set_title("AR rollout RMSE vs depth (closed-loop, teacher-forced)")
    ax.set_xlabel("Rollout depth (steps)")
    ax.set_ylabel("RMSE (AUX space)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ar_growth.png"), dpi=120)
    plt.close(fig)

    # --- team_symmetry.png: sigma_team0 vs sigma_team1 per feature group ---
    s0 = [feats[n]["sigma_team0"] for n in feat_names]
    s1 = [feats[n]["sigma_team1"] for n in feat_names]
    x  = np.arange(len(feat_names))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, s0, w, label="team 0", color="royalblue", alpha=0.85)
    ax.bar(x + w / 2, s1, w, label="team 1", color="tomato",    alpha=0.85)
    ax.set_title("Sigma per team (should be symmetric)")
    ax.set_ylabel("sigma")
    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "team_symmetry.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(data: dict) -> None:
    feats = data["features"]
    rec   = data["recommended_noise"]
    print(f"\n{'='*70}")
    print(f"{'Feature':<14}  {'sigma':>8}  {'bias':>8}  {'rho':>7}  "
          f"{'sym_ok':>6}  {'sigma_cmbt':>10}  {'rec_sigma':>9}  {'rec_rho':>7}")
    print(f"{'-'*70}")
    for name in _FEAT_GROUPS:
        f = feats[name]
        r = rec[name]
        sym = "yes" if f["team_symmetry_ok"] else "NO"
        print(
            f"{name:<14}  {f['sigma']:>8.5f}  {f['bias']:>8.5f}  {f['rho_lag1']:>7.3f}  "
            f"{sym:>6}  {f['sigma_combat']:>10.5f}  {r['sigma']:>9.5f}  {r['rho']:>7.3f}"
        )
    print(f"{'='*70}")
    print("\nRecommended noise_params.json written with sigma and rho per feature.")
