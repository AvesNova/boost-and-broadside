"""``noise-calibration`` mode: measure NextStateHead prediction error statistics.

Runs two phases:
  Phase 1 — 512 envs × 512 steps, collecting per-feature single-step prediction
             errors to estimate sigma (noise std) and rho (lag-1 autocorrelation).
  Phase 2 — 256 envs × 20 windows, running short closed-loop AR rollouts (teacher-
             forced with real sim actions) to measure how RMSE grows with rollout depth.

Report dimensions are the eleven physical next-state channels.

Errors are measured in target space (predicted target vs true target).

The measurement writes one ``noise-calibration`` artifact: ``result.json`` holds
the aggregates every report is built from, and an optional ignored
``samples/phase1_errors.npz`` retains a bounded prefix of the raw single-step
errors so a later estimator can be tried without replaying the environment. The
figures are rendered from the artifact by ``bnb figures``.
"""

import datetime
import math
import time

import numpy as np
import torch

from boost_and_broadside.artifacts import ArtifactRecipe, ArtifactStore
from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.observation import YemongObservation
from boost_and_broadside.evaluation.agents import (
    ResolvedAgent,
    resolve_agent_spec,
)
from boost_and_broadside.evaluation.environment import (
    create_evaluation_env,
    resolve_evaluation_environment,
)
from boost_and_broadside.evaluation.match import MatchRunner
from boost_and_broadside.evaluation.next_state import means_to_observation
from boost_and_broadside.evaluation.subjects import describe_agents, describe_environment
from boost_and_broadside.train.rl.physical_belief import (
    PHYSICAL_MEAN_DIM,
    PHYSICAL_MEAN_NAMES,
    PhysicalNextState,
    physical_means_from_observation,
    physical_means_from_state,
)

_AR_WINDOW = 20
_WARMUP_STEPS = 50
_TEAM_SYMMETRY_REL_TOL = 0.15
_SCHEMA_VERSION = 1

# Raw single-step errors are retained as an ignored local payload so a later
# estimator can be tried without replaying the environment. The cap keeps that
# payload bounded: a full run is hundreds of millions of rows, and the
# aggregates in ``result.json`` already answer everything the report asks.
_MAX_RAW_SAMPLE_ROWS = 262_144


# One report panel per physical channel, in the calibration's order. The error is
# the normalized physical delta's residual, so every panel is in units of that
# channel's Phase-1 scale and the panels are directly comparable.
_REPORT_FEATURES = {
    "position_x": ("pos_x", "position x (px / 2.5)"),
    "position_y": ("pos_y", "position y (px / 2.5)"),
    "velocity_x": ("velocity_x", "velocity x (px/s / 4)"),
    "velocity_y": ("velocity_y", "velocity y (px/s / 4)"),
    "attitude": ("att", "attitude (rad / 0.1)"),
    "angular_velocity": ("ang_vel", "angular velocity (rad/s / 2.5pi)"),
    "shield_delay": ("shield_delay", "shield recharge delay (s / 5)"),
    "health": ("health", "health (/ 10)"),
    "power": ("power", "power (/ 0.75)"),
    "cooldown": ("cooldown", "cooldown (s / 0.1)"),
    "local_log_index": ("local_log_index", "natural log-index (/ 0.05)"),
}


class _RawSampleBuffer:
    """A bounded, in-order prefix of raw error rows and the step each came from.

    Retention is deliberately a prefix rather than a sample: it costs nothing to
    collect, keeps the rows contiguous in time, and is honest about what it is.
    Anything that needs an unbiased sample of the whole run should read the
    aggregates, which cover every row.
    """

    def __init__(self, max_rows: int, num_targets: int) -> None:
        self._max_rows = max_rows
        self._num_targets = num_targets
        self._chunks: list[torch.Tensor] = []
        self._steps: list[np.ndarray] = []
        self._kept = 0

    def add(self, rows: torch.Tensor, step: int) -> None:
        if self._kept >= self._max_rows or not rows.shape[0]:
            return
        take = min(int(rows.shape[0]), self._max_rows - self._kept)
        self._chunks.append(rows[:take].detach().to(torch.float16).cpu())
        self._steps.append(np.full(take, step, dtype=np.int32))
        self._kept += take

    def errors(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros((0, self._num_targets), dtype=np.float16)
        return torch.cat(self._chunks).numpy()

    def steps(self) -> np.ndarray:
        return np.concatenate(self._steps) if self._steps else np.zeros(0, dtype=np.int32)


def _report_layout() -> tuple[dict[str, tuple[list[int], str]], list[str]]:
    """Build report dimension groups from the physical mean layout.

    One dimension per panel now: the next-state model predicts eleven physical
    quantities rather than a Fourier expansion of them, so the harmonic grouping
    this used to do has nothing left to group.
    """

    groups: dict[str, tuple[list[int], str]] = {}
    dim_names = [""] * PHYSICAL_MEAN_DIM
    for channel, name in enumerate(PHYSICAL_MEAN_NAMES):
        if name not in _REPORT_FEATURES:
            # A channel the report does not know about would be measured anyway
            # and published as a nameless empty panel. Fail here instead.
            raise ValueError(
                f"noise report layout names no panel for physical channel {name!r}; "
                "add it to _REPORT_FEATURES"
            )
        report_name, description = _REPORT_FEATURES[name]
        groups[report_name] = ([channel], description)
        dim_names[channel] = report_name
    return groups, dim_names


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
    store: ArtifactStore | None = None,
) -> dict:
    dev = torch.device(device)
    N = env_config.num_ships

    print("Resolving agents...")
    agent0 = resolve_agent_spec(
        team0_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N
    )
    agent1 = resolve_agent_spec(
        team1_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N
    )

    # The policy's own provenance decides the field distribution: a fields policy
    # is measured in a field arena or it is not being measured at all.
    env_config = resolve_evaluation_environment(
        env_config, (agent0, agent1), ship_config=ship_config
    )

    if agent0.kind != "policy":
        raise ValueError(
            f"noise-calibration requires team0 to be a policy checkpoint, "
            f"got kind={agent0.kind!r}. Pass a .pt path via --team0."
        )

    next_state = PhysicalNextState.from_ship_config(ship_config)
    index_log_scale = 2.0 * math.log(ship_config.field_index_step)
    feature_groups, dim_names = _report_layout()
    scripted_for_warmup = resolve_agent_spec(
        "scripted", ship_config, model_config, device, num_ships=N
    )

    print(f"\n{'=' * 60}")
    print("Phase 1: single-step error collection")
    print(f"  {num_envs} envs × {num_steps} steps")
    print(f"{'=' * 60}")
    phase1 = _run_phase1(
        agent0,
        agent1,
        num_envs,
        N,
        num_steps,
        ship_config,
        env_config,
        dev,
        next_state,
        index_log_scale,
    )

    print(f"\n{'=' * 60}")
    print("Phase 2: short closed-loop AR rollouts")
    print(f"  {num_ar_envs} envs × {num_ar_windows} windows × {_AR_WINDOW}-step chains")
    print(f"{'=' * 60}")
    phase2 = _run_phase2(
        agent0,
        scripted_for_warmup,
        num_ar_envs,
        N,
        num_ar_windows,
        ship_config,
        env_config,
        dev,
        next_state,
        index_log_scale,
    )

    print("\nBuilding output...")
    output = _build_output(
        phase1,
        phase2,
        team0_spec,
        num_envs,
        num_steps,
        num_ar_envs,
        num_ar_windows,
        feature_groups,
    )
    output["dim_names"] = dim_names

    store = store or ArtifactStore(checkpoint_root=checkpoint_dir)
    recipe = ArtifactRecipe(
        artifact_type="noise-calibration",
        result_schema_version=_SCHEMA_VERSION,
        subjects=describe_agents(
            checkpoint_root=checkpoint_dir, team0=team0_spec, team1=team1_spec
        ),
        parameters={
            "num_envs": num_envs,
            "num_steps": num_steps,
            "num_ar_envs": num_ar_envs,
            "num_ar_windows": num_ar_windows,
            "ar_window_len": _AR_WINDOW,
            "warmup_steps": _WARMUP_STEPS,
            "environment": describe_environment(env_config, ship_config=ship_config),
        },
    )
    owner = store.owner_for(
        store.owning_run_for_paths(
            [spec for spec in (team0_spec, team1_spec) if spec.endswith(".pt")]
        )
    )
    artifact = store.create(recipe, owner)
    artifact.write_json(output)
    raw_errors = phase1.get("raw_errors")
    if raw_errors is not None and raw_errors.size:
        artifact.write_samples_npz(
            {
                "errors": raw_errors,
                "step": phase1["raw_steps"],
                "dim_names": np.asarray(dim_names),
            },
            "phase1_errors.npz",
        )
    artifact.complete()

    _print_summary(output, feature_groups)
    print(f"\nWrote results to {artifact.path}")
    return output


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------


def _run_phase1(
    agent0: ResolvedAgent,
    agent1: ResolvedAgent,
    B: int,
    N: int,
    num_steps: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    dev: torch.device,
    next_state: PhysicalNextState,
    index_log_scale: float,
) -> dict:
    env = create_evaluation_env(B, ship_config, env_config, dev)
    env.reset()
    runner = MatchRunner(
        env,
        [agent0, agent1],
        torch.zeros(B, dtype=torch.long, device=dev),
        torch.ones(B, dtype=torch.long, device=dev),
        ship_config,
        N,
    )
    runner.init_hidden()

    num_targets = PHYSICAL_MEAN_DIM
    err_sum = torch.zeros(num_targets, device=dev)
    err_sq_sum = torch.zeros(num_targets, device=dev)
    err_count = torch.zeros(1, device=dev)
    lag1_cross_sum = torch.zeros(num_targets, device=dev)
    lag1_sq_sum = torch.zeros(num_targets, device=dev)
    lag1_count = torch.zeros(1, device=dev)
    team_err_sq_sum = torch.zeros(2, num_targets, device=dev)
    team_count = torch.zeros(2, device=dev)
    combat_err_sq_sum = torch.zeros(2, num_targets, device=dev)
    combat_count = torch.zeros(2, device=dev)

    prev_err = torch.zeros(B, N, num_targets, device=dev)
    prev_valid = torch.zeros(B, N, dtype=torch.bool, device=dev)

    raw_samples = _RawSampleBuffer(_MAX_RAW_SAMPLE_ROWS, num_targets)

    t0 = time.perf_counter()
    print(f"Collecting {num_steps} steps across {B} envs...")

    for step in range(num_steps):
        runner.observe()
        selection = runner.select_actions(trace_agents=frozenset({0}))

        # Capture combat flag before step
        combat = (runner.action_state.pending[:, :N, 2] > 0).any(dim=1)  # (B,)
        pred_next_scaled = selection.predictions.get(0)
        model_obs = selection.observations[0]
        curr_alive = env.state.ship_alive.clone()  # (B, N) bool, before step
        curr_means = physical_means_from_observation(model_obs, index_log_scale, num_ships=N)

        dones, truncated = runner.advance(selection.action)
        done_any = dones | truncated  # (B,)

        next_alive = env.state.ship_alive  # (B, N) bool, after step

        if pred_next_scaled is not None:
            pred_means = next_state.apply_means(curr_means, pred_next_scaled.float())
            true_means = physical_means_from_state(env.state)[:, :N]
            # In normalized units, so the eleven channels are comparable and the
            # residual reads against the same scales the objective uses.
            err = next_state.labels(true_means, pred_means)  # (B, N, 11)

            # valid: alive at both ends, no episode boundary
            episode_end = done_any.unsqueeze(-1)  # (B, 1)
            valid = curr_alive & next_alive & ~episode_end  # (B, N)

            if valid.any():
                v_err = err[valid]  # (K, target_dim)
                err_sum += v_err.sum(0)
                err_sq_sum += v_err.pow(2).sum(0)
                err_count += valid.sum().float()
                raw_samples.add(v_err, step)

                # Lag-1 autocorrelation
                lag_valid = valid & prev_valid  # (B, N)
                if lag_valid.any():
                    lv_curr = err[lag_valid]  # (M, target_dim)
                    lv_prev = prev_err[lag_valid]
                    lag1_cross_sum += (lv_prev * lv_curr).sum(0)
                    lag1_sq_sum += lv_prev.pow(2).sum(0)
                    lag1_count += lag_valid.sum().float()

                # Team-stratified
                for t in range(2):
                    tm = valid & (env.state.ship_team_id == t)
                    if tm.any():
                        te = err[tm]
                        team_err_sq_sum[t] += te.pow(2).sum(0)
                        team_count[t] += tm.sum().float()

                # Combat-stratified
                c_expand = combat.unsqueeze(-1).expand_as(valid)  # (B, N)
                for c_idx, c_cond in enumerate([~c_expand, c_expand]):
                    m = valid & c_cond
                    if m.any():
                        combat_err_sq_sum[c_idx] += err[m].pow(2).sum(0)
                        combat_count[c_idx] += m.sum().float()

            prev_valid = valid.clone()
            prev_err = err.detach().clone()

        runner.reset_finished(done_any)

        if (step + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  step {step + 1}/{num_steps}  "
                f"valid samples: {int(err_count.item()):,}  "
                f"elapsed: {elapsed:.1f}s"
            )

    elapsed = time.perf_counter() - t0
    print(f"Phase 1 done in {elapsed:.1f}s — {int(err_count.item()):,} valid (ship, step) pairs.")

    return {
        "err_sum": err_sum.cpu().numpy(),
        "err_sq_sum": err_sq_sum.cpu().numpy(),
        "err_count": float(err_count.cpu().item()),
        "lag1_cross_sum": lag1_cross_sum.cpu().numpy(),
        "lag1_sq_sum": lag1_sq_sum.cpu().numpy(),
        "lag1_count": float(lag1_count.cpu().item()),
        "team_err_sq_sum": team_err_sq_sum.cpu().numpy(),
        "team_count": team_count.cpu().numpy(),
        "combat_err_sq_sum": combat_err_sq_sum.cpu().numpy(),
        "combat_count": combat_count.cpu().numpy(),
        "raw_errors": raw_samples.errors(),
        "raw_steps": raw_samples.steps(),
    }


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------


def _run_phase2(
    agent0: ResolvedAgent,
    warmup_agent1: ResolvedAgent,
    B: int,
    N: int,
    num_windows: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    dev: torch.device,
    next_state: PhysicalNextState,
    index_log_scale: float,
) -> dict:
    env = create_evaluation_env(B, ship_config, env_config, dev)
    env.reset()
    runner = MatchRunner(
        env,
        [agent0, warmup_agent1],
        torch.zeros(B, dtype=torch.long, device=dev),
        torch.ones(B, dtype=torch.long, device=dev),
        ship_config,
        N,
    )
    runner.init_hidden()

    ar_sq_sum = torch.zeros(_AR_WINDOW, PHYSICAL_MEAN_DIM, device=dev)
    ar_count = torch.zeros(_AR_WINDOW, device=dev)

    t0 = time.perf_counter()

    for window in range(num_windows):
        # --- Warmup ---
        for _ in range(_WARMUP_STEPS):
            dones, truncated = runner.step()
            done_any = dones | truncated
            runner.reset_finished(done_any)

        # --- Real-sim recording ---
        stored_actions = []  # list of (B, N, 3) int tensors
        stored_true_means = []  # list of (B, N, 11) float tensors
        stored_alive = []  # list of (B, N) bool tensors
        window_valid = torch.ones(B, dtype=torch.bool, device=dev)

        for k in range(_AR_WINDOW):
            hidden_before = agent0.hidden.clone() if k == 0 else None
            runner.observe()
            selection = runner.select_actions(trace_agents=frozenset({0}))
            if k == 0:
                ar_start_obs = selection.observations[0]
                ar_start_hidden = hidden_before
                ar_start_means = physical_means_from_observation(
                    ar_start_obs, index_log_scale, num_ships=N
                )
            stored_actions.append(selection.action.clone())

            dones, truncated = runner.advance(selection.action)
            done_any = dones | truncated
            window_valid &= ~done_any

            stored_true_means.append(physical_means_from_state(env.state)[:, :N].clone())
            stored_alive.append(env.state.ship_alive.clone())

            runner.reset_finished(done_any)

        # --- AR replay from snapshot ---
        curr_obs = YemongObservation(data={k: v.clone() for k, v in ar_start_obs.items()})
        curr_hidden = ar_start_hidden.clone()
        curr_means = ar_start_means.clone()

        with torch.no_grad():
            for k in range(_AR_WINDOW):
                action_k, _, _, pred_next_scaled, curr_hidden = agent0.agent.get_action_and_value(
                    curr_obs, curr_hidden
                )
                if pred_next_scaled is None:
                    break

                curr_means = next_state.apply_means(curr_means, pred_next_scaled.float())
                err_k = next_state.labels(stored_true_means[k], curr_means).pow(2)

                # valid: window not terminated + ship alive in ground truth
                valid_k = window_valid.unsqueeze(-1) & stored_alive[k]  # (B, N)

                if valid_k.any():
                    mask = valid_k.unsqueeze(-1).float()  # (B, N, 1)
                    ar_sq_sum[k] += (err_k * mask).sum(dim=(0, 1))
                    ar_count[k] += valid_k.sum().float()

                curr_obs = means_to_observation(
                    curr_means,
                    curr_obs,
                    stored_actions[k],
                    N,
                    index_log_scale,
                )

        elapsed = time.perf_counter() - t0
        if (window + 1) % 5 == 0 or window == 0:
            print(
                f"  window {window + 1}/{num_windows}  elapsed: {elapsed:.1f}s  "
                f"valid at depth-1: {int(ar_count[0].item()):,}"
            )

    elapsed = time.perf_counter() - t0
    print(f"Phase 2 done in {elapsed:.1f}s.")

    return {
        "ar_sq_sum": ar_sq_sum.cpu().numpy(),  # (AR_WINDOW, 11)
        "ar_count": ar_count.cpu().numpy(),  # (20,)
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
    feature_groups: dict[str, tuple[list[int], str]],
) -> dict:
    n = max(phase1["err_count"], 1.0)
    sigma_per_dim = np.sqrt(phase1["err_sq_sum"] / n)  # (target_dim,)
    bias_per_dim = phase1["err_sum"] / n  # (target_dim,)

    # Both lag-1 accumulators are summed over the same mask, so a dimension no
    # ship was valid on twice in a row leaves each of them at exactly zero --
    # routine on a short run, and the whole of one two-step smoke case.
    #
    # The guard has to be on the division rather than on its result: np.where
    # picks between two arrays that have both already been evaluated, so
    # guarding there still computes 0/0, still makes a nan, and still warns
    # before discarding it. `where=` skips the divide instead, leaving those
    # dimensions at the zero they were initialised to.
    lag_denom = phase1["lag1_sq_sum"]
    rho_per_dim = np.zeros_like(lag_denom)
    np.divide(
        phase1["lag1_cross_sum"],
        lag_denom,
        out=rho_per_dim,
        where=lag_denom > 1e-9,
    )
    rho_per_dim = rho_per_dim.clip(-1.0, 1.0)  # (target_dim,)

    team_sigma = np.sqrt(
        phase1["team_err_sq_sum"] / np.maximum(phase1["team_count"][:, None], 1.0)
    )  # (2, target_dim)

    combat_sigma = np.sqrt(
        phase1["combat_err_sq_sum"] / np.maximum(phase1["combat_count"][:, None], 1.0)
    )  # (2, target_dim)

    ar_rmse = np.sqrt(
        phase2["ar_sq_sum"] / np.maximum(phase2["ar_count"][:, None], 1.0)
    )  # (AR_WINDOW, target_dim)

    features_json: dict = {}
    for name, (dims, _desc) in feature_groups.items():
        sig = float(np.mean(sigma_per_dim[dims]))
        s0 = float(np.mean(team_sigma[0, dims]))
        s1 = float(np.mean(team_sigma[1, dims]))
        features_json[name] = {
            "aux_dims": dims,
            "sigma": sig,
            "bias": float(np.mean(bias_per_dim[dims])),
            "rho_lag1": float(np.mean(rho_per_dim[dims])),
            "sigma_team0": s0,
            "sigma_team1": s1,
            "team_symmetry_ok": bool(abs(s0 - s1) / max(sig, 1e-9) < _TEAM_SYMMETRY_REL_TOL),
            "sigma_combat": float(np.mean(combat_sigma[1, dims])),
            "sigma_noncombat": float(np.mean(combat_sigma[0, dims])),
        }

    ar_growth_json = {
        "depth": list(range(1, _AR_WINDOW + 1)),
        "rmse_per_feature": {
            name: [float(np.mean(ar_rmse[k, dims])) for k in range(_AR_WINDOW)]
            for name, (dims, _) in feature_groups.items()
        },
    }

    recommended_noise = {
        name: {
            "sigma": features_json[name]["sigma"],
            "rho": features_json[name]["rho_lag1"],
        }
        for name in feature_groups
    }

    return {
        "metadata": {
            "checkpoint": checkpoint_path,
            "num_envs": num_envs,
            "num_steps": num_steps,
            "num_ar_envs": num_ar_envs,
            "num_ar_windows": num_ar_windows,
            "ar_window_len": _AR_WINDOW,
            "warmup_steps": _WARMUP_STEPS,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        },
        "features": features_json,
        "ar_growth": ar_growth_json,
        "recommended_noise": recommended_noise,
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(data: dict, feature_groups: dict[str, tuple[list[int], str]]) -> None:
    feats = data["features"]
    rec = data["recommended_noise"]
    print(f"\n{'=' * 70}")
    print(
        f"{'Feature':<14}  {'sigma':>8}  {'bias':>8}  {'rho':>7}  "
        f"{'sym_ok':>6}  {'sigma_cmbt':>10}  {'rec_sigma':>9}  {'rec_rho':>7}"
    )
    print(f"{'-' * 70}")
    for name in feature_groups:
        f = feats[name]
        r = rec[name]
        sym = "yes" if f["team_symmetry_ok"] else "NO"
        print(
            f"{name:<14}  {f['sigma']:>8.5f}  {f['bias']:>8.5f}  {f['rho_lag1']:>7.3f}  "
            f"{sym:>6}  {f['sigma_combat']:>10.5f}  {r['sigma']:>9.5f}  {r['rho']:>7.3f}"
        )
    print(f"{'=' * 70}")
    print("\nRecommended sigma and rho per feature are recorded in the artifact.")
