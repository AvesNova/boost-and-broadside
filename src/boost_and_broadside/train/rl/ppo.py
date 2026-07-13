"""Clean recurrent PPO trainer for the MVP policy.

Zero Mamba, zero auxiliary losses. One clean loop:
    collect rollout → compute GAE → update epochs → log async → repeat.

Logging is async (CPU-side via wandb) to avoid GPU sync on the hot path.
"""

import dataclasses
import time
import threading
from collections import deque
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import (
    TrainConfig,
    ModelConfig,
    RewardConfig,
    ShipConfig,
    EnvConfig,
    TrainingSchedule,
)
from boost_and_broadside.constants import POWER_SLICE, TURN_SLICE, SHOOT_SLICE
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import MVPObservation, ObsKey
from boost_and_broadside.env.obstacle_cache import ObstacleCache
from boost_and_broadside.env.rewards import REWARD_COMPONENT_NAMES
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.env.wrapper import MVPEnvWrapper
from boost_and_broadside.models.mvp.policy import MVPPolicy
from boost_and_broadside.train.rl.buffer import (
    RolloutBuffer,
    ReturnScaler,
    AdvantageScaler,
    symlog,
)
from boost_and_broadside.train.rl.features import build_standard_coordinator, FeatureCoordinator
from boost_and_broadside.train.rl.sigreg import SIGReg
from boost_and_broadside.train.rl.roster import EloRoster, RosterEntry


def _cast_norms_bf16(module: nn.Module) -> None:
    """Cast all RMSNorm weights to bf16 so the fused kernel can run end-to-end.

    Without this, autocast leaves norm weights in fp32 while activations are bf16,
    forcing a fallback to the slower unfused implementation.  Norm scale vectors are
    tiny (D=128) so bf16 Adam precision is entirely acceptable.  Only applied on
    CUDA — on CPU (e.g. tests) inputs stay fp32 so we leave weights fp32 too.
    """
    for m in module.modules():
        if isinstance(m, nn.RMSNorm) and m.weight.is_cuda:
            m.weight.data = m.weight.data.bfloat16()


# ------------------------------------------------------------------
# Per-component gamma / lambda tensor builder
# ------------------------------------------------------------------


def _build_component_tensor(
    global_val: float,
    overrides: dict[str, float],
    names: tuple[str, ...],
    device: torch.device,
) -> torch.Tensor:
    """Build a (K,) tensor of per-component values.

    Each component uses overrides[name] if present, else global_val.
    """
    return torch.tensor(
        [overrides.get(n, global_val) for n in names],
        dtype=torch.float32,
        device=device,
    )


# ------------------------------------------------------------------
# Opponent-management helpers (module-level, no class coupling)
# ------------------------------------------------------------------


def _slice_obs(obs: MVPObservation, start: int, end: int) -> MVPObservation:
    """Return a view of obs tensors for envs [start, end)."""
    return obs.slice_envs(slice(start, end))


def _slice_state(state: TensorState, start: int, end: int) -> TensorState:
    """Return a new TensorState containing only envs [start, end)."""
    return TensorState(
        step_count=state.step_count[start:end],
        ship_pos=state.ship_pos[start:end],
        ship_vel=state.ship_vel[start:end],
        ship_attitude=state.ship_attitude[start:end],
        ship_ang_vel=state.ship_ang_vel[start:end],
        ship_health=state.ship_health[start:end],
        ship_power=state.ship_power[start:end],
        ship_cooldown=state.ship_cooldown[start:end],
        ship_team_id=state.ship_team_id[start:end],
        ship_alive=state.ship_alive[start:end],
        ship_is_shooting=state.ship_is_shooting[start:end],
        prev_action=state.prev_action[start:end],
        bullet_pos=state.bullet_pos[start:end],
        bullet_vel=state.bullet_vel[start:end],
        bullet_time=state.bullet_time[start:end],
        bullet_active=state.bullet_active[start:end],
        bullet_cursor=state.bullet_cursor[start:end],
        damage_matrix=state.damage_matrix[start:end],
        cumulative_damage_matrix=state.cumulative_damage_matrix[start:end],
        obstacle_pos=state.obstacle_pos[start:end],
        obstacle_vel=state.obstacle_vel[start:end],
        obstacle_radius=state.obstacle_radius[start:end],
        obstacle_gcenter=state.obstacle_gcenter[start:end],
        ship_hit_obstacle=state.ship_hit_obstacle[start:end],
    )


def _flip_team_obs(obs: MVPObservation, N: int) -> MVPObservation:
    """Return a copy of obs with ship team IDs flipped (0↔1).

    Obstacles occupy positions ≥N and always have team_id=2 — they are unchanged.
    """
    return obs.flip_team(N)


# Maps reward component name → the TrainingSchedule group-scale field to apply.
# Effective weight = group_scale * individual_weight (from RewardConfig).
# Groups:
#   true_reward → win components (ally_win, enemy_win)
#   global      → global outcome rewards + shaping (team-aggregated via lambda)
#   local       → self-only per-ship rewards (diagonal lambda, no teammate propagation)
_GROUP: dict[str, str] = {
    "ally_win": "true_reward_scale",
    "enemy_win": "true_reward_scale",
    "ally_damage": "global_scale",
    "enemy_damage": "global_scale",
    "ally_death": "global_scale",
    "enemy_death": "global_scale",
    "facing": "local_scale",
    "closing_speed": "local_scale",
    "shoot_quality": "local_scale",
    "kill_shot": "local_scale",
    "kill_assist": "local_scale",
    "damage_taken": "local_scale",
    "damage_dealt_enemy": "local_scale",
    "damage_dealt_ally": "local_scale",
    "death": "local_scale",
    "obstacle_death": "local_scale",
    "obstacle_proximity": "local_scale",
    "obstacle_closing_speed": "local_scale",
    "obstacle_tti": "local_scale",
    "shooting_penalty": "local_scale",
    "speed": "local_scale",
}

# Components that use diagonal lambda (self-only: i==j). These must match the
# "local_scale" entries above. Any component NOT in _LOCAL_COMPONENTS uses the
# standard team-based lambda aggregation.
_LOCAL_COMPONENTS: frozenset[str] = frozenset(
    {
        "ally_win",
        "enemy_win",
        "facing",
        "closing_speed",
        "shoot_quality",
        "kill_shot",
        "kill_assist",
        "damage_taken",
        "damage_dealt_enemy",
        "damage_dealt_ally",
        "death",
        "obstacle_death",
        "obstacle_proximity",
        "obstacle_closing_speed",
        "obstacle_tti",
        "shooting_penalty",
        "speed",
    }
)


@dataclasses.dataclass
class _ResolvedSchedule:
    """Training schedule evaluated at a single global step.

    Produced by ``_resolve_schedule``; replaces the old ``PhaseConfig`` snapshot.
    All fields are plain values — no callables, no Nones.
    """

    learning_rate: float
    policy_gradient_coef: float
    entropy_coef: float
    behavior_cloning_coef: float
    value_function_coef: float
    sigreg_coef: float
    true_reward_scale: float
    global_scale: float
    local_scale: float
    scripted_fraction: float
    avg_model_fraction: float
    league_fraction: float
    allow_scripted_in_roster: bool
    elo_eval_games: int
    checkpoint_interval: int
    num_epochs: int
    target_kl: float | None


def _resolve_schedule(schedule: TrainingSchedule, step: int) -> _ResolvedSchedule:
    """Evaluate every schedule field at ``step`` and return a resolved snapshot."""
    return _ResolvedSchedule(
        learning_rate=schedule.learning_rate(step),
        policy_gradient_coef=schedule.policy_gradient_coef(step),
        entropy_coef=schedule.entropy_coef(step),
        behavior_cloning_coef=schedule.behavior_cloning_coef(step),
        value_function_coef=schedule.value_function_coef(step),
        sigreg_coef=schedule.sigreg_coef(step),
        true_reward_scale=schedule.true_reward_scale(step),
        global_scale=schedule.global_scale(step),
        local_scale=schedule.local_scale(step),
        scripted_fraction=schedule.scripted_fraction(step),
        avg_model_fraction=schedule.avg_model_fraction(step),
        league_fraction=schedule.league_fraction(step),
        allow_scripted_in_roster=schedule.allow_scripted_in_roster(step),
        elo_eval_games=schedule.elo_eval_games(step),
        checkpoint_interval=schedule.checkpoint_interval(step),
        num_epochs=schedule.num_epochs(step),
        target_kl=schedule.target_kl(step),
    )


def _max_schedule_value(
    schedule_fn: "Callable[[int], float]",
    total_steps: int,
    n_samples: int = 1000,
) -> float:
    """Sample ``schedule_fn`` at ``n_samples`` evenly-spaced steps and return the max.

    Used to pre-allocate env group slots sized for the peak fraction over the run.
    """
    step_size = max(1, total_steps // n_samples)
    return max(schedule_fn(s) for s in range(0, total_steps + step_size, step_size))


def _compute_optimal_eval_ratio(training_elo: float) -> float:
    """Compute optimal ratio (fraction of matches against Random) using Information-Proportional Allocation."""
    # expected win rate against Random (r = 0)
    p_rand = 1.0 / (1.0 + 10.0 ** ((0.0 - training_elo) / 400.0))
    # expected win rate against Scripted (r = 1000)
    p_sc = 1.0 / (1.0 + 10.0 ** ((1000.0 - training_elo) / 400.0))
    
    # Bernoulli variances (Fisher Information contribution)
    v_rand = p_rand * (1.0 - p_rand)
    v_sc = p_sc * (1.0 - p_sc)
    
    # Information-Proportional Allocation: fraction of games to route to Random
    total_val = v_rand + v_sc
    if total_val <= 1e-8:
        return 0.5  # fallback if variance is zero
    return v_rand / total_val


def _compute_optimal_eval_ratio_tensor(elo: torch.Tensor) -> torch.Tensor:
    """Tensor version of _compute_optimal_eval_ratio.

    Same math, but on a device-resident ELO scalar so anchor re-routing inside
    the rollout never forces a host-device sync.
    """
    p_rand = 1.0 / (1.0 + 10.0 ** ((0.0 - elo) / 400.0))
    p_sc = 1.0 / (1.0 + 10.0 ** ((1000.0 - elo) / 400.0))
    v_rand = p_rand * (1.0 - p_rand)
    v_sc = p_sc * (1.0 - p_sc)
    total = v_rand + v_sc
    return torch.where(
        total <= 1e-8, torch.full_like(total, 0.5), v_rand / total.clamp(min=1e-8)
    )




def _clone_to_cpu(obj: Any) -> Any:
    """Recursively copy all tensors to CPU and clone them with non_blocking=True."""
    if isinstance(obj, torch.Tensor):
        return obj.to("cpu", non_blocking=True).clone()
    elif isinstance(obj, dict):
        return {k: _clone_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clone_to_cpu(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_clone_to_cpu(x) for x in obj)
    elif isinstance(obj, (set, frozenset)):
        return type(obj)(_clone_to_cpu(x) for x in obj)
    else:
        return obj


class PPOTrainer:
    """Proximal Policy Optimization for the MVP multi-agent policy.

    Args:
        train_config:    PPO hyperparameters and timeline.
        model_config:    Policy architecture.
        ship_config:     Physics constants.
        device:          Torch device.
        use_wandb:       Whether to log metrics to W&B.
        scripted_agent:  Stochastic scripted agent for BC loss targets and scripted opponents.
    """

    def __init__(
        self,
        train_config: TrainConfig,
        model_config: ModelConfig,
        ship_config: ShipConfig,
        device: str | torch.device,
        use_wandb: bool = False,
        scripted_agent: StochasticScriptedAgent | None = None,
        compile_mode: str | None = "reduce-overhead",
        obs_config=None,  # deprecated, ignored
        resume_wandb_run_id: str | None = None,
    ) -> None:
        self.cfg = train_config
        self.model_config = model_config
        self.ship_config = ship_config
        # Paradigm: "ego_pass" (dual-perspective pass, team 0 trains) vs
        # "shared_pass" (single pass, both teams train). See TrainConfig docstring.
        self._ego_pass = train_config.paradigm == "ego_pass"
        self.coordinator: FeatureCoordinator = build_standard_coordinator(ship_config)
        self.env_config = train_config.scales[0].env_config
        self.device = torch.device(device)
        self._zero_tensor = torch.zeros((), device=self.device)
        self.use_wandb = use_wandb
        self.scripted_agent = scripted_agent

        base_state = _resolve_schedule(train_config.schedule, 0)

        # Primary scale — supports scripted / avg-model / league opponents.
        # Env groups are sized from the MAXIMUM fraction seen across the entire run
        # so that slots exist when a later phase activates a higher fraction.
        # Whether each group is ACTIVE each step is controlled by the current schedule
        # fraction (> 0 → active, == 0 → those envs run self-play silently).
        # Env groups are contiguous slices of the B primary envs:
        #   [0, B_self)                          → pure self-play (+ overflow from inactive groups)
        #   [B_self, B_self+B_sc)               → scripted opponent (+ BC targets)
        #   [B_self+B_sc, B_self+B_sc+B_avg)   → avg-model opponent
        #   [B_self+B_sc+B_avg, B)              → league roster opponent
        B = train_config.scales[0].num_envs
        max_sc_frac = _max_schedule_value(
            train_config.schedule.scripted_fraction, train_config.total_timesteps
        )
        max_avg_frac = _max_schedule_value(
            train_config.schedule.avg_model_fraction, train_config.total_timesteps
        )
        max_league_frac = _max_schedule_value(
            train_config.schedule.league_fraction, train_config.total_timesteps
        )
        self.B_sc = round(max_sc_frac * B)
        self.B_avg = round(max_avg_frac * B)
        self.B_league = round(max_league_frac * B)
        self.B_self = B - self.B_sc - self.B_avg - self.B_league

        if self.B_sc > 0 and scripted_agent is None:
            raise ValueError(
                "scripted_fraction > 0 in schedule requires a scripted_agent to be provided."
            )
        if base_state.policy_gradient_coef == 0.0 and scripted_agent is None:
            raise ValueError(
                "policy_gradient_coef=0.0 (BC mode) requires a scripted_agent."
            )

        # Generate converged obstacle maps before training begins.
        # The cache is shared across all wrappers (primary + aux scales).
        M = train_config.scales[0].env_config.num_obstacles
        if M > 0 and train_config.obstacle_cache is not None:
            cache_cfg = train_config.obstacle_cache
            print(
                f"[PPOTrainer] Generating obstacle cache "
                f"({cache_cfg.num_cache_envs} envs, "
                f"target {cache_cfg.cache_size} maps)..."
            )
            self._obstacle_cache = ObstacleCache.generate(
                ship_config,
                train_config.scales[0].env_config,
                cache_cfg,
                self.device,
            )
            print(f"[PPOTrainer] Obstacle cache ready: {len(self._obstacle_cache)} maps")
        else:
            self._obstacle_cache = None

        self.wrapper = MVPEnvWrapper(
            num_envs=train_config.scales[0].num_envs,
            ship_config=ship_config,
            env_config=train_config.scales[0].env_config,
            rewards=train_config.rewards,
            device=device,
            obstacle_cache=self._obstacle_cache,
        )
        K = self.wrapper.num_active_components
        self._active_names = self.wrapper.active_names  # stable ref used throughout
        # Indices of win/loss components in the active set — these use the TeamPMA
        # value path; all other components use the local (per-ship) path.
        self._win_k: tuple[int, ...] = tuple(
            i for i, n in enumerate(self._active_names) if n in {"ally_win", "enemy_win"}
        )

        # Build per-component (K,) discount tensors — used by all RolloutBuffers.
        self._gamma_t = _build_component_tensor(
            train_config.gamma, train_config.component_gammas, self._active_names, device
        )
        self._lambda_t = _build_component_tensor(
            train_config.gae_lambda, train_config.component_lambdas, self._active_names, device
        )

        N = train_config.scales[0].env_config.num_ships
        self._compile_mode = compile_mode
        self._policy_module = MVPPolicy(
            model_config, self.coordinator, num_value_components=K, num_ships=N,
            team_pma_k=self._win_k,
        ).to(self.device)
        _cast_norms_bf16(self._policy_module)
        self.sigreg = SIGReg(d_model=model_config.d_model, num_proj=64).to(self.device)
        self.policy = (
            torch.compile(self._policy_module, mode=compile_mode)
            if compile_mode is not None
            else self._policy_module
        )
        self.optim = optim.Adam(
            self._policy_module.parameters(), lr=base_state.learning_rate, eps=1e-5
        )

        # Build the buffer using a sample observation to infer shapes and dtypes
        sample_obs = self.wrapper.reset()

        self.buffer = RolloutBuffer(
            num_steps=train_config.num_steps,
            num_envs=train_config.scales[0].num_envs,
            num_ships=N,
            num_components=K,
            obs_sample=sample_obs,
            gamma=self._gamma_t,
            gae_lambda=self._lambda_t,
            device=self.device,
            num_tokens=N + M,
        )

        # Pre-compute lambda masks for active components only.
        # Static for the entire run — derived from RewardConfig.
        self.enemy_neg_k = self._make_enemy_neg_k(
            train_config.rewards.enemy_neg_lambda_components
        )
        self.ally_zero_k = self._make_ally_zero_k(
            train_config.rewards.ally_zero_components
        )
        self.local_k = self._make_local_k()

        self.aux_weights = self.coordinator.get_loss_weights(self.device)

        # Per-component return scaler: EMA of p5/p95 in symlog-reward space (critic)
        self.scaler = ReturnScaler(
            num_components=K,
            device=self.device,
            ema_alpha=train_config.return_ema_alpha,
            min_span=train_config.return_min_span,
        )
        # Per-component advantage scaler: EMA of RMS in symlog-reward space (actor)
        self.adv_scaler = AdvantageScaler(
            num_components=K,
            device=self.device,
        )

        # Per-component aggregated-return diagnostic — refreshed once per update
        # by _precompute_lambda_aggregates (primary scale).
        self._ret_per_comp_mean_k = torch.zeros(K, device=self.device)

        # --- Avg-model opponent (uniform mean of all post-warmup policy snapshots) ---
        # Weights initialized as a copy of the training policy.
        # Accumulation starts once normalized training ELO reaches
        # cfg.avg_model_elo_threshold; once started it never stops.
        self._avg_policy_module = MVPPolicy(
            model_config, self.coordinator, num_value_components=K, num_ships=N,
            team_pma_k=self._win_k,
        ).to(self.device)
        self.avg_policy = (
            torch.compile(self._avg_policy_module, mode=compile_mode)
            if compile_mode is not None
            else self._avg_policy_module
        )
        self._avg_policy_module.load_state_dict(self._policy_module.state_dict())
        _cast_norms_bf16(self._avg_policy_module)
        for p in self._avg_policy_module.parameters():
            p.requires_grad_(False)
        self._avg_param_cumsum: list[torch.Tensor] = [
            torch.zeros(p.shape, dtype=torch.float32, device=p.device)
            for p in self._policy_module.parameters()
        ]
        self._avg_update_count: int = 0

        # Warmup: force torch.compile to trace both policies under autocast so
        # the BF16 norm weights always see BF16 inputs during compilation.
        # Without this, the internal fake-tensor trace runs in fp32 and emits a
        # dtype-mismatch warning (and misses the fused RMSNorm kernel).
        if compile_mode is not None and self.device.type == "cuda":
            _nt = N + M
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _h = self._policy_module.initial_hidden(B, _nt, self.device)
                self.policy.get_action_and_value(sample_obs, _h)
                if self._ego_pass:
                    # Warm up the 2B batch used by the combined team-0/team-1 rollout pass.
                    _obs_t1 = sample_obs.flip_team(N)
                    _obs_2B = sample_obs.concat_batch(_obs_t1)
                    _h_2B = torch.cat([_h, _h], dim=1)
                    self.policy.get_action_and_value(_obs_2B, _h_2B)
                _h_avg = self._avg_policy_module.initial_hidden(B, _nt, self.device)
                self.avg_policy.get_action_and_value(sample_obs, _h_avg)

        # Per-env flag (shared_pass only): which team_id is the opponent in
        # scripted/avg/league groups. In ego_pass opponents always play team 1.
        # Randomised at init and re-randomised each episode reset.
        # Shape: (B_sc + B_avg + B_league,) — indexed relative to the non-self-play slice.
        #   [:B_sc]                → scripted group
        #   [B_sc : B_sc+B_avg]   → avg-model group
        #   [B_sc+B_avg :]         → league group
        n_opp_envs = self.B_sc + self.B_avg + self.B_league
        self._opp_team_flag = (
            torch.randint(0, 2, (n_opp_envs,), device=self.device, dtype=torch.int32)
            if n_opp_envs > 0
            else torch.empty(0, device=self.device, dtype=torch.int32)
        )

        # --- League play + ELO ---
        self.roster = EloRoster(
            max_size=train_config.league_size,
            k_factor=train_config.elo_k_factor,
            elo_temperature=train_config.elo_temperature,
            uniform_sampling=train_config.league_uniform_sampling,
        )
        # Random anchor is added by EloRoster.__init__ (ELO=0, fixed).
        # "avg" entry is added when _update_avg_model() is first called.
        # "scripted" entry is added lazily after scripted_roster_min_steps.

        # Training ELO starts at 0 — all ratings begin
        # at the same point and diverge as eval matchups accumulate.
        self._training_elo: float = 0.0
        self._avg_training_elo: float = 0.0
        self._eval_window_rand = deque(maxlen=100)
        self._eval_window_sc = deque(maxlen=100)
        self._eval_window_avg_vs_sc = deque(maxlen=100)
        self._eval_window_live_vs_avg = deque(maxlen=100)
        self._elo_milestone: float = (
            0.0  # normalized training ELO (vs random) at last milestone
        )
        self._best_training_elo_norm: float = (
            0.0  # best normalized training ELO seen so far
        )
        self._best_avg_elo_norm: float = 0.0  # best normalized avg ELO seen so far
        self._last_checkpoint_path: Path | None = None

        # Current league opponent for the ongoing rollout (rotated each rollout).
        self._current_league_entry: RosterEntry | None = None
        self._current_league_policy: MVPPolicy | None = None

        # Async logging queue
        self._log_queue: Queue = Queue()
        if use_wandb:
            self._init_wandb(train_config, model_config, ship_config, self.env_config, resume_wandb_run_id)
            self._log_thread = threading.Thread(target=self._log_worker, daemon=True)
            self._log_thread.start()

        self._global_step = 0
        self._start_update = 1
        # Cumulative counters persisted across checkpoint resumes so throughput
        # metrics behave as if training never stopped.
        self._ship_steps = 0  # ship tokens (all teams, all envs, all scales)
        self._elapsed_train_time = 0.0  # wall-clock seconds spent training
        self._train_start_time = time.time()  # reset at the top of train()
        total_envs_all = sum(sc.num_envs for sc in train_config.scales)
        self._num_updates = train_config.total_timesteps // (
            total_envs_all * train_config.num_steps
        )

        # Run name used as checkpoint subdirectory (e.g. "checkpoints/good-spaceship-223/")
        if use_wandb:
            import wandb as _wandb

            self._run_name: str = _wandb.run.name
            run_id_path = Path(train_config.checkpoint_dir) / self._run_name / "wandb_run_id.txt"
            run_id_path.parent.mkdir(parents=True, exist_ok=True)
            run_id_path.write_text(_wandb.run.id)
        else:
            from datetime import datetime

            self._run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Schedule state — evaluated from the schedule functions each update.
        # Initialized from step=0 and refreshed after every PPO update.
        self._schedule_state: _ResolvedSchedule = base_state
        self._policy_gradient_coef: float = base_state.policy_gradient_coef
        self._behavior_cloning_coef: float = base_state.behavior_cloning_coef


        # --- Auxiliary training scales (multi-scale curriculum) ---
        # Each scale has its own env + buffer; policy, optimizer, and scaler are shared.
        # Pure self-play only — no scripted/avg/league opponents on aux scales.
        self.aux_wrappers: list[MVPEnvWrapper] = []
        self.aux_buffers: list[RolloutBuffer] = []

        for sc in train_config.scales[1:]:
            aux_w = MVPEnvWrapper(
                num_envs=sc.num_envs,
                ship_config=ship_config,
                env_config=sc.env_config,
                rewards=train_config.rewards,
                device=device,
                obstacle_cache=self._obstacle_cache,
            )
            aux_sample_obs = aux_w.reset()
            aux_buf = RolloutBuffer(
                num_steps=train_config.num_steps,
                num_envs=sc.num_envs,
                num_ships=sc.env_config.num_ships,
                num_components=K,
                obs_sample=aux_sample_obs,
                gamma=self._gamma_t,
                gae_lambda=self._lambda_t,
                device=self.device,
                num_tokens=sc.env_config.num_ships + sc.env_config.num_obstacles,
            )
            self.aux_wrappers.append(aux_w)
            self.aux_buffers.append(aux_buf)

        self._active_save_thread = None
        self._active_best_thread = None

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def _make_enemy_neg_k(self, enemy_neg_set: frozenset[str]) -> torch.Tensor:
        """Build the (K,) bool tensor marking components with lambda=-1 for enemy ships."""
        return torch.tensor(
            [name in enemy_neg_set for name in self._active_names],
            dtype=torch.bool,
            device=self.device,
        )

    def _make_ally_zero_k(self, ally_zero_set: frozenset[str]) -> torch.Tensor:
        """Build the (K,) bool tensor marking components where same-team lambda=0.

        Used for enemy-perspective components (enemy_damage, enemy_death, enemy_win)
        where allies should not contribute their own signal to the aggregated advantage.
        """
        return torch.tensor(
            [name in ally_zero_set for name in self._active_names],
            dtype=torch.bool,
            device=self.device,
        )

    def _make_local_k(self) -> torch.Tensor:
        """Build the (K,) bool tensor marking self-only (local) reward components.

        Local components use a diagonal lambda matrix (lambda_ij = 1 if i==j, else 0)
        so each ship's reward signal never propagates to teammates or enemies.
        """
        return torch.tensor(
            [name in _LOCAL_COMPONENTS for name in self._active_names],
            dtype=torch.bool,
            device=self.device,
        )

    def _update_avg_model(self) -> None:
        """Add the current training policy snapshot to the uniform running average."""
        first_update = self._avg_update_count == 0
        self._avg_update_count += 1
        for cum, p in zip(self._avg_param_cumsum, self._policy_module.parameters()):
            cum.add_(p.detach().float())
        for avg_p, cum in zip(
            self._avg_policy_module.parameters(), self._avg_param_cumsum
        ):
            avg_p.data.copy_(cum / self._avg_update_count)
        # Register the avg model as a roster entry the first time it's ready,
        # seeded at current training ELO (it's a recent snapshot, so it's a
        # reasonable starting estimate that will quickly self-correct via eval).
        if first_update:
            self.roster.add_special(
                "avg", self._global_step, 0, initial_elo=self._training_elo
            )

    def _rollout_policy_pass(
        self,
        obs: MVPObservation,
        hidden: torch.Tensor,
        hidden_t1: torch.Tensor | None,
        num_ships: int,
        num_tokens: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Run the training policy's rollout forward pass(es) for one step.

        ego_pass: one batched 2B pass over both team perspectives. Team 1 ships
        act from the flipped-obs half (action_t1); logprob/value/pred_next are
        stored from the raw-obs half only.
        shared_pass: one B pass on raw obs — every ship acts from it.

        Args:
            obs:        MVPObservation with (B, N+M, ...) tensors (raw team IDs).
            hidden:     (n_layers, B*(N+M), D) raw-perspective hidden state.
            hidden_t1:  Flipped-perspective hidden state; None in shared_pass.
            num_ships:  N — ship token count for team flipping.
            num_tokens: N+M — used to split the 2B hidden state.

        Returns:
            action_t0:  (B, N, 3) raw-perspective actions.
            action_t1:  (B, N, 3) flipped-perspective actions; None in shared_pass.
            logprob:    (B, N) raw-perspective log probs.
            value_norm: (B, N, K) raw-perspective values (normalized space).
            pred_next:  (B, N, pred_dim) raw-perspective next-state predictions.
            hidden:     Updated raw-perspective hidden state.
            hidden_t1:  Updated flipped-perspective hidden state; None in shared_pass.
        """
        if not self._ego_pass:
            action, logprob, value_norm, pred_next, hidden = (
                self.policy.get_action_and_value(obs, hidden)
            )
            return action, None, logprob, value_norm, pred_next, hidden, None

        batch = hidden.shape[1] // num_tokens
        obs_t1 = _flip_team_obs(obs, num_ships)
        obs_both = obs.concat_batch(obs_t1)
        hidden_both = torch.cat([hidden, hidden_t1], dim=1)  # (n_layers, 2B*(N+M), K*D)
        action_both, logprob_both, value_both, pred_next_both, hidden_out = (
            self.policy.get_action_and_value(obs_both, hidden_both)
        )
        return (
            action_both[:batch],                        # (B, N, 3)
            action_both[batch:],                        # (B, N, 3)
            logprob_both[:batch],                       # (B, N)
            value_both[:batch],                         # (B, N, K)
            pred_next_both[:batch],                     # (B, N, pred_dim)
            hidden_out[:, : batch * num_tokens, :],     # (n_layers, B*(N+M), K*D)
            hidden_out[:, batch * num_tokens :, :],     # (n_layers, B*(N+M), K*D)
        )

    def _opponent_obs(self, obs_slice: MVPObservation, num_ships: int) -> MVPObservation:
        """Return the observation perspective policy opponents act from.

        ego_pass: opponents always play team 1 but must see themselves as
        team 0, so ship team IDs are flipped. shared_pass: opponents act on
        raw obs and play whichever team ``_opp_team_flag`` assigns.
        """
        return _flip_team_obs(obs_slice, num_ships) if self._ego_pass else obs_slice

    def _combine_actions(
        self,
        action_t0: torch.Tensor,
        action_t1: torch.Tensor | None,
        team_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge per-perspective policy actions and build the actor-loss mask.

        ego_pass: team 0 ships act from the raw-obs pass and are the only ships
        that train; team 1 ships act from the flipped-obs pass.
        shared_pass: every ship acts from the single raw-obs pass and trains.

        Args:
            action_t0: (B, N, 3) raw-perspective actions.
            action_t1: (B, N, 3) flipped-perspective actions; None in shared_pass.
            team_id:   (B, N) int team assignment per ship.

        Returns:
            action:     (B, N, 3) executed actions — a fresh tensor, safe for
                        in-place opponent overrides.
            actor_mask: (B, N) bool — ships contributing to actor/BC losses.
        """
        if self._ego_pass:
            team0_mask = team_id == 0  # (B, N)
            action = torch.where(team0_mask.unsqueeze(-1), action_t0, action_t1)  # (B, N, 3)
            return action, team0_mask
        return action_t0.clone(), torch.ones_like(team_id, dtype=torch.bool)

    def _apply_opponent_override(
        self,
        action: torch.Tensor,
        actor_mask: torch.Tensor,
        team_id: torch.Tensor,
        start: int,
        end: int,
        opp_action: torch.Tensor,
    ) -> None:
        """Replace opponent-team ship actions in envs [start, end) in-place.

        ego_pass: the opponent always controls team 1 (already excluded from
        actor_mask). shared_pass: the opponent controls the per-episode random
        ``_opp_team_flag`` team, which is removed from the actor-loss mask here.

        Args:
            action:     (B, N, 3) combined action tensor — modified in-place.
            actor_mask: (B, N) bool actor-loss mask — modified in-place.
            team_id:    (B, N) int team assignment per ship.
            start, end: env slice controlled by this opponent group.
            opp_action: (end-start, N, 3) opponent agent's actions.
        """
        if self._ego_pass:
            opp_mask = team_id[start:end] == 1  # (end-start, N)
        else:
            flags = self._opp_team_flag[start - self.B_self : end - self.B_self]
            opp_mask = team_id[start:end] == flags.unsqueeze(1)  # (end-start, N)
        action[start:end] = torch.where(
            opp_mask.unsqueeze(-1), opp_action, action[start:end]
        )
        actor_mask[start:end] &= ~opp_mask

    def train(self) -> None:
        """Run the full PPO training loop."""
        B = self.cfg.scales[0].num_envs
        N = self.wrapper.num_ships
        M = self.env_config.num_obstacles
        num_tokens = N + M  # ships + obstacles; hidden state covers all entity tokens
        
        # -- ELO Evaluation Env & State Initialization (Parallel Vectorized Slots) --
        # Three fixed 512-env matchup slices, ordered so each network agent covers
        # one contiguous span and runs a single sliced forward pass:
        #   [0:512)     live vs anchors  → updates live ELO
        #   [512:1024)  live vs avg      → head-to-head win rate
        #   [1024:1536) avg  vs anchors  → updates avg-model ELO
        # Live policy acts on [0:1024); avg policy acts on [512:1536);
        # anchors (scripted/random) act on [0:512) and [1024:1536).
        S_eval = 512
        B_eval = 3 * S_eval
        sl_live_anchor = slice(0, S_eval)
        sl_live_avg = slice(S_eval, 2 * S_eval)
        sl_avg_anchor = slice(2 * S_eval, 3 * S_eval)
        eval_env = TensorEnv(
            B_eval,
            self.ship_config,
            self.env_config,
            self.device,
            self._obstacle_cache,
        )
        eval_env.reset()
        eval_env.state.step_count.random_(0, self.env_config.max_episode_steps)

        # Load and resolve agents for ELO evaluation.
        # NOTE: the avg agent must use kind="policy" — get_actions dispatches on
        # kind, and unknown kinds fall through to the null (all-zero-action) path.
        from boost_and_broadside.modes.agent_factory import ResolvedAgent, get_actions, init_hidden, reset_done_envs
        from boost_and_broadside.modes.collect import _obs_from_state

        agent_policy = ResolvedAgent("policy", self.policy)
        agent_avg = ResolvedAgent("policy", self.avg_policy)
        agent_sc = ResolvedAgent("scripted", self.scripted_agent) if self.scripted_agent else None
        agent_rand = ResolvedAgent("random", None)

        # Each policy agent only carries hidden state for the envs it acts on.
        init_hidden(agent_policy, 2 * S_eval, num_tokens, self.device)
        init_hidden(agent_avg, 2 * S_eval, num_tokens, self.device)

        # Optimal information-proportional routing variables
        K_eval = 4.0
        f_star = _compute_optimal_eval_ratio(self._training_elo)
        eval_anchor_is_scripted = (torch.rand(B_eval, device=self.device) > f_star)

        # On-GPU ELO scalars — updated branchlessly inside the rollout, synced to
        # the Python-float attributes once per update. Score history accumulates
        # on-GPU and flushes to the CPU win-rate deques at the same point.
        elo_live_gpu = torch.tensor(
            float(self._training_elo), device=self.device, dtype=torch.float64
        )
        elo_avg_gpu = torch.tensor(
            float(self._avg_training_elo), device=self.device, dtype=torch.float64
        )
        eval_win_hist: list[torch.Tensor] = []
        eval_done_hist: list[torch.Tensor] = []
        eval_anchor_hist: list[torch.Tensor] = []



        sc_start = self.B_self
        sc_end = self.B_self + self.B_sc
        avg_start = sc_end
        avg_end = avg_start + self.B_avg
        league_start = avg_end
        league_end = B

        obs = self.wrapper.reset()
        # Stagger initial step counts so envs don't all truncate simultaneously.
        # Uniformly distributed over [0, max_episode_steps) — after the first wave
        # of truncations they naturally desynchronize on their own.
        self.wrapper.env.state.step_count.random_(0, self.env_config.max_episode_steps)
        hidden = self.policy.initial_hidden(B, num_tokens, self.device)
        # Flipped-perspective hidden state — ego_pass only.
        hidden_t1 = (
            self.policy.initial_hidden(B, num_tokens, self.device)
            if self._ego_pass
            else None
        )

        # Avg-model hidden state — lives across the whole training run.
        avg_hidden: torch.Tensor | None = None
        if self.B_avg > 0:
            avg_hidden = self.avg_policy.initial_hidden(self.B_avg, num_tokens, self.device)

        # League hidden state — re-initialised each rollout when the entry changes.
        league_hidden: torch.Tensor | None = None

        # Action buffer: the most-recently-decided combined action, applied by env.step
        # next iteration (1-step delay). Initialized to zero-action (coast/straight/no-shoot).
        # Maintained between rollouts like hidden state.
        action_buffer = torch.zeros(B, N, 3, dtype=torch.int32, device=self.device)

        # Aux-scale obs, hidden states, action buffers, and last-done flags — live across the whole run.
        aux_obs: list[dict[str, torch.Tensor]] = []
        aux_hiddens: list[torch.Tensor] = []
        aux_hidden_t1s: list[torch.Tensor | None] = []
        aux_action_buffers: list[torch.Tensor] = []
        aux_last_dones: list[torch.Tensor] = []
        for sc, aux_w in zip(self.cfg.scales[1:], self.aux_wrappers):
            aux_obs.append(aux_w.reset())
            aux_w.env.state.step_count.random_(0, sc.env_config.max_episode_steps)
            aux_num_tokens = sc.env_config.num_ships + sc.env_config.num_obstacles
            aux_hiddens.append(
                self.policy.initial_hidden(
                    sc.num_envs, aux_num_tokens, self.device
                )
            )
            aux_hidden_t1s.append(
                self.policy.initial_hidden(sc.num_envs, aux_num_tokens, self.device)
                if self._ego_pass
                else None
            )
            aux_action_buffers.append(
                torch.zeros(sc.num_envs, sc.env_config.num_ships, 3, dtype=torch.int32, device=self.device)
            )
            aux_last_dones.append(
                torch.zeros(sc.num_envs, dtype=torch.bool, device=self.device)
            )

        # CUDA streams for overlapping env physics with network forward passes.
        # env_stream: runs wrapper.step (physics + obs extraction)
        # net_stream: runs all policy forward passes
        env_stream = torch.cuda.Stream() if self.device.type == "cuda" else None
        net_stream = torch.cuda.Stream() if self.device.type == "cuda" else None

        self._train_start_time = time.time()
        # Ship tokens (friendly + enemy, all envs, all scales) processed per update.
        ship_tokens_per_update = self.cfg.num_steps * sum(
            sc.num_envs * sc.env_config.num_ships for sc in self.cfg.scales
        )

        for update in range(self._start_update, self._num_updates + 1):
            # Avg-model eval slots only produce meaningful stats once the avg
            # model has been initialized. Constant for the whole rollout — the
            # count only changes in the post-update section below.
            avg_eval_active = self._avg_update_count > 0

            self.buffer.reset()
            self.buffer.store_initial_hidden(hidden)
            for aux_buf, aux_h in zip(self.aux_buffers, aux_hiddens):
                aux_buf.reset()
                aux_buf.store_initial_hidden(aux_h)

            # Sample a league opponent for this rollout (rotated each update).
            # Only runs when the current phase has league_frac > 0 AND slots are allocated.
            # Evict the previous checkpoint's weights before loading the new one.
            league_active = (
                self.B_league > 0 and self._schedule_state.league_fraction > 0.0
            )
            if league_active:
                entry = self.roster.sample(self._training_elo)
                self._current_league_entry = entry
                if entry is None or (
                    entry.kind == "avg" and self._avg_update_count == 0
                ):
                    # No valid opponent yet — league group falls back to self-play this rollout.
                    self._current_league_entry = None
                    self._current_league_policy = None
                elif entry.kind == "checkpoint":
                    self.roster.load_policy(
                        entry,
                        self.model_config,
                        self.coordinator,
                        self.wrapper.num_active_components,
                        self.wrapper.num_ships,
                        self.device,
                        self._compile_mode,
                        team_pma_k=self._win_k,
                    )
                    self._current_league_policy = entry._policy
                    league_hidden = self._current_league_policy.initial_hidden(
                        self.B_league, num_tokens, self.device
                    )
                elif entry.kind == "avg":
                    self._current_league_policy = self.avg_policy
                    league_hidden = self._current_league_policy.initial_hidden(
                        self.B_league, num_tokens, self.device
                    )
                else:  # "scripted" — no policy forward pass needed
                    self._current_league_policy = None
                    league_hidden = None
            else:
                # League inactive this phase — evict any loaded policy and fall back to self-play.
                self.roster.evict_all_checkpoint_policies()
                self._current_league_entry = None
                self._current_league_policy = None

            # ----------------------------------------------------------------
            # Rollout collection  (1-step action delay + parallel env/net streams)
            #
            # Semantics: at step t, obs(t) = {state(t), prev_action(t-1)}.
            # action_buffer holds action(t-1) — applied to the env this step.
            # The policy computes action(t) in parallel; it is stored in action_buffer
            # and injected into obs(t+1).prev_action so the next policy call sees
            # what it just decided.
            # ----------------------------------------------------------------
            for _step in range(self.cfg.num_steps):
                team_id = obs["team_id"][:, :N]  # (B, N) — stable within a step

                # -- Phase 1: scripted computations on CURRENT state (before env stream) --
                # Scripted agent reads env.state directly; must run before stream launch
                # to avoid data hazard with the physics kernels.
                if self._policy_gradient_coef == 0.0:
                    # BC pretraining: scripted BC targets only, no opponent overrides
                    with torch.no_grad():
                        _, expert_probs_step = (
                            self.scripted_agent.get_actions_and_probs(
                                self.wrapper.env.state
                            )
                        )
                    use_avg = False
                    use_league = False
                    use_sc_bc = False
                    use_sc_opponent = False
                    action_scripted = None
                    action_league_scripted = None
                else:
                    use_avg = (
                        self._avg_update_count > 0
                        and self.B_avg > 0
                        and self._schedule_state.avg_model_fraction > 0.0
                    )
                    use_league = (
                        self.B_league > 0
                        and self._current_league_entry is not None
                        and self._schedule_state.league_fraction > 0.0
                    )
                    use_sc_bc = self.B_sc > 0
                    use_sc_opponent = (
                        use_sc_bc and self._schedule_state.scripted_fraction > 0.0
                    )

                    expert_probs_step: torch.Tensor | None = None
                    action_scripted = None
                    action_league_scripted = None

                    # BC targets + scripted opponent actions (both read env.state)
                    if (
                        self._behavior_cloning_coef > 0.0
                        and self.scripted_agent is not None
                    ):
                        with torch.no_grad():
                            action_scripted_all, expert_probs_step = (
                                self.scripted_agent.get_actions_and_probs(
                                    self.wrapper.env.state
                                )
                            )
                        if use_sc_opponent:
                            action_scripted = action_scripted_all[sc_start:sc_end]
                    elif use_sc_bc:
                        with torch.no_grad():
                            state_sc = _slice_state(
                                self.wrapper.env.state, sc_start, sc_end
                            )
                            action_scripted, _ = (
                                self.scripted_agent.get_actions_and_probs(state_sc)
                            )

                    # Scripted league opponent also reads env.state before the env stream
                    if use_league and self._current_league_policy is None:
                        with torch.no_grad():
                            state_league = _slice_state(
                                self.wrapper.env.state, league_start, league_end
                            )
                            action_league_scripted = self.scripted_agent.get_actions(
                                state_league
                            )

                # -- Phase 2: parallel env step + all network forwards --
                # env_stream: apply action_buffer (previous step's decision) to physics
                # net_stream: compute new actions and values for all policies
                if env_stream is not None:
                    env_stream.wait_stream(torch.cuda.current_stream())
                    net_stream.wait_stream(torch.cuda.current_stream())

                    with torch.cuda.stream(env_stream):
                        next_obs, reward, dones, truncated, info = self.wrapper.step(
                            action_buffer
                        )

                    with torch.cuda.stream(net_stream):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            # get_action_and_value is @no_grad, so no gradient concerns.
                            (
                                action_t0,
                                action_t1,
                                logprob,
                                value_norm,
                                pred_next,
                                hidden,
                                hidden_t1,
                            ) = self._rollout_policy_pass(obs, hidden, hidden_t1, N, num_tokens)
                        if use_avg:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                obs_avg = self._opponent_obs(_slice_obs(obs, avg_start, avg_end), N)
                                action_avg, _, _, _, avg_hidden = (
                                    self.avg_policy.get_action_and_value(
                                        obs_avg, avg_hidden
                                    )
                                )
                        if use_league and self._current_league_policy is not None:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                obs_league = self._opponent_obs(
                                    _slice_obs(obs, league_start, league_end), N
                                )
                                action_league_net, _, _, _, league_hidden = (
                                    self._current_league_policy.get_action_and_value(
                                        obs_league, league_hidden
                                    )
                                )
                        else:
                            action_league_net = None

                    torch.cuda.current_stream().wait_stream(env_stream)
                    torch.cuda.current_stream().wait_stream(net_stream)
                else:
                    # CPU fallback (no streams)
                    next_obs, reward, dones, truncated, info = self.wrapper.step(
                        action_buffer
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        action_t0, action_t1, logprob, value_norm, pred_next, hidden, hidden_t1 = (
                            self._rollout_policy_pass(obs, hidden, hidden_t1, N, num_tokens)
                        )
                    if use_avg:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            obs_avg = self._opponent_obs(_slice_obs(obs, avg_start, avg_end), N)
                            action_avg, _, _, _, avg_hidden = (
                                self.avg_policy.get_action_and_value(obs_avg, avg_hidden)
                            )
                    if use_league and self._current_league_policy is not None:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            obs_league = self._opponent_obs(
                                _slice_obs(obs, league_start, league_end), N
                            )
                            action_league_net, _, _, _, league_hidden = (
                                self._current_league_policy.get_action_and_value(
                                    obs_league, league_hidden
                                )
                            )
                    else:
                        action_league_net = None

                # -- Phase 3: combine actions and compute actor mask --
                action, actor_mask = self._combine_actions(action_t0, action_t1, team_id)

                # Override opponent-controlled ships with actual opponent actions.
                if self._policy_gradient_coef != 0.0:
                    action_league = (
                        action_league_scripted
                        if action_league_scripted is not None
                        else action_league_net
                    )
                    if use_sc_opponent:
                        self._apply_opponent_override(
                            action, actor_mask, team_id, sc_start, sc_end, action_scripted
                        )
                    if use_avg:
                        self._apply_opponent_override(
                            action, actor_mask, team_id, avg_start, avg_end, action_avg
                        )
                    if use_league:
                        self._apply_opponent_override(
                            action, actor_mask, team_id, league_start, league_end, action_league
                        )

                # -- Phase 4: inject decided action into next obs as prev_action --
                # obs(t+1).previous_action = action(t) — what the policy just decided,
                # will be executed by env.step next iteration.
                next_obs[ObsKey.PREVIOUS_ACTION][:, :N] = action

                done_any = dones | truncated
                self.buffer.add(
                    obs=obs,
                    action=action,
                    logprob=logprob,
                    reward=reward,
                    done=dones.float(),  # only true termination cuts GAE bootstrap
                    value=self.scaler.denormalize(value_norm),  # symlog-reward space for GAE
                    alive=obs["alive"][:, :N].bool(),
                    actor_mask=actor_mask,
                    expert_probs=expert_probs_step,
                    terminated=done_any,
                )

                # Update action buffer: action(t) will be applied by env.step next step
                action_buffer = action.detach()

                # Reset hidden states and action buffer for terminated envs
                hidden = self.policy.reset_hidden_for_envs(hidden, done_any, num_tokens)
                if self._ego_pass:
                    hidden_t1 = self.policy.reset_hidden_for_envs(
                        hidden_t1, done_any, num_tokens
                    )
                action_buffer = action_buffer.clone()
                action_buffer[done_any] = 0
                if use_avg and self.B_avg > 0:
                    avg_hidden = self.avg_policy.reset_hidden_for_envs(
                        avg_hidden, done_any[avg_start:avg_end], num_tokens
                    )
                if use_league and self._current_league_policy is not None:
                    league_hidden = self._current_league_policy.reset_hidden_for_envs(
                        league_hidden, done_any[league_start:league_end], num_tokens
                    )

                # Re-randomise which team the opponent controls for envs that just
                # ended an episode (shared_pass only; ego_pass opponents are fixed
                # to team 1).
                if not self._ego_pass and self._opp_team_flag.numel() > 0:
                    done_non_self = done_any[self.B_self :]
                    new_flags = torch.randint(
                        0,
                        2,
                        self._opp_team_flag.shape,
                        device=self.device,
                        dtype=torch.int32,
                    )
                    self._opp_team_flag = torch.where(
                        done_non_self, new_flags, self._opp_team_flag
                    )

                obs = next_obs
                self._global_step += B

                # Aux-scale rollout steps (pure self-play, 1-step delay)
                for i, (sc, aux_w, aux_buf) in enumerate(
                    zip(self.cfg.scales[1:], self.aux_wrappers, self.aux_buffers)
                ):
                    aux_N = sc.env_config.num_ships
                    aux_num_tokens = aux_N + sc.env_config.num_obstacles
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        (
                            aux_action_t0,
                            aux_action_t1,
                            aux_logprob,
                            aux_value_norm,
                            _,
                            aux_hiddens[i],
                            aux_hidden_t1s[i],
                        ) = self._rollout_policy_pass(
                            aux_obs[i], aux_hiddens[i], aux_hidden_t1s[i], aux_N, aux_num_tokens
                        )
                    aux_team_id = aux_obs[i]["team_id"][:, :aux_N]  # (B_aux, N_aux)
                    aux_action, aux_actor_mask = self._combine_actions(
                        aux_action_t0, aux_action_t1, aux_team_id
                    )
                    next_aux_obs, aux_reward, aux_dones, aux_truncated, _ = aux_w.step(
                        aux_action_buffers[i]
                    )
                    # Inject aux decided action into next obs previous_action
                    next_aux_obs[ObsKey.PREVIOUS_ACTION][:, :aux_N] = aux_action
                    aux_done_any = aux_dones | aux_truncated
                    aux_buf.add(
                        obs=aux_obs[i],
                        action=aux_action,
                        logprob=aux_logprob,
                        reward=aux_reward,
                        done=aux_dones.float(),
                        value=self.scaler.denormalize(aux_value_norm),
                        alive=aux_obs[i]["alive"][:, :aux_N].bool(),
                        actor_mask=aux_actor_mask,
                        expert_probs=None,
                        terminated=aux_done_any,
                    )
                    aux_hiddens[i] = self.policy.reset_hidden_for_envs(
                        aux_hiddens[i], aux_done_any, aux_num_tokens
                    )
                    if self._ego_pass:
                        aux_hidden_t1s[i] = self.policy.reset_hidden_for_envs(
                            aux_hidden_t1s[i], aux_done_any, aux_num_tokens
                        )
                    aux_action_buffers[i] = aux_action.detach().clone()
                    aux_action_buffers[i][aux_done_any] = 0
                    aux_last_dones[i] = aux_dones
                    aux_obs[i] = next_aux_obs
                    self._global_step += sc.num_envs

                # -- Continuous Live ELO Evaluation (every 4 steps) --
                # Branchless: outcome scoring, ELO updates, anchor re-routing, and
                # env resets are all masked tensor ops on fixed shapes. The only
                # host sync for the whole ELO system happens once per update, when
                # the GPU scalars and score history flush (after the rollout loop).
                if _step % 4 == 0:
                    with torch.no_grad():
                        eval_state = eval_env.state
                        eval_obs_obj = _obs_from_state(eval_state, self.ship_config)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            # Sliced forward passes — each agent covers only the
                            # contiguous env span whose actions it provides.
                            action_live = get_actions(
                                agent_policy,
                                eval_obs_obj.slice_envs(slice(0, 2 * S_eval)),
                                eval_state, 2 * S_eval, N, self.device,
                            ).long()  # envs [0:1024): live-vs-anchor + live-vs-avg
                            action_avg = get_actions(
                                agent_avg,
                                eval_obs_obj.slice_envs(slice(S_eval, 3 * S_eval)),
                                eval_state, 2 * S_eval, N, self.device,
                            ).long()  # envs [512:1536): live-vs-avg + avg-vs-anchor

                            # Anchor actions — rows map to envs [0:512) + [1024:1536).
                            if agent_sc is not None:
                                action_scripted = torch.cat([
                                    get_actions(
                                        agent_sc, None,
                                        _slice_state(eval_state, 0, S_eval),
                                        S_eval, N, self.device,
                                    ),
                                    get_actions(
                                        agent_sc, None,
                                        _slice_state(eval_state, 2 * S_eval, 3 * S_eval),
                                        S_eval, N, self.device,
                                    ),
                                ], dim=0).long()
                            else:
                                action_scripted = torch.zeros(
                                    2 * S_eval, N, 3, dtype=torch.long, device=self.device
                                )
                            action_random = get_actions(
                                agent_rand, None, eval_state, 2 * S_eval, N, self.device
                            ).long()
                            anchor_flags = torch.cat([
                                eval_anchor_is_scripted[sl_live_anchor],
                                eval_anchor_is_scripted[sl_avg_anchor],
                            ])
                            action_anchor = torch.where(
                                anchor_flags.view(-1, 1, 1), action_scripted, action_random
                            )

                            # Assemble the full eval batch from the slices.
                            action_team0 = torch.cat([
                                action_live,               # live: matchups 0 + 2
                                action_avg[S_eval:],       # avg vs anchors
                            ], dim=0)
                            action_team1 = torch.cat([
                                action_anchor[:S_eval],    # anchors vs live
                                action_avg[:S_eval],       # avg vs live
                                action_anchor[S_eval:],    # anchors vs avg
                            ], dim=0)
                            eval_team_id = eval_state.ship_team_id
                            eval_action = torch.where(
                                (eval_team_id == 0).unsqueeze(-1),
                                action_team0,
                                action_team1,
                            )

                            eval_dones, eval_truncated = eval_env.step(eval_action)
                            eval_done_any = eval_dones | eval_truncated

                        # Outcome scoring — masked so unfinished envs contribute zero.
                        eval_alive = eval_env.state.ship_alive
                        eval_team = eval_env.state.ship_team_id
                        eval_team0_alive = (eval_alive & (eval_team == 0)).any(dim=1)
                        eval_team1_alive = (eval_alive & (eval_team == 1)).any(dim=1)
                        eval_team0_won = eval_done_any & eval_team0_alive & ~eval_team1_alive
                        eval_team1_won = eval_done_any & eval_team1_alive & ~eval_team0_alive
                        eval_tied = eval_done_any & ~eval_team0_won & ~eval_team1_won
                        score = eval_team0_won.float() + 0.5 * eval_tied.float()  # (B_eval,)
                        done_f = eval_done_any.float()
                        anchor_elo = eval_anchor_is_scripted.float() * 1000.0

                        # Live ELO vs anchors — every game finishing this step uses
                        # the pre-update rating, matching the old batched update.
                        expected_live = 1.0 / (1.0 + 10.0 ** (
                            (anchor_elo[sl_live_anchor] - elo_live_gpu) / 400.0
                        ))
                        elo_live_gpu = elo_live_gpu + (
                            K_eval
                            * (score[sl_live_anchor] - expected_live)
                            * done_f[sl_live_anchor]
                        ).sum()

                        # Avg-model ELO vs anchors — frozen until the avg model
                        # has been initialized (its slots play meaningless games
                        # with a copy of the initial policy weights until then).
                        if avg_eval_active:
                            expected_avg = 1.0 / (1.0 + 10.0 ** (
                                (anchor_elo[sl_avg_anchor] - elo_avg_gpu) / 400.0
                            ))
                            elo_avg_gpu = elo_avg_gpu + (
                                K_eval
                                * (score[sl_avg_anchor] - expected_avg)
                                * done_f[sl_avg_anchor]
                            ).sum()

                        # Win history for the win-rate windows — flushed to the
                        # CPU deques once per update. Only outright wins count
                        # (ties are not half-wins), so an untrained policy that
                        # merely survives to truncation reports ~0%, not ~50%.
                        # Anchor flags are snapshotted before re-routing so they
                        # match the finished games.
                        eval_win_hist.append(eval_team0_won.float())
                        eval_done_hist.append(eval_done_any)
                        eval_anchor_hist.append(eval_anchor_is_scripted.clone())

                        # Information-Proportional anchor re-routing from the on-GPU
                        # live ELO. Matchup slices stay fixed so the 3-way split is
                        # exact.
                        f_star_t = _compute_optimal_eval_ratio_tensor(elo_live_gpu)
                        eval_anchor_is_scripted = torch.where(
                            eval_done_any,
                            torch.rand(B_eval, device=self.device) > f_star_t,
                            eval_anchor_is_scripted,
                        )

                        eval_env.reset_envs(eval_done_any)
                        # Scripted/random anchors are stateless; only the two policy
                        # agents carry hidden state, each over its own env span.
                        reset_done_envs(agent_policy, eval_done_any[: 2 * S_eval], num_tokens)
                        reset_done_envs(agent_avg, eval_done_any[S_eval:], num_tokens)

            # ----------------------------------------------------------------
            # Flush ELO bookkeeping — the eval system's only host sync, once
            # per update. Must run before _update_epochs / the schedule refresh,
            # which read self._training_elo (target_kl gate, BC decay factor).
            # ----------------------------------------------------------------
            self._training_elo = float(elo_live_gpu.item())
            self._avg_training_elo = float(elo_avg_gpu.item())
            if eval_win_hist:
                wins_h = torch.stack(eval_win_hist).cpu()
                dones_h = torch.stack(eval_done_hist).cpu()
                anchors_h = torch.stack(eval_anchor_hist).cpu()
                eval_win_hist.clear()
                eval_done_hist.clear()
                eval_anchor_hist.clear()
                # Extend the deques game-by-game in eval-step order — identical
                # contents to the old per-step extension.
                for i in range(wins_h.shape[0]):
                    w, d, a = wins_h[i], dones_h[i], anchors_h[i]
                    d_live = d[sl_live_anchor]
                    a_live = a[sl_live_anchor]
                    w_live = w[sl_live_anchor]
                    self._eval_window_sc.extend(w_live[d_live & a_live].tolist())
                    self._eval_window_rand.extend(w_live[d_live & ~a_live].tolist())
                    if avg_eval_active:
                        d_avg = d[sl_avg_anchor]
                        a_avg = a[sl_avg_anchor]
                        self._eval_window_avg_vs_sc.extend(
                            w[sl_avg_anchor][d_avg & a_avg].tolist()
                        )
                        self._eval_window_live_vs_avg.extend(
                            w[sl_live_avg][d[sl_live_avg]].tolist()
                        )

            # Store final obs for T+1 aux loss label computation
            self.buffer.store_final_obs(obs)
            for i, aux_buf in enumerate(self.aux_buffers):
                aux_buf.store_final_obs(aux_obs[i])

            # ----------------------------------------------------------------
            # GAE computation
            # ----------------------------------------------------------------
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, next_value_norm, _, _ = self.policy.get_action_and_value(obs, hidden)
            next_value = self.scaler.denormalize(next_value_norm)  # symlog-reward space
            self.buffer.compute_gae(next_value, dones.float())

            # Aux-scale GAE
            for i, (aux_buf, aux_h) in enumerate(zip(self.aux_buffers, aux_hiddens)):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _, _, next_aux_val_norm, _, _ = self.policy.get_action_and_value(
                        aux_obs[i], aux_h
                    )
                next_aux_val = self.scaler.denormalize(next_aux_val_norm)
                aux_buf.compute_gae(next_aux_val, aux_last_dones[i].float())

            # Update per-component return percentiles and advantage RMS from primary rollout only
            self.scaler.update(self.buffer.returns)
            self.adv_scaler.update(self.buffer.advantages, self.buffer.alive_mask)

            # ----------------------------------------------------------------
            # PPO update epochs
            # ----------------------------------------------------------------
            record_hist = update % 10 == 0
            metrics = self._update_epochs(
                all_buffers=[self.buffer] + self.aux_buffers,
                record_histograms=record_hist,
            )

            # Refresh schedule state — syncs LR, loss coefficients, and reward weights.
            # Runs after the PPO update so changes take effect on the next rollout.
            self._schedule_state = _resolve_schedule(
                self.cfg.schedule, self._global_step
            )
            self._policy_gradient_coef = self._schedule_state.policy_gradient_coef
            # BC weight: P(bc_target_beats_current), remapped to [0, 1].
            # Scale=200 gives a steeper sigmoid; target=950 zeroes out BC before 1000.
            elo_norm = self._training_elo - self._random_elo()
            p_bc_wins = 1.0 / (1.0 + 10.0 ** ((elo_norm - 950.0) / 200.0))
            bc_factor = max(0.0, 2.0 * (p_bc_wins - 0.5))
            self._behavior_cloning_coef = self._schedule_state.behavior_cloning_coef * bc_factor
            self.optim.param_groups[0]["lr"] = self._schedule_state.learning_rate
            for comp in self.wrapper._all_components:
                scale_attr = _GROUP[comp.name]
                raw: float = getattr(self.cfg.rewards, f"{comp.name}_weight")
                setattr(
                    comp,
                    f"{comp.name}_weight",
                    raw * getattr(self._schedule_state, scale_attr),
                )
            self.wrapper.refresh_component_weights()
            metrics["schedule/learning_rate"] = self._schedule_state.learning_rate
            metrics["schedule/policy_gradient_coef"] = (
                self._schedule_state.policy_gradient_coef
            )
            metrics["schedule/behavior_cloning_coef"] = self._behavior_cloning_coef
            metrics["schedule/bc_decay_factor"] = bc_factor
            metrics["schedule/target_kl"] = 0.02 if (self._training_elo - self._random_elo()) >= 900.0 else self._schedule_state.target_kl
            metrics["schedule/true_reward_scale"] = (
                self._schedule_state.true_reward_scale
            )
            metrics["schedule/global_scale"] = self._schedule_state.global_scale
            metrics["schedule/local_scale"] = self._schedule_state.local_scale

            if self._policy_gradient_coef > 0.0:
                # Avg-model accumulation starts once normalized training ELO
                # crosses the barrier; once started it never stops.
                elo_barrier_reached = (
                    self._training_elo - self._random_elo()
                    >= self.cfg.avg_model_elo_threshold
                )
                if self.B_avg > 0 and (
                    self._avg_update_count > 0 or elo_barrier_reached
                ):
                    first_avg_update = self._avg_update_count == 0
                    self._update_avg_model()
                    if first_avg_update:
                        # Seed the avg ELO at the live ELO — the first avg
                        # snapshot is exactly the current policy.
                        elo_avg_gpu = elo_live_gpu.clone()
                        self._avg_training_elo = self._training_elo

            # Scaler stats — one CPU transfer per component group
            p5_cpu = self.scaler._p5.cpu()
            p95_cpu = self.scaler._p95.cpu()
            span_cpu = p95_cpu - p5_cpu
            adv_rms_cpu = self.adv_scaler._rms.cpu()
            for i, name in enumerate(self._active_names):
                metrics[f"scaler/p5/{name}"] = p5_cpu[i].item()
                metrics[f"scaler/p95/{name}"] = p95_cpu[i].item()
                metrics[f"scaler/span/{name}"] = span_cpu[i].item()
                metrics[f"scaler/adv_rms/{name}"] = adv_rms_cpu[i].item()

            # Scaler span minimum — flags components where normalization may be degenerate
            metrics["scaler/span_min"] = span_cpu.min().item()

            # Merge episode stats accumulated on-GPU by the wrapper — one sync per update
            ep_stats = self.wrapper.pop_episode_stats()
            for aux_w in self.aux_wrappers:
                aux_w.pop_episode_stats()  # discarded, but keeps accumulators bounded
            n_eps = ep_stats["episodes"].item()
            if n_eps > 0:
                n_ship_eps = n_eps * self.wrapper.num_ships
                comp_sum = ep_stats["comp_sum"].cpu()
                comp_scaled_sum = ep_stats["comp_scaled_sum"].cpu()
                metrics["episode/reward_mean"] = ep_stats["reward_sum"].item() / n_ship_eps
                metrics["episode/reward_min"] = ep_stats["reward_min"].item()
                metrics["episode/reward_max"] = ep_stats["reward_max"].item()
                metrics["episode/length_mean"] = ep_stats["length_sum"].item() / n_eps
                for i, name in enumerate(self._active_names):
                    metrics[f"episode/reward_{name}"] = comp_sum[i].item() / n_ship_eps
                    metrics[f"episode/scaled_{name}"] = comp_scaled_sum[i].item() / n_ship_eps
                metrics["episode/win_rate"] = ep_stats["wins_sum"].item() / n_ship_eps
                metrics["episode/lifespan_mean"] = ep_stats["lifespan_sum"].item() / n_ship_eps

            self._ship_steps += ship_tokens_per_update
            # Cumulative work / cumulative training time — spans checkpoint
            # resumes, as if the run never stopped.
            elapsed = self._elapsed_train_time + (time.time() - self._train_start_time)
            sps = int(self._global_step / elapsed)
            ship_tps = int(self._ship_steps / elapsed)
            metrics["train/global_step"] = self._global_step
            metrics["train/sps"] = sps
            metrics["train/ship_tokens_per_sec"] = ship_tps

            # ELO evaluation — continuous live statistics from parallel slots.
            # Avg-model metrics only exist once the avg model has been initialized.
            metrics["elo/training"] = self._training_elo
            if self._eval_window_rand:
                metrics["elo/training_vs_random"] = sum(self._eval_window_rand) / len(self._eval_window_rand)
            if self._eval_window_sc:
                metrics["elo/training_vs_scripted"] = sum(self._eval_window_sc) / len(self._eval_window_sc)
            if self._avg_update_count > 0:
                metrics["elo/avg"] = self._avg_training_elo
                if self._eval_window_live_vs_avg:
                    metrics["elo/training_vs_avg"] = sum(self._eval_window_live_vs_avg) / len(self._eval_window_live_vs_avg)
                if self._eval_window_avg_vs_sc:
                    metrics["elo/avg_vs_scripted"] = sum(self._eval_window_avg_vs_sc) / len(self._eval_window_avg_vs_sc)

            # Save overwriting best-model checkpoints when normalized ELO improves.
            random_elo = self._random_elo()
            training_elo_norm = self._training_elo - random_elo
            if training_elo_norm > self._best_training_elo_norm:
                self._best_training_elo_norm = training_elo_norm
                self._save_best_checkpoint("best_training.pt")

            # Overview — redundant copies of the most important global metrics
            for src, dst in [
                ("elo/training",                   "overview/elo"),
                ("elo/training_vs_scripted",        "overview/win_rate_vs_scripted"),
                ("elo/training_vs_random",          "overview/win_rate_vs_random"),
                ("elo/training_vs_avg",             "overview/win_rate_vs_avg"),
                ("loss/total",                      "overview/loss_total"),
                ("loss_proxy/policy_gradient",      "overview/loss_proxy_pg"),
                ("loss_proxy/behavioral_cloning",   "overview/loss_proxy_bc"),
                ("policy/kl",                       "overview/kl"),
                ("policy/clip_fraction",            "overview/clip_fraction"),
                ("episode/win_rate",                "overview/win_rate"),
                ("episode/reward_mean",             "overview/reward_mean"),
                ("train/gradient_norm",             "overview/gradient_norm"),
                ("schedule/behavior_cloning_coef",  "overview/bc_coef"),
            ]:
                if src in metrics:
                    metrics[dst] = metrics[src]
            ev_vals = [v for k, v in metrics.items() if k.startswith("critic/explained_variance/")]
            if ev_vals:
                metrics["overview/explained_variance"] = sum(ev_vals) / len(ev_vals)

            # Single log call per update — all metrics at the same step
            self._enqueue_log(metrics, step=self._global_step)

            if update % self.cfg.log_interval == 0:
                elo_str = f"  elo={self._training_elo:.0f}"
                lifespan_str = (
                    f"  lifespan={metrics['episode/lifespan_mean']:.1f}"
                    if "episode/lifespan_mean" in metrics else ""
                )
                print(
                    f"update={update}/{self._num_updates}  "
                    f"step={self._global_step:,}  "
                    f"sps={sps:,}  "
                    f"ship_tps={ship_tps:,}  "
                    f"loss={metrics.get('loss/total', 0.0):.4f}"
                    f"{elo_str}"
                    f"{lifespan_str}"
                )

            checkpoint_interval: int = self._schedule_state.checkpoint_interval
            if checkpoint_interval > 0 and update % checkpoint_interval == 0:
                self._save_checkpoint(update)
                # Add to roster when normalized training ELO (vs random) crosses the next milestone.
                # Skip during pretraining — ELO is not evaluated and the policy is imitating, not competing.
                training_elo_norm = self._training_elo - self._random_elo()
                if (
                    self._policy_gradient_coef > 0.0
                    and self._last_checkpoint_path is not None
                    and self._last_checkpoint_path.exists()
                    and self.cfg.elo_milestone_gap > 0
                    and training_elo_norm - self._elo_milestone
                    >= self.cfg.elo_milestone_gap
                ):
                    self.roster.add_checkpoint(
                        str(self._last_checkpoint_path),
                        self._global_step,
                        update,
                        initial_elo=self._training_elo,
                    )
                    self._elo_milestone = training_elo_norm
                    self._save_roster_json()

        self._shutdown()

    def _shutdown(self) -> None:
        """Release GPU memory and cleanly terminate background threads/processes.

        Safe to call more than once.
        """
        if getattr(self, "_shutdown_called", False):
            return
        self._shutdown_called = True
        self.roster.evict_all_checkpoint_policies()
        self._current_league_policy = None
        if self.use_wandb:
            self._log_queue.put(None)
            if hasattr(self, "_log_thread"):
                self._log_thread.join(timeout=10)
            import wandb

            wandb.finish()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PPO update inner loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _minibatch_denominators(
        self, chunks: list[tuple], buf: RolloutBuffer, is_primary: bool
    ) -> dict:
        """Minibatch-total loss denominators, summed over the micro-batches.

        Masked-mean loss terms in _compute_minibatch_loss divide by these
        totals instead of micro-batch-local counts, so micro-batch losses sum
        exactly to the unsplit minibatch loss and gradient accumulation is
        equivalent to one large minibatch. All masks are rollout data, so this
        is cheap and policy-independent.
        """
        _z = torch.zeros((), device=self.device)
        alive_sum = _z.clone()
        actor_sum = _z.clone()
        bc_sum = _z.clone()
        ns_sum = _z.clone()
        numel = 0
        need_bc = is_primary and self._behavior_cloning_coef > 0.0
        need_ns = is_primary and (
            self.cfg.next_state_coef > 0.0 or self.cfg.windowed_loss_coef > 0.0
        )
        for chunk in chunks:
            mb_alive, mb_actor_mask = chunk[6], chunk[8]
            mb_expert_probs, mb_terminated = chunk[9], chunk[10]
            alive_sum += mb_alive.sum()
            actor_sum += (mb_actor_mask & mb_alive).sum()
            numel += mb_alive.numel()
            if need_bc:
                bc_valid = mb_expert_probs.sum(-1) > 0
                bc_sum += (bc_valid & mb_actor_mask & mb_alive).sum()
            if need_ns:
                ns_sum += (mb_alive & ~mb_terminated.unsqueeze(-1)).sum()
        return {
            "mask_sum": alive_sum.clamp(min=1.0),
            "actor_sum": actor_sum.clamp(min=1.0),
            "bc_sum": bc_sum.clamp(min=1.0),
            "ns_sum": ns_sum.clamp(min=1.0),
            "numel": float(numel),
            "adv_rms": buf.adv_rms,
        }

    def _compute_minibatch_loss(
        self,
        batch: tuple,
        is_primary: bool,
        denoms: dict,
        frac: float,
    ) -> tuple[torch.Tensor, dict]:
        """Compute PPO loss for one micro-batch. Does NOT call zero_grad / backward / step.

        Loss coefficients are read from ``self._policy_gradient_coef``, ``self._behavior_cloning_coef``,
        and ``self._schedule_state`` (``value_function_coef``, ``entropy_coef``) which are updated
        each update step.  Setting ``policy_gradient_coef=0.0`` in the base schedule activates
        BC pretraining mode (no policy gradient or entropy loss).

        Lambda-aggregated advantages/returns and aux next-state labels arrive
        precomputed in the batch (see _precompute_lambda_aggregates /
        _precompute_ns_labels) — they depend only on rollout data, so they are
        built once per update instead of once per minibatch.

        Masked-mean terms divide by the minibatch-total denominators in
        ``denoms`` rather than micro-batch-local counts, so losses and additive
        diagnostics from a minibatch's micro-batches sum exactly to the unsplit
        minibatch values — gradient accumulation over micro-batches is then
        equivalent to one large minibatch. Batch-statistic terms (sigreg,
        windowed next-state) can't decompose that way and are weighted by
        ``frac`` instead (exact when the minibatch is unsplit, i.e. frac=1).

        Args:
            batch:        One micro-batch tuple from RolloutBuffer.get_minibatch_iterator.
            is_primary:   True for the primary scale — enables BC loss and per-component
                          critic diagnostics. Aux scales skip these to avoid shape mismatches
                          (different N) and because BC targets only exist in the primary env.
            denoms:       Minibatch-total denominators from _minibatch_denominators,
                          plus "adv_rms" (whole-buffer advantage normalizer).
            frac:         This micro-batch's env count / minibatch env count.

        Returns:
            (loss, diag) where diag is a dict of scalar/tensor diagnostics.
            Except for "ratio_max" (combine with max) and the histogram tensors,
            diag entries are additive contributions to the minibatch value.
        """
        cfg = self.cfg
        K = self.buffer.num_components

        (
            mb_obs,
            mb_actions,
            mb_old_logprobs,
            mb_advantages,
            mb_returns,
            mb_values,
            mb_alive,
            mb_hidden,
            mb_actor_mask,
            mb_expert_probs,
            mb_terminated,
            mb_adv_agg,
            mb_ret_agg,
            mb_ns_labels,
        ) = batch

        # mb_obs has T+1 steps; first T for encode/evaluate, last T for next-state aux loss.
        T = mb_alive.shape[0]
        curr_mb_obs = mb_obs.slice_time(0, T)

        need_sigreg = self._schedule_state.sigreg_coef > 0.0
        # evaluate_actions needs the full (T, B, N+M) alive mask so Yemong layers
        # can attend to obstacle tokens; mb_alive is ships-only and used for loss masking.
        alive_mask_full = curr_mb_obs["alive"].bool()  # (T, B_mb, N+M)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logprob, entropy, new_value, policy_logits, z, pred_next = self.policy.evaluate_actions(
                obs=curr_mb_obs,
                actions=mb_actions.long(),
                initial_hidden=mb_hidden,
                alive_mask=alive_mask_full,
                done_mask=mb_terminated,
                return_encoder_output=need_sigreg,
            )

        alive_f = mb_alive.float()  # (T, B_mb, N)
        alive_k = alive_f.unsqueeze(-1)  # (T, B_mb, N, 1)
        mask_sum = denoms["mask_sum"]

        actor_f = (mb_actor_mask & mb_alive).float()  # (T, B_mb, N)
        actor_sum = denoms["actor_sum"]

        # ---- Lambda aggregation (precomputed once per update) --------------
        # See _precompute_lambda_aggregates: the (T, B, N_i, N_j, K) lambda
        # tensor depends only on rollout data + per-update scalers, so it is
        # built and reduced once per update, not per minibatch.
        adv_agg = mb_adv_agg  # (T, B_mb, N)
        ret_agg = mb_ret_agg  # (T, B_mb, N)

        adv_norm = adv_agg / (denoms["adv_rms"].sqrt().clamp(min=0.1) + 1e-8)

        # ---- Policy gradient loss ----------------------------------------
        log_ratio = logprob - mb_old_logprobs
        ratio = log_ratio.exp()
        pg_loss1 = -adv_norm * ratio
        pg_loss2 = -adv_norm * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
        pg_loss = (torch.max(pg_loss1, pg_loss2) * actor_f).sum() / actor_sum

        # ---- Value loss --------------------------------------------------
        target_norm = self.scaler.normalize(mb_returns).detach()  # (T, B_mb, N, K)
        vf_loss_raw = (new_value - target_norm).pow(2)  # (T, B_mb, N, K)
        vf_loss = (vf_loss_raw * alive_k).sum() / (mask_sum * K)

        # ---- Entropy bonus -----------------------------------------------
        ent_loss = -(entropy * actor_f).sum() / actor_sum

        # ---- Behavioral cloning loss (primary scale only) ----------------
        bc_loss = self._zero_tensor
        scripted_entropy = self._zero_tensor
        if is_primary and self._behavior_cloning_coef > 0.0:
            bc_valid = mb_expert_probs.sum(-1) > 0  # (T, B_mb, N)
            bc_f = (bc_valid & mb_actor_mask & mb_alive).float()
            bc_sum = denoms["bc_sum"]
            ce = (
                -(
                    mb_expert_probs[..., POWER_SLICE]
                    * F.log_softmax(policy_logits[..., POWER_SLICE], dim=-1)
                ).sum(-1)
                - (
                    mb_expert_probs[..., TURN_SLICE]
                    * F.log_softmax(policy_logits[..., TURN_SLICE], dim=-1)
                ).sum(-1)
                - (
                    mb_expert_probs[..., SHOOT_SLICE]
                    * F.log_softmax(policy_logits[..., SHOOT_SLICE], dim=-1)
                ).sum(-1)
            )  # (T, B_mb, N)
            bc_loss = (ce * bc_f).sum() / bc_sum
            # Entropy of the scripted agent's distribution (the BC loss floor).
            # KL(scripted || policy) = CE - H(scripted); 0 = perfect imitation.
            with torch.no_grad():
                p = mb_expert_probs.clamp(min=1e-8)
                scripted_ent_per_token = (
                    -(p[..., POWER_SLICE] * p[..., POWER_SLICE].log()).sum(-1)
                    - (p[..., TURN_SLICE] * p[..., TURN_SLICE].log()).sum(-1)
                    - (p[..., SHOOT_SLICE] * p[..., SHOOT_SLICE].log()).sum(-1)
                )  # (T, B_mb, N)
                scripted_entropy = (scripted_ent_per_token * bc_f).sum() / bc_sum

        # ---- SIGReg encoder regularization ----------------------------------
        # Batch-statistic term: not decomposable over micro-batches, so weight
        # by env fraction (a per-chunk estimate of the minibatch value).
        sigreg_loss = self._zero_tensor
        if need_sigreg:
            T_mb, B_mb, N_mb, D_mb = z.shape
            z_flat = z.reshape(T_mb, B_mb * N_mb, D_mb)  # (T, B*N, D)
            sigreg_loss = self.sigreg(z_flat) * frac

        # ---- Next-state prediction loss (primary scale only) ----------------
        next_state_loss = self._zero_tensor
        next_state_cont_loss = self._zero_tensor
        windowed_ns_loss = self._zero_tensor
        next_state_per_feat: torch.Tensor | None = None  # (pred_dim,) gpu, for logging
        _need_aux = is_primary and (
            self.cfg.next_state_coef > 0.0 or self.cfg.windowed_loss_coef > 0.0
        )
        if _need_aux:
            non_terminal = ~mb_terminated.unsqueeze(-1)  # (T, B_mb, 1)
            ns_mask = mb_alive & non_terminal             # (T, B_mb, N)
            ns_mask_f = ns_mask.float()
            ns_sum = denoms["ns_sum"]

            # Labels precomputed once per update from the T+1 obs storage
            # (see _precompute_ns_labels) — they depend only on rollout data.
            labels = mb_ns_labels  # (T, B_mb, N, pred_dim)

            P = self.coordinator.total_prediction_dimension
            sq_err = (pred_next.float() - labels.detach()).pow(2)  # (T, B, N, pred_dim)
            sq_err = sq_err * self.aux_weights  # per-prediction weight

            if self.cfg.next_state_coef > 0.0:
                next_state_cont_loss = (
                    sq_err * ns_mask_f.unsqueeze(-1)
                ).sum() / (ns_sum * P)
                next_state_loss = next_state_cont_loss

            if self.cfg.windowed_loss_coef > 0.0:
                # Internally a masked mean over its own validity mask — weight
                # by env fraction like sigreg (exact when the minibatch is unsplit).
                windowed_ns_loss = self.coordinator.compute_windowed_loss(
                    pred_next.float(),
                    labels.detach(),
                    ns_mask,
                    mb_terminated,
                ) * frac

            with torch.no_grad():
                next_state_per_feat = (
                    sq_err * ns_mask_f.unsqueeze(-1)
                ).sum((0, 1, 2)) / ns_sum  # (pred_dim,) gpu, additive across chunks

        loss = (
            self._policy_gradient_coef * pg_loss
            + self._schedule_state.value_function_coef * vf_loss
            + self._schedule_state.entropy_coef * ent_loss
            + self._behavior_cloning_coef * bc_loss
            + self._schedule_state.sigreg_coef * sigreg_loss
            + self.cfg.next_state_coef * next_state_loss
            + self.cfg.windowed_loss_coef * windowed_ns_loss
        )

        # ---- Diagnostics (no grad) — kept as GPU tensors, .item() deferred to logging ----
        diag: dict = {}
        with torch.no_grad():
            diag["loss"] = loss.detach()
            diag["pg_loss"] = pg_loss.detach()
            diag["vf_loss"] = vf_loss.detach()
            diag["ent_loss"] = ent_loss.detach()
            diag["bc_loss"] = bc_loss.detach()
            diag["sigreg_loss"] = sigreg_loss.detach()
            diag["next_state_loss"] = next_state_loss.detach()
            diag["next_state_cont_loss"] = next_state_cont_loss.detach()
            diag["windowed_ns_loss"] = windowed_ns_loss.detach()
            diag["next_state_per_feat"] = next_state_per_feat  # (pred_dim,) gpu or None
            diag["scripted_entropy"] = scripted_entropy.detach()
            diag["bc_kl"] = bc_loss.detach() - scripted_entropy.detach()
            diag["approx_kl"] = (((ratio - 1) - log_ratio) * actor_f).sum() / actor_sum
            diag["clip_frac"] = (
                ((ratio - 1).abs() > cfg.clip_coef).float() * actor_f
            ).sum() / actor_sum
            diag["alive_frac"] = alive_f.sum() / denoms["numel"]
            diag["ratio_mean"] = (ratio * actor_f).sum() / actor_sum
            diag["ratio_max"] = ratio.max()  # combine across chunks with max, not sum

            # Per-head entropy — recomputed from policy_logits (already returned by evaluate_actions)
            power_ent = Categorical(logits=policy_logits[..., POWER_SLICE]).entropy()
            turn_ent = Categorical(logits=policy_logits[..., TURN_SLICE]).entropy()
            shoot_ent = Categorical(logits=policy_logits[..., SHOOT_SLICE]).entropy()
            diag["entropy_power"] = (power_ent * actor_f).sum() / actor_sum
            diag["entropy_turn"] = (turn_ent * actor_f).sum() / actor_sum
            diag["entropy_shoot"] = (shoot_ent * actor_f).sum() / actor_sum

            # First/second moments over minibatch-total actor count — variances
            # are finalized (E[x²] − E[x]²) at the scale level in _update_epochs
            # so they stay exact under micro-batch accumulation.
            diag["ret_agg_mean"] = (ret_agg * actor_f).sum() / actor_sum
            diag["ret_agg_sq"] = (ret_agg.pow(2) * actor_f).sum() / actor_sum

            # Per-component critic stats — primary scale only (K matches buffer.num_components)
            # All additive GPU tensors; ev/std finalization and the single CPU
            # transfer happen once per minibatch in _update_epochs.
            if is_primary:
                pred_k = self.scaler.denormalize(new_value.detach())  # (T, B_mb, N, K)
                residuals_k = mb_returns - pred_k  # (T, B_mb, N, K)
                diag["value_loss_k"] = (vf_loss_raw.detach() * alive_k).sum(
                    (0, 1, 2)
                ) / mask_sum  # (K,)
                diag["ret_mean_k"] = (mb_returns * alive_k).sum((0, 1, 2)) / mask_sum
                diag["ret_sq_k"] = (mb_returns.pow(2) * alive_k).sum((0, 1, 2)) / mask_sum
                diag["res_mean_k"] = (residuals_k * alive_k).sum((0, 1, 2)) / mask_sum
                diag["res_sq_k"] = (residuals_k.pow(2) * alive_k).sum((0, 1, 2)) / mask_sum
                diag["pred_mean_k"] = (pred_k * alive_k).sum((0, 1, 2)) / mask_sum
                # Per-component advantage second moment — raw, unweighted, un-aggregated
                diag["adv_sq_k"] = (mb_advantages.pow(2) * alive_k).sum((0, 1, 2)) / mask_sum
                diag["alive_flat"] = mb_alive.reshape(-1).bool()
                diag["mb_returns"] = mb_returns
                diag["logprob_flat"] = logprob.detach().float().reshape(-1)

        return loss, diag

    @torch.no_grad()
    def _precompute_lambda_aggregates(
        self, buf: RolloutBuffer, comp_weights: torch.Tensor, is_primary: bool
    ) -> None:
        """Fill buf.adv_agg / buf.ret_agg with lambda-aggregated advantages/returns.

        The (T, B, N_i, N_j, K) lambda tensor depends only on rollout data and
        the per-update scalers/weights — not on the policy — so it is built once
        per update here instead of once per minibatch inside the epoch loop.
        Work is chunked over envs to keep peak memory at the per-minibatch level.

        Lambda semantics (unchanged from the previous in-loss computation):
        allies share signals, enemies are zero-sum (enemy_neg_k), enemy-only
        components zero the ally contribution (ally_zero_k), local components
        use a diagonal lambda, dead contributing ships are zeroed, and each
        ship's weights are normalized to a weighted mean over alive ships.

        Also fills buf.adv_rms — the actor-masked mean squared aggregated
        advantage over the whole buffer. Computing it globally (not per
        minibatch) makes the advantage normalization independent of the
        minibatch/micro-batch split.

        For the primary buffer this also computes the per-component
        aggregated-return diagnostic mean (self._ret_per_comp_mean_k).
        """
        T = buf.num_steps
        B = buf.num_envs
        N = buf.num_ships

        ally_lam = torch.where(self.ally_zero_k, 0.0, 1.0)    # (K,)
        enemy_lam = torch.where(self.enemy_neg_k, -1.0, 0.0)  # (K,)
        identity = torch.eye(N, dtype=torch.float32, device=self.device)
        local_lambda = identity[None, None, :, :, None]        # (1, 1, N, N, 1)

        adv_sq_sum = torch.zeros((), device=self.device)
        adv_cnt = torch.zeros((), device=self.device)
        if is_primary:
            ret_pc_sum = torch.zeros(buf.num_components, device=self.device)
            actor_sum = torch.zeros((), device=self.device)

        chunk = max(1, B // self.cfg.num_minibatches)
        if self.cfg.microbatch_tokens is not None:
            chunk = min(
                chunk, max(1, self.cfg.microbatch_tokens // (T * buf.num_tokens))
            )
        for start in range(0, B, chunk):
            sl = slice(start, start + chunk)
            alive = buf.alive_mask[:, sl]                             # (T, b, N)
            team_id_t = buf.obs[ObsKey.TEAM_ID][:T, sl, :N].long()    # (T, b, N)
            same_team_t = team_id_t.unsqueeze(3) == team_id_t.unsqueeze(2)  # (T, b, N, N)
            alive_j = alive.float().unsqueeze(2).unsqueeze(-1)        # (T, b, 1, N_j, 1)

            global_lambda = (
                same_team_t.float().unsqueeze(-1) * ally_lam
                + (~same_team_t).float().unsqueeze(-1) * enemy_lam
            )  # (T, b, N_i, N_j, K)
            lambda_ij_t = (
                torch.where(self.local_k, local_lambda, global_lambda)
                * comp_weights
                * alive_j
            )
            lambda_norm = lambda_ij_t.abs().sum(dim=3, keepdim=True).clamp(min=1.0)
            lambda_ij_t = lambda_ij_t / lambda_norm

            adv_normed = self.adv_scaler.normalize(buf.advantages[:, sl])
            buf.adv_agg[:, sl] = torch.einsum(
                "tbijk,tbjk->tbi", lambda_ij_t, adv_normed
            )
            buf.ret_agg[:, sl] = torch.einsum(
                "tbijk,tbjk->tbi", lambda_ij_t, buf.returns[:, sl]
            )

            actor_f = (buf.actor_masks[:, sl] & alive).float()
            adv_sq_sum += (buf.adv_agg[:, sl].pow(2) * actor_f).sum()
            adv_cnt += actor_f.sum()

            if is_primary:
                ret_pc = torch.einsum(
                    "tbijk,tbjk->tbik", lambda_ij_t, buf.returns[:, sl]
                )  # (T, b, N, K)
                ret_pc_sum += (ret_pc * actor_f.unsqueeze(-1)).sum((0, 1, 2))
                actor_sum += actor_f.sum()

        buf.adv_rms = adv_sq_sum / adv_cnt.clamp(min=1.0)
        if is_primary:
            self._ret_per_comp_mean_k = ret_pc_sum / actor_sum.clamp(min=1.0)

    @torch.no_grad()
    def _precompute_ns_labels(self, buf: RolloutBuffer) -> None:
        """Compute next-state prediction labels once per update.

        Labels come from the stored T+1 observations only — not the policy — so
        computing them here saves num_epochs × num_minibatches redundant passes
        through the coordinator. Targets are computed once over all T+1 steps
        and diffed (labels[t] = f(target[t], target[t+1])).
        """
        if self.cfg.next_state_coef <= 0.0 and self.cfg.windowed_loss_coef <= 0.0:
            buf.ns_labels = None
            return
        T, B, N = buf.num_steps, buf.num_envs, buf.num_ships
        ship_obs = MVPObservation(data={
            k: (v[:, :, :N].reshape((T + 1) * B, N, *v.shape[3:])
                if v.dim() > 3
                else v[:, :, :N].reshape((T + 1) * B, N))
            for k, v in buf.obs.items()
        })
        targets = self.coordinator.get_target_vector(ship_obs)   # ((T+1)*B, N, t_dim)
        targets = targets.reshape(T + 1, B, N, -1)
        buf.ns_labels = self.coordinator.compute_labels(
            targets[:T], targets[1:]
        )  # (T, B, N, pred_dim)

    def _update_epochs(
        self,
        all_buffers: list[RolloutBuffer],
        record_histograms: bool = False,
    ) -> dict:
        """Run num_epochs × num_minibatches of PPO updates across all scales.

        Gradients from every scale are accumulated before each optimizer step so
        that each parameter update reflects all game sizes simultaneously. When
        cfg.microbatch_tokens is set, each scale's minibatch is further split
        into micro-batches whose gradients are accumulated within the same step
        (normalized so the update matches the unsplit minibatch exactly) —
        a memory-only knob for fitting the backward pass on smaller GPUs.

        Args:
            all_buffers:       Primary buffer first, then aux buffers in order.
            record_histograms: If True, capture return/logprob distributions from
                the last primary-scale minibatch for async histogram logging.

        Returns:
            Dict of mean metric values over all minibatch updates.
        """
        cfg = self.cfg
        K = self.buffer.num_components
        n_scales = len(all_buffers)

        comp_weights = torch.tensor(
            [c.weight for c in self.wrapper._active_components],
            dtype=torch.float32,
            device=self.device,
        )  # (K,)

        # Precompute everything that depends only on rollout data (not the
        # policy) once per update instead of once per minibatch: the lambda
        # aggregation and the aux next-state labels (primary scale only).
        for scale_idx, buf in enumerate(all_buffers):
            self._precompute_lambda_aggregates(
                buf, comp_weights, is_primary=(scale_idx == 0)
            )
            if scale_idx > 0:
                buf.ns_labels = None  # aux scales never use the aux losses
        self._precompute_ns_labels(all_buffers[0])

        accum_scalar: dict[str, list[torch.Tensor]] = {
            "loss/total": [],
            "loss/policy_gradient": [],
            "loss/value": [],
            "loss/entropy": [],
            "loss/behavioral_cloning": [],
            "loss/behavioral_cloning_kl": [],
            "loss/scripted_entropy": [],
            "loss/sigreg": [],
            "loss/next_state": [],
            "loss/next_state_cont": [],
            "loss/windowed_ns": [],
            "loss_proxy/policy_gradient": [],
            "loss_proxy/value": [],
            "loss_proxy/entropy": [],
            "loss_proxy/behavioral_cloning": [],
            "loss_proxy/sigreg": [],
            "loss_proxy/next_state": [],
            "policy/kl": [],
            "policy/clip_fraction": [],
            "policy/ratio_mean": [],
            "policy/ratio_max": [],
            "policy/entropy_power": [],
            "policy/entropy_turn": [],
            "policy/entropy_shoot": [],
            "returns/aggregate": [],
            "returns/aggregate_std": [],
            "returns/advantage_std": [],
            "episode/alive_fraction": [],
            "train/gradient_norm": [],
        }
        accum_k: dict[str, list[torch.Tensor]] = {
            "critic/value_loss": [],
            "critic/explained_variance": [],
            "critic/return_mean": [],
            "critic/value_pred_mean": [],
            "returns/component": [],
            "returns/advantage_std": [],
        }
        _NS_FEAT_NAMES = (
            "pos_x_dphase",
            "pos_y_dphase",
            "vel_dvx_norm", "vel_dvy_norm",
            "att_dphase",
            "ang_vel_abs",
            "health_dphase",
            "power_dphase",
            "cooldown_dphase",
        )  # 9 total — matches coordinator.total_prediction_dimension
        ns_per_feat_accum: list[torch.Tensor] = []
        last_returns_np = None
        last_logprob_np = None

        num_epochs = self._schedule_state.num_epochs
        target_kl = 0.02 if (self._training_elo - self._random_elo()) >= 900.0 else self._schedule_state.target_kl

        for epoch_idx in range(num_epochs):
            kl_start = len(accum_scalar["policy/kl"])
            iters = [
                buf.get_minibatch_iterator(cfg.num_minibatches, cfg.microbatch_tokens)
                for buf in all_buffers
            ]
            for batches in zip(*iters):
                self.optim.zero_grad()

                # Accumulate gradients across all scales — and each scale's
                # micro-batches when cfg.microbatch_tokens splits minibatches —
                # before stepping. Each loss is divided by n_scales so the total
                # gradient magnitude stays comparable to single-scale training;
                # micro-batch losses already sum to the exact minibatch loss via
                # the shared minibatch-total denominators.
                _z = torch.zeros((), device=self.device)
                # (accumulator key, diag key) for diagnostics that are additive
                # across micro-batches and averaged across scales.
                _additive = (
                    ("loss", "loss"),
                    ("pg", "pg_loss"),
                    ("vf", "vf_loss"),
                    ("ent", "ent_loss"),
                    ("bc", "bc_loss"),
                    ("sigreg", "sigreg_loss"),
                    ("ns_loss", "next_state_loss"),
                    ("ns_cont", "next_state_cont_loss"),
                    ("windowed_ns", "windowed_ns_loss"),
                    ("bc_kl", "bc_kl"),
                    ("scripted_entropy", "scripted_entropy"),
                    ("kl", "approx_kl"),
                    ("clip", "clip_frac"),
                    ("alive_frac", "alive_frac"),
                    ("ratio_mean", "ratio_mean"),
                    ("entropy_power", "entropy_power"),
                    ("entropy_turn", "entropy_turn"),
                    ("entropy_shoot", "entropy_shoot"),
                )
                _primary_k = (
                    "value_loss_k",
                    "ret_mean_k",
                    "ret_sq_k",
                    "res_mean_k",
                    "res_sq_k",
                    "pred_mean_k",
                    "adv_sq_k",
                )
                scalar_accum_step: dict[str, torch.Tensor] = {
                    key: _z.clone() for key, _ in _additive
                }
                for key in ("adv_var", "ret_agg_mean", "ret_agg_std", "ratio_max"):
                    scalar_accum_step[key] = _z.clone()

                k_stats: dict[str, torch.Tensor] = {}  # primary per-K moments
                ns_feat_step: torch.Tensor | None = None
                hist_diag: dict = {}

                for scale_idx, (buf, chunks) in enumerate(zip(all_buffers, batches)):
                    is_primary = scale_idx == 0
                    denoms = self._minibatch_denominators(chunks, buf, is_primary)
                    mb_envs = sum(c[6].shape[1] for c in chunks)  # c[6] = mb_alive
                    ratio_max = _z.clone()
                    ret_agg_mean = _z.clone()
                    ret_agg_sq = _z.clone()

                    for chunk in chunks:
                        frac = chunk[6].shape[1] / mb_envs
                        loss, diag = self._compute_minibatch_loss(
                            chunk, is_primary, denoms, frac
                        )
                        (loss / n_scales).backward()

                        for key, dkey in _additive:
                            scalar_accum_step[key] += diag[dkey] / n_scales
                        ratio_max = torch.maximum(ratio_max, diag["ratio_max"])
                        ret_agg_mean += diag["ret_agg_mean"]
                        ret_agg_sq += diag["ret_agg_sq"]

                        if is_primary:
                            for kk in _primary_k:
                                k_stats[kk] = (
                                    diag[kk]
                                    if kk not in k_stats
                                    else k_stats[kk] + diag[kk]
                                )
                            if diag.get("next_state_per_feat") is not None:
                                ns_feat_step = (
                                    diag["next_state_per_feat"]
                                    if ns_feat_step is None
                                    else ns_feat_step + diag["next_state_per_feat"]
                                )
                            hist_diag = diag

                    # Non-additive stats finalized per scale: max for the ratio,
                    # E[x²] − E[x]² for the aggregated-return variance.
                    scalar_accum_step["ratio_max"] += ratio_max / n_scales
                    scalar_accum_step["ret_agg_mean"] += ret_agg_mean / n_scales
                    ret_agg_var = (ret_agg_sq - ret_agg_mean.pow(2)).clamp(min=0.0)
                    scalar_accum_step["ret_agg_std"] += ret_agg_var.sqrt() / n_scales
                    scalar_accum_step["adv_var"] += buf.adv_rms / n_scales

                grad_norm = nn.utils.clip_grad_norm_(
                    self._policy_module.parameters(), cfg.max_grad_norm
                )
                self.optim.step()

                accum_scalar["loss/total"].append(scalar_accum_step["loss"])
                accum_scalar["loss/policy_gradient"].append(scalar_accum_step["pg"])
                accum_scalar["loss/value"].append(scalar_accum_step["vf"])
                accum_scalar["loss/entropy"].append(scalar_accum_step["ent"])
                accum_scalar["loss/behavioral_cloning"].append(scalar_accum_step["bc"])
                accum_scalar["loss/behavioral_cloning_kl"].append(scalar_accum_step["bc_kl"])
                accum_scalar["loss/scripted_entropy"].append(
                    scalar_accum_step["scripted_entropy"]
                )
                accum_scalar["loss/sigreg"].append(scalar_accum_step["sigreg"])
                accum_scalar["loss/next_state"].append(scalar_accum_step["ns_loss"])
                accum_scalar["loss/next_state_cont"].append(scalar_accum_step["ns_cont"])
                accum_scalar["loss/windowed_ns"].append(scalar_accum_step["windowed_ns"])
                accum_scalar["loss_proxy/policy_gradient"].append(
                    self._policy_gradient_coef * scalar_accum_step["pg"]
                )
                accum_scalar["loss_proxy/value"].append(
                    self._schedule_state.value_function_coef * scalar_accum_step["vf"]
                )
                accum_scalar["loss_proxy/entropy"].append(
                    self._schedule_state.entropy_coef * scalar_accum_step["ent"]
                )
                accum_scalar["loss_proxy/behavioral_cloning"].append(
                    self._behavior_cloning_coef * scalar_accum_step["bc"]
                )
                accum_scalar["loss_proxy/sigreg"].append(
                    self._schedule_state.sigreg_coef * scalar_accum_step["sigreg"]
                )
                accum_scalar["loss_proxy/next_state"].append(
                    self.cfg.next_state_coef * scalar_accum_step["ns_loss"]
                )
                accum_scalar["policy/kl"].append(scalar_accum_step["kl"])
                accum_scalar["policy/clip_fraction"].append(scalar_accum_step["clip"])
                accum_scalar["policy/ratio_mean"].append(scalar_accum_step["ratio_mean"])
                accum_scalar["policy/ratio_max"].append(scalar_accum_step["ratio_max"])
                accum_scalar["policy/entropy_power"].append(
                    scalar_accum_step["entropy_power"]
                )
                accum_scalar["policy/entropy_turn"].append(
                    scalar_accum_step["entropy_turn"]
                )
                accum_scalar["policy/entropy_shoot"].append(
                    scalar_accum_step["entropy_shoot"]
                )
                accum_scalar["returns/aggregate"].append(
                    scalar_accum_step["ret_agg_mean"]
                )
                accum_scalar["returns/aggregate_std"].append(
                    scalar_accum_step["ret_agg_std"]
                )
                accum_scalar["returns/advantage_std"].append(
                    scalar_accum_step["adv_var"] ** 0.5
                )
                accum_scalar["episode/alive_fraction"].append(
                    scalar_accum_step["alive_frac"]
                )
                accum_scalar["train/gradient_norm"].append(grad_norm.detach())

                if k_stats:
                    # Finalize variance-based stats from the accumulated moments,
                    # then one GPU→CPU transfer per minibatch: stack → (6, K) → cpu
                    ret_var_k = k_stats["ret_sq_k"] - k_stats["ret_mean_k"].pow(2)
                    res_var_k = k_stats["res_sq_k"] - k_stats["res_mean_k"].pow(2)
                    ev_k = 1.0 - res_var_k / (ret_var_k + 1e-8)  # (K,)
                    stats_k_cpu = torch.stack(
                        [
                            k_stats["value_loss_k"],
                            ev_k,
                            k_stats["ret_mean_k"],
                            self._ret_per_comp_mean_k,
                            k_stats["pred_mean_k"],
                            k_stats["adv_sq_k"],
                        ]
                    ).cpu()
                    accum_k["critic/value_loss"].append(stats_k_cpu[0])
                    accum_k["critic/return_mean"].append(stats_k_cpu[2])
                    accum_k["returns/component"].append(stats_k_cpu[3])
                    accum_k["critic/value_pred_mean"].append(stats_k_cpu[4])
                    accum_k["returns/advantage_std"].append(
                        stats_k_cpu[5].clamp(min=0.0).sqrt()
                    )
                    if epoch_idx == num_epochs - 1:
                        accum_k["critic/explained_variance"].append(stats_k_cpu[1])

                if ns_feat_step is not None:
                    ns_per_feat_accum.append(ns_feat_step.cpu())

                if record_histograms and "alive_flat" in hist_diag:
                    # Sampled from the last micro-batch of the last primary
                    # minibatch — a large-enough sample for the histograms.
                    alive_flat = hist_diag["alive_flat"]
                    last_returns_np = (
                        hist_diag["mb_returns"]
                        .reshape(-1, K)[alive_flat]
                        .cpu()
                        .numpy()
                    )
                    last_logprob_np = (
                        hist_diag["logprob_flat"][alive_flat].cpu().numpy()
                    )

            if target_kl is not None:
                epoch_kls = accum_scalar["policy/kl"][kl_start:]
                if epoch_kls and torch.stack(epoch_kls).mean().item() > target_kl:
                    break

        metrics: dict = {
            k: torch.stack(v).mean().item() for k, v in accum_scalar.items() if v
        }
        metrics["train/epochs_completed"] = float(epoch_idx + 1)

        for key, tensors in accum_k.items():
            if not tensors:
                continue
            avg = torch.stack(tensors).mean(0)  # (K,) CPU
            prefix = "returns" if key == "returns/component" else key
            for i, name in enumerate(self._active_names):
                metrics[f"{prefix}/{name}"] = avg[i].item()

        if ns_per_feat_accum:
            avg_per_feat = torch.stack(ns_per_feat_accum).mean(0)  # (16,) cpu
            for i, name in enumerate(_NS_FEAT_NAMES):
                metrics[f"next_state/{name}"] = avg_per_feat[i].item()

        if last_returns_np is not None:
            metrics["hist/returns"] = last_returns_np
            metrics["hist/logprob"] = last_logprob_np

        return metrics

    # ------------------------------------------------------------------
    # ELO evaluation
    # ------------------------------------------------------------------

    def _random_elo(self) -> float:
        """Return the current ELO of the random anchor roster entry."""
        for e in self.roster.entries:
            if e.kind == "random":
                return e.elo
        return 0.0  # fallback; random entry should always exist


    def _save_roster_json(self) -> None:
        """Persist roster metadata alongside the run's checkpoints."""
        ckpt_dir = Path(self.cfg.checkpoint_dir) / self._run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.roster.save_json(ckpt_dir / "roster.json")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint_payload(self, update: int) -> dict:
        """Build the data dict shared by all checkpoint saves."""
        return {
            "policy_state_dict": self._policy_module.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "adv_scaler_state_dict": self.adv_scaler.state_dict(),
            "avg_policy_state_dict": self._avg_policy_module.state_dict(),
            "avg_param_cumsum": [c.to("cpu", non_blocking=True) for c in self._avg_param_cumsum],
            "avg_update_count": self._avg_update_count,
            "update": update,
            "global_step": self._global_step,
            "ship_steps": self._ship_steps,
            "elapsed_train_time": self._elapsed_train_time
            + (time.time() - self._train_start_time),
            "training_elo": self._training_elo,
            "avg_training_elo": self._avg_training_elo,
            "eval_window_rand": list(self._eval_window_rand),
            "eval_window_sc": list(self._eval_window_sc),
            "eval_window_avg_vs_sc": list(self._eval_window_avg_vs_sc),
            "eval_window_live_vs_avg": list(self._eval_window_live_vs_avg),
            "elo_milestone": self._elo_milestone,
            "train_config": {
                k: v for k, v in dataclasses.asdict(self.cfg).items() if k != "schedule"
            },
            "model_config": dataclasses.asdict(self.model_config),
            "env_config": dataclasses.asdict(self.env_config),
        }

    def _save_checkpoint(self, update: int) -> None:
        """Save policy and optimizer state to a .pt file asynchronously.

        Written to cfg.checkpoint_dir/checkpoint_{update:06d}.pt.
        Directory is created if it does not exist.

        Args:
            update: Current update index (used as filename suffix).
        """
        # Check if the previous standard saving thread is still running
        if hasattr(self, "_active_save_thread") and self._active_save_thread is not None and self._active_save_thread.is_alive():
            print("[PPOTrainer] Warning: Previous standard checkpoint saving is still in progress. Skipping this save to prevent disk/GIL congestion.")
            return

        ckpt_dir = Path(self.cfg.checkpoint_dir) / self._run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{self._global_step:012d}.pt"
        
        # Build and copy checkpoints to CPU synchronously on the main thread (very fast, ~5-10ms)
        cpu_payload = _clone_to_cpu(self._checkpoint_payload(update))
        
        avg_path = None
        avg_cpu_payload = None
        if self._avg_update_count > 0:
            avg_path = ckpt_dir / "recent_avg.pt"
            avg_cpu_payload = _clone_to_cpu(self._avg_checkpoint_payload(update))
            
        self._last_checkpoint_path = path

        def _async_save():
            # Write to a temp file then rename atomically so .exists() only
            # returns True once the file is complete (avoids partial-read crashes).
            tmp = path.with_suffix(".tmp")
            torch.save(cpu_payload, tmp)
            tmp.replace(path)
            print(f"Checkpoint saved asynchronously: {path}")

            if avg_cpu_payload is not None and avg_path is not None:
                tmp_avg = avg_path.with_suffix(".tmp")
                torch.save(avg_cpu_payload, tmp_avg)
                tmp_avg.replace(avg_path)
                print(f"Recent avg checkpoint saved asynchronously: {avg_path}")

            # Prune: keep only the latest checkpoint + all roster-referenced files.
            # best_*.pt files are not touched (they don't match the step_*.pt glob).
            kept = self.roster.kept_paths()
            kept.add(str(path))
            for old_path in ckpt_dir.glob("step_*.pt"):
                if str(old_path) not in kept:
                    old_path.unlink(missing_ok=True)

        self._active_save_thread = threading.Thread(target=_async_save, daemon=True)
        self._active_save_thread.start()

    def _avg_checkpoint_payload(self, update: int) -> dict:
        """Build checkpoint payload with avg_policy as the primary policy_state_dict.

        Allows best_avg.pt / recent_avg.pt to be loaded by _load_checkpoint_agent
        in elo_stats.py, which reads ``ckpt["policy_state_dict"]``.
        """
        payload = self._checkpoint_payload(update)
        payload["policy_state_dict"] = self._avg_policy_module.state_dict()
        return payload

    def _checkpoint_payload_lightweight(self, update: int) -> dict:
        """Build a lightweight data dict for best-model saves, omitting heavy optimizer and avg states."""
        return {
            "policy_state_dict": self._policy_module.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "adv_scaler_state_dict": self.adv_scaler.state_dict(),
            "update": update,
            "global_step": self._global_step,
            "training_elo": self._training_elo,
            "eval_window_rand": list(self._eval_window_rand),
            "eval_window_sc": list(self._eval_window_sc),
            "elo_milestone": self._elo_milestone,
            "team_pma_k": self._win_k,
            "train_config": {
                k: v for k, v in dataclasses.asdict(self.cfg).items() if k != "schedule"
            },
            "model_config": dataclasses.asdict(self.model_config),
            "env_config": dataclasses.asdict(self.env_config),
        }

    def _save_best_checkpoint(self, name: str, payload: dict | None = None) -> None:
        """Save a named best-model checkpoint asynchronously, overwriting any previous version.

        Args:
            name:    Filename, e.g. "best_training.pt" or "best_avg.pt".
            payload: Custom payload dict; defaults to _checkpoint_payload_lightweight(update=0).
        """
        # Check if the previous best saving thread is still running
        if hasattr(self, "_active_best_thread") and self._active_best_thread is not None and self._active_best_thread.is_alive():
            print(f"[PPOTrainer] Warning: Previous best checkpoint saving for '{name}' is still in progress. Skipping this save to prevent disk/GIL congestion.")
            return

        ckpt_dir = Path(self.cfg.checkpoint_dir) / self._run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / name
        
        # Build and copy payload synchronously on the main thread (extremely lightweight, ~1-3ms)
        raw_payload = payload if payload is not None else self._checkpoint_payload_lightweight(update=0)
        cpu_payload = _clone_to_cpu(raw_payload)
        
        def _async_save():
            tmp = path.with_suffix(".tmp")
            torch.save(cpu_payload, tmp)
            tmp.replace(path)
            print(f"Best checkpoint saved asynchronously: {path}")

        self._active_best_thread = threading.Thread(target=_async_save, daemon=True)
        self._active_best_thread.start()

    def load_pretrained_weights(self, path: str) -> None:
        """Load policy and scaler from a pretrained checkpoint, discarding optimizer state.

        Use this when starting an RL run from a BC-pretrained policy. The optimizer
        is left in its freshly-initialised state so Adam calibrates to RL gradients
        from scratch — avoiding contamination from BC gradient statistics.

        The avg_policy is synced to the loaded weights so that if avg-model opponents
        are used, they start from the same pretrained base rather than random init.

        Args:
            path: Path to any .pt checkpoint (step_*.pt or best_*.pt).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._policy_module.load_state_dict(ckpt["policy_state_dict"])
        self._avg_policy_module.load_state_dict(ckpt["policy_state_dict"])
        _cast_norms_bf16(self._policy_module)
        _cast_norms_bf16(self._avg_policy_module)
        self._avg_param_cumsum = [
            torch.zeros_like(p) for p in self._policy_module.parameters()
        ]
        self._avg_update_count = 0
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "adv_scaler_state_dict" in ckpt:
            self.adv_scaler.load_state_dict(ckpt["adv_scaler_state_dict"])
        print(f"Pretrained weights loaded from: {path} (optimizer state discarded)")

    def load_checkpoint(self, path: str) -> int:
        """Load policy and optimizer weights from a checkpoint file.

        Args:
            path: Path to a .pt checkpoint file.

        Returns:
            The update index stored in the checkpoint.

        Raises:
            ValueError: If the checkpoint was trained under a different paradigm —
                a policy trained in one paradigm misbehaves when resumed in the
                other (ego_pass policies only ever act as team 0).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        ckpt_paradigm = ckpt.get("train_config", {}).get("paradigm")
        if ckpt_paradigm is not None and ckpt_paradigm != self.cfg.paradigm:
            raise ValueError(
                f"Checkpoint was trained with paradigm={ckpt_paradigm!r} but this "
                f"run uses paradigm={self.cfg.paradigm!r}. Resuming across "
                f"paradigms is not supported."
            )
        self._policy_module.load_state_dict(ckpt["policy_state_dict"])
        _cast_norms_bf16(self._policy_module)
        self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "adv_scaler_state_dict" in ckpt:
            self.adv_scaler.load_state_dict(ckpt["adv_scaler_state_dict"])
        if "avg_policy_state_dict" in ckpt:
            self._avg_policy_module.load_state_dict(ckpt["avg_policy_state_dict"])
            _cast_norms_bf16(self._avg_policy_module)
            self._avg_param_cumsum = [
                c.to(self.device) for c in ckpt["avg_param_cumsum"]
            ]
            self._avg_update_count = ckpt["avg_update_count"]
        if "training_elo" in ckpt:
            self._training_elo = ckpt["training_elo"]
            self._elo_milestone = ckpt.get("elo_milestone", 0.0)
        self._avg_training_elo = ckpt.get("avg_training_elo", 0.0)
        if "eval_window_rand" in ckpt:
            self._eval_window_rand = deque(ckpt["eval_window_rand"], maxlen=100)
        if "eval_window_sc" in ckpt:
            self._eval_window_sc = deque(ckpt["eval_window_sc"], maxlen=100)
        if "eval_window_avg_vs_sc" in ckpt:
            self._eval_window_avg_vs_sc = deque(ckpt["eval_window_avg_vs_sc"], maxlen=100)
        if "eval_window_live_vs_avg" in ckpt:
            self._eval_window_live_vs_avg = deque(ckpt["eval_window_live_vs_avg"], maxlen=100)
        if "global_step" in ckpt:
            self._global_step = ckpt["global_step"]
            self._start_update = ckpt["update"] + 1
        self._elapsed_train_time = ckpt.get("elapsed_train_time", 0.0)
        # Older checkpoints lack ship_steps — reconstruct from update count,
        # exact as long as the scale config hasn't changed between runs.
        ship_tokens_per_update = self.cfg.num_steps * sum(
            sc.num_envs * sc.env_config.num_ships for sc in self.cfg.scales
        )
        self._ship_steps = ckpt.get(
            "ship_steps", ckpt.get("update", 0) * ship_tokens_per_update
        )

        # Restore roster if its JSON exists alongside the checkpoint
        roster_path = Path(path).parent / "roster.json"
        if roster_path.exists():
            self.roster.load_json(roster_path)

        print(f"Checkpoint loaded from: {path} (resuming from update {self._start_update}, step {self._global_step:,})")
        return ckpt["update"]

    # ------------------------------------------------------------------
    # Async logging
    # ------------------------------------------------------------------

    def _init_wandb(
        self,
        train_config: TrainConfig,
        model_config: ModelConfig,
        ship_config: ShipConfig,
        env_config: EnvConfig,
        resume_run_id: str | None = None,
    ) -> None:
        """Initialize W&B run with all configs serialized as the run config."""
        import wandb

        def _sanitize(obj: object) -> object:
            """Recursively convert frozenset/set → sorted list for JSON serialization."""
            if isinstance(obj, (frozenset, set)):
                return sorted(_sanitize(x) for x in obj)  # type: ignore[misc]
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(x) for x in obj]
            return obj

        config: dict = {}
        for prefix, cfg in [
            ("train", train_config),
            ("model", model_config),
            ("ship", ship_config),
            ("env", env_config),
        ]:
            for k, v in dataclasses.asdict(cfg).items():
                if k == "schedule":
                    continue  # TrainingSchedule contains callables — not serializable
                config[f"{prefix}/{k}"] = _sanitize(v)

        if resume_run_id is not None:
            wandb.init(project="boost-and-broadside", config=config, id=resume_run_id, resume="must")
        else:
            wandb.init(project="boost-and-broadside", config=config)

    def _enqueue_log(self, metrics: dict, step: int) -> None:
        """Put metrics onto the async log queue (non-blocking)."""
        self._log_queue.put((metrics, step))

    def _log_worker(self) -> None:
        """Background thread: drains the log queue and calls wandb.log().

        Handles two special value types so the training thread stays off the
        W&B serialization path:
          - ``np.ndarray`` with key ``"hist/returns"`` → one ``wandb.Histogram``
            per reward component, keyed ``hist/returns/<name>``.
          - ``np.ndarray`` with any other key → ``wandb.Histogram`` directly.
        """
        import numpy as np
        import wandb

        while True:
            try:
                item = self._log_queue.get(timeout=1.0)
            except Empty:
                continue
            if item is None:
                break
            raw_metrics, step = item
            processed: dict = {}
            for k, v in raw_metrics.items():
                if isinstance(v, np.ndarray):
                    if k == "hist/returns":
                        # v shape: (alive_count, K) — one histogram per active component
                        for i, name in enumerate(self._active_names):
                            processed[f"hist/returns/{name}"] = wandb.Histogram(v[:, i])
                    else:
                        processed[k] = wandb.Histogram(v)
                else:
                    processed[k] = v
            wandb.log(processed, step=step)
