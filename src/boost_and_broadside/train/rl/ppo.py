"""Recurrent PPO trainer for the Yemong policy.

Core loop: collect rollout → compute per-component GAE → PPO update epochs →
log async → repeat. On top of that, PPOTrainer coordinates:

  - the decomposed critic (per-component returns, lambda aggregation,
    schedule-driven group scales),
  - auxiliary losses (behavior cloning from the scripted agent with
    win-rate-gated decay, next-state prediction, windowed cumulative loss,
    optional SIGReg),
  - opponent management (scripted / avg-model / league fractions, OpponentMixin),
  - continuous in-training Elo ladder evaluation (EloEvaluator) and the roster,
  - checkpointing (CheckpointMixin) and async W&B logging (LoggingMixin).

Rollout and update work run on CUDA streams where available, with a CPU
fallback path; logging stays off the GPU hot path.
"""

import dataclasses
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from pathlib import Path
from queue import Queue

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import (
    ModelConfig,
    ShipConfig,
    TrainConfig,
    TrainingSchedule,
)
from boost_and_broadside.constants import POWER_SLICE, SHOOT_SLICE, TURN_SLICE
from boost_and_broadside.env.field_cache import FieldMapCache
from boost_and_broadside.env.observation import ObsKey, YemongObservation
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.models.yemong.policy import YemongPolicy
from boost_and_broadside.train.rl.buffer import (
    AdvantageScaler,
    LogicalRolloutBuffer,
    MicroBatch,
    ReturnScaler,
    RolloutBuffer,
    StoredRollout,
)
from boost_and_broadside.train.rl.checkpoint import CheckpointMixin
from boost_and_broadside.train.rl.elo_eval import MAX_ANCHORS, EloEvaluator, LadderOpponent
from boost_and_broadside.train.rl.features import (
    FeatureCoordinator,
    build_bullet_coordinator,
    build_standard_coordinator,
)
from boost_and_broadside.train.rl.logging import LoggingMixin
from boost_and_broadside.train.rl.opponents import (
    OpponentMixin,
    flip_team_obs,
)
from boost_and_broadside.train.rl.policy_io import build_policy
from boost_and_broadside.train.rl.roster import EloRoster, RosterEntry
from boost_and_broadside.train.rl.sigreg import SIGReg

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


# Consecutive updates the scripted win rate must hold at bc_winrate_target before
# avg-model accumulation latches on. The eval window refills in roughly two updates,
# so three gives the trigger two near-independent looks at the win rate.
_BC_CUTOFF_UPDATES = 3

# Maps reward component name → the TrainingSchedule group-scale field to apply.
# Effective weight = group_scale * individual_weight (from RewardConfig).
# Groups:
#   true_reward → win components (ally_win, enemy_win)
#   global      → global outcome rewards + shaping (team-aggregated via lambda)
#   local       → self-only per-ship rewards (diagonal lambda, no teammate propagation)
_GROUP: dict[str, str] = {
    "ally_win": "true_reward_scale",
    "enemy_win": "true_reward_scale",
    "ally_combat_damage": "global_scale",
    "enemy_combat_damage": "global_scale",
    "ally_field_damage": "global_scale",
    "enemy_field_damage": "global_scale",
    "ally_combat_death": "global_scale",
    "enemy_combat_death": "global_scale",
    "ally_field_death": "global_scale",
    "enemy_field_death": "global_scale",
    "facing": "local_scale",
    "closing_speed": "local_scale",
    "shoot_quality": "local_scale",
    "kill_shot": "local_scale",
    "kill_assist": "local_scale",
    "combat_damage_taken": "local_scale",
    "field_damage_taken": "local_scale",
    "damage_dealt_enemy": "local_scale",
    "damage_dealt_ally": "local_scale",
    "combat_death": "local_scale",
    "field_death": "local_scale",
    "shooting_penalty": "local_scale",
    "speed": "local_scale",
}

# Components with self-only rewards use a diagonal lambda (i == j); all others
# use team-based lambda aggregation. Derived from _GROUP so the two registries
# cannot silently drift: a "local_scale" component is exactly a self-only one.
_LOCAL_COMPONENTS: frozenset[str] = frozenset(
    name for name, group in _GROUP.items() if group == "local_scale"
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
    checkpoint_interval: int
    num_epochs: int
    target_kl: float | None
    high_elo_threshold: float | None
    high_elo_target_kl: float | None


@dataclasses.dataclass
class _RolloutRuntime:
    """Mutable state that persists across rollout updates."""

    num_envs: int
    num_ships: int
    # Recurrent tokens per env (ships). Fields are non-recurrent, so this is
    # deliberately not N+M — it is the stride for every hidden-state operation.
    num_recurrent: int
    scripted_start: int
    scripted_end: int
    avg_start: int
    avg_end: int
    league_start: int
    league_end: int
    elo_eval: EloEvaluator
    obs: YemongObservation
    hidden: torch.Tensor
    hidden_t1: torch.Tensor | None
    avg_hidden: torch.Tensor | None
    action_buffer: torch.Tensor
    aux_obs: list[YemongObservation]
    aux_hiddens: list[torch.Tensor]
    aux_hidden_t1s: list[torch.Tensor | None]
    aux_action_buffers: list[torch.Tensor]
    aux_last_dones: list[torch.Tensor]
    env_stream: torch.cuda.Stream | None
    net_stream: torch.cuda.Stream | None
    ship_tokens_per_update: int


@dataclasses.dataclass
class _StagedMicroBatch:
    """Pinned source, device copy, and readiness event for one prefetched batch."""

    pinned: MicroBatch
    device: MicroBatch
    ready: torch.cuda.Event


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
        checkpoint_interval=schedule.checkpoint_interval(step),
        num_epochs=schedule.num_epochs(step),
        target_kl=schedule.target_kl(step),
        high_elo_threshold=schedule.high_elo_threshold(step),
        high_elo_target_kl=schedule.high_elo_target_kl(step),
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


class PPOTrainer(CheckpointMixin, LoggingMixin, OpponentMixin):
    """Proximal Policy Optimization for the Yemong multi-agent policy.

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
        resume_wandb_run_id: str | None = None,
    ) -> None:
        self.cfg = train_config
        self.model_config = model_config
        self.ship_config = ship_config
        # Paradigm: "ego_pass" (dual-perspective pass, team 0 trains) vs
        # "shared_pass" (single pass, both teams train). See TrainConfig docstring.
        self._ego_pass = train_config.paradigm == "ego_pass"
        self.coordinator: FeatureCoordinator = build_standard_coordinator(ship_config)
        # Built only when the trunk reads bullets; None keeps the bullet axis off
        # the observation, out of the rollout buffer, and out of the model.
        self.bullet_coordinator: FeatureCoordinator | None = (
            build_bullet_coordinator(ship_config) if model_config.reads_bullets else None
        )
        self.env_config = train_config.scales[0].env_config
        self.device = torch.device(device)
        self._zero_tensor = torch.zeros((), device=self.device)
        self._host_transfer_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
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
            raise ValueError("policy_gradient_coef=0.0 (BC mode) requires a scripted_agent.")

        # Generate valid static field maps before training begins. The cache is
        # shared across all wrappers (primary + auxiliary scales).
        M = train_config.scales[0].env_config.num_fields
        if M > 0 and train_config.field_map is not None:
            cache_cfg = train_config.field_map
            print(
                f"[PPOTrainer] Generating refractive-field map cache "
                f"({cache_cfg.cache_size} maps)..."
            )
            self._field_map = FieldMapCache.generate(
                ship_config,
                train_config.scales[0].env_config,
                cache_cfg,
                self.device,
            )
            print(f"[PPOTrainer] Field map cache ready: {len(self._field_map)} maps")
        else:
            self._field_map = None
        if M > 0 and self._field_map is None:
            raise ValueError("field-enabled training requires TrainConfig.field_map")

        collision_compile_mode = (
            ("max-autotune" if compile_mode == "max-autotune" else "default")
            if compile_mode is not None
            else None
        )
        self.wrapper = YemongEnvWrapper(
            num_envs=train_config.scales[0].num_envs,
            ship_config=ship_config,
            env_config=train_config.scales[0].env_config,
            rewards=train_config.rewards,
            device=device,
            field_map=self._field_map,
            collision_compile_mode=collision_compile_mode,
            include_bullets=model_config.reads_bullets,
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
        self._policy_module = build_policy(
            model_config,
            ship_config,
            num_value_components=K,
            num_ships=N,
            team_pma_k=self._win_k,
        ).to(self.device)
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
        self.enemy_neg_k = self._make_enemy_neg_k(train_config.rewards.enemy_neg_lambda_components)
        self.ally_zero_k = self._make_ally_zero_k(train_config.rewards.ally_zero_components)
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
            min_rms=train_config.advantage_min_rms,
        )
        # Components whose scaler floor has already been reported, so a binding
        # floor warns once rather than every update.
        self._floor_warned: set[str] = set()

        # Per-component aggregated-return diagnostic — refreshed once per update
        # by _precompute_lambda_aggregates (primary scale).
        self._ret_per_comp_mean_k = torch.zeros(K, device=self.device)

        # --- Avg-model opponent (uniform mean of all post-warmup policy snapshots) ---
        # Weights initialized as a copy of the training policy.
        # Accumulation starts when the BC aux loss decays to zero (scripted win
        # rate reaches cfg.bc_winrate_target); once started it never stops.
        self._avg_policy_module = build_policy(
            model_config,
            ship_config,
            num_value_components=K,
            num_ships=N,
            team_pma_k=self._win_k,
        ).to(self.device)
        self.avg_policy = (
            torch.compile(self._avg_policy_module, mode=compile_mode)
            if compile_mode is not None
            else self._avg_policy_module
        )
        self._avg_policy_module.load_state_dict(self._policy_module.state_dict())
        for p in self._avg_policy_module.parameters():
            p.requires_grad_(False)
        self._avg_param_cumsum: list[torch.Tensor] = [
            torch.zeros(p.shape, dtype=torch.float32, device=p.device)
            for p in self._policy_module.parameters()
        ]
        self._avg_update_count: int = 0

        # Warmup: force torch.compile to trace both policies under autocast, so
        # the graph it specializes on matches the one training actually runs.
        # Without this the internal fake-tensor trace runs in fp32 and compiles
        # a graph the first real autocast call immediately invalidates.
        if compile_mode is not None and self.device.type == "cuda":
            # Hidden state covers ship tokens only; fields are non-recurrent.
            _nt = N
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

        # --- League play + Elo ---
        self.roster = EloRoster(
            max_size=train_config.league_size,
            elo_temperature=train_config.elo_temperature,
            uniform_sampling=train_config.league_uniform_sampling,
        )
        # Random anchor is added by EloRoster.__init__ (Elo=0, fixed).
        # "avg" entry is added when _update_avg_model() is first called.

        # Training Elo starts at 0 — all ratings begin
        # at the same point and diverge as eval matchups accumulate.
        self._training_elo: float = 0.0
        self._avg_training_elo: float = 0.0
        self._scripted_elo: float = train_config.elo_eval.scripted_elo_init
        self._floating_games: int = 0  # rated games of the floating ladder checkpoint
        self._bc_cutoff_streak: int = 0  # consecutive updates past the BC win-rate target
        # Latest update's rated outcomes, opponent label → (win, loss, tie).
        self._match_counts: dict[str, tuple[int, int, int]] = {}
        eval_window_size = train_config.elo_eval.window_size
        self._eval_window_rand = deque(maxlen=eval_window_size)
        self._eval_window_sc = deque(maxlen=eval_window_size)
        self._eval_window_ladder = deque(maxlen=eval_window_size)
        self._eval_window_floating = deque(maxlen=eval_window_size)
        self._eval_window_live_vs_avg = deque(maxlen=eval_window_size)
        # Highest claimed ladder-milestone grid point, in normalized Elo (vs random).
        # Always a multiple of cfg.elo_milestone_gap once the first snapshot lands;
        # runs resumed from before the grid existed carry one off-grid value forward
        # and snap to the grid at their next snapshot.
        self._elo_milestone: float = 0.0
        self._best_training_elo_norm: float = 0.0  # best normalized training Elo seen so far
        self._best_avg_elo_norm: float = 0.0  # best normalized avg Elo seen so far
        self._last_checkpoint_path: Path | None = None

        # Current league opponent for the ongoing rollout (rotated each rollout).
        self._current_league_entry: RosterEntry | None = None
        self._current_league_policy: YemongPolicy | None = None

        # Async logging queue
        self._log_queue: Queue = Queue()
        if use_wandb:
            self._init_wandb(
                train_config, model_config, ship_config, self.env_config, resume_wandb_run_id
            )
            self._log_thread = threading.Thread(target=self._log_worker, daemon=True)
            self._log_thread.start()

        self._global_step = 0
        self._start_update = 1
        # Cumulative counters persisted across checkpoint resumes so throughput
        # metrics behave as if training never stopped.
        self._ship_steps = 0  # ship tokens (all teams, all envs, all scales)
        # Entity tokens (ships + fields) the update phase processes per epoch —
        # one full pass over all scales' rollouts.
        self._entity_tokens_per_epoch = (
            train_config.num_steps
            * sum(
                sc.num_envs * (sc.env_config.num_ships + sc.env_config.num_fields)
                for sc in train_config.scales
            )
            * train_config.rollouts_per_update
        )
        # Cumulative entity tokens consumed by backward passes (counts actual
        # epochs completed, so target_kl early stops are reflected). The compute
        # x-axis for comparing runs with different batch/update configurations.
        self._grad_tokens = 0
        self._elapsed_train_time = 0.0  # wall-clock seconds spent training
        self._train_start_time = time.time()  # reset at the top of train()
        total_envs_all = sum(sc.num_envs for sc in train_config.scales)
        self._num_updates = train_config.total_timesteps // (
            total_envs_all * train_config.num_steps * train_config.rollouts_per_update
        )

        # Run name used as checkpoint subdirectory (e.g. "checkpoints/good-spaceship-223/")
        if use_wandb:
            import wandb as _wandb

            self.run_name: str = _wandb.run.name
            run_id_path = Path(train_config.checkpoint_dir) / self.run_name / "wandb_run_id.txt"
            run_id_path.parent.mkdir(parents=True, exist_ok=True)
            run_id_path.write_text(_wandb.run.id)
        else:
            from datetime import datetime

            self.run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Schedule state — evaluated from the schedule functions each update.
        # Initialized from step=0 and refreshed after every PPO update.
        self._schedule_state: _ResolvedSchedule = base_state
        self._policy_gradient_coef: float = base_state.policy_gradient_coef
        self._behavior_cloning_coef: float = base_state.behavior_cloning_coef

        # --- Auxiliary training scales (multi-scale curriculum) ---
        # Each scale has its own env + buffer; policy, optimizer, and scaler are shared.
        # Pure self-play only — no scripted/avg/league opponents on aux scales.
        self.aux_wrappers: list[YemongEnvWrapper] = []
        self.aux_buffers: list[RolloutBuffer] = []

        for sc in train_config.scales[1:]:
            aux_w = YemongEnvWrapper(
                num_envs=sc.num_envs,
                ship_config=ship_config,
                env_config=sc.env_config,
                rewards=train_config.rewards,
                device=device,
                field_map=self._field_map,
                collision_compile_mode=collision_compile_mode,
                include_bullets=model_config.reads_bullets,
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
                num_tokens=sc.env_config.num_ships + sc.env_config.num_fields,
            )
            self.aux_wrappers.append(aux_w)
            self.aux_buffers.append(aux_buf)

        self._active_save_thread = None
        self._active_best_thread = None
        self._active_best_avg_thread = None

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

        Used for enemy-perspective source-split damage/death components and enemy_win,
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

    def _rollout_policy_pass(
        self,
        obs: YemongObservation,
        hidden: torch.Tensor,
        hidden_t1: torch.Tensor | None,
        num_ships: int,
        num_recurrent: int,
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
            obs:        YemongObservation with (B, N+M, ...) tensors (raw team IDs).
            hidden:     (n_layers, B*N, CONV_KERNEL*D) raw-perspective hidden state.
            hidden_t1:  Flipped-perspective hidden state; None in shared_pass.
            num_ships:  N — ship token count for team flipping.
            num_recurrent: N — recurrent tokens per env, used to split the 2B hidden
                state. Fields are non-recurrent, so this is ships, not N+M.

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
            action, logprob, value_norm, pred_next, hidden = self.policy.get_action_and_value(
                obs, hidden
            )
            return action, None, logprob, value_norm, pred_next, hidden, None

        batch = hidden.shape[1] // num_recurrent
        obs_t1 = flip_team_obs(obs, num_ships)
        obs_both = obs.concat_batch(obs_t1)
        hidden_both = torch.cat([hidden, hidden_t1], dim=1)  # (n_layers, 2B*N, CONV_KERNEL*D)
        action_both, logprob_both, value_both, pred_next_both, hidden_out = (
            self.policy.get_action_and_value(obs_both, hidden_both)
        )
        return (
            action_both[:batch],  # (B, N, 3)
            action_both[batch:],  # (B, N, 3)
            logprob_both[:batch],  # (B, N)
            value_both[:batch],  # (B, N, K)
            pred_next_both[:batch],  # (B, N, pred_dim)
            hidden_out[:, : batch * num_recurrent, :],  # (n_layers, B*N, CONV_KERNEL*D)
            hidden_out[:, batch * num_recurrent :, :],  # (n_layers, B*N, CONV_KERNEL*D)
        )

    def _collect_aux_steps(
        self,
        aux_obs: list[YemongObservation],
        aux_hiddens: list[torch.Tensor],
        aux_hidden_t1s: list[torch.Tensor | None],
        aux_action_buffers: list[torch.Tensor],
        aux_last_dones: list[torch.Tensor],
    ) -> None:
        """Collect one pure-self-play transition for every auxiliary scale."""
        # Aux-scale rollout steps (pure self-play, 1-step delay)
        for i, (sc, aux_w, aux_buf) in enumerate(
            zip(self.cfg.scales[1:], self.aux_wrappers, self.aux_buffers)
        ):
            aux_N = sc.env_config.num_ships
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
                    aux_obs[i], aux_hiddens[i], aux_hidden_t1s[i], aux_N, aux_N
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
            aux_hiddens[i] = self.policy.reset_hidden_for_envs(aux_hiddens[i], aux_done_any, aux_N)
            if self._ego_pass:
                aux_hidden_t1s[i] = self.policy.reset_hidden_for_envs(
                    aux_hidden_t1s[i], aux_done_any, aux_N
                )
            aux_action_buffers[i] = aux_action.detach().clone()
            aux_action_buffers[i][aux_done_any] = 0
            aux_last_dones[i] = aux_dones
            aux_obs[i] = next_aux_obs
            self._global_step += sc.num_envs

    def _initialize_rollout_runtime(self) -> _RolloutRuntime:
        """Initialize persistent primary, auxiliary, and evaluation rollout state."""
        num_envs = self.cfg.scales[0].num_envs
        num_ships = self.wrapper.num_ships
        # Only ships carry recurrent state; field tokens take the non-recurrent path.
        num_recurrent = num_ships
        scripted_start = self.B_self
        scripted_end = scripted_start + self.B_sc
        avg_start = scripted_end
        avg_end = avg_start + self.B_avg
        league_start = avg_end

        obs = self.wrapper.reset()
        self.wrapper.env.state.step_count.random_(0, self.env_config.max_episode_steps)
        hidden = self.policy.initial_hidden(num_envs, num_recurrent, self.device)
        hidden_t1 = (
            self.policy.initial_hidden(num_envs, num_recurrent, self.device)
            if self._ego_pass
            else None
        )
        avg_hidden = (
            self.avg_policy.initial_hidden(self.B_avg, num_recurrent, self.device)
            if self.B_avg > 0
            else None
        )
        action_buffer = torch.zeros(num_envs, num_ships, 3, dtype=torch.int32, device=self.device)

        aux_obs: list[YemongObservation] = []
        aux_hiddens: list[torch.Tensor] = []
        aux_hidden_t1s: list[torch.Tensor | None] = []
        aux_action_buffers: list[torch.Tensor] = []
        aux_last_dones: list[torch.Tensor] = []
        for scale, wrapper in zip(self.cfg.scales[1:], self.aux_wrappers):
            aux_obs.append(wrapper.reset())
            wrapper.env.state.step_count.random_(0, scale.env_config.max_episode_steps)
            aux_tokens = scale.env_config.num_ships  # recurrent tokens: ships only
            aux_hiddens.append(self.policy.initial_hidden(scale.num_envs, aux_tokens, self.device))
            aux_hidden_t1s.append(
                self.policy.initial_hidden(scale.num_envs, aux_tokens, self.device)
                if self._ego_pass
                else None
            )
            aux_action_buffers.append(
                torch.zeros(
                    scale.num_envs,
                    scale.env_config.num_ships,
                    3,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            aux_last_dones.append(torch.zeros(scale.num_envs, dtype=torch.bool, device=self.device))

        anchors, floating = self._ladder_eval_state()
        # The ladder can hold entries from before the live architecture, so the
        # bullet axis follows the union: a policy that ignores bullets is
        # unaffected by their presence, one that reads them and is handed an
        # observation without them plays blind.
        eval_reads_bullets = self.model_config.reads_bullets or any(
            opponent.reads_bullets for opponent in [*anchors, floating] if opponent is not None
        )
        return _RolloutRuntime(
            num_envs=num_envs,
            num_ships=num_ships,
            num_recurrent=num_recurrent,
            scripted_start=scripted_start,
            scripted_end=scripted_end,
            avg_start=avg_start,
            avg_end=avg_end,
            league_start=league_start,
            league_end=num_envs,
            elo_eval=EloEvaluator(
                config=self.cfg.elo_eval,
                ship_config=self.ship_config,
                env_config=self.env_config,
                device=self.device,
                field_map=self._field_map,
                live_policy=self.policy,
                avg_policy=self.avg_policy,
                scripted_agent=self.scripted_agent,
                num_ships=num_ships,
                num_tokens=num_recurrent,
                ego_pass=self._ego_pass,
                live_elo=self._training_elo,
                avg_elo=self._avg_training_elo,
                scripted_elo=self._scripted_elo,
                anchors=anchors,
                floating=floating,
                floating_games=self._floating_games,
                random_window=self._eval_window_rand,
                ladder_window=self._eval_window_ladder,
                floating_window=self._eval_window_floating,
                scripted_window=self._eval_window_sc,
                live_vs_avg_window=self._eval_window_live_vs_avg,
                include_bullets=eval_reads_bullets,
            ),
            obs=obs,
            hidden=hidden,
            hidden_t1=hidden_t1,
            avg_hidden=avg_hidden,
            action_buffer=action_buffer,
            aux_obs=aux_obs,
            aux_hiddens=aux_hiddens,
            aux_hidden_t1s=aux_hidden_t1s,
            aux_action_buffers=aux_action_buffers,
            aux_last_dones=aux_last_dones,
            env_stream=torch.cuda.Stream() if self.device.type == "cuda" else None,
            net_stream=torch.cuda.Stream() if self.device.type == "cuda" else None,
            ship_tokens_per_update=self.cfg.num_steps
            * sum(scale.num_envs * scale.env_config.num_ships for scale in self.cfg.scales)
            * self.cfg.rollouts_per_update,
        )

    def _collect_rollout(self, runtime: _RolloutRuntime, avg_eval_active: bool) -> torch.Tensor:
        """Collect one complete primary and auxiliary rollout."""
        self.buffer.reset()
        self.buffer.store_initial_hidden(runtime.hidden)
        for aux_buffer, aux_hidden in zip(self.aux_buffers, runtime.aux_hiddens):
            aux_buffer.reset()
            aux_buffer.store_initial_hidden(aux_hidden)

        league_hidden = self._prepare_league_opponent(runtime.num_recurrent)
        for rollout_step in range(self.cfg.num_steps):
            primary = self._collect_primary_step(
                obs=runtime.obs,
                hidden=runtime.hidden,
                hidden_t1=runtime.hidden_t1,
                avg_hidden=runtime.avg_hidden,
                league_hidden=league_hidden,
                action_buffer=runtime.action_buffer,
                num_envs=runtime.num_envs,
                num_ships=runtime.num_ships,
                num_recurrent=runtime.num_recurrent,
                scripted_start=runtime.scripted_start,
                scripted_end=runtime.scripted_end,
                avg_start=runtime.avg_start,
                avg_end=runtime.avg_end,
                league_start=runtime.league_start,
                league_end=runtime.league_end,
                env_stream=runtime.env_stream,
                net_stream=runtime.net_stream,
            )
            (
                runtime.obs,
                runtime.hidden,
                runtime.hidden_t1,
                runtime.avg_hidden,
                league_hidden,
                runtime.action_buffer,
                dones,
            ) = primary
            self._collect_aux_steps(
                runtime.aux_obs,
                runtime.aux_hiddens,
                runtime.aux_hidden_t1s,
                runtime.aux_action_buffers,
                runtime.aux_last_dones,
            )
            runtime.elo_eval.step(rollout_step, avg_eval_active)

        elo_snapshot = runtime.elo_eval.flush(avg_eval_active)
        self._training_elo = elo_snapshot.live_elo
        self._avg_training_elo = elo_snapshot.avg_elo
        self._scripted_elo = elo_snapshot.scripted_elo
        self._floating_games = elo_snapshot.floating_games
        self._match_counts = elo_snapshot.match_counts
        if elo_snapshot.floating_elo is not None:
            self.roster.set_floating_elo(elo_snapshot.floating_elo)
        return dones

    def _compute_rollout_gae(
        self,
        runtime: _RolloutRuntime,
        dones: torch.Tensor,
        update_scalers: bool = True,
    ) -> None:
        """Store final observations and compute GAE for every scale.

        Args:
            runtime: Persistent environment and recurrent rollout state.
            dones: Primary-scale terminal flags after the final step.
            update_scalers: Update statistics immediately for a single-shard batch.
                Logical host batches defer this until every shard is available.
        """
        self.buffer.store_final_obs(runtime.obs)
        for index, aux_buffer in enumerate(self.aux_buffers):
            aux_buffer.store_final_obs(runtime.aux_obs[index])

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, next_value_norm, _, _ = self.policy.get_action_and_value(
                runtime.obs, runtime.hidden
            )
        self.buffer.compute_gae(self.scaler.denormalize(next_value_norm), dones.float())
        for index, (aux_buffer, aux_hidden) in enumerate(
            zip(self.aux_buffers, runtime.aux_hiddens)
        ):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, next_aux_norm, _, _ = self.policy.get_action_and_value(
                    runtime.aux_obs[index], aux_hidden
                )
            aux_buffer.compute_gae(
                self.scaler.denormalize(next_aux_norm),
                runtime.aux_last_dones[index].float(),
            )
        if update_scalers:
            self.scaler.update(self.buffer.returns)
            self.adv_scaler.update(self.buffer.advantages, self.buffer.alive_mask)

    def _collect_host_rollouts(
        self,
        runtime: _RolloutRuntime,
        avg_eval_active: bool,
    ) -> list[LogicalRolloutBuffer]:
        """Collect a logical PPO batch into CPU-resident rollout shards.

        Args:
            runtime: Persistent rollout state.
            avg_eval_active: Whether average-policy Elo evaluation is active.

        Returns:
            One logical host buffer per training scale.
        """
        device_buffers = [self.buffer] + self.aux_buffers
        stored_by_scale: list[list[StoredRollout]] = [[] for _ in device_buffers]
        for _ in range(self.cfg.rollouts_per_update):
            dones = self._collect_rollout(runtime, avg_eval_active)
            self._compute_rollout_gae(runtime, dones, update_scalers=False)
            for scale_index, (shards, buffer) in enumerate(
                zip(stored_by_scale, device_buffers, strict=True)
            ):
                if scale_index == 0:
                    self._precompute_ns_labels(buffer)
                else:
                    buffer.ns_labels = None
                shards.append(StoredRollout(buffer))

        primary_shards = stored_by_scale[0]
        self.scaler.update_chunks(
            [shard.returns for shard in primary_shards],
            max_samples=self.cfg.return_quantile_samples,
        )
        self.adv_scaler.update_chunks(
            [shard.advantages for shard in primary_shards],
            [shard.alive_mask for shard in primary_shards],
        )
        return self._prepare_host_rollouts(device_buffers, stored_by_scale)

    @torch.no_grad()
    def _prepare_host_rollouts(
        self,
        device_buffers: list[RolloutBuffer],
        stored_by_scale: list[list[StoredRollout]],
    ) -> list[LogicalRolloutBuffer]:
        """Compute derived PPO data a shard at a time on the GPU.

        Args:
            device_buffers: Reusable fixed-width GPU buffers, one per scale.
            stored_by_scale: CPU rollout shards grouped by scale.

        Returns:
            Logical host buffers ready for PPO epoch iteration.
        """
        comp_weights = torch.tensor(
            [component.weight for component in self.wrapper.active_components],
            dtype=torch.float32,
            device=self.device,
        )  # (K,)
        logical_buffers = []
        for scale_index, (device_buffer, stored_shards) in enumerate(
            zip(device_buffers, stored_by_scale, strict=True)
        ):
            is_primary = scale_index == 0
            adv_square_sum = torch.zeros((), device=self.device)
            adv_count = torch.zeros((), device=self.device)
            ret_component_sum = torch.zeros(device_buffer.num_components, device=self.device)
            ret_actor_count = torch.zeros((), device=self.device)

            for stored in stored_shards:
                stored.restore_aggregate_inputs(device_buffer)
                shard_stats = self._precompute_lambda_aggregates(
                    device_buffer,
                    comp_weights,
                    is_primary=is_primary,
                )
                shard_adv_sum, shard_adv_count, shard_ret_sum, shard_actor_count = shard_stats
                adv_square_sum += shard_adv_sum
                adv_count += shard_adv_count
                if shard_ret_sum is not None and shard_actor_count is not None:
                    ret_component_sum += shard_ret_sum
                    ret_actor_count += shard_actor_count

                stored.capture_aggregates(device_buffer)

            adv_rms = adv_square_sum / adv_count.clamp(min=1.0)
            if is_primary:
                self._ret_per_comp_mean_k = ret_component_sum / ret_actor_count.clamp(min=1.0)
            logical_buffers.append(LogicalRolloutBuffer(stored_shards, adv_rms))
        return logical_buffers

    def _refresh_training_schedule(self, metrics: dict, elo_eval: EloEvaluator) -> None:
        """Refresh schedule-controlled optimization, reward, and averaging state."""
        self._schedule_state = _resolve_schedule(self.cfg.schedule, self._global_step)
        self._policy_gradient_coef = self._schedule_state.policy_gradient_coef
        # BC aux loss decays linearly with the win rate against the scripted
        # agent, reaching zero at bc_winrate_target (full strength before any
        # scripted games have been recorded).
        window_sc = self._eval_window_sc
        scripted_win_rate = sum(window_sc) / len(window_sc) if window_sc else 0.0
        bc_factor = max(0.0, 1.0 - scripted_win_rate / self.cfg.bc_winrate_target)
        self._behavior_cloning_coef = self._schedule_state.behavior_cloning_coef * bc_factor
        self.optim.param_groups[0]["lr"] = self._schedule_state.learning_rate
        for component in self.wrapper.reward_components:
            raw_weight = getattr(self.cfg.rewards, f"{component.name}_weight")
            component.weight = raw_weight * getattr(self._schedule_state, _GROUP[component.name])
        self.wrapper.refresh_component_weights()

        metrics["schedule/learning_rate"] = self._schedule_state.learning_rate
        metrics["schedule/policy_gradient_coef"] = self._policy_gradient_coef
        metrics["schedule/behavior_cloning_coef"] = self._behavior_cloning_coef
        metrics["schedule/bc_decay_factor"] = bc_factor
        metrics["schedule/target_kl"] = self._effective_target_kl()
        metrics["schedule/true_reward_scale"] = self._schedule_state.true_reward_scale
        metrics["schedule/global_scale"] = self._schedule_state.global_scale
        metrics["schedule/local_scale"] = self._schedule_state.local_scale

        # Avg-model accumulation picks up exactly where the BC aux loss lets go:
        # bc_factor hits zero when the scripted win rate reaches bc_winrate_target.
        # Keyed to the win rate rather than to _behavior_cloning_coef, because
        # profiles that disable BC entirely (behavior_cloning_coef=0) would
        # otherwise trip the trigger on update one.
        #
        # The gate latches forever, so it is guarded against a lucky window: the
        # window must be full, and the target must hold for _BC_CUTOFF_UPDATES
        # consecutive updates — long enough to refresh the window end to end.
        # The streak is not checkpointed; a resume mid-streak just re-earns it.
        if bc_factor <= 0.0 and len(window_sc) == window_sc.maxlen:
            self._bc_cutoff_streak += 1
        else:
            self._bc_cutoff_streak = 0
        bc_cutoff_reached = self._bc_cutoff_streak >= _BC_CUTOFF_UPDATES
        metrics["schedule/bc_cutoff_streak"] = self._bc_cutoff_streak
        if self._policy_gradient_coef > 0.0 and self.B_avg > 0:
            if self._avg_update_count > 0 or bc_cutoff_reached:
                first_avg_update = self._avg_update_count == 0
                self._update_avg_model()
                if first_avg_update:
                    elo_eval.seed_avg_elo_from_live()
                    self._avg_training_elo = self._training_elo

    def train(self) -> None:
        """Run the full PPO training loop."""
        runtime = self._initialize_rollout_runtime()
        self._train_start_time = time.time()

        for update in range(self._start_update, self._num_updates + 1):
            avg_eval_active = self._avg_update_count > 0
            if self.cfg.rollouts_per_update == 1:
                dones = self._collect_rollout(runtime, avg_eval_active)
                self._compute_rollout_gae(runtime, dones)
                update_buffers: list[RolloutBuffer | LogicalRolloutBuffer] = [
                    self.buffer,
                    *self.aux_buffers,
                ]
                precomputed = False
            else:
                update_buffers = self._collect_host_rollouts(runtime, avg_eval_active)
                precomputed = True

            record_hist = update % self.cfg.histogram_interval == 0
            metrics = self._update_epochs(
                all_buffers=update_buffers,
                record_histograms=record_hist,
                precomputed=precomputed,
            )

            self._refresh_training_schedule(metrics, runtime.elo_eval)
            sps, ship_tps = self._assemble_metrics(metrics, update, runtime.ship_tokens_per_update)

            self._log_training_update(metrics, update, sps, ship_tps)
            self._maybe_save_checkpoint(update)
            self._maybe_advance_ladder(update, runtime.elo_eval)

        self.shutdown()

    def shutdown(self) -> None:
        """Release GPU memory and cleanly terminate background threads/processes.

        Safe to call more than once.
        """
        if getattr(self, "_shutdown_called", False):
            return
        self._shutdown_called = True
        self._wait_for_checkpoint_saves()
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

    def _stage_microbatch(self, batch: MicroBatch) -> _StagedMicroBatch:
        """Pin and enqueue one host micro-batch on the dedicated copy stream."""
        if self._host_transfer_stream is None:
            raise RuntimeError("host transfer staging requires a CUDA device")
        pinned = batch.pin_memory()
        with torch.cuda.stream(self._host_transfer_stream):
            device_batch = pinned.to(self.device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self._host_transfer_stream)
        return _StagedMicroBatch(pinned=pinned, device=device_batch, ready=ready)

    def _iter_device_chunks(
        self,
        chunks: list[MicroBatch],
        buffer: RolloutBuffer | LogicalRolloutBuffer,
    ) -> Generator[tuple[MicroBatch, MicroBatch]]:
        """Yield device chunks while prefetching the following host chunk.

        Args:
            chunks: Micro-batches forming one optimizer minibatch.
            buffer: Source buffer, used to split staged logical shard minibatches.

        Yields:
            Source/device pairs. The source supplies shape metadata without a sync.
        """
        if not isinstance(buffer, LogicalRolloutBuffer):
            for chunk in chunks:
                yield chunk, chunk.to(self.device)
            return

        tokens_per_env = buffer.num_steps * buffer.num_tokens

        def split_count(batch: MicroBatch) -> int:
            if self.cfg.microbatch_tokens is None:
                return 1
            batch_tokens = batch.actions.shape[1] * tokens_per_env
            count = -(-batch_tokens // self.cfg.microbatch_tokens)
            return min(max(count, 1), batch.actions.shape[1])

        if self.device.type != "cuda":
            for chunk in chunks:
                for microbatch in chunk.split_envs(split_count(chunk), buffer.num_ships):
                    yield microbatch, microbatch
            return

        current_stream = torch.cuda.current_stream(self.device)
        staged = self._stage_microbatch(chunks[0])
        for index, source in enumerate(chunks):
            current_stream.wait_event(staged.ready)
            staged.device.record_stream(current_stream)
            next_staged = (
                self._stage_microbatch(chunks[index + 1]) if index + 1 < len(chunks) else None
            )
            source_microbatches = source.split_envs(split_count(source), buffer.num_ships)
            device_microbatches = staged.device.split_envs(
                split_count(source),
                buffer.num_ships,
            )
            yield from zip(source_microbatches, device_microbatches, strict=True)
            if next_staged is not None:
                staged = next_staged

    @torch.no_grad()
    def _minibatch_denominators(
        self,
        chunks: list[MicroBatch],
        buf: RolloutBuffer | LogicalRolloutBuffer,
        is_primary: bool,
    ) -> dict:
        """Minibatch-total loss denominators, summed over the micro-batches.

        Masked-mean loss terms in _compute_minibatch_loss divide by these
        totals instead of micro-batch-local counts, so micro-batch losses sum
        exactly to the unsplit minibatch loss and gradient accumulation is
        equivalent to one large minibatch. All masks are rollout data, so this
        is cheap and policy-independent.
        """
        source_device = chunks[0].alive.device
        _z = torch.zeros((), device=source_device)
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
            mb_alive = chunk.alive
            mb_actor_mask = chunk.actor_mask
            mb_expert_probs = chunk.expert_probs
            mb_terminated = chunk.terminated
            alive_sum += mb_alive.sum()
            actor_sum += (mb_actor_mask & mb_alive).sum()
            numel += mb_alive.numel()
            if need_bc:
                bc_valid = mb_expert_probs.sum(-1) > 0
                bc_sum += (bc_valid & mb_actor_mask & mb_alive).sum()
            if need_ns:
                ns_sum += (mb_alive & ~mb_terminated.unsqueeze(-1)).sum()
        return {
            "mask_sum": alive_sum.clamp(min=1.0).to(self.device),
            "actor_sum": actor_sum.clamp(min=1.0).to(self.device),
            "bc_sum": bc_sum.clamp(min=1.0).to(self.device),
            "ns_sum": ns_sum.clamp(min=1.0).to(self.device),
            "numel": float(numel),
            "adv_rms": buf.adv_rms,
        }

    def _compute_minibatch_loss(
        self,
        batch: MicroBatch,
        is_primary: bool,
        denoms: dict,
        frac: float,
    ) -> tuple[torch.Tensor, dict]:
        """Compute PPO loss for one micro-batch. Does NOT call zero_grad / backward / step.

        Loss coefficients are read from ``self._policy_gradient_coef``,
        ``self._behavior_cloning_coef``, and ``self._schedule_state``. Setting
        ``policy_gradient_coef=0.0`` activates BC pretraining mode.

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

        mb_obs = batch.obs
        mb_actions = batch.actions
        mb_old_logprobs = batch.old_logprobs
        mb_advantages = batch.advantages
        mb_returns = batch.returns
        mb_alive = batch.alive
        mb_hidden = batch.hidden
        mb_actor_mask = batch.actor_mask
        mb_expert_probs = batch.expert_probs
        mb_terminated = batch.terminated
        mb_adv_agg = batch.adv_agg
        mb_ret_agg = batch.ret_agg
        mb_ns_labels = batch.ns_labels

        # mb_obs has T+1 steps; first T for encode/evaluate, last T for next-state aux loss.
        T = mb_alive.shape[0]
        curr_mb_obs = mb_obs.slice_time(0, T)

        need_sigreg = self._schedule_state.sigreg_coef > 0.0
        # evaluate_actions needs the full (T, B, N+M) alive mask so Yemong layers
        # can attend to field tokens; mb_alive is ships-only and used for loss masking.
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
            ns_mask = mb_alive & non_terminal  # (T, B_mb, N)
            ns_mask_f = ns_mask.float()
            ns_sum = denoms["ns_sum"]

            # Labels precomputed once per update from the T+1 obs storage
            # (see _precompute_ns_labels) — they depend only on rollout data.
            labels = mb_ns_labels  # (T, B_mb, N, pred_dim)

            P = self.coordinator.total_prediction_dimension
            sq_err = (pred_next.float() - labels.detach()).pow(2)  # (T, B, N, pred_dim)
            sq_err = sq_err * self.aux_weights  # per-prediction weight

            if self.cfg.next_state_coef > 0.0:
                next_state_cont_loss = (sq_err * ns_mask_f.unsqueeze(-1)).sum() / (ns_sum * P)
                next_state_loss = next_state_cont_loss

            if self.cfg.windowed_loss_coef > 0.0:
                # Internally a masked mean over its own validity mask — weight
                # by env fraction like sigreg (exact when the minibatch is unsplit).
                windowed_ns_loss = (
                    self.coordinator.compute_windowed_loss(
                        pred_next.float(),
                        labels.detach(),
                        ns_mask,
                        mb_terminated,
                    )
                    * frac
                )

            with torch.no_grad():
                next_state_per_feat = (sq_err * ns_mask_f.unsqueeze(-1)).sum(
                    (0, 1, 2)
                ) / ns_sum  # (pred_dim,) gpu, additive across chunks

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

            # Per-head entropy from the logits already returned by evaluate_actions.
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
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

        ally_lam = torch.where(self.ally_zero_k, 0.0, 1.0)  # (K,)
        enemy_lam = torch.where(self.enemy_neg_k, -1.0, 0.0)  # (K,)
        identity = torch.eye(N, dtype=torch.float32, device=self.device)
        local_lambda = identity[None, None, :, :, None]  # (1, 1, N, N, 1)

        adv_sq_sum = torch.zeros((), device=self.device)
        adv_cnt = torch.zeros((), device=self.device)
        if is_primary:
            ret_pc_sum = torch.zeros(buf.num_components, device=self.device)
            actor_sum = torch.zeros((), device=self.device)

        chunk = max(1, B // self.cfg.num_minibatches)
        if self.cfg.microbatch_tokens is not None:
            chunk = min(chunk, max(1, self.cfg.microbatch_tokens // (T * buf.num_tokens)))
        for start in range(0, B, chunk):
            sl = slice(start, start + chunk)
            alive = buf.alive_mask[:, sl]  # (T, b, N)
            team_id_t = buf.obs[ObsKey.TEAM_ID][:T, sl, :N].long()  # (T, b, N)
            same_team_t = team_id_t.unsqueeze(3) == team_id_t.unsqueeze(2)  # (T, b, N, N)
            alive_j = alive.float().unsqueeze(2).unsqueeze(-1)  # (T, b, 1, N_j, 1)

            global_lambda = (
                same_team_t.float().unsqueeze(-1) * ally_lam
                + (~same_team_t).float().unsqueeze(-1) * enemy_lam
            )  # (T, b, N_i, N_j, K)
            lambda_ij_t = (
                torch.where(self.local_k, local_lambda, global_lambda) * comp_weights * alive_j
            )
            lambda_norm = lambda_ij_t.abs().sum(dim=3, keepdim=True).clamp(min=1.0)
            lambda_ij_t = lambda_ij_t / lambda_norm

            # advantages/returns are bf16-stored; normalize() promotes advantages via
            # the fp32 rms divisor, and returns is upcast explicitly so the einsum with
            # the fp32 lambda tensor stays fp32 (einsum will not mix dtypes).
            adv_normed = self.adv_scaler.normalize(buf.advantages[:, sl])
            returns_sl = buf.returns[:, sl].float()
            buf.adv_agg[:, sl] = torch.einsum("tbijk,tbjk->tbi", lambda_ij_t, adv_normed)
            buf.ret_agg[:, sl] = torch.einsum("tbijk,tbjk->tbi", lambda_ij_t, returns_sl)

            actor_f = (buf.actor_masks[:, sl] & alive).float()
            adv_sq_sum += (buf.adv_agg[:, sl].pow(2) * actor_f).sum()
            adv_cnt += actor_f.sum()

            if is_primary:
                ret_pc = torch.einsum("tbijk,tbjk->tbik", lambda_ij_t, returns_sl)  # (T, b, N, K)
                ret_pc_sum += (ret_pc * actor_f.unsqueeze(-1)).sum((0, 1, 2))
                actor_sum += actor_f.sum()

        buf.adv_rms = adv_sq_sum / adv_cnt.clamp(min=1.0)
        if is_primary:
            self._ret_per_comp_mean_k = ret_pc_sum / actor_sum.clamp(min=1.0)
            return adv_sq_sum, adv_cnt, ret_pc_sum, actor_sum
        return adv_sq_sum, adv_cnt, None, None

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
        ship_obs = YemongObservation(
            data={
                k: (
                    v[:, :, :N].reshape((T + 1) * B, N, *v.shape[3:])
                    if v.dim() > 3
                    else v[:, :, :N].reshape((T + 1) * B, N)
                )
                for k, v in buf.obs.items()
            }
        )
        targets = self.coordinator.get_target_vector(ship_obs)  # ((T+1)*B, N, t_dim)
        targets = targets.reshape(T + 1, B, N, -1)
        buf.ns_labels = self.coordinator.compute_labels(
            targets[:T], targets[1:]
        )  # (T, B, N, pred_dim)

    def _update_epochs(
        self,
        all_buffers: list[RolloutBuffer | LogicalRolloutBuffer],
        record_histograms: bool = False,
        precomputed: bool = False,
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
            precomputed: Derived tensors already exist in host-backed logical buffers.

        Returns:
            Dict of mean metric values over all minibatch updates.
        """
        cfg = self.cfg
        K = self.buffer.num_components
        n_scales = len(all_buffers)

        comp_weights = torch.tensor(
            [c.weight for c in self.wrapper.active_components],
            dtype=torch.float32,
            device=self.device,
        )  # (K,)

        # Precompute everything that depends only on rollout data (not the
        # policy) once per update instead of once per minibatch: the lambda
        # aggregation and the aux next-state labels (primary scale only).
        if not precomputed:
            for scale_idx, buf in enumerate(all_buffers):
                assert isinstance(buf, RolloutBuffer)
                self._precompute_lambda_aggregates(buf, comp_weights, is_primary=(scale_idx == 0))
                if scale_idx > 0:
                    buf.ns_labels = None  # aux scales never use the aux losses
            primary = all_buffers[0]
            assert isinstance(primary, RolloutBuffer)
            self._precompute_ns_labels(primary)

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
            # Fraction of optimizer steps whose gradients were non-finite and
            # got scrubbed. Any sustained non-zero reading means the forward or
            # backward pass is overflowing and needs investigating at source.
            "train/nonfinite_grad_fraction": [],
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
            "vel_dvx_norm",
            "vel_dvy_norm",
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
        target_kl = self._effective_target_kl()

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
                # (accum_scalar output key, scalar_accum_step key) for metrics that
                # are a direct 1:1 copy at the end of the minibatch step — i.e. every
                # entry that isn't scaled by a loss coefficient or read from a
                # variable outside scalar_accum_step (those stay as explicit lines
                # below since this table only covers the pure-rename case).
                _direct_metrics = (
                    ("loss/total", "loss"),
                    ("loss/policy_gradient", "pg"),
                    ("loss/value", "vf"),
                    ("loss/entropy", "ent"),
                    ("loss/behavioral_cloning", "bc"),
                    ("loss/behavioral_cloning_kl", "bc_kl"),
                    ("loss/scripted_entropy", "scripted_entropy"),
                    ("loss/sigreg", "sigreg"),
                    ("loss/next_state", "ns_loss"),
                    ("loss/next_state_cont", "ns_cont"),
                    ("loss/windowed_ns", "windowed_ns"),
                    ("policy/kl", "kl"),
                    ("policy/clip_fraction", "clip"),
                    ("policy/ratio_mean", "ratio_mean"),
                    ("policy/ratio_max", "ratio_max"),
                    ("policy/entropy_power", "entropy_power"),
                    ("policy/entropy_turn", "entropy_turn"),
                    ("policy/entropy_shoot", "entropy_shoot"),
                    ("returns/aggregate", "ret_agg_mean"),
                    ("returns/aggregate_std", "ret_agg_std"),
                    ("episode/alive_fraction", "alive_frac"),
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
                    mb_envs = sum(chunk.alive.shape[1] for chunk in chunks)
                    ratio_max = _z.clone()
                    ret_agg_mean = _z.clone()
                    ret_agg_sq = _z.clone()

                    for source_chunk, device_chunk in self._iter_device_chunks(chunks, buf):
                        frac = source_chunk.alive.shape[1] / mb_envs
                        loss, diag = self._compute_minibatch_loss(
                            device_chunk,
                            is_primary,
                            denoms,
                            frac,
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
                                    diag[kk] if kk not in k_stats else k_stats[kk] + diag[kk]
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

                params = list(self._policy_module.parameters())
                grad_norm = nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                # One inf/NaN gradient element makes the total norm non-finite,
                # and clip_grad_norm_ then scales every gradient by
                # max_norm/inf == 0 — turning that element into NaN (inf * 0)
                # while zeroing all the others. Adam folds the NaN into exp_avg,
                # so the parameter stays NaN for the rest of the run and the
                # policy emits NaN logits until something samples them and the
                # CUDA multinomial assert fires, far from the real cause.
                # Scrubbing degrades the bad micro-batch to a no-op step.
                # The norm is finite only if every gradient is, so this is a
                # no-op on healthy steps. Kept on-device: the flag rides along
                # with the other metrics rather than forcing a host sync here.
                nonfinite_grad = ~torch.isfinite(grad_norm)
                for param in params:
                    if param.grad is not None:
                        torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                self.optim.step()

                for out_key, short_key in _direct_metrics:
                    accum_scalar[out_key].append(scalar_accum_step[short_key])
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
                accum_scalar["returns/advantage_std"].append(scalar_accum_step["adv_var"] ** 0.5)
                accum_scalar["train/gradient_norm"].append(grad_norm.detach())
                accum_scalar["train/nonfinite_grad_fraction"].append(nonfinite_grad.float())

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
                    accum_k["returns/advantage_std"].append(stats_k_cpu[5].clamp(min=0.0).sqrt())
                    if epoch_idx == num_epochs - 1:
                        accum_k["critic/explained_variance"].append(stats_k_cpu[1])

                if ns_feat_step is not None:
                    ns_per_feat_accum.append(ns_feat_step.cpu())

                if record_histograms and "alive_flat" in hist_diag:
                    # Sampled from the last micro-batch of the last primary
                    # minibatch — a large-enough sample for the histograms.
                    alive_flat = hist_diag["alive_flat"]
                    # returns are bf16-stored; upcast before numpy (no bf16 dtype there).
                    last_returns_np = (
                        hist_diag["mb_returns"].reshape(-1, K)[alive_flat].float().cpu().numpy()
                    )
                    last_logprob_np = hist_diag["logprob_flat"][alive_flat].cpu().numpy()

            if target_kl is not None:
                epoch_kls = accum_scalar["policy/kl"][kl_start:]
                if epoch_kls and torch.stack(epoch_kls).mean().item() > target_kl:
                    break

        metrics: dict = {k: torch.stack(v).mean().item() for k, v in accum_scalar.items() if v}
        metrics["train/epochs_completed"] = float(epoch_idx + 1)

        for key, tensors in accum_k.items():
            if not tensors:
                continue
            avg = torch.stack(tensors).mean(0)  # (K,) CPU
            prefix = "returns" if key == "returns/component" else key
            for i, name in enumerate(self._active_names):
                metrics[f"{prefix}/{name}"] = avg[i].item()

        if ns_per_feat_accum:
            avg_per_feat = torch.stack(ns_per_feat_accum).mean(0)  # (pred_dim,) CPU
            for i, name in enumerate(_NS_FEAT_NAMES):
                metrics[f"next_state/{name}"] = avg_per_feat[i].item()

        if last_returns_np is not None:
            metrics["hist/returns"] = last_returns_np
            metrics["hist/logprob"] = last_logprob_np

        return metrics

    # ------------------------------------------------------------------
    # Elo evaluation
    # ------------------------------------------------------------------

    def _random_elo(self) -> float:
        """Return the current Elo of the random anchor roster entry."""
        for e in self.roster.entries:
            if e.kind == "random":
                return e.elo
        return 0.0  # fallback; random entry should always exist

    def _ladder_eval_state(
        self,
    ) -> tuple[list[LadderOpponent], LadderOpponent | None]:
        """Build the evaluator's (anchors, floating) ladder state from the roster.

        Loads anchor and floating checkpoint policies from disk (resume path);
        a None policy stands for the random agent.
        """

        def _opponent(entry: RosterEntry) -> LadderOpponent:
            if entry.kind == "random":
                return LadderOpponent(policy=None, elo=entry.elo, label=entry.label)
            self.roster.load_policy(
                entry,
                self.ship_config,
                self.wrapper.num_ships,
                self.device,
                model_config=self.model_config,
                compile_mode=self._compile_mode,
                team_pma_k=self._win_k,
            )
            return LadderOpponent(
                policy=entry.policy,
                elo=entry.elo,
                label=entry.label,
                reads_bullets=entry.bundle.reads_bullets,
            )

        anchors = [_opponent(entry) for entry in self.roster.ladder_anchors(MAX_ANCHORS)]
        floating_entry = self.roster.floating_checkpoint()
        if floating_entry is None:
            return anchors, None
        return anchors, _opponent(floating_entry)

    def _effective_target_kl(self) -> float | None:
        """Resolve the Elo-gated target KL from the current schedule snapshot."""
        threshold = self._schedule_state.high_elo_threshold
        elo_norm = self._training_elo - self._random_elo()
        if threshold is not None and elo_norm >= threshold:
            return self._schedule_state.high_elo_target_kl
        return self._schedule_state.target_kl
