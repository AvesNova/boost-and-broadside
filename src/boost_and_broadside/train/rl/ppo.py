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
from typing import Callable

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


def _override_opponent(
    action: torch.Tensor,
    team_id: torch.Tensor,
    opp_team_flag: torch.Tensor,
    start: int,
    end: int,
    opp_action: torch.Tensor,
) -> None:
    """Replace team-opp_team_flag actions in envs [start, end) with opp_action in-place.

    Args:
        action:        (B, N, 3) combined action tensor — modified in-place.
        team_id:       (B, N) int — team assignment per ship.
        opp_team_flag: (end-start,) int — which team_id is the opponent per env.
        start, end:    slice of envs to update.
        opp_action:    (end-start, N, 3) — opponent agent's actions.
    """
    opp_mask = team_id[start:end] == opp_team_flag.unsqueeze(1)  # (slice, N)
    action[start:end] = torch.where(
        opp_mask.unsqueeze(-1), opp_action, action[start:end]
    )


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
    allow_avg_model_updates: bool
    allow_scripted_in_roster: bool
    elo_eval_games: int
    elo_eval_interval: int
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
        allow_avg_model_updates=schedule.allow_avg_model_updates(step),
        allow_scripted_in_roster=schedule.allow_scripted_in_roster(step),
        elo_eval_games=schedule.elo_eval_games(step),
        elo_eval_interval=schedule.elo_eval_interval(step),
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
            model_config, self.coordinator, num_value_components=K, num_ships=N
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

        # --- Avg-model opponent (uniform mean of all post-warmup policy snapshots) ---
        # Weights initialized as a copy of the training policy.
        # Only updated when allow_avg_model_updates is True in the current phase.
        self._avg_policy_module = MVPPolicy(
            model_config, self.coordinator, num_value_components=K, num_ships=N
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
                _h_avg = self._avg_policy_module.initial_hidden(B, _nt, self.device)
                self.avg_policy.get_action_and_value(sample_obs, _h_avg)

        # Per-env flag: which team_id is the opponent in scripted/avg/league groups.
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
        self._eval_window_rand = deque(maxlen=100)
        self._eval_window_sc = deque(maxlen=100)
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

        # BC decay: None until ELO first reaches 1000, then records that step.
        # Factor = 1.0 before milestone, 0.1**(steps_since/10M) for next 20M steps, then 0.
        self._bc_1000_elo_step: int | None = None

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

    def train(self) -> None:
        """Run the full PPO training loop."""
        B = self.cfg.scales[0].num_envs
        N = self.wrapper.num_ships
        M = self.env_config.num_obstacles
        num_tokens = N + M  # ships + obstacles; hidden state covers all entity tokens
        
        # -- ELO Evaluation Env & State Initialization (Parallel Vectorized Slots) --
        B_eval = 128
        eval_env = TensorEnv(
            B_eval,
            self.ship_config,
            self.env_config,
            self.device,
            self._obstacle_cache,
        )
        eval_obs = eval_env.reset()
        eval_env.state.step_count.random_(0, self.env_config.max_episode_steps)
        
        # Load and resolve agents for ELO evaluation
        from boost_and_broadside.modes.agent_factory import ResolvedAgent, get_actions, init_hidden, reset_done_envs
        from boost_and_broadside.modes.collect import _obs_from_state
        
        agent_policy = ResolvedAgent("policy", self.policy)
        agent_sc = ResolvedAgent("scripted", self.scripted_agent) if self.scripted_agent else None
        agent_rand = ResolvedAgent("random", None)
        
        init_hidden(agent_policy, B_eval, num_tokens, self.device)
        if agent_sc is not None:
            init_hidden(agent_sc, B_eval, num_tokens, self.device)
        init_hidden(agent_rand, B_eval, num_tokens, self.device)
        
        # Optimal information-proportional routing variables
        K_eval = 4.0
        f_star = _compute_optimal_eval_ratio(self._training_elo)
        eval_is_scripted = (torch.rand(B_eval, device=self.device) > f_star)
        


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
        hidden    = self.policy.initial_hidden(B, num_tokens, self.device)
        hidden_t1 = self.policy.initial_hidden(B, num_tokens, self.device)

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
        aux_hidden_t1s: list[torch.Tensor] = []
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
                self.policy.initial_hidden(
                    sc.num_envs, aux_num_tokens, self.device
                )
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

        start_time = time.time()

        for update in range(self._start_update, self._num_updates + 1):
            self.buffer.reset()
            self.buffer.store_initial_hidden(hidden)
            for aux_buf, aux_h in zip(self.aux_buffers, aux_hiddens):
                aux_buf.reset()
                aux_buf.store_initial_hidden(aux_h)

            # Accumulate episode stats across the rollout — flushed once per update
            ep_rewards: list[torch.Tensor] = []
            ep_lengths: list[torch.Tensor] = []
            ep_components: dict[str, list[torch.Tensor]] = {}
            ep_scaled_components: dict[str, list[torch.Tensor]] = {}
            ep_wins: list[torch.Tensor] = []
            ep_lifespans: list[torch.Tensor] = []

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
            for _ in range(self.cfg.num_steps):
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
                            # Pass 1: team 0 perspective — training pass (logprob/value stored)
                            action_t0, logprob, value_norm, pred_next, hidden = (
                                self.policy.get_action_and_value(obs, hidden)
                            )
                            # Pass 2: team 1 perspective — action generation only
                            with torch.no_grad():
                                obs_t1 = _flip_team_obs(obs, N)
                                action_t1, _, _, _, hidden_t1 = (
                                    self.policy.get_action_and_value(obs_t1, hidden_t1)
                                )
                        if use_avg:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                obs_avg = _flip_team_obs(_slice_obs(obs, avg_start, avg_end), N)
                                action_avg, _, _, _, avg_hidden = (
                                    self.avg_policy.get_action_and_value(
                                        obs_avg, avg_hidden
                                    )
                                )
                        if use_league and self._current_league_policy is not None:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                obs_league = _flip_team_obs(_slice_obs(obs, league_start, league_end), N)
                                action_league_net, _, _, _, league_hidden = (
                                    self._current_league_policy.get_action_and_value(
                                        obs_league, league_hidden
                                    )
                                )
                        else:
                            action_league_net = None

                    torch.cuda.synchronize()
                else:
                    # CPU fallback (no streams)
                    next_obs, reward, dones, truncated, info = self.wrapper.step(
                        action_buffer
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        # Pass 1: team 0 perspective — training pass
                        action_t0, logprob, value_norm, pred_next, hidden = (
                            self.policy.get_action_and_value(obs, hidden)
                        )
                        # Pass 2: team 1 perspective — action generation only
                        with torch.no_grad():
                            obs_t1 = _flip_team_obs(obs, N)
                            action_t1, _, _, _, hidden_t1 = (
                                self.policy.get_action_and_value(obs_t1, hidden_t1)
                            )
                    if use_avg:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            obs_avg = _flip_team_obs(_slice_obs(obs, avg_start, avg_end), N)
                            action_avg, _, _, _, avg_hidden = (
                                self.avg_policy.get_action_and_value(obs_avg, avg_hidden)
                            )
                    if use_league and self._current_league_policy is not None:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            obs_league = _flip_team_obs(_slice_obs(obs, league_start, league_end), N)
                            action_league_net, _, _, _, league_hidden = (
                                self._current_league_policy.get_action_and_value(
                                    obs_league, league_hidden
                                )
                            )
                    else:
                        action_league_net = None

                # -- Phase 3: combine actions and compute actor mask --
                # team 0 ships always use Pass 1 (training pass); team 1 always use Pass 2.
                team0_mask = (team_id == 0)  # (B, N)
                action = torch.where(team0_mask.unsqueeze(-1), action_t0, action_t1)

                # Override team 1 ships in opponent envs with actual opponent actions.
                # Opponents always play team 1 in the new paradigm.
                if self._policy_gradient_coef != 0.0:
                    action_league = (
                        action_league_scripted
                        if action_league_scripted is not None
                        else action_league_net
                    )
                    if use_sc_opponent:
                        opp_mask = team_id[sc_start:sc_end] == 1
                        action[sc_start:sc_end] = torch.where(
                            opp_mask.unsqueeze(-1), action_scripted, action[sc_start:sc_end]
                        )
                    if use_avg:
                        opp_mask = team_id[avg_start:avg_end] == 1
                        action[avg_start:avg_end] = torch.where(
                            opp_mask.unsqueeze(-1), action_avg, action[avg_start:avg_end]
                        )
                    if use_league:
                        opp_mask = team_id[league_start:league_end] == 1
                        action[league_start:league_end] = torch.where(
                            opp_mask.unsqueeze(-1), action_league, action[league_start:league_end]
                        )

                # Actor mask: only team 0 ships train (Pass 1 = training pass).
                actor_mask = team0_mask

                # -- Phase 4: inject decided action into next obs as prev_action --
                # obs(t+1).previous_action = action(t) — what the policy just decided,
                # will be executed by env.step next iteration.
                next_obs[ObsKey.PREVIOUS_ACTION][:, :N] = action

                if info.get("ep_reward") is not None:
                    ep_rewards.append(info["ep_reward"])
                    ep_lengths.append(info["ep_length"].float())
                    for name, t in info["ep_reward_components"].items():
                        ep_components.setdefault(name, []).append(t)
                    for name, t in info["ep_scaled_reward_components"].items():
                        ep_scaled_components.setdefault(name, []).append(t)
                    ep_wins.append(info["ep_wins"])
                    if "ep_lifespan" in info:
                        ep_lifespans.append(info["ep_lifespan"])

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
                hidden    = self.policy.reset_hidden_for_envs(hidden,    done_any, num_tokens)
                hidden_t1 = self.policy.reset_hidden_for_envs(hidden_t1, done_any, num_tokens)
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

                obs = next_obs
                self._global_step += B

                # Aux-scale rollout steps (pure self-play, 1-step delay)
                for i, (sc, aux_w, aux_buf) in enumerate(
                    zip(self.cfg.scales[1:], self.aux_wrappers, self.aux_buffers)
                ):
                    aux_N = sc.env_config.num_ships
                    aux_num_tokens = aux_N + sc.env_config.num_obstacles
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        # Pass 1: team 0 perspective — training pass
                        aux_action_t0, aux_logprob, aux_value_norm, _, aux_hiddens[i] = (
                            self.policy.get_action_and_value(aux_obs[i], aux_hiddens[i])
                        )
                        # Pass 2: team 1 perspective — action generation only
                        aux_obs_t1 = _flip_team_obs(aux_obs[i], aux_N)
                        aux_action_t1, _, _, _, aux_hidden_t1s[i] = (
                            self.policy.get_action_and_value(aux_obs_t1, aux_hidden_t1s[i])
                        )
                    aux_team0 = (aux_obs[i]["team_id"][:, :aux_N] == 0)
                    aux_action = torch.where(aux_team0.unsqueeze(-1), aux_action_t0, aux_action_t1)
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
                        actor_mask=(aux_obs[i]["team_id"][:, :aux_N] == 0),
                        expert_probs=None,
                        terminated=aux_done_any,
                    )
                    aux_hiddens[i] = self.policy.reset_hidden_for_envs(
                        aux_hiddens[i], aux_done_any, aux_num_tokens
                    )
                    aux_hidden_t1s[i] = self.policy.reset_hidden_for_envs(
                        aux_hidden_t1s[i], aux_done_any, aux_num_tokens
                    )
                    aux_action_buffers[i] = aux_action.detach().clone()
                    aux_action_buffers[i][aux_done_any] = 0
                    aux_last_dones[i] = aux_dones
                    aux_obs[i] = next_aux_obs
                    self._global_step += sc.num_envs

                # -- Continuous Live ELO Evaluation --
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    eval_state = eval_env.state
                    eval_obs_obj = _obs_from_state(eval_state, self.ship_config)
                    eval_action_policy = get_actions(agent_policy, eval_obs_obj, eval_state, B_eval, N, self.device)
                    
                    if agent_sc is not None:
                        eval_action_scripted = get_actions(agent_sc, eval_obs_obj, eval_state, B_eval, N, self.device)
                    else:
                        eval_action_scripted = torch.zeros_like(eval_action_policy)
                        
                    eval_action_random = get_actions(agent_rand, eval_obs_obj, eval_state, B_eval, N, self.device)
                    
                    eval_action_opp = torch.where(
                        eval_is_scripted.unsqueeze(-1).unsqueeze(-1),
                        eval_action_scripted,
                        eval_action_random
                    )
                    
                    eval_team_id = eval_state.ship_team_id
                    eval_action = torch.where(
                        (eval_team_id == 0).unsqueeze(-1),
                        eval_action_policy,
                        eval_action_opp
                    )
                    
                    eval_dones, eval_truncated = eval_env.step(eval_action)
                    eval_done_any = eval_dones | eval_truncated
                    
                    if eval_done_any.any():
                        eval_alive = eval_env.state.ship_alive
                        eval_team = eval_env.state.ship_team_id
                        eval_team0_alive = (eval_alive & (eval_team == 0)).any(dim=1)
                        eval_team1_alive = (eval_alive & (eval_team == 1)).any(dim=1)
                        
                        eval_team1_won = eval_done_any & eval_team1_alive & ~eval_team0_alive
                        eval_team0_won = eval_done_any & eval_team0_alive & ~eval_team1_alive
                        eval_tied = eval_done_any & ~eval_team0_won & ~eval_team1_won
                        
                        finished_indices = torch.where(eval_done_any)[0].cpu().tolist()
                        for idx in finished_indices:
                            opp_is_scripted = eval_is_scripted[idx].item()
                            opp_elo = 1000.0 if opp_is_scripted else 0.0
                            
                            if eval_team0_won[idx]:
                                score = 1.0
                            elif eval_tied[idx]:
                                score = 0.5
                            else:
                                score = 0.0
                                
                            expected = 1.0 / (1.0 + 10.0 ** ((opp_elo - self._training_elo) / 400.0))
                            delta = K_eval * (score - expected)
                            self._training_elo += delta
                            
                            # Dynamic Information-Proportional Re-routing
                            f_star = _compute_optimal_eval_ratio(self._training_elo)
                            eval_is_scripted[idx] = (torch.rand(1).item() > f_star)
                            
                            if opp_is_scripted:
                                self._eval_window_sc.append(score)
                            else:
                                self._eval_window_rand.append(score)
                                
                        eval_env.reset_envs(eval_done_any)
                        reset_done_envs(agent_policy, eval_done_any, num_tokens)
                        if agent_sc is not None:
                            reset_done_envs(agent_sc, eval_done_any, num_tokens)
                        reset_done_envs(agent_rand, eval_done_any, num_tokens)

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
            # BC decay: full until ELO reaches 1000, then exponential decay to zero.
            elo_norm = self._training_elo - self._random_elo()
            if self._bc_1000_elo_step is None and elo_norm >= 900.0:
                self._bc_1000_elo_step = self._global_step
            if self._bc_1000_elo_step is None:
                bc_factor = 1.0
            else:
                steps_since = self._global_step - self._bc_1000_elo_step
                if steps_since >= 20_000_000:
                    bc_factor = 0.0
                else:
                    bc_factor = 0.1 ** (steps_since / 10_000_000)
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
            metrics["schedule/learning_rate"] = self._schedule_state.learning_rate
            metrics["schedule/policy_gradient_coef"] = (
                self._schedule_state.policy_gradient_coef
            )
            metrics["schedule/behavior_cloning_coef"] = self._behavior_cloning_coef
            metrics["schedule/bc_decay_factor"] = bc_factor
            metrics["schedule/target_kl"] = 0.02 if self._bc_1000_elo_step is not None else self._schedule_state.target_kl
            metrics["schedule/true_reward_scale"] = (
                self._schedule_state.true_reward_scale
            )
            metrics["schedule/global_scale"] = self._schedule_state.global_scale
            metrics["schedule/local_scale"] = self._schedule_state.local_scale

            if self._policy_gradient_coef > 0.0:
                # Update avg model when allowed by the current phase.
                # The timeline's allow_avg_model_updates is the sole gate — no min_steps.
                avg_model_ready = self._schedule_state.allow_avg_model_updates
                if self.B_avg > 0 and avg_model_ready:
                    self._update_avg_model()

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

            # Merge episode stats collected during rollout into the metrics dict
            if ep_rewards:
                all_rewards = torch.cat(ep_rewards)  # (num_finished_eps * N,)
                all_lengths = torch.cat(ep_lengths)
                metrics["episode/reward_mean"] = all_rewards.mean().item()
                metrics["episode/reward_min"] = all_rewards.min().item()
                metrics["episode/reward_max"] = all_rewards.max().item()
                metrics["episode/length_mean"] = all_lengths.mean().item()
                for name, tensors in ep_components.items():
                    metrics[f"episode/reward_{name}"] = torch.cat(tensors).mean().item()
                for name, tensors in ep_scaled_components.items():
                    metrics[f"episode/scaled_{name}"] = torch.cat(tensors).mean().item()
                if ep_wins:
                    metrics["episode/win_rate"] = torch.cat(ep_wins).mean().item()
                if ep_lifespans:
                    metrics["episode/lifespan_mean"] = torch.cat(ep_lifespans).mean().item()

            sps = int(self._global_step / (time.time() - start_time))
            metrics["train/global_step"] = self._global_step
            metrics["train/sps"] = sps

            # ELO evaluation — continuous live statistics from parallel slots
            metrics["elo/training"] = self._training_elo
            if self._eval_window_rand:
                metrics["elo/win_rate_vs_random"] = sum(self._eval_window_rand) / len(self._eval_window_rand)
            if self._eval_window_sc:
                metrics["elo/win_rate_vs_scripted"] = sum(self._eval_window_sc) / len(self._eval_window_sc)

            # Save overwriting best-model checkpoints when normalized ELO improves.
            random_elo = self._random_elo()
            training_elo_norm = self._training_elo - random_elo
            if training_elo_norm > self._best_training_elo_norm:
                self._best_training_elo_norm = training_elo_norm
                self._save_best_checkpoint("best_training.pt")

            # Overview — redundant copies of the most important global metrics
            for src, dst in [
                ("elo/training",                   "overview/elo"),
                ("elo/win_rate_vs_scripted",        "overview/win_rate_vs_scripted"),
                ("elo/win_rate_vs_random",          "overview/win_rate_vs_random"),
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

    def _compute_minibatch_loss(
        self,
        batch: tuple,
        comp_weights: torch.Tensor,
        is_primary: bool,
    ) -> tuple[torch.Tensor, dict]:
        """Compute PPO loss for one minibatch. Does NOT call zero_grad / backward / step.

        Loss coefficients are read from ``self._policy_gradient_coef``, ``self._behavior_cloning_coef``,
        and ``self._schedule_state`` (``value_function_coef``, ``entropy_coef``) which are updated
        each update step.  Setting ``policy_gradient_coef=0.0`` in the base schedule activates
        BC pretraining mode (no policy gradient or entropy loss).

        Args:
            batch:        Output of RolloutBuffer.get_minibatch_iterator.
            comp_weights: (K,) per-component lambda weights for this update step.
            is_primary:   True for the primary scale — enables BC loss and per-component
                          critic diagnostics. Aux scales skip these to avoid shape mismatches
                          (different N) and because BC targets only exist in the primary env.

        Returns:
            (loss, diag) where diag is a dict of scalar/tensor diagnostics.
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
        mask_sum = alive_f.sum().clamp(min=1.0)

        actor_f = (mb_actor_mask & mb_alive).float()  # (T, B_mb, N)
        actor_sum = actor_f.sum().clamp(min=1.0)

        # ---- Lambda aggregation -------------------------------------------
        # Per-timestep team IDs: correctly tracks re-assignments after mid-rollout resets.
        # Buffer stores obs as float32; cast to long for comparison.
        # Slice to ship tokens only — obstacle tokens (team_id=2) have no rewards/actions.
        N_ships = mb_alive.shape[-1]
        team_id_t = curr_mb_obs["team_id"][:, :, :N_ships].long()  # (T, B_mb, N)
        same_team_t = team_id_t.unsqueeze(3) == team_id_t.unsqueeze(
            2
        )  # (T, B_mb, N_i, N_j)
        N = team_id_t.shape[-1]

        ally_lam = torch.where(
            self.ally_zero_k, 0.0, 1.0
        )  # (K,) — 0 for enemy-only components
        enemy_lam = torch.where(
            self.enemy_neg_k, -1.0, 0.0
        )  # (K,) — -1 for zero-sum components

        # Zero out dead contributing ships (j): dead ships have untrained critic values
        # and must not contaminate surviving ships' aggregated advantages.
        alive_j = mb_alive.float().unsqueeze(2).unsqueeze(-1)  # (T, B_mb, 1, N_j, 1)

        # Global lambda (team-based): allies share signals, enemies are zero-sum.
        global_lambda = (
            same_team_t.float().unsqueeze(-1) * ally_lam
            + (~same_team_t).float().unsqueeze(-1) * enemy_lam
        )  # (T, B_mb, N_i, N_j, K)

        # Local lambda (diagonal): ship i only receives its own signal (i==j).
        # Shape (1, 1, N, N, 1) broadcasts across T, B_mb, and K dims.
        identity = torch.eye(N, dtype=torch.float32, device=self.device)
        local_lambda = identity[None, None, :, :, None]  # (1, 1, N, N, 1)

        lambda_ij_t = (
            torch.where(self.local_k, local_lambda, global_lambda)
            * comp_weights
            * alive_j
        )  # (T, B_mb, N_i, N_j, K)

        # Normalize each ship i's lambda weights by the sum of absolute contributions
        # across alive ships j. This makes the aggregated signal a weighted mean rather
        # than a sum, so the policy gradient magnitude is consistent regardless of how
        # many ships are alive at each timestep and comparable across game sizes (N).
        # Local components (diagonal lambda) always sum to 1.0 so are unaffected.
        # clamp(min=1.0) handles the degenerate case where all contributing ships are dead.
        lambda_norm = lambda_ij_t.abs().sum(dim=3, keepdim=True).clamp(min=1.0)
        lambda_ij_t = lambda_ij_t / lambda_norm  # (T, B_mb, N_i, N_j, K)

        mb_advantages_normed = self.adv_scaler.normalize(
            mb_advantages
        )  # (T, B_mb, N, K)
        adv_agg = torch.einsum(
            "tbijk,tbjk->tbi", lambda_ij_t, mb_advantages_normed
        )  # (T, B_mb, N)
        ret_agg = torch.einsum(
            "tbijk,tbjk->tbi", lambda_ij_t, mb_returns
        )  # (T, B_mb, N)
        ret_per_comp = torch.einsum(
            "tbijk,tbjk->tbik", lambda_ij_t, mb_returns
        )  # (T, B_mb, N, K)

        adv_rms = (adv_agg.pow(2) * actor_f).sum() / actor_sum
        adv_norm = adv_agg / (adv_rms.sqrt().clamp(min=0.1) + 1e-8)

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
            bc_sum = bc_f.sum().clamp(min=1.0)
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
        sigreg_loss = self._zero_tensor
        if need_sigreg:
            T_mb, B_mb, N_mb, D_mb = z.shape
            z_flat = z.reshape(T_mb, B_mb * N_mb, D_mb)  # (T, B*N, D)
            sigreg_loss = self.sigreg(z_flat)

        # ---- Next-state prediction loss (primary scale only) ----------------
        next_state_loss = self._zero_tensor
        next_state_cont_loss = self._zero_tensor
        windowed_ns_loss = self._zero_tensor
        next_state_per_feat: torch.Tensor | None = None  # (pred_dim,) cpu, for logging
        _need_aux = is_primary and (
            self.cfg.next_state_coef > 0.0 or self.cfg.windowed_loss_coef > 0.0
        )
        if _need_aux:
            non_terminal = ~mb_terminated.unsqueeze(-1)  # (T, B_mb, 1)
            ns_mask = mb_alive & non_terminal             # (T, B_mb, N)
            ns_mask_f = ns_mask.float()
            ns_sum = ns_mask_f.sum().clamp(min=1.0)

            # Compute aux labels from T+1 observation storage.
            # obs[0:T] = current, obs[1:T+1] = next — same layout as the stored buffer.
            T_flat = mb_alive.shape[0]
            B_flat = mb_alive.shape[1]
            N_ships_aux = mb_alive.shape[-1]
            next_mb_obs = mb_obs.slice_time(1, T_flat + 1)  # (T, B_mb, N+M, ...)

            # Flatten T and B for batch processing through the coordinator.
            def _flat_ship(o: MVPObservation) -> MVPObservation:
                return MVPObservation(data={
                    k: (v[:, :, :N_ships_aux].reshape(T_flat * B_flat, N_ships_aux, *v.shape[3:])
                        if v.dim() > 3
                        else v[:, :, :N_ships_aux].reshape(T_flat * B_flat, N_ships_aux))
                    for k, v in o.items()
                })

            curr_flat = _flat_ship(curr_mb_obs)  # (T*B, N, ...)
            next_flat = _flat_ship(next_mb_obs)  # (T*B, N, ...)

            curr_targets = self.coordinator.get_target_vector(curr_flat)  # (T*B, N, target_dim)
            next_targets = self.coordinator.get_target_vector(next_flat)  # (T*B, N, target_dim)
            labels = self.coordinator.compute_labels(curr_targets, next_targets)  # (T*B, N, pred_dim)
            labels = labels.reshape(T_flat, B_flat, N_ships_aux, -1)  # (T, B, N, pred_dim)

            P = self.coordinator.total_prediction_dimension
            sq_err = (pred_next.float() - labels.detach()).pow(2)  # (T, B, N, pred_dim)
            sq_err = sq_err * self.aux_weights  # per-prediction weight

            if self.cfg.next_state_coef > 0.0:
                next_state_cont_loss = (
                    sq_err * ns_mask_f.unsqueeze(-1)
                ).sum() / (ns_sum * P)
                next_state_loss = next_state_cont_loss

            if self.cfg.windowed_loss_coef > 0.0:
                windowed_ns_loss = self.coordinator.compute_windowed_loss(
                    pred_next.float(),
                    labels.detach(),
                    ns_mask,
                    mb_terminated,
                )

            with torch.no_grad():
                per_feat_cont = (sq_err * ns_mask_f.unsqueeze(-1)).sum((0, 1, 2)) / ns_sum  # (pred_dim,)
                next_state_per_feat = per_feat_cont.cpu()

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
            diag["next_state_per_feat"] = next_state_per_feat  # (16,) cpu or None
            diag["scripted_entropy"] = scripted_entropy.detach()
            diag["bc_kl"] = bc_loss.detach() - scripted_entropy.detach()
            diag["adv_var"] = adv_rms
            diag["approx_kl"] = (((ratio - 1) - log_ratio) * actor_f).sum() / actor_sum
            diag["clip_frac"] = (
                ((ratio - 1).abs() > cfg.clip_coef).float() * actor_f
            ).sum() / actor_sum
            diag["alive_frac"] = alive_f.mean()
            diag["ratio_mean"] = (ratio * actor_f).sum() / actor_sum
            diag["ratio_max"] = ratio.max()

            # Per-head entropy — recomputed from policy_logits (already returned by evaluate_actions)
            power_ent = Categorical(logits=policy_logits[..., POWER_SLICE]).entropy()
            turn_ent = Categorical(logits=policy_logits[..., TURN_SLICE]).entropy()
            shoot_ent = Categorical(logits=policy_logits[..., SHOOT_SLICE]).entropy()
            diag["entropy_power"] = (power_ent * actor_f).sum() / actor_sum
            diag["entropy_turn"] = (turn_ent * actor_f).sum() / actor_sum
            diag["entropy_shoot"] = (shoot_ent * actor_f).sum() / actor_sum

            ret_agg_mean = (ret_agg * actor_f).sum() / actor_sum
            ret_agg_var = ((ret_agg - ret_agg_mean).pow(2) * actor_f).sum() / actor_sum
            diag["ret_agg_mean"] = ret_agg_mean
            diag["ret_agg_std"] = ret_agg_var.sqrt()

            # Per-component critic stats — primary scale only (K matches buffer.num_components)
            if is_primary:
                actor_k = actor_f.unsqueeze(-1)  # (T, B_mb, N, 1)
                ret_per_comp_mean_k = (ret_per_comp * actor_k).sum(
                    (0, 1, 2)
                ) / actor_sum  # (K,)

                pred_k = self.scaler.denormalize(new_value.detach())  # (T, B_mb, N, K)
                value_loss_k = (vf_loss_raw.detach() * alive_k).sum(
                    (0, 1, 2)
                ) / mask_sum  # (K,)
                ret_mean_k = (mb_returns * alive_k).sum((0, 1, 2)) / mask_sum  # (K,)
                ret_var_k = ((mb_returns - ret_mean_k) ** 2 * alive_k).sum(
                    (0, 1, 2)
                ) / mask_sum
                residuals_k = mb_returns - pred_k  # (T, B_mb, N, K)
                res_mean_k = (residuals_k * alive_k).sum((0, 1, 2)) / mask_sum  # (K,)
                res_var_k = ((residuals_k - res_mean_k) ** 2 * alive_k).sum(
                    (0, 1, 2)
                ) / mask_sum
                ev_k = 1.0 - res_var_k / (ret_var_k + 1e-8)  # (K,)
                pred_mean_k = (pred_k * alive_k).sum((0, 1, 2)) / mask_sum  # (K,)
                # One GPU→CPU transfer: stack → (5, K) → cpu
                diag["stats_k_cpu"] = torch.stack(
                    [value_loss_k, ev_k, ret_mean_k, ret_per_comp_mean_k, pred_mean_k]
                ).cpu()
                # Per-component advantage std — raw, unweighted, un-aggregated
                adv_var_k = (mb_advantages.pow(2) * alive_k).sum((0, 1, 2)) / mask_sum
                diag["adv_std_k"] = adv_var_k.sqrt().cpu()  # (K,)
                diag["alive_flat"] = mb_alive.reshape(-1).bool()
                diag["mb_returns"] = mb_returns
                diag["logprob_flat"] = logprob.detach().float().reshape(-1)

        return loss, diag

    def _update_epochs(
        self,
        all_buffers: list[RolloutBuffer],
        record_histograms: bool = False,
    ) -> dict:
        """Run num_epochs × num_minibatches of PPO updates across all scales.

        Gradients from every scale are accumulated before each optimizer step so
        that each parameter update reflects all game sizes simultaneously.

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
        target_kl = 0.02 if self._bc_1000_elo_step is not None else self._schedule_state.target_kl

        for epoch_idx in range(num_epochs):
            kl_start = len(accum_scalar["policy/kl"])
            iters = [
                buf.get_minibatch_iterator(cfg.num_minibatches) for buf in all_buffers
            ]
            for batches in zip(*iters):
                self.optim.zero_grad()

                # Accumulate gradients across all scales before stepping.
                # Each loss is divided by n_scales so the total gradient magnitude
                # stays comparable to single-scale training.
                diag_primary: dict = {}
                _z = torch.zeros((), device=self.device)
                scalar_accum_step: dict[str, torch.Tensor] = {
                    "loss": _z.clone(),
                    "pg": _z.clone(),
                    "vf": _z.clone(),
                    "ent": _z.clone(),
                    "bc": _z.clone(),
                    "sigreg": _z.clone(),
                    "ns_loss": _z.clone(),
                    "ns_cont": _z.clone(),
                    "windowed_ns": _z.clone(),
                    "bc_kl": _z.clone(),
                    "scripted_entropy": _z.clone(),
                    "kl": _z.clone(),
                    "clip": _z.clone(),
                    "adv_var": _z.clone(),
                    "ret_agg_mean": _z.clone(),
                    "ret_agg_std": _z.clone(),
                    "alive_frac": _z.clone(),
                    "ratio_mean": _z.clone(),
                    "ratio_max": _z.clone(),
                    "entropy_power": _z.clone(),
                    "entropy_turn": _z.clone(),
                    "entropy_shoot": _z.clone(),
                }

                for scale_idx, (buf, batch) in enumerate(zip(all_buffers, batches)):
                    is_primary = scale_idx == 0
                    loss, diag = self._compute_minibatch_loss(
                        batch, comp_weights, is_primary
                    )
                    (loss / n_scales).backward()

                    # Accumulate scalar diagnostics (average across scales)
                    scalar_accum_step["loss"] += diag["loss"] / n_scales
                    scalar_accum_step["pg"] += diag["pg_loss"] / n_scales
                    scalar_accum_step["vf"] += diag["vf_loss"] / n_scales
                    scalar_accum_step["ent"] += diag["ent_loss"] / n_scales
                    scalar_accum_step["bc"] += diag["bc_loss"] / n_scales
                    scalar_accum_step["sigreg"] += diag["sigreg_loss"] / n_scales
                    scalar_accum_step["ns_loss"] += diag["next_state_loss"] / n_scales
                    scalar_accum_step["ns_cont"] += diag["next_state_cont_loss"] / n_scales
                    scalar_accum_step["windowed_ns"] += diag["windowed_ns_loss"] / n_scales
                    scalar_accum_step["bc_kl"] += diag["bc_kl"] / n_scales
                    scalar_accum_step["scripted_entropy"] += (
                        diag["scripted_entropy"] / n_scales
                    )
                    scalar_accum_step["kl"] += diag["approx_kl"] / n_scales
                    scalar_accum_step["clip"] += diag["clip_frac"] / n_scales
                    scalar_accum_step["adv_var"] += diag["adv_var"] / n_scales
                    scalar_accum_step["ret_agg_mean"] += diag["ret_agg_mean"] / n_scales
                    scalar_accum_step["ret_agg_std"] += diag["ret_agg_std"] / n_scales
                    scalar_accum_step["alive_frac"] += diag["alive_frac"] / n_scales
                    scalar_accum_step["ratio_mean"] += diag["ratio_mean"] / n_scales
                    scalar_accum_step["ratio_max"] += diag["ratio_max"] / n_scales
                    scalar_accum_step["entropy_power"] += (
                        diag["entropy_power"] / n_scales
                    )
                    scalar_accum_step["entropy_turn"] += diag["entropy_turn"] / n_scales
                    scalar_accum_step["entropy_shoot"] += (
                        diag["entropy_shoot"] / n_scales
                    )

                    if is_primary:
                        diag_primary = diag

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

                if "stats_k_cpu" in diag_primary:
                    stats_k_cpu = diag_primary["stats_k_cpu"]
                    accum_k["critic/value_loss"].append(stats_k_cpu[0])
                    accum_k["critic/return_mean"].append(stats_k_cpu[2])
                    accum_k["returns/component"].append(stats_k_cpu[3])
                    accum_k["critic/value_pred_mean"].append(stats_k_cpu[4])
                    if epoch_idx == num_epochs - 1:
                        accum_k["critic/explained_variance"].append(stats_k_cpu[1])

                if "adv_std_k" in diag_primary:
                    accum_k["returns/advantage_std"].append(diag_primary["adv_std_k"])

                if diag_primary.get("next_state_per_feat") is not None:
                    ns_per_feat_accum.append(diag_primary["next_state_per_feat"])

                if record_histograms and "alive_flat" in diag_primary:
                    alive_flat = diag_primary["alive_flat"]
                    last_returns_np = (
                        diag_primary["mb_returns"]
                        .reshape(-1, K)[alive_flat]
                        .cpu()
                        .numpy()
                    )
                    last_logprob_np = (
                        diag_primary["logprob_flat"][alive_flat].cpu().numpy()
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
            "avg_param_cumsum": [c.cpu() for c in self._avg_param_cumsum],
            "avg_update_count": self._avg_update_count,
            "update": update,
            "global_step": self._global_step,
            "training_elo": self._training_elo,
            "eval_window_rand": list(self._eval_window_rand),
            "eval_window_sc": list(self._eval_window_sc),
            "elo_milestone": self._elo_milestone,
            "bc_1000_elo_step": self._bc_1000_elo_step,
            "train_config": {
                k: v for k, v in dataclasses.asdict(self.cfg).items() if k != "schedule"
            },
            "model_config": dataclasses.asdict(self.model_config),
            "env_config": dataclasses.asdict(self.env_config),
        }

    def _save_checkpoint(self, update: int) -> None:
        """Save policy and optimizer state to a .pt file.

        Written to cfg.checkpoint_dir/checkpoint_{update:06d}.pt.
        Directory is created if it does not exist.

        Args:
            update: Current update index (used as filename suffix).
        """
        ckpt_dir = Path(self.cfg.checkpoint_dir) / self._run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{self._global_step:012d}.pt"
        torch.save(self._checkpoint_payload(update), path)
        self._last_checkpoint_path = path
        print(f"Checkpoint saved: {path}")

        # Save most-recent avg model checkpoint when available.
        if self._avg_update_count > 0:
            avg_path = ckpt_dir / "recent_avg.pt"
            torch.save(self._avg_checkpoint_payload(update), avg_path)
            print(f"Recent avg checkpoint saved: {avg_path}")

        # Prune: keep only the latest checkpoint + all roster-referenced files.
        # best_*.pt files are not touched (they don't match the step_*.pt glob).
        kept = self.roster.kept_paths()
        kept.add(str(path))
        for old_path in ckpt_dir.glob("step_*.pt"):
            if str(old_path) not in kept:
                old_path.unlink(missing_ok=True)

    def _avg_checkpoint_payload(self, update: int) -> dict:
        """Build checkpoint payload with avg_policy as the primary policy_state_dict.

        Allows best_avg.pt / recent_avg.pt to be loaded by _load_checkpoint_agent
        in elo_stats.py, which reads ``ckpt["policy_state_dict"]``.
        """
        payload = self._checkpoint_payload(update)
        payload["policy_state_dict"] = self._avg_policy_module.state_dict()
        return payload

    def _save_best_checkpoint(self, name: str, payload: dict | None = None) -> None:
        """Save a named best-model checkpoint, overwriting any previous version.

        Args:
            name:    Filename, e.g. "best_training.pt" or "best_avg.pt".
            payload: Custom payload dict; defaults to _checkpoint_payload(update=0).
        """
        ckpt_dir = Path(self.cfg.checkpoint_dir) / self._run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / name
        torch.save(
            payload if payload is not None else self._checkpoint_payload(update=0),
            path,
        )
        print(f"Best checkpoint saved: {path}")

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
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
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
        if "eval_window_rand" in ckpt:
            self._eval_window_rand = deque(ckpt["eval_window_rand"], maxlen=100)
        if "eval_window_sc" in ckpt:
            self._eval_window_sc = deque(ckpt["eval_window_sc"], maxlen=100)
        if "bc_1000_elo_step" in ckpt:
            self._bc_1000_elo_step = ckpt["bc_1000_elo_step"]
        if "global_step" in ckpt:
            self._global_step = ckpt["global_step"]
            self._start_update = ckpt["update"] + 1

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
