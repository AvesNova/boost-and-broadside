"""Clean recurrent PPO trainer for the MVP policy.

Zero Mamba, zero auxiliary losses. One clean loop:
    collect rollout → compute GAE → update epochs → log async → repeat.

Logging is async (CPU-side via wandb) to avoid GPU sync on the hot path.
"""

import dataclasses
import time
import threading
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
from boost_and_broadside.train.rl.roster import EloRoster, RosterEntry


# ------------------------------------------------------------------
# Opponent-management helpers (module-level, no class coupling)
# ------------------------------------------------------------------


def _slice_obs(
    obs: dict[str, torch.Tensor], start: int, end: int
) -> dict[str, torch.Tensor]:
    """Return a view of obs tensors for envs [start, end)."""
    return {k: v[start:end] for k, v in obs.items()}


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
    )


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
}

# Components that use diagonal lambda (self-only: i==j). These must match the
# "local_scale" entries above. Any component NOT in _LOCAL_COMPONENTS uses the
# standard team-based lambda aggregation.
_LOCAL_COMPONENTS: frozenset[str] = frozenset(
    {
        "facing",
        "closing_speed",
        "shoot_quality",
        "kill_shot",
        "kill_assist",
        "damage_taken",
        "damage_dealt_enemy",
        "damage_dealt_ally",
        "death",
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


def _resolve_schedule(schedule: TrainingSchedule, step: int) -> _ResolvedSchedule:
    """Evaluate every schedule field at ``step`` and return a resolved snapshot."""
    return _ResolvedSchedule(
        learning_rate=schedule.learning_rate(step),
        policy_gradient_coef=schedule.policy_gradient_coef(step),
        entropy_coef=schedule.entropy_coef(step),
        behavior_cloning_coef=schedule.behavior_cloning_coef(step),
        value_function_coef=schedule.value_function_coef(step),
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
    ) -> None:
        self.cfg = train_config
        self.model_config = model_config
        self.ship_config = ship_config
        self.env_config = train_config.scales[0].env_config
        self.device = torch.device(device)
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

        self.wrapper = MVPEnvWrapper(
            num_envs=train_config.scales[0].num_envs,
            ship_config=ship_config,
            env_config=train_config.scales[0].env_config,
            rewards=train_config.rewards,
            device=device,
        )
        K = self.wrapper.num_active_components
        self._active_names = self.wrapper.active_names  # stable ref used throughout

        self._compile_mode = compile_mode
        self._policy_module = MVPPolicy(
            model_config, ship_config, num_value_components=K
        ).to(self.device)
        self.policy = (
            torch.compile(self._policy_module, mode=compile_mode)
            if compile_mode is not None
            else self._policy_module
        )
        self.optim = optim.Adam(
            self._policy_module.parameters(), lr=base_state.learning_rate, eps=1e-5
        )

        N = train_config.scales[0].env_config.num_ships

        # Build obs shapes for the buffer from a dummy reset
        sample_obs = self.wrapper.reset()
        obs_shapes = {
            k: v.shape[1:]  # strip batch dim → (N, ...) or (N,)
            for k, v in sample_obs.items()
            if v.dtype in (torch.float32, torch.int32, torch.int64, torch.bool)
        }

        self.buffer = RolloutBuffer(
            num_steps=train_config.num_steps,
            num_envs=train_config.scales[0].num_envs,
            num_ships=N,
            num_components=K,
            obs_shapes=obs_shapes,
            gamma=train_config.gamma,
            gae_lambda=train_config.gae_lambda,
            device=self.device,
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
            model_config, ship_config, num_value_components=K
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
            torch.zeros_like(p) for p in self._policy_module.parameters()
        ]
        self._avg_update_count: int = 0

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
            self._init_wandb(train_config, model_config, ship_config, self.env_config)
            self._log_thread = threading.Thread(target=self._log_worker, daemon=True)
            self._log_thread.start()

        self._global_step = 0
        total_envs_all = sum(sc.num_envs for sc in train_config.scales)
        self._num_updates = train_config.total_timesteps // (
            total_envs_all * train_config.num_steps
        )

        # Run name used as checkpoint subdirectory (e.g. "checkpoints/good-spaceship-223/")
        if use_wandb:
            import wandb as _wandb

            self._run_name: str = _wandb.run.name
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
            )
            aux_sample_obs = aux_w.reset()
            aux_obs_shapes = {
                k: v.shape[1:]
                for k, v in aux_sample_obs.items()
                if v.dtype in (torch.float32, torch.int32, torch.int64, torch.bool)
            }
            aux_buf = RolloutBuffer(
                num_steps=train_config.num_steps,
                num_envs=sc.num_envs,
                num_ships=sc.env_config.num_ships,
                num_components=K,
                obs_shapes=aux_obs_shapes,
                gamma=train_config.gamma,
                gae_lambda=train_config.gae_lambda,
                device=self.device,
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
            cum.add_(p.detach())
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
        hidden = self.policy.initial_hidden(B, N, self.device)

        # Avg-model hidden state — lives across the whole training run.
        avg_hidden: torch.Tensor | None = None
        if self.B_avg > 0:
            avg_hidden = self.avg_policy.initial_hidden(self.B_avg, N, self.device)

        # League hidden state — re-initialised each rollout when the entry changes.
        league_hidden: torch.Tensor | None = None

        # Aux-scale obs, hidden states, and last-done flags — live across the whole run.
        aux_obs: list[dict[str, torch.Tensor]] = []
        aux_hiddens: list[torch.Tensor] = []
        aux_last_dones: list[torch.Tensor] = []
        for sc, aux_w in zip(self.cfg.scales[1:], self.aux_wrappers):
            aux_obs.append(aux_w.reset())
            aux_w.env.state.step_count.random_(0, sc.env_config.max_episode_steps)
            aux_hiddens.append(
                self.policy.initial_hidden(
                    sc.num_envs, sc.env_config.num_ships, self.device
                )
            )
            aux_last_dones.append(
                torch.zeros(sc.num_envs, dtype=torch.bool, device=self.device)
            )

        start_time = time.time()

        for update in range(1, self._num_updates + 1):
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
                        self.ship_config,
                        self.wrapper.num_active_components,
                        self.device,
                        self._compile_mode,
                    )
                    self._current_league_policy = entry._policy
                    league_hidden = self._current_league_policy.initial_hidden(
                        self.B_league, N, self.device
                    )
                elif entry.kind == "avg":
                    self._current_league_policy = self.avg_policy
                    league_hidden = self._current_league_policy.initial_hidden(
                        self.B_league, N, self.device
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
            # Rollout collection
            # ----------------------------------------------------------------
            for _ in range(self.cfg.num_steps):
                # Training policy: all envs, all ships
                action, logprob, value_norm, hidden = self.policy.get_action_and_value(
                    obs, hidden
                )

                # Team IDs — used for opponent overrides, actor mask, and BC expert masking
                team_id = obs["team_id"]  # (B, N) int

                if self._policy_gradient_coef == 0.0:
                    # BC pretraining: pure self-play — live policy controls all ships.
                    # Query scripted agent on all B envs for supervised targets.
                    # No opponent overrides; actor_mask is all-True.
                    with torch.no_grad():
                        _, expert_probs_step = (
                            self.scripted_agent.get_actions_and_probs(
                                self.wrapper.env.state
                            )
                        )  # expert_probs_step: (B, N, 12)
                    actor_mask = torch.ones(B, N, dtype=torch.bool, device=self.device)
                    use_avg = False
                    use_league = False
                else:
                    # avg active when: slots exist, phase wants it, and avg model has been snapshotted
                    use_avg = (
                        self._avg_update_count > 0
                        and self.B_avg > 0
                        and self._schedule_state.avg_model_fraction > 0.0
                    )

                    # Avg-model opponent actions (when ready)
                    if use_avg:
                        with torch.no_grad():
                            obs_avg = _slice_obs(obs, avg_start, avg_end)
                            action_avg, _, _, avg_hidden = (
                                self.avg_policy.get_action_and_value(
                                    obs_avg, avg_hidden
                                )
                            )

                    # League opponent actions (when a valid entry is loaded for this rollout)
                    use_league = (
                        self.B_league > 0
                        and self._current_league_entry is not None
                        and self._schedule_state.league_fraction > 0.0
                    )
                    if use_league:
                        if self._current_league_policy is not None:
                            with torch.no_grad():
                                obs_league = _slice_obs(obs, league_start, league_end)
                                action_league, _, _, league_hidden = (
                                    self._current_league_policy.get_action_and_value(
                                        obs_league, league_hidden
                                    )
                                )
                        else:  # scripted
                            state_league = _slice_state(
                                self.wrapper.env.state, league_start, league_end
                            )
                            with torch.no_grad():
                                action_league = self.scripted_agent.get_actions(
                                    state_league
                                )

                    # use_sc_opponent: scripted agent plays as opponent when phase requests it.
                    # Controlled entirely by scripted_frac > 0 in the current phase.
                    use_sc_bc = self.B_sc > 0
                    use_sc_opponent = (
                        use_sc_bc and self._schedule_state.scripted_fraction > 0.0
                    )

                    # BC target collection: query scripted agent on ALL envs whenever
                    # bc_coef is active. This is independent of scripted_frac and pg_coef.
                    # actor_mask in the loss already excludes opponent ships from the BC
                    # gradient, so no need to zero them out here.
                    expert_probs_step: torch.Tensor | None = None
                    action_scripted = None
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
                        # bc_coef == 0 but scripted opponent is active — need actions only
                        with torch.no_grad():
                            state_sc = _slice_state(
                                self.wrapper.env.state, sc_start, sc_end
                            )
                            action_scripted, _ = (
                                self.scripted_agent.get_actions_and_probs(state_sc)
                            )

                    # Override opponent team actions; training-policy actions are the base
                    action = action.clone()
                    sc_flags = self._opp_team_flag[: self.B_sc]
                    avg_flags = self._opp_team_flag[self.B_sc : self.B_sc + self.B_avg]
                    lg_flags = self._opp_team_flag[self.B_sc + self.B_avg :]
                    if use_sc_opponent:
                        _override_opponent(
                            action, team_id, sc_flags, sc_start, sc_end, action_scripted
                        )
                    if use_avg:
                        _override_opponent(
                            action, team_id, avg_flags, avg_start, avg_end, action_avg
                        )
                    if use_league:
                        _override_opponent(
                            action,
                            team_id,
                            lg_flags,
                            league_start,
                            league_end,
                            action_league,
                        )

                    # Actor mask: True for ships whose actions were chosen by training policy
                    actor_mask = torch.ones(B, N, dtype=torch.bool, device=self.device)
                    if use_sc_opponent:
                        actor_mask[sc_start:sc_end] = team_id[
                            sc_start:sc_end
                        ] != sc_flags.unsqueeze(1)
                    if use_avg:
                        actor_mask[avg_start:avg_end] = team_id[
                            avg_start:avg_end
                        ] != avg_flags.unsqueeze(1)
                    if use_league:
                        actor_mask[league_start:league_end] = team_id[
                            league_start:league_end
                        ] != lg_flags.unsqueeze(1)

                next_obs, reward, dones, truncated, info = self.wrapper.step(action)

                if info.get("ep_reward") is not None:
                    ep_rewards.append(info["ep_reward"])
                    ep_lengths.append(info["ep_length"].float())
                    for name, t in info["ep_reward_components"].items():
                        ep_components.setdefault(name, []).append(t)
                    for name, t in info["ep_scaled_reward_components"].items():
                        ep_scaled_components.setdefault(name, []).append(t)
                    ep_wins.append(info["ep_wins"])

                done_any = dones | truncated
                self.buffer.add(
                    obs=obs,
                    action=action,
                    logprob=logprob,
                    reward=reward,
                    done=dones.float(),  # only true termination cuts GAE bootstrap
                    value=self.scaler.denormalize(
                        value_norm
                    ),  # symlog-reward space for GAE
                    alive=obs["alive"].bool(),
                    actor_mask=actor_mask,
                    expert_probs=expert_probs_step,
                )

                # Reset hidden states for terminated envs
                hidden = self.policy.reset_hidden_for_envs(hidden, done_any, N)
                if use_avg and self.B_avg > 0:
                    avg_hidden = self.avg_policy.reset_hidden_for_envs(
                        avg_hidden, done_any[avg_start:avg_end], N
                    )
                if use_league and self._current_league_policy is not None:
                    league_hidden = self._current_league_policy.reset_hidden_for_envs(
                        league_hidden, done_any[league_start:league_end], N
                    )

                # Re-randomise opponent team for envs that just ended an episode
                if self.B_sc + self.B_avg + self.B_league > 0:
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

                # Aux-scale rollout steps (pure self-play, no opponent overrides)
                for i, (sc, aux_w, aux_buf) in enumerate(
                    zip(self.cfg.scales[1:], self.aux_wrappers, self.aux_buffers)
                ):
                    aux_N = sc.env_config.num_ships
                    with torch.no_grad():
                        aux_action, aux_logprob, aux_value_norm, aux_hiddens[i] = (
                            self.policy.get_action_and_value(aux_obs[i], aux_hiddens[i])
                        )
                    next_aux_obs, aux_reward, aux_dones, aux_truncated, _ = aux_w.step(
                        aux_action
                    )
                    aux_buf.add(
                        obs=aux_obs[i],
                        action=aux_action,
                        logprob=aux_logprob,
                        reward=aux_reward,
                        done=aux_dones.float(),
                        value=self.scaler.denormalize(aux_value_norm),
                        alive=aux_obs[i]["alive"].bool(),
                        actor_mask=torch.ones(
                            sc.num_envs, aux_N, dtype=torch.bool, device=self.device
                        ),
                        expert_probs=None,
                    )
                    aux_done_any = aux_dones | aux_truncated
                    aux_hiddens[i] = self.policy.reset_hidden_for_envs(
                        aux_hiddens[i], aux_done_any, aux_N
                    )
                    aux_last_dones[i] = aux_dones
                    aux_obs[i] = next_aux_obs
                    self._global_step += sc.num_envs

            # ----------------------------------------------------------------
            # GAE computation
            # ----------------------------------------------------------------
            with torch.no_grad():
                _, _, next_value_norm, _ = self.policy.get_action_and_value(obs, hidden)
            next_value = self.scaler.denormalize(next_value_norm)  # symlog-reward space
            self.buffer.compute_gae(next_value, dones.float())

            # Aux-scale GAE
            for i, (aux_buf, aux_h) in enumerate(zip(self.aux_buffers, aux_hiddens)):
                with torch.no_grad():
                    _, _, next_aux_val_norm, _ = self.policy.get_action_and_value(
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
            self._behavior_cloning_coef = self._schedule_state.behavior_cloning_coef
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
            metrics["schedule/behavior_cloning_coef"] = (
                self._schedule_state.behavior_cloning_coef
            )
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
                metrics["ep/reward_mean"] = all_rewards.mean().item()
                metrics["ep/reward_min"] = all_rewards.min().item()
                metrics["ep/reward_max"] = all_rewards.max().item()
                metrics["ep/length_mean"] = all_lengths.mean().item()
                for name, tensors in ep_components.items():
                    metrics[f"ep/reward_{name}"] = torch.cat(tensors).mean().item()
                for name, tensors in ep_scaled_components.items():
                    metrics[f"ep/scaled_{name}"] = torch.cat(tensors).mean().item()
                if ep_wins:
                    metrics["ep/win_rate"] = torch.cat(ep_wins).mean().item()

            sps = int(self._global_step / (time.time() - start_time))
            metrics["train/lr"] = self.optim.param_groups[0]["lr"]
            metrics["train/global_step"] = self._global_step
            metrics["train/sps"] = sps

            # ELO evaluation — runs sync matchups against all roster entries
            elo_eval_interval: int = self._schedule_state.elo_eval_interval
            if elo_eval_interval > 0 and update % elo_eval_interval == 0:
                elo_metrics = self._run_elo_eval()
                metrics.update(elo_metrics)
                # Save overwriting best-model checkpoints when normalized ELO improves.
                random_elo = self._random_elo()
                training_elo_norm = self._training_elo - random_elo
                if training_elo_norm > self._best_training_elo_norm:
                    self._best_training_elo_norm = training_elo_norm
                    self._save_best_checkpoint("best_training.pt")
                avg_entry = next(
                    (e for e in self.roster.entries if e.kind == "avg"), None
                )
                if avg_entry is not None:
                    avg_elo_norm = avg_entry.elo - random_elo
                    if avg_elo_norm > self._best_avg_elo_norm:
                        self._best_avg_elo_norm = avg_elo_norm
                        self._save_best_checkpoint(
                            "best_avg.pt", self._avg_checkpoint_payload(update=0)
                        )

            # Single log call per update — all metrics at the same step
            self._enqueue_log(metrics, step=self._global_step)

            if update % 10 == 0:
                elo_str = (
                    f"  elo={self._training_elo:.0f}" if elo_eval_interval > 0 else ""
                )
                print(
                    f"update={update}/{self._num_updates}  "
                    f"step={self._global_step:,}  "
                    f"sps={sps:,}  "
                    f"loss={metrics.get('train/loss', 0.0):.4f}"
                    f"{elo_str}"
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
        ) = batch

        logprob, entropy, new_value, policy_logits = self.policy.evaluate_actions(
            obs=mb_obs,
            actions=mb_actions.long(),
            initial_hidden=mb_hidden,
            alive_mask=mb_alive,
        )

        alive_f = mb_alive.float()  # (T, B_mb, N)
        alive_k = alive_f.unsqueeze(-1)  # (T, B_mb, N, 1)
        mask_sum = alive_f.sum().clamp(min=1.0)

        actor_f = (mb_actor_mask & mb_alive).float()  # (T, B_mb, N)
        actor_sum = actor_f.sum().clamp(min=1.0)

        # ---- Lambda aggregation -------------------------------------------
        # Per-timestep team IDs: correctly tracks re-assignments after mid-rollout resets.
        # Buffer stores obs as float32; cast to long for comparison.
        team_id_t = mb_obs["team_id"].long()  # (T, B_mb, N)
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
        bc_loss = torch.tensor(0.0, device=self.device)
        scripted_entropy = torch.tensor(0.0, device=self.device)
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

        loss = (
            self._policy_gradient_coef * pg_loss
            + self._schedule_state.value_function_coef * vf_loss
            + self._schedule_state.entropy_coef * ent_loss
            + self._behavior_cloning_coef * bc_loss
        )

        # ---- Diagnostics (no grad) ---------------------------------------
        diag: dict = {}
        with torch.no_grad():
            diag["loss"] = loss.item()
            diag["pg_loss"] = pg_loss.item()
            diag["vf_loss"] = vf_loss.item()
            diag["ent_loss"] = ent_loss.item()
            diag["bc_loss"] = bc_loss.item()
            diag["scripted_entropy"] = scripted_entropy.item()
            diag["bc_kl"] = bc_loss.item() - scripted_entropy.item()
            diag["adv_var"] = adv_rms.item()
            diag["approx_kl"] = (
                ((ratio - 1) - log_ratio) * actor_f
            ).sum().item() / actor_sum.item()
            diag["clip_frac"] = (
                ((ratio - 1).abs() > cfg.clip_coef).float() * actor_f
            ).sum().item() / actor_sum.item()
            diag["alive_frac"] = alive_f.mean().item()
            diag["ratio_mean"] = (ratio * actor_f).sum().item() / actor_sum.item()
            actor_bool = actor_f.bool()
            diag["ratio_max"] = (
                ratio[actor_bool].max().item() if actor_bool.any() else 1.0
            )

            # Per-head entropy — recomputed from policy_logits (already returned by evaluate_actions)
            power_ent = Categorical(logits=policy_logits[..., POWER_SLICE]).entropy()
            turn_ent = Categorical(logits=policy_logits[..., TURN_SLICE]).entropy()
            shoot_ent = Categorical(logits=policy_logits[..., SHOOT_SLICE]).entropy()
            diag["entropy_power"] = (
                power_ent * actor_f
            ).sum().item() / actor_sum.item()
            diag["entropy_turn"] = (turn_ent * actor_f).sum().item() / actor_sum.item()
            diag["entropy_shoot"] = (
                shoot_ent * actor_f
            ).sum().item() / actor_sum.item()

            ret_agg_mean = (ret_agg * actor_f).sum() / actor_sum
            ret_agg_var = ((ret_agg - ret_agg_mean).pow(2) * actor_f).sum() / actor_sum
            diag["ret_agg_mean"] = ret_agg_mean.item()
            diag["ret_agg_std"] = ret_agg_var.sqrt().item()

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
                diag["logprob_flat"] = logprob.detach().reshape(-1)

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

        accum_scalar: dict[str, list[float]] = {
            "train/loss": [],
            "train/policy_gradient_loss": [],
            "train/value_loss": [],
            "train/entropy_loss": [],
            "train/behavioral_cloning_loss": [],
            "train/bc_kl": [],
            "train/scripted_entropy": [],
            "train/approximate_kl": [],
            "train/clip_fraction": [],
            "train/gradient_norm": [],
            "train/advantage_std": [],
            "train/alive_fraction": [],
            "train/ratio_mean": [],
            "train/ratio_max": [],
            "train/entropy_power": [],
            "train/entropy_turn": [],
            "train/entropy_shoot": [],
            "returns/aggregate": [],
            "returns/aggregate_std": [],
        }
        accum_k: dict[str, list[torch.Tensor]] = {
            "critic/value_loss": [],
            "critic/explained_variance": [],
            "critic/return_mean": [],
            "critic/value_pred_mean": [],
            "returns/component": [],
            "train/advantage_std_k": [],
        }
        last_returns_np = None
        last_logprob_np = None

        for epoch_idx in range(cfg.num_epochs):
            iters = [
                buf.get_minibatch_iterator(cfg.num_minibatches) for buf in all_buffers
            ]
            for batches in zip(*iters):
                self.optim.zero_grad()

                # Accumulate gradients across all scales before stepping.
                # Each loss is divided by n_scales so the total gradient magnitude
                # stays comparable to single-scale training.
                diag_primary: dict = {}
                scalar_accum_step: dict[str, float] = {
                    "loss": 0.0,
                    "pg": 0.0,
                    "vf": 0.0,
                    "ent": 0.0,
                    "bc": 0.0,
                    "bc_kl": 0.0,
                    "scripted_entropy": 0.0,
                    "kl": 0.0,
                    "clip": 0.0,
                    "adv_var": 0.0,
                    "ret_agg_mean": 0.0,
                    "ret_agg_std": 0.0,
                    "alive_frac": 0.0,
                    "ratio_mean": 0.0,
                    "ratio_max": 0.0,
                    "entropy_power": 0.0,
                    "entropy_turn": 0.0,
                    "entropy_shoot": 0.0,
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

                accum_scalar["train/loss"].append(scalar_accum_step["loss"])
                accum_scalar["train/policy_gradient_loss"].append(
                    scalar_accum_step["pg"]
                )
                accum_scalar["train/value_loss"].append(scalar_accum_step["vf"])
                accum_scalar["train/entropy_loss"].append(scalar_accum_step["ent"])
                accum_scalar["train/behavioral_cloning_loss"].append(
                    scalar_accum_step["bc"]
                )
                accum_scalar["train/bc_kl"].append(scalar_accum_step["bc_kl"])
                accum_scalar["train/scripted_entropy"].append(
                    scalar_accum_step["scripted_entropy"]
                )
                accum_scalar["train/approximate_kl"].append(scalar_accum_step["kl"])
                accum_scalar["train/clip_fraction"].append(scalar_accum_step["clip"])
                accum_scalar["train/gradient_norm"].append(grad_norm.item())
                accum_scalar["train/advantage_std"].append(
                    scalar_accum_step["adv_var"] ** 0.5
                )
                accum_scalar["train/alive_fraction"].append(
                    scalar_accum_step["alive_frac"]
                )
                accum_scalar["train/ratio_mean"].append(scalar_accum_step["ratio_mean"])
                accum_scalar["train/ratio_max"].append(scalar_accum_step["ratio_max"])
                accum_scalar["train/entropy_power"].append(
                    scalar_accum_step["entropy_power"]
                )
                accum_scalar["train/entropy_turn"].append(
                    scalar_accum_step["entropy_turn"]
                )
                accum_scalar["train/entropy_shoot"].append(
                    scalar_accum_step["entropy_shoot"]
                )
                accum_scalar["returns/aggregate"].append(
                    scalar_accum_step["ret_agg_mean"]
                )
                accum_scalar["returns/aggregate_std"].append(
                    scalar_accum_step["ret_agg_std"]
                )

                if "stats_k_cpu" in diag_primary:
                    stats_k_cpu = diag_primary["stats_k_cpu"]
                    accum_k["critic/value_loss"].append(stats_k_cpu[0])
                    accum_k["critic/return_mean"].append(stats_k_cpu[2])
                    accum_k["returns/component"].append(stats_k_cpu[3])
                    accum_k["critic/value_pred_mean"].append(stats_k_cpu[4])
                    if epoch_idx == cfg.num_epochs - 1:
                        accum_k["critic/explained_variance"].append(stats_k_cpu[1])

                if "adv_std_k" in diag_primary:
                    accum_k["train/advantage_std_k"].append(diag_primary["adv_std_k"])

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

        metrics: dict = {k: sum(v) / len(v) for k, v in accum_scalar.items() if v}

        for key, tensors in accum_k.items():
            if not tensors:
                continue
            avg = torch.stack(tensors).mean(0)  # (K,) CPU
            prefix = "returns" if key == "returns/component" else key
            for i, name in enumerate(self._active_names):
                metrics[f"{prefix}/{name}"] = avg[i].item()

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

    def _run_elo_eval(self) -> dict[str, float]:
        """Run a single synchronized mixed-batch evaluations against Random and Scripted anchors.

        Computes a statistical Maximum Likelihood Estimation of the Universal Elo,
        then injects this as the new `_training_elo`. This provides an extremely
        clean signal to the league training system.
        """
        from boost_and_broadside.modes.agent_factory import (
            ResolvedAgent,
            get_actions,
            init_hidden,
            reset_done_envs,
        )
        from boost_and_broadside.modes.collect import _obs_from_state
        from scipy.optimize import root_scalar

        # Lazily add the scripted entry to the roster if enough time has passed and the
        # current phase permits it.
        if (
            self.scripted_agent is not None
            and self._schedule_state.allow_scripted_in_roster
            and self._global_step >= self.cfg.scripted_roster_min_steps
            and not any(e.kind == "scripted" for e in self.roster.entries)
        ):
            self.roster.add_special(
                "scripted", self._global_step, 0, initial_elo=1000.0
            )

        # 1. Determine batch routing
        games: int = self._schedule_state.elo_eval_games
        if self.scripted_agent is not None:
            ratio = max(0.0, min(1.0, self._training_elo / 1000.0))
            games_sc = int(ratio * games)
            games_rand = games - games_sc
        else:
            games_sc = 0
            games_rand = games

        # 2. Setup Env and Agents
        B = games
        N = self.env_config.num_ships
        dev = self.device

        env = TensorEnv(B, self.ship_config, self.env_config, self.device)
        self.policy.eval()

        agent_policy = ResolvedAgent("policy", self.policy)
        agent_sc = (
            ResolvedAgent("scripted", self.scripted_agent)
            if self.scripted_agent
            else None
        )
        agent_rand = ResolvedAgent("random", None)

        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        is_scripted = torch.zeros(B, dtype=torch.bool, device=dev)
        if games_sc > 0:
            is_scripted[:games_sc] = True

        init_hidden(agent_policy, B, N, dev)
        if games_sc > 0:
            init_hidden(agent_sc, B, N, dev)
        if games_rand > 0:
            init_hidden(agent_rand, B, N, dev)

        env.reset()

        wins_vs_sc = 0.0
        wins_vs_rand = 0.0

        metrics: dict[str, float] = {}

        try:
            with torch.no_grad():
                while not finished.all():
                    state = env.state
                    obs = _obs_from_state(state, self.ship_config)
                    action_a = get_actions(agent_policy, obs, state, B, N, dev)

                    # Route opposing actions accurately per batch index
                    action_b = torch.zeros_like(action_a)
                    if games_sc > 0:
                        action_sc = get_actions(agent_sc, obs, state, B, N, dev)
                        action_b = torch.where(
                            is_scripted.unsqueeze(1).unsqueeze(2), action_sc, action_b
                        )
                    if games_rand > 0:
                        action_rand = get_actions(agent_rand, obs, state, B, N, dev)
                        action_b = torch.where(
                            ~is_scripted.unsqueeze(1).unsqueeze(2),
                            action_rand,
                            action_b,
                        )

                    team_id = state.ship_team_id
                    action = torch.where(
                        (team_id == 0).unsqueeze(-1), action_a, action_b
                    )

                    dones, truncated = env.step(action)
                    done_any = dones | truncated

                    new_done = done_any & ~finished
                    if new_done.any():
                        alive = env.state.ship_alive
                        team = env.state.ship_team_id
                        team0_alive = (alive & (team == 0)).any(dim=1)
                        team1_alive = (alive & (team == 1)).any(dim=1)
                        team1_won = new_done & team1_alive & ~team0_alive
                        team0_won = new_done & team0_alive & ~team1_alive
                        tied = new_done & ~team0_won & ~team1_won

                        wins_vs_sc += float(
                            (team0_won & is_scripted).sum()
                        ) + 0.5 * float((tied & is_scripted).sum())
                        wins_vs_rand += float(
                            (team0_won & ~is_scripted).sum()
                        ) + 0.5 * float((tied & ~is_scripted).sum())

                        finished |= new_done

                    if done_any.any():
                        env.reset_envs(done_any)
                        reset_done_envs(agent_policy, done_any, N)
                        if games_sc > 0:
                            reset_done_envs(agent_sc, done_any, N)
                        if games_rand > 0:
                            reset_done_envs(agent_rand, done_any, N)

                # 3. Solve MLE Universal Elo
                def compute_eval_elo(w_rand, g_rand, w_sc, g_sc) -> float:
                    # Mild Laplace smoothing bounds the infinity of 100% win/loss
                    w_rand += 0.5
                    g_rand += 1.0
                    w_sc += 0.5
                    g_sc += 1.0
                    total_expected_wins = w_rand + w_sc

                    def score_fn(r):
                        expected_rand = g_rand / (1 + 10 ** ((0 - r) / 400))
                        expected_scripted = g_sc / (1 + 10 ** ((1000 - r) / 400))
                        return total_expected_wins - (expected_rand + expected_scripted)

                    res = root_scalar(score_fn, bracket=[-2000, 3000])
                    return float(res.root)

                self._training_elo = compute_eval_elo(
                    wins_vs_rand, games_rand, wins_vs_sc, games_sc
                )

                metrics["elo/training"] = self._training_elo
                if games_rand > 0:
                    metrics["elo/win_rate_vs_random"] = wins_vs_rand / games_rand
                if games_sc > 0:
                    metrics["elo/win_rate_vs_scripted"] = wins_vs_sc / games_sc

                # Evaluate avg model ELO if it has been initialized.
                if self._avg_update_count > 0:
                    avg_wins_vs_sc = 0.0
                    avg_wins_vs_rand = 0.0
                    avg_finished = torch.zeros(B, dtype=torch.bool, device=dev)
                    agent_avg = ResolvedAgent("policy", self.avg_policy)
                    init_hidden(agent_avg, B, N, dev)
                    env.reset()

                    while not avg_finished.all():
                        state = env.state
                        obs = _obs_from_state(state, self.ship_config)
                        action_a = get_actions(agent_avg, obs, state, B, N, dev)

                        action_b = torch.zeros_like(action_a)
                        if games_sc > 0:
                            action_sc = get_actions(agent_sc, obs, state, B, N, dev)
                            action_b = torch.where(
                                is_scripted.unsqueeze(1).unsqueeze(2),
                                action_sc,
                                action_b,
                            )
                        if games_rand > 0:
                            action_rand = get_actions(agent_rand, obs, state, B, N, dev)
                            action_b = torch.where(
                                ~is_scripted.unsqueeze(1).unsqueeze(2),
                                action_rand,
                                action_b,
                            )

                        team_id = state.ship_team_id
                        action = torch.where(
                            (team_id == 0).unsqueeze(-1), action_a, action_b
                        )

                        dones, truncated = env.step(action)
                        done_any = dones | truncated

                        new_done = done_any & ~avg_finished
                        if new_done.any():
                            alive = env.state.ship_alive
                            team = env.state.ship_team_id
                            team0_alive = (alive & (team == 0)).any(dim=1)
                            team1_alive = (alive & (team == 1)).any(dim=1)
                            team0_won = new_done & team0_alive & ~team1_alive
                            tied = (
                                new_done
                                & ~team0_won
                                & ~(new_done & team1_alive & ~team0_alive)
                            )
                            avg_wins_vs_sc += float(
                                (team0_won & is_scripted).sum()
                            ) + 0.5 * float((tied & is_scripted).sum())
                            avg_wins_vs_rand += float(
                                (team0_won & ~is_scripted).sum()
                            ) + 0.5 * float((tied & ~is_scripted).sum())
                            avg_finished |= new_done

                        if done_any.any():
                            env.reset_envs(done_any)
                            reset_done_envs(agent_avg, done_any, N)

                    avg_elo = compute_eval_elo(
                        avg_wins_vs_rand, games_rand, avg_wins_vs_sc, games_sc
                    )
                    avg_entry = next(
                        (e for e in self.roster.entries if e.kind == "avg"), None
                    )
                    if avg_entry is not None:
                        avg_entry.elo = avg_elo
                    metrics["elo/avg"] = avg_elo

        finally:
            self.policy.train()
            # Still evict checkpoint policies if any were loaded via the training rollout's league
            self.roster.evict_all_checkpoint_policies()

        return metrics

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
            "elo_milestone": self._elo_milestone,
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
        ckpt = torch.load(path, map_location=self.device)
        self._policy_module.load_state_dict(ckpt["policy_state_dict"])
        self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "adv_scaler_state_dict" in ckpt:
            self.adv_scaler.load_state_dict(ckpt["adv_scaler_state_dict"])
        if "avg_policy_state_dict" in ckpt:
            self._avg_policy_module.load_state_dict(ckpt["avg_policy_state_dict"])
            self._avg_param_cumsum = [
                c.to(self.device) for c in ckpt["avg_param_cumsum"]
            ]
            self._avg_update_count = ckpt["avg_update_count"]
        if "training_elo" in ckpt:
            self._training_elo = ckpt["training_elo"]
            self._elo_milestone = ckpt.get("elo_milestone", 0.0)

        # Restore roster if its JSON exists alongside the checkpoint
        roster_path = Path(path).parent / "roster.json"
        if roster_path.exists():
            self.roster.load_json(roster_path)

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
    ) -> None:
        """Initialize W&B run with all configs serialized as the run config.

        Args:
            train_config: PPO hyperparameters and timeline.
            model_config: Policy architecture.
            ship_config:  Physics constants.
            env_config:   Environment sizing.
        """
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
