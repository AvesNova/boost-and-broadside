"""Recurrent PPO trainer for the Yemong policy.

Core loop: collect rollout → compute per-component GAE → PPO update epochs →
log async → repeat. On top of that, PPOTrainer coordinates:

  - the decomposed critic (per-component returns, lambda aggregation,
    schedule-driven group scales),
  - auxiliary losses (behavior cloning from the scripted agent with
    win-rate-gated decay, next-state prediction,
    optional SIGReg),
  - opponent management (scripted / avg-model / league fractions, OpponentMixin),
  - continuous in-training Elo ladder evaluation (EloEvaluator) and the roster,
  - checkpointing (CheckpointMixin) and async W&B logging (LoggingMixin).

Rollout and update work run on CUDA streams where available, with a CPU
fallback path; logging stays off the GPU hot path.
"""

import dataclasses
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from typing import NamedTuple

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
from boost_and_broadside.config.diagnostics import (
    GRADIENT_DIAGNOSTICS_LEVELS,
    GRADIENT_DIAGNOSTICS_OFF,
    GradientDiagnosticsConfig,
)
from boost_and_broadside.config.live_elo import LIVE_RANDOM_ELO, live_reference_ladder
from boost_and_broadside.constants import (
    NUM_JOINT_ACTIONS,
    NUM_OUTCOME_CLASSES,
    NUM_POWER_ACTIONS,
    NUM_SHOOT_ACTIONS,
    NUM_TURN_ACTIONS,
    POWER_SLICE,
    SHOOT_SLICE,
    TURN_SLICE,
)
from boost_and_broadside.env.observation import ObsKey, YemongObservation, compile_observation
from boost_and_broadside.env.rewards import (
    GLOBAL_VALUE_COMPONENTS,
    REWARD_COMPONENT_NAMES,
    component_payout_ratios,
    component_weights,
)
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.execution import CUDA_GRAPH_COMPILE_MODES
from boost_and_broadside.run_manifest import RunStatus
from boost_and_broadside.runtime.actions import (
    PendingActionState,
    encode_joint_action_unchecked,
)
from boost_and_broadside.train.rl.allocation import allocation_weights
from boost_and_broadside.train.rl.belief import DualBeliefTracker
from boost_and_broadside.train.rl.buffer import (
    AdvantageScaler,
    LogicalRolloutBuffer,
    MicroBatch,
    ReturnScaler,
    RolloutBuffer,
    StoredRollout,
)
from boost_and_broadside.train.rl.checkpoint import CheckpointMixin
from boost_and_broadside.train.rl.elo_diagnostics import LiveEloDiagnostics
from boost_and_broadside.train.rl.elo_eval import MAX_ANCHORS, EloEvaluator, LadderOpponent
from boost_and_broadside.train.rl.features import (
    FeatureCoordinator,
    build_bullet_coordinator,
    build_standard_coordinator,
)
from boost_and_broadside.train.rl.grad_diagnostics import (
    TermGradientAccumulator,
    scope_metric_records,
    scope_statistics,
)
from boost_and_broadside.train.rl.hex_density import HEX_DENSITY_DIM, HexDensityTarget
from boost_and_broadside.train.rl.live_rating import TwoStageRating
from boost_and_broadside.train.rl.logging import LoggingMixin
from boost_and_broadside.train.rl.match_matrix import MatchMatrix
from boost_and_broadside.train.rl.next_state_populations import (
    candidate_losses,
    environment_halves,
    gradient_terms,
    population_masks,
    population_metric_records,
    population_moments,
)
from boost_and_broadside.train.rl.next_state_populations import (
    gradient_metric_records as next_state_gradient_records,
)
from boost_and_broadside.train.rl.opponents import (
    LeagueSlot,
    OpponentMixin,
    flip_team_obs,
)
from boost_and_broadside.train.rl.physical_belief import (
    ANGULAR_VELOCITY,
    ATTITUDE,
    COOLDOWN,
    HEALTH,
    LOCAL_LOG_INDEX,
    NEXT_STATE_OUTPUT_DIM,
    PHYSICAL_MEAN_DIM,
    PHYSICAL_MEAN_NAMES,
    PHYSICAL_UNCERTAINTY_DIM,
    POSITION_X,
    POSITION_Y,
    POWER,
    SHIELD_DELAY,
    VELOCITY_X,
    VELOCITY_Y,
    PhysicalNextState,
    physical_means_from_observation,
    physical_means_from_state,
    wrap_symmetric,
)
from boost_and_broadside.train.rl.physical_deltas import PHYSICAL_DELTA_SCALES
from boost_and_broadside.train.rl.policy_io import build_policy, compile_policy
from boost_and_broadside.train.rl.roster import EloRoster, RosterEntry
from boost_and_broadside.train.rl.sigreg import SIGReg

# ------------------------------------------------------------------
# Per-component gamma / lambda tensor builder
# ------------------------------------------------------------------


def _build_component_tensor(
    global_val: float,
    overrides: Mapping[str, float],
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

# Maps reward level → the TrainingSchedule tier-scale field to apply.
# Effective weight = tier_scale * level weight (from RewardConfig).
#
# The tiers are a credit-assignment ladder, and the per-level gammas and lambdas
# in config/defaults.py follow the same partition: the outcome and the zones are
# the objective, discounted over the match; a kill over an engagement; damage
# over an exchange. Scaling a whole tier at once is how a run shifts weight
# between "what actually wins" and the proxies for it.
_TIER: dict[str, str] = {
    "outcome": "outcome_scale",
    "zone_capture": "outcome_scale",
    "zone_progress": "outcome_scale",
    "kill_death": "kill_death_scale",
    "damage": "damage_scale",
}


@dataclasses.dataclass
class _ResolvedSchedule:
    """Training schedule evaluated at a single global step.

    Produced by ``_resolve_schedule``; replaces the old ``PhaseConfig`` snapshot.
    All fields are plain values — no callables, no Nones.
    """

    offensive_bias: float
    learning_rate: float
    policy_gradient_coef: float
    entropy_coef: float
    behavior_cloning_coef: float
    value_function_coef: float
    sigreg_coef: float
    outcome_scale: float
    kill_death_scale: float
    damage_scale: float
    league_fraction: float
    checkpoint_interval: int
    num_epochs: int
    target_kl: float | None
    high_winrate_threshold: float | None
    high_winrate_target_kl: float | None


@dataclasses.dataclass
class _RolloutRuntime:
    """Mutable state that persists across rollout updates."""

    num_envs: int
    num_ships: int
    # Recurrent tokens per env: ships plus the promoted global token (N+G). Map
    # tokens are non-recurrent, so this is deliberately neither N nor N+G+M --
    # it is the stride for every hidden-state operation.
    num_recurrent: int
    elo_eval: EloEvaluator
    obs: YemongObservation
    beliefs: DualBeliefTracker | None
    hidden: torch.Tensor
    hidden_t1: torch.Tensor | None
    action_state: PendingActionState
    league_slots: list[LeagueSlot]
    aux_obs: list[YemongObservation]
    aux_beliefs: list[DualBeliefTracker | None]
    aux_hiddens: list[torch.Tensor]
    aux_hidden_t1s: list[torch.Tensor | None]
    aux_action_states: list[PendingActionState]
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


def _huber(error: torch.Tensor, delta: float) -> torch.Tensor:
    """Squared error inside ``delta``, linear outside, continuous in both value
    and slope at the join.

    Scaled to agree with ``error**2`` in the quadratic region rather than with
    the textbook ``0.5 * error**2``, so switching a squared-error critic to this
    changes the tails and leaves the bulk of the loss — and therefore the critic's
    gradient scale — where it was.

    The tails are the point. Normalizing each component by its own statistics
    necessarily exposes them: a sparse component's returns are a spike at zero
    with rare large excursions, so its normalized error reaches values a dense
    component never sees. Under squared error one such token can outweigh a
    minibatch of ordinary ones, which is what the oversized ``return_min_span``
    floor was compensating for by shrinking every sparse component instead.

    Args:
        error: Any shape — the critic residual in normalized space.
        delta: Half-width of the quadratic region, in normalized units.

    Returns:
        Elementwise loss, same shape as ``error``.
    """
    magnitude = error.abs()
    return torch.where(
        magnitude <= delta,
        error.pow(2),
        delta * (2.0 * magnitude - delta),
    )


def _actor_entropy_coef(
    scheduled: float, *, policy_gradient_coef: float, behavior_cloning_coef: float
) -> float:
    """The entropy weight, dropped to zero when nothing else trains the actor.

    Entropy is a regularizer on an objective: it keeps a policy gradient from
    collapsing onto one action, and it keeps a cloned policy from over-sharpening
    past its teacher. It is not itself an objective. With both of those weights at
    zero it becomes the only gradient reaching the actor, and its optimum is the
    uniform distribution — so the run spends the rest of its budget undoing
    whatever the actor had learned.

    That is exactly the state a behavior-cloning run enters when its scripted win
    rate reaches ``bc_winrate_target``: ``_behavior_cloning_coef`` decays to zero
    while ``policy_gradient_coef`` is zero for the whole BC schedule. Measured at
    a reduced launch width (64 envs, d_model 64), a policy cloned to a KL of 1.12
    and 60% of maximum action entropy returned to 99.8% of maximum entropy and a
    KL of 2.66 — its untrained value — within 400 updates of the cutoff, while the
    control arm held at 1.10 and 60% over the same span.

    RL is unaffected: its policy gradient is positive throughout, so the
    scheduled value passes through unchanged. The critic, next-state, and SIGReg
    terms keep training through the shared trunk either way.
    """

    if policy_gradient_coef > 0.0 or behavior_cloning_coef > 0.0:
        return scheduled
    return 0.0


def _resolve_schedule(schedule: TrainingSchedule, step: int) -> _ResolvedSchedule:
    """Evaluate every schedule field at ``step`` and return a resolved snapshot."""
    return _ResolvedSchedule(
        offensive_bias=schedule.offensive_bias(step),
        learning_rate=schedule.learning_rate(step),
        policy_gradient_coef=schedule.policy_gradient_coef(step),
        entropy_coef=schedule.entropy_coef(step),
        behavior_cloning_coef=schedule.behavior_cloning_coef(step),
        value_function_coef=schedule.value_function_coef(step),
        sigreg_coef=schedule.sigreg_coef(step),
        outcome_scale=schedule.outcome_scale(step),
        kill_death_scale=schedule.kill_death_scale(step),
        damage_scale=schedule.damage_scale(step),
        league_fraction=schedule.league_fraction(step),
        checkpoint_interval=schedule.checkpoint_interval(step),
        num_epochs=schedule.num_epochs(step),
        target_kl=schedule.target_kl(step),
        high_winrate_threshold=schedule.high_winrate_threshold(step),
        high_winrate_target_kl=schedule.high_winrate_target_kl(step),
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


class _DensityStats(NamedTuple):
    """One micro-batch's global-density statistics, ally and enemy apart.

    Every field is already divided by the minibatch's step-environment total,
    so micro-batch contributions add.
    """

    loss: torch.Tensor
    deviance: torch.Tensor
    ally_loss: torch.Tensor
    enemy_loss: torch.Tensor
    ally_deviance: torch.Tensor
    enemy_deviance: torch.Tensor

    @classmethod
    def zeros(cls, zero: torch.Tensor) -> "_DensityStats":
        """All-zero stats for a scale or a schedule that does not train the head."""

        return cls(zero, zero, zero, zero, zero, zero)


class PPOTrainer(CheckpointMixin, LoggingMixin, OpponentMixin):
    """Proximal Policy Optimization for the Yemong multi-agent policy.

    Args:
        train_config:    PPO hyperparameters and timeline.
        model_config:    Policy architecture.
        ship_config:     Physics constants.
        device:          Torch device.
        use_wandb:       Whether to log metrics to W&B.
        scripted_agent:  Stochastic scripted agent for BC loss targets and scripted opponents.
        resolved_config_document: Complete resolved launch config/fingerprints for checkpoints.
        launch_provenance: Execution settings resolved by the installed CLI.
        gradient_diagnostics: Gradient decomposition depth and cadence. At its
            default the trainer takes no diagnostic code path at all.
    """

    def __init__(
        self,
        train_config: TrainConfig,
        model_config: ModelConfig,
        ship_config: ShipConfig,
        device: str | torch.device,
        use_wandb: bool = False,
        scripted_agent: StochasticScriptedAgent | None = None,
        compile_mode: str | None = "default",
        resume_wandb_run_id: str | None = None,
        resolved_config_document: Mapping[str, object] | None = None,
        launch_provenance: Mapping[str, object] | None = None,
        gradient_diagnostics: GradientDiagnosticsConfig = GRADIENT_DIAGNOSTICS_OFF,
    ) -> None:
        self.cfg = train_config
        self.model_config = model_config
        self.ship_config = ship_config
        self.resolved_config_document = resolved_config_document
        self.launch_provenance = launch_provenance
        # Paradigm: "ego_pass" (dual-perspective pass, team 0 trains) vs
        # "shared_pass" (single pass, both teams train). See TrainConfig docstring.
        self._ego_pass = train_config.paradigm == "ego_pass"
        if not self._ego_pass and any(
            scale.env_config.vision_range is not None for scale in train_config.scales
        ):
            raise ValueError(
                "finite vision requires paradigm='ego_pass'; shared_pass exposes one "
                "team's masked observation to both teams"
            )
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
        self._stores_bc_targets = (
            _max_schedule_value(
                train_config.schedule.behavior_cloning_coef,
                train_config.total_timesteps,
            )
            > 0.0
        )

        # Primary scale — two contiguous env groups:
        #   [0, B_self)        → self-play
        #   [B_self, B)        → league, split into cfg.league_slots slots that each
        #                        draw an opponent from the roster by Elo proximity
        # The block is sized from the MAXIMUM league fraction over the run so the
        # envs exist when a later phase widens it; the ACTIVE width inside it comes
        # from the current schedule value each rollout (see _active_league_width),
        # so a fraction that steps down genuinely returns envs to self-play.
        B = train_config.scales[0].num_envs
        max_league_frac = _max_schedule_value(
            train_config.schedule.league_fraction, train_config.total_timesteps
        )
        self.B_league = round(max_league_frac * B)
        self.B_self = B - self.B_league

        if base_state.policy_gradient_coef == 0.0 and scripted_agent is None:
            raise ValueError("policy_gradient_coef=0.0 (BC mode) requires a scripted_agent.")

        collision_compile_mode = (
            ("max-autotune" if compile_mode == "max-autotune" else "default")
            if compile_mode is not None
            else None
        )
        # The pure, unbuffered perception builder is part of the demonstrated
        # compiled training path.  Keep one launch-level switch: compile_mode
        # enables both collision and perception fusion, while None remains the
        # explicit eager/debug escape hatch.
        self._env_compile_mode = collision_compile_mode
        # A league opponent composes its own single-observer view, fused on the
        # same switch the rest of perception follows.
        self._build_opponent_view = compile_observation(collision_compile_mode)
        self.wrapper = YemongEnvWrapper(
            num_envs=train_config.scales[0].num_envs,
            ship_config=ship_config,
            env_config=train_config.scales[0].env_config,
            rewards=train_config.rewards,
            device=device,
            collision_compile_mode=collision_compile_mode,
            include_bullets=model_config.reads_bullets,
            perception_compile_mode=collision_compile_mode,
        )
        K = self.wrapper.num_active_components
        self._active_names = self.wrapper.active_names  # stable ref used throughout
        # Components whose reward is identical for every ship on a side by
        # construction, so their return is too and one estimate serves the team:
        # all three are a function of team and match result alone, paid to the
        # living and the dead alike. They read the global token; everything else
        # keeps a per-ship head, including the zone rewards, whose whole point is
        # that they pay the ships that showed up and charge the ones that did
        # not. ``tests/train/test_global_value.py`` asserts the uniformity rather
        # than trusting this list.
        self._global_value_k: tuple[int, ...] = tuple(
            i for i, n in enumerate(self._active_names) if n in GLOBAL_VALUE_COMPONENTS
        )
        # Where the categorical head reads its labels from. None when `outcome`
        # carries no weight, which is also when the head's loss is off.
        self._outcome_k: int | None = (
            self._active_names.index("outcome") if "outcome" in self._active_names else None
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
            global_value_k=self._global_value_k,
            predict_outcome=train_config.outcome_categorical_coef > 0.0,
            predict_density=train_config.global_density_coef > 0.0,
        ).to(self.device)
        self.sigreg = SIGReg(d_model=model_config.d_model, num_proj=64).to(self.device)
        # Captured before compiling. Two things re-traverse a micro-batch's
        # backward graph, which a compiled backward does not survive: the
        # actor/critic split probe, which gets its own forward pass, and the
        # gradient diagnostics, which decompose into too many terms to give each
        # one and so run the whole update eagerly. See _update_evaluate_actions.
        self._eager_evaluate_actions = self._policy_module.evaluate_actions
        self.policy = compile_policy(self._policy_module, compile_mode)
        # eps 1e-8, Torch's default, rather than the 1e-5 the PPO reference
        # implementations carry. Measured on run 748's own ``exp_avg_sq``, 1e-5
        # was 32% of the Adam denominator at the *median* parameter and at least
        # 90% of it for 18.6% of them -- a third of the network was effectively
        # on SGD, not Adam, and that is a hyperparameter rather than the
        # numerical guard eps is supposed to be. At 1e-8 it is 0.1% at the
        # median and the optimizer is scale-invariant again, which is also what
        # lets the loss coefficients be rescaled to put the gradient under
        # ``max_grad_norm`` without changing what the optimizer does.
        #
        # Consequence to watch: the parameters that were damped now take full
        # steps, so this is an effective learning-rate increase across much of
        # the network.
        self.optim = optim.Adam(
            self._policy_module.parameters(), lr=base_state.learning_rate, eps=1e-8
        )
        # CUDA-graph modes capture the backward too, and a `.grad` tensor first
        # allocated inside that capture lives in the graph's private pool -- the
        # next replay overwrites it, and accumulating into it across
        # micro-batches raises "accessing gradient tensor output of CUDAGraphs
        # that has been overwritten by a subsequent run". Torch's remedy is
        # stable buffers allocated before any capture, which then have to stay
        # allocated: `zero_grad(set_to_none=True)` would free them again and put
        # the next backward right back inside the pool.
        #
        # Keeping a zeroed grad where there would otherwise be None is exactly
        # equivalent here. Adam skips a parameter whose grad is None; for one
        # whose grad is all zeros it decays moments that are themselves zero and
        # applies `-lr * 0 / (sqrt(0) + eps)`, which is exactly zero, with no
        # weight decay configured to make it otherwise. The case that would
        # differ -- a parameter that gets a gradient on some steps and not
        # others -- cannot arise: participation is decided by token *counts*
        # (`field_sub` runs only when the observation carries map tokens), and
        # those are fixed for a run.
        self._zero_grad_to_none = compile_mode not in CUDA_GRAPH_COMPILE_MODES
        if not self._zero_grad_to_none:
            for parameter in self._policy_module.parameters():
                parameter.grad = torch.zeros_like(parameter)

        # --- Gradient diagnostics (observability; off changes nothing) ---
        # The parameter list and its trunk membership are fixed for the run, so
        # they are resolved once here rather than per diagnosed minibatch. At
        # level "off" both stay empty and nothing downstream ever runs.
        self._grad_diag = gradient_diagnostics
        self._grad_diag_params: list[nn.Parameter] = []
        self._grad_diag_trunk: list[bool] = []
        if self._grad_diag.enabled:
            trunk_ids = self._policy_module.trunk_parameter_ids()
            for parameter in self._policy_module.parameters():
                if not parameter.requires_grad:
                    continue
                self._grad_diag_params.append(parameter)
                self._grad_diag_trunk.append(id(parameter) in trunk_ids)

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
            num_tokens=sample_obs.pos.shape[1],
            prediction_target_dim=PHYSICAL_MEAN_DIM,
            prediction_dim=NEXT_STATE_OUTPUT_DIM,
            uncertainty_dim=PHYSICAL_UNCERTAINTY_DIM,
            density_dim=HEX_DENSITY_DIM if train_config.global_density_coef > 0.0 else 0,
            store_expert_probs=self._stores_bc_targets,
        )

        # OpenAI Five's team spirit per active level: how much of the mean
        # teammate advantage each ship's own advantage absorbs. Zero is pure
        # per-ship credit, which is what the rewards already assign.
        spirit = dict(zip(REWARD_COMPONENT_NAMES, train_config.rewards.team_spirit, strict=True))
        self._team_spirit_k = torch.tensor(
            [spirit[name] for name in self._active_names], dtype=torch.float32, device=self.device
        )  # (K,)

        # The physical next-state model: fixed Phase-1 delta scales, physical
        # bounds for the belief recursion, and the Gaussian likelihood over them.
        self.next_state = PhysicalNextState.from_ship_config(ship_config)
        # The privileged hex density target, grid resident on device. Built only
        # when the head that consumes it exists, and fused on the same launch
        # switch the rest of perception follows -- it runs once per rollout step
        # beside the observation, so it is part of that cost.
        self.density_target = (
            HexDensityTarget(ship_config.world_size, self.device, collision_compile_mode)
            if train_config.global_density_coef > 0.0
            else None
        )
        # Every stored primary observation is team 0's view, so that is the
        # observer whose allies fill the target's first half. Preallocated: this
        # is read once per rollout step.
        self._density_observer_team = torch.zeros(
            train_config.scales[0].num_envs, dtype=torch.int32, device=self.device
        )
        # One decision of game time, and the divisor that turns the observation's
        # normalized log index back into the natural log the belief plane stores.
        self._decision_dt = ship_config.dt * self.env_config.action_repeat
        self._index_log_scale = 2.0 * math.log(ship_config.field_index_step)

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
            global_value_k=self._global_value_k,
            predict_outcome=train_config.outcome_categorical_coef > 0.0,
            predict_density=train_config.global_density_coef > 0.0,
        ).to(self.device)
        self.avg_policy = compile_policy(self._avg_policy_module, compile_mode)
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
            # Hidden state covers ships and the global token; map tokens are
            # non-recurrent.
            _nt = self._policy_module.num_recurrent_tokens
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _h = self._policy_module.initial_hidden(B, _nt, self.device)
                self.policy.get_action_and_value(sample_obs, _h, return_enemy_action=self._ego_pass)
                if self._ego_pass:
                    # Warm up the 2B batch used by the combined team-0/team-1 rollout pass.
                    _obs_t1 = sample_obs.flip_team(N)
                    _obs_2B = sample_obs.concat_batch(_obs_t1)
                    _h_2B = torch.cat([_h, _h], dim=1)
                    self.policy.get_action_and_value(_obs_2B, _h_2B, return_enemy_action=True)
                _h_avg = self._avg_policy_module.initial_hidden(B, _nt, self.device)
                self.avg_policy.get_action_and_value(
                    sample_obs, _h_avg, return_enemy_action=self._ego_pass
                )

        # Per-env flag (shared_pass only): which team_id the league opponent plays.
        # In ego_pass opponents always play team 1. Randomised at init and
        # re-randomised each episode reset. Shape: (B_league,), indexed relative
        # to the start of the league block.
        self._opp_team_flag = (
            torch.randint(0, 2, (self.B_league,), device=self.device, dtype=torch.int32)
            if self.B_league > 0
            else torch.empty(0, device=self.device, dtype=torch.int32)
        )

        # --- League play + Elo ---
        self.roster = EloRoster(
            max_size=train_config.league_size,
            elo_temperature=train_config.elo_temperature,
            uniform_sampling=train_config.league_uniform_sampling,
        )
        # Random anchor is added by EloRoster.__init__ (Elo=0, fixed) and is
        # excluded from opponent sampling. "scripted" is registered below;
        # "avg" joins when _update_avg_model() first runs.
        self._register_special_opponents()

        # Seeded at the random reference's rating: an untrained policy is a
        # random one, and on the live gauge that is a defined point rather than
        # an arbitrary zero. Starting elsewhere just costs eval games to walk
        # back.
        self._live_elo: float = LIVE_RANDOM_ELO
        self._avg_live_elo: float = LIVE_RANDOM_ELO
        self._floating_games: int = 0  # rated games of the floating ladder checkpoint
        self._bc_cutoff_streak: int = 0  # consecutive updates past the BC win-rate target
        # Raw win rate against the scripted controller, refreshed each update.
        # Gates both the behavior-cloning decay and the target-KL tightening.
        self._scripted_win_rate: float = 0.0
        # Latest update's rated outcomes, opponent label → (win, loss, tie).
        self._match_counts: dict[str, tuple[int, int, int]] = {}
        # Read-only instrumentation for the rating filter above. Logs only; see
        # elo_diagnostics.
        self._elo_diagnostics = LiveEloDiagnostics(scripted_label="scripted")
        # Accumulated ladder record among weight-frozen players. Replaced
        # wholesale on resume; see _save_roster_json for why it is a sidecar.
        self.match_matrix = MatchMatrix()
        # The candidate replacement for the K-factor filter, running alongside
        # it. It gates nothing: swapping the estimator would change promotion
        # timing, which changes the pool, which changes every rating — so the
        # old one keeps gating for the whole comparison run and this one only
        # observes.
        self._two_stage = TwoStageRating(
            anchor_label="scripted",
            anchor_elo=train_config.elo_eval.scripted_live_elo,
        )
        eval_window_size = train_config.elo_eval.window_size
        self._eval_window_rand = deque(maxlen=eval_window_size)
        self._eval_window_sc = deque(maxlen=eval_window_size)
        # Monotone floor for the behavior-cloning gate; see _apply_schedule_state.
        self._eval_window_ladder = deque(maxlen=eval_window_size)
        self._eval_window_floating = deque(maxlen=eval_window_size)
        self._eval_window_live_vs_avg = deque(maxlen=eval_window_size)
        # Highest claimed ladder-milestone grid point, in normalized Elo (vs random).
        # Always a multiple of cfg.elo_milestone_gap once the first snapshot lands;
        # runs resumed from before the grid existed carry one off-grid value forward
        # and snap to the grid at their next snapshot.
        # Grid points are absolute, so the first one to claim is the highest
        # multiple of the gap at or below where the run starts.
        self._elo_milestone: float = (
            (LIVE_RANDOM_ELO // train_config.elo_milestone_gap) * train_config.elo_milestone_gap
            if train_config.elo_milestone_gap > 0
            else 0.0
        )
        # Best ratings seen, on the live gauge.
        self._best_live_elo: float = -float("inf")
        self._best_avg_live_elo: float = -float("inf")
        self._last_checkpoint_path: Path | None = None

        # Async logging queue
        self._log_queue: Queue = Queue()
        if use_wandb:
            self._init_wandb(
                train_config, model_config, ship_config, self.env_config, resume_wandb_run_id
            )
            self._log_thread = threading.Thread(target=self._log_worker, daemon=True)
            self._log_thread.start()

        self._global_step = 0
        # Where this process's configuration starts applying. A fresh run owns
        # the whole history; a resume owns everything from the step it restored
        # at, which is what makes `--continue` with changed settings recordable.
        self._start_step = 0
        self._segment_recorded_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._start_update = 1
        # Last update the loop carried all the way through. An interrupt lands
        # mid-update, so this -- not the update in progress -- is the only index
        # a final save can honestly claim.
        self._completed_update = 0
        # Cumulative counters persisted across checkpoint resumes so throughput
        # metrics behave as if training never stopped.
        self._ship_steps = 0  # ship tokens (all teams, all envs, all scales)
        # Entity tokens (ships + fields) the update phase processes per epoch —
        # one full pass over all scales' rollouts.
        self._entity_tokens_per_epoch = (
            train_config.num_steps
            * sum(sc.num_envs * sc.env_config.num_entity_tokens for sc in train_config.scales)
            * train_config.rollouts_per_update
        )
        # Cumulative entity tokens consumed by backward passes (counts actual
        # epochs completed, so target_kl early stops are reflected). The compute
        # x-axis for comparing runs with different batch/update configurations.
        self._grad_tokens = 0
        self._elapsed_train_time = 0.0  # wall-clock seconds spent training
        self._train_start_time = time.time()  # reset at the top of train()
        # Marks for the instantaneous throughput metrics: the step counters and
        # the clock as they stood at the previous logging point. Reset at the
        # top of train(), so a resumed run's first interval measures only the
        # updates it ran itself.
        self._perf_mark_time = self._train_start_time
        self._perf_mark_step = 0
        self._perf_mark_ship_steps = 0
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
            self.run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Schedule state — evaluated from the schedule functions each update.
        # Initialized from step=0 and refreshed after every PPO update.
        self._schedule_state: _ResolvedSchedule = base_state
        self._policy_gradient_coef: float = base_state.policy_gradient_coef
        self._behavior_cloning_coef: float = base_state.behavior_cloning_coef
        self._entropy_coef: float = _actor_entropy_coef(
            base_state.entropy_coef,
            policy_gradient_coef=base_state.policy_gradient_coef,
            behavior_cloning_coef=base_state.behavior_cloning_coef,
        )

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
                collision_compile_mode=collision_compile_mode,
                include_bullets=model_config.reads_bullets,
                perception_compile_mode=collision_compile_mode,
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
                num_tokens=aux_sample_obs.pos.shape[1],
                store_expert_probs=False,
            )
            self.aux_wrappers.append(aux_w)
            self.aux_buffers.append(aux_buf)

        self._active_save_thread = None
        self._active_best_thread = None
        self._active_best_avg_thread = None

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def _rollout_policy_pass(
        self,
        obs: YemongObservation,
        hidden: torch.Tensor,
        hidden_t1: torch.Tensor | None,
        num_ships: int,
        num_recurrent: int,
    ) -> tuple:
        """Run the training policy's rollout forward pass(es) for one step.

        ego_pass: one batched 2B pass over both team perspectives. Team 1 ships
        act from the flipped-obs half (action_t1); logprob/value/pred_next are
        stored from the raw-obs half only.
        shared_pass: one B pass on raw obs — every ship acts from it.

        Args:
            obs:        YemongObservation with (B, N+G+M, ...) tensors (raw team IDs).
            hidden:     (n_layers, B*(N+G), CONV_KERNEL*D) raw-perspective hidden state.
            hidden_t1:  Flipped-perspective hidden state; None in shared_pass.
            num_ships:  N — ship token count for team flipping.
            num_recurrent: N+G — recurrent tokens per env, used to split the 2B
                hidden state. Map tokens are non-recurrent, so this is not N+G+M.

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
            action, logprob, value_norm, pred_next, enemy_action_logits, hidden = (
                self.policy.get_action_and_value(obs, hidden, return_enemy_action=True)
            )
            return (
                action,
                None,
                logprob,
                value_norm,
                pred_next,
                None,
                hidden,
                enemy_action_logits,
                None,
                None,
            )
        batch = hidden.shape[1] // num_recurrent
        obs_t1 = flip_team_obs(obs.for_team(1), num_ships)
        obs_both = obs.concat_batch(obs_t1)
        hidden_both = torch.cat([hidden, hidden_t1], dim=1)  # (n_layers, 2B*(N+G), CK*D)
        action_both, logprob_both, value_both, pred_next_both, enemy_both, hidden_out = (
            self.policy.get_action_and_value(obs_both, hidden_both, return_enemy_action=True)
        )
        return (
            action_both[:batch],  # (B, N, 3)
            action_both[batch:],  # (B, N, 3)
            logprob_both[:batch],  # (B, N)
            value_both[:batch],  # (B, N, K)
            pred_next_both[:batch],  # (B, N, pred_dim)
            pred_next_both[batch:],  # (B, N, pred_dim)
            hidden_out[:, : batch * num_recurrent, :],  # (n_layers, B*(N+G), CK*D)
            enemy_both[:batch],  # physical indices, Team-0 observer
            enemy_both[batch:],  # physical indices, Team-1 canonical observer
            hidden_out[:, batch * num_recurrent :, :],  # (n_layers, B*(N+G), CK*D)
        )

    def _collect_aux_steps(
        self,
        aux_obs: list[YemongObservation],
        aux_beliefs: list[DualBeliefTracker | None],
        aux_hiddens: list[torch.Tensor],
        aux_hidden_t1s: list[torch.Tensor | None],
        aux_action_states: list[PendingActionState],
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
                    aux_pred_t0,
                    aux_pred_t1,
                    aux_hiddens[i],
                    aux_enemy_t0,
                    aux_enemy_t1,
                    aux_hidden_t1s[i],
                ) = self._rollout_policy_pass(
                    aux_obs[i],
                    aux_hiddens[i],
                    aux_hidden_t1s[i],
                    aux_N,
                    self.policy.num_recurrent_tokens,
                )
            # Ground truth, not the observation's masked copy (see
            # _collect_primary_step). Read before the step, which may reset.
            aux_team_id = aux_w.env.state.ship_team_id[:, :aux_N]  # (B_aux, N_aux)
            aux_action, aux_actor_mask = self._combine_actions(
                aux_action_t0, aux_action_t1, aux_team_id
            )
            action_state = aux_action_states[i]
            _, aux_reward, aux_dones, aux_truncated, aux_info = aux_w.step(
                action_state.applied_action(), observe=False
            )
            aux_done_any = aux_dones | aux_truncated
            decision_committed = action_state.commit(
                aux_action,
                aux_info["actuator_contiguous"],
                aux_done_any,
            )
            aux_buf.add(
                obs=aux_obs[i],
                action=aux_action,
                logprob=aux_logprob,
                reward=aux_reward,
                value=self.scaler.denormalize(aux_value_norm),
                alive=aux_obs[i]["alive"][:, :aux_N].bool(),
                actor_mask=aux_actor_mask,
                decision_committed=decision_committed,
                expert_probs=None,
                terminated=aux_done_any,
                transition_contiguous=aux_info["transition_contiguous"],
            )
            aux_recurrent = self.policy.num_recurrent_tokens
            aux_hiddens[i] = self.policy.reset_hidden_for_envs(
                aux_hiddens[i], aux_done_any, aux_recurrent
            )
            if self._ego_pass:
                aux_hidden_t1s[i] = self.policy.reset_hidden_for_envs(
                    aux_hidden_t1s[i], aux_done_any, aux_recurrent
                )
            aux_last_dones[i] = aux_done_any
            sources = None
            if aux_beliefs[i] is not None:
                aux_beliefs[i].advance(aux_pred_t0, aux_pred_t1, aux_enemy_t0, aux_enemy_t1)
                aux_beliefs[i].reset(aux_done_any)
                sources = aux_beliefs[i].observe(aux_w.env.state, aux_w.last_visibility.ship)
            next_aux_obs = aux_w.observe(sources)
            # Re-read after the step: the privacy mask must match the post-reset
            # observation it is written into.
            action_state.write_observation(
                next_aux_obs,
                aux_w.env.state.ship_team_id[:, :aux_N],
                aux_w.env.state.ship_spawned[:, :aux_N],
                aux_N,
                belief_action=None if sources is None else (sources[0].action, sources[1].action),
            )
            aux_obs[i] = next_aux_obs
            self._global_step += sc.num_envs

    def _register_special_opponents(self) -> None:
        """Ensure the stationary league entries exist, at the gauge's ratings.

        Idempotent, and called again after a resume restores roster.json, so a
        run resumed from a roster written before these were entries picks them
        up rather than silently losing them.

        Every stationary rating is *re-pinned* here rather than read back from
        the roster. All three kinds — random, the semi-random rungs, and
        scripted — are defined by the live gauge (config/live_elo), so a stored
        roster that disagrees is out of date, not evidence. Resuming is the case
        that matters: it is the one path where the on-disk numbers could quietly
        outrank the configured gauge.

        Every stationary player is ``fixed``: their strength does not change, so
        their ratings stay constants rather than estimates to be dragged around
        by in-training games the live policy is busy overfitting.
        """
        self.roster.pin_stationary_elo("random", LIVE_RANDOM_ELO)  # the gauge's zero
        if self.scripted_agent is None:
            return
        self.roster.add_special("scripted", initial_elo=self.cfg.elo_eval.scripted_live_elo)
        self.roster.pin_stationary_elo(  # the gauge's unit
            "scripted", self.cfg.elo_eval.scripted_live_elo
        )
        ladder = live_reference_ladder(
            self.cfg.live_reference_probabilities,
            scripted_elo=self.cfg.elo_eval.scripted_live_elo,
        )
        for p_scripted, elo in ladder:
            self.roster.add_reference(p_scripted=p_scripted, elo=elo)

    def _initialize_rollout_runtime(self) -> _RolloutRuntime:
        """Initialize persistent primary, auxiliary, and evaluation rollout state."""
        num_envs = self.cfg.scales[0].num_envs
        num_ships = self.wrapper.num_ships
        # Ships and the promoted global token carry recurrent state; map tokens
        # take the non-recurrent path. The policy owns the count.
        num_recurrent = self.policy.num_recurrent_tokens

        # Stagger truncation so episodes do not all end in one synchronized block.
        # The seeded first episode in each env is a fragment, so it is withheld
        # from the episode statistics until the env recycles -- otherwise the
        # first update of every run, and of every resume, reports a reward and
        # win rate that measure the seeding.
        beliefs = (
            DualBeliefTracker(
                num_envs,
                num_ships,
                self._decision_dt,
                self.ship_config,
                self.device,
            )
            if self._ego_pass
            else None
        )
        obs = self.wrapper.reset()
        self.wrapper.env.state.step_count.random_(0, self.env_config.max_episode_steps)
        self.wrapper.mark_seeded_uncounted()
        # The opening view is composed the same way every later one is. The reset
        # above returns a beliefless view, so recompose once the tracker has
        # assimilated the spawn -- every ship is revealed on the decision it
        # spawns, so this is a full acquisition rather than a patch.
        if beliefs is not None:
            obs = self.wrapper.observe(
                beliefs.observe(self.wrapper.env.state, self.wrapper.last_visibility.ship)
            )
        hidden = self.policy.initial_hidden(num_envs, num_recurrent, self.device)
        hidden_t1 = (
            self.policy.initial_hidden(num_envs, num_recurrent, self.device)
            if self._ego_pass
            else None
        )
        action_state = PendingActionState.allocate(num_envs, num_ships, self.device)

        aux_obs: list[YemongObservation] = []
        aux_beliefs: list[DualBeliefTracker | None] = []
        aux_hiddens: list[torch.Tensor] = []
        aux_hidden_t1s: list[torch.Tensor | None] = []
        aux_action_states: list[PendingActionState] = []
        aux_last_dones: list[torch.Tensor] = []
        for scale, wrapper in zip(self.cfg.scales[1:], self.aux_wrappers):
            raw_aux_obs = wrapper.reset()
            wrapper.env.state.step_count.random_(0, scale.env_config.max_episode_steps)
            wrapper.mark_seeded_uncounted()
            aux_belief = (
                DualBeliefTracker(
                    scale.num_envs,
                    scale.env_config.num_ships,
                    self.ship_config.dt * scale.env_config.action_repeat,
                    self.ship_config,
                    self.device,
                )
                if self._ego_pass
                else None
            )
            aux_beliefs.append(aux_belief)
            aux_obs.append(
                wrapper.observe(aux_belief.observe(wrapper.env.state, wrapper.last_visibility.ship))
                if aux_belief is not None
                else raw_aux_obs
            )
            aux_tokens = self.policy.num_recurrent_tokens
            aux_hiddens.append(self.policy.initial_hidden(scale.num_envs, aux_tokens, self.device))
            aux_hidden_t1s.append(
                self.policy.initial_hidden(scale.num_envs, aux_tokens, self.device)
                if self._ego_pass
                else None
            )
            aux_action_states.append(
                PendingActionState.allocate(
                    scale.num_envs,
                    scale.env_config.num_ships,
                    self.device,
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
            elo_eval=EloEvaluator(
                config=self.cfg.elo_eval,
                ship_config=self.ship_config,
                env_config=self.env_config,
                device=self.device,
                live_policy=self.policy,
                avg_policy=self.avg_policy,
                scripted_agent=self.scripted_agent,
                num_ships=num_ships,
                ego_pass=self._ego_pass,
                live_elo=self._live_elo,
                avg_elo=self._avg_live_elo,
                anchors=anchors,
                floating=floating,
                floating_games=self._floating_games,
                random_window=self._eval_window_rand,
                ladder_window=self._eval_window_ladder,
                floating_window=self._eval_window_floating,
                scripted_window=self._eval_window_sc,
                live_vs_avg_window=self._eval_window_live_vs_avg,
                include_bullets=eval_reads_bullets,
                compile_mode=self._env_compile_mode,
            ),
            obs=obs,
            beliefs=beliefs,
            hidden=hidden,
            hidden_t1=hidden_t1,
            action_state=action_state,
            league_slots=[],
            aux_obs=aux_obs,
            aux_beliefs=aux_beliefs,
            aux_hiddens=aux_hiddens,
            aux_hidden_t1s=aux_hidden_t1s,
            aux_action_states=aux_action_states,
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

        if not runtime.league_slots:
            runtime.league_slots = self._prepare_league_slots()
            if self._global_step > 0:
                for slot in runtime.league_slots:
                    slot.active.zero_()
        else:
            self._begin_league_replacement(runtime.league_slots)
        slots = runtime.league_slots
        # A slot drawn or replaced this shard has no view yet, and the first
        # forward of the loop runs concurrently with a physics step that makes
        # the state unreadable. Compose here, where it is settled -- and only for
        # those slots, because nothing stepped since the previous shard's last
        # decision composed the rest.
        self._compose_league_views(slots, runtime.action_state, runtime.num_ships, only_new=True)
        for rollout_step in range(self.cfg.num_steps):
            primary = self._collect_primary_step(
                obs=runtime.obs,
                beliefs=runtime.beliefs,
                hidden=runtime.hidden,
                hidden_t1=runtime.hidden_t1,
                action_state=runtime.action_state,
                num_envs=runtime.num_envs,
                num_ships=runtime.num_ships,
                num_recurrent=runtime.num_recurrent,
                slots=slots,
                env_stream=runtime.env_stream,
                net_stream=runtime.net_stream,
            )
            (
                runtime.obs,
                runtime.hidden,
                runtime.hidden_t1,
                runtime.action_state,
                terminated,
            ) = primary
            self._collect_aux_steps(
                runtime.aux_obs,
                runtime.aux_beliefs,
                runtime.aux_hiddens,
                runtime.aux_hidden_t1s,
                runtime.aux_action_states,
                runtime.aux_last_dones,
            )
            runtime.elo_eval.step(rollout_step, avg_eval_active)

        elo_snapshot = runtime.elo_eval.flush(avg_eval_active)
        self._live_elo = elo_snapshot.live_elo
        self._avg_live_elo = elo_snapshot.avg_elo
        self._floating_games = elo_snapshot.floating_games
        self._match_counts = elo_snapshot.match_counts
        if elo_snapshot.floating_label is not None:
            # The floating checkpoint's weights are fixed at snapshot time — only
            # its rating is unsettled — so its games are worth keeping forever.
            # The live and avg policies are excluded because they change under
            # the record, and a count matrix cannot say when a game was played.
            self.match_matrix.record_all(elo_snapshot.floating_label, elo_snapshot.ladder_counts)
        if elo_snapshot.floating_elo is not None:
            self.roster.set_floating_elo(elo_snapshot.floating_elo)
        # Proximity sampling reads this, so it has to track the evaluator rather
        # than keep the rating the entry was created with. The scripted entry
        # needs no such sync: the live gauge pins it and it never moves.
        self.roster.set_special_elo("avg", elo_snapshot.avg_elo)
        self._allocate_ladder_games(runtime.elo_eval)
        return terminated

    def _allocate_ladder_games(self, elo_eval: EloEvaluator) -> None:
        """Point slot 4 at the games that most sharpen the rung's floor offset.

        Recomputed every update because the target moves: the floating rung's
        rating settles, the accumulated graph gains edges, and the direct link to
        the anchor saturates. Falls back to the evaluator's local rule whenever
        the graph cannot yet identify the difference, which is the normal state
        until the new rung has played its first games.
        """
        floating_label = elo_eval.floating_label
        if not floating_label:
            return
        ratings = {entry.label: entry.elo for entry in self.roster.entries}
        ratings[floating_label] = float(elo_eval.floating_elo.item())
        ratings["scripted"] = self.cfg.elo_eval.scripted_live_elo
        candidates = elo_eval.anchor_labels()
        elo_eval.set_float_anchor_weights(
            allocation_weights(
                self.match_matrix,
                ratings,
                protagonist=floating_label,
                anchor="scripted",
                candidates=candidates,
            )
            if candidates
            else None
        )

    def _compute_rollout_gae(
        self,
        runtime: _RolloutRuntime,
        terminated: torch.Tensor,
        update_scalers: bool = True,
    ) -> None:
        """Store final observations and compute GAE for every scale.

        Args:
            runtime: Persistent environment and recurrent rollout state.
            terminated: Primary-scale episode-boundary flags (done | truncated)
                after the final step.
            update_scalers: Update statistics immediately for a single-shard batch.
                Logical host batches defer this until every shard is available.
        """
        final_truth = physical_means_from_state(self.wrapper.env.state)[:, : runtime.num_ships]
        self.buffer.store_final_obs(runtime.obs, privileged_means=final_truth)
        for index, aux_buffer in enumerate(self.aux_buffers):
            aux_buffer.store_final_obs(runtime.aux_obs[index])

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, next_value_norm, _, _ = self.policy.get_action_and_value(
                runtime.obs, runtime.hidden
            )
        self.buffer.compute_gae(self.scaler.denormalize(next_value_norm), terminated.float())
        if self._outcome_k is not None and self.cfg.outcome_categorical_coef > 0.0:
            self.buffer.fill_outcome_class(self._outcome_k)
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
            if self._outcome_k is not None and self.cfg.outcome_categorical_coef > 0.0:
                aux_buffer.fill_outcome_class(self._outcome_k)
        if update_scalers:
            self.scaler.update(self.buffer.returns, self.buffer.alive_mask)
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
            terminated = self._collect_rollout(runtime, avg_eval_active)
            self._compute_rollout_gae(runtime, terminated, update_scalers=False)
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
            [shard.alive_mask for shard in primary_shards],
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
                    device_buffer, is_primary=is_primary
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

    def _apply_schedule_state(self, step: int) -> float:
        """Resolve the schedule at ``step`` and apply every value it controls.

        Pure in the sense that matters: it reads ``step`` and the restored eval
        window and writes the coefficients, the optimizer learning rate, and the
        reward-component weights, without touching the averaging or league state
        that only an update boundary should advance. That makes it safe to call
        from ``load_checkpoint``, which must reproduce the coefficients the run
        had when it stopped rather than inherit the step-zero values ``__init__``
        computed. Returns the behavior-cloning decay factor, which the caller
        needs for the cutoff streak.
        """
        self._schedule_state = _resolve_schedule(self.cfg.schedule, step)
        self._policy_gradient_coef = self._schedule_state.policy_gradient_coef
        # BC aux loss decays linearly with the win rate against the scripted
        # agent, reaching zero at bc_winrate_target (full strength before any
        # scripted games have been recorded).
        window_sc = self._eval_window_sc
        self._scripted_win_rate = sum(window_sc) / len(window_sc) if window_sc else 0.0
        bc_factor = (
            1.0
            if self.cfg.bc_winrate_target is None
            else max(0.0, 1.0 - self._scripted_win_rate / self.cfg.bc_winrate_target)
        )
        # Tracks the current win rate, and is allowed back up.
        #
        # This was ratcheted -- min() against a running floor -- to stop the
        # coefficient oscillating with a boxcar window that wanders even over 500
        # rated games. It did stop that, and it also turned a plateau into a trap.
        # Run 740 peaked at 0.345 against the scripted agent at 45.7M steps, which
        # pinned the floor at 1 - 0.345/0.45 = 0.233 and held it there for the
        # next 28M steps while the score sat at 0.31 and never moved again. A
        # ratchet keyed on best-ever performance cannot distinguish "the warm
        # start is no longer needed" from "the policy stopped improving", and it
        # answers both by freezing.
        #
        # Oscillation is the lesser problem: it is visible in the logs and it
        # averages out. A frozen coefficient is neither.
        self._behavior_cloning_coef = self._schedule_state.behavior_cloning_coef * bc_factor
        self._entropy_coef = _actor_entropy_coef(
            self._schedule_state.entropy_coef,
            policy_gradient_coef=self._policy_gradient_coef,
            behavior_cloning_coef=self._behavior_cloning_coef,
        )
        self.optim.param_groups[0]["lr"] = self._schedule_state.learning_rate
        weights = component_weights(self.cfg.rewards)
        ratios = component_payout_ratios(self.cfg.rewards, self._schedule_state.offensive_bias)
        for wrapper in (self.wrapper, *self.aux_wrappers):
            for component in wrapper.reward_components:
                component.payout_ratio = ratios[component.name]
                tier = getattr(self._schedule_state, _TIER[component.name])
                component.weight = weights[component.name] * tier
            wrapper.refresh_component_weights()
        return bc_factor

    @torch.no_grad()
    def _seat_symmetry_metrics(self, obs, num_ships: int) -> dict[str, float]:
        """Check exact allied actions and normalized enemy beliefs for both seats."""
        if obs.team1_data is None:
            return {}
        team_id = self.wrapper.env.state.ship_team_id[:, :num_ships]
        own_invalid = 0.0
        enemy_invalid = 0.0
        for observer, view in ((0, obs.data), (1, obs.team1_data)):
            pending = view[ObsKey.PREVIOUS_ACTION][:, :num_ships].float()
            own = team_id == observer
            enemy = ~own
            invalid_distribution = (
                ~torch.isfinite(pending).all(-1)
                | (pending < 0).any(-1)
                | ~torch.isclose(
                    pending.sum(-1), torch.ones_like(pending[..., 0]), atol=2e-3, rtol=0.0
                )
            )
            own_wrong = (
                invalid_distribution
                | (pending == 1.0).sum(-1).ne(1)
                | ((pending != 0.0) & (pending != 1.0)).any(-1)
            )
            if own.any():
                own_invalid = max(own_invalid, float((own_wrong & own).sum() / own.sum()))
            if enemy.any():
                enemy_invalid = max(
                    enemy_invalid,
                    float((invalid_distribution & enemy).sum() / enemy.sum()),
                )
        return {
            "seat/own_pending_invalid": own_invalid,
            "seat/enemy_pending_invalid": enemy_invalid,
        }

    def _refresh_training_schedule(self, metrics: dict, elo_eval: EloEvaluator) -> None:
        """Refresh schedule-controlled optimization, reward, and averaging state."""
        bc_factor = self._apply_schedule_state(self._global_step)
        window_sc = self._eval_window_sc

        metrics["schedule/learning_rate"] = self._schedule_state.learning_rate
        metrics["schedule/policy_gradient_coef"] = self._policy_gradient_coef
        metrics["schedule/behavior_cloning_coef"] = self._behavior_cloning_coef
        metrics["schedule/entropy_coef"] = self._entropy_coef
        metrics["schedule/bc_decay_factor"] = bc_factor
        metrics["schedule/scripted_win_rate"] = self._scripted_win_rate
        metrics["schedule/target_kl"] = self._effective_target_kl()
        metrics["schedule/outcome_scale"] = self._schedule_state.outcome_scale
        metrics["schedule/kill_death_scale"] = self._schedule_state.kill_death_scale
        metrics["schedule/offensive_bias"] = self._schedule_state.offensive_bias
        metrics["schedule/damage_scale"] = self._schedule_state.damage_scale

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
        # No group-size gate: the average policy is rated by the evaluator in
        # every RL run, and the league draws it as an ordinary entry when one is
        # configured, so there is no longer a "reserved avg envs" count to key off.
        if self._policy_gradient_coef > 0.0:
            if self._avg_update_count > 0 or bc_cutoff_reached:
                first_avg_update = self._avg_update_count == 0
                self._update_avg_model()
                if first_avg_update:
                    elo_eval.seed_avg_elo_from_live()
                    self._avg_live_elo = self._live_elo

    def train(self) -> None:
        """Run the full PPO training loop."""
        runtime = self._initialize_rollout_runtime()
        self._train_start_time = time.time()
        self._perf_mark_time = self._train_start_time
        self._perf_mark_step = self._global_step
        self._perf_mark_ship_steps = self._ship_steps

        for update in range(self._start_update, self._num_updates + 1):
            avg_eval_active = self._avg_update_count > 0
            if self.cfg.rollouts_per_update == 1:
                terminated = self._collect_rollout(runtime, avg_eval_active)
                self._compute_rollout_gae(runtime, terminated)
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
                update=update,
            )

            self._refresh_training_schedule(metrics, runtime.elo_eval)
            metrics.update(self._seat_symmetry_metrics(runtime.obs, runtime.num_ships))
            sps, ship_tps = self._assemble_metrics(metrics, update, runtime.ship_tokens_per_update)

            self._log_training_update(metrics, update, sps, ship_tps)
            self._maybe_save_checkpoint(update)
            self._maybe_advance_ladder(update, runtime.elo_eval)
            self._completed_update = update

        self.save_final_checkpoint()
        self.record_run_status(RunStatus.COMPLETE)
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
                for microbatch in chunk.split_envs(split_count(chunk)):
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
            source_microbatches = source.split_envs(split_count(source))
            device_microbatches = staged.device.split_envs(split_count(source))
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
        pg_sum = _z.clone()
        bc_sum = _z.clone()
        ns_sum = _z.clone()
        enemy_action_sum = _z.clone()
        persistence_sum = _z.clone()
        ns_visible_sum = _z.clone()
        numel = 0
        need_bc = is_primary and self._behavior_cloning_coef > 0.0
        need_ns = is_primary and self.cfg.next_state_coef > 0.0
        for chunk in chunks:
            mb_alive = chunk.alive
            mb_actor_mask = chunk.actor_mask
            mb_decision_committed = chunk.decision_committed
            mb_expert_probs = chunk.expert_probs
            mb_terminated = chunk.terminated
            alive_sum += mb_alive.sum()
            if self.cfg.enemy_action_coef > 0.0:
                team_id = chunk.obs[ObsKey.TEAM_ID][: mb_alive.shape[0], :, : self.buffer.num_ships]
                enemy_action_sum += ((team_id == 1) & mb_alive & mb_decision_committed).sum()
                persistence_mask = torch.zeros_like(mb_alive)
                persistence_mask[1:] = (
                    (team_id[1:] == 1)
                    & mb_alive[1:]
                    & mb_decision_committed[1:]
                    & mb_alive[:-1]
                    & mb_decision_committed[:-1]
                    & chunk.transition_contiguous[:-1]
                )
                persistence_sum += persistence_mask.sum()
            actor_sum += (mb_actor_mask & mb_alive).sum()
            pg_sum += (mb_actor_mask & mb_alive & mb_decision_committed).sum()
            numel += mb_alive.numel()
            if need_bc:
                bc_valid = mb_expert_probs.sum(-1) > 0
                bc_sum += (bc_valid & mb_actor_mask & mb_alive).sum()
            if need_ns:
                belief_valid = chunk.obs[ObsKey.BELIEF_VALID][
                    : mb_alive.shape[0], :, : self.buffer.num_ships
                ].bool()
                supervised = (
                    belief_valid & ~mb_terminated.unsqueeze(-1) & chunk.transition_contiguous
                )
                ns_sum += supervised.sum()
                # Split the same supervised set by whether the ship was in sight.
                # A visible token's label is one step of real dynamics; a hidden
                # one's is mostly belief error nobody could have predicted, and
                # the aggregate series cannot tell those apart.
                ns_visible_sum += (
                    supervised
                    & chunk.obs[ObsKey.VISIBLE][
                        : mb_alive.shape[0], :, : self.buffer.num_ships
                    ].bool()
                ).sum()
        return {
            "density_sum": (
                float(sum(chunk.alive.shape[0] * chunk.alive.shape[1] for chunk in chunks))
                if is_primary and self.cfg.global_density_coef > 0.0
                else 1.0
            ),
            "mask_sum": alive_sum.clamp(min=1.0).to(self.device),
            "actor_sum": actor_sum.clamp(min=1.0).to(self.device),
            "pg_sum": pg_sum.clamp(min=1.0).to(self.device),
            "bc_sum": bc_sum.clamp(min=1.0).to(self.device),
            "ns_sum": ns_sum.clamp(min=1.0).to(self.device),
            "ns_visible_sum": ns_visible_sum.clamp(min=1.0).to(self.device),
            "ns_hidden_sum": (ns_sum - ns_visible_sum).clamp(min=1.0).to(self.device),
            "enemy_action_sum": enemy_action_sum.clamp(min=1.0).to(self.device),
            "persistence_sum": persistence_sum.clamp(min=1.0).to(self.device),
            "numel": float(numel),
            "adv_rms": buf.adv_rms,
        }

    def _compute_minibatch_loss(
        self,
        batch: MicroBatch,
        is_primary: bool,
        denoms: dict,
        frac: float,
        measure_grad_split: bool = False,
        grad_terms: TermGradientAccumulator | None = None,
        grad_scale: float = 1.0,
        evaluate_actions: Callable[..., tuple] | None = None,
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
        equivalent to one large minibatch. SIGReg is a batch statistic and
        cannot decompose that way, so it is weighted by ``frac`` instead
        (exact when the minibatch is unsplit, i.e. frac=1); it is the only
        such term, and with it disabled the accumulation is exact.

        Args:
            batch:        One micro-batch tuple from RolloutBuffer.get_minibatch_iterator.
            is_primary:   True for the primary scale — enables BC loss and per-component
                          critic diagnostics. Aux scales skip these to avoid shape mismatches
                          (different N) and because BC targets only exist in the primary env.
            denoms:       Minibatch-total denominators from _minibatch_denominators,
                          plus "adv_rms" (whole-buffer advantage normalizer).
            frac:         This micro-batch's env count / minibatch env count.
            grad_terms:   Accumulator collecting this micro-batch's per-term
                          gradients, or None (the default) for no diagnostics.
            grad_scale:   The factor the training backward applies to this
                          micro-batch's loss, so accumulated term gradients sum
                          to the gradient the optimizer step receives.
            evaluate_actions: Policy entry point to evaluate with. Defaults to
                          the run's own, which is compiled unless gradient
                          diagnostics are on. The actor/critic split probe
                          passes the eager one so it can traverse its own graph
                          twice without touching the training graph.

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
        mb_decision_committed = batch.decision_committed
        mb_expert_probs = batch.expert_probs
        mb_terminated = batch.terminated
        mb_transition_contiguous = batch.transition_contiguous
        mb_adv_agg = batch.adv_agg
        mb_ret_agg = batch.ret_agg
        mb_ns_labels = batch.ns_labels

        # mb_obs has T+1 steps; first T for encode/evaluate, last T for next-state aux loss.
        T = mb_alive.shape[0]
        curr_mb_obs = mb_obs.slice_time(0, T)

        need_sigreg = self._schedule_state.sigreg_coef > 0.0
        # evaluate_actions needs the full (T, B, N+M) alive mask so Yemong layers
        # can attend to field tokens; mb_alive is ships-only and used for loss masking.
        alive_mask_full = curr_mb_obs[ObsKey.BELIEF_VALID].bool()  # (T, B_mb, N+M)
        evaluate = evaluate_actions or self._update_evaluate_actions()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (
                logprob,
                entropy,
                new_value,
                policy_logits,
                z,
                pred_next,
                outcome_logits,
                enemy_action_logits,
                density_pred,
            ) = evaluate(
                obs=curr_mb_obs,
                actions=mb_actions.long(),
                initial_hidden=mb_hidden,
                alive_mask=alive_mask_full,
                done_mask=mb_terminated,
                return_encoder_output=need_sigreg,
                return_enemy_action=True,
                return_density=True,
            )

        alive_f = mb_alive.float()  # (T, B_mb, N)
        alive_k = alive_f.unsqueeze(-1)  # (T, B_mb, N, 1)
        mask_sum = denoms["mask_sum"]

        actor_f = (mb_actor_mask & mb_alive).float()  # (T, B_mb, N)
        actor_sum = denoms["actor_sum"]
        pg_f = (mb_actor_mask & mb_alive & mb_decision_committed).float()
        pg_sum = denoms["pg_sum"]

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
        pg_loss = (torch.max(pg_loss1, pg_loss2) * pg_f).sum() / pg_sum

        # ---- Value loss --------------------------------------------------
        target_norm = self.scaler.normalize(mb_returns).detach()  # (T, B_mb, N, K)
        vf_loss_raw = _huber(new_value - target_norm, cfg.value_huber_delta)  # (T, B_mb, N, K)
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
            p_power = mb_expert_probs[..., POWER_SLICE]
            p_turn = mb_expert_probs[..., TURN_SLICE]
            p_shoot = mb_expert_probs[..., SHOOT_SLICE]
            teacher_joint = (
                p_power.unsqueeze(-1).unsqueeze(-1)
                * p_turn.unsqueeze(-2).unsqueeze(-1)
                * p_shoot.unsqueeze(-2).unsqueeze(-2)
            ).flatten(-3)
            ce = -(teacher_joint * F.log_softmax(policy_logits, dim=-1)).sum(-1)
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
        next_state_per_feat: torch.Tensor | None = None  # (pred_dim,) gpu, for logging
        next_state_visible_per_feat: torch.Tensor | None = None  # (pred_dim,) gpu
        next_state_hidden_per_feat: torch.Tensor | None = None  # (pred_dim,) gpu
        label_sq_per_feat: torch.Tensor | None = None  # (pred_dim,) gpu, for logging
        # Per-population calibration sums, (3, 4, 11) and (3,), float64 on gpu.
        ns_population_sums: torch.Tensor | None = None
        ns_population_counts: torch.Tensor | None = None
        ns_population_terms: dict[str, torch.Tensor] = {}
        _need_aux = is_primary and self.cfg.next_state_coef > 0.0
        if _need_aux:
            non_terminal = ~mb_terminated.unsqueeze(-1)  # (T, B_mb, 1)
            # Privileged dynamics supervision covers visible and previously-seen
            # hidden tokens. Never-seen enemies do not exist in the policy input,
            # so training their unknowable state would add contradictory noise.
            belief_valid = curr_mb_obs[ObsKey.BELIEF_VALID][:, :, : self.buffer.num_ships].bool()
            ns_mask = belief_valid & non_terminal & mb_transition_contiguous
            ns_mask_f = ns_mask.float()
            ns_sum = denoms["ns_sum"]

            # Labels precomputed once per update from the T+1 obs storage
            # (see _precompute_ns_labels) — they depend only on rollout data.
            labels = mb_ns_labels  # (T, B_mb, N, pred_dim)

            P = PHYSICAL_MEAN_DIM
            # Gaussian negative log likelihood over the eleven physical deltas:
            # full 2D covariance for position and velocity, scalar for the rest.
            # It is scale-free in the label, so nothing here depends on a fitted
            # weight being right, and a token whose label is mostly unpredictable
            # belief error earns a wide sigma instead of dominating the sum.
            per_dim = self.next_state.loss(
                pred_next.float(), labels.detach(), beta=self.cfg.next_state_beta
            )

            if self.cfg.next_state_coef > 0.0:
                next_state_cont_loss = (per_dim * ns_mask_f.unsqueeze(-1)).sum() / (ns_sum * P)
                next_state_loss = next_state_cont_loss

            # Allies, visible enemies, and hidden enemies carry labels that mean
            # different things (see next_state_populations), so the objective's
            # balance between them is measured rather than read off the sum.
            populations = population_masks(
                ns_mask,
                curr_mb_obs[ObsKey.TEAM_ID][:, :, : self.buffer.num_ships],
                curr_mb_obs[ObsKey.VISIBLE][:, :, : self.buffer.num_ships].bool(),
            )  # (3, T, B_mb, N)
            ns_population_sums, ns_population_counts = population_moments(
                self.next_state, pred_next, labels.detach(), per_dim, populations
            )
            if grad_terms is not None and self._grad_diag.decomposes_next_state_by_population:
                ns_population_terms = gradient_terms(
                    candidate_losses(self.next_state, pred_next.float(), labels.detach(), per_dim),
                    populations,
                    environment_halves(populations.shape[2], populations.device),
                    ns_sum,
                    self.cfg.next_state_coef,
                )

            with torch.no_grad():
                # Squared error, not the objective: this series predates the NLL
                # and has to keep meaning the same thing across the change, and a
                # likelihood is not an error anyone can read in physical units.
                sq_err = self.next_state.residual(pred_next.float(), labels.detach()).pow(2)
                next_state_per_feat = (sq_err * ns_mask_f.unsqueeze(-1)).sum(
                    (0, 1, 2)
                ) / ns_sum  # (pred_dim,) gpu, additive across chunks
                # The same error split by whether the ship was in sight. The two
                # halves answer different questions: a visible token's label is
                # one step of real dynamics and measures the learned model, while
                # a hidden one's is dominated by belief error and measures how far
                # the recursion has drifted. Aggregated they are a mixture whose
                # proportions move with the fog, so neither is readable alone.
                #
                # Masked reductions over a tensor already materialised for the
                # aggregate, so the cost is two more passes over ``sq_err`` and
                # no additional forward or backward work.
                visible_f = (
                    curr_mb_obs[ObsKey.VISIBLE][:, :, : self.buffer.num_ships].bool() & ns_mask
                ).float()
                next_state_visible_per_feat = (sq_err * visible_f.unsqueeze(-1)).sum(
                    (0, 1, 2)
                ) / denoms["ns_visible_sum"]
                next_state_hidden_per_feat = (sq_err * (ns_mask_f - visible_f).unsqueeze(-1)).sum(
                    (0, 1, 2)
                ) / denoms["ns_hidden_sum"]
                # Mean square of the *label* itself, against the fixed Phase-1
                # scales: a channel whose scale still conditions its labels has
                # mean square near 1, and the null model scores 1.
                #
                # Worth logging rather than measuring offline because the label
                # steps from the *believed* state, so its spread depends on how
                # good this policy's own next-state head currently is. The
                # Phase-1 calibration measured truth-to-truth deltas, which is
                # the visible half of this; the series shows how far the hidden
                # half has moved away from it. The constants themselves stay
                # fixed -- there is no online scaler by contract -- so this is a
                # diagnostic, not a control input.
                label_sq_per_feat = (labels.detach().float().pow(2) * ns_mask_f.unsqueeze(-1)).sum(
                    (0, 1, 2)
                ) / ns_sum

        # ---- Match-outcome classification ---------------------------------
        outcome_ce_loss = policy_logits.new_zeros(())
        zero = policy_logits.new_zeros(())
        diag_outcome = {"outcome_ce": zero, "outcome_correct": zero, "outcome_labelled": zero}
        if self.cfg.outcome_categorical_coef > 0.0:
            outcome_ce_loss, outcome_graded, outcome_target_class = self._outcome_categorical_loss(
                outcome_logits, batch.outcome_class, alive_f, mask_sum
            )
            with torch.no_grad():
                # Divide by the minibatch-total token count because additive
                # diagnostics sum across micro-batches before finalization.
                graded = outcome_graded & mb_alive
                outcome_predicted = outcome_logits.argmax(-1).to(batch.outcome_class.dtype)
                correct = (outcome_predicted == outcome_target_class) & graded
                numel = denoms["numel"]
                diag_outcome.update(
                    {
                        "outcome_ce": outcome_ce_loss.detach(),
                        "outcome_labelled": graded.sum() / numel,
                        "outcome_correct": correct.sum() / numel,
                    }
                )

        # ---- Enemy pending-action prediction -------------------------------
        enemy_action_loss = self._zero_tensor
        enemy_action_probability = self._zero_tensor
        enemy_action_entropy = self._zero_tensor
        enemy_action_brier = self._zero_tensor
        enemy_action_accuracy = self._zero_tensor
        persistence_accuracy = self._zero_tensor
        persistence_ce = self._zero_tensor
        if self.cfg.enemy_action_coef > 0.0:
            team_id = curr_mb_obs[ObsKey.TEAM_ID][:, :, : self.buffer.num_ships]
            enemy_mask = (team_id == 1) & mb_alive & mb_decision_committed
            enemy_f = enemy_mask.float()
            enemy_sum = denoms["enemy_action_sum"]
            action_target = encode_joint_action_unchecked(mb_actions)
            per_token_ce = F.cross_entropy(
                enemy_action_logits.float().flatten(0, -2),
                action_target.flatten(),
                reduction="none",
            ).reshape_as(action_target)
            enemy_action_loss = (per_token_ce * enemy_f).sum() / enemy_sum
            with torch.no_grad():
                probabilities = F.softmax(enemy_action_logits.float(), dim=-1)
                realised_probability = probabilities.gather(
                    -1, action_target.unsqueeze(-1)
                ).squeeze(-1)
                entropy = Categorical(probs=probabilities).entropy()
                target_one_hot = F.one_hot(action_target, NUM_JOINT_ACTIONS).to(probabilities.dtype)
                brier = (probabilities - target_one_hot).pow(2).sum(-1)
                predicted = probabilities.argmax(-1)
                previous_action = torch.roll(action_target, shifts=1, dims=0)
                persistence_mask = torch.zeros_like(enemy_mask)
                persistence_mask[1:] = (
                    enemy_mask[1:]
                    & mb_alive[:-1]
                    & mb_decision_committed[:-1]
                    & mb_transition_contiguous[:-1]
                )
                persistence_f = persistence_mask.float()
                same = previous_action == action_target
                persistence_probability = torch.where(
                    same,
                    torch.full_like(realised_probability, 0.99),
                    torch.full_like(realised_probability, 0.01 / (NUM_JOINT_ACTIONS - 1)),
                )
                enemy_action_probability = (realised_probability * enemy_f).sum() / enemy_sum
                enemy_action_entropy = (entropy * enemy_f).sum() / enemy_sum
                enemy_action_brier = (brier * enemy_f).sum() / enemy_sum
                enemy_action_accuracy = (
                    (predicted == action_target).float() * enemy_f
                ).sum() / enemy_sum
                persistence_sum = denoms["persistence_sum"]
                persistence_accuracy = (same.float() * persistence_f).sum() / persistence_sum
                persistence_ce = (
                    -persistence_probability.log() * persistence_f
                ).sum() / persistence_sum

        # ---- Global density prediction (primary scale only) -----------------
        density = _DensityStats.zeros(self._zero_tensor)
        if is_primary and self.cfg.global_density_coef > 0.0:
            density = self._global_density_loss(
                density_pred, batch.density_targets, denoms["density_sum"]
            )
        density_loss = density.loss

        loss = (
            self.cfg.outcome_categorical_coef * outcome_ce_loss
            + self._policy_gradient_coef * pg_loss
            + self._schedule_state.value_function_coef * vf_loss
            + self._entropy_coef * ent_loss
            + self._behavior_cloning_coef * bc_loss
            + self.cfg.enemy_action_coef * enemy_action_loss
            + self._schedule_state.sigreg_coef * sigreg_loss
            + self.cfg.next_state_coef * next_state_loss
            + self.cfg.global_density_coef * density_loss
        )

        diag: dict = dict(diag_outcome)

        # ---- Gradient decomposition -------------------------------------------
        # Differentiates the weighted terms that make up `loss` above, one
        # autograd traversal each, and hands the results to the accumulator that
        # spans this optimizer minibatch. Everything here is opt-in: with
        # grad_terms None not a single extra graph node is built.
        if grad_terms is not None:
            terms = {
                "policy": self._policy_gradient_coef * pg_loss,
                "value": self._schedule_state.value_function_coef * vf_loss,
                "outcome": self.cfg.outcome_categorical_coef * outcome_ce_loss,
                "enemy_action": self.cfg.enemy_action_coef * enemy_action_loss,
                "entropy": self._entropy_coef * ent_loss,
                "bc": self._behavior_cloning_coef * bc_loss,
                "sigreg": self._schedule_state.sigreg_coef * sigreg_loss,
                "next_state": self.cfg.next_state_coef * next_state_loss,
                "density": self.cfg.global_density_coef * density_loss,
            }
            if self._grad_diag.decomposes_policy_by_reward:
                terms.update(
                    self._reward_policy_terms(
                        batch=batch,
                        ratio=ratio,
                        adv_norm=adv_norm,
                        pg_f=pg_f,
                        pg_sum=pg_sum,
                        adv_rms=denoms["adv_rms"],
                    )
                )
            if self._grad_diag.decomposes_value_by_reward:
                terms.update(
                    self._reward_value_terms(
                        vf_loss_raw=vf_loss_raw,
                        alive_k=alive_k,
                        mask_sum=mask_sum,
                        num_components=K,
                    )
                )
            terms.update(ns_population_terms)
            grad_terms.accumulate(terms, scale=grad_scale)

        # ---- Actor / critic gradient split ------------------------------------
        # Both terms land on the same trunk, so max_grad_norm renormalizes them
        # together: whichever sends more gradient takes a larger share of every
        # clipped step, and the other loses it. Inferring that split from the
        # total norm is how a 3.4x imbalance in the categorical critic's favor
        # survived two full runs unnoticed. Costs two extra backward passes, so
        # it runs on one micro-batch per update at the histogram cadence.
        if measure_grad_split:
            params = [p for p in self._policy_module.parameters() if p.requires_grad]
            terms = {
                "actor": self._policy_gradient_coef * pg_loss + self._entropy_coef * ent_loss,
                "critic": self._schedule_state.value_function_coef * vf_loss,
            }
            for term_name, term in terms.items():
                grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
                present = [g.norm() for g in grads if g is not None]
                diag[f"grad_norm_{term_name}"] = (
                    torch.stack(present).norm() if present else self._zero_tensor
                )

        # ---- Diagnostics (no grad) — kept as GPU tensors, .item() deferred to logging ----
        with torch.no_grad():
            diag["loss"] = loss.detach()
            diag["pg_loss"] = pg_loss.detach()
            diag["vf_loss"] = vf_loss.detach()
            diag["ent_loss"] = ent_loss.detach()
            diag["bc_loss"] = bc_loss.detach()
            diag["sigreg_loss"] = sigreg_loss.detach()
            diag["next_state_loss"] = next_state_loss.detach()
            diag["density_loss"] = density.loss.detach()
            diag["density_deviance"] = density.deviance
            diag["density_ally_loss"] = density.ally_loss.detach()
            diag["density_enemy_loss"] = density.enemy_loss.detach()
            diag["density_ally_deviance"] = density.ally_deviance
            diag["density_enemy_deviance"] = density.enemy_deviance
            diag["next_state_cont_loss"] = next_state_cont_loss.detach()
            diag["next_state_per_feat"] = next_state_per_feat  # (pred_dim,) gpu or None
            diag["next_state_visible_per_feat"] = next_state_visible_per_feat
            diag["next_state_hidden_per_feat"] = next_state_hidden_per_feat
            diag["label_sq_per_feat"] = label_sq_per_feat  # (pred_dim,) gpu or None
            diag["ns_population_sums"] = ns_population_sums  # (3, 4, 11) gpu or None
            diag["ns_population_counts"] = ns_population_counts  # (3,) gpu or None
            diag["scripted_entropy"] = scripted_entropy.detach()
            diag["bc_kl"] = bc_loss.detach() - scripted_entropy.detach()
            diag["approx_kl"] = (((ratio - 1) - log_ratio) * pg_f).sum() / pg_sum
            diag["clip_frac"] = (((ratio - 1).abs() > cfg.clip_coef).float() * pg_f).sum() / pg_sum
            diag["alive_frac"] = alive_f.sum() / denoms["numel"]
            diag["enemy_action_loss"] = enemy_action_loss.detach()
            diag["enemy_action_probability"] = enemy_action_probability.detach()
            diag["enemy_action_entropy"] = enemy_action_entropy.detach()
            diag["enemy_action_brier"] = enemy_action_brier.detach()
            diag["enemy_action_accuracy"] = enemy_action_accuracy.detach()
            diag["enemy_action_persistence_accuracy"] = persistence_accuracy.detach()
            diag["enemy_action_persistence_ce"] = persistence_ce.detach()
            diag["ratio_mean"] = (ratio * pg_f).sum() / pg_sum
            diag["ratio_max"] = ratio.max()  # combine across chunks with max, not sum

            # Marginals are diagnostics only. The objective entropy above is the
            # joint categorical entropy; summing these would be wrong once the
            # actor learns correlations among action factors.
            #
            # fp32 before the softmax, because ``Categorical`` validates that its
            # ``probs`` lie on the simplex and bf16 marginals do not: summing six
            # bf16 probabilities to get the turn marginal lands a few thousandths
            # either side of one, and the check raises rather than tolerating it.
            # The block is under ``no_grad``, so the upcast costs nothing that
            # reaches the backward pass. Diagnostics must not be able to stop a
            # run.
            joint_prob = F.softmax(policy_logits.float(), dim=-1).reshape(
                *policy_logits.shape[:-1],
                NUM_POWER_ACTIONS,
                NUM_TURN_ACTIONS,
                NUM_SHOOT_ACTIONS,
            )
            power_ent = Categorical(probs=joint_prob.sum(dim=(-1, -2))).entropy()
            turn_ent = Categorical(probs=joint_prob.sum(dim=(-3, -1))).entropy()
            shoot_ent = Categorical(probs=joint_prob.sum(dim=(-3, -2))).entropy()
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

    def _global_density_loss(
        self,
        prediction: torch.Tensor | None,
        target: torch.Tensor | None,
        density_sum: float,
    ) -> "_DensityStats":
        """Poisson negative log likelihood of the hex ship-count field.

        The target is a soft count per cell -- every living ship deposits one
        unit, split barycentrically across the three cells whose centres form
        the triangle containing it -- so the field sums to the living count per
        side, not to one. The head emits a log-rate and the likelihood is

            L = sum_c [ exp(l_c) - y_c * l_c ]

        dropping the ``log y!`` constant, which does not depend on the model.
        Its gradient in the logit is ``exp(l_c) - y_c``: bounded, and zero
        exactly when the rate matches the count. That is the whole reason this
        is a Poisson head rather than a Gaussian one -- the next-state head's
        ``r / sigma^2`` is what concentrates gradient on whatever it already
        predicts best, and this form cannot.

        Averaged over cells and summed over steps and environments, then
        divided by the minibatch's total step-environment count, so a
        micro-batch's contribution is additive exactly as every other
        masked-mean term is.

        Every step has a target -- the field is a property of the state the
        observation was built from, not of a transition -- so there is nothing
        to mask here. A step whose episode ends still has a legitimate field.

        Returns:
            :class:`_DensityStats` -- the loss and the **deviance**, and both
            again for the ally and enemy halves alone.

            A Poisson likelihood has no natural zero: its value depends on the
            counts as well as on the fit, so the bare number says nothing about
            how good the head is. The deviance subtracts the likelihood of a
            perfect predictor (``mu_c = y_c``), leaving excess nats per cell
            that are zero when the head is exact and positive otherwise. That
            is the series to read; the loss is what is optimized.

            The halves are reported apart because they are not the same
            problem. The observer sees every ally, so that field is a
            restatement of what it already knows; the enemy field is mostly
            belief. One mean over all ``2C`` cells lets the easy half carry the
            hard one.

        Raises:
            ValueError: If the head or the target is missing while the
                coefficient asks for the loss.
        """

        if prediction is None or target is None:
            raise ValueError(
                "global_density_coef is positive but this rollout carries no "
                "density head or no density target"
            )
        counts = target.float()  # (T, B, 2C)
        log_rate = prediction.float()
        # Clamped only against overflow; the head is free inside this range and
        # the bound is far outside any rate the grid can legitimately carry
        # (e2 cells hold every ship in the game; e-20 is empty to any precision).
        log_rate = log_rate.clamp(-20.0, 2.0)
        per_cell = log_rate.exp() - counts * log_rate  # (T, B, 2C)
        with torch.no_grad():
            # The same likelihood at mu = y, which is its minimum. ``y log y``
            # is taken as zero at y = 0, its limit, which is also almost every
            # cell.
            saturated = counts - counts * torch.log(counts.clamp(min=1e-12))
            saturated = torch.where(counts > 0.0, saturated, torch.zeros_like(saturated))

        cells = counts.shape[-1] // 2

        def halves(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Per-half cell means, summed over steps and envs and normalized."""
            ally = values[..., :cells].mean(-1).sum() / density_sum
            enemy = values[..., cells:].mean(-1).sum() / density_sum
            return ally, enemy

        ally_loss, enemy_loss = halves(per_cell)
        with torch.no_grad():
            ally_floor, enemy_floor = halves(saturated)
        # The mean over 2C equally weighted cells is the mean of the two half
        # means, so the training term stays a plain mean over the field.
        return _DensityStats(
            loss=0.5 * (ally_loss + enemy_loss),
            deviance=0.5 * (ally_loss + enemy_loss).detach() - 0.5 * (ally_floor + enemy_floor),
            ally_loss=ally_loss,
            enemy_loss=enemy_loss,
            ally_deviance=ally_loss.detach() - ally_floor,
            enemy_deviance=enemy_loss.detach() - enemy_floor,
        )

    def _outcome_categorical_loss(
        self,
        logits: torch.Tensor,
        outcome_class: torch.Tensor,
        alive_f: torch.Tensor,
        mask_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy for the win / loss / tie head.

        With ``outcome``'s gamma at 1.0 and no reward before the terminal, the
        return from any state *is* the match result, so the distributional
        Bellman backup degenerates to "carry the terminal one-hot backwards".
        No projection step, no discounting to shift the support -- the atoms are
        the three real outcomes and they do not move.

        A step whose episode ends inside the rollout takes the nearest *future*
        terminal, which is what keeps a step landing after one episode end from
        inheriting the previous episode's result. Steps with no terminal ahead
        of them inside the chunk are **not graded at all**.

        They used to bootstrap from the head's own belief at the chunk's last
        step, detached, to densify a signal that is otherwise sparse -- an
        episode runs about 8,700 steps and a chunk is 128, so under 1% of steps
        carry a realised result. That densification had a fixed point. Every
        unlabelled step took the *same* target, ``logits[-1]``, so any
        prediction constant across the chunk satisfied it exactly; and for
        softmax cross-entropy ``d/dlogits`` is ``p - target``, which at such a
        fixed point is identically zero. Uniform is one of those fixed points
        and is where orthogonal init starts. Run 748 sat there for 84.5M steps:
        cross-entropy pinned at ln 3 = 1.0986 to four decimals, accuracy at
        chance, and 5.6e-05 of trunk gradient -- 1/200,000 of behaviour
        cloning's. The 1% of real labels never outweighed the pull back to it,
        and were further divided by a denominator counting every alive token.

        So the bootstrap is gone and the mean is over graded tokens only. The
        signal is sparse -- a few thousand tokens an update -- but it is signal,
        and the reported number is now cross-entropy against realised results,
        which is comparable to ``ln 3`` as a bar rather than being pinned to it.

        Args:
            logits:        (T, B, N, 3) head output.
            outcome_class: (T, B, N) int8 realised result, -1 where unknown.
            alive_f:       (T, B, N) float liveness mask.
            mask_sum:      Unused; the mean is over graded tokens, which is a
                count this function alone knows. Kept so the call site reads
                like the other masked means.

        Returns:
            ``(loss, graded, graded_class)`` -- the cross-entropy averaged over
            graded tokens, the mask of tokens that had a realised result ahead
            of them, and the class each was graded against.
        """

        horizon = logits.shape[0]
        with torch.no_grad():
            labelled = outcome_class >= 0  # (T, B, N)
            step = torch.arange(horizon, device=logits.device).view(-1, 1, 1)
            # Nearest labelled step at or after t; `horizon` where there is none.
            reach = torch.where(labelled, step.expand_as(labelled), horizon)
            nearest = reach.flip(0).cummin(0).values.flip(0)  # (T, B, N)
            found = nearest < horizon
            realised = torch.nn.functional.one_hot(
                outcome_class.clamp(min=0).long(), NUM_OUTCOME_CLASSES
            ).float()  # (T, B, N, 3)
            carried = torch.gather(
                realised,
                0,
                nearest.clamp(max=horizon - 1)
                .unsqueeze(-1)
                .expand(-1, -1, -1, NUM_OUTCOME_CLASSES),
            )
            graded_f = (found & (alive_f > 0.0)).float()  # (T, B, N)
            # Every micro-batch of a minibatch divides by its own graded count.
            # The alternative, a shared minibatch total, would need a count this
            # loss cannot see; the head is a probe on a sparse label and the
            # slight non-additivity across micro-batches costs it nothing.
            graded_sum = graded_f.sum().clamp(min=1.0)

        log_probabilities = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        cross_entropy = -(carried * log_probabilities).sum(-1)  # (T, B, N)
        graded_class = torch.gather(
            outcome_class.clamp(min=0).long(), 0, nearest.clamp(max=horizon - 1)
        ).to(outcome_class.dtype)
        return (cross_entropy * graded_f).sum() / graded_sum, found, graded_class

    def _reward_policy_terms(
        self,
        *,
        batch: MicroBatch,
        ratio: torch.Tensor,
        adv_norm: torch.Tensor,
        pg_f: torch.Tensor,
        pg_sum: torch.Tensor,
        adv_rms: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Split this micro-batch's policy loss across reward components.

        PPO's clipping decision belongs to the aggregate objective: the ratio is
        clipped or not for a *token*, not for a reward. Choosing a branch per
        component would let each component pick the branch that flatters it, and
        the parts would no longer add up to the update being run. So the branch
        the aggregate objective selected is taken as given, and only the
        advantage is decomposed:

            pg = -sum_k adv_k * ratio_selected,   sum_k adv_k = adv_aggregate

        which makes the component gradients an exact linear attribution of the
        real policy gradient rather than an approximation of it.

        Args:
            batch:     The micro-batch being differentiated.
            ratio:     (T, b, N) new/old probability ratio.
            adv_norm:  (T, b, N) normalized aggregate advantage — the quantity
                       the live objective uses.
            pg_f:      (T, b, N) causal policy-gradient mask.
            pg_sum:    Minibatch-total committed policy-decision count.
            adv_rms:   Whole-buffer aggregated-advantage mean square.

        Returns:
            Term name → weighted scalar loss, one per component with a non-zero
            weight. Components scheduled to zero contribute no gradient and are
            left out rather than logged as an empty series.
        """
        T, _, N = batch.alive.shape
        team_id = batch.obs[ObsKey.TEAM_ID][:T, :, :N].long()  # (T, b, N)
        with torch.no_grad():
            lambda_ij = self._team_mixing(team_id, batch.alive)
            adv_normed = self.adv_scaler.normalize(batch.advantages)  # (T, b, N, K)
            # Same aggregation as _precompute_lambda_aggregates, minus the sum
            # over components: adv_agg_k.sum(-1) is the adv_agg it produced.
            adv_agg_k = torch.einsum("tbijk,tbjk->tbik", lambda_ij, adv_normed)  # (T, b, N, K)
            adv_norm_k = adv_agg_k / (adv_rms.sqrt().clamp(min=0.1) + 1e-8)  # (T, b, N, K)

        clipped = ratio.clamp(1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef)  # (T, b, N)
        # torch.max(-A*r, -A*clip(r)) selects a branch; reproduce that selection
        # from the aggregate advantage and reuse it for every component.
        use_clipped = (-adv_norm * clipped) > (-adv_norm * ratio)  # (T, b, N)
        ratio_selected = torch.where(use_clipped, clipped, ratio)  # (T, b, N)

        weighted = (ratio_selected * pg_f).unsqueeze(-1)  # (T, b, N, 1)
        per_component = -(adv_norm_k * weighted).sum((0, 1, 2)) / pg_sum  # (K,)
        coefficient = self._policy_gradient_coef
        return {
            f"policy/{name}": coefficient * per_component[index]
            for index, name in enumerate(self._active_names)
            if self.wrapper.active_components[index].weight != 0.0
        }

    def _reward_value_terms(
        self,
        *,
        vf_loss_raw: torch.Tensor,
        alive_k: torch.Tensor,
        mask_sum: torch.Tensor,
        num_components: int,
    ) -> dict[str, torch.Tensor]:
        """Split this micro-batch's critic loss across reward components.

        The critic objective is already a sum of independent per-component
        squared errors, so this is the existing loss regrouped rather than a
        second objective: the components sum back to ``vf_loss`` by construction.

        Args:
            vf_loss_raw:    (T, b, N, K) per-component squared critic error.
            alive_k:        (T, b, N, 1) float alive mask.
            mask_sum:       Minibatch-total alive token count.
            num_components: K — the critic's own averaging divisor.

        Returns:
            Term name → weighted scalar loss, one per active component.
        """
        per_component = (vf_loss_raw * alive_k).sum((0, 1, 2)) / (mask_sum * num_components)  # (K,)
        coefficient = self._schedule_state.value_function_coef
        return {
            f"value/{name}": coefficient * per_component[index]
            for index, name in enumerate(self._active_names)
        }

    def _team_mixing(self, team_id: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        """Per-level advantage mixing: own advantage plus team spirit times the team's.

        ``adv_i + s_k * mean_{j != i, same side, alive} adv_j``. With every
        ``s_k`` zero this is the identity: each ship keeps the credit its
        rewards already assigned it.

        Shared by the per-update aggregation and by the reward-decomposed
        gradient diagnostic, so the diagnostic cannot drift from the credit
        assignment the policy gradient actually used.

        Args:
            team_id: (T, b, N) long — raw team labels.
            alive:   (T, b, N) bool — living ships.

        Returns:
            (T, b, N_i, N_j, K) float32 mixing tensor.
        """
        N = alive.shape[-1]
        itself = torch.eye(N, dtype=torch.bool, device=self.device)
        teammates = (team_id.unsqueeze(3) == team_id.unsqueeze(2)) & ~itself  # (T, b, N, N)
        teammates = teammates & alive.unsqueeze(2)
        mean = teammates.float() / teammates.sum(3, keepdim=True).clamp(min=1)
        own = itself.float()[None, None, :, :, None]  # (1, 1, N, N, 1)
        return own + mean.unsqueeze(-1) * self._team_spirit_k  # (T, b, N, N, K)

    @torch.no_grad()
    def _precompute_lambda_aggregates(
        self, buf: RolloutBuffer, is_primary: bool
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
            lambda_ij_t = self._team_mixing(team_id_t, alive)

            # advantages/returns are bf16-stored; normalize() promotes advantages via
            # the fp32 rms divisor, and returns is upcast explicitly so the einsum with
            # the fp32 lambda tensor stays fp32 (einsum will not mix dtypes).
            adv_normed = self.adv_scaler.normalize(buf.advantages[:, sl])
            returns_sl = buf.returns[:, sl].float()
            buf.adv_agg[:, sl] = torch.einsum("tbijk,tbjk->tbi", lambda_ij_t, adv_normed)
            buf.ret_agg[:, sl] = torch.einsum("tbijk,tbjk->tbi", lambda_ij_t, returns_sl)

            actor_f = (buf.actor_masks[:, sl] & buf.decision_committed[:, sl] & alive).float()
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

        Labels come from the stored T+1 observations and the privileged physical
        truth beside them -- not the policy -- so computing them here saves
        num_epochs x num_minibatches redundant passes.

        The label is the normalized physical step from the *believed* current
        state to the *true* next state, not truth to truth. That is what the
        head's output is used for: ``BeliefTracker.advance`` applies the forecast
        to the belief, so ``belief[t+1] = belief[t] + pred[t]``. Training it on
        ``true[t+1] - true[t]`` instead makes the substitution

            error[t+1] = belief[t] + (true[t+1] - true[t]) - true[t+1] = error[t]

        -- the belief error is conserved exactly, every step's noise is retained
        forever, and the head is never once shown what "too far" looks like. The
        drift that killed run 734 (velocity error 99 -> 1178 px/s in the 30s+
        hidden bucket, then a non-finite logit) is that identity, not an
        incidental instability.

        Re-basing on the belief makes the target the correction that carries the
        believed state onto the true next one, so error is nulled each step to
        whatever extent it is inferable. For a ship the observer can see, the
        belief *is* truth, so its label is exactly the Phase-1 truth-to-truth
        delta the scales were calibrated on: this adds signal where the drift
        happens and leaves the rest of the supervision alone.

        The residual is not fully predictable, so the head regresses toward the
        conditional mean of the correction -- shrinkage of a stale belief toward
        the prior, which is the right behaviour for a mean estimate. The
        likelihood is what makes that safe: a token whose label is mostly
        unpredictable belief error earns a wide sigma rather than dominating the
        sum, so the objective is comparable across visible and hidden tokens
        without any weight being tuned.
        """
        T = buf.num_steps
        believed = self._believed_means(buf, T + 1)
        # Ground truth where the rollout captured it. Without it there is no
        # privileged signal to correct towards and belief is the only account of
        # the world, which recovers the original truth-to-truth labels.
        truth = buf.privileged_means if buf.privileged_means is not None else believed
        buf.ns_labels = (  # (T, B, N, 11)
            self.next_state.labels(believed[:T], truth[1:])
            if self.cfg.next_state_coef > 0.0
            else None
        )
        # Diagnostics compare forecasts against hidden *truth*, so they keep
        # reading the privileged means rather than the re-based labels.
        self._precompute_belief_diagnostics(buf, believed, truth)

    def _believed_means(self, buf: RolloutBuffer, steps: int) -> torch.Tensor:
        """``(steps, B, N, 11)`` physical state the stored observations carry."""

        N = buf.num_ships
        return physical_means_from_observation(
            YemongObservation(data={key: value[:steps] for key, value in buf.obs.items()}),
            self._index_log_scale,
            num_ships=N,
        ).float()

    @torch.no_grad()
    def _precompute_belief_diagnostics(
        self,
        buf: RolloutBuffer,
        believed: torch.Tensor,
        truth: torch.Tensor,
    ) -> None:
        """Compare behavior-policy forecasts with hidden truth in physical units.

        Every series is the absolute error of a physical quantity in its own
        unit, so it is directly comparable with the persistence and
        dead-reckoning baselines reported beside it. There is no decode step:
        belief and truth are the same eleven numbers.
        """

        if buf.rollout_predictions is None:
            buf.belief_diagnostics = {}
            return
        T, N = buf.num_steps, buf.num_ships
        current = believed[:T]
        truth_next = truth[1:]
        forecast = self.next_state.apply_means(current, buf.rollout_predictions.float())
        # Two baselines the model has to beat to be worth its cost. Persistence
        # is the belief standing still; dead reckoning carries it forward on its
        # own believed velocity for one decision, which is the strongest thing
        # available without a learned model.
        persistence = current
        reckoned = current.clone()
        reckoned[..., POSITION_X] = current[..., POSITION_X] + (
            current[..., VELOCITY_X] * self._decision_dt
        )
        reckoned[..., POSITION_Y] = current[..., POSITION_Y] + (
            current[..., VELOCITY_Y] * self._decision_dt
        )
        errors = self._physical_errors(forecast, truth_next)
        errors.update(
            {
                f"persist_{name}": value
                for name, value in self._physical_errors(persistence, truth_next).items()
                if name in ("position_px", "velocity_px_s")
            }
        )
        errors["reckon_position_px"] = self._physical_errors(reckoned, truth_next)["position_px"]

        visible = buf.obs[ObsKey.VISIBLE][:T, :, :N].bool()
        valid = buf.obs[ObsKey.BELIEF_VALID][:T, :, :N].bool()
        enemy = buf.obs[ObsKey.TEAM_ID][:T, :, :N] == 1
        transition = ~buf.terminated.unsqueeze(-1) & buf.transition_contiguous
        visible_enemy = visible & enemy & transition
        hidden_enemy = ~visible & valid & enemy & transition
        age = buf.obs[ObsKey.TIME_SINCE_OBSERVATION][:T, :, :N, 0].float()

        diagnostics: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        def add(prefix: str, mask: torch.Tensor) -> None:
            count = mask.sum().float()
            for name, error in errors.items():
                diagnostics[f"belief/{prefix}/{name}"] = ((error * mask).sum(), count)

        add("visible", visible_enemy)
        add("hidden", hidden_enemy)
        lower = 0.0
        for upper in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0):
            bucket = hidden_enemy & (age > lower) & (age <= upper)
            add(f"hidden_age_{lower:g}_{upper:g}s", bucket)
            lower = upper
        add("hidden_age_30_inf_s", hidden_enemy & (age > 30.0))
        buf.belief_diagnostics = diagnostics

    def _physical_errors(
        self, estimate: torch.Tensor, truth: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Absolute physical error per channel, position and velocity as norms."""

        world = torch.tensor(self.ship_config.world_size, device=estimate.device)
        pos_delta = (
            torch.remainder(
                estimate[..., POSITION_X : POSITION_Y + 1]
                - truth[..., POSITION_X : POSITION_Y + 1]
                + world / 2.0,
                world,
            )
            - world / 2.0
        )
        position_error = pos_delta.norm(dim=-1)
        # Ten calibrated position scales is an empirical outlier threshold, not
        # a physics limit. The production scripted calibration's largest valid
        # component was 11.32 px; even the worst per-axis bound implies a
        # 16.01 px norm against this 25 px threshold. On a trained head, a
        # visible-cell excursion beyond it is therefore a useful lifecycle
        # leak signal; on a fresh head it still measures ordinary forecast error.
        # Hidden-cell excursions measure legitimate recursive drift.
        leak_threshold = 10.0 * PHYSICAL_DELTA_SCALES[POSITION_X]
        return {
            "position_px": position_error,
            "position_beyond_legal_frac": (position_error > leak_threshold).float(),
            "velocity_px_s": (
                estimate[..., VELOCITY_X : VELOCITY_Y + 1] - truth[..., VELOCITY_X : VELOCITY_Y + 1]
            ).norm(dim=-1),
            "attitude_rad": wrap_symmetric(
                estimate[..., ATTITUDE] - truth[..., ATTITUDE], 2.0 * math.pi
            ).abs(),
            "angular_velocity": (
                estimate[..., ANGULAR_VELOCITY] - truth[..., ANGULAR_VELOCITY]
            ).abs(),
            "shield_delay_s": (estimate[..., SHIELD_DELAY] - truth[..., SHIELD_DELAY]).abs(),
            "health": (estimate[..., HEALTH] - truth[..., HEALTH]).abs(),
            "power": (estimate[..., POWER] - truth[..., POWER]).abs(),
            "cooldown_s": (estimate[..., COOLDOWN] - truth[..., COOLDOWN]).abs(),
            "local_log_index": (estimate[..., LOCAL_LOG_INDEX] - truth[..., LOCAL_LOG_INDEX]).abs(),
        }

    def _gradient_diagnostic_groups(
        self, accumulator: TermGradientAccumulator
    ) -> dict[str, list[str]]:
        """Group the accumulated terms into the families that get compared.

        A cosine is only meaningful between terms measuring the same kind of
        thing, and the summed-gradient statistics describe exactly one group, so
        the top-level terms, the reward-decomposed policy terms, and the
        reward-decomposed critic terms are kept apart.

        Args:
            accumulator: The minibatch's accumulated term gradients.

        Returns:
            Group name → accumulator term names, skipping empty groups.
        """
        groups = {
            "top_level": [name for name in accumulator.term_names if "/" not in name],
            "reward_policy": [
                name for name in accumulator.term_names if name.startswith("policy/")
            ],
            "reward_value": [name for name in accumulator.term_names if name.startswith("value/")],
        }
        return {group: names for group, names in groups.items() if names}

    def _gradient_diagnostic_metrics(
        self, accumulator: TermGradientAccumulator, seconds: float
    ) -> dict[str, float]:
        """Turn one diagnosed minibatch's accumulated gradients into metrics.

        Every group is measured over two parameter scopes. The whole-model scope
        answers "how much of the clipped step is this term asking for"; the
        shared-trunk scope answers "do these two terms want the trunk to move the
        same way", which the whole-model cosine cannot, because task-specific
        heads have disjoint parameters and drag every pairing toward zero.

        Args:
            accumulator: The minibatch's accumulated term gradients.
            seconds:     Wall-clock cost of measuring this minibatch.

        Returns:
            Metric name → value.
        """
        records: dict[str, float] = {}
        for group, names in self._gradient_diagnostic_groups(accumulator).items():
            # Reward groups are keyed "<term>/<reward>"; publish the reward.
            display = [name.split("/")[-1] for name in names]
            for trunk in (False, True):
                statistics = scope_statistics(display, accumulator.gram(names, trunk=trunk))
                prefix = f"trunk_{group}" if trunk else group
                records.update(scope_metric_records(prefix, statistics))

        records.update(self._actor_critic_split(accumulator))
        records.update(next_state_gradient_records(accumulator))
        records["grad_diag/microbatches"] = float(accumulator.microbatches)
        records["grad_diag/terms"] = float(len(accumulator.term_names))
        records["grad_diag/seconds"] = seconds
        records["grad_diag/level"] = float(GRADIENT_DIAGNOSTICS_LEVELS.index(self._grad_diag.level))
        return records

    def _update_evaluate_actions(self) -> Callable[..., tuple]:
        """The policy entry point the PPO update evaluates with.

        Compiled, except while gradient diagnostics are on. Those differentiate
        the micro-batch once per decomposed term -- up to seventeen traversals at
        ``reward_full`` -- and a compiled backward is one fused function whose
        saved tensors do not survive the second. Giving each term its own forward
        is not affordable at that count, so a diagnosed run evaluates eagerly
        throughout.

        Read from the diagnostic level rather than fixed at construction, so it
        is a property of the *run* and never of the micro-batch. That is what
        keeps the applied gradient independent of whether it was measured, which
        ``test_measuring_does_not_disturb_the_gradient_that_gets_applied``
        asserts bit-for-bit.
        """

        if self._grad_diag.enabled:
            return self._eager_evaluate_actions
        return self.policy.evaluate_actions

    def _measure_actor_critic_split(
        self,
        batch: MicroBatch,
        denoms: dict,
        frac: float,
    ) -> dict[str, torch.Tensor]:
        """Measure the actor/critic gradient split on a graph of the probe's own.

        The probe differentiates two loss terms separately, which means two
        traversals, and the training backward is a third. A compiled backward is
        one fused function whose saved tensors do not survive the second, so the
        probe re-evaluates the micro-batch eagerly and takes both gradients off
        that. It costs one extra forward and two extra backwards, on one
        micro-batch per histogram interval, against a probe that already budgets
        two extra backwards.

        The point of paying that rather than measuring on the training graph is
        that the training forward and backward stay exactly what an unmeasured
        micro-batch runs, so switching the histogram cadence on cannot move the
        applied gradient. The reported norms come from the eager implementation
        instead of the compiled one, which is a ratio of two gradients either
        way.
        """

        _, diag = self._compute_minibatch_loss(
            batch,
            True,
            denoms,
            frac,
            measure_grad_split=True,
            evaluate_actions=self._eager_evaluate_actions,
        )
        return {key: diag[key] for key in ("grad_norm_actor", "grad_norm_critic")}

    def _actor_critic_split(self, accumulator: TermGradientAccumulator) -> dict[str, float]:
        """The long-standing actor/critic split, read off the top-level terms.

        Same quantity the histogram-cadence probe reports, measured over the
        whole optimizer minibatch rather than one micro-batch of it. The actor
        side is the norm of the *combined* policy and entropy gradient, not the
        sum of their norms, because that combination is what reaches the trunk.

        Args:
            accumulator: The minibatch's accumulated term gradients.

        Returns:
            Metric name → value, empty when neither side received a gradient.
        """
        accumulated = set(accumulator.term_names)
        actor_names = [name for name in ("policy", "entropy") if name in accumulated]
        critic_names = [name for name in ("value",) if name in accumulated]
        if not actor_names and not critic_names:
            return {}
        actor = scope_statistics(actor_names, accumulator.gram(actor_names, trunk=False)).total_norm
        critic = scope_statistics(
            critic_names, accumulator.gram(critic_names, trunk=False)
        ).total_norm
        return {
            "train/grad_norm_actor": actor,
            "train/grad_norm_critic": critic,
            "train/actor_grad_share": actor / (actor + critic + 1e-12),
        }

    def _update_epochs(
        self,
        all_buffers: list[RolloutBuffer | LogicalRolloutBuffer],
        record_histograms: bool = False,
        precomputed: bool = False,
        update: int = 1,
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
            update: This update's index, which decides whether the gradient
                diagnostic cadence fires.

        Returns:
            Dict of mean metric values over all minibatch updates.
        """
        cfg = self.cfg
        K = self.buffer.num_components
        n_scales = len(all_buffers)

        # Precompute everything that depends only on rollout data (not the
        # policy) once per update instead of once per minibatch: the lambda
        # aggregation and the aux next-state labels (primary scale only).
        if not precomputed:
            for scale_idx, buf in enumerate(all_buffers):
                assert isinstance(buf, RolloutBuffer)
                self._precompute_lambda_aggregates(buf, is_primary=(scale_idx == 0))
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
            "loss/global_density": [],
            # Excess nats per cell over a perfect predictor. A Poisson
            # likelihood has no natural zero, so this is the series that says
            # how good the head is; the loss is what is optimized.
            "global_density/deviance": [],
            # The same two numbers per half. The observer sees every ally, so the
            # ally field is a smoothing of what it already knows and the enemy
            # field is mostly belief -- an aggregate over both lets the easy half
            # carry the hard one.
            "loss/global_density_ally": [],
            "loss/global_density_enemy": [],
            "global_density/ally_deviance": [],
            "global_density/enemy_deviance": [],
            "loss_proxy/policy_gradient": [],
            "loss/enemy_action": [],
            "enemy_action/realized_probability": [],
            "enemy_action/entropy": [],
            "enemy_action/brier": [],
            "enemy_action/accuracy": [],
            "enemy_action/persistence_accuracy": [],
            "enemy_action/persistence_cross_entropy": [],
            "enemy_action/uniform_cross_entropy": [],
            "loss_proxy/value": [],
            "loss_proxy/entropy": [],
            "loss_proxy/behavioral_cloning": [],
            "loss_proxy/sigreg": [],
            "loss_proxy/next_state": [],
            "loss_proxy/global_density": [],
            "loss_proxy/enemy_action": [],
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
            "outcome_head/cross_entropy": [],
            "outcome_head/correct_fraction": [],
            "outcome_head/labelled_fraction": [],
            "train/gradient_norm": [],
            # Fraction of optimizer steps on which max_grad_norm actually bound.
            # A guard should fire rarely; a value near one means the clip is not
            # a guard but a reparameterization to normalized-gradient descent,
            # and the loss coefficients want rescaling rather than the clip
            # raising.
            "train/clip_fire_rate": [],
            # Fraction of optimizer steps whose gradients were non-finite and
            # got scrubbed. Any sustained non-zero reading means the forward or
            # backward pass is overflowing and needs investigating at source.
            "train/nonfinite_grad_fraction": [],
            # Actor / critic split of the pre-clip gradient, and the actor's share
            # of it. Measured on one micro-batch per update; the share is what
            # max_grad_norm hands to the policy after renormalizing both together.
            "train/grad_norm_actor": [],
            "train/grad_norm_critic": [],
            "train/actor_grad_share": [],
        }
        accum_k: dict[str, list[torch.Tensor]] = {
            "critic/value_loss": [],
            "critic/explained_variance": [],
            "critic/return_mean": [],
            "critic/value_pred_mean": [],
            "returns/component": [],
            "returns/advantage_std": [],
        }
        # Derived, never hand-listed: a parallel name list drifts from the
        # prediction width silently. It already had, dropping local_log_index --
        # the one channel that says whether fields are being modelled -- off the
        # end of a 9-name list against 10 dimensions.
        ns_feat_names = list(PHYSICAL_MEAN_NAMES)
        ns_per_feat_accum: list[torch.Tensor] = []
        ns_visible_accum: list[torch.Tensor] = []
        ns_hidden_accum: list[torch.Tensor] = []
        label_sq_accum: list[torch.Tensor] = []
        # Raw sums over every pass of the update, finalized once at the end: a
        # small population's mean is a ratio of sums, not a mean of ratios.
        ns_population_sums: torch.Tensor | None = None
        ns_population_counts: torch.Tensor | None = None
        hist_returns: torch.Tensor | None = None
        hist_logprob: torch.Tensor | None = None
        hist_alive: torch.Tensor | None = None

        num_epochs = self._schedule_state.num_epochs
        target_kl = self._effective_target_kl()

        # Explained variance describes the critic at the *end* of the update, so
        # unlike its sibling metrics it is not averaged over epochs -- it is taken
        # from the last epoch that actually ran. Holding it in its own list that
        # resets per epoch is what makes "last epoch that ran" different from
        # "epoch num_epochs-1": target_kl can break the loop early, and gating on
        # the final index instead dropped the whole family for those updates.
        ev_epoch: list[torch.Tensor] = []
        # Gradient diagnostics measure whole optimizer minibatches from the
        # first epoch, so every measurement describes a step taken against the
        # same rollout under a comparably fresh policy.
        diagnose_update = self._grad_diag.measures_update(update)
        diagnosed_minibatches = 0
        grad_diag_records: list[dict[str, float]] = []
        # Armed once per call; the first primary micro-batch consumes it. A
        # diagnostic update measures the actor/critic split over the full
        # minibatch instead, so the cheap single-micro-batch probe stands down
        # rather than measuring the same thing twice, less well.
        measure_split = record_histograms and not diagnose_update

        for epoch_idx in range(num_epochs):
            kl_start = len(accum_scalar["policy/kl"])
            ev_epoch = []
            iters = [
                buf.get_minibatch_iterator(cfg.num_minibatches, cfg.microbatch_tokens)
                for buf in all_buffers
            ]
            for batches in zip(*iters):
                self.optim.zero_grad(set_to_none=self._zero_grad_to_none)

                measure_gradients = (
                    diagnose_update
                    and epoch_idx == 0
                    and diagnosed_minibatches < self._grad_diag.minibatches
                )
                accumulator = (
                    TermGradientAccumulator(self._grad_diag_params, self._grad_diag_trunk)
                    if measure_gradients
                    else None
                )
                diagnostic_start = time.perf_counter() if measure_gradients else 0.0

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
                    ("density", "density_loss"),
                    ("density_deviance", "density_deviance"),
                    ("density_ally", "density_ally_loss"),
                    ("density_enemy", "density_enemy_loss"),
                    ("density_ally_deviance", "density_ally_deviance"),
                    ("density_enemy_deviance", "density_enemy_deviance"),
                    ("bc_kl", "bc_kl"),
                    ("scripted_entropy", "scripted_entropy"),
                    ("kl", "approx_kl"),
                    ("clip", "clip_frac"),
                    ("alive_frac", "alive_frac"),
                    ("ratio_mean", "ratio_mean"),
                    ("entropy_power", "entropy_power"),
                    ("entropy_turn", "entropy_turn"),
                    ("entropy_shoot", "entropy_shoot"),
                    ("enemy_action", "enemy_action_loss"),
                    ("enemy_action_probability", "enemy_action_probability"),
                    ("enemy_action_entropy", "enemy_action_entropy"),
                    ("enemy_action_brier", "enemy_action_brier"),
                    ("enemy_action_accuracy", "enemy_action_accuracy"),
                    ("enemy_action_persistence_accuracy", "enemy_action_persistence_accuracy"),
                    ("enemy_action_persistence_ce", "enemy_action_persistence_ce"),
                    ("outcome_ce", "outcome_ce"),
                    ("outcome_correct", "outcome_correct"),
                    ("outcome_labelled", "outcome_labelled"),
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
                    ("loss/enemy_action", "enemy_action"),
                    ("enemy_action/realized_probability", "enemy_action_probability"),
                    ("enemy_action/entropy", "enemy_action_entropy"),
                    ("enemy_action/brier", "enemy_action_brier"),
                    ("enemy_action/accuracy", "enemy_action_accuracy"),
                    ("enemy_action/persistence_accuracy", "enemy_action_persistence_accuracy"),
                    ("enemy_action/persistence_cross_entropy", "enemy_action_persistence_ce"),
                    ("loss/behavioral_cloning_kl", "bc_kl"),
                    ("loss/scripted_entropy", "scripted_entropy"),
                    ("loss/sigreg", "sigreg"),
                    ("loss/next_state", "ns_loss"),
                    ("loss/next_state_cont", "ns_cont"),
                    ("loss/global_density", "density"),
                    ("global_density/deviance", "density_deviance"),
                    ("loss/global_density_ally", "density_ally"),
                    ("loss/global_density_enemy", "density_enemy"),
                    ("global_density/ally_deviance", "density_ally_deviance"),
                    ("global_density/enemy_deviance", "density_enemy_deviance"),
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
                    ("outcome_head/cross_entropy", "outcome_ce"),
                    ("outcome_head/correct_fraction", "outcome_correct"),
                    ("outcome_head/labelled_fraction", "outcome_labelled"),
                )
                scalar_accum_step: dict[str, torch.Tensor] = {
                    key: _z.clone() for key, _ in _additive
                }
                for key in ("adv_var", "ret_agg_mean", "ret_agg_std", "ratio_max"):
                    scalar_accum_step[key] = _z.clone()

                k_stats: dict[str, torch.Tensor] = {}  # primary per-K moments
                ns_feat_step: torch.Tensor | None = None
                ns_vis_step: torch.Tensor | None = None
                ns_hid_step: torch.Tensor | None = None
                label_sq_step: torch.Tensor | None = None
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
                        # Measured before the training pass and on its own graph,
                        # so the forward and backward that actually move the
                        # policy are identical to an unmeasured micro-batch's.
                        if measure_split and is_primary:
                            split = self._measure_actor_critic_split(device_chunk, denoms, frac)
                            a, c = split["grad_norm_actor"], split["grad_norm_critic"]
                            accum_scalar["train/grad_norm_actor"].append(a)
                            accum_scalar["train/grad_norm_critic"].append(c)
                            accum_scalar["train/actor_grad_share"].append(a / (a + c + 1e-12))
                            measure_split = False

                        loss, diag = self._compute_minibatch_loss(
                            device_chunk,
                            is_primary,
                            denoms,
                            frac,
                            grad_terms=accumulator,
                            grad_scale=1.0 / n_scales,
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
                            for _key, _name in (
                                ("next_state_visible_per_feat", "vis"),
                                ("next_state_hidden_per_feat", "hid"),
                            ):
                                if diag.get(_key) is None:
                                    continue
                                if _name == "vis":
                                    ns_vis_step = (
                                        diag[_key]
                                        if ns_vis_step is None
                                        else ns_vis_step + diag[_key]
                                    )
                                else:
                                    ns_hid_step = (
                                        diag[_key]
                                        if ns_hid_step is None
                                        else ns_hid_step + diag[_key]
                                    )
                            if diag.get("ns_population_sums") is not None:
                                if ns_population_sums is None:
                                    ns_population_sums = diag["ns_population_sums"]
                                    ns_population_counts = diag["ns_population_counts"]
                                else:
                                    ns_population_sums = (
                                        ns_population_sums + diag["ns_population_sums"]
                                    )
                                    ns_population_counts = (
                                        ns_population_counts + diag["ns_population_counts"]
                                    )
                            if diag.get("label_sq_per_feat") is not None:
                                label_sq_step = (
                                    diag["label_sq_per_feat"]
                                    if label_sq_step is None
                                    else label_sq_step + diag["label_sq_per_feat"]
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
                # Whether the clip actually bound. The point of the rescale is
                # that it should be a spike-catcher, and nothing said how often
                # it fired -- on run 748 it was every step, by a factor of 100.
                clipped = (grad_norm > cfg.max_grad_norm).float()
                nonfinite_grad = ~torch.isfinite(grad_norm)
                for param in params:
                    if param.grad is not None:
                        torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                self.optim.step()

                if accumulator is not None:
                    # Taken after the step is launched: the statistics sync on
                    # the host, and there is no reason to make the optimizer
                    # wait behind that.
                    grad_diag_records.append(
                        self._gradient_diagnostic_metrics(
                            accumulator, time.perf_counter() - diagnostic_start
                        )
                    )
                    diagnosed_minibatches += 1
                    accumulator = None  # release the accumulated gradient copies

                for out_key, short_key in _direct_metrics:
                    accum_scalar[out_key].append(scalar_accum_step[short_key])
                accum_scalar["loss_proxy/policy_gradient"].append(
                    self._policy_gradient_coef * scalar_accum_step["pg"]
                )
                accum_scalar["loss_proxy/value"].append(
                    self._schedule_state.value_function_coef * scalar_accum_step["vf"]
                )
                accum_scalar["loss_proxy/entropy"].append(
                    self._entropy_coef * scalar_accum_step["ent"]
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
                accum_scalar["loss_proxy/global_density"].append(
                    self.cfg.global_density_coef * scalar_accum_step["density"]
                )
                accum_scalar["loss_proxy/enemy_action"].append(
                    self.cfg.enemy_action_coef * scalar_accum_step["enemy_action"]
                )
                accum_scalar["returns/advantage_std"].append(scalar_accum_step["adv_var"] ** 0.5)
                accum_scalar["train/gradient_norm"].append(grad_norm.detach())
                accum_scalar["train/clip_fire_rate"].append(clipped)
                accum_scalar["train/nonfinite_grad_fraction"].append(nonfinite_grad.float())

                if k_stats:
                    # Finalized from the accumulated moments but kept on device.
                    # A .cpu() here would be a blocking copy once per optimizer
                    # step, draining the launch queue and destroying CPU
                    # run-ahead for a model that issues hundreds of small kernels
                    # per micro-batch. Everything transfers once, after the loop.
                    ret_var_k = k_stats["ret_sq_k"] - k_stats["ret_mean_k"].pow(2)
                    res_var_k = k_stats["res_sq_k"] - k_stats["res_mean_k"].pow(2)
                    ev_k = 1.0 - res_var_k / (ret_var_k + 1e-8)  # (K,)
                    accum_k["critic/value_loss"].append(k_stats["value_loss_k"])
                    accum_k["critic/return_mean"].append(k_stats["ret_mean_k"])
                    accum_k["returns/component"].append(self._ret_per_comp_mean_k)
                    accum_k["critic/value_pred_mean"].append(k_stats["pred_mean_k"])
                    accum_k["returns/advantage_std"].append(
                        k_stats["adv_sq_k"].clamp(min=0.0).sqrt()
                    )
                    ev_epoch.append(ev_k)

                if ns_feat_step is not None:
                    ns_per_feat_accum.append(ns_feat_step)
                if ns_vis_step is not None:
                    ns_visible_accum.append(ns_vis_step)
                if ns_hid_step is not None:
                    ns_hidden_accum.append(ns_hid_step)
                if label_sq_step is not None:
                    label_sq_accum.append(label_sq_step)

                if record_histograms and "alive_flat" in hist_diag:
                    # Sampled from the last micro-batch of the last primary
                    # minibatch — a large-enough sample for the histograms. Held
                    # as device tensors and converted once below, so a histogram
                    # update does not sync once per minibatch.
                    hist_returns = hist_diag["mb_returns"]
                    hist_logprob = hist_diag["logprob_flat"]
                    hist_alive = hist_diag["alive_flat"]

            if target_kl is not None:
                epoch_kls = accum_scalar["policy/kl"][kl_start:]
                if epoch_kls and torch.stack(epoch_kls).mean().item() > target_kl:
                    break

        accum_k["critic/explained_variance"] = ev_epoch

        metrics: dict = {k: torch.stack(v).mean().item() for k, v in accum_scalar.items() if v}
        # Accuracy is a ratio of two accumulated shares, so it is formed after
        # the sum rather than inside it. Undefined when no match ended anywhere
        # in the batch, which is ordinary: a 128-step rollout against ~8,000-step
        # episodes labels well under a percent of steps.
        labelled = metrics.get("outcome_head/labelled_fraction", 0.0)
        correct = metrics.pop("outcome_head/correct_fraction", 0.0)
        if labelled > 0.0:
            metrics["outcome_head/accuracy"] = correct / labelled
        metrics["train/epochs_completed"] = float(epoch_idx + 1)
        metrics["enemy_action/uniform_cross_entropy"] = math.log(NUM_JOINT_ACTIONS)
        metrics["enemy_action/uniform_probability"] = 1.0 / NUM_JOINT_ACTIONS
        metrics["enemy_action/uniform_accuracy"] = 1.0 / NUM_JOINT_ACTIONS
        metrics["enemy_action/uniform_entropy"] = math.log(NUM_JOINT_ACTIONS)

        for key, tensors in accum_k.items():
            if not tensors:
                continue
            avg = torch.stack(tensors).mean(0).cpu()  # (K,)
            prefix = "returns" if key == "returns/component" else key
            for i, name in enumerate(self._active_names):
                metrics[f"{prefix}/{name}"] = avg[i].item()

        if ns_per_feat_accum:
            avg_per_feat = torch.stack(ns_per_feat_accum).mean(0).cpu()  # (pred_dim,)
            for i, name in enumerate(ns_feat_names):
                metrics[f"next_state/{name}"] = avg_per_feat[i].item()

        # The same error split by sight. A visible token's label is one step of
        # real dynamics and measures the learned model; a hidden one's is mostly
        # belief error and measures how far the recursion has drifted. The
        # aggregate above is a mixture of the two whose proportions move with the
        # fog, so a change in it cannot be attributed without these.
        for accum, prefix in (
            (ns_visible_accum, "next_state_visible"),
            (ns_hidden_accum, "next_state_hidden"),
        ):
            if not accum:
                continue
            avg = torch.stack(accum).mean(0).cpu()  # (pred_dim,)
            for i, name in enumerate(ns_feat_names):
                metrics[f"{prefix}/{name}"] = avg[i].item()

        if label_sq_accum:
            # A calibrated label reads 1.0 here. The suggested scale is the
            # correction that would restore that, so it can be read off the
            # chart and written straight into the feature's label_scale.
            #
            # A feature whose label never moved in this update has no scale to
            # suggest -- health and shield delay do that in any sample without
            # combat. It gets no suggestion rather than one divided by roughly
            # zero, because this series exists to be copied into a config and a
            # plausible-looking wrong number is worse there than a gap.
            avg_label_sq = torch.stack(label_sq_accum).mean(0).cpu()  # (11,)
            for i, name in enumerate(ns_feat_names):
                mean_sq = avg_label_sq[i].item()
                metrics[f"next_state_label_sq/{name}"] = mean_sq
                if mean_sq > 0.0 and math.isfinite(mean_sq):
                    metrics[f"next_state_label_scale/{name}"] = self.next_state.scales[
                        i
                    ] * math.sqrt(mean_sq)

        if ns_population_sums is not None:
            metrics.update(population_metric_records(ns_population_sums, ns_population_counts))

        for name, (total, count) in all_buffers[0].belief_diagnostics.items():
            metrics[name] = (total / count.clamp(min=1.0)).item()

        if grad_diag_records:
            # Averaged across the diagnosed minibatches. Every key is present in
            # every record, so a mean is over the same measurement each time.
            for key in grad_diag_records[0]:
                metrics[key] = sum(record[key] for record in grad_diag_records) / len(
                    grad_diag_records
                )

        if hist_returns is not None:
            # returns are bf16-stored; upcast before numpy (no bf16 dtype there).
            metrics["hist/returns"] = hist_returns.reshape(-1, K)[hist_alive].float().cpu().numpy()
            metrics["hist/logprob"] = hist_logprob[hist_alive].cpu().numpy()

        return metrics

    # ------------------------------------------------------------------
    # Elo evaluation
    # ------------------------------------------------------------------

    def _random_elo(self) -> float:
        """Return the Elo of the random reference on this run's gauge."""
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
            if entry.is_stationary:
                # Stationary references act from the scripted/uniform blend the
                # evaluator computes itself — no weights, no recurrent state.
                # p_scripted=1.0 is the scripted controller; None is uniform.
                p_scripted = 1.0 if entry.kind == "scripted" else entry.p_scripted
                if entry.kind == "semi_random" and p_scripted is None:
                    # Fail rather than fall through to the uniform agent. A rung
                    # that plays as random while keeping a rating of 200-950 is
                    # invisible in every metric and inflates the live rating for
                    # the rest of the run; a resume that cannot rebuild the
                    # ladder has to stop instead.
                    raise ValueError(
                        f"semi-random rung {entry.label!r} has no p_scripted; its roster "
                        "entry cannot be rebuilt and it would silently play as random"
                    )
                return LadderOpponent(
                    policy=None, elo=entry.elo, label=entry.label, p_scripted=p_scripted
                )
            self.roster.load_policy(
                entry,
                self.ship_config,
                self.wrapper.num_ships,
                self.device,
                model_config=self.model_config,
                compile_mode=self._compile_mode,
                global_value_k=self._global_value_k,
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
        """Resolve the win-rate-gated target KL from the current schedule snapshot.

        Reads the same scripted win rate that decays the behavior-cloning
        weight, so "is the policy strong yet" is one measure rather than two.
        Lags by one update — the update phase runs before the schedule refresh —
        exactly as the rating-based gate it replaces did.
        """
        threshold = self._schedule_state.high_winrate_threshold
        if threshold is not None and self._scripted_win_rate >= threshold:
            return self._schedule_state.high_winrate_target_kl
        return self._schedule_state.target_kl
