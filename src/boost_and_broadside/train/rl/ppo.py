"""Recurrent PPO trainer for the Yemong policy.

Core loop: collect rollout → compute per-component GAE → PPO update epochs →
log async → repeat. On top of that, PPOTrainer coordinates:

  - the decomposed critic (per-component returns, lambda aggregation,
    schedule-driven group scales),
  - auxiliary losses (behavior cloning from the scripted agent with
    win-rate-gated decay, an iterated predictive belief state supervised on
    future transitions and future actions, optional SIGReg),
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
from collections.abc import Callable, Generator, Mapping
from datetime import UTC, datetime
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
from boost_and_broadside.config.diagnostics import (
    GRADIENT_DIAGNOSTICS_LEVELS,
    GRADIENT_DIAGNOSTICS_OFF,
    GradientDiagnosticsConfig,
)
from boost_and_broadside.config.live_elo import LIVE_RANDOM_ELO, live_reference_ladder
from boost_and_broadside.constants import (
    ACTION_FACTOR_MAX_ENTROPY,
    ACTION_FACTOR_SLICES,
    POWER_SLICE,
    SHOOT_SLICE,
    TURN_SLICE,
)
from boost_and_broadside.env.field_cache import FieldMapCache
from boost_and_broadside.env.observation import ObsKey, YemongObservation
from boost_and_broadside.env.rewards import component_weights
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.run_manifest import RunStatus
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
from boost_and_broadside.train.rl.grad_diagnostics import (
    TermGradientAccumulator,
    scope_metric_records,
    scope_statistics,
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

# Maps reward component name → the TrainingSchedule tier-scale field to apply.
# Effective weight = tier_scale * individual_weight (from RewardConfig).
#
# The tiers are a credit-assignment ladder, and the per-component gammas and
# lambdas in config/defaults.py already follow the same partition: an outcome is
# discounted over a whole episode, a kill over an engagement, damage over an
# exchange, geometry over the next moment. Scaling a whole tier at once is how a
# run shifts weight between "what actually wins" and the proxies for it.
_TIER: dict[str, str] = {
    "ally_win": "outcome_scale",
    "enemy_win": "outcome_scale",
    "ally_combat_death": "kill_death_scale",
    "enemy_combat_death": "kill_death_scale",
    "ally_field_death": "kill_death_scale",
    "enemy_field_death": "kill_death_scale",
    "combat_death": "kill_death_scale",
    "field_death": "kill_death_scale",
    "kill_shot": "kill_death_scale",
    "kill_assist": "kill_death_scale",
    "kill_ally_shot": "kill_death_scale",
    "kill_ally_assist": "kill_death_scale",
    "ally_combat_damage": "damage_scale",
    "enemy_combat_damage": "damage_scale",
    "ally_field_damage": "damage_scale",
    "enemy_field_damage": "damage_scale",
    "combat_damage_taken": "damage_scale",
    "field_damage_taken": "damage_scale",
    "damage_dealt_enemy": "damage_scale",
    "damage_dealt_ally": "damage_scale",
    "facing": "shaping_scale",
    "closing_speed": "shaping_scale",
    "shoot_quality": "shaping_scale",
    "shooting_penalty": "shaping_scale",
    "speed": "shaping_scale",
}

# Components with self-only rewards use a diagonal lambda (i == j); all others
# use team-based lambda aggregation.
#
# Stated outright rather than derived from the tier map. Locality is a
# credit-assignment property and the tier is a weighting one, and they are
# orthogonal: combat_death and ally_combat_death sit in the same tier and differ
# only in whether the signal propagates to teammates. The previous registry
# derived one from the other, which worked only because the scale groups
# happened to be drawn along the locality line. ``test_every_component_is
# _classified`` pins that both maps stay complete.
_LOCAL_COMPONENTS: frozenset[str] = frozenset(
    {
        "facing",
        "closing_speed",
        "shoot_quality",
        "kill_shot",
        "kill_assist",
        "kill_ally_shot",
        "kill_ally_assist",
        "combat_damage_taken",
        "field_damage_taken",
        "damage_dealt_enemy",
        "damage_dealt_ally",
        "combat_death",
        "field_death",
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
    outcome_scale: float
    kill_death_scale: float
    damage_scale: float
    shaping_scale: float
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
    # Recurrent tokens per env (ships). Fields are non-recurrent, so this is
    # deliberately not N+M — it is the stride for every hidden-state operation.
    num_recurrent: int
    elo_eval: EloEvaluator
    obs: YemongObservation
    hidden: torch.Tensor
    hidden_t1: torch.Tensor | None
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


def predictive_horizon_masks(
    alive: torch.Tensor,
    terminated: torch.Tensor,
    actor_mask: torch.Tensor,
    prediction_horizon: int,
) -> Generator[tuple[int, torch.Tensor, torch.Tensor]]:
    """Yield the validity masks of each predictive horizon, aligned on the base step.

    The belief state at base step ``t`` and horizon ``h`` describes step
    ``t + h``, so a prediction only counts where the rollout actually reaches
    that step: no episode boundary anywhere in ``[t, t + h)``, and a ship that
    is alive at both ends. Horizon ``h`` therefore covers base steps
    ``[0, T - h)`` — the tail of the rollout simply supervises fewer horizons
    rather than borrowing steps from the episode that follows it.

    The two masks differ in one place. A *state* prediction is a transition out
    of step ``t + h``, so it additionally needs that step to be non-terminal. An
    *action* prediction is about the decision taken at ``t + h``, which exists
    whether or not the episode ends there.

    Args:
        alive:      (T, B, N) bool — living ships per step.
        terminated: (T, B) bool — True at the step an episode ended.
        actor_mask: (T, B, N) bool — ships whose stored action was sampled from
            this very forward pass. Only used to exclude horizon 0; see
            ``_predictive_losses``.
        prediction_horizon: Horizons to supervise, counting horizon 0.

    Yields:
        ``(horizon, state_mask, action_mask)`` with both masks shaped
        ``(T - horizon, B, N)`` and indexed by the base step ``t``.
    """
    steps = alive.shape[0]
    # chain[t] — the trajectory from t has not yet crossed an episode boundary.
    chain = torch.ones_like(terminated)  # (T, B)
    for horizon in range(min(prediction_horizon, steps)):
        span = steps - horizon
        # alive[t + horizon] and terminated[t + horizon], re-indexed by base step.
        reached_alive = alive[horizon:steps]
        reached_terminal = terminated[horizon:steps]
        entity_mask = chain[:span].unsqueeze(-1) & alive[:span] & reached_alive
        state_mask = entity_mask & ~reached_terminal.unsqueeze(-1)
        action_mask = entity_mask
        if horizon == 0:
            action_mask = action_mask & ~actor_mask[:span]
        yield horizon, state_mask, action_mask
        chain = chain[:span] & ~reached_terminal


# Diagnostic groups the action prediction is split across: the whole valid set,
# then the two sides of it. The suffix is both the denominator key and the metric
# key, so a group cannot be normalized by one team's count and published as the
# other's.
ACTION_PREDICTION_GROUPS: tuple[str, ...] = ("", "_ally", "_enemy")


def stratified_depth_assignment(
    num_steps: int,
    prediction_horizon: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Assign each rollout step one predictive depth, split evenly across depths.

    Independent uniform draws would leave each depth with a random number of
    steps — Binomial(T, 1/H), so 10.7 ± 3.1 at the shipping shape. Dealing a
    fixed multiset instead pins the count per depth to within one, which buys
    two things beyond the obvious variance reduction:

    * the *shapes* of every tensor in the rollout become identical from step to
      step, so ``torch.compile`` sees one graph rather than a new one per draw;
    * the total transition work is exactly ``T * (H - 1) / 2`` rather than a
      random variable around it, so peak memory is what the average says.

    ``T`` need not divide ``H``. The remainder steps go to a *randomly chosen*
    subset of depths, so across a run's minibatches every depth draws the extra
    step equally often, and the multiset of counts — and therefore every shape —
    stays fixed.

    The assignment is a permutation, never ``t % H``: the modulo pattern would
    lock each depth to one phase of any periodic structure in the game, and the
    firing cooldown is three decisions against a twelve-deep horizon.

    Args:
        num_steps: T — rollout steps to assign.
        prediction_horizon: H — depths to spread across, capped at T.
        device: Device for the returned tensor. The draw itself always runs on
            the CPU generator, the one ``--seed`` covers and the one that
            already orders minibatches.

    Returns:
        (T,) int64 depths in ``[0, min(H, T))``, counts differing by at most one.
    """
    depths = min(prediction_horizon, num_steps)
    base, remainder = divmod(num_steps, depths)
    counts = torch.full((depths,), base, dtype=torch.long)
    if remainder:
        counts[torch.randperm(depths)[:remainder]] += 1
    assignment = torch.repeat_interleave(torch.arange(depths), counts)
    assignment = assignment[torch.randperm(num_steps)]
    return assignment.to(device) if device is not None else assignment


def predictive_masks_at(
    base: torch.Tensor,
    depth: int,
    alive: torch.Tensor,
    terminated: torch.Tensor,
    actor_mask: torch.Tensor,
    boundary_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validity of a horizon-``depth`` prediction made from the given base steps.

    The index-addressed counterpart of :func:`predictive_horizon_masks`, for the
    sampled mode where each step sits at its own depth and the contiguous
    per-horizon slabs no longer exist. Same rules, stated the same way: the
    trajectory must not cross an episode boundary in ``[t, t + depth)``, the ship
    must be alive at both ends, a *state* prediction additionally needs the
    reached step to be non-terminal, and horizon 0 excludes the actions the
    latent itself generated.

    Episode boundaries are read off a prefix sum rather than walked step by step,
    which is what lets an arbitrary set of base steps be checked at once.

    Args:
        base:  (n,) int64 rollout steps the predictions are made from.
        depth: Horizon these predictions sit at.
        alive: (T, B, N) bool.
        terminated: (T, B) bool.
        actor_mask: (T, B, N) bool — actions sampled from this very pass.
        boundary_counts: (T + 1, B) int64 prefix sum of ``terminated``.

    Returns:
        ``(state_mask, action_mask, reached)`` — masks shaped (n, B, N) and the
        clamped ``base + depth`` index the targets are gathered from. Rows whose
        target falls past the rollout are already masked out.
    """
    steps = alive.shape[0]
    reached = base + depth
    in_range = reached < steps
    reached = reached.clamp(max=steps - 1)

    crossed = boundary_counts[reached] - boundary_counts[base]  # (n, B)
    entity = (
        (crossed == 0).unsqueeze(-1)
        & alive[base]
        & alive[reached]
        & in_range[:, None, None]
    )
    state_mask = entity & ~terminated[reached].unsqueeze(-1)
    action_mask = entity if depth else entity & ~actor_mask[base]
    return state_mask, action_mask, reached


def episode_boundary_counts(terminated: torch.Tensor) -> torch.Tensor:
    """(T + 1, B) prefix sum of episode endings, for O(1) boundary queries."""
    counts = terminated.new_zeros((terminated.shape[0] + 1, terminated.shape[1]), dtype=torch.long)
    counts[1:] = terminated.long().cumsum(0)
    return counts


def ally_token_mask(
    obs: YemongObservation,
    num_ships: int,
    num_steps: int,
) -> torch.Tensor:
    """(T, B, N) bool — ship tokens on team 0 of the perspective this batch stores.

    "Ally" is a statement about a perspective, not about a pair of ships. The
    rollout stores the raw observation, and in ``ego_pass`` that observation is
    always written from team 0's point of view — every ship acts from a pass
    where its own side is team 0, and the ego half is the half re-evaluated
    here. So team 0 is the side whose decisions the belief is forecasting from
    the inside, and team 1 is the opposition.

    In ``shared_pass`` there is no ego side: both teams train from one pass and a
    league opponent may play either. The split is still exactly "team 0 versus
    team 1" and still separates the two sides of each battle, but which side is
    "ours" is then a labelling convention rather than a perspective.

    Args:
        obs:       The micro-batch observation, with T+1 stored steps.
        num_ships: N — field tokens carry team 2 and are excluded by the slice.
        num_steps: T — the steps the predictions are indexed by.

    Returns:
        (T, B, N) bool, True on team 0.
    """
    return obs[ObsKey.TEAM_ID][:num_steps, :, :num_ships].long() == 0


def factored_action_statistics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token cross-entropy, entropy, and per-factor hits of an action prediction.

    The three action factors are independent categoricals sharing one logit
    vector, exactly as the actor's own head lays them out. Each factor's term is
    divided by its own maximum entropy before they are summed, so a factor
    contributes according to how much of its uncertainty is left rather than to
    how many options it offers: unnormalized, the seven-way turn would be 52% of
    an untrained total and the binary shoot 19%, and the auxiliary would press
    the trunk hardest on whichever factor happened to have the most values. Both
    returned quantities are then in units of "fraction of maximum uncertainty
    per factor", summing to the factor count when the prediction is uniform and
    to zero when it is exact.

    This is deliberately *not* what the behavior-cloning loss or the entropy
    bonus do. Those are the log-likelihood and the entropy of the distribution
    the policy genuinely samples from, and rescaling their factors would stop
    them being either. This one is a representation-shaping pressure, where
    balance across the factors is worth more than being a likelihood — the same
    reason the state side scales every channel by ``label_scale``.

    Args:
        logits:  (..., TOTAL_ACTION_LOGITS) — predicted [power | turn | shoot].
        targets: (..., 3) long — the actions the rollout actually took.

    Returns:
        cross_entropy: (...) — normalized negative log-likelihood of the target.
        entropy:       (...) — normalized entropy of the predicted distribution.
        hits:          (..., 3) bool — whether each factor's mode is the target.
    """
    cross_entropy = torch.zeros_like(logits[..., 0])
    entropy = torch.zeros_like(cross_entropy)
    hits = []
    factors = zip(ACTION_FACTOR_SLICES, ACTION_FACTOR_MAX_ENTROPY, strict=True)
    for factor, (logit_slice, max_entropy) in enumerate(factors):
        log_probs = F.log_softmax(logits[..., logit_slice], dim=-1)
        target = targets[..., factor]
        likelihood = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        cross_entropy = cross_entropy - likelihood / max_entropy
        entropy = entropy - (log_probs.exp() * log_probs).sum(-1) / max_entropy
        hits.append(log_probs.argmax(-1) == target)
    return cross_entropy, entropy, torch.stack(hits, dim=-1)


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
        learning_rate=schedule.learning_rate(step),
        policy_gradient_coef=schedule.policy_gradient_coef(step),
        entropy_coef=schedule.entropy_coef(step),
        behavior_cloning_coef=schedule.behavior_cloning_coef(step),
        value_function_coef=schedule.value_function_coef(step),
        sigreg_coef=schedule.sigreg_coef(step),
        outcome_scale=schedule.outcome_scale(step),
        kill_death_scale=schedule.kill_death_scale(step),
        damage_scale=schedule.damage_scale(step),
        shaping_scale=schedule.shaping_scale(step),
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
        compile_mode: str | None = "reduce-overhead",
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
            num_tokens=N + M,
        )

        # Pre-compute lambda masks for active components only.
        # Static for the entire run — derived from RewardConfig.
        # Derived once: the per-component weights the four event weights imply.
        self._component_weights = component_weights(train_config.rewards)
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
        # Grid points are absolute, so the first one to claim is the highest
        # multiple of the gap at or below where the run starts.
        self._elo_milestone: float = (
            (LIVE_RANDOM_ELO // train_config.elo_milestone_gap)
            * train_config.elo_milestone_gap
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
        torch.Tensor | None,
    ]:
        """Run the training policy's rollout forward pass(es) for one step.

        ego_pass: one batched 2B pass over both team perspectives. Team 1 ships
        act from the flipped-obs half (action_t1); logprob and value are stored
        from the raw-obs half only. No state prediction is decoded: collecting
        experience never reads one.
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
            hidden:     Updated raw-perspective hidden state.
            hidden_t1:  Updated flipped-perspective hidden state; None in shared_pass.
        """
        if not self._ego_pass:
            action, logprob, value_norm, _, hidden = self.policy.get_action_and_value(obs, hidden)
            return action, None, logprob, value_norm, hidden, None

        batch = hidden.shape[1] // num_recurrent
        obs_t1 = flip_team_obs(obs, num_ships)
        obs_both = obs.concat_batch(obs_t1)
        hidden_both = torch.cat([hidden, hidden_t1], dim=1)  # (n_layers, 2B*N, CONV_KERNEL*D)
        action_both, logprob_both, value_both, _, hidden_out = self.policy.get_action_and_value(
            obs_both, hidden_both
        )
        return (
            action_both[:batch],  # (B, N, 3)
            action_both[batch:],  # (B, N, 3)
            logprob_both[:batch],  # (B, N)
            value_both[:batch],  # (B, N, K)
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
            aux_last_dones[i] = aux_done_any
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
        # Only ships carry recurrent state; field tokens take the non-recurrent path.
        num_recurrent = num_ships

        obs = self.wrapper.reset()
        self.wrapper.env.state.step_count.random_(0, self.env_config.max_episode_steps)
        hidden = self.policy.initial_hidden(num_envs, num_recurrent, self.device)
        hidden_t1 = (
            self.policy.initial_hidden(num_envs, num_recurrent, self.device)
            if self._ego_pass
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
            ),
            obs=obs,
            hidden=hidden,
            hidden_t1=hidden_t1,
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

        # Fresh maps every rollout. A fixed bank is a small enough distribution
        # that a full run sees each map thousands of times; regenerating it here
        # gives roughly one distinct map per episode for a few milliseconds,
        # entirely on device and without a host sync.
        if self._field_map is not None:
            self._field_map.refresh()
        slots = self._prepare_league_slots(runtime.num_recurrent)
        for rollout_step in range(self.cfg.num_steps):
            primary = self._collect_primary_step(
                obs=runtime.obs,
                hidden=runtime.hidden,
                hidden_t1=runtime.hidden_t1,
                action_buffer=runtime.action_buffer,
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
                runtime.action_buffer,
                terminated,
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
        self._live_elo = elo_snapshot.live_elo
        self._avg_live_elo = elo_snapshot.avg_elo
        self._floating_games = elo_snapshot.floating_games
        self._match_counts = elo_snapshot.match_counts
        if elo_snapshot.floating_elo is not None:
            self.roster.set_floating_elo(elo_snapshot.floating_elo)
        # Proximity sampling reads this, so it has to track the evaluator rather
        # than keep the rating the entry was created with. The scripted entry
        # needs no such sync: the live gauge pins it and it never moves.
        self.roster.set_special_elo("avg", elo_snapshot.avg_elo)
        return terminated

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
        self.buffer.store_final_obs(runtime.obs)
        for index, aux_buffer in enumerate(self.aux_buffers):
            aux_buffer.store_final_obs(runtime.aux_obs[index])

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, next_value_norm, _, _ = self.policy.get_action_and_value(
                runtime.obs, runtime.hidden
            )
        self.buffer.compute_gae(self.scaler.denormalize(next_value_norm), terminated.float())
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
                    self._precompute_transition_labels(buffer)
                else:
                    buffer.transition_labels = None
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
        bc_factor = max(0.0, 1.0 - self._scripted_win_rate / self.cfg.bc_winrate_target)
        self._behavior_cloning_coef = self._schedule_state.behavior_cloning_coef * bc_factor
        self._entropy_coef = _actor_entropy_coef(
            self._schedule_state.entropy_coef,
            policy_gradient_coef=self._policy_gradient_coef,
            behavior_cloning_coef=self._behavior_cloning_coef,
        )
        self.optim.param_groups[0]["lr"] = self._schedule_state.learning_rate
        for component in self.wrapper.reward_components:
            raw_weight = self._component_weights[component.name]
            component.weight = raw_weight * getattr(self._schedule_state, _TIER[component.name])
        self.wrapper.refresh_component_weights()
        return bc_factor

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
        metrics["schedule/damage_scale"] = self._schedule_state.damage_scale
        metrics["schedule/shaping_scale"] = self._schedule_state.shaping_scale

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
        numel = 0
        need_bc = is_primary and self._behavior_cloning_coef > 0.0
        need_predictive = is_primary and self._predictive_enabled
        steps, _, ships = chunks[0].alive.shape
        horizons = min(self._predictive_horizon, steps)
        # Drawn once per optimizer minibatch and shared by its micro-batches, so
        # every micro-batch scores the same depths and the denominators below
        # describe the whole minibatch. It rides along in the returned dict
        # rather than being redrawn in the loss, where it would disagree.
        assignment = (
            stratified_depth_assignment(steps, self.cfg.prediction_horizon)
            if need_predictive and self.cfg.predictive_mode == "sampled"
            else None
        )
        # Counted on the host once, so the belief rollout can slice its prefixes
        # from Python ints instead of syncing the device every depth.
        depth_counts = (
            None
            if assignment is None
            else tuple(torch.bincount(assignment, minlength=horizons).tolist())
        )
        if assignment is not None:
            assignment = assignment.to(source_device)
        state_counts = torch.zeros(horizons, device=source_device)
        action_counts = {
            group: torch.zeros(horizons, device=source_device) for group in ACTION_PREDICTION_GROUPS
        }
        for chunk in chunks:
            mb_alive = chunk.alive
            mb_actor_mask = chunk.actor_mask
            mb_expert_probs = chunk.expert_probs
            alive_sum += mb_alive.sum()
            actor_sum += (mb_actor_mask & mb_alive).sum()
            numel += mb_alive.numel()
            if need_bc:
                bc_valid = mb_expert_probs.sum(-1) > 0
                bc_sum += (bc_valid & mb_actor_mask & mb_alive).sum()
            if not need_predictive:
                continue
            # One count per horizon: the predictive losses average each horizon
            # over its own valid tokens, so every horizon needs a denominator
            # that spans the whole minibatch rather than this micro-batch.
            ally = ally_token_mask(chunk.obs, ships, steps)
            if assignment is None:
                masks = (
                    (horizon, state, action, slice(None))
                    for horizon, state, action in predictive_horizon_masks(
                        mb_alive, chunk.terminated, mb_actor_mask, self._predictive_horizon
                    )
                )
            else:
                masks = self._sampled_masks(assignment, mb_alive, chunk.terminated, mb_actor_mask)
            for horizon, state_mask, action_mask, rows in masks:
                ally_rows = ally[rows] if assignment is not None else ally[: state_mask.shape[0]]
                state_counts[horizon] += state_mask.sum()
                action_counts[""][horizon] += action_mask.sum()
                action_counts["_ally"][horizon] += (action_mask & ally_rows).sum()
                action_counts["_enemy"][horizon] += (action_mask & ~ally_rows).sum()
        denominators = {
            "depth_assignment": None if assignment is None else assignment.to(self.device),
            "depth_counts": depth_counts,
            "mask_sum": alive_sum.clamp(min=1.0).to(self.device),
            "actor_sum": actor_sum.clamp(min=1.0).to(self.device),
            "bc_sum": bc_sum.clamp(min=1.0).to(self.device),
            "state_counts": state_counts.clamp(min=1.0).to(self.device),
            "numel": float(numel),
            "adv_rms": buf.adv_rms,
        }
        # Depths that scored anything. A depth with no valid tokens contributes a
        # zero numerator, so dividing the total by the full horizon count would
        # pull the mean toward zero by however many depths happened to come up
        # empty — a bias that exists in either mode and that sampling, with far
        # fewer steps per depth, makes easy to hit.
        denominators["state_depths"] = (state_counts > 0).sum().clamp(min=1).float().to(self.device)
        denominators["action_depths"] = (
            (action_counts[""] > 0).sum().clamp(min=1).float().to(self.device)
        )
        for group, counts in action_counts.items():
            # The unsplit group normalizes the loss and must never be zero. The
            # two sides are diagnostics, and an empty side is left to divide by
            # zero on purpose: in ego_pass the ally group is empty at horizon 0
            # by construction — those are exactly the self-generated actions the
            # objective excludes — and a NaN says "not measured" where a zero
            # would read as a perfect prediction.
            counts = counts.clamp(min=1.0) if group == "" else counts
            denominators[f"action_counts{group}"] = counts.to(self.device)
            denominators[f"action_count_total{group}"] = counts.sum().to(self.device)
        return denominators

    def _compute_minibatch_loss(
        self,
        batch: MicroBatch,
        is_primary: bool,
        denoms: dict,
        frac: float,
        measure_grad_split: bool = False,
        grad_terms: TermGradientAccumulator | None = None,
        grad_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """Compute PPO loss for one micro-batch. Does NOT call zero_grad / backward / step.

        Loss coefficients are read from ``self._policy_gradient_coef``,
        ``self._behavior_cloning_coef``, and ``self._schedule_state``. Setting
        ``policy_gradient_coef=0.0`` activates BC pretraining mode.

        Lambda-aggregated advantages/returns and state-transition labels arrive
        precomputed in the batch (see _precompute_lambda_aggregates /
        _precompute_transition_labels) — they depend only on rollout data, so
        they are built once per update instead of once per minibatch.

        Masked-mean terms divide by the minibatch-total denominators in
        ``denoms`` rather than micro-batch-local counts, so losses and additive
        diagnostics from a minibatch's micro-batches sum exactly to the unsplit
        minibatch values — gradient accumulation over micro-batches is then
        equivalent to one large minibatch. Batch-statistic terms (sigreg) can't
        decompose that way and are weighted by ``frac`` instead (exact when the
        minibatch is unsplit, i.e. frac=1).

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

        # mb_obs has T+1 steps; first T for encode/evaluate, last T for next-state aux loss.
        T = mb_alive.shape[0]
        curr_mb_obs = mb_obs.slice_time(0, T)

        need_sigreg = self._schedule_state.sigreg_coef > 0.0
        need_predictive = is_primary and self._predictive_enabled
        # evaluate_actions needs the full (T, B, N+M) alive mask so Yemong layers
        # can attend to field tokens; mb_alive is ships-only and used for loss masking.
        alive_mask_full = curr_mb_obs["alive"].bool()  # (T, B_mb, N+M)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (
                logprob,
                entropy,
                new_value,
                policy_logits,
                z,
                predictive_latent,
            ) = self.policy.evaluate_actions(
                obs=curr_mb_obs,
                actions=mb_actions.long(),
                initial_hidden=mb_hidden,
                alive_mask=alive_mask_full,
                done_mask=mb_terminated,
                return_encoder_output=need_sigreg,
                return_predictive_latent=need_predictive,
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

        # ---- Predictive belief-state losses (primary scale only) -------------
        predictive_state_loss = self._zero_tensor
        predictive_action_loss = self._zero_tensor
        predictive_diag: dict = {}
        if need_predictive:
            predictive_state_loss, predictive_action_loss, predictive_diag = (
                self._predictive_losses(predictive_latent, batch, denoms)
            )

        loss = (
            self._policy_gradient_coef * pg_loss
            + self._schedule_state.value_function_coef * vf_loss
            + self._entropy_coef * ent_loss
            + self._behavior_cloning_coef * bc_loss
            + self._schedule_state.sigreg_coef * sigreg_loss
            + self.cfg.predictive_state_coef * predictive_state_loss
            + self.cfg.predictive_action_coef * predictive_action_loss
        )

        diag: dict = dict(predictive_diag)

        # ---- Gradient decomposition -------------------------------------------
        # Differentiates the weighted terms that make up `loss` above, one
        # autograd traversal each, and hands the results to the accumulator that
        # spans this optimizer minibatch. Everything here is opt-in: with
        # grad_terms None not a single extra graph node is built.
        if grad_terms is not None:
            terms = {
                "policy": self._policy_gradient_coef * pg_loss,
                "value": self._schedule_state.value_function_coef * vf_loss,
                "entropy": self._entropy_coef * ent_loss,
                "bc": self._behavior_cloning_coef * bc_loss,
                "sigreg": self._schedule_state.sigreg_coef * sigreg_loss,
                "predictive_state": self.cfg.predictive_state_coef * predictive_state_loss,
                "predictive_action": self.cfg.predictive_action_coef * predictive_action_loss,
            }
            if self._grad_diag.decomposes_policy_by_reward:
                terms.update(
                    self._reward_policy_terms(
                        batch=batch,
                        ratio=ratio,
                        adv_norm=adv_norm,
                        actor_f=actor_f,
                        actor_sum=actor_sum,
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
            diag["predictive_state_loss"] = predictive_state_loss.detach()
            diag["predictive_action_loss"] = predictive_action_loss.detach()
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

    def _predictive_plan(
        self,
        predictive_latent: torch.Tensor,
        batch: MicroBatch,
        denoms: dict,
        ally: torch.Tensor,
        predictive: torch.nn.Module,
    ):
        """Yield every decode point of the belief rollout, belief already advanced.

        The two modes differ only in *which* rollout steps decode at *which*
        depth; the scoring is identical, so it lives in the caller and this
        supplies the plan.

        ``full`` decodes every step at every depth: the belief is one contiguous
        slab that loses its last step each time, since that step's next horizon
        would need data past the end of the rollout.

        ``sampled`` gives each step a single depth. Steps are visited
        deepest-first, so the ones still advancing are always a prefix and the
        belief shrinks by slicing rather than gathering. The prefix lengths come
        from host-side counts, which is what keeps a twelve-deep rollout free of
        twelve device synchronisations.

        Yields:
            ``(depth, belief, reached, state_mask, action_mask, ally_rows)``,
            where ``reached`` indexes the labels and action targets.
        """
        assignment = denoms.get("depth_assignment")
        horizon = self._predictive_horizon
        if assignment is None:
            belief = predictive_latent
            depths = min(horizon, batch.alive.shape[0])
            for depth, state_mask, action_mask in predictive_horizon_masks(
                batch.alive, batch.terminated, batch.actor_mask, horizon
            ):
                span = state_mask.shape[0]
                reached = torch.arange(span, device=belief.device) + depth
                yield depth, belief, reached, state_mask, action_mask, ally[:span]
                # Only when another horizon will consume it. Advancing past the
                # last one is a transition nothing decodes -- wasted work here,
                # and the one-step arm has no transition to call at all.
                if depth + 1 < depths and span > 1:
                    belief = predictive.advance(belief[: span - 1])
            return

        counts = denoms["depth_counts"]
        boundary_counts = episode_boundary_counts(batch.terminated)
        # Deepest first, so "still advancing" is a prefix at every depth.
        active = torch.argsort(assignment, descending=True, stable=True)
        belief = predictive_latent[active]
        for depth in range(len(counts)):
            remaining = sum(counts[depth + 1 :])
            rows = active[remaining:]
            state_mask, action_mask, reached = predictive_masks_at(
                rows, depth, batch.alive, batch.terminated, batch.actor_mask, boundary_counts
            )
            yield depth, belief[remaining:], reached, state_mask, action_mask, ally[rows]
            active = active[:remaining]
            belief = predictive.advance(belief[:remaining])

    def _sampled_masks(
        self,
        assignment: torch.Tensor,
        alive: torch.Tensor,
        terminated: torch.Tensor,
        actor_mask: torch.Tensor,
    ):
        """Yield ``(depth, state_mask, action_mask, rows)`` for the sampled plan.

        ``rows`` are the rollout steps decoding at that depth, ordered so that
        the belief rollout can hand this method a contiguous slice: steps are
        visited deepest-first, so "still advancing" is always a prefix.
        """
        boundary_counts = episode_boundary_counts(terminated)
        order = torch.argsort(assignment, descending=True, stable=True)
        ordered = assignment[order]
        horizons = min(self._predictive_horizon, alive.shape[0])
        for depth in range(horizons):
            rows = order[ordered == depth]
            state_mask, action_mask, _ = predictive_masks_at(
                rows, depth, alive, terminated, actor_mask, boundary_counts
            )
            yield depth, state_mask, action_mask, rows

    @property
    def _predictive_horizon(self) -> int:
        """Decode depth for this run: one step for the control arm, else configured.

        ``next_step`` is the horizon-0 slice of the same machinery, so it needs
        no separate loss path -- capping the depth at one leaves the ``full``
        plan yielding exactly the immediate transition.
        """
        return 1 if self.cfg.predictive_mode == "next_step" else self.cfg.prediction_horizon

    @property
    def _predictive_action_enabled(self) -> bool:
        """Whether the action family trains. The control arm has no action head."""

        if self.cfg.predictive_mode == "next_step":
            return False
        return self.cfg.predictive_action_coef > 0.0

    @property
    def _predictive_enabled(self) -> bool:
        """Whether either predictive auxiliary family carries weight this run."""

        if self.cfg.predictive_mode == "off":
            return False
        return self.cfg.predictive_state_coef > 0.0 or self._predictive_action_enabled

    def _predictive_losses(
        self,
        predictive_latent: torch.Tensor,
        batch: MicroBatch,
        denoms: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Roll the belief state forward and score it against the realized future.

        The rollout is open-loop: after the projection that produced
        ``predictive_latent``, the only input to each step is the previous
        belief. Rollout states and actions enter as targets and nothing else,
        which is what leaves the later horizons genuinely uncertain.

        Timing, which is the part that is easy to get wrong. The observation at
        step ``t`` already carries the action pending on the transition to
        ``t + 1``, so:

          - the state prediction at horizon ``h`` targets the local transition
            ``t + h -> t + h + 1``, i.e. ``transition_labels[t + h]``;
          - the action prediction at horizon ``h`` targets ``actions[t + h]``,
            the decision *made* at that step, which only becomes visible in the
            observation at ``t + h + 1``.

        Horizon 0's action target is skipped wherever the rollout sampled that
        action from this very latent (``actor_mask``): predicting one's own
        output is self-imitation, not a belief about anything. Actions that came
        from another perspective, another policy, or the scripted controller
        stay in — the latent did not produce those.

        Each horizon is averaged over its own valid tokens and the horizons are
        then averaged with equal weight, so the easy immediate transition cannot
        dominate the total by sheer count.

        The action statistics are additionally reported for each side of the
        battle on its own (see ``ally_token_mask``). Only the unsplit group
        enters the loss; the split exists because anticipating your own fleet's
        decisions and anticipating the opposition's are different problems that
        the total silently averages together. It also makes the horizon-0
        exclusion visible rather than implicit: under ``ego_pass`` the ally group
        there is empty by construction and reads NaN, because those are exactly
        the actions this latent generated.

        Args:
            predictive_latent: (T, b, N, predictive_latent_dim) horizon-0 belief.
            batch:  The micro-batch being scored.
            denoms: Minibatch-total denominators from ``_minibatch_denominators``.

        Returns:
            ``(state_loss, action_loss, diagnostics)``. The diagnostics are GPU
            tensors that add across micro-batches, never host scalars.
        """
        predictive = self._policy_module.predictive
        labels = batch.transition_labels
        targets = batch.actions.long()
        want_state = self.cfg.predictive_state_coef > 0.0 and labels is not None
        want_action = self._predictive_action_enabled
        prediction_dim = self.coordinator.total_prediction_dimension

        state_counts = denoms["state_counts"]
        action_counts = denoms["action_counts"]
        state_total = self._zero_tensor
        action_total = self._zero_tensor
        state_by_horizon: list[torch.Tensor] = []
        state_per_feature: torch.Tensor | None = None
        # One series per diagnostic group, keyed by the same suffix the
        # denominators use. The unsplit group is what the loss is built from;
        # the other two only describe it.
        cross_entropy_by_horizon: dict[str, list[torch.Tensor]] = {
            group: [] for group in ACTION_PREDICTION_GROUPS
        }
        entropy_by_horizon: dict[str, list[torch.Tensor]] = {
            group: [] for group in ACTION_PREDICTION_GROUPS
        }
        accuracy_numerator = {
            group: torch.zeros(3, device=self.device) for group in ACTION_PREDICTION_GROUPS
        }
        steps, _, ships = batch.alive.shape
        # Always built: the rollout plan slices it per decode point, and a team
        # comparison is far cheaper than branching the plan on which family is on.
        ally = ally_token_mask(batch.obs, ships, steps)

        # The rollout runs in the same reduced precision as the trunk that fed
        # it: its activation memory scales with the horizon, and it is the one
        # part of the update whose cost is linear in a configurable depth. The
        # reductions stay fp32 — autocast leaves them alone, and every predicted
        # tensor is upcast before it enters one.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for horizon, belief, reached, state_mask, action_mask, ally_rows in (
                self._predictive_plan(predictive_latent, batch, denoms, ally, predictive)
            ):
                if want_state:
                    prediction = predictive.predict_state(belief).float()
                    squared = (prediction - labels[reached].detach()).pow(2)
                    squared = squared * self.aux_weights * state_mask.unsqueeze(-1)
                    horizon_loss = squared.sum() / (state_counts[horizon] * prediction_dim)
                    state_total = state_total + horizon_loss
                    with torch.no_grad():
                        state_by_horizon.append(horizon_loss.detach())
                        if horizon == 0:
                            # Per-channel error of the immediate transition — the
                            # same measurement the one-step head used to report.
                            state_per_feature = squared.sum((0, 1, 2)) / state_counts[0]
                if want_action:
                    logits = predictive.predict_action_logits(belief).float()
                    cross_entropy, entropy, hits = factored_action_statistics(
                        logits, targets[reached]
                    )
                    mask = action_mask.float()
                    horizon_loss = (cross_entropy * mask).sum() / action_counts[horizon]
                    action_total = action_total + horizon_loss
                    with torch.no_grad():
                        groups = {
                            "": mask,
                            "_ally": (action_mask & ally_rows).float(),
                            "_enemy": (action_mask & ~ally_rows).float(),
                        }
                        for group, weights in groups.items():
                            count = denoms[f"action_counts{group}"][horizon]
                            cross_entropy_by_horizon[group].append(
                                (cross_entropy * weights).sum() / count
                            )
                            entropy_by_horizon[group].append((entropy * weights).sum() / count)
                            accuracy_numerator[group] += (hits.float() * weights.unsqueeze(-1)).sum(
                                (0, 1, 2)
                            )

        diagnostics: dict = {}
        if want_state:
            diagnostics["predictive_state_by_horizon"] = torch.stack(state_by_horizon)
            diagnostics["next_state_per_feat"] = state_per_feature
        if want_action:
            for group in ACTION_PREDICTION_GROUPS:
                diagnostics[f"predictive_action_ce{group}_by_horizon"] = torch.stack(
                    cross_entropy_by_horizon[group]
                )
                diagnostics[f"predictive_action_entropy{group}_by_horizon"] = torch.stack(
                    entropy_by_horizon[group]
                )
                diagnostics[f"predictive_action_accuracy{group}"] = (
                    accuracy_numerator[group] / denoms[f"action_count_total{group}"]
                )
        return (
            state_total / denoms["state_depths"],
            action_total / denoms["action_depths"],
            diagnostics,
        )

    def _active_component_weights(self) -> torch.Tensor:
        """(K,) current effective weight of every active reward component."""
        return torch.tensor(
            [component.weight for component in self.wrapper.active_components],
            dtype=torch.float32,
            device=self.device,
        )

    def _reward_policy_terms(
        self,
        *,
        batch: MicroBatch,
        ratio: torch.Tensor,
        adv_norm: torch.Tensor,
        actor_f: torch.Tensor,
        actor_sum: torch.Tensor,
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
            actor_f:   (T, b, N) float mask of tokens the actor loss covers.
            actor_sum: Minibatch-total actor token count.
            adv_rms:   Whole-buffer aggregated-advantage mean square.

        Returns:
            Term name → weighted scalar loss, one per component with a non-zero
            weight. Components scheduled to zero contribute no gradient and are
            left out rather than logged as an empty series.
        """
        T, _, N = batch.alive.shape
        team_id = batch.obs[ObsKey.TEAM_ID][:T, :, :N].long()  # (T, b, N)
        comp_weights = self._active_component_weights()  # (K,)
        with torch.no_grad():
            lambda_ij = self._lambda_matrix(team_id, batch.alive, comp_weights)
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

        weighted = (ratio_selected * actor_f).unsqueeze(-1)  # (T, b, N, 1)
        per_component = -(adv_norm_k * weighted).sum((0, 1, 2)) / actor_sum  # (K,)
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
        per_component = (vf_loss_raw * alive_k).sum((0, 1, 2)) / (
            mask_sum * num_components
        )  # (K,)
        coefficient = self._schedule_state.value_function_coef
        return {
            f"value/{name}": coefficient * per_component[index]
            for index, name in enumerate(self._active_names)
        }

    def _lambda_matrix(
        self,
        team_id: torch.Tensor,
        alive: torch.Tensor,
        comp_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Build the normalized credit-assignment weights for one env chunk.

        Allies share signals, enemies are zero-sum (``enemy_neg_k``),
        enemy-only components zero the ally contribution (``ally_zero_k``),
        local components use a diagonal lambda, dead contributing ships are
        zeroed, and each ship's row is normalized to a mean over its alive
        contributors before the component weight is applied.

        Normalization runs on the unweighted pattern so that ``comp_weights``
        stays linear: a row normalized by its own weighted sum divides the
        weight back out. ``clamp(min=1.0)`` therefore bounds the number of
        contributors, not the weight — a single-contributor row (every local
        component, and any global one down to its last alive ship) passes
        through at exactly its weight.

        Shared by the per-update aggregation and by the reward-decomposed
        gradient diagnostic, so the diagnostic cannot drift from the credit
        assignment the policy gradient actually used.

        Args:
            team_id:      (T, b, N) long — raw team labels.
            alive:        (T, b, N) bool — living ships.
            comp_weights: (K,) — current effective per-component weights.

        Returns:
            (T, b, N_i, N_j, K) float32 lambda tensor.
        """
        N = alive.shape[-1]
        ally_lam = torch.where(self.ally_zero_k, 0.0, 1.0)  # (K,)
        enemy_lam = torch.where(self.enemy_neg_k, -1.0, 0.0)  # (K,)
        identity = torch.eye(N, dtype=torch.float32, device=self.device)
        local_lambda = identity[None, None, :, :, None]  # (1, 1, N, N, 1)

        same_team = team_id.unsqueeze(3) == team_id.unsqueeze(2)  # (T, b, N, N)
        alive_j = alive.float().unsqueeze(2).unsqueeze(-1)  # (T, b, 1, N_j, 1)
        global_lambda = (
            same_team.float().unsqueeze(-1) * ally_lam
            + (~same_team).float().unsqueeze(-1) * enemy_lam
        )  # (T, b, N_i, N_j, K)
        # Normalize the *unweighted* pattern, then apply the weight. Dividing a
        # weighted row by its own weighted sum cancels the weight: local
        # components came out at min(w, 1) and global ones lost their weight
        # entirely once w * n_alive exceeded the clamp, so ally_win_weight=1.5
        # trained identically to 0.25. Splitting the two steps keeps the row a
        # mean over contributors while leaving the weight a linear knob.
        pattern = torch.where(self.local_k, local_lambda, global_lambda) * alive_j
        row_sum = pattern.abs().sum(dim=3, keepdim=True).clamp(min=1.0)
        return pattern / row_sum * comp_weights  # (T, b, N_i, N_j, K)

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
            lambda_ij_t = self._lambda_matrix(team_id_t, alive, comp_weights)

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
    def _precompute_transition_labels(self, buf: RolloutBuffer) -> None:
        """Compute one-step state-transition labels once per update.

        Labels come from the stored T+1 observations only — not the policy — so
        computing them here saves num_epochs × num_minibatches redundant passes
        through the coordinator. Targets are computed once over all T+1 steps
        and diffed (labels[t] = f(target[t], target[t+1])).

        The predictive rollout reads the same tensor at an offset: its horizon-h
        belief for base step t is scored against ``labels[t + h]``, the local
        transition out of step t + h. One label array therefore serves every
        horizon, and no target is stored that the rollout data does not already
        determine.
        """
        if not self._predictive_enabled or self.cfg.predictive_state_coef <= 0.0:
            buf.transition_labels = None
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
        buf.transition_labels = self.coordinator.compute_labels(
            targets[:T], targets[1:]
        )  # (T, B, N, pred_dim)

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
        records["grad_diag/microbatches"] = float(accumulator.microbatches)
        records["grad_diag/terms"] = float(len(accumulator.term_names))
        records["grad_diag/seconds"] = seconds
        records["grad_diag/level"] = float(GRADIENT_DIAGNOSTICS_LEVELS.index(self._grad_diag.level))
        return records

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

    def _vector_diagnostic_labels(self) -> dict[str, tuple[str, list[str]]]:
        """Diagnostic key → metric prefix and one label per element of its vector.

        These are the per-element measurements the loss returns as GPU tensors:
        the immediate transition's error per prediction channel, and the three
        horizon-resolved predictive series. Whether prediction quality decays
        smoothly with horizon, and whether the action belief grows less certain
        further out, is exactly what these are for.

        Returns:
            Key → ``(metric prefix, element labels)``. Keys absent from a
            micro-batch's diagnostics simply go unlogged.
        """
        horizons = [f"h{index:02d}" for index in range(self.cfg.prediction_horizon)]
        factors = ["power", "turn", "shoot"]
        labels = {
            # Horizon 0 of the predictive state head is the same quantity the
            # standalone one-step head reported, so it keeps the metric names.
            "next_state_per_feat": ("next_state/", self.coordinator.get_feature_names()),
            "predictive_state_by_horizon": ("predictive/state_loss/", horizons),
        }
        # The unsplit action series, then the same three measurements for each
        # side on its own. Forecasting what your own fleet will do and
        # forecasting what the opposition will do are different problems, and
        # only the split says which one the total is describing.
        for group in ACTION_PREDICTION_GROUPS:
            labels[f"predictive_action_ce{group}_by_horizon"] = (
                f"predictive/action_cross_entropy{group}/",
                horizons,
            )
            labels[f"predictive_action_entropy{group}_by_horizon"] = (
                f"predictive/action_entropy{group}/",
                horizons,
            )
            labels[f"predictive_action_accuracy{group}"] = (
                f"predictive/action_accuracy{group}/",
                factors,
            )
        return labels

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

        comp_weights = torch.tensor(
            [c.weight for c in self.wrapper.active_components],
            dtype=torch.float32,
            device=self.device,
        )  # (K,)

        # Precompute everything that depends only on rollout data (not the
        # policy) once per update instead of once per minibatch: the lambda
        # aggregation and the state-transition labels (primary scale only).
        if not precomputed:
            for scale_idx, buf in enumerate(all_buffers):
                assert isinstance(buf, RolloutBuffer)
                self._precompute_lambda_aggregates(buf, comp_weights, is_primary=(scale_idx == 0))
                if scale_idx > 0:
                    buf.transition_labels = None  # aux scales never use the aux losses
            primary = all_buffers[0]
            assert isinstance(primary, RolloutBuffer)
            self._precompute_transition_labels(primary)

        accum_scalar: dict[str, list[torch.Tensor]] = {
            "loss/total": [],
            "loss/policy_gradient": [],
            "loss/value": [],
            "loss/entropy": [],
            "loss/behavioral_cloning": [],
            "loss/behavioral_cloning_kl": [],
            "loss/scripted_entropy": [],
            "loss/sigreg": [],
            "loss/predictive_state": [],
            "loss/predictive_action": [],
            "loss_proxy/policy_gradient": [],
            "loss_proxy/value": [],
            "loss_proxy/entropy": [],
            "loss_proxy/behavioral_cloning": [],
            "loss_proxy/sigreg": [],
            "loss_proxy/predictive_state": [],
            "loss_proxy/predictive_action": [],
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
        # Per-element diagnostics: one GPU vector per diagnosed quantity, summed
        # across micro-batches and averaged across minibatches, with the labels
        # derived rather than hand-listed. A parallel name list drifts from the
        # coordinator's prediction width silently, and already had — dropping
        # local_log_index, the one channel that says whether fields are being
        # modelled, off the end of a 9-name list against 10 dimensions.
        vector_metric_labels = self._vector_diagnostic_labels()
        vector_accum: dict[str, list[torch.Tensor]] = {key: [] for key in vector_metric_labels}
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
                self.optim.zero_grad()

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
                    ("predictive_state", "predictive_state_loss"),
                    ("predictive_action", "predictive_action_loss"),
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
                    ("loss/predictive_state", "predictive_state"),
                    ("loss/predictive_action", "predictive_action"),
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
                vector_step: dict[str, torch.Tensor] = {}
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
                            measure_grad_split=measure_split and is_primary,
                            grad_terms=accumulator,
                            grad_scale=1.0 / n_scales,
                        )
                        (loss / n_scales).backward()

                        if "grad_norm_actor" in diag:
                            a, c = diag["grad_norm_actor"], diag["grad_norm_critic"]
                            accum_scalar["train/grad_norm_actor"].append(a)
                            accum_scalar["train/grad_norm_critic"].append(c)
                            accum_scalar["train/actor_grad_share"].append(a / (a + c + 1e-12))
                            measure_split = False

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
                            for key in vector_metric_labels:
                                value = diag.get(key)
                                if value is None:
                                    continue
                                stored = vector_step.get(key)
                                vector_step[key] = value if stored is None else stored + value
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
                accum_scalar["loss_proxy/predictive_state"].append(
                    self.cfg.predictive_state_coef * scalar_accum_step["predictive_state"]
                )
                accum_scalar["loss_proxy/predictive_action"].append(
                    self.cfg.predictive_action_coef * scalar_accum_step["predictive_action"]
                )
                accum_scalar["returns/advantage_std"].append(scalar_accum_step["adv_var"] ** 0.5)
                accum_scalar["train/gradient_norm"].append(grad_norm.detach())
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

                for key, value in vector_step.items():
                    vector_accum[key].append(value)

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
        metrics["train/epochs_completed"] = float(epoch_idx + 1)

        for key, tensors in accum_k.items():
            if not tensors:
                continue
            avg = torch.stack(tensors).mean(0).cpu()  # (K,)
            prefix = "returns" if key == "returns/component" else key
            for i, name in enumerate(self._active_names):
                metrics[f"{prefix}/{name}"] = avg[i].item()

        for key, values in vector_accum.items():
            if not values:
                continue
            prefix, names = vector_metric_labels[key]
            # zip, not enumerate over the names: a rollout shorter than the
            # prediction horizon supervises fewer horizons than the label list
            # spells, and the vector is the authority on how many there were.
            averaged = torch.stack(values).mean(0).cpu()
            for name, value in zip(names, averaged, strict=False):
                metrics[f"{prefix}{name}"] = value.item()

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
