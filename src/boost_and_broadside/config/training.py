"""Training configuration: scale, PPO hyperparameters, and run assembly."""

import dataclasses
from dataclasses import dataclass

from boost_and_broadside.config.core import EnvConfig, RewardConfig
from boost_and_broadside.config.schedule import TrainingSchedule


@dataclass(frozen=True)
class FieldMapConfig:
    """Bounded static refractive-field map generation settings.

    Map construction happens once before rollout. Every candidate is checked for
    a laminar (strictly nested or disjoint) relationship, including transition
    bands; no hierarchy discovery or rejection sampling occurs in ``step()``.
    """

    cache_size: int
    max_generation_attempts: int = 128
    nesting_probability: float = 0.25

    def __post_init__(self) -> None:
        if self.cache_size < 1:
            raise ValueError("cache_size must be positive")
        if self.max_generation_attempts < 1:
            raise ValueError("max_generation_attempts must be positive")
        if not 0.0 <= self.nesting_probability <= 1.0:
            raise ValueError("nesting_probability must lie in [0, 1]")


# Import compatibility for callers that only stored the old class name. Its
# constructor intentionally follows the new static-map API: orbital/PBD options
# cannot be meaningfully migrated.
ObstacleCacheConfig = FieldMapConfig


@dataclass(frozen=True)
class EloCalibrateConfig:
    """Configuration for the post-training Elo calibration tournament.

    Used by ``--mode elo_calibrate`` to re-rate a finished run. Unlike
    EloEvalConfig this costs nothing during training — it runs once afterwards,
    so the budget is bounded by patience rather than by throughput.

    num_envs is both the parallel width and the games per batch: every env plays
    exactly one episode per batch. Raising it buys precision per batch at a
    proportional increase in batch wall time, so it trades against max_batches
    at roughly fixed total cost; prefer more envs to more batches, since each
    batch also pays for its slowest episode to reach the horizon.

    Precision improves as 1/sqrt(games), so halving target_stderr costs about
    four times the games.
    """

    num_envs: int  # parallel envs per batch, i.e. games played per batch
    target_stderr: float  # stop once every rating is pinned to within this
    max_batches: int  # cap, so an unreachable target cannot run forever
    # How draws enter the likelihood: "half_win" or "decisive". See TIE_MODES in
    # modes/elo_calibrate.py. Both are always fit and reported; this selects
    # which one drives allocation, the gauge, and the convergence test.
    tie_mode: str = "half_win"
    # Render the charts built on the secondary draw convention. Both conventions
    # are always fit and written to JSON; this only controls the extra plots,
    # which are a diagnostic rather than a result.
    plot_decisive: bool = False
    # Virtual decisive games per player, split for and against the anchor.
    # Keeps a player that never loses from having an infinite rating.
    prior_games: float = 1.0
    # Interior semi-random reference rungs added to the tournament field, as
    # scripted-action probabilities in (0, 1). They connect random to scripted
    # through informative matchups instead of one near-deterministic link,
    # tightening the weak end of the scale. Empty disables the ladder.
    reference_probabilities: tuple[float, ...] = ()


@dataclass(frozen=True)
class EloEvalConfig:
    """Configuration for continuous in-training Elo ladder evaluation.

    envs_per_matchup and step_interval trade against each other at fixed cost.
    The eval env advances num_steps / step_interval steps per update, so with
    both scaled together the rated-game rate and the env-step cost hold constant
    while the episode span does not:

        games per update = envs_per_matchup × (num_steps / step_interval)
                           / max_episode_steps                    — invariant
        episode span     = max_episode_steps
                           / (num_steps / step_interval)          — halves

    Episode span is the lag between the live policy and what its rating
    reflects, since the evaluator holds a live reference to the policy and its
    weights change mid-episode. Span also sets how long live Elo stays flat
    after a milestone, because promotion reseeds both slots that feed it.

    The floor is kernel efficiency: each eval step fires up to eight separate
    policy forward passes, which go launch-bound as envs_per_matchup shrinks.
    """

    envs_per_matchup: int
    step_interval: int  # eval steps once per this many rollout steps
    k_factor: float
    scripted_elo_init: float  # initial estimate for the scripted agent's floating rating
    window_size: int
    # Rated games the floating checkpoint must accumulate before it may be
    # frozen at the next milestone. 0 disables the gate.
    min_games_to_freeze: int = 0


@dataclass(frozen=True)
class ScaleConfig:
    """One training scale: an environment config paired with a batch size.

    All scales share the same policy, optimizer, and return scaler.
    Gradients are accumulated across scales before each optimizer step.
    scales[0] in TrainConfig is the primary scale and supports scripted /
    avg-model / league opponents; scales[1:] run pure self-play.

    Args:
        env_config: Environment config for this scale — num_ships defines N.
        num_envs:   Parallel environments. Set inversely proportional to N so
                    total ships-per-update stays constant across scales.
    """

    env_config: EnvConfig
    num_envs: int


@dataclass(frozen=True)
class TrainConfig:
    """Complete PPO training configuration.

    Sections:
        scales    — environment scale(s); scales[0] is primary.
        paradigm  — how rollout actions are generated and which ships train.
        schedule  — all time-varying parameters (LR, loss coefficients, fractions).
        rewards   — static reward weights and geometry params.
        ppo       — static PPO hyperparameters.
        league    — league play and Elo tournament parameters.

    All scalar values that vary over training live in ``schedule``.
    Everything here is fixed for the entire run. Core training values are
    required; defaults are reserved for optional or disabled features.

    Paradigms:
        "ego_pass"    — two batched policy passes per step (raw obs + team-flipped
                        obs) so every ship acts from a perspective where its own
                        team is labelled 0. Only team 0 ships contribute to the
                        actor/BC losses; opponents always play team 1.
        "shared_pass" — one policy pass per step on raw obs; the model emits
                        actions for both teams and all ships contribute to the
                        actor/BC losses. In opponent envs a per-episode random
                        flag picks which team the opponent controls.
    """

    # --- Scales ---
    scales: tuple[ScaleConfig, ...]  # at least one entry

    # --- Training paradigm ---
    paradigm: str  # "ego_pass" | "shared_pass" — see class docstring

    # --- Schedule (time-varying) ---
    schedule: TrainingSchedule

    # --- Rewards (static) ---
    rewards: RewardConfig

    # --- PPO hyperparameters ---
    num_steps: int  # rollout length per environment
    # Fixed-width rollout shards collected under one policy snapshot before each
    # PPO update. This grows the logical batch in host RAM without growing the
    # GPU-resident environment or rollout buffer.
    rollouts_per_update: int
    num_minibatches: int  # minibatches per epoch (scales[0].num_envs must be divisible)
    gamma: float  # discount factor
    gae_lambda: float  # GAE lambda
    clip_coef: float  # PPO clip epsilon
    max_grad_norm: float  # gradient clipping norm
    total_timesteps: int  # total environment steps before stopping
    return_ema_alpha: float  # EMA decay for per-component return percentile scaler
    # Degeneracy epsilons for the two per-component scalers. Both are
    # divide-by-zero guards and nothing more: they must sit far below the
    # smallest span/RMS any *active* component really has, or they quietly
    # rescale that component's critic targets and policy-gradient share. The
    # trainer logs scaler/floor_bound/<name> and warns when one binds.
    return_min_span: float  # ReturnScaler p95-p5 epsilon (symlog-space)
    advantage_min_rms: float  # AdvantageScaler RMS epsilon (symlog-space)
    checkpoint_dir: str  # directory to write .pt files

    # --- League play + Elo (static tournament parameters) ---
    league_size: int  # max number of checkpoint policies kept loaded for league play
    # Ladder-snapshot grid spacing: snapshots are taken as normalized Elo crosses
    # each multiple of this value, so rungs land at absolute heights (200, 400, …)
    # that are comparable across runs rather than drifting from run history.
    elo_milestone_gap: float
    elo_temperature: float  # Elo bandwidth for proximity-weighted sampling
    league_uniform_sampling: bool  # if True, sample league opponents uniformly
    elo_eval: EloEvalConfig  # continuous evaluation batch and rating parameters
    bc_winrate_target: float  # win rate vs scripted at which the BC aux loss reaches zero
    histogram_interval: int  # record expensive histograms every N updates

    # Maximum entity samples used for host-backed return percentiles. None keeps
    # exact quantiles; a bounded sample prevents CPU sorting from dominating large
    # logical batches.
    return_quantile_samples: int | None = None

    # --- Gradient accumulation (memory-only, per-machine knob) ---
    # Max entity-tokens (envs × num_steps × (N+M)) per backward pass. Minibatches
    # larger than this are split into micro-batches whose gradients are accumulated
    # before each optimizer step, with loss terms normalized by minibatch-total
    # denominators so the update is equivalent to the unsplit minibatch. Does not
    # change training statistics — set it per GPU to fit VRAM. None = no splitting.
    microbatch_tokens: int | None = None

    # --- Next-state prediction loss ---
    next_state_coef: float = 1.0  # weight for per-step aux prediction loss; 0 to disable
    windowed_loss_coef: float = 0.1  # weight for windowed cumulative bias loss; 0 to disable

    # --- Static field map cache (None when num_fields=0) ---
    field_map: FieldMapConfig | None = None

    @property
    def obstacle_cache(self) -> FieldMapConfig | None:
        """Deprecated read-only alias for pre-field integrations."""

        return self.field_map

    # --- Logging ---
    log_interval: int = 10  # print to terminal every N updates

    # --- Per-component GAE discounts (override global gamma/gae_lambda by name) ---
    # Missing keys fall back to the global gamma / gae_lambda values.
    component_gammas: dict[str, float] = dataclasses.field(default_factory=dict)
    component_lambdas: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.scales) == 0:
            raise ValueError("scales must contain at least one ScaleConfig")
        has_fields = any(scale.env_config.num_fields > 0 for scale in self.scales)
        if has_fields and self.field_map is None:
            raise ValueError("field_map is required when any training scale has num_fields > 0")
        if not has_fields and self.field_map is not None:
            raise ValueError("field_map must be None when every training scale has zero fields")
        if self.paradigm not in ("ego_pass", "shared_pass"):
            raise ValueError(f"paradigm must be 'ego_pass' or 'shared_pass', got {self.paradigm!r}")
        primary_envs = self.scales[0].num_envs
        if primary_envs % self.num_minibatches != 0:
            raise ValueError(
                f"scales[0].num_envs={primary_envs} must be divisible by "
                f"num_minibatches={self.num_minibatches}"
            )
        if self.microbatch_tokens is not None and self.microbatch_tokens < 1:
            raise ValueError(
                f"microbatch_tokens must be positive or None, got {self.microbatch_tokens}"
            )
        if self.rollouts_per_update < 1:
            raise ValueError(
                f"rollouts_per_update must be positive, got {self.rollouts_per_update}"
            )
        if self.return_quantile_samples is not None and self.return_quantile_samples < 1:
            raise ValueError(
                "return_quantile_samples must be positive or None, "
                f"got {self.return_quantile_samples}"
            )
        if not 0.0 < self.bc_winrate_target <= 1.0:
            raise ValueError(f"bc_winrate_target must be in (0, 1], got {self.bc_winrate_target}")
