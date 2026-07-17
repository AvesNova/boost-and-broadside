"""Training configuration: scale, PPO hyperparameters, and run assembly."""

import dataclasses
from dataclasses import dataclass

from boost_and_broadside.config.core import EnvConfig, RewardConfig
from boost_and_broadside.config.schedule import TrainingSchedule


@dataclass(frozen=True)
class ObstacleCacheConfig:
    """Config for pre-training obstacle map generation.

    A large batch of environments is simulated with harmonic gravity + PBD until
    obstacles converge to stable orbits. Converged snapshots are stored and
    replayed (with random rotation + translation) throughout training.

    Args:
        num_cache_envs: Parallel envs to simulate during generation.
        cache_size:     Desired number of stored converged snapshots.
        max_steps:      Max simulation steps before giving up on stragglers.
    """

    num_cache_envs: int
    cache_size: int
    max_steps: int


@dataclass(frozen=True)
class LeagueEvalConfig:
    """Configuration for continuous information-scheduled league evaluation."""

    eval_num_envs: int
    step_interval: int
    eval_pairs: int
    eval_slots: int
    # Rollouts each scheduled pair block persists. Envs are hard-reset at each
    # block assignment, so blocks should span at least one full episode at the
    # eval stride: eval steps per block = eval_block_rollouts * num_steps /
    # step_interval >= max_episode_steps.
    eval_block_rollouts: int
    # Eval-only episode truncation. Shorter than the training env's cap so
    # draw-bound (evenly matched) pairs complete and record sooner; decisive
    # games end well under this anyway.
    max_episode_steps: int

    def __post_init__(self) -> None:
        for name in (
            "eval_num_envs",
            "step_interval",
            "eval_pairs",
            "eval_slots",
            "eval_block_rollouts",
            "max_episode_steps",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


@dataclass(frozen=True)
class ScaleConfig:
    """One training scale: an environment config paired with a batch size.

    All scales share the same policy, optimizer, and return scaler.
    Gradients are accumulated across scales before each optimizer step.
    scales[0] in TrainConfig is the primary scale and supports the unified
    PFSP opponent slice; scales[1:] run pure self-play.

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
        league    — league play and ELO tournament parameters.

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
    num_minibatches: int  # minibatches per epoch (scales[0].num_envs must be divisible)
    gamma: float  # discount factor
    gae_lambda: float  # GAE lambda
    clip_coef: float  # PPO clip epsilon
    max_grad_norm: float  # gradient clipping norm
    total_timesteps: int  # total environment steps before stopping
    return_ema_alpha: float  # EMA decay for per-component return percentile scaler
    return_min_span: float  # minimum p95-p5 span (symlog-space) — guards disabled components
    checkpoint_dir: str  # directory to write .pt files

    # --- League curriculum and Bradley-Terry rating ---
    league_size: int  # max number of checkpoint entries in the roster
    league_k: int  # maximum distinct PFSP opponents per rollout
    league_admission_interval: int  # PPO updates between frozen checkpoint admissions
    # Rollouts each sampled opponent is held before staggered replacement —
    # episodes must usually finish under one opponent for valid attribution.
    opponent_hold_rollouts: int
    pfsp_mode: str  # "hard" or "variance"
    pfsp_exponent: float  # exponent for hard-PFSP weights
    live_rating_decay: float  # per-update count decay for the live policy
    avg_rating_decay: float  # per-update count decay for the average policy
    bt_prior_draws: float  # virtual draws regularizing each played pair
    # Volume-proportional extra prior per played pair (draws per game). Caps a
    # fully saturated pair's implied gap at ~400*log10(2/frac) regardless of
    # how many games it logs.
    bt_prior_frac: float
    admission_prior_games: float  # virtual live-vs-copy draws on admission
    league_eval: LeagueEvalConfig
    # BC aux decay is gated on the measured live-vs-scripted win rate: the
    # coefficient scales by max(0, 1 - wr / bc_winrate_target), i.e. full BC
    # while scripted dominates, zero at wr >= target (parity with the teacher).
    bc_winrate_target: float
    histogram_interval: int  # record expensive histograms every N updates
    # Avg-model accumulation starts once the measured live-vs-scripted win
    # rate reaches this barrier; once started it never stops.
    avg_model_winrate_threshold: float = 0.5
    # Early checkpoint admission: admit a new rung when live's expected score
    # vs the newest checkpoint reaches this trigger (both must be identified),
    # at most once per admission_min_interval updates. The fixed
    # league_admission_interval remains as the slow-cadence fallback.
    admission_winrate_trigger: float = 0.65
    admission_min_interval: int = 3

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

    # --- Obstacle cache (None when num_obstacles=0) ---
    obstacle_cache: ObstacleCacheConfig | None = None

    # --- Logging ---
    log_interval: int = 10  # print to terminal every N updates

    # --- Per-component GAE discounts (override global gamma/gae_lambda by name) ---
    # Missing keys fall back to the global gamma / gae_lambda values.
    component_gammas: dict[str, float] = dataclasses.field(default_factory=dict)
    component_lambdas: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.scales) == 0:
            raise ValueError("scales must contain at least one ScaleConfig")
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
        if self.league_k < 1:
            raise ValueError(f"league_k must be positive, got {self.league_k}")
        if self.league_admission_interval < 1:
            raise ValueError(
                f"league_admission_interval must be positive, got {self.league_admission_interval}"
            )
        if self.opponent_hold_rollouts < 1:
            raise ValueError(
                f"opponent_hold_rollouts must be positive, got {self.opponent_hold_rollouts}"
            )
        if not 0.0 <= self.live_rating_decay <= 1.0:
            raise ValueError("live_rating_decay must be in [0, 1]")
        if not 0.0 <= self.avg_rating_decay <= 1.0:
            raise ValueError("avg_rating_decay must be in [0, 1]")
        if self.pfsp_mode not in ("hard", "variance"):
            raise ValueError(f"invalid pfsp_mode {self.pfsp_mode!r}")
        if self.pfsp_exponent <= 0.0:
            raise ValueError("pfsp_exponent must be positive")
        if self.bt_prior_draws < 0.0 or self.admission_prior_games < 0.0:
            raise ValueError("BT prior counts must be non-negative")
        if self.bt_prior_frac < 0.0:
            raise ValueError("bt_prior_frac must be non-negative")
        if not 0.0 < self.bc_winrate_target <= 1.0:
            raise ValueError("bc_winrate_target must be in (0, 1]")
        if not 0.0 < self.avg_model_winrate_threshold <= 1.0:
            raise ValueError("avg_model_winrate_threshold must be in (0, 1]")
        if not 0.0 < self.admission_winrate_trigger < 1.0:
            raise ValueError("admission_winrate_trigger must be in (0, 1)")
        if self.admission_min_interval < 1:
            raise ValueError("admission_min_interval must be positive")
