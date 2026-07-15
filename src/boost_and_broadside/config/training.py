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
    """Complete PPO training configuration. No defaults — all values required.

    Sections:
        scales    — environment scale(s); scales[0] is primary.
        paradigm  — how rollout actions are generated and which ships train.
        schedule  — all time-varying parameters (LR, loss coefficients, fractions).
        rewards   — static reward weights and geometry params.
        ppo       — static PPO hyperparameters.
        league    — league play and ELO tournament parameters.

    All scalar values that vary over training live in ``schedule``.
    Everything here is fixed for the entire run.

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

    # --- League play + ELO (static tournament parameters) ---
    league_size: int  # max number of checkpoint entries in the roster
    elo_milestone_gap: float  # add checkpoint to roster every N ELO points gained
    elo_k_factor: float  # ELO K-factor (score sensitivity per match)
    elo_temperature: float  # ELO bandwidth for proximity-weighted sampling
    league_uniform_sampling: bool  # if True, sample league opponents uniformly
    # Avg-model accumulation starts once normalized training ELO (vs the random
    # anchor) reaches this barrier; once started it never stops.
    avg_model_elo_threshold: float = 1000.0

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
