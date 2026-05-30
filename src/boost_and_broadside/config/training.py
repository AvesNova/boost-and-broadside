"""Training configuration: scale, PPO hyperparameters, and run assembly."""

import dataclasses
from dataclasses import dataclass
from typing import Callable

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
    """One training scale: environment config, token budget fraction, and opponent mix.

    All scales are first-class citizens — they share the same policy, optimizer,
    and return scaler, with gradients weighted by ship-token contribution so every
    token has equal gradient impact.

    Args:
        env_config:         Environment config for this scale.
        token_fraction:     Fraction of TrainConfig.max_tokens allocated to this scale.
                            num_envs is computed as
                            int(max_tokens * fraction / total_tokens / num_steps / num_minibatches)
                            * num_minibatches, where total_tokens = num_ships + num_obstacles.
        scripted_fraction:  Schedule: fraction of this scale's envs that play a scripted opponent.
        avg_model_fraction: Schedule: fraction that play the running-average model opponent.
        league_fraction:    Schedule: fraction that play a league checkpoint opponent.
    """

    env_config: EnvConfig
    token_fraction: float
    scripted_fraction: Callable[[int], float]
    avg_model_fraction: Callable[[int], float]
    league_fraction: Callable[[int], float]


@dataclass(frozen=True)
class TrainConfig:
    """Complete PPO training configuration. No defaults — all values required.

    Sections:
        scales    — environment scale(s); all are first-class (no primary/aux).
        schedule  — time-varying parameters (LR, loss coefficients, roster management).
        rewards   — static reward weights and geometry params.
        ppo       — static PPO hyperparameters.
        league    — league play and ELO tournament parameters.

    All scalar values that vary over training live in ``schedule``.
    Per-scale opponent fractions live in each ``ScaleConfig``.
    Everything here is fixed for the entire run.
    """

    # --- Token budget ---
    max_tokens: int  # target total entity tokens per rollout across all scales

    # --- Scales ---
    scales: tuple[ScaleConfig, ...]  # at least one entry

    # --- Schedule (time-varying) ---
    schedule: TrainingSchedule

    # --- Rewards (static) ---
    rewards: RewardConfig

    # --- PPO hyperparameters ---
    num_steps: int  # rollout length per environment
    num_minibatches: int  # minibatches per epoch
    gamma: float  # discount factor
    gae_lambda: float  # GAE lambda
    clip_coef: float  # PPO clip epsilon
    max_grad_norm: float  # gradient clipping norm
    total_timesteps: int  # total environment steps before stopping
    return_ema_alpha: float  # EMA decay for per-component return percentile scaler
    return_min_span: (
        float  # minimum p95-p5 span (symlog-space) — guards disabled components
    )
    checkpoint_dir: str  # directory to write .pt files

    # --- League play + ELO (static tournament parameters) ---
    league_size: int  # max number of checkpoint entries in the roster
    elo_milestone_gap: float  # add checkpoint to roster every N ELO points gained
    elo_k_factor: float  # ELO K-factor (score sensitivity per match)
    elo_temperature: float  # ELO bandwidth for proximity-weighted sampling
    league_uniform_sampling: bool  # if True, sample league opponents uniformly
    scripted_roster_min_steps: (
        int  # delay adding scripted to roster until this many steps
    )

    # --- Next-state prediction loss ---
    next_state_coef: float = 1.0       # weight for per-step aux prediction loss; 0 to disable
    windowed_loss_coef: float = 0.1    # weight for windowed cumulative bias loss; 0 to disable

    # --- Obstacle cache (None when num_obstacles=0) ---
    obstacle_cache: ObstacleCacheConfig | None = None

    # --- Logging ---
    log_interval: int = 10  # print to terminal every N updates

    # --- Per-component GAE discounts (override global gamma/gae_lambda by name) ---
    # Missing keys fall back to the global gamma / gae_lambda values.
    component_gammas: dict[str, float] = dataclasses.field(default_factory=dict)
    component_lambdas: dict[str, float] = dataclasses.field(default_factory=dict)

    def scale_num_envs(self, sc: ScaleConfig) -> int:
        """Compute num_envs for a scale from its token_fraction and this config's budget.

        Uses integer division to match the original formula and guarantee divisibility:
            num_envs = (max_tokens * fraction // total_tokens // num_steps // num_minibatches)
                       * num_minibatches
        """
        total_tokens = sc.env_config.num_ships + sc.env_config.num_obstacles
        effective = int(self.max_tokens * sc.token_fraction)
        raw = effective // total_tokens // self.num_steps // self.num_minibatches
        return raw * self.num_minibatches

    def __post_init__(self) -> None:
        if len(self.scales) == 0:
            raise ValueError("scales must contain at least one ScaleConfig")
        total_frac = sum(sc.token_fraction for sc in self.scales)
        if total_frac > 1.0 + 1e-9:
            raise ValueError(
                f"sum of token_fractions={total_frac:.4f} exceeds 1.0"
            )
        for i, sc in enumerate(self.scales):
            ne = self.scale_num_envs(sc)
            if ne == 0:
                raise ValueError(
                    f"scales[{i}] computes to num_envs=0 "
                    f"(token_fraction={sc.token_fraction}, max_tokens={self.max_tokens})"
                )
            if ne % self.num_minibatches != 0:
                raise ValueError(
                    f"scales[{i}] num_envs={ne} must be divisible by "
                    f"num_minibatches={self.num_minibatches}"
                )
