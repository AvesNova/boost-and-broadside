"""Training configuration: scale, PPO hyperparameters, and run assembly."""

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
        schedule  — all time-varying parameters (LR, loss coefficients, fractions).
        rewards   — static reward weights and geometry params.
        ppo       — static PPO hyperparameters.
        league    — league play and ELO tournament parameters.

    All scalar values that vary over training live in ``schedule``.
    Everything here is fixed for the entire run.
    """

    # --- Scales ---
    scales: tuple[ScaleConfig, ...]  # at least one entry

    # --- Schedule (time-varying) ---
    schedule: TrainingSchedule

    # --- Rewards (static) ---
    rewards: RewardConfig

    # --- PPO hyperparameters ---
    num_steps: int  # rollout length per environment
    num_epochs: int  # PPO update epochs per rollout
    num_minibatches: int  # minibatches per epoch (scales[0].num_envs must be divisible)
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

    # --- Obstacle cache (None when num_obstacles=0) ---
    obstacle_cache: ObstacleCacheConfig | None = None

    # --- Logging ---
    log_interval: int = 10  # print to terminal every N updates

    def __post_init__(self) -> None:
        if len(self.scales) == 0:
            raise ValueError("scales must contain at least one ScaleConfig")
        primary_envs = self.scales[0].num_envs
        if primary_envs % self.num_minibatches != 0:
            raise ValueError(
                f"scales[0].num_envs={primary_envs} must be divisible by "
                f"num_minibatches={self.num_minibatches}"
            )
