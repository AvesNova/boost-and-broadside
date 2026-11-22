"""Settings of the learner: PPO, reward terms, team-size draws and the self-play mix."""

from dataclasses import dataclass

LEVELS = 4
OUTCOME, KILL, DAMAGE, SHAPING = 0, 1, 2, 3


@dataclass
class LearnConfig:
    """Every number the learner uses.

    Discounts are per tick. The game ticks at 30 Hz and the policy decides every tick, so
    one tick is one decision; v1 decided at 30 Hz (60 Hz physics, action repeat 2) and its
    per-decision discounts (0.998, 0.990, 0.982, 0.951) therefore carry over unchanged.
    A different tick rate `f` would use `discount ** (30 / f)`.
    """

    # Returns. Order of levels: outcome, kill and death, damage, shaping.
    discounts: tuple[float, ...] = (0.998, 0.990, 0.982, 0.951)
    gae_lambda: float = 0.95
    advantage_normaliser: str = "std"  # "std" or "percentile"
    advantage_percentiles: tuple[float, float] = (0.05, 0.95)

    # PPO.
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    segment_ticks: int = 128
    learning_rate: float = 3e-4
    warmup_updates: int = 10
    adam_epsilon: float = 1e-5
    gradient_clip: float = 1.0
    entropy_coefficient: float = 0.003
    value_coefficient: float = 0.05
    next_state_coefficient: float = 0.1
    bc_coefficient: float = 2.0
    bc_gate: float = 0.45

    # Categorical critic: the outermost bin centres per level, in return units. The learner
    # owns the support; each must hold that level's returns with three bins of margin.
    value_low: tuple[float, ...] = (-1.5, -2.0, -2.0, -4.0)
    value_high: tuple[float, ...] = (1.5, 2.0, 2.0, 4.0)
    value_sigma_ratio: float = 0.75  # Gaussian width of a target, in bin widths

    # Rewards. The side causing an event is paid twice the victim's charge.
    win_reward: float = 1.0
    death_penalty: float = 0.283
    kill_reward: float = 0.283
    damage_dealt_reward: float = 0.548
    damage_taken_penalty: float = 0.274
    friendly_damage_penalty: float = 0.548
    damage_unit: float = 1.0  # damage that counts as 1.0, in the game's damage units
    individual_share: float = 0.5  # rest of kill and damage terms is the team mean
    facing_reward: float = 0.09
    closing_reward: float = 0.08
    speed_unit: float = 100.0  # metres per second that count as closing speed 1.0
    shaping_full_steps: float = 100e6  # shaping scale is 1 until here
    shaping_floor: float = 0.05
    shaping_floor_steps: float = 400e6  # and decays exponentially to the floor here

    # Scenario draws at reset.
    team_size_minimum: int = 4
    team_size_maximum: int = 4
    equal_teams: bool = True

    # Self-play.
    scripted_share: float = 0.5  # share of environments that always face the scripted agent
    snapshot_every: int = 10  # updates between additions to the pool
    snapshot_capacity: int = 8
    persona_spread: float = 0.0
    win_rate_momentum: float = 0.9


def bc_weight(config: LearnConfig, scripted_win_rate: float) -> float:
    """Coefficient of the BC term: bc_coefficient * max(0, 1 - win_rate / gate)."""
    return config.bc_coefficient * max(0.0, 1.0 - scripted_win_rate / config.bc_gate)


def shaping_scale(config: LearnConfig, steps: float) -> float:
    """Taper of the shaping level: 1 until shaping_full_steps, then exponential decay
    reaching shaping_floor at shaping_floor_steps and held there. `steps` counts
    environment ticks collected so far."""
    if steps <= config.shaping_full_steps:
        return 1.0
    span = config.shaping_floor_steps - config.shaping_full_steps
    fraction = min(1.0, (steps - config.shaping_full_steps) / span)
    return config.shaping_floor**fraction


def learning_rate(config: LearnConfig, update_index: int) -> float:
    """Learning rate of update `update_index` (from 0): linear warm-up, then constant."""
    return config.learning_rate * min(1.0, (update_index + 1) / max(1, config.warmup_updates))
