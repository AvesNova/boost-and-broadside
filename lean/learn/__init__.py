"""The learner: rewards, rollout collection, returns, the update of the BC stage and of RL,
weight averages and checkpoints."""

from lean.contract import LEVEL_NAMES, LEVELS
from lean.learn.config import LearnConfig, learning_rate
from lean.learn.pool import League
from lean.learn.returns import (
    fallen_team_values,
    generalised_advantages,
    normalise_advantages,
    value_centres,
    value_knees,
)
from lean.learn.rewards import reward_step, tick_events
from lean.learn.rollout import (
    CollectCarry,
    Opponent,
    Rollout,
    collect,
    draw_team_sizes,
    start,
)
from lean.learn.store import RolloutStore
from lean.learn.training import TeacherTerm, make_optimiser, update
from lean.learn.weights import WeightAverages, load_checkpoint, policy_weights, save_checkpoint

__all__ = [
    "CollectCarry",
    "LEVELS",
    "LEVEL_NAMES",
    "LearnConfig",
    "League",
    "Opponent",
    "Rollout",
    "RolloutStore",
    "TeacherTerm",
    "WeightAverages",
    "collect",
    "draw_team_sizes",
    "fallen_team_values",
    "generalised_advantages",
    "learning_rate",
    "load_checkpoint",
    "make_optimiser",
    "normalise_advantages",
    "policy_weights",
    "reward_step",
    "save_checkpoint",
    "start",
    "tick_events",
    "update",
    "value_centres",
    "value_knees",
]
