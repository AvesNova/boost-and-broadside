"""A rollout as the learner receives it: L steps of R learner views of S ships each."""

from typing import Any

import torch

from .action import Action
from .obs import Obs
from .tree import contract

# The reward levels, from the least to the most directly tied to winning (rewards spec 1):
# the order of Trajectory.reward's last axis and of the critic's levels.
REWARD_LEVELS = ("damage", "kill_death", "objective_progress", "objective", "outcome")


@contract
class Trajectory:
    """obs: the parts the model reads, (L, R, ...). action, logp, value: (L, R, S).
    reward: (L, R, S, len(REWARD_LEVELS)), per level, weighted.
    done: (L, R) bool, the view's environment reset after this step.

    When the learner's controller gives them (else None): values (L, R, S, K), outcome
    (L, R, 4) and replay (a pytree, (L, R, ...)) from its Outputs, and initial_carry, the
    learner's carry at the segment's start (R, ...).
    """

    obs: Obs
    action: Action
    logp: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    values: torch.Tensor | None = None
    outcome: torch.Tensor | None = None
    replay: Any = None
    initial_carry: Any = None
