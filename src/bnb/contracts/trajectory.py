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
    """obs: the parts the model reads, (L, R, ...). action: every row's action (L, R, N), in
    the view's row order (own ships first; the learner's are [..., :S]). log_probability, value:
    (L, R, S). reward: (L, R, S, len(REWARD_LEVELS)), per level, weighted.
    done: (L, R) bool, the view's environment reset after this step. result: (L, R) int32,
    the match result for the view's team (+1 win, -1 loss, 0 draw), nonzero only where done.

    When the learner's controller gives them (else None): values (L, R, S, levels), outcome
    (L, R, 4), replay (a pytree, (L, R, ...)) and probabilities (L, R, S, NUM_JOINT) float16
    (the rollout's whole joint action distribution) from its Outputs, and initial_carry, the
    learner's carry at the segment's start (R, ...).

    The label inputs (learner spec 2), from the runtime: alive (L, R, N) and valued
    (L, R, S) bool, each row alive at t and each own ship alive or waiting to respawn (the
    observation's, kept apart for the returns); truth (L, R, N - S, len(SHIP_COLUMNS)), the
    enemy rows of the fog-off privileged view (`Privileged.ships`; an own row equals the
    observation's); final_truth (R, N, len(SHIP_COLUMNS)), the privileged rows of the
    observation after the last step; capture, the shot capture (a pytree, (L, R, N, ...):
    each row's launch state and its move this tick, `bnb.learn.shots.ShotCapture`).
    """

    obs: Obs
    action: Action
    log_probability: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    result: torch.Tensor | None = None
    values: torch.Tensor | None = None
    outcome: torch.Tensor | None = None
    replay: Any = None
    initial_carry: Any = None
    probabilities: torch.Tensor | None = None
    alive: torch.Tensor | None = None
    valued: torch.Tensor | None = None
    truth: torch.Tensor | None = None
    final_truth: torch.Tensor | None = None
    capture: Any = None
