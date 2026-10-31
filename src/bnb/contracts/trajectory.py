"""A rollout as the learner receives it: L steps of R learner views of S ships each."""

import torch

from .action import Action
from .obs import Obs
from .tree import contract


@contract
class Trajectory:
    """obs: the parts the model reads, (L, R, ...). action, logp, value, reward: (L, R, S).
    done: (L, R) bool, the view's environment reset after this step.
    """

    obs: Obs
    action: Action
    logp: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
