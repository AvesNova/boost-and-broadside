"""The interface every controller implements: models, scripted agents, random, humans.

A controller plays one team. It receives that team's view and returns actions for that
team's S ships. Following PyTorch idiom, a model's parameters live in the controller's own
module; a league slot is a separate controller whose weights are copied in.
"""

from typing import Any, Protocol

import torch

from .action import Action
from .obs import Obs
from .tree import contract

Carry = Any  # a pytree of tensors the runtime stores but never inspects (None if stateless)


@contract
class Outputs:
    """Per controlled ship, shaped (R, S) for R views. None for controllers that do not learn.

    logp: log-probability of the chosen action; entropy: of the policy; value: the critic.
    """

    logp: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    value: torch.Tensor | None = None


class Controller(Protocol):
    """reads: the Obs parts the controller uses, so rollouts store only those."""

    reads: tuple[str, ...]

    def init_carry(self, views: int, device: torch.device) -> Carry: ...

    def step(self, carry: Carry, obs: Obs, done: torch.Tensor) -> tuple[Action, Carry, Outputs]:
        """obs: views (R, ...); done: (R,) bool, the view's environment reset before this
        observation (a lifecycle flag, also in obs.globals)."""
        ...
