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

    logp: log-probability of the chosen action; entropy: of the policy; value: the critic,
    the sum of `values` over levels when a controller gives them.
    values (R, S, K): the critic per reward level, in raw units and the configured level
        order (an outcome level is P(win) - P(loss)).
    outcome (R, 4): the outcome probabilities (win, tie, loss, unresolved).
    replay: an opaque pytree with leading (R,), what the decision read from the controller's
        carry, so the learner can recompute a segment without replaying it.
    """

    logp: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    value: torch.Tensor | None = None
    values: torch.Tensor | None = None
    outcome: torch.Tensor | None = None
    replay: Any = None


class Controller(Protocol):
    """reads: the Obs parts the controller uses, so rollouts store only those."""

    reads: tuple[str, ...]

    def init_carry(self, views: int, device: torch.device) -> Carry: ...

    def step(self, carry: Carry, obs: Obs, done: torch.Tensor) -> tuple[Action, Carry, Outputs]:
        """obs: views (R, ...); done: (R,) bool, the view's environment reset before this
        observation (a lifecycle flag, also in obs.globals)."""
        ...
