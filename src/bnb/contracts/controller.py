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

    log_probability: of the chosen action; entropy: of the policy; value: the critic,
    the sum of `values` over levels when a controller gives them.
    values (R, S, levels): the critic per value level, in raw units and the configured level
        order (an outcome level is P(win) - P(loss)).
    outcome (R, 4): the outcome probabilities (win, tie, loss, unresolved), when the critic
        has an outcome level.
    replay: an opaque pytree with leading (R,), what the decision read from the controller's
        carry, so the learner can recompute a segment without replaying it.
    probabilities (R, S, NUM_JOINT) float16: the whole joint action distribution, when the
        learner's trust region needs it (`LearnerConfig.records_distribution`).
    """

    log_probability: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    value: torch.Tensor | None = None
    values: torch.Tensor | None = None
    outcome: torch.Tensor | None = None
    replay: Any = None
    probabilities: torch.Tensor | None = None


class Controller(Protocol):
    """reads: the Obs parts the controller uses, so rollouts store only those."""

    reads: tuple[str, ...]

    def init_carry(self, views: int, device: torch.device) -> Carry:
        """The carry before the first step of R = views views, on device (None for a stateless
        controller)."""

    def step(
        self, carry: Carry, obs: Obs, reset: torch.Tensor, decision_key: torch.Tensor
    ) -> tuple[Action, Carry, Outputs]:
        """obs: views (R, ...); reset: (R,) bool, the view's environment reset before this
        observation (a lifecycle flag, also in obs.globals); decision_key: (R,) int64 in
        [0, 2^32), the decision's key: every random draw of the decision is a pure function
        of it (`bnb.streams`; the rollout keys it by the seed, the environment, its episode
        and tick, and the row's role)."""
        ...
