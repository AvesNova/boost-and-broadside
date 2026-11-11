"""The toy relational model (and the nearest-neighbour MLP, which has its outputs) as a
controller: sample the joint action (Gumbel-max), report its log-probability, entropy and
the critic. Stateless."""

import torch

from bnb.contracts import Action, Obs, Outputs
from bnb.contracts.action import from_joint
from bnb.models.nearest import Nearest
from bnb.models.relational import Relational, outcome_probabilities
from bnb.models.yemong import policy


class RelationalController:
    """The relational model as a controller: samples the joint action from its policy logits and
    reports its log-probability, the entropy, the critic's values per level and the outcome
    probabilities."""

    def __init__(self, model: Relational | Nearest) -> None:
        self.model = model
        self.reads = model.reads

    def init_carry(self, views: int, device: torch.device) -> None:
        """No carry: the relational model is stateless."""
        return None

    def step(
        self, carry: None, obs: Obs, reset: torch.Tensor, temperature: torch.Tensor | None = None
    ) -> tuple[Action, None, Outputs]:
        """One decision for views (R, ...); see `Controller.step`. temperature: (R,) each
        view's sampling temperature (`policy.tempered`; None: the policy as it is)."""
        decision = self.model(obs)
        logits = decision.policy_logits
        if temperature is not None:
            logits = policy.tempered(logits, temperature)
        joint, log_probability, entropy = policy.sample(logits)
        outputs = Outputs(
            log_probability=log_probability,
            entropy=entropy,
            value=decision.values.sum(-1),
            values=decision.values,
            outcome=outcome_probabilities(decision),
        )
        return from_joint(joint), carry, outputs
