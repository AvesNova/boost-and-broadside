"""The toy relational model as a controller: sample the joint action (Gumbel-max), report
its log-probability, entropy and the critic. Stateless."""

import torch

from bnb.contracts import Action, Obs, Outputs
from bnb.contracts.action import from_joint
from bnb.models.relational import Relational, outcome_probabilities
from bnb.models.yemong import policy


class RelationalController:
    def __init__(self, model: Relational) -> None:
        self.model = model
        self.reads = model.reads

    def init_carry(self, views: int, device: torch.device) -> None:
        return None

    def step(self, carry: None, obs: Obs, reset: torch.Tensor) -> tuple[Action, None, Outputs]:
        decision = self.model(obs)
        joint, log_probability, entropy = policy.sample(decision.policy_logits)
        outputs = Outputs(
            log_probability=log_probability,
            entropy=entropy,
            value=decision.values.sum(-1),
            values=decision.values,
            outcome=outcome_probabilities(decision),
        )
        return from_joint(joint), carry, outputs
