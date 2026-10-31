"""Yemong as a controller: one team's views in, its ships' actions out.

The controller samples the joint action (Gumbel-max) and makes the sample the carry's
previous action; everything else is the model's.
"""

import dataclasses

import torch

from bnb.contracts import Action, Obs, Outputs
from bnb.contracts.action import from_joint
from bnb.models.yemong import policy
from bnb.models.yemong.model import Carry, Replay, Unrolled, Yemong


class YemongController:
    reads = ("ships", "fields", "globals", "zones")

    def __init__(self, model: Yemong, ships_per_team: int, num_ships: int) -> None:
        self.model = model
        self.ships_per_team, self.num_ships = ships_per_team, num_ships

    def init_carry(self, views: int, device: torch.device) -> Carry:
        return self.model.initial_carry(views, self.ships_per_team, self.num_ships, device)

    def step(self, carry: Carry, obs: Obs, done: torch.Tensor) -> tuple[Action, Carry, Outputs]:
        decision = self.model.step(carry, obs, done)
        joint, logp, entropy = policy.sample(decision.policy_logits)
        carry = dataclasses.replace(decision.carry, previous_action=joint)
        outputs = Outputs(logp=logp, entropy=entropy, value=decision.values.sum(-1),
                          values=decision.values, outcome=decision.outcome_probabilities,
                          replay=decision.replay)
        return from_joint(joint), carry, outputs

    def unroll(self, carry: Carry, obs: Obs, replay: Replay, reset: torch.Tensor) -> Unrolled:
        """A stored segment (L, R, ...): carry is the learner's carry at its start, reset
        (L, R) the per-step `done` flags `step` received."""
        return self.model.unroll(carry, obs, replay, reset)
