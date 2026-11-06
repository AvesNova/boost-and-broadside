"""Yemong as a controller: one team's views in, its ships' actions out.

The controller samples the joint action (Gumbel-max) and makes the sample the carry's
previous action; everything else is the model's.
"""

import dataclasses

import torch

from bnb.contracts import Action, Obs, Outputs
from bnb.contracts.action import from_joint
from bnb.models.yemong import policy
from bnb.models.yemong.model import Ablations, Carry, Replay, Unrolled, Yemong


class YemongController:
    """Yemong as a controller: the carry is the model's, with the sampled joint action as its
    previous action; the outputs add the critic's values per level, the outcome probabilities and
    the replay, and with `distribution` the whole joint action distribution (float16)."""

    reads = ("ships", "fields", "globals", "zones")

    def __init__(
        self, model: Yemong, ships_per_team: int, num_ships: int, distribution: bool = False
    ) -> None:
        self.model = model
        self.ships_per_team, self.num_ships = ships_per_team, num_ships
        self.distribution = distribution

    def init_carry(self, views: int, device: torch.device) -> Carry:
        """The model's initial carry for R = views views, on device."""
        return self.model.initial_carry(views, self.ships_per_team, self.num_ships, device)

    @staticmethod
    def forget(carry: Carry, rows: torch.Tensor) -> Carry:
        """The carry with the trunk's memory (its state and convolution buffer) zeroed on
        `rows` (R,) bool and the rest kept: the belief (observations, last sightings,
        forecasts), the previous actions and the position offset. What a league slot's rows
        keep when another network takes the slot (`bnb.runtime.opponents.OpponentSlots.cut`)."""

        def zeroed(tensor: torch.Tensor) -> torch.Tensor:
            shape = rows.shape + (1,) * (tensor.dim() - rows.dim())
            return torch.where(rows.view(shape), 0, tensor)

        return dataclasses.replace(
            carry,
            state=zeroed(carry.state),
            convolution_buffer=zeroed(carry.convolution_buffer),
        )

    def step(
        self, carry: Carry, obs: Obs, reset: torch.Tensor, ablations: Ablations | None = None
    ) -> tuple[Action, Carry, Outputs]:
        """One decision for views (R, ...); see `Controller.step`. ablations: the model's
        inference-time ablations per view (`Ablations`; none by default)."""
        decision = self.model.step(carry, obs, reset, ablations)
        joint, log_probability, entropy = policy.sample(decision.policy_logits)
        carry = dataclasses.replace(decision.carry, previous_action=joint)
        outputs = Outputs(
            log_probability=log_probability,
            entropy=entropy,
            value=decision.values.sum(-1),
            values=decision.values,
            outcome=decision.outcome_probabilities,
            replay=decision.replay,
            probabilities=(
                torch.softmax(decision.policy_logits.float(), -1).to(torch.float16)
                if self.distribution
                else None
            ),
        )
        return from_joint(joint), carry, outputs

    def unroll(self, carry: Carry, obs: Obs, replay: Replay, reset: torch.Tensor) -> Unrolled:
        """A stored segment (L, R, ...): carry is the learner's carry at its start, reset
        (L, R) the per-step `reset` flags `step` received."""
        return self.model.unroll(carry, obs, replay, reset)
