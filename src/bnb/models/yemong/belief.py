"""The belief store (spec 5.2): what a team believes about every ship row, assimilating
sight before a decision and storing the forecast after it.

Visible ships are truth; a hidden ship the team has seen since it last spawned keeps the
model's forecast; a ship never seen since its spawn is not valid (zeros, uniform action).
"""

import torch

from bnb.contracts.tree import contract


@contract
class Belief:
    """Per view and ship row, (..., N) leading axes.

    valid: bool, the row holds a belief. age: int32 ticks since the ship was last seen.
    means (..., N, 12) and spreads (..., N, 15): the ship state (state_code.MEANS and
    SPREADS order, physical units). action (..., N, 42): forecast probabilities of the
    ship's next joint action (read for enemy rows only).
    """

    valid: torch.Tensor
    age: torch.Tensor
    means: torch.Tensor
    spreads: torch.Tensor
    action: torch.Tensor


def blank(leading: tuple[int, ...], means: int, spreads: int, actions: int, device=None,
          dtype=torch.float32) -> Belief:
    """No beliefs: invalid, zero, uniform actions."""
    return Belief(valid=torch.zeros(leading, dtype=torch.bool, device=device),
                  age=torch.zeros(leading, dtype=torch.int32, device=device),
                  means=torch.zeros(leading + (means,), dtype=dtype, device=device),
                  spreads=torch.zeros(leading + (spreads,), dtype=dtype, device=device),
                  action=torch.full(leading + (actions,), 1.0 / actions, dtype=dtype,
                                    device=device))


def observe(belief: Belief, truth: torch.Tensor, visible: torch.Tensor,
            spawned: torch.Tensor) -> Belief:
    """Assimilate one observation: truth (..., N, 12) is read only where visible."""
    valid = (belief.valid & ~spawned) | visible
    seen = visible[..., None]
    held = valid[..., None]
    means = torch.where(seen, truth, torch.where(held, belief.means, 0.0))
    spreads = torch.where(seen | ~held, 0.0, belief.spreads)
    age = torch.where(visible, 0, torch.where(valid, belief.age + 1, 0)).to(torch.int32)
    actions = belief.action.shape[-1]
    action = torch.where(held, belief.action, 1.0 / actions)
    return Belief(valid=valid, age=age, means=means, spreads=spreads, action=action)


def advance(belief: Belief, prediction: torch.Tensor,
            action_logits: torch.Tensor) -> tuple[Belief, torch.Tensor]:
    """Store the forecast: prediction (..., N, 12 + 15) means then spreads, decoded from the
    next-state head, and the enemy-action head's logits for every row. Non-finite prediction
    entries become 0 and are counted, per leading index (...)."""
    finite = torch.isfinite(prediction)
    nonfinite = (~finite).flatten(-2).sum(-1)
    prediction = torch.where(finite, prediction, 0.0)
    count = belief.means.shape[-1]
    held = belief.valid[..., None]
    means = torch.where(held, prediction[..., :count], belief.means)
    spreads = torch.where(held, prediction[..., count:], belief.spreads)
    action = torch.softmax(action_logits, -1)
    return Belief(valid=belief.valid, age=belief.age, means=means, spreads=spreads,
                  action=action), nonfinite
