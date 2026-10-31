"""The factored categorical policy over (power, stick, shoot)."""

import torch
import torch.nn.functional as F

from bnb.contracts import Action
from bnb.contracts.action import NUM_POWER, NUM_SHOOT, NUM_STICK

SIZES = (NUM_POWER, NUM_STICK, NUM_SHOOT)


def split(logits: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """(..., 12) logits into the three components' logits."""
    return logits.split(SIZES, -1)


def sample(logits: torch.Tensor) -> tuple[Action, torch.Tensor, torch.Tensor]:
    """A Gumbel-max sample per component. Returns (action, log-prob, entropy) per ship."""
    picks, logp, entropy = [], 0.0, 0.0
    for part in split(logits):
        u = torch.rand_like(part).clamp_min(1e-20)
        pick = torch.argmax(part - torch.log(-torch.log(u)), -1)
        logs = F.log_softmax(part, -1)
        picks.append(pick.to(torch.int32))
        logp = logp + logs.gather(-1, pick[..., None])[..., 0]
        entropy = entropy - (logs.exp() * logs).sum(-1)
    return Action(*picks), logp, entropy


def evaluate(logits: torch.Tensor, action: Action) -> tuple[torch.Tensor, torch.Tensor]:
    """(log-prob of `action`, entropy) per ship."""
    logp, entropy = 0.0, 0.0
    for part, pick in zip(split(logits), (action.power, action.stick, action.shoot)):
        logs = F.log_softmax(part, -1)
        logp = logp + logs.gather(-1, pick.long()[..., None])[..., 0]
        entropy = entropy - (logs.exp() * logs).sum(-1)
    return logp, entropy
