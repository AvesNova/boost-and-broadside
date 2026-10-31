"""Output heads (spec 7): MLP heads, the next-state head over its baseline, the critic's
values, and the shot trajectory head queried at bullet ages."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import codes, critic
from .layers import norm

FIRST_GAIN = math.sqrt(2.0)
LAST_GAIN = 0.01


def _orthogonal(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class MLPHead(nn.Module):
    """last(gelu(norm(first x))): first orthogonal with gain sqrt 2, last with gain 0.01,
    biases zero."""

    def __init__(self, inputs: int, hidden: int, outputs: int) -> None:
        super().__init__()
        self.first = nn.Linear(inputs, hidden)
        self.norm = norm(hidden)
        self.last = nn.Linear(hidden, outputs)
        _orthogonal(self.first, FIRST_GAIN)
        _orthogonal(self.last, LAST_GAIN)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.last(F.gelu(self.norm(self.first(inputs))))


class NextStateHead(MLPHead):
    """The next state's code logits as a change from the current code: it starts as
    "nothing changes" (its last weight is zero)."""

    def __init__(self, inputs: int, hidden: int, outputs: int) -> None:
        super().__init__(inputs, hidden, outputs)
        nn.init.zeros_(self.last.weight)

    def forward(self, inputs: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        return codes.baseline(code) + super().forward(inputs)


def critic_values(local_logits: torch.Tensor, outcome_logits: torch.Tensor, bins: torch.Tensor,
                  local_levels: tuple[int, ...], outcome_level: int) -> torch.Tensor:
    """(..., S, K) values in the configured level order.

    local_logits (..., S, K - 1, bins): per-ship levels, each the expectation over the bins;
    outcome_logits (..., 4): the outcome level P(win) - P(loss), the same for every ship.
    """
    local = critic.expectation(local_logits, bins)
    outcome = critic.outcome_values(torch.softmax(outcome_logits, -1))
    stacked = torch.cat([local, outcome[..., None, None].expand(local.shape[:-1] + (1,))], -1)
    order = list(local_levels) + [outcome_level]
    return stacked[..., [order.index(level) for level in range(len(order))]]


class ShotTrajectoryHead(nn.Module):
    """Bullet snapshot code logits at any ages: the latent part is computed once per
    ship-step, and the age part is added per query, so ages are queried in parallel.

    h = latent(x) + age(phi(a)), then norm, gelu, middle, norm, gelu, last.
    """

    def __init__(self, width: int, age_features: int, outputs: int) -> None:
        super().__init__()
        self.latent = nn.Linear(width, width)
        self.age = nn.Linear(age_features, width, bias=False)
        self.first_norm = norm(width)
        self.middle = nn.Linear(width, width)
        self.middle_norm = norm(width)
        self.last = nn.Linear(width, outputs)
        _orthogonal(self.latent, FIRST_GAIN)
        _orthogonal(self.age, FIRST_GAIN)
        _orthogonal(self.middle, FIRST_GAIN)
        _orthogonal(self.last, LAST_GAIN)

    def query(self, latent: torch.Tensor, age_features: torch.Tensor) -> torch.Tensor:
        """latent (..., D) from `self.latent`; age_features (..., ages, F) broadcasting
        against latent[..., None, :]. Returns (..., ages, outputs)."""
        hidden = F.gelu(self.first_norm(latent[..., None, :] + self.age(age_features)))
        return self.last(F.gelu(self.middle_norm(self.middle(hidden))))
