"""Output heads (spec 7): groups of MLP heads over one token, the next-state logits over
their baseline, the critic's values, and the shot trajectory head queried at bullet ages."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import codes, critic
from .layers import norm

FIRST_GAIN = math.sqrt(2.0)
LAST_GAIN = 0.01


def _orthogonal(weight: torch.Tensor, gain: float) -> None:
    with torch.no_grad():
        nn.init.orthogonal_(weight, gain)


class HeadGroup(nn.Module):
    """MLP heads reading the same token, each last(gelu(norm(first x))). Their first
    projections are one matrix, so the group costs one matrix product to start.

    Initialisation (spec 7): each head's block of `first` orthogonal with gain sqrt 2, each
    `last` orthogonal with gain 0.01, biases zero.
    """

    def __init__(self, inputs: int, hidden: int, outputs: tuple[int, ...]) -> None:
        super().__init__()
        self.hidden = hidden
        self.first = nn.Linear(inputs, hidden * len(outputs))
        self.norms = nn.ModuleList(norm(hidden) for _ in outputs)
        self.lasts = nn.ModuleList(nn.Linear(hidden, size) for size in outputs)
        for block in self.first.weight.split(hidden):
            _orthogonal(block, FIRST_GAIN)
        nn.init.zeros_(self.first.bias)
        for last in self.lasts:
            _orthogonal(last.weight, LAST_GAIN)
            nn.init.zeros_(last.bias)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.first(inputs).split(self.hidden, -1)
        return [last(F.gelu(normed(part)))
                for part, normed, last in zip(hidden, self.norms, self.lasts, strict=True)]


def next_state_logits(code: torch.Tensor, change: torch.Tensor) -> torch.Tensor:
    """The next state's code logits as a change from the current code. The head's last
    weight starts at zero, so the forecast starts as "nothing changes"."""
    return codes.baseline(code) + change


def critic_values(local_logits: torch.Tensor, outcome_logits: torch.Tensor, bins: torch.Tensor,
                  local_levels: tuple[int, ...], outcome_level: int) -> torch.Tensor:
    """(..., S, K) values in the configured level order.

    local_logits (..., S, K - 1, bins): per-ship levels, each the expectation over the bins;
    outcome_logits (..., 4): the outcome level P(win) - P(loss), the same for every ship.
    """
    local = critic.expectation(local_logits, bins).unbind(-1)
    outcome = critic.outcome_values(torch.softmax(outcome_logits, -1))[..., None]
    outcome = outcome.expand(local[0].shape)  # K >= 2: at least one per-ship level
    columns = {level: column for level, column in zip(local_levels, local)}
    columns[outcome_level] = outcome
    # Stacked from slices (no index tensor), so eager CUDA-graph capture needs no host copy.
    return torch.stack([columns[level] for level in range(len(columns))], -1)


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
        for layer, gain in ((self.latent, FIRST_GAIN), (self.age, FIRST_GAIN),
                            (self.middle, FIRST_GAIN), (self.last, LAST_GAIN)):
            _orthogonal(layer.weight, gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def query(self, latent: torch.Tensor, age_features: torch.Tensor) -> torch.Tensor:
        """latent (..., D) from `self.latent`; age_features (..., ages, F) broadcasting
        against latent[..., None, :]. Returns (..., ages, outputs)."""
        hidden = F.gelu(self.first_norm(latent[..., None, :] + self.age(age_features)))
        return self.last(F.gelu(self.middle_norm(self.middle(hidden))))
