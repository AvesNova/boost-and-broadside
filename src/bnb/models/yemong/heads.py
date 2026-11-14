"""Output heads (spec 7): groups of MLP heads over one token, the next-state logits over
their baseline, the critic's values, and the shot trajectory head queried at bullet ages."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bnb.models.initialise import orthogonal

from . import codes, critic, kernels
from .kernels import heads as head_kernel
from .layers import norm

FIRST_GAIN = math.sqrt(2.0)
LAST_GAIN = 0.01


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

    def initialise_parameters(self, generator) -> None:
        """Spec 7's initialisation (the class docstring), each head's draws its own."""
        for index, block in enumerate(self.first.weight.split(self.hidden)):
            orthogonal(block, FIRST_GAIN, generator(f"first.weight[{index}]"))
        nn.init.zeros_(self.first.bias)
        for index, last in enumerate(self.lasts):
            orthogonal(last.weight, LAST_GAIN, generator(f"lasts.{index}.weight"))
            nn.init.zeros_(last.bias)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """One output (..., size) per head, in the order of the constructor's outputs."""
        if kernels.switches.heads and head_kernel.applies(inputs, self.hidden):
            blocks = zip(
                self.first.weight.split(self.hidden),
                self.first.bias.split(self.hidden),
                self.norms,
                self.lasts,
                strict=True,
            )
            return [
                head_kernel.head(inputs, (weight, bias), normed, last)
                for weight, bias, normed, last in blocks
            ]
        hidden = self.first(inputs).split(self.hidden, -1)
        return [
            last(F.gelu(normed(part)))
            for part, normed, last in zip(hidden, self.norms, self.lasts, strict=True)
        ]


def next_state_logits(code: torch.Tensor, change: torch.Tensor) -> torch.Tensor:
    """The next state's code logits as a change from the current code. The head's last
    weight starts at zero, so the forecast starts as "nothing changes"."""
    return codes.baseline(code) + change


def critic_values(
    local_logits: torch.Tensor,
    outcome_logits: torch.Tensor | None,
    bins: torch.Tensor,
    local_levels: tuple[int, ...],
    outcome_level: int | None,
) -> torch.Tensor:
    """(..., S, levels) values in the configured level order.

    local_logits (..., S, per-ship levels, bins): per-ship levels, each the expectation over
    the bins; outcome_logits (..., 4): the outcome level P(win) - P(loss), the same for every
    ship. Without an outcome level (outcome_logits and outcome_level None) every level is
    per-ship, in order.
    """
    if outcome_level is None:
        return critic.expectation(local_logits, bins)
    local = critic.expectation(local_logits, bins).unbind(-1)
    outcome = critic.outcome_values(torch.softmax(outcome_logits, -1))[..., None]
    outcome = outcome.expand(local[0].shape)  # levels >= 2: at least one per-ship level
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

    def initialise_parameters(self, generator) -> None:
        """Every weight orthogonal (gain sqrt 2, the last's 0.01), every bias zero."""
        for name, gain in (
            ("latent", FIRST_GAIN),
            ("age", FIRST_GAIN),
            ("middle", FIRST_GAIN),
            ("last", LAST_GAIN),
        ):
            layer = getattr(self, name)
            orthogonal(layer.weight, gain, generator(f"{name}.weight"))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def query(
        self, latent: torch.Tensor, age_features: torch.Tensor, ages: torch.Tensor
    ) -> torch.Tensor:
        """latent (..., D) from `self.latent`; age_features (T, F), a table of the ages;
        ages (Q,) or (..., Q), int64 indices into it, the same for every row or each row's
        own. Returns (..., Q, outputs). The staged head kernel makes the whole query in one."""
        age = self.age(age_features)  # (T, D)
        if kernels.switches.heads and head_kernel.staged_applies(
            latent, age, self.middle.out_features
        ):
            return head_kernel.staged_head(
                latent, age, ages, self.first_norm, self.middle, self.middle_norm, self.last
            )
        hidden = F.gelu(self.first_norm(latent[..., None, :] + age[ages]))
        if kernels.switches.heads and head_kernel.applies(hidden, self.middle.out_features):
            return head_kernel.head(hidden, self.middle, self.middle_norm, self.last)
        return self.last(F.gelu(self.middle_norm(self.middle(hidden))))
