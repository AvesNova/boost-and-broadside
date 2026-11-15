"""Output heads (spec 7): groups of MLP heads over one token, the next-state logits over
their baseline, the critic's values, the shot trajectory head queried at bullet ages, and
the shot damage head over pairs of ship rows."""

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
    levels: tuple[str, ...],
    outcome_level: str | None,
) -> torch.Tensor:
    """(..., S, levels) values in the configured level order.

    local_logits (..., S, per-ship levels, bins): the per-ship levels in order
    (`critic.local_levels`), each the expectation over the bins; outcome_logits (..., 4):
    the outcome level's parts (`critic.outcome_part`), the same for every ship. Without an
    outcome level (outcome_logits and outcome_level None) every level is per-ship.
    """
    if outcome_level is None:
        return critic.expectation(local_logits, bins)
    local = iter(critic.expectation(local_logits, bins).unbind(-1))
    probabilities = torch.softmax(outcome_logits, -1)
    shape = local_logits.shape[:-2]  # (..., S)
    columns = []
    for name in levels:
        part = critic.outcome_part(name, outcome_level)
        if part is None:
            columns.append(next(local))
        else:
            columns.append(critic.outcome_part_values(probabilities, part)[..., None].expand(shape))
    # Stacked from slices (no index tensor), so eager CUDA-graph capture needs no host copy.
    return torch.stack(columns, -1)


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


class ShotDamageHead(nn.Module):
    """Pair logits (..., S, N, outputs) (arm A22, `YemongConfig.shot_head` "damage"): for
    each of the first S rows (the shooters, the own ships) and every row (the targets), the
    code of the damage a shot fired now would deal. latent(x) is computed once per row,
    its first half the shooter's part and its second the target's; a pair's logits are
    last(gelu(norm(shooter part + target part))), one hidden layer as a `HeadGroup` head's.
    Initialised as the geometry head: each width-row block of the latent orthogonal (gain
    sqrt 2), the last orthogonal (gain 0.01), every bias zero."""

    def __init__(self, width: int, outputs: int) -> None:
        super().__init__()
        self.width = width
        self.latent = nn.Linear(width, 2 * width)
        self.first_norm = norm(width)
        self.last = nn.Linear(width, outputs)

    def initialise_parameters(self, generator) -> None:
        """The class docstring's initialisation, each block its own draw."""
        for index, block in enumerate(self.latent.weight.split(self.width)):
            orthogonal(block, FIRST_GAIN, generator(f"latent.weight[{index}]"))
        nn.init.zeros_(self.latent.bias)
        orthogonal(self.last.weight, LAST_GAIN, generator("last.weight"))
        nn.init.zeros_(self.last.bias)

    def query(self, latents: torch.Tensor, shooters: int) -> torch.Tensor:
        """latents (..., N, 2 D) from `self.latent`; the first `shooters` rows shoot. Returns
        (..., S, N, outputs), a shooter's own row included (no loss reads it)."""
        shooter = latents[..., :shooters, None, : self.width]
        target = latents[..., None, :, self.width :]
        return self.last(F.gelu(self.first_norm(shooter + target)))
