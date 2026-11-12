"""The pool's entries, their sampling weights and the snapshots' retention (league spec 1, 3
and 5).

Entries: frozen snapshots of the live policy (checkpoints), the uniform average and the
exponential moving average of its weights, and the full-strength scripted agent (no weaker
rung and no random player: the author, 2026-11-12). Self-play is not an entry: it is a fixed
share of each mode's batch.
"""

import dataclasses

import torch

from bnb.runtime.lanes import EXPONENTIAL_AVERAGE, UNIFORM_AVERAGE
from bnb.runtime.opponents import SCRIPTED_PLAYER, LeagueConfig

from .gauge import KINDS

AVERAGES = (UNIFORM_AVERAGE, EXPONENTIAL_AVERAGE)
RECENT_SNAPSHOTS = 8  # the newest snapshots retention keeps
HISTORY_FROM = 3  # then the snapshot nearest each age 2^k snapshots back, k >= HISTORY_FROM


@dataclasses.dataclass(frozen=True)
class Entry:
    """One pool entry. kind: "checkpoint", "average" or "scripted" (the full-strength
    scripted agent, the pool's one scripted entry); steps: a checkpoint's environment steps
    at its snapshot; ordinal: a checkpoint's place in the order of snapshots (0, 1, 2,
    ...)."""

    name: str
    kind: str
    steps: int | None = None
    ordinal: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"entry kind {self.kind!r} is not one of {KINDS}")
        if (self.kind == "scripted") != (self.name == SCRIPTED_PLAYER):
            raise ValueError(f"{self.name}: the scripted entry is {SCRIPTED_PLAYER!r}")

    @property
    def network(self) -> bool:
        """Whether the entry is a network (a checkpoint or an average), loaded into a slot."""
        return self.kind != "scripted"


SCRIPTED_ENTRY = Entry(SCRIPTED_PLAYER, "scripted")  # the pool's one scripted entry


def snapshot_name(steps: int) -> str:
    """A checkpoint entry's name: `snapshot_<environment steps>`."""
    return f"snapshot_{steps}"


def sampling_weights(win_probabilities: torch.Tensor, floor: float) -> torch.Tensor:
    """(K,) float64 summing to 1: each entry's `p (1 - p)` on the predicted probability `p`
    that the live policy beats it, scaled to `1 - floor` of the pool's weight, plus `floor`
    spread evenly over the entries (all of it when every `p (1 - p)` is zero)."""
    probability = win_probabilities.to(torch.float64)
    weight = probability * (1.0 - probability)
    count = weight.shape[0]
    total = float(weight.sum())
    if total == 0.0:
        return torch.full((count,), 1.0 / count, dtype=torch.float64)
    return (1.0 - floor) * weight / total + floor / count


def retained(ordinals: list[int] | set[int]) -> set[int]:
    """The snapshots retention keeps, by ordinal: the newest RECENT_SNAPSHOTS, plus of the
    older ones the snapshot nearest each age 2^k snapshots back for k >= HISTORY_FROM (an
    equal distance keeps the older). Ages are counted from the newest snapshot."""
    ordinals = set(ordinals)
    if not ordinals:
        return set()
    newest = max(ordinals)
    keep = {ordinal for ordinal in ordinals if newest - ordinal < RECENT_SNAPSHOTS}
    older = sorted(newest - ordinal for ordinal in ordinals if newest - ordinal >= RECENT_SNAPSHOTS)
    if older:
        power = HISTORY_FROM
        while True:
            target = 1 << power
            nearest = min(older, key=lambda age: (abs(age - target), -age))
            keep.add(newest - nearest)
            if target >= older[-1]:  # every larger age is nearest the oldest, kept now
                break
            power += 1
    return keep


@torch.no_grad()
def average_into(
    uniform: dict[str, torch.Tensor],
    exponential: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
    count: int,
    decay: float,
) -> None:
    """Fold one update's weights into the averages in place: the uniform average of `count`
    updates' weights (this one included) and the exponential moving average with `decay`
    per update. Integer buffers are copied, not averaged."""
    for name, value in weights.items():
        if not value.is_floating_point():
            uniform[name].copy_(value)
            exponential[name].copy_(value)
            continue
        uniform[name].add_(value - uniform[name], alpha=1.0 / count)
        exponential[name].mul_(decay).add_(value, alpha=1.0 - decay)


class Averages:
    """The uniform average and the exponential moving average of the live weights (league
    spec 3), each a state dict by name (`weights`, AVERAGES), folded in after every update
    (`fold`). The uniform average counts the updates from `uniform_from` environment steps
    on (`count`; before them it is the live weights: `LeagueConfig.uniform_start` of the
    run's warm-up); the exponential one
    decays by `LeagueConfig.exponential_decay` per update throughout. The league keeps them
    as pool entries, behaviour cloning as its product (`bnb.league.lanes.LaneHooks`).

    weights: the live model's state dict, which both start from; a resume's `count` and
    `tensors` (`tensors()` of the saving run) restore them, on the weights' devices."""

    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        config: LeagueConfig,
        uniform_from: int,
        count: int = 0,
        tensors: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.config, self.uniform_from = config, uniform_from
        self.count = count  # the updates in the uniform average
        if tensors is None:
            self.weights = {
                name: {key: value.detach().clone() for key, value in weights.items()}
                for name in AVERAGES
            }
        else:
            self.weights = {
                name: {
                    key: tensors[f"{name}.{key}"].to(value.device) for key, value in weights.items()
                }
                for name in AVERAGES
            }

    @property
    def uniform_started(self) -> bool:
        """Whether the uniform average holds an update past the warm-up (before, it is the
        live weights)."""
        return self.count > 0

    def fold(self, weights: dict[str, torch.Tensor], environment_steps: int) -> None:
        """Fold an update's live weights in, `environment_steps` the run's total after it."""
        if environment_steps >= self.uniform_from:
            self.count += 1
        average_into(
            self.weights[UNIFORM_AVERAGE],
            self.weights[EXPONENTIAL_AVERAGE],
            weights,
            max(self.count, 1),  # before uniform_from, the live weights
            self.config.exponential_decay,
        )

    def tensors(self) -> dict[str, torch.Tensor]:
        """Both averages' weights by `<average>.<key>`, for a training checkpoint."""
        return {
            f"{name}.{key}": value
            for name, weights in self.weights.items()
            for key, value in weights.items()
        }
