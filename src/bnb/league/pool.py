"""The pool's entries, their sampling weights and the snapshots' retention (league spec 1, 3
and 5).

Entries: frozen snapshots of the live policy (checkpoints), the uniform average and the
exponential moving average of its weights, and the stationary ladder (random, the rungs and
the scripted agent). Self-play is not an entry: it is a fixed share of each mode's batch.
"""

import dataclasses

import torch

from .gauge import KINDS

UNIFORM_AVERAGE = "uniform_average"
EXPONENTIAL_AVERAGE = "exponential_average"
AVERAGES = (UNIFORM_AVERAGE, EXPONENTIAL_AVERAGE)
RECENT = 8  # the newest snapshots retention keeps
HISTORY_FROM = 3  # then the snapshot nearest each age 2^k snapshots back, k >= HISTORY_FROM


@dataclasses.dataclass(frozen=True)
class Entry:
    """One pool entry. kind: "checkpoint", "average" or "ladder"; scripted_probability: a
    ladder entry's `p` (0 random, 1 the scripted agent), None otherwise; steps: a
    checkpoint's environment steps at its snapshot; ordinal: a checkpoint's place in the
    order of snapshots (0, 1, 2, ...)."""

    name: str
    kind: str
    scripted_probability: float | None = None
    steps: int | None = None
    ordinal: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"entry kind {self.kind!r} is not one of {KINDS}")
        if (self.kind == "ladder") != (self.scripted_probability is not None):
            raise ValueError(f"{self.name}: only a ladder entry has a scripted probability")

    @property
    def network(self) -> bool:
        """Whether the entry is a network (a checkpoint or an average), loaded into a slot."""
        return self.kind != "ladder"


def ladder_name(scripted_probability: float) -> str:
    """ "random" (0), "scripted" (1), or "rung_<p>"."""
    if scripted_probability == 0.0:
        return "random"
    if scripted_probability == 1.0:
        return "scripted"
    return f"rung_{scripted_probability:g}"


def ladder_entries(probabilities: tuple[float, ...]) -> list[Entry]:
    """The stationary ladder's entries, one per scripted probability."""
    return [Entry(ladder_name(p), "ladder", scripted_probability=p) for p in probabilities]


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
    """The snapshots retention keeps, by ordinal: the newest RECENT, plus of the older ones
    the snapshot nearest each age 2^k snapshots back for k >= HISTORY_FROM (an equal
    distance keeps the older). Ages are counted from the newest snapshot."""
    ordinals = set(ordinals)
    if not ordinals:
        return set()
    newest = max(ordinals)
    keep = {ordinal for ordinal in ordinals if newest - ordinal < RECENT}
    older = sorted(newest - ordinal for ordinal in ordinals if newest - ordinal >= RECENT)
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
