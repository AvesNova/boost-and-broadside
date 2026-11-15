"""The figure sets, by name: what `bnb figures` can draw from a capture.

Each set's renderer takes the artifact (`bnb.figures.capture.load`) and an output
directory and returns (the PNGs it wrote, its numbers). The docs page
`docs/guides/figures.md` describes every set; a docs test keeps the two in step. A set of a
head the policy does not build has nothing to draw (`drawable`): its record is not in the
capture.
"""

from collections.abc import Callable

from . import belief, critic, density, enemy_action, next_state, scenarios, shots

FIGURE_SETS: dict[str, Callable] = {
    "shots": shots.render,
    "scenarios": scenarios.render,
    "belief": belief.render,
    "next_state": next_state.render,
    "density": density.render,
    "enemy_action": enemy_action.render,
    "critic": critic.render,
}
# The record each head's set reads, by its path in the artifact: absent when the policy
# does not build the head (`bnb.models.yemong.model.Yemong.auxiliary_heads`).
HEAD_RECORDS: dict[str, tuple[str, ...]] = {
    "shots": ("rollout", "shots"),
    "scenarios": ("scenarios",),
    "next_state": ("rollout", "next_state"),
    "density": ("rollout", "density"),
    "enemy_action": ("rollout", "enemy_action"),
}


def drawable(name: str, artifact: dict) -> bool:
    """The capture holds what the set reads (a head's set: the policy builds the head)."""
    part = artifact
    for key in HEAD_RECORDS.get(name, ()):
        if not isinstance(part, dict) or key not in part:
            return False
        part = part[key]
    return True
