"""The figure sets, by name: what `bnb figures` can draw from a capture.

Each set's renderer takes the artifact (`bnb.figures.capture.load`) and an output
directory and returns (the PNGs it wrote, its numbers). The docs page
`docs/guides/figures.md` describes every set; a docs test keeps the two in step.
"""

from collections.abc import Callable

from . import belief, density, next_state, scenarios, shots

FIGURE_SETS: dict[str, Callable] = {
    "shots": shots.render,
    "scenarios": scenarios.render,
    "belief": belief.render,
    "next_state": next_state.render,
    "density": density.render,
}
