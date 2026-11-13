"""The scripted agent's step as the play shell runs it: one decision for one team's view,
uncompiled or built ahead of time (`bnb.runtime.built`).

Uncompiled, the step is a few hundred small torch operations, about 4 ms a side on one CPU
thread whatever the fight; built, about 0.1 ms. The ship characteristics are an input, so a
tuned card plays on the same library; it is keyed (`library_path`) by the code it holds,
the parameters, the doctrines it flies (a stored-doctrine file, `scripted_doctrine.stored`:
its contents and the doctrine table's width, which the library holds as constants) and the
shapes of its inputs (the fleet, the fields, the zones). Its draws
come from torch's default generator, built or not, so a side that swaps its own generator
in gets the same draws from either.
"""

import hashlib
import pathlib

import torch
from torch.utils import _pytree

from bnb import cache
from bnb.contracts import Obs
from bnb.controllers import scripted, scripted_doctrine
from bnb.controllers.scripted_defaults import MODE_DEFAULTS
from bnb.controllers.scripted_memory import ScriptedCarry
from bnb.controllers.scripted_parameters import ModeParameters, ScriptedParameters
from bnb.game.handling.characteristics import Characteristics

_STEP_CODE = ("controllers", "contracts", "game", "runtime/scripted_step.py", "runtime/built.py")


class ScriptedStep:
    """One decision of the scripted agent (`bnb.controllers.scripted.Scripted`, temperature
    1, memory on) for a team of `ships_per_team` ships: the team's (S, 3) int32 actions and
    the carry after it. doctrines: the stored-doctrine file it flies (`scripted_doctrine.
    for_card` of the card; the default, the training's stored ones), read when it is made."""

    def __init__(
        self,
        ships_per_team: int,
        parameters: ScriptedParameters | ModeParameters = MODE_DEFAULTS,
        doctrines: pathlib.Path = scripted_doctrine.DOCTRINES,
    ) -> None:
        self.ships_per_team, self.parameters = ships_per_team, parameters
        self.doctrines = doctrines
        # read here, kept as plain floats: the traced step reads no files
        self._rows = scripted_doctrine.rows_by_count(doctrines)

    def table(self, characteristics: Characteristics) -> torch.Tensor | None:
        """(T, WIDTH) the doctrines for the card's types, on its device (None: the file has
        none for its type count)."""
        rows = self._rows.get(characteristics.gun_range.shape[-1])
        if rows is None:
            return None
        return torch.tensor(rows, dtype=torch.float32, device=characteristics.gun_range.device)

    def __call__(
        self,
        carry: ScriptedCarry,
        view: Obs,
        reset: torch.Tensor,
        characteristics: Characteristics,
    ) -> tuple[torch.Tensor, ScriptedCarry]:
        controller = scripted.Scripted(
            characteristics,
            self.ships_per_team,
            self.parameters,
            doctrines=self.table(characteristics),
        )
        action, after, _ = controller.step(carry, view, reset)
        return action.packed()[0], after


def library_path(step: ScriptedStep, arguments: tuple) -> pathlib.Path:
    """Where the built step for these arguments' shapes is kept (`bnb.cache`)."""
    shapes = tuple(
        (tuple(leaf.shape), leaf.dtype)
        for leaf in _pytree.tree_leaves(arguments)
        if isinstance(leaf, torch.Tensor)
    )
    doctrines = hashlib.sha256(step.doctrines.read_bytes()).hexdigest()
    parts = (step.ships_per_team, step.parameters, scripted_doctrine.WIDTH, doctrines, shapes)
    return cache.path("scripted_steps", *parts, packages=_STEP_CODE, suffix=".pt2")
