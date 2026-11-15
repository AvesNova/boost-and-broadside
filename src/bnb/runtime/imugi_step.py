"""Imugi's step as the play shell runs it: one decision for one team's view,
uncompiled or built ahead of time (`bnb.runtime.built`).

Uncompiled, the step is a few hundred small torch operations, about 4 ms a side on one CPU
thread whatever the fight; built, about 0.1 ms. The ship characteristics are an input, so a
tuned card plays on the same library; it is keyed (`library_path`) by the code it holds,
the parameters, the doctrines it flies (a stored-doctrine file, `imugi_doctrine.stored`:
its contents and the doctrine table's width, which the library holds as constants) and the
shapes of its inputs (the fleet, the fields, the zones). Its draws are keyed by the
decision key it is given (`Controller.step`), so built or not it draws the same.
"""

import hashlib
import pathlib

import torch
from torch.utils import _pytree

from bnb import cache
from bnb.contracts import Obs
from bnb.controllers import imugi, imugi_doctrine
from bnb.controllers.imugi_defaults import MODE_DEFAULTS
from bnb.controllers.imugi_memory import ImugiCarry
from bnb.controllers.imugi_parameters import ImugiParameters, ModeParameters
from bnb.game.handling.characteristics import Characteristics

_STEP_CODE = ("controllers", "contracts", "game", "runtime/imugi_step.py", "runtime/built.py")


class ImugiStep:
    """One decision of Imugi (`bnb.controllers.imugi.Imugi`, temperature
    1, memory on) for a team of `ships_per_team` ships: the team's (S, 3) int32 actions and
    the carry after it. doctrines: the stored-doctrine file it flies (`imugi_doctrine.
    for_card` of the card; the default, the training's stored ones), read when it is made."""

    def __init__(
        self,
        ships_per_team: int,
        parameters: ImugiParameters | ModeParameters = MODE_DEFAULTS,
        doctrines: pathlib.Path = imugi_doctrine.DOCTRINES,
    ) -> None:
        self.ships_per_team, self.parameters = ships_per_team, parameters
        self.doctrines = doctrines
        # read here, kept as plain floats: the traced step reads no files
        self._rows = imugi_doctrine.rows_by_kind(doctrines)

    def table(self, characteristics: Characteristics) -> torch.Tensor | None:
        """(T, WIDTH) the doctrines for the card's types, on its device: the file's set with
        the card's type count and scale, chosen by the card's gravity in the traced step (a
        scale without a set flies the default doctrine, off); None when the file has no set
        with the card's type count."""
        count = characteristics.gun_range.shape[-1]
        real, standard = self._rows.get((count, True)), self._rows.get((count, False))
        if real is None and standard is None:
            return None
        device = characteristics.gun_range.device
        off = imugi_doctrine.off_rows(count)
        real_table, standard_table = (
            torch.tensor(rows or off, dtype=torch.float32, device=device)
            for rows in (real, standard)
        )
        chosen = imugi_doctrine.real_scale_card(characteristics.gravity.amax())
        return torch.where(chosen, real_table, standard_table)

    def __call__(
        self,
        carry: ImugiCarry,
        view: Obs,
        reset: torch.Tensor,
        decision_key: torch.Tensor,
        characteristics: Characteristics,
    ) -> tuple[torch.Tensor, ImugiCarry]:
        controller = imugi.Imugi(
            characteristics,
            self.ships_per_team,
            self.parameters,
            doctrines=self.table(characteristics),
        )
        action, after, _ = controller.step(carry, view, reset, decision_key)
        return action.packed()[0], after


def library_path(step: ImugiStep, arguments: tuple) -> pathlib.Path:
    """Where the built step for these arguments' shapes is kept (`bnb.cache`)."""
    shapes = tuple(
        (tuple(leaf.shape), leaf.dtype)
        for leaf in _pytree.tree_leaves(arguments)
        if isinstance(leaf, torch.Tensor)
    )
    doctrines = hashlib.sha256(step.doctrines.read_bytes()).hexdigest()
    parts = (step.ships_per_team, step.parameters, imugi_doctrine.WIDTH, doctrines, shapes)
    return cache.path("imugi_steps", *parts, packages=_STEP_CODE, suffix=".pt2")
