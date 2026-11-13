"""The league-only fast chaser (game v12): vizia with a faster cruise (top coasting speed)
and a stronger boost, solved under the real-scale rules and written beside this script as
chaser.toml, outside the set (so it is no playable type and the set's order is unchanged).

    uv run --no-sync python docs/rewrite/sections/lethality/chaser.py
"""

import dataclasses
import pathlib

import torch

from bnb.contracts import GameConfig
from bnb.game import ships
from bnb.game.api import rules_of
from bnb.game.handling import NAMES, Handles, solve
from bnb.game.ship_types import FIELDS

HERE = pathlib.Path(__file__).parent
FASTER = dict(cruise_speed=1.15, dive_acceleration=1.5)  # vizia's, times these

rules = rules_of(GameConfig(1, 1, 7800.0, ship_set="real-scale"))
vizia = ships.read(ships.directory_of("real-scale") / "vizia.toml", "vizia")
values = {n: vizia.handles[n] * FASTER.get(n, 1.0) for n in NAMES}
handles = Handles(**{n: torch.tensor([v], dtype=torch.float64) for n, v in values.items()})
solution = solve(handles, rules)
print("worst error", float(solution.error.amax()), "converged", solution.converged.tolist())
card = {name: float(getattr(solution.types, name)[0]) for name in FIELDS}
notes = {n: f"vizia's x {f:g} (league-only fast chaser)" for n, f in FASTER.items()}
chaser = dataclasses.replace(
    vizia, key="chaser", name="Chaser", handles=values, card=card, notes=notes
)
text = (
    "# Chaser: the league-only fast chaser (docs/rewrite/sections/lethality/chaser.py), not a\n"
    "# playable type; vizia with a faster cruise and a stronger boost.\n"
    f"name = {chaser.name!r}\n\n[handles]\n{ships._table(chaser.handles)}\n[card]\n"
    f"{ships._table(chaser.card)}\n[notes]\n{ships._table(chaser.notes)}"
)
(HERE / "chaser.toml").write_text(text)
print(
    {
        k: round(card[k], 4)
        for k in ("engine_power", "static_thrust", "zero_lift_drag", "hit_damage")
    }
)
