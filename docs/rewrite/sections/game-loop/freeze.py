"""The freeze: the final card factors (on the ship files, with the base's hit damage x 0.65)
written as the real-scale ship files of Yongma and Jujak (their handles changed and the
card re-solved, as `docs/rewrite/sections/doctrines/ships.py` does), vizia's hit damage x
0.65, and the champion's doctrines as the stored ones (`scripted_doctrines.json`, which the
scripted agent flies by default) and, if it carries them, its scripted parameters as the
real-scale deathmatch set.

    PYTHONPATH=src:. python docs/rewrite/sections/game-loop/freeze.py --card CARD.json \
        --champion NAME
"""

import argparse
import json
import math
import pathlib
import sys
import tomllib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import league  # noqa: E402

from bench.game_loop import BASE_FACTORS  # noqa: E402
from bnb.controllers import scripted_defaults  # noqa: E402
from bnb.controllers.scripted_doctrine import DOCTRINES, named  # noqa: E402
from bnb.controllers.scripted_parameters import from_row  # noqa: E402
from bnb.game import api, ships  # noqa: E402
from bnb.game.handling import Handles  # noqa: E402

SET = "real-scale"
HERE = pathlib.Path(__file__).parent
# a card field's factor or value -> its handle's
HANDLE_OF = {
    "hit_damage": "hit_damage",
    "explosive_share": "explosive_share",
    "head_on_fraction_scale": "head_on_fraction",
    "head_on_angle_scale": "head_on_angle",
    "shield_recharge_rate_scale": "shield_recharge_rate",
    "shield_recharge_delay_scale": "shield_recharge_delay",
    "max_shield": "max_shield",
    "hit_radius": "hit_radius",
}
NOTE = "the game loop's tuning (docs/rewrite/sections/game-loop/progress.md): "


def rewrite(key: str, factors: dict, rules, note: str) -> None:
    """One ship file's handles times the factors (BASE_FACTORS included), re-solved."""
    path = ships.directory_of(SET) / f"{key}.toml"
    data = tomllib.loads(path.read_text())
    handles = dict(data["handles"])
    notes = dict(data.get("notes", {}))
    factors = dict(factors)
    for name, factor in BASE_FACTORS.items():
        factors[name] = (
            factors.get(name, 1.0) * factor
            if not isinstance(factors.get(name), dict)
            else factors[name]
        )
    for field, factor in factors.items():
        if field not in HANDLE_OF:
            raise ValueError(f"no handle for the card field {field}")
        handle = HANDLE_OF[field]
        old = handles[handle]
        handles[handle] = factor["set"] if isinstance(factor, dict) else old * factor
        if not math.isclose(handles[handle], old):
            notes[handle] = (
                f"{notes.get(handle, '')} {NOTE}{old:.6g} -> {handles[handle]:.6g} ({note})".strip()
            )
    solution = api.solve_handles(
        Handles(**{n: torch.tensor([v], dtype=torch.float64) for n, v in handles.items()}),
        rules,
    )
    card = {
        field: float(getattr(solution.types, field)[0])
        for field in api.ship_types(ship_set=SET).__dataclass_fields__
    }
    ships.write_ship(
        ships.Ship(key=key, name=data["name"], handles=handles, card=card, notes=notes), SET
    )
    print(key, {h: round(handles[h], 4) for h in set(HANDLE_OF.values()) if h in handles})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--source", default="the game loop's champion")
    args = parser.parse_args()
    card = json.loads(pathlib.Path(args.card).read_text())
    game = api.PRESETS["real_scale_deathmatch"]
    rules = api.rules_of(game, types=api.ship_types(ship_set=SET))
    rewrite("yongma", card["energy"], rules, "Yongma's final card")
    rewrite("jujak", card["turn"], rules, "Jujak's final card")
    rewrite("vizia", {}, rules, "hit damage x 0.65 restores v5's damage per landed hit")
    champion = {p.name: p for p in league.load()}[args.champion]
    stored = json.loads(DOCTRINES.read_text())
    stored[SET] = {
        "yongma": {"source": args.source, "doctrine": named(champion.energy)},
        "jujak": {"source": args.source, "doctrine": named(champion.turn)},
    }
    DOCTRINES.write_text(json.dumps(stored, indent=1) + "\n")
    if champion.parameters is not None:
        scripted_defaults.write_real_scale(
            "real_scale_deathmatch", from_row(list(champion.parameters)), args.source
        )
    print("doctrines", json.dumps(named(champion.energy)), json.dumps(named(champion.turn)))


if __name__ == "__main__":
    main()
