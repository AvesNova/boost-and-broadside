"""Write the chosen re-trim (game v25) into the real-scale ship files: the v24 files' handles
with the trim's max_shield factors (relative to G's, as retrim.py), the card fields re-solved
from the handles (`bnb.game.api.solve_handles`) and checked against the measured entry in
trims.json before writing.

    PYTHONPATH=$PWD/src:$PWD uv run --no-sync python docs/rewrite/sections/retrim-v25/apply.py Y0.42

Run it once, on game v24's files: the factors multiply the handles as they are.
"""

import json
import pathlib
import sys
import tomllib

import torch

from bnb.game import api, ships
from bnb.game.handling import Handles

HERE = pathlib.Path(__file__).parent
SET = "real-scale"
G_YONGMA, G_JUJAK = 0.46, 1.1586  # the v24 files' max_shield factors (of v23's)
TRIMS = {"Y0.42": (0.42, 1.1586), "Y0.38": (0.38, 1.1586), "Y0.42-J1.22": (0.42, 1.22)}
NOTE = "game v25's re-trim {name} for the committed-turn Imugi (docs/rewrite/sections/retrim-v25)"


def main(name: str) -> None:
    """Apply the trim `name` to the ship files."""
    measured = next(e for e in json.loads((HERE / "trims.json").read_text()) if e["name"] == name)
    yongma, jujak = TRIMS[name]
    factors = {"energy": ("yongma", yongma / G_YONGMA), "turn": ("jujak", jujak / G_JUJAK)}
    rules = api.rules_of(api.PRESETS["real_scale_deathmatch"], types=api.ship_types(ship_set=SET))
    for side, (key, factor) in factors.items():
        data = tomllib.loads((ships.directory_of(SET) / f"{key}.toml").read_text())
        handles, notes = dict(data["handles"]), dict(data.get("notes", {}))
        if abs(factor - 1.0) > 1e-9:
            old = handles["max_shield"]
            handles["max_shield"] = old * factor
            note = NOTE.format(name=name)
            notes["max_shield"] = (
                f"{notes.get('max_shield', '')} {note}: {old:.6g} -> {handles['max_shield']:.6g}"
            ).strip()
            print(f"{key} max_shield: {old:.6g} -> {handles['max_shield']:.6g}")
        solution = api.solve_handles(
            Handles(**{n: torch.tensor([v], dtype=torch.float64) for n, v in handles.items()}),
            rules,
        )
        card = dict(data["card"])
        for field in card:
            value = float(getattr(solution.types, field)[0])
            if abs(value - card[field]) > 1e-5 * max(abs(card[field]), 1e-9):
                print(f"  {key} card {field}: {card[field]:.6g} -> {value:.6g}")
            card[field] = value
        for field, entry in measured["card"][side].items():  # the card the runs measured
            if abs(card[field] - entry["set"]) > 1e-5 * max(abs(entry["set"]), 1e-9):
                raise SystemExit(f"{key} {field}: solved {card[field]}, measured {entry['set']}")
        ships.write_ship(ships.Ship(key, data["name"], handles, card, notes), SET)
    print("wrote", ships.directory_of(SET))


if __name__ == "__main__":
    main(sys.argv[1])
