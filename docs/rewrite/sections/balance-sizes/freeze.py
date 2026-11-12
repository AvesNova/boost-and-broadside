"""The freeze: a card variant (`sizes.py`'s card factors on the ship files) written into the
real-scale ship files of Yongma and Jujak in handle space. The variant card is flown
(`api.measure_handles`), every handle it changes replaces the file's, and the card is
re-solved from the handles (`api.solve_handles`) as the tuning panel does. The doctrines
and the scripted parameters stay as they are.

    PYTHONPATH=src:. uv run --no-sync --active python \
        docs/rewrite/sections/balance-sizes/freeze.py --candidates round5/candidates.json \
        --candidate 2 --note "the size balance's final card"
"""

import argparse
import dataclasses
import json
import math
import pathlib
import sys
import tomllib

import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "game-loop"))

from bench.game_loop import PRESET, base_card, variant_types  # noqa: E402
from bnb.game import api, ships  # noqa: E402
from bnb.game.handling import Handles  # noqa: E402
from bnb.game.handling.errors import Infeasible  # noqa: E402

SET = "real-scale"
NOTE = "the size balance (docs/rewrite/sections/balance-sizes/progress.md): "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--note", default="the size balance's card")
    parser.add_argument("--dry", action="store_true", help="measure and print, write nothing")
    parser.add_argument(
        "--out", default=None, help="write the ship files here, not into the ship set"
    )
    args = parser.parse_args()
    candidate = json.loads((HERE / args.candidates).read_text())[args.candidate]
    base = base_card("cpu")
    types = variant_types(base, [candidate["energy"]], [candidate["turn"]])
    types = dataclasses.replace(
        types, **{f.name: getattr(types, f.name).double() for f in dataclasses.fields(types)}
    )
    rules = api.rules_of(api.PRESETS[PRESET], types=types)
    measured = api.measure_handles(types, rules)
    for row, key in enumerate(("yongma", "jujak")):
        path = ships.directory_of(SET) / f"{key}.toml"
        data = tomllib.loads(path.read_text())
        handles, notes = dict(data["handles"]), dict(data.get("notes", {}))
        for name, old in data["handles"].items():
            new = float(getattr(measured, name)[row])
            if not math.isclose(new, old, rel_tol=2e-3, abs_tol=1e-9):
                handles[name] = new
                notes[name] = f"{notes.get(name, '')} {NOTE}{old:.6g} -> {new:.6g} ({args.note})"
                print(f"{key} {name}: {old:.6g} -> {new:.6g}")
        if args.dry:
            continue
        try:
            solution = api.solve_handles(
                Handles(**{n: torch.tensor([v], dtype=torch.float64) for n, v in handles.items()}),
                rules,
            )
            card = {
                field: float(getattr(solution.types, field)[0])
                for field in api.ship_types(ship_set=SET).__dataclass_fields__
            }
        except Infeasible as error:
            # the solver cannot re-fit these handles: the variant card itself, whose flown
            # handles these are (consistent by construction)
            print(f"  {key}: the solve failed ({error}); the variant card is written")
            card = {
                field: float(getattr(types, field)[row])
                for field in api.ship_types(ship_set=SET).__dataclass_fields__
            }
        for field in ("induced_drag_scale", "max_shield", "bullet_life", "engine_power"):
            print(f"  {key} card {field}: {float(getattr(types, field)[row]):.6g} (variant) "
                  f"{card[field]:.6g} (solved)")  # fmt: skip
        ship = ships.Ship(key=key, name=data["name"], handles=handles, card=card, notes=notes)
        if args.out is None:
            ships.write_ship(ship, SET)
        else:  # the same file, into another directory (the set's files untouched)
            directory_of = ships.directory_of
            ships.directory_of = lambda _: HERE / args.out
            try:
                ships.write_ship(ship, SET)
            finally:
                ships.directory_of = directory_of


if __name__ == "__main__":
    main()
