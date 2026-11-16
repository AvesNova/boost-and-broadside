"""Write the chosen balance card, Y0.46 with a head-on damage fraction of 0.05 (the author,
2026-11-16), into the real-scale ship files: option G's handle factors on game v23's files
with Yongma's max_shield factor 0.46 (tune/g-v23, docs/rewrite/sections/g-v23/ there:
solve.py, trims.py "Y0.46", headon.json "Y0.46-H0.05"), and head_on_fraction 0.05 for both
ships. The card fields are re-solved from the handles (`bnb.game.api.solve_handles`, as
solve.py does) and checked against the measured entry in card.json before writing.

    PYTHONPATH=$PWD/src:$PWD uv run --no-sync python docs/rewrite/sections/balance-y046/apply.py

Run it once, on game v23's files: the factors multiply the handles as they are.
"""

import json
import pathlib
import tomllib

import torch

from bnb.game import api, ships
from bnb.game.handling import Handles

HERE = pathlib.Path(__file__).parent
SET = "real-scale"
HEAD_ON_FRACTION = 0.05
FACTORS = {
    "energy": ("yongma", {"sustained_turn_rate": 1.113, "max_shield": 0.46, "ceiling": 1.4438,
                          "climb_time": 1.4443, "shots_per_store": 2.1373}),
    "turn": ("jujak", {"sustained_turn_rate": 0.95, "max_shield": 1.1586}),
}  # fmt: skip
NOTE = (
    "the Y0.46 balance card with head-on damage 0.05 (docs/rewrite/sections/g-v23/"
    "results-fix.md on tune/g-v23, 2026-11-16)"
)


def main() -> None:
    measured = json.loads((HERE / "card.json").read_text())[0]["card"]
    rules = api.rules_of(api.PRESETS["real_scale_deathmatch"], types=api.ship_types(ship_set=SET))
    for side, (key, factors) in FACTORS.items():
        data = tomllib.loads((ships.directory_of(SET) / f"{key}.toml").read_text())
        handles, notes = dict(data["handles"]), dict(data.get("notes", {}))
        for name, factor in factors.items():
            old = handles[name]
            handles[name] = old * factor
            notes[name] = f"{notes.get(name, '')} {NOTE}: {old:.6g} -> {handles[name]:.6g}".strip()
            print(f"{key} {name}: {old:.6g} -> {handles[name]:.6g}")
        old = handles["head_on_fraction"]
        handles["head_on_fraction"] = HEAD_ON_FRACTION
        notes["head_on_fraction"] = (
            f"{notes.get('head_on_fraction', '')} {NOTE}: {old:.6g} -> {HEAD_ON_FRACTION:g}"
        ).strip()
        print(f"{key} head_on_fraction: {old:.6g} -> {HEAD_ON_FRACTION:g}")
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
        for field, entry in measured[side].items():  # the card the balance runs measured
            if abs(card[field] - entry["set"]) > 1e-5 * max(abs(entry["set"]), 1e-9):
                raise SystemExit(f"{key} {field}: solved {card[field]}, measured {entry['set']}")
        ships.write_ship(ships.Ship(key, data["name"], handles, card, notes), SET)
    print("wrote", ships.directory_of(SET))


if __name__ == "__main__":
    main()
