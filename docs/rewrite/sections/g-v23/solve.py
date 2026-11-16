"""Option G's handle factors (balance-final, candidates/g-handles.json) on game v23's ship files
(lethality-sticky: shields and recharge x 5.4), re-solved into card fields and written as
bench.balance_sweep candidates: "v23" (the files as they are) and "G-v23".

    PYTHONPATH=src:. uv run --no-sync python docs/rewrite/sections/g-v23/solve.py
"""

import json
import pathlib
import tomllib

import torch

from bnb.game import api, ships
from bnb.game.handling import Handles

HERE = pathlib.Path(__file__).parent
SET = "real-scale"
G = {
    "energy": ("yongma", {"sustained_turn_rate": 1.113, "max_shield": 0.55, "ceiling": 1.4438,
                          "climb_time": 1.4443, "shots_per_store": 2.1373}),
    "turn": ("jujak", {"sustained_turn_rate": 0.95, "max_shield": 1.1586}),
}


def solve(key: str, factors: dict) -> dict:
    data = tomllib.loads((ships.directory_of(SET) / f"{key}.toml").read_text())
    handles = {name: value * factors.get(name, 1.0) for name, value in data["handles"].items()}
    rules = api.rules_of(api.PRESETS["real_scale_deathmatch"], types=api.ship_types(ship_set=SET))
    solution = api.solve_handles(
        Handles(**{n: torch.tensor([v], dtype=torch.float64) for n, v in handles.items()}), rules
    )
    card, out = data["card"], {}
    for field in api.ship_types(ship_set=SET).__dataclass_fields__:
        value = float(getattr(solution.types, field)[0])
        if abs(value - card[field]) > 1e-5 * max(abs(card[field]), 1e-9):
            out[field] = {"set": value}
            print(f"{key} {field}: {card[field]:.6g} -> {value:.6g}")
    return out


candidates = [
    {"name": "v23", "card": {"energy": {}, "turn": {}}},
    {"name": "G-v23", "card": {side: solve(key, f) for side, (key, f) in G.items()}},
]
(HERE / "candidates.json").write_text(json.dumps(candidates, indent=1) + "\n")
print("wrote", HERE / "candidates.json")
