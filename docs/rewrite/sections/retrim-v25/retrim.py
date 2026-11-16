"""Shield trims of the v24 card (option G with Yongma's max_shield factor 0.46, head-on 0.05)
for the committed-turn Imugi (feat/imugi-initiative): Yongma's G max_shield factor 0.42 and
0.38, and 0.42 with Jujak's 1.22 (G: 1.1586). The factors are relative to G's (the v24 files
carry G's), applied to the v24 handles and re-solved as tune/g-v23's solve.py does; writes
bench.balance_sweep candidates (trims.json) with "Y0.46" the files as they are.

    PYTHONPATH=$PWD/src:$PWD uv run --no-sync python docs/rewrite/sections/retrim-v25/retrim.py
"""

import json
import pathlib
import tomllib

import torch

from bnb.game import api, ships
from bnb.game.handling import Handles

HERE = pathlib.Path(__file__).parent
SET = "real-scale"
G_YONGMA, G_JUJAK = 0.46, 1.1586  # the v24 files' max_shield factors (of v23's)
TRIMS = {"Y0.46": (0.46, 1.1586), "Y0.42": (0.42, 1.1586), "Y0.38": (0.38, 1.1586),
         "Y0.42-J1.22": (0.42, 1.22)}  # fmt: skip


def solve(key: str, factors: dict) -> dict:
    """The card fields that change when the v24 file's handles are multiplied by `factors`."""
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


if __name__ == "__main__":
    out = []
    for name, (yongma, jujak) in TRIMS.items():
        energy = solve("yongma", {"max_shield": yongma / G_YONGMA})
        turn = solve("jujak", {"max_shield": jujak / G_JUJAK})
        out.append({"name": name, "card": {"energy": energy, "turn": turn}})
    (HERE / "trims.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", HERE / "trims.json")
