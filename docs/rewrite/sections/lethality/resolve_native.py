"""Game v13: re-solve Yongma, Jujak and vizia from their x2 handles under the x2 rules (the
similarity's card the start), so each file's card is the solve of its handles; prints the
largest relative card change against the similarity's.

    uv run --no-sync python docs/rewrite/sections/lethality/resolve_native.py
"""

import concurrent.futures
import dataclasses

import torch

from bnb.contracts import GameConfig
from bnb.game import ships
from bnb.game.api import rules_of
from bnb.game.handling import NAMES, Handles, solve
from bnb.game.ship_types import FIELDS, ShipTypes

SET = "real-scale"
KEYS = ("vizia", "yongma", "jujak")


def one(key: str) -> tuple[str, dict, float, float]:
    torch.set_num_threads(1)
    rules = rules_of(GameConfig(1, 1, 7800.0, ship_set=SET))
    ship = ships.read(ships.directory_of(SET) / f"{key}.toml", key)
    handles = Handles(**{n: torch.tensor([ship.handles[n]], dtype=torch.float64) for n in NAMES})
    start = ShipTypes(**{n: torch.tensor([ship.card[n]], dtype=torch.float64) for n in FIELDS})
    solution = solve(handles, rules, start=start)
    card = {n: float(getattr(solution.types, n)[0]) for n in FIELDS}
    change = max(abs(card[n] - ship.card[n]) / max(abs(ship.card[n]), 1e-9) for n in FIELDS)
    return key, card, float(solution.error.amax()), change


def main() -> None:
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(KEYS)) as pool:
        for key, card, error, change in pool.map(one, KEYS):
            ship = ships.read(ships.directory_of(SET) / f"{key}.toml", key)
            ships.write_ship(dataclasses.replace(ship, card=card), SET)
            print(f"{key}: worst flown error {error:.2e}, card change vs similarity {change:.2e}")


if __name__ == "__main__":
    main()
