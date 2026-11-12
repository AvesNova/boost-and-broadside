"""Card variants in handle space: a ship file's handles times factors, re-solved into the
card (`api.solve_handles`), given as the card fields that changed ({"set": value}), so a
variant `sizes.py` measures is one a ship file can carry (`freeze.py` writes it).

    PYTHONPATH=src:. uv run --no-sync --active python \
        docs/rewrite/sections/balance-sizes/handles.py \
        yongma cruise_speed=1.05
"""

import functools
import sys
import tomllib

import torch

from bnb.game import api, ships
from bnb.game.handling import Handles

SET = "real-scale"


@functools.cache
def rules():
    return api.rules_of(api.PRESETS["real_scale_deathmatch"], types=api.ship_types(ship_set=SET))


def handles_of(key: str) -> dict:
    return dict(tomllib.loads((ships.directory_of(SET) / f"{key}.toml").read_text())["handles"])


def solved(handles: dict) -> dict:
    solution = api.solve_handles(
        Handles(**{n: torch.tensor([v], dtype=torch.float64) for n, v in handles.items()}),
        rules(),
    )
    return {
        field: float(getattr(solution.types, field)[0])
        for field in api.ship_types(ship_set=SET).__dataclass_fields__
    }


def card_variant(key: str, factors: dict) -> dict:
    """The card fields that change when the handles are multiplied by `factors`."""
    if not factors:
        return {}
    base = handles_of(key)
    changed = dict(base)
    for name, factor in factors.items():
        changed[name] = base[name] * factor
    before, after = solved(base), solved(changed)
    return {
        field: {"set": value}
        for field, value in after.items()
        if abs(value - before[field]) > 1e-6 * max(abs(before[field]), 1e-9)
    }


if __name__ == "__main__":
    key = sys.argv[1]
    factors = {a.split("=")[0]: float(a.split("=")[1]) for a in sys.argv[2:]}
    print(card_variant(key, factors))
