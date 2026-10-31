"""Random fleet, field and map sizes, drawn the way the physics race drew them.

Benchmarking at sizes nobody tuned for keeps optimizations general. SKELETON_SEED is
public; a section agent's acceptance run uses a seed it is not given.
"""

import math

import numpy as np

SHIPS = (2, 128)  # even, inclusive
FIELDS = (1, 60)
DENSITY = (0.7, 1.4)  # arena radius factor on 2600 sqrt(N / 10)
SHIP_BUDGET = 40_960  # environments x ships per batch, so cells are comparable
SKELETON_SEED = 20261031


def draw(seed: int, count: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        n = 2 * int(rng.integers(SHIPS[0] // 2, SHIPS[1] // 2 + 1))
        m = int(rng.integers(FIELDS[0], FIELDS[1] + 1))
        radius = 2600.0 * math.sqrt(n / 10.0) * float(rng.uniform(*DENSITY))
        out.append(dict(ships_per_team=n // 2, num_fields=m, arena_radius=round(radius, 1)))
    return out


def envs_for(size: dict, budget: int = SHIP_BUDGET) -> int:
    """Environments for a ship budget, rounded to a multiple of 8 (pool shares are eighths)."""
    return max(8, 8 * round(budget / (2 * size["ships_per_team"]) / 8))
