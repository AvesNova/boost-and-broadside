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


def envs_for(size: dict, budget: int = SHIP_BUDGET, preset: str | None = None) -> int:
    """Environments for a ship budget, rounded to a whole multiple of what the pools need:
    8 for the skeleton lineup (its shares are eighths), 64 for the mixed preset."""
    multiple = 8
    if preset == "mixed":
        from bnb.contracts.pools import environment_multiple
        from bnb.runtime.train import mixed

        multiple = environment_multiple(mixed()[1])
    return max(multiple, multiple * round(budget / (2 * size["ships_per_team"]) / multiple))
