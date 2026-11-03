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
    """`count` sizes (ships per team, fields, arena radius) drawn from the seed."""
    generator = np.random.default_rng(seed)
    drawn = []
    for _ in range(count):
        ships = 2 * int(generator.integers(SHIPS[0] // 2, SHIPS[1] // 2 + 1))
        fields = int(generator.integers(FIELDS[0], FIELDS[1] + 1))
        radius = 2600.0 * math.sqrt(ships / 10.0) * float(generator.uniform(*DENSITY))
        drawn.append(
            dict(ships_per_team=ships // 2, num_fields=fields, arena_radius=round(radius, 1))
        )
    return drawn


def environments_for(size: dict, budget: int = SHIP_BUDGET, preset: str | None = None) -> int:
    """Environments for a ship budget, rounded to a whole multiple of what the pools need
    (`environment_multiple`: the skeleton lineup's, or the mixed preset's pools)."""
    from bnb.contracts.pools import environment_multiple
    from bnb.runtime.train import LINEUP, mixed

    pools = mixed()[1] if preset == "mixed" else LINEUP
    multiple = environment_multiple(pools)
    return max(multiple, multiple * round(budget / (2 * size["ships_per_team"]) / multiple))
