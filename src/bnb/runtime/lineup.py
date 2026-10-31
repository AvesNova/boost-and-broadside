"""Which controller plays which (environment, role), from the fixed-ratio pools."""

import torch

from bnb.contracts import Pool
from bnb.contracts.pools import pool_sizes


class Lineup:
    """Rows per controller: env[name] (R,) environment index, role[name] (R,) role index.

    A row's team is (role + seat[env]) mod T, with the seat redrawn at every reset, so every
    controller plays both sides. Rows are listed pool by pool, environment by environment.

    fogged (B,) bool: the first fog_share of each pool's environments have fog of war, a
    fixed contiguous split (the seat is redrawn per episode, so neither side is favoured).
    """

    def __init__(self, pools: tuple[Pool, ...], num_envs: int, device: torch.device) -> None:
        sizes = pool_sizes(pools, num_envs)
        env: dict[str, list[int]] = {}
        role: dict[str, list[int]] = {}
        pool_of: dict[str, list[int]] = {}
        fogged: list[bool] = []
        start = 0
        for index, (pool, size) in enumerate(zip(pools, sizes)):
            count = round(pool.fog_share * size)
            fogged.extend([True] * count + [False] * (size - count))
            for r, name in enumerate(pool.seats):
                env.setdefault(name, []).extend(range(start, start + size))
                role.setdefault(name, []).extend([r] * size)
                pool_of.setdefault(name, []).extend([index] * size)
            start += size
        self.pools, self.sizes = pools, sizes
        self.names = tuple(env)
        self.env = {k: torch.tensor(v, device=device) for k, v in env.items()}
        self.role = {k: torch.tensor(v, device=device) for k, v in role.items()}
        self.pool_of = {k: torch.tensor(v, device=device) for k, v in pool_of.items()}
        self.fogged = torch.tensor(fogged, dtype=torch.bool, device=device)

    def rows(self, name: str) -> int:
        return self.env[name].shape[0]
