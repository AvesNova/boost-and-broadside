"""Which controller plays which (environment, role), from the fixed-ratio pools."""

import torch

from bnb.contracts import GameConfig, Pool
from bnb.contracts.pools import environments_per_pool, pool_fog_share


class Lineup:
    """Rows per controller: environments[name] (R,) environment index, role[name] (R,) role
    index.

    A row's team is (role + seat[environment]) mod T, with the seat redrawn at every reset, so
    every controller plays both sides. Rows are listed pool by pool, environment by environment.

    fogged (B,) bool: the first `pool_fog_share` of each pool's environments (its fog_share
    times its game's: none in team deathmatch) have fog of war, a fixed contiguous split
    (the seat is redrawn per episode, so neither side is favoured).
    """

    def __init__(
        self,
        pools: tuple[Pool, ...],
        environments: int,
        device: torch.device,
        game: GameConfig,
    ) -> None:
        sizes = environments_per_pool(pools, environments)
        environment_of: dict[str, list[int]] = {}
        role: dict[str, list[int]] = {}
        pool_of: dict[str, list[int]] = {}
        fogged: list[bool] = []
        start = 0
        for index, (pool, size) in enumerate(zip(pools, sizes)):
            count = round(pool_fog_share(game, pool) * size)
            fogged.extend([True] * count + [False] * (size - count))
            for role_index, name in enumerate(pool.seats):
                environment_of.setdefault(name, []).extend(range(start, start + size))
                role.setdefault(name, []).extend([role_index] * size)
                pool_of.setdefault(name, []).extend([index] * size)
            start += size
        self.pools, self.sizes = pools, sizes
        self.names = tuple(environment_of)
        self.environments = {
            name: torch.tensor(rows, device=device) for name, rows in environment_of.items()
        }
        self.role = {k: torch.tensor(v, device=device) for k, v in role.items()}
        self.pool_of = {k: torch.tensor(v, device=device) for k, v in pool_of.items()}
        self.fogged = torch.tensor(fogged, dtype=torch.bool, device=device)

    def rows(self, name: str) -> int:
        """How many rows (views) the controller `name` plays."""
        return self.environments[name].shape[0]
