"""Fresh games to their end: every environment plays one game from a fresh clock, and each
of a player's views reports that game's result.

The rollout is the training rollout (`Rollout` with `staggered=False`), so a player is any
controller, a model or the scripted agent, and its opponents are the pools' other seats.
"""

import torch

from bnb.contracts import Controller, GameConfig, Pool

from .rollout import Rollout


@torch.no_grad()
def first_results(
    game: GameConfig,
    pools: tuple[Pool, ...],
    controllers: dict[str, Controller],
    player: str,
    games: int,
    device: torch.device,
    length: int = 32,
    compiled: bool = True,
) -> torch.Tensor:
    """(views,) int32: the result of each of `player`'s views in its environment's first
    game, +1 a win, 0 a draw, -1 a loss. games: environments (the pools' batch)."""
    rollout = Rollout(
        game, pools, controllers, player, games, length, device, compiled=compiled, staggered=False
    )
    views = rollout.lineup.rows(player)
    result = torch.zeros(views, dtype=torch.int32, device=device)
    finished = torch.zeros(views, dtype=torch.bool, device=device)
    while not bool(finished.all()):
        trajectory = rollout.run()  # (L, views)
        for tick in range(trajectory.result.shape[0]):
            first = trajectory.done[tick] & ~finished
            result = torch.where(first, trajectory.result[tick], result)
            finished = finished | trajectory.done[tick]
    return result


def score(results: torch.Tensor) -> float:
    """Wins plus half the draws, as a share of the games."""
    return float(((results > 0).float() + 0.5 * (results == 0).float()).mean())


def shares(results: torch.Tensor) -> dict:
    """The shares won, drawn and lost."""
    return dict(
        wins=float((results > 0).float().mean()),
        draws=float((results == 0).float().mean()),
        losses=float((results < 0).float().mean()),
    )
