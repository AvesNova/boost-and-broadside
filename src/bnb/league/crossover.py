"""Crossover fights: `bnb crossover`, a player against another at unequal ship counts, to
find the handicap at which the first player's score crosses one half.

A game is built at one size, `ships_per_team = M`; each pair of active counts (a, b) plays
the first player with a ships against the second with b, the rest of each team's slots
benched (`bnb.game.api`: benched ships stay dead, out of play, and team deathmatch's winning
score counts the enemy's active ships). Every pair plays on the match engine built with
benching (`bnb.league.matches.Arena`), in the head-to-head layout
(`bnb.league.tournament.head_to_head_layout`): blocks of games, each block one pair, all
the pairs in the same batches, each block's games split evenly between the seats.

Per pair: the first player's wins, draws and losses, its score (a draw half a win) and the
score's 95% Wilson interval (`wilson_interval`). The crossover (`crossings`) is read along
the pairs in the order given: where the score crosses one half between two neighbouring
pairs, the counts there by linear interpolation. Refining is the caller's: a finer grid, or
more games, around the crossing.
"""

import math

import torch

from bnb.contracts import GameConfig

from .matches import Arena, ArenaConfig, Player, network_groups
from .tournament import head_to_head_layout, identity

Z_95 = 1.959963984540054  # the standard normal's two-sided 95% quantile


def parse_pair(text: str) -> tuple[int, int]:
    """`a:b` as (a, b): the first player's active ships and the second's. Raises ValueError
    for anything else, or a count below 1."""
    first, colon, second = text.partition(":")
    try:
        counts = (int(first), int(second))
    except ValueError:
        counts = None
    if not colon or counts is None or min(counts) < 1:
        raise ValueError(f"{text!r}: a pair is a:b, two positive ship counts (such as 5:6)")
    return counts


def wilson_interval(score: float, games: int, z: float = Z_95) -> tuple[float, float]:
    """The Wilson score interval of a proportion `score` over `games` games at `z` standard
    deviations (a draw counted as half a win, as the score is): (low, high), or (0, 1) for
    no games."""
    if games == 0:
        return 0.0, 1.0
    z2 = z * z
    centre = (score + z2 / (2 * games)) / (1 + z2 / games)
    half = z * math.sqrt(score * (1 - score) / games + z2 / (4 * games * games)) / (1 + z2 / games)
    return max(centre - half, 0.0), min(centre + half, 1.0)


def summary(counts: torch.Tensor) -> dict:
    """games, wins, draws, losses, the score and its 95% Wilson interval from (wins, ties,
    losses)."""
    won, tied, lost = (int(value) for value in counts.tolist())
    games = won + tied + lost
    score = (won + tied / 2) / games if games else math.nan
    low, high = wilson_interval(score, games) if games else (0.0, 1.0)
    return dict(
        games=games, wins=won, draws=tied, losses=lost, score=score, score_low=low, score_high=high
    )


def crossover(
    first: Player,
    second: Player,
    pairs: list[tuple[int, int]],
    games: int,
    game: GameConfig,
    device: torch.device,
    config: ArenaConfig = ArenaConfig(),
) -> list[dict]:
    """`first` against `second` in `game` at each pair of active ship counts (first's,
    second's; each from 1 to the game's ships per team), `games` games per pair (even:
    split evenly between the seats). A row per pair, in order: the players, their
    identities (`bnb.league.tournament.identity`), the counts, the handicap (second's
    count less first's) and `summary`. The pairs' blocks share batches; a last batch's spare
    blocks replay the first pairs, and their games are left out. A player may be named on
    both sides (two players of the same name, such as `scripted` against itself)."""
    if not pairs:
        raise ValueError("no pairs of active ship counts to play")
    if games < 2 or games % 2:
        raise ValueError(f"{games} games per pair must be even (split evenly between the seats)")
    size = game.ships_per_team
    for pair in pairs:
        if not all(1 <= count <= size for count in pair):
            raise ValueError(
                f"pair {pair[0]}:{pair[1]}: active ships from 1 to the game's {size} ships per "
                "team (a larger fleet needs a larger game)"
            )
    players = [first, second]
    layout = head_to_head_layout(len(pairs), games, config)
    queue = [
        number for number in range(len(pairs)) for _ in range(games // layout.block_environments)
    ]
    batches = [
        queue[start : start + layout.blocks] for start in range(0, len(queue), layout.blocks)
    ]
    batches = [batch + queue[: layout.blocks - len(batch)] for batch in batches]  # spare blocks
    groups = network_groups(players)
    lanes = tuple(
        layout.blocks * groups.count(group)
        for group in range(1 + max((group for group in groups if group is not None), default=-1))
    )
    arena = Arena(players, game, device, layout, lanes, benching=True)
    counts = torch.zeros((len(pairs), 3), dtype=torch.int64)
    for start, batch in zip(range(0, len(queue), layout.blocks), batches, strict=True):
        played = queue[start : start + layout.blocks]
        results = arena.play([(0, 1)] * layout.blocks, [pairs[number] for number in batch])
        for number, result in zip(played, results[: len(played)], strict=True):
            counts[number] += result
    return [
        dict(
            first=first.name,
            second=second.name,
            first_identity=identity(first),
            second_identity=identity(second),
            first_active=pair[0],
            second_active=pair[1],
            handicap=pair[1] - pair[0],
            **summary(counts[number]),
        )
        for number, pair in enumerate(pairs)
    ]


def crossings(rows: list[dict]) -> list[dict]:
    """Where the first player's score crosses one half along the rows (in their order): a
    row at one half exactly, and between two neighbouring rows whose scores lie strictly on
    either side of it, the counts at one half by linear interpolation in the score. Each:
    the rows' pairs (`between`, a:b), and first_active, second_active and the handicap
    there (floats)."""
    found = []

    def crossing(before: dict, after: dict, fraction: float) -> dict:
        def at(name: str) -> float:
            return before[name] + fraction * (after[name] - before[name])

        pairs = [f"{row['first_active']}:{row['second_active']}" for row in (before, after)]
        return dict(
            between=pairs[:1] if before is after else pairs,
            first_active=at("first_active"),
            second_active=at("second_active"),
            handicap=at("handicap"),
        )

    for index, row in enumerate(rows):
        if row["score"] == 0.5:
            found.append(crossing(row, row, 0.0))
        if index + 1 < len(rows):
            low, high = row["score"] - 0.5, rows[index + 1]["score"] - 0.5
            if low * high < 0:
                found.append(crossing(row, rows[index + 1], low / (low - high)))
    return found
