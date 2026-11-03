"""Published ratings and head-to-head results (league spec 6): `bnb ratings` and
`bnb head-to-head`.

The ratings are an offline tournament over a field, game preset by game preset, in batches:
the first spread evenly over the pairs, the later ones split by A-optimal value
(`bnb.league.ratings.allocate`, at least one game per pair, a pair's games rounded up to an
even count so they split evenly between the seats). After each batch, the Bradley-Terry fit
(`fit_ratings`, draws as half a win, prior 1 game), shifted so the scripted agent is 1000:
only it is pinned. The tournament stops when every rating's standard error is below the
target, or when the game budget is spent. The result is the leaderboard (`calibrated_elo`),
keyed by game version, mode, map and fleet size.
"""

import dataclasses
import math
from collections.abc import Callable

import torch

from bnb.contracts import GameConfig
from bnb.game import VERSION as GAME_VERSION
from bnb.game.api import PRESETS

from .gauge import GAUGE_TOP
from .matches import Player, play
from .ratings import allocate, fit_ratings

PRESET_NAMES = ("frontline", "deathmatch", "deathmatch_wide")  # the training mix's games
PRIOR_GAMES = 1.0


@dataclasses.dataclass(frozen=True)
class TournamentConfig:
    """first_games: games per pair in the first batch; batch_games: games per later batch;
    target_stderr: rating points every standard error must be under to stop; budget: games
    per preset at most; length: ticks per rollout segment."""

    first_games: int = 4
    batch_games: int = 1024
    target_stderr: float = 25.0
    budget: int = 20_000
    length: int = 64

    def __post_init__(self) -> None:
        if self.first_games < 2 or self.first_games % 2:
            raise ValueError(f"first_games {self.first_games} must be even and at least 2")
        if self.target_stderr <= 0.0:
            raise ValueError(f"target_stderr {self.target_stderr} must be positive")


def _even(games: int) -> int:
    return games + games % 2


def rate(
    players: list[Player],
    preset: str,
    device: torch.device,
    config: TournamentConfig = TournamentConfig(),
    log: Callable[[str], None] = print,
    game: GameConfig | None = None,
) -> list[dict]:
    """The field's ratings in one game preset (`game`: its config, PRESETS[preset] by
    default): a row per player (its name, `calibrated_elo`, its standard error and its
    games) plus the leaderboard's key. The field must hold the scripted agent, the
    anchor."""
    game = PRESETS[preset] if game is None else game
    names = [p.name for p in players]
    if "scripted" not in names:
        raise ValueError("the field needs the scripted agent: it anchors the ratings at 1000")
    count, anchor = len(players), names.index("scripted")
    wins = torch.zeros((count, count), dtype=torch.float64)
    pair_games = torch.zeros((count, count), dtype=torch.float64)
    upper = torch.triu_indices(count, count, 1)
    allocation = allocate(
        pair_games, torch.zeros(count), anchor, config.first_games * upper.shape[1], 0
    )
    played = 0
    while True:
        pairs = [
            (int(i), int(j), _even(int(allocation[i, j])))
            for i, j in upper.T
            if allocation[i, j] > 0
        ]
        results = play(players, pairs, game, device, config.length)
        for (first, second, games), (won, tied, lost) in zip(pairs, results.tolist()):
            wins[first, second] += won + tied / 2
            wins[second, first] += lost + tied / 2
            pair_games[first, second] += games
            pair_games[second, first] += games
            played += games
        ratings, stderr = fit_ratings(wins, anchor, PRIOR_GAMES)
        worst = float(stderr.max())
        log(f"{preset}: {played} games, the largest standard error {worst:.1f}")
        if worst < config.target_stderr or played >= config.budget:
            break
        batch = min(config.batch_games, config.budget - played)
        allocation = allocate(pair_games, ratings, anchor, batch, 1)
    key = dict(
        game_version=GAME_VERSION, mode=game.mode, map=preset, fleet_size=game.ships_per_team
    )
    return [
        dict(
            key,
            player=name,
            calibrated_elo=float(ratings[index]) + GAUGE_TOP,
            stderr=float(stderr[index]),
            games=int(pair_games[index].sum()),
        )
        for index, name in enumerate(names)
    ]


def head_to_head(
    players: list[Player],
    games: int,
    presets: tuple[str, ...],
    device: torch.device,
    length: int = 64,
    games_of: dict[str, GameConfig] | None = None,
) -> list[dict]:
    """Each pair of `players` (in order: the first against the second, the third against
    the fourth, ...) over `games` games per preset (split evenly between the seats; a
    preset's config from `games_of`, PRESETS by default): the
    first player's wins, losses and draws, its score (a draw half a win) with the score's
    standard error, in total and per preset."""
    if len(players) % 2:
        raise ValueError("head-to-head takes players in pairs: A B [C D ...]")
    if games < 2 or games % 2:
        raise ValueError(f"--games {games} must be even (split evenly between the seats)")
    pairs = [(index, index + 1, games) for index in range(0, len(players), 2)]
    games_of = games_of or PRESETS
    split = {preset: play(players, pairs, games_of[preset], device, length) for preset in presets}
    rows = []
    for number, (first, second, _) in enumerate(pairs):
        per_preset = {preset: _summary(split[preset][number]) for preset in presets}
        total = sum(
            (split[preset][number] for preset in presets), torch.zeros(3, dtype=torch.int64)
        )
        rows.append(
            dict(
                first=players[first].name,
                second=players[second].name,
                **_summary(total),
                presets=per_preset,
            )
        )
    return rows


def _summary(counts: torch.Tensor) -> dict:
    """wins, draws, losses, the score and its standard error from (wins, ties, losses)."""
    won, tied, lost = (int(value) for value in counts.tolist())
    games = won + tied + lost
    if games == 0:
        return dict(wins=0, draws=0, losses=0, score=math.nan, stderr=math.inf)
    score = (won + tied / 2) / games
    second_moment = (won + tied / 4) / games
    variance = max(second_moment - score * score, 0.0)
    return dict(
        wins=won,
        draws=tied,
        losses=lost,
        score=score,
        stderr=math.sqrt(variance / games),
    )
