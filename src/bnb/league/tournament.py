"""Published ratings and head-to-head results (league spec 6): `bnb ratings` and
`bnb head-to-head`, on the match engine's fixed slots (`bnb.league.matches.Arena`).

The ratings are an offline tournament over a field, game preset by game preset, in batches
of the engine's fixed matchup blocks. Before each batch every block is given a pair, among
the allowed pairs, by A-optimal value (`bnb.league.ratings.allocate_blocks`, at the current
ratings and every game so far); after it, the Bradley-Terry fit on every game so far
(`fit_ratings`, draws as half a win, prior 1 game), shifted so the scripted agent is 1000:
only it is pinned. The tournament stops when every rating's standard error is below the
target, or when the next batch would pass the game budget. The result is the leaderboard
(`calibrated_elo`), keyed by game version, mode, map and fleet size.

The field may restrict who plays whom (`allowed_pairs`): the scripted agent can be kept to
a few players, which anchor the scale for the others through their games.
"""

import dataclasses
import math
import time
from collections.abc import Callable

import torch

from bnb.contracts import GameConfig
from bnb.game import VERSION as GAME_VERSION
from bnb.game.api import PRESETS

from .gauge import GAUGE_TOP
from .matches import Arena, ArenaConfig, Player
from .ratings import allocate_blocks, fit_ratings

PRESET_NAMES = ("frontline", "deathmatch", "deathmatch_wide")  # the training mix's games
PRIOR_GAMES = 1.0
ANCHOR = "scripted"


@dataclasses.dataclass(frozen=True)
class TournamentConfig:
    """arena: the match engine's layout (a batch is its blocks' games); target_stderr:
    rating points every standard error must be under to stop; budget: games per preset at
    most (whole batches)."""

    arena: ArenaConfig = ArenaConfig()
    target_stderr: float = 25.0
    budget: int = 20_000

    def __post_init__(self) -> None:
        if self.target_stderr <= 0.0:
            raise ValueError(f"target_stderr {self.target_stderr} must be positive")
        if self.budget < self.arena.games:
            raise ValueError(
                f"a budget of {self.budget} games is less than one batch ({self.arena.games})"
            )


def allowed_pairs(names: list[str], anchor_partners: tuple[str, ...] | None = None) -> torch.Tensor:
    """Who may play whom in a field of `names`, (K, K) bool: every pair of different
    players, except that the scripted agent plays only `anchor_partners` when they are
    given."""
    count = len(names)
    allowed = ~torch.eye(count, dtype=torch.bool)
    if anchor_partners is not None:
        unknown = [name for name in anchor_partners if name not in names or name == ANCHOR]
        if unknown or not anchor_partners:
            raise ValueError(
                f"the scripted agent's partners {list(anchor_partners)} must be other players "
                "of the field"
            )
        anchor = names.index(ANCHOR)
        partner = torch.tensor([name in anchor_partners for name in names])
        allowed[anchor] = partner
        allowed[:, anchor] = partner
    return allowed


def _connected(allowed: torch.Tensor, anchor: int) -> bool:
    """Whether every player reaches the anchor through allowed pairs."""
    reached = torch.zeros(allowed.shape[0], dtype=torch.bool)
    reached[anchor] = True
    while True:
        grown = reached | allowed[reached].any(0)
        if bool((grown == reached).all()):
            return bool(reached.all())
        reached = grown


def fit_field(
    play: Callable[[list[tuple[int, int]]], torch.Tensor],
    count: int,
    anchor: int,
    allowed: torch.Tensor,
    config: TournamentConfig,
    log: Callable[[str], None] = print,
    label: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The tournament's batches over a field of `count` players: `play` plays one batch
    (a pair per block, first player and second) and gives each block's (wins, ties,
    losses) from its first player's side. Returns the ratings (K,) with the anchor at 0,
    their standard errors (K,) and the games of each pair (K, K)."""
    if not _connected(allowed, anchor):
        raise ValueError("every player must reach the scripted agent through allowed pairs")
    arena = config.arena
    wins = torch.zeros((count, count), dtype=torch.float64)
    pair_games = torch.zeros((count, count), dtype=torch.float64)
    ratings = torch.zeros(count, dtype=torch.float64)
    played = 0
    while True:
        started = time.perf_counter()
        matchups = allocate_blocks(
            pair_games,
            ratings,
            anchor,
            arena.blocks,
            arena.block_environments,
            allowed,
            PRIOR_GAMES,
        )
        results = play(matchups)
        for (first, second), (won, tied, lost) in zip(matchups, results.tolist(), strict=True):
            wins[first, second] += won + tied / 2
            wins[second, first] += lost + tied / 2
            games = won + tied + lost
            pair_games[first, second] += games
            pair_games[second, first] += games
            played += games
        ratings, stderr = fit_ratings(wins, anchor, PRIOR_GAMES)
        worst = float(stderr.max())
        seconds = time.perf_counter() - started
        log(
            f"{label}: {played} games, the largest standard error {worst:.1f} "
            f"(the batch took {seconds:.0f} s)"
        )
        if worst < config.target_stderr or played + arena.games > config.budget:
            return ratings, stderr, pair_games


def rate(
    players: list[Player],
    preset: str,
    device: torch.device,
    config: TournamentConfig = TournamentConfig(),
    log: Callable[[str], None] = print,
    game: GameConfig | None = None,
    allowed: torch.Tensor | None = None,
) -> list[dict]:
    """The field's ratings in one game preset (`game`: its config, PRESETS[preset] by
    default): a row per player (its name, `calibrated_elo`, its standard error and its
    games) plus the leaderboard's key. The field must hold the scripted agent, the
    anchor; allowed (K, K) bool: who may play whom (`allowed_pairs`; every pair by
    default)."""
    game = PRESETS[preset] if game is None else game
    names = [p.name for p in players]
    if ANCHOR not in names:
        raise ValueError("the field needs the scripted agent: it anchors the ratings at 1000")
    allowed = allowed_pairs(names) if allowed is None else allowed
    arena = Arena(players, game, device, config.arena)
    ratings, stderr, pair_games = fit_field(
        arena.play, len(players), names.index(ANCHOR), allowed, config, log, preset
    )
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


def head_to_head_layout(pairs: int, games: int, config: ArenaConfig) -> ArenaConfig:
    """The engine's layout for `pairs` pairs of `games` games each: blocks of the largest
    even divisor of `games` up to `config.block_environments` games, and as many blocks as
    the pairs need, up to `config.blocks`."""
    environments = max(
        size for size in range(2, config.block_environments + 1, 2) if games % size == 0
    )
    blocks = min(config.blocks, pairs * games // environments)
    return dataclasses.replace(config, blocks=blocks, block_environments=environments)


def head_to_head(
    players: list[Player],
    games: int,
    presets: tuple[str, ...],
    device: torch.device,
    config: ArenaConfig = ArenaConfig(),
    games_of: dict[str, GameConfig] | None = None,
) -> list[dict]:
    """Each pair of `players` (in order: the first against the second, the third against
    the fourth, ...) over `games` games per preset (split evenly between the seats; a
    preset's config from `games_of`, PRESETS by default): the first player's wins, losses
    and draws, its score (a draw half a win) with the score's standard error, in total and
    per preset. The pairs' blocks are played in batches of the engine's layout
    (`head_to_head_layout`); a last batch's spare blocks replay the first ones, and their
    games are left out."""
    if len(players) % 2:
        raise ValueError("head-to-head takes players in pairs: A B [C D ...]")
    if games < 2 or games % 2:
        raise ValueError(f"--games {games} must be even (split evenly between the seats)")
    pairs = [(index, index + 1) for index in range(0, len(players), 2)]
    layout = head_to_head_layout(len(pairs), games, config)
    queue = [
        number for number in range(len(pairs)) for _ in range(games // layout.block_environments)
    ]
    games_of = games_of or PRESETS
    split = {}
    for preset in presets:
        arena = Arena(players, games_of[preset], device, layout)
        counts = torch.zeros((len(pairs), 3), dtype=torch.int64)
        for start in range(0, len(queue), layout.blocks):
            batch = queue[start : start + layout.blocks]
            spare = queue[: layout.blocks - len(batch)]
            results = arena.play([pairs[number] for number in batch + spare])
            for number, result in zip(batch, results[: len(batch)], strict=True):
                counts[number] += result
        split[preset] = counts
        del arena
    rows = []
    for number, (first, second) in enumerate(pairs):
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
