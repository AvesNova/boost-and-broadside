"""Published ratings and head-to-head results (league spec 6): `bnb ratings` and
`bnb head-to-head`, on the match engine's fixed slots (`bnb.league.matches.Arena`).

The ratings are an offline tournament over a field, game preset by game preset, in batches
of the engine's fixed matchup blocks. Before each batch the blocks are planned
(`bnb.league.design.plan_batch`: an A-optimal design over the allowed pairs for the unmet
standard-error targets, at the current posterior, rounded to whole blocks within the
batch's players and lanes); after it, the Bradley-Terry posterior on every game so far
(`bnb.league.design.fit_posterior`, draws as half a win, a Gaussian prior about the
anchor), shifted so the scripted agent is 1000: only it is pinned. The targets
(`TournamentConfig`) are a standard error for every player, a tighter one for the decision
players, and one for each named contrast (a difference between two players' ratings). The
tournament stops when every target is met, when the next batch would pass the game budget,
or when every unmet target is out of reach within the budget. The result is the
leaderboard (`calibrated_elo`), keyed by game version, mode, map and fleet size.

The field may restrict who plays whom (`allowed_pairs`): the scripted agent can be kept to
a few players, which anchor the scale for the others through their games.

A field may instead be balanced (`TournamentConfig.allocation`): each block goes to the
allowed pair with the fewest games so far, so a budget of every pair's games is a full
round robin; and anchored on another player (`rate`'s `anchor`). The ship-type tournament
(`bench/ship_type_tournament.py`) is one: a ship set's types, each flown by the scripted
agent (`matches.ship_type_players`), anchored on one of them (the P-51D, while the
real-scale set held the researched types: docs/explanation/game/real-scale-tournament.md).

Earlier tournaments' games may be reused (`reuse_record`): every recorded game whose two
players are in the field, matched by identity (`identity`: a policy by the hash of its
weights, a scripted player by its name), starts the fit, and only the batches still needed
are played.

Team deathmatch is scored by kill share by default, or by win, draw or loss by the sign of
the final kill difference (`deathmatch_score`, `DEATHMATCH_SCORES`). The scoring is applied
when the games are fitted, not when they are recorded: each block of a batch keeps its full
outcome (`outcomes`: wins, draws and losses by the game's win rule, the summed kill shares
and each side's summed kills), so a record is resumed and reused under either scoring
(`recorded_results`). A record written before the full outcome was kept holds only the
results of the scoring it names (`scoring`; none named, the default's), and is used only
under that scoring. A budget of 0 plays nothing: the field is refitted from its record and
the reused records alone.
"""

import dataclasses
import datetime
import json
import math
import os
import pathlib
import time
from collections.abc import Callable

import torch

from bnb.contracts import GameConfig
from bnb.contracts.config import TEAM_DEATHMATCH
from bnb.game import VERSION as GAME_VERSION
from bnb.game.api import PRESETS
from bnb.runs import checkpoint

from .design import (
    HOPELESS,
    INFORMATION_MARGIN,
    PRIOR_STDERR,
    Layout,
    Targets,
    fit_posterior,
    plan_batch,
    target_vector,
)
from .gauge import GAUGE_TOP
from .matches import (
    Arena,
    ArenaConfig,
    Player,
    default_lanes,
    kill_share_sums,
    network_groups,
    player,
)

# The rating games (the author, 2026-11-13): frontline, elimination (the clock to the team
# with more health left) and team deathmatch, rated by kill share (`kill_share_scored`).
PRESET_NAMES = ("frontline", "elimination", "deathmatch")
# How a team deathmatch game is scored (`kill_share_scored`): "kill_share", the default, the
# game's kill share; "win_loss", a win, draw or loss by the sign of the final kill
# difference (the mode's score), as the other modes' games are.
DEATHMATCH_SCORES = ("kill_share", "win_loss")
ANCHOR = "scripted"
ALLOCATIONS = ("a_optimal", "balanced")


@dataclasses.dataclass(frozen=True)
class TournamentConfig:
    """arena: the match engine's layout (a batch is its blocks' games); target_stderr:
    rating points every player's standard error must be under (the floor of a ladder of
    checkpoints); decision: the players a decision rests on, whose target is
    decision_stderr; contrasts: pairs of players (A, B) whose difference r_A - r_B must
    have a standard error under contrast_stderr; budget: games per preset at most (whole
    batches), 0 to play none and refit from the recorded games; allocation: how a batch's
    blocks choose their pairs, "a_optimal" (`bnb.league.design.plan_batch`) or "balanced"
    (`balanced_blocks`); players_per_batch: the distinct players a batch may hold at most
    (None: as many as its slots); prior_stderr: the posterior's prior about the anchor
    (Elo); information_margin: the standard errors of a pair's difference its games are
    planned at (`bnb.league.design.planning_information`); triangle_share: the share of a
    batch's blocks that goes first to the least-played pairs among the decision players
    (closing their triangles for the loop check); hopeless: when every unmet target is out
    of reach, the tournament stops if each needs more than this many times the batches its
    budget has left (`bnb.league.design.plan_batch`; inf: it plays its budget)."""

    arena: ArenaConfig = ArenaConfig()
    target_stderr: float = 25.0
    budget: int = 20_000
    allocation: str = "a_optimal"
    decision: tuple[str, ...] = ()
    decision_stderr: float = 20.0
    contrasts: tuple[tuple[str, str], ...] = ()
    contrast_stderr: float = 25.0
    players_per_batch: int | None = None
    prior_stderr: float = PRIOR_STDERR
    information_margin: float = INFORMATION_MARGIN
    triangle_share: float = 0.0
    hopeless: float = HOPELESS

    def __post_init__(self) -> None:
        if self.allocation not in ALLOCATIONS:
            raise ValueError(f"allocation {self.allocation!r} is not one of {ALLOCATIONS}")
        for name in ("target_stderr", "decision_stderr", "contrast_stderr", "prior_stderr"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} {getattr(self, name)} must be positive")
        if self.budget < 0 or 0 < self.budget < self.arena.games:
            raise ValueError(
                f"a budget of {self.budget} games is less than one batch ({self.arena.games}) "
                "and not 0 (refit only)"
            )
        if not 0.0 <= self.triangle_share <= 1.0:
            raise ValueError(f"triangle_share {self.triangle_share} must be within 0 to 1")
        if self.players_per_batch is not None and self.players_per_batch < 2:
            raise ValueError(f"players_per_batch {self.players_per_batch}: a block needs two")

    def targets(self, names: list[str]) -> Targets:
        """The field's targets: every player's, the decision players' tighter ones and the
        contrasts'. Raises ValueError for a name not in the field."""
        unknown = [
            name
            for name in (*self.decision, *(name for pair in self.contrasts for name in pair))
            if name not in names
        ]
        if unknown:
            raise ValueError(f"decision players and contrasts {unknown} are not in the field")
        stderr = target_vector(names, self.target_stderr, dict.fromkeys(self.decision, 0.0))
        for name in self.decision:
            stderr[names.index(name)] = min(self.decision_stderr, self.target_stderr)
        contrasts = tuple(
            (names.index(first), names.index(second), self.contrast_stderr)
            for first, second in self.contrasts
        )
        return Targets(stderr, contrasts)

    def layout(self, lanes: tuple[list[int | None], tuple[int, ...]] | None = None) -> Layout:
        """The plan's view of a batch: the arena's blocks, the players per batch and the
        engine's network groups and lanes (None: none)."""
        groups, lane_counts = lanes if lanes is not None else (None, None)
        return Layout(
            self.arena.blocks,
            self.arena.block_environments,
            self.players_per_batch,
            None if groups is None else tuple(groups),
            lane_counts,
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


def balanced_blocks(pair_games, blocks: int, block_games: int, allowed) -> list[tuple[int, int]]:
    """Which pair each of `blocks` blocks of `block_games` games plays next: each the allowed
    pair (i, j), i < j, with the fewest games so far (the blocks already chosen counted),
    the first in row order on a tie. pair_games (K, K); allowed (K, K) bool, symmetric."""
    games = torch.as_tensor(pair_games, dtype=torch.float64).clone()
    allowed = torch.as_tensor(allowed, dtype=torch.bool)
    upper = torch.triu_indices(games.shape[0], games.shape[0], 1)
    upper = upper[:, allowed[upper[0], upper[1]]]
    if upper.shape[1] == 0:
        raise ValueError("no pair is allowed to play")
    chosen = []
    for _ in range(blocks):
        best = int(torch.argmin(games[upper[0], upper[1]]))
        first, second = int(upper[0, best]), int(upper[1, best])
        chosen.append((first, second))
        games[first, second] += block_games
        games[second, first] += block_games
    return chosen


def report(line: str) -> None:
    """The tournament's default log: a line to stdout, flushed (a run's log is a file)."""
    print(line, flush=True)


def _clock(seconds: float) -> str:
    """The local clock time `seconds` from now, HH:MM."""
    later = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=seconds)
    return later.strftime("%H:%M")


def _connected(allowed: torch.Tensor, anchor: int) -> bool:
    """Whether every player reaches the anchor through allowed pairs."""
    reached = torch.zeros(allowed.shape[0], dtype=torch.bool)
    reached[anchor] = True
    while True:
        grown = reached | allowed[reached].any(0)
        if bool((grown == reached).all()):
            return bool(reached.all())
        reached = grown


def _check_deathmatch_score(deathmatch_score: str) -> None:
    if deathmatch_score not in DEATHMATCH_SCORES:
        raise ValueError(f"deathmatch score {deathmatch_score!r} is not one of {DEATHMATCH_SCORES}")


def kill_share_scored(game: GameConfig, deathmatch_score: str = "kill_share") -> bool:
    """Whether a game is rated by kill share (the author, 2026-11-13): team deathmatch with
    deathmatch_score "kill_share" (the default), each game scored by the enemy's deaths over
    both sides' (any cause, as the mode's score counts them: `Arena.kill_shares`), a
    fractional outcome the Bradley-Terry fit takes as it takes a draw's half. With
    deathmatch_score "win_loss" team deathmatch is rated, as elimination and frontline are,
    on the match's win, draw or loss (in team deathmatch the sign of the final kill
    difference)."""
    _check_deathmatch_score(deathmatch_score)
    return game.mode == TEAM_DEATHMATCH and deathmatch_score == "kill_share"


def scoring_of(game: GameConfig, deathmatch_score: str = "kill_share") -> str:
    """How the game's results are scored, as a record names it: "kill_share" or "win_loss"
    (`kill_share_scored`). A record that names none was scored with the default
    (`scoring_of(game)`)."""
    return "kill_share" if kill_share_scored(game, deathmatch_score) else "win_loss"


def outcomes(arena: Arena, matchups: list[tuple[int, int]]) -> tuple[torch.Tensor, dict]:
    """One batch's full outcome by block, whatever the scoring: (blocks, 3) int64 the first
    player's (wins, draws, losses) by the game's win rule (`Arena.play`; in team deathmatch
    the sign of the final kill difference), and the record's further columns, "kill_share"
    each block's summed kill shares (`kill_share_sums`) and "kills" each block's (first
    player's, second player's) kills summed over its games. The kills are read once after
    the batch (`Arena.deaths`), as the kill-share scoring always read them."""
    results = arena.play(matchups)
    deaths = arena.deaths()
    kills = torch.stack([deaths[:, 1].sum(-1), deaths[:, 0].sum(-1)], -1)
    return results, dict(
        kill_share=kill_share_sums(deaths).tolist(), kills=kills.round().long().tolist()
    )


def score_results(results: list, kill_share: list, scoring: str) -> list[list[float]]:
    """Full outcomes by block (`outcomes`) as `scoring` rates them: "win_loss" the (wins,
    draws, losses) as they are; "kill_share" (the summed shares, 0, the games less them)."""
    if scoring == "kill_share":
        return [
            [share, 0.0, sum(row) - share] for row, share in zip(results, kill_share, strict=True)
        ]
    return [list(row) for row in results]


def recorded_results(entry: dict, scoring: tuple[str, str] | None) -> list | None:
    """A recorded batch's results by block as a fit of scoring[0] reads them, or None when
    it cannot. A batch with its full outcome (a "kill_share" column; `outcomes`) is scored
    as asked (`score_results`); an older one holds only the results of the scoring it names
    (none named: scoring[1], the default's), usable only under that scoring. scoring None:
    the results as they were recorded."""
    if scoring is None:
        return entry["results"]
    if "kill_share" in entry:
        return score_results(entry["results"], entry["kill_share"], scoring[0])
    return entry["results"] if entry.get("scoring", scoring[1]) == scoring[0] else None


def scored(
    arena: Arena,
    game: GameConfig,
    matchups: list[tuple[int, int]],
    deathmatch_score: str = "kill_share",
) -> torch.Tensor:
    """One batch's results by block, as the game is rated: the first player's (wins, ties,
    losses), or its kill-share score as (score, 0, games - score) (`kill_share_scored`)."""
    results = arena.play(matchups)
    return arena.kill_shares() if kill_share_scored(game, deathmatch_score) else results


def _add(wins, pair_games, matchups, results) -> int:
    """Add a batch's results (a block's first player's (wins, ties, losses), fractional for a
    kill-share game) to the wins and the pair games; returns its games."""
    played = 0
    for (first, second), (won, tied, lost) in zip(matchups, results, strict=True):
        wins[first, second] += won + tied / 2
        wins[second, first] += lost + tied / 2
        games = round(won + tied + lost)  # whole games; a kill-share row sums to them
        pair_games[first, second] += games
        pair_games[second, first] += games
        played += games
    return played


def _append(record: pathlib.Path, entry: dict) -> None:
    """One JSON line appended to `record`, flushed and synced to disk."""
    record.parent.mkdir(parents=True, exist_ok=True)
    with record.open("a") as file:
        file.write(json.dumps(entry) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _replay(
    record: pathlib.Path,
    names: list[str],
    wins,
    pair_games,
    identities: list[str] | None = None,
    scoring: tuple[str, str] | None = None,
) -> tuple[int, int, list]:
    """The batches a record holds, added to the wins and the pair games: (batches, games,
    the batch seconds). Its field must be `names` (and `identities`, where it records them),
    and every batch readable under scoring (`recorded_results`; `fit_field`)."""
    batches, played, seconds = 0, 0, []
    for line in record.read_text().splitlines():
        entry = json.loads(line)
        results = recorded_results(entry, scoring)
        if results is None:
            raise ValueError(
                f"{record} was scored by {entry.get('scoring', scoring[1])}, not {scoring[0]}, "
                "before records kept every game's full outcome; move it aside or name another "
                "--out"
            )
        recorded = entry.get("identities")
        if [row["player"] for row in entry["ratings"]] != names or (
            recorded is not None
            and identities is not None
            and [recorded.get(name) for name in names] != identities
        ):
            raise ValueError(f"{record} rates another field; move it aside to start over")
        index = {name: number for number, name in enumerate(names)}
        matchups = [(index[first], index[second]) for first, second in entry["matchups"]]
        played += _add(wins, pair_games, matchups, results)
        batches += 1
        seconds.append(entry["seconds"])
    return batches, played, seconds


def _policy_record(policy: pathlib.Path) -> dict:
    """A policy's checkpoint.json, unchecked (its files are hashed when it is loaded)."""
    return json.loads((pathlib.Path(policy) / checkpoint.RECORD).read_text())


def identity(player: Player) -> str:
    """A player's identity across tournaments: a policy's is its weights, `sha256:<hash>` of
    its `weights.safetensors` as its checkpoint.json records it (then `:<ablations>` for an
    ablated one and `@T=<temperature>` for a tempered one), so a policy moved keeps it and a
    path written anew changes it; a scripted player's (the ladder's, a teacher's) is its
    name."""
    if player.policy is None:
        return player.name
    weights = "sha256:" + _policy_record(player.policy)["files"]["weights.safetensors"]
    if player.ablations:
        weights = ":".join((weights, "+".join(player.ablations)))
    if player.temperature != 1.0:
        weights = f"{weights}@T={player.temperature!r}"
    return weights


def _legacy_identities(
    names: list[str], ended: datetime.datetime, resolve: Callable[[str], Player]
) -> tuple[dict[str, str], dict[str, str]]:
    """The identities of a record's `names` that it does not record (`identity` of the player
    each name means now, by `resolve`), and the names left out with the reason. A record
    holds no hashes, so it trusts the paths, with one check: a policy whose checkpoint.json
    was written after the record's first batch ended (`ended`; both times to the second)
    did not play its games (a rewind moves a policy and writes another at its path), and
    is left out."""
    found, skipped = {}, {}
    for name in names:
        try:
            now = resolve(name)
            if now.policy is not None:
                written = _policy_record(now.policy).get("time")
                if written is None or datetime.datetime.fromisoformat(written) > ended:
                    skipped[name] = f"its policy was written at {written}, after the games"
                    continue
            found[name] = identity(now)
        except (ValueError, OSError, KeyError) as error:
            skipped[name] = f"not found now ({error})"
    return found, skipped


@dataclasses.dataclass(frozen=True)
class Reused:
    """An earlier record's games among a field's players (`reuse_record`): wins and
    pair_games (K, K) as `fit_field` keeps them, games the games added, batches and recorded
    the record's batches and players; matched: each recorded name in the field and the
    field's name for it; skipped: each recorded name left out and why (a name merely not in
    the field is in neither); unscored: the batches left out because they hold only
    another scoring's results (`recorded_results`)."""

    wins: torch.Tensor
    pair_games: torch.Tensor
    games: int
    batches: int
    recorded: tuple[str, ...]
    matched: dict[str, str]
    skipped: dict[str, str]
    unscored: int = 0


def reuse_record(
    record: pathlib.Path,
    names: list[str],
    identities: list[str],
    resolve: Callable[[str], Player] = player,
    scoring: tuple[str, str] | None = None,
) -> Reused:
    """The games of an earlier tournament's `record` (a `<preset>.batches.jsonl`, of any
    field and layout) whose two players are both in the field `names`, matched by identity
    (`identity`, the field's `identities`). A record that holds its players' identities is
    matched by them; a legacy one (none recorded) by the players its names mean now
    (`resolve`, `_legacy_identities`). A recorded identity that two of the field's players
    share is left out (ambiguous). A record of another game version (`bnb.game.VERSION`; a
    record without one is version 1's) is not reused: its games were another game's. scoring:
    (this tournament's scoring, the one a batch that names none was scored by; `scoring_of`),
    None to take the results as recorded: a batch with its full outcome is read under any
    scoring, an older one only under its own (`recorded_results`)."""
    count = len(names)
    wins = torch.zeros((count, count), dtype=torch.float64)
    pair_games = torch.zeros((count, count), dtype=torch.float64)
    entries = [json.loads(line) for line in record.read_text().splitlines() if line.strip()]
    if not entries:
        return Reused(wins, pair_games, 0, 0, (), {}, {})
    first = entries[0]
    recorded = [row["player"] for row in first["ratings"]]
    version = first.get("game_version", 1)
    readable = [recorded_results(entry, scoring) for entry in entries]
    reason = None
    if version != GAME_VERSION:
        reason = f"game version {version}, not {GAME_VERSION}"
    elif all(results is None for results in readable):
        reason = f"scored by {first.get('scoring', scoring[1])}, not {scoring[0]}"
    if reason is not None:
        return Reused(
            wins, pair_games, 0, len(entries), tuple(recorded), {}, dict.fromkeys(recorded, reason)
        )
    if "identities" in first:
        known, skipped = first["identities"], {}
    else:
        ended = datetime.datetime.fromisoformat(first["wall_time"])
        known, skipped = _legacy_identities(recorded, ended, resolve)
    holders: dict[str, list[int]] = {}
    for number, value in enumerate(identities):
        holders.setdefault(value, []).append(number)
    index = {}
    for name in recorded:
        held = holders.get(known.get(name), [])
        if len(held) == 1:
            index[name] = held[0]
        elif held:
            skipped[name] = f"ambiguous: {[names[number] for number in held]} share its weights"
    games = 0
    for entry, results in zip(entries, readable, strict=True):
        if results is None:
            continue
        kept = [
            (pair, result)
            for pair, result in zip(entry["matchups"], results, strict=True)
            if all(name in index for name in pair) and index[pair[0]] != index[pair[1]]
        ]
        games += _add(
            wins,
            pair_games,
            [(index[one], index[other]) for (one, other), _ in kept],
            [result for _, result in kept],
        )
    matched = {name: names[number] for name, number in index.items()}
    unscored = sum(results is None for results in readable)
    return Reused(
        wins, pair_games, games, len(entries), tuple(recorded), matched, skipped, unscored
    )


def fit_field(
    play: Callable[[list[tuple[int, int]]], torch.Tensor],
    names: list[str],
    anchor: int,
    allowed: torch.Tensor,
    config: TournamentConfig,
    log: Callable[[str], None] = report,
    label: str = "",
    record: pathlib.Path | None = None,
    lanes: tuple[list[int | None], tuple[int, ...]] | None = None,
    identities: list[str] | None = None,
    reused: tuple[torch.Tensor, torch.Tensor] | None = None,
    scoring: tuple[str, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The tournament's batches over a field of players `names`: `play` plays one batch (a
    pair per block, first player and second) and gives each block's (wins, ties, losses)
    from its first player's side, or those and a dict of more of the batch's results, which
    the record keeps; with a "kill_share" among them (`outcomes`) the results are the full
    outcome, scored as `scoring` says (`recorded_results`). Returns the ratings (K,) with
    the anchor at 0, their standard errors (K,) and the games of each pair (K, K).

    record: a JSON-lines file of the batches, one line appended (and synced) after each:
    its index, the blocks' matchups and results, the games so far, every player's rating
    and standard error, the worst error, the batch's seconds and the wall time, the games
    reused, the targets left out as out of reach, and the players' identities (`identity`)
    when `identities` are given. A record that exists already is read first (its field
    must be this one), and the tournament goes on from its games.

    lanes: the engine's network groups and lanes, which each batch's blocks keep to
    (`TournamentConfig.layout`); None for an engine without lanes. reused: earlier
    tournaments' wins and pair games (`reuse_record`), added to the record's before the
    first fit: the batches are played only while a target is unmet (none when the reused
    games meet them), and the budget counts only the games played here; a budget of 0
    plays none, the fit then the recorded and reused games' alone. scoring: (how the fit
    scores the games, recorded with each batch as "scoring", the scoring of its ratings;
    the scoring of a recorded batch that names none): a record whose batches hold only
    another scoring's results is not resumed (`recorded_results`)."""
    if not _connected(allowed, anchor):
        raise ValueError("every player must reach the scripted agent through allowed pairs")
    arena, count = config.arena, len(names)
    targets = config.targets(names)
    decision = tuple(names.index(name) for name in config.decision)
    labels = targets.labels(names, anchor)
    layout = config.layout(lanes)
    wins = torch.zeros((count, count), dtype=torch.float64)
    pair_games = torch.zeros((count, count), dtype=torch.float64)
    batches, played, seconds = 0, 0, []
    if record is not None and record.exists():
        batches, played, seconds = _replay(record, names, wins, pair_games, identities, scoring)
        log(f"{label}: resumed from {record}: {batches} batches, {played} games")
    reused_games = 0
    if reused is not None:
        wins += reused[0]
        pair_games += reused[1]
        reused_games = round(float(reused[1].sum()) / 2)
    if config.budget == 0 and batches == 0 and reused_games == 0:
        raise ValueError(f"{label}: a budget of 0 refits recorded games, and none are recorded")
    posterior = fit_posterior(wins, anchor, config.prior_stderr)
    while config.budget > 0 and (
        (batches == 0 and reused_games == 0)
        or (not targets.met(posterior) and played + arena.games <= config.budget)
    ):
        started = time.perf_counter()
        left = (config.budget - played) // arena.games  # the budget's bound, in batches
        out_of_reach: dict[int, float] = {}
        if config.allocation == "balanced":
            matchups = balanced_blocks(pair_games, arena.blocks, arena.block_environments, allowed)
        else:
            plan = plan_batch(
                posterior,
                targets,
                allowed,
                layout,
                left,
                config.information_margin,
                config.triangle_share,
                decision,
                config.hopeless,
            )
            out_of_reach = plan.out_of_reach
            if plan.criterion <= 0.0 and (batches > 0 or reused_games > 0):
                log(
                    f"{label}: stopped: every unmet target is out of reach within the budget's "
                    f"{left} batches ({_reach_note(labels, out_of_reach)})"
                )
                break
            matchups = plan.blocks
        outcome, extra = play(matchups), {}
        if isinstance(outcome, tuple):
            outcome, extra = outcome
        results = outcome.tolist()
        line = dict(results=results, **extra)
        if scoring is not None:
            line["scoring"] = scoring[0]
        counted = recorded_results(line, scoring)
        played += _add(wins, pair_games, matchups, counted)
        posterior = fit_posterior(wins, anchor, config.prior_stderr)
        ratings, stderr = posterior.ratings, posterior.stderr
        seconds.append(time.perf_counter() - started)
        worst = float(stderr.max())
        if record is not None:
            _append(
                record,
                dict(
                    batch=batches,
                    game_version=GAME_VERSION,
                    **({} if scoring is None else dict(scoring=scoring[0])),
                    matchups=[[names[first], names[second]] for first, second in matchups],
                    results=results,
                    games=played,
                    reused_games=reused_games,
                    ratings=[
                        dict(player=name, rating=float(ratings[i]), stderr=float(stderr[i]))
                        for i, name in enumerate(names)
                    ],
                    worst_stderr=worst,
                    out_of_reach=[labels[t] for t in sorted(out_of_reach)],
                    seconds=seconds[-1],
                    wall_time=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                    **extra,
                    **({} if identities is None else dict(identities=dict(zip(names, identities)))),
                ),
            )
        batches += 1
        left = (config.budget - played) // arena.games
        # the errors fall as one over the root of the games: the batches to the targets
        errors, goals = targets.standard_errors(posterior)
        playing = [t for t in range(len(goals)) if t not in out_of_reach]
        ratio = max((float(errors[t] / goals[t]) for t in playing), default=1.0)
        total = played + reused_games
        needed = math.ceil(total * (ratio**2 - 1.0) / arena.games)
        expected = min(left, max(needed, 0))
        mean = sum(seconds) / len(seconds)
        log(
            f"{label}: batch {batches}, {played} games, the largest standard error {worst:.1f}"
            f"{'' if not out_of_reach else f' ({_reach_note(labels, out_of_reach)})'} (the "
            f"batch took {seconds[-1]:.0f} s); done about {_clock(expected * mean)} "
            f"({expected} batches), by {_clock(left * mean)} at the latest ({left} batches)"
        )
    return posterior.ratings, posterior.stderr, pair_games


def _reach_note(labels: list[str], out_of_reach: dict[int, float]) -> str:
    """The targets left out as out of reach, with the batches each would need."""
    listed = ", ".join(f"{labels[t]} needs {batches:.0f}" for t, batches in out_of_reach.items())
    return f"out of reach: {listed}" if listed else "none out of reach"


def rate(
    players: list[Player],
    preset: str,
    device: torch.device,
    config: TournamentConfig = TournamentConfig(),
    log: Callable[[str], None] = report,
    game: GameConfig | None = None,
    allowed: torch.Tensor | None = None,
    record: pathlib.Path | None = None,
    anchor: str = ANCHOR,
    statistics: bool = False,
    reuse: tuple[pathlib.Path, ...] = (),
    resolve: Callable[[str], Player] = player,
    deathmatch_score: str = "kill_share",
) -> list[dict]:
    """The field's ratings in one game preset (`game`: its config, PRESETS[preset] by
    default): a row per player (its name, `calibrated_elo`, its standard error and its
    games) plus the leaderboard's key. The field must hold the anchor (the scripted agent
    by default), rated 1000; allowed (K, K) bool: who may play whom (`allowed_pairs`; every
    pair by default); record: the batches' record (`fit_field`), which with `statistics`
    also keeps each block's game statistics (`Arena.statistics`, under "statistics").
    reuse: earlier tournaments' records of this preset whose games among the field start
    the fit (`reuse_record`, a legacy record's names resolved by `resolve`; `record` itself
    is left out of them), each logged. deathmatch_score: how the fit scores a team
    deathmatch game (`DEATHMATCH_SCORES`, `kill_share_scored`); each batch records its full
    outcome (`outcomes`), so any record is read under either, except an older one that
    holds only the other scoring's results (refused on resume, left out of reuse;
    `recorded_results`). The engine is built at the first batch, so a finished record, or
    a budget of 0, plays nothing."""
    game = PRESETS[preset] if game is None else game
    scoring = (scoring_of(game, deathmatch_score), scoring_of(game))
    names = [p.name for p in players]
    if anchor not in names:
        raise ValueError(f"the field needs its anchor {anchor!r}: it anchors the ratings at 1000")
    allowed = allowed_pairs(names) if allowed is None else allowed
    identities = [identity(p) for p in players]
    own = None if record is None else record.resolve()
    reused = None
    for earlier in reuse:
        if earlier.resolve() == own:
            continue
        found = reuse_record(earlier, names, identities, resolve, scoring)
        log(
            f"{preset}: reused {found.games} games of {earlier} ({found.batches} batches), "
            f"{len(found.matched)} of its {len(found.recorded)} players matched: "
            f"{sorted(found.matched.values())}"
            + "".join(f"; skipped {name}: {why}" for name, why in found.skipped.items())
            + (
                f"; left out {found.unscored} batches scored by another scoring"
                if found.unscored and found.unscored < found.batches
                else ""
            )
        )
        reused = (
            (found.wins, found.pair_games)
            if reused is None
            else (reused[0] + found.wins, reused[1] + found.pair_games)
        )
    groups = network_groups(players)
    lanes = default_lanes(groups, 2 * config.arena.blocks)
    arenas: list[Arena] = []

    def play(matchups: list[tuple[int, int]]):
        if not arenas:
            arenas.append(Arena(players, game, device, config.arena, lanes))
        results, extra = outcomes(arenas[0], matchups)
        if statistics:
            extra["statistics"] = arenas[0].statistics().tolist()
        return results, extra

    ratings, stderr, pair_games = fit_field(
        play,
        names,
        names.index(anchor),
        allowed,
        config,
        log,
        preset,
        record,
        (groups, lanes),
        identities,
        reused,
        scoring,
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
    deathmatch_score: str = "kill_share",
) -> list[dict]:
    """Each pair of `players` (in order: the first against the second, the third against
    the fourth, ...) over `games` games per preset (split evenly between the seats; a
    preset's config from `games_of`, PRESETS by default): the first player's wins, losses
    and draws, its score (a draw half a win) with the score's standard error, in total and
    per preset. The pairs' blocks are played in batches of the engine's layout
    (`head_to_head_layout`); a last batch's spare blocks replay the first ones, and their
    games are left out. deathmatch_score: how a team deathmatch game is scored
    (`kill_share_scored`)."""
    _check_deathmatch_score(deathmatch_score)
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
    batches = [
        queue[start : start + layout.blocks] for start in range(0, len(queue), layout.blocks)
    ]
    batches = [batch + queue[: layout.blocks - len(batch)] for batch in batches]  # spare blocks
    # each group's lanes: the most of its slots a batch holds
    groups = network_groups(players)
    held = [[groups[index] for number in batch for index in pairs[number]] for batch in batches]
    lanes = tuple(
        max(max(one.count(group) for one in held), 1)
        for group in range(1 + max((group for group in groups if group is not None), default=-1))
    )
    split = {}
    for preset in presets:
        arena = Arena(players, games_of[preset], device, layout, lanes)
        counts = torch.zeros((len(pairs), 3), dtype=torch.float64)
        for start, batch in zip(range(0, len(queue), layout.blocks), batches, strict=True):
            played = queue[start : start + layout.blocks]
            results = scored(
                arena, games_of[preset], [pairs[number] for number in batch], deathmatch_score
            )
            for number, result in zip(played, results[: len(played)], strict=True):
                counts[number] += result
        split[preset] = counts
        del arena
    rows = []
    for number, (first, second) in enumerate(pairs):
        per_preset = {preset: _summary(split[preset][number]) for preset in presets}
        total = sum(
            (split[preset][number] for preset in presets), torch.zeros(3, dtype=torch.float64)
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
    """wins, draws, losses, the score and its standard error from (wins, ties, losses). A
    kill-share game's counts are fractional (score, 0, games - score): its "wins" are the
    summed shares, and its standard error is the Bernoulli bound (a fractional outcome
    varies less than a win or loss of the same mean)."""
    won, tied, lost = (_count(value) for value in counts.tolist())
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


def _count(value: float) -> int | float:
    """A count as an int where it is whole (a win, draw or loss count), else the float."""
    return int(value) if float(value).is_integer() else float(value)
