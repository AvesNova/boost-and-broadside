"""Accumulated match record among players whose strength does not change.

Phase 1 of ``docs/internal/live-elo-plan.md``. The live rating is carried up
from the floor through a chain of ladder rungs, and the chain is only as good as
the ratings of its links. Those ratings are a *static* estimation problem —
every rung is a file on disk whose weights never move again — so the right
treatment is to keep counting and refit, rather than to nudge with a K-factor.
This is the counting half.

What goes in is decided by weights, not by ratings. A player belongs here when
its play is fixed forever: the random agent, the semi-random rungs, the scripted
controller, and every checkpoint snapshot including the newest one whose
*rating* is still settling. Its rating being unsettled is the reason to
accumulate its games, not a reason to withhold them. What must never go in is a
player whose strength changes under the record — the live policy and the running
average — because a count matrix has no way to say when a game was played, so
pooling a moving player's results across a run fits the average of a thing that
was never the same twice.

The games themselves already exist. The evaluator's slot 4 plays the floating
checkpoint against a stationary anchor every update to settle the floating
rating, and until now those outcomes were used once and dropped. Over a run they
build exactly the graph the estimator needs — each rung against the references
below it, and against the rungs frozen before it.

This is deliberately *not* recorded in ``elo_history.jsonl``. That file holds the
run's irreplaceable measurements, the live and average policies' records, which
exist in one form for one update and can never be replayed. Everything here can
be replayed from disk at any precision later; it is kept because the *training
run* needs it now, which is a different reason and belongs in a different file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


class MatchMatrix:
    """Symmetric win/loss/tie totals between fixed-strength players.

    Stored sparsely by label pair rather than as a dense array: the pool grows
    by a rung every milestone and most pairs never meet, so a dense matrix would
    be mostly zeros and would have to be re-indexed on every promotion.
    """

    def __init__(self) -> None:
        # (low, high) label pair → [wins by low, wins by high, ties]. The key is
        # ordered so a pair has exactly one entry however it is reported.
        self._pairs: dict[tuple[str, str], list[float]] = {}

    def __len__(self) -> int:
        return len(self._pairs)

    @property
    def total_games(self) -> float:
        """Every rated episode recorded, over all pairs."""
        return float(sum(sum(record) for record in self._pairs.values()))

    def record(self, player: str, opponent: str, wins: float, losses: float, ties: float) -> None:
        """Add one batch of outcomes from ``player``'s perspective."""
        if player == opponent:
            raise ValueError(f"a player cannot be recorded against itself: {player!r}")
        if wins < 0 or losses < 0 or ties < 0:
            raise ValueError(
                f"counts must be non-negative, got {wins}, {losses}, {ties} "
                f"for {player!r} vs {opponent!r}"
            )
        key = (player, opponent) if player < opponent else (opponent, player)
        forward = key[0] == player
        record = self._pairs.setdefault(key, [0.0, 0.0, 0.0])
        record[0] += wins if forward else losses
        record[1] += losses if forward else wins
        record[2] += ties

    def labels(self) -> list[str]:
        """Every player with at least one recorded game, in sorted order."""
        seen: set[str] = set()
        for low, high in self._pairs:
            seen.add(low)
            seen.add(high)
        return sorted(seen)

    def restrict(self, keep: Iterable[str]) -> "MatchMatrix":
        """Return a copy holding only pairs where both players are in ``keep``.

        Used to drop a label the roster has retired. Dropping a *player* is not
        the same as dropping its games: every pair it appears in goes with it,
        which can disconnect the graph, so the caller should check connectivity
        afterwards rather than assume the rest is unaffected.
        """
        kept = set(keep)
        restricted = MatchMatrix()
        restricted._pairs = {
            key: list(record)
            for key, record in self._pairs.items()
            if key[0] in kept and key[1] in kept
        }
        return restricted

    def pair_games(self, labels: Sequence[str]) -> np.ndarray:
        """(K, K) symmetric count of every rated episode, ties included.

        This is the game count the Fisher information wants. Ties are evidence
        about the expected score like any other result, and this game's draw rate
        is strongly level-dependent, so dropping them would bias the weak end of
        the ladder far more than the strong end.
        """
        index = {label: position for position, label in enumerate(labels)}
        games = np.zeros((len(labels), len(labels)), dtype=np.float64)
        for (low, high), (wins_low, wins_high, ties) in self._pairs.items():
            if low not in index or high not in index:
                continue
            total = wins_low + wins_high + ties
            games[index[low], index[high]] += total
            games[index[high], index[low]] += total
        return games

    def scored_wins(self, labels: Sequence[str]) -> np.ndarray:
        """(K, K) win counts with draws scored as half a win to each side.

        The convention every other caller of ``fit_bradley_terry`` uses, and the
        one that fits the expected score without modelling a draw process. See
        that module's header for why the tie-aware alternatives are wrong here.
        """
        index = {label: position for position, label in enumerate(labels)}
        wins = np.zeros((len(labels), len(labels)), dtype=np.float64)
        for (low, high), (wins_low, wins_high, ties) in self._pairs.items():
            if low not in index or high not in index:
                continue
            wins[index[low], index[high]] += wins_low + 0.5 * ties
            wins[index[high], index[low]] += wins_high + 0.5 * ties
        return wins

    def as_records(self) -> list[dict[str, float | str]]:
        """Serializable view, sorted so the file is stable across saves."""
        return [
            {
                "a": low,
                "b": high,
                "wins_a": record[0],
                "wins_b": record[1],
                "ties": record[2],
            }
            for (low, high), record in sorted(self._pairs.items())
        ]

    def save_json(self, path: str | Path) -> None:
        """Persist beside ``roster.json``, atomically.

        Written through a temporary file and renamed, because a run is killed
        mid-save often enough that a truncated matrix would otherwise be a
        routine way to lose the whole accumulation.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"version": SCHEMA_VERSION, "pairs": self.as_records()}, indent=1)
        )
        temporary.replace(destination)

    @classmethod
    def load_json(cls, path: str | Path) -> "MatchMatrix":
        """Restore from disk. A missing file is an empty matrix, not an error.

        Resuming a run that predates this file, or one killed before its first
        save, simply starts counting from zero. Nothing else depends on the
        history being complete — the accumulation is an optimization of
        precision, not a correctness requirement.
        """
        source = Path(path)
        if not source.exists():
            return cls()
        data = json.loads(source.read_text())
        version = data.get("version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"match matrix {str(source)!r} has schema version {version!r}, "
                f"expected {SCHEMA_VERSION}"
            )
        matrix = cls()
        for record in data["pairs"]:
            matrix.record(
                str(record["a"]),
                str(record["b"]),
                float(record["wins_a"]),
                float(record["wins_b"]),
                float(record["ties"]),
            )
        return matrix

    def record_all(
        self, player: str, counts: Mapping[str, tuple[float, float, float]]
    ) -> None:
        """Record one player's batch of results against several opponents."""
        for opponent, (wins, losses, ties) in counts.items():
            if wins or losses or ties:
                self.record(player, opponent, wins, losses, ties)
