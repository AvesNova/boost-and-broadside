"""The result store: one SQLite file of agents, match conditions and games, and the Sobol
streams of the scripted search.

- agents: id (a short hash of the agent's identity: a scripted agent's full parameter vector
  with the rows' names; a network's checkpoint file hash, which weights and the inference
  version), kind, and the vector or the path as JSON. The inference version is a hash of the
  source a network acts through: lean/model (the model and lean.model.agent's step) and
  lean/eval/runner.py.
- conditions: id (a short hash of the game config, the seats' team sizes and the game
  version), the version, and the conditions as JSON. The game version is a hash of the
  source that decides a game: lean/game (the card, derived by tuning.py, and the config's
  defaults are there), lean/scripted and lean/contract.py. A change in either
  version starts new agents or conditions, so old games are kept but not mixed in.
- games: agent A, agent B, the team A played, conditions, result (0 A won, 1 B won, 2 a
  draw), length in ticks, and each seat's style (lean.eval.style.MEASURES, columns `a_...`
  and `b_...`; NULL for games stored before the style counters or where undefined). Counts
  are a view (`counts`); a pairing is read in both orientations. `styles` is a view of each
  agent's style per conditions over all its games (either seat): means weighted by
  ship-minutes, two_circle by turning-fight time, hit_range by hits.
- sobol: a scrambled Sobol sequence per name: dimension, seed, points used.
"""

import dataclasses
import functools
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import torch

from lean.eval.style import MEASURES
from lean.scripted import PARAMETERS, ScriptedConfig
from lean.scripted.parameters import from_vector, vector

LEAN = Path(__file__).resolve().parent.parent
SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (id TEXT PRIMARY KEY, kind TEXT NOT NULL, description TEXT NOT
NULL);
CREATE TABLE IF NOT EXISTS conditions (id TEXT PRIMARY KEY, version TEXT NOT NULL, description TEXT
NOT NULL);
CREATE TABLE IF NOT EXISTS games (
    agent_a TEXT NOT NULL, agent_b TEXT NOT NULL, side INTEGER NOT NULL, conditions TEXT NOT NULL,
    result INTEGER NOT NULL, length INTEGER NOT NULL, {style_columns}
);
CREATE INDEX IF NOT EXISTS games_by_pairing ON games (conditions, agent_a, agent_b);
CREATE VIEW IF NOT EXISTS counts AS
    SELECT conditions, agent_a, agent_b, COUNT(*) AS games, SUM(result = 0) AS wins, SUM(result = 2)
    AS draws,
           SUM(result = 1) AS losses, SUM(side = 0) AS as_team_zero, AVG(length) AS mean_length
    FROM games GROUP BY conditions, agent_a, agent_b;
{style_view}
CREATE TABLE IF NOT EXISTS sobol (name TEXT PRIMARY KEY, dimension INTEGER NOT NULL, seed INTEGER
NOT NULL,
                                  used INTEGER NOT NULL);
"""

STYLE_COLUMNS = tuple(f"{seat}_{name}" for seat in "ab" for name in MEASURES)


def _style_view() -> str:
    """The `styles` view: per conditions and agent, the agent's games from both seats."""
    seats = " UNION ALL ".join(
        f"SELECT conditions, agent_{seat} AS agent, "
        + ", ".join(f"{seat}_{name} AS {name}" for name in MEASURES)
        + f" FROM games WHERE {seat}_minutes IS NOT NULL"
        for seat in "ab"
    )
    weighted = ", ".join(
        f"SUM({name} * minutes) / SUM(minutes) AS {name}"
        for name in MEASURES
        if name not in ("minutes", "two_circle", "hits", "hit_range")
    )
    return (
        "CREATE VIEW IF NOT EXISTS styles AS SELECT conditions, agent, COUNT(*) AS games, "
        "SUM(minutes) AS minutes, " + weighted + ", "
        "SUM(two_circle * turning * minutes) / SUM(CASE WHEN two_circle IS NOT NULL THEN turning * "
        "minutes END) "
        "AS two_circle, SUM(hits) AS hits, SUM(hit_range * hits) / SUM(CASE WHEN hit_range IS NOT "
        "NULL THEN hits END) "
        f"AS hit_range FROM ({seats}) GROUP BY conditions, agent;"
    )


SCHEMA = SCHEMA.format(
    style_columns=", ".join(f"{column} REAL" for column in STYLE_COLUMNS), style_view=_style_view()
)


def short_hash(*parts) -> str:
    """12 hex digits of the SHA-256 of the parts' JSON."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def game_files() -> list[Path]:
    """The source that decides a game: lean/game (with tuning.py, which derives the card),
    lean/scripted (not their tests) and lean/contract.py."""
    return sorted(
        [*(LEAN / "game").glob("*.py"), *(LEAN / "scripted").glob("*.py"), LEAN / "contract.py"]
    )


def inference_files() -> list[Path]:
    """The source a network agent acts through besides the game: lean/model (not its tests)
    and the runner."""
    return sorted([*(LEAN / "model").glob("*.py"), LEAN / "eval" / "runner.py"])


def source_hash(files) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(LEAN)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


@functools.cache
def game_version() -> str:
    """A hash of game_files()."""
    return source_hash(game_files())


@functools.cache
def inference_version() -> str:
    """A hash of inference_files(): part of a network agent's id."""
    return source_hash(inference_files())


@dataclass(frozen=True)
class ScriptedAgent:
    """A scripted agent by its full parameter vector (lean.scripted.parameters.vector)."""

    values: tuple[float, ...]

    @classmethod
    def of(cls, config: ScriptedConfig | None = None) -> ScriptedAgent:
        return cls(vector(config or ScriptedConfig()))

    @property
    def id(self) -> str:
        return short_hash("scripted", dict(zip(PARAMETERS, self.values, strict=True)))

    def config(self) -> ScriptedConfig:
        return from_vector(self.values)

    def description(self) -> dict:
        return dict(zip(PARAMETERS, self.values, strict=True))


@dataclass(frozen=True)
class NetworkAgent:
    """A network: a checkpoint (lean.learn.save_checkpoint) and which weights ("live", "ema",
    "uniform")."""

    checkpoint: str
    weights: str = "ema"

    @property
    def id(self) -> str:
        return short_hash("network", file_hash(self.checkpoint), self.weights, inference_version())

    def description(self) -> dict:
        return {
            "checkpoint": str(self.checkpoint),
            "weights": self.weights,
            "file_hash": file_hash(self.checkpoint),
            "inference_version": inference_version(),
        }


@functools.cache
def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def conditions_of(game_config, sizes: tuple[int, int]) -> tuple[str, dict]:
    """(id, description) of the match conditions: the game config, the team sizes of seats A
    and B, and the game version."""
    description = {"game": dataclasses.asdict(game_config), "sizes": [int(sizes[0]), int(sizes[1])]}
    version = game_version()
    return short_hash(description, version), description


class Store:
    """The SQLite file (created when missing)."""

    def __init__(self, path):
        self.connection = sqlite3.connect(str(path))
        # A job commits once per pairing (tens of thousands per call): write-ahead logging
        # with normal sync does not wait on the disk per commit (laptop NVMe: 3.5 ms against
        # 0.03 ms a commit). A power cut may lose the last commits, never the file.
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        # Stores from before the style counters: their games take the style columns as NULL.
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(games)")}
        if columns and "a_minutes" not in columns:
            with self.connection:
                for column in STYLE_COLUMNS:
                    self.connection.execute(f"ALTER TABLE games ADD COLUMN {column} REAL")
        self.connection.executescript(SCHEMA)

    def add_agent(self, agent) -> str:
        kind = "scripted" if isinstance(agent, ScriptedAgent) else "network"
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO agents VALUES (?, ?, ?)",
                (agent.id, kind, json.dumps(agent.description())),
            )
        return agent.id

    def agent(self, agent_id: str):
        """The agent of an id (a ScriptedAgent or NetworkAgent), or None."""
        row = self.connection.execute(
            "SELECT kind, description FROM agents WHERE id = ?", (agent_id,)
        ).fetchone()
        if row is None:
            return None
        description = json.loads(row[1])
        if row[0] == "scripted":
            return ScriptedAgent(tuple(description[name] for name in PARAMETERS))
        return NetworkAgent(description["checkpoint"], description["weights"])

    def add_conditions(self, game_config, sizes) -> str:
        identity, description = conditions_of(game_config, sizes)
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO conditions VALUES (?, ?, ?)",
                (identity, game_version(), json.dumps(description)),
            )
        return identity

    def add_games(
        self, agent_a: str, agent_b: str, conditions: str, sides, results, lengths, styles=None
    ) -> None:
        """`styles`: per game a tuple of STYLE_COLUMNS' values (lean.eval.style.measures), or
        None (NULL style)."""
        styles = styles if styles is not None else [(None,) * len(STYLE_COLUMNS)] * len(sides)
        rows = [
            (agent_a, agent_b, int(side), conditions, int(result), int(length), *style)
            for side, result, length, style in zip(sides, results, lengths, styles, strict=True)
        ]
        names = ("agent_a", "agent_b", "side", "conditions", "result", "length", *STYLE_COLUMNS)
        with self.connection:
            self.connection.executemany(
                f"INSERT INTO games ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
                rows,
            )

    def styles(self, conditions: str, agents=None) -> list[dict]:
        """The `styles` view's rows under the conditions (of `agents`' ids only, if given), as
        dicts: agent, games, minutes and lean.eval.style.MEASURES."""
        cursor = self.connection.execute("SELECT * FROM styles WHERE conditions = ?", (conditions,))
        names = [column[0] for column in cursor.description]
        rows = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
        return [row for row in rows if agents is None or row["agent"] in set(agents)]

    def counts(self, agent_a: str, agent_b: str, conditions: str) -> dict[str, int]:
        """A's games against B under the conditions, from both orientations: games, wins,
        draws, losses (A's) and A's games as team 0."""
        total = {"games": 0, "wins": 0, "draws": 0, "losses": 0, "as_team_zero": 0}
        query = (
            "SELECT games, wins, draws, losses, as_team_zero FROM counts "
            "WHERE conditions = ? AND agent_a = ? AND agent_b = ?"
        )
        row = self.connection.execute(query, (conditions, agent_a, agent_b)).fetchone()
        if row is not None:
            for key, value in zip(total, row, strict=True):
                total[key] += value
        if agent_a != agent_b:
            row = self.connection.execute(query, (conditions, agent_b, agent_a)).fetchone()
            if row is not None:
                games, wins, draws, losses, as_team_zero = row
                total["games"] += games
                total["wins"] += losses
                total["draws"] += draws
                total["losses"] += wins
                total["as_team_zero"] += games - as_team_zero
        return total

    def pairing_totals(self, conditions: str) -> list[tuple[str, str, int, float]]:
        """(agent A, agent B, games, A's score: wins plus half the draws) of every stored
        orientation under the conditions."""
        return self.connection.execute(
            "SELECT agent_a, agent_b, games, wins + 0.5 * draws FROM counts WHERE conditions = ?",
            (conditions,),
        ).fetchall()

    def total_games(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM games").fetchone()[0]


class SobolStream:
    """A scrambled Sobol sequence that resumes: its seed and the points used are stored under
    its name, and `next(n)` continues where the last call stopped (the first points never
    change when the stream is extended)."""

    def __init__(self, store: Store, name: str, dimension: int, seed: int = 0):
        connection = store.connection
        row = connection.execute(
            "SELECT dimension, seed FROM sobol WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            with connection:
                connection.execute("INSERT INTO sobol VALUES (?, ?, ?, 0)", (name, dimension, seed))
        elif row[0] != dimension:
            raise ValueError(f"Sobol stream {name!r} has dimension {row[0]}, not {dimension}")
        else:
            seed = row[1]
        self.store, self.name, self.dimension, self.seed = store, name, dimension, seed

    @property
    def used(self) -> int:
        return self.store.connection.execute(
            "SELECT used FROM sobol WHERE name = ?", (self.name,)
        ).fetchone()[0]

    def next(self, count: int) -> torch.Tensor:
        """The next `count` points, (count, dimension) float64 in [0, 1)."""
        used = self.used
        engine = torch.quasirandom.SobolEngine(self.dimension, scramble=True, seed=self.seed)
        engine.fast_forward(used)
        points = engine.draw(count, dtype=torch.float64)
        with self.store.connection:
            self.store.connection.execute(
                "UPDATE sobol SET used = ? WHERE name = ?", (used + count, self.name)
            )
        return points
