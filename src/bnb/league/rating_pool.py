"""The shared rating pool (league spec 6): every rated checkpoint of many runs and arms, and
every game played between them, in one directory per pool that several machines may write
at once (a shared volume), one fit per game preset.

**Players** have stable names: a policy's is `<run>/step_<steps>/<weights kind>` (from its
directory, `<run>/policies/step_<steps>/<kind>`, or a run's `league/` or `checkpoints/`),
a scripted player's its own name ("scripted", a ladder rung, a teacher). Each is held with
its identity (`bnb.league.tournament.identity`: a policy's SHA-256 of its weights, read
from its checkpoint.json after every file's hash is checked) and the paths where writers
found it. A name registered twice with other weights is a conflict: the first registration
stands, and the games recorded under the other weights are not counted.

**Layout:**

    POOL/pool.json                        conditions every game shares (written once)
    POOL/players/<writer>.jsonl           player registrations
    POOL/games/<preset>/<writer>.jsonl    batches, one JSON line each

**Concurrency: one file per writer, merged when read.** A writer is one process (host,
process id and start time), and only it appends to its files; a reader lists the
directories and reads every file. No two processes ever write one file, so nothing needs a
lock: file locks are unreliable on network and synchronised volumes (NFS's are advisory and
depend on its lock daemon, a synchronised folder has none), and appends from two machines
to one file can interleave or be lost. Each line is written whole and synced; a reader skips
a last line without its newline (a write in progress, or a copy cut short) and reports any
other unreadable line. Every batch carries an id, so a file copied in twice is counted once.

**Batches** keep each block's full outcome (`bnb.league.tournament.outcomes`), the
players' identities and the game version, and are scored at fit time
(`bnb.league.tournament.recorded_results`) by the pool's team-deathmatch scoring or the
one asked for. Records of `bnb ratings` are brought in once by `import_records` (their
names mapped to pool names, their batches given ids from their content).

**Rating** (`rate`): every game of the preset is fitted (`bnb.league.design.fit_posterior`),
and a session plays batches among at most `session_players` players that can be played on
this machine (the players the continuous design over every playable pair weighs most), each
batch planned on the posterior's marginal over them (`plan_batch`). Before each batch the
pool is read again, so the games other writers have added count. A new player starts at
the prior, so its first games go to players whose ratings are well known and near the
middle of the field; as it is rated, to the players closest to it.
"""

import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import re
import socket
import time
import uuid
from collections.abc import Callable

import torch

from bnb.game import VERSION as GAME_VERSION
from bnb.runs import checkpoint

from . import goodness_of_fit
from .design import Posterior, fit_posterior, pair_weights, plan_batch
from .gauge import GAUGE_TOP
from .matches import default_lanes, network_groups
from .tournament import (
    ANCHOR,
    TournamentConfig,
    _append,
    _legacy_identities,
    _reach_note,
    recorded_results,
)

FORMAT = 1
POOL_RECORD = "pool.json"
PLAYERS = "players"
GAMES = "games"
SESSION_PLAYERS = 16
# a policy directory: <run>/{policies,league,checkpoints}/step_<steps>/<weights kind>
POLICY_PATH = re.compile(r"(?:^|/)([^/]+)/(?:policies|league|checkpoints)/step_(\d+)/([^/]+)/?$")


@dataclasses.dataclass(frozen=True)
class PoolPlayer:
    """A registered player: its pool name, identity, run, steps and weights kind (None for a
    scripted player), and every path writers found it at (where it may be played)."""

    name: str
    identity: str
    run: str | None = None
    step: int | None = None
    kind: str | None = None
    paths: tuple[str, ...] = ()


def policy_name(path: str | os.PathLike) -> tuple[str, str, int, str] | None:
    """(pool name, run, steps, weights kind) of a policy directory's path, or None when the
    path is not a run's `policies/`, `league/` or `checkpoints/` step directory."""
    found = POLICY_PATH.search(str(pathlib.Path(path)).replace(os.sep, "/"))
    if found is None:
        return None
    run, step, kind = found.group(1), int(found.group(2)), found.group(3)
    return f"{run}/step_{step}/{kind}", run, step, kind


def policy_identity(path: str | os.PathLike) -> str:
    """A policy's identity, `sha256:<hash of weights.safetensors>`, after every file of the
    checkpoint is checked against its record (`bnb.runs.checkpoint.read_record`)."""
    record = checkpoint.read_record(path)
    return "sha256:" + record["files"]["weights.safetensors"]


def _writer() -> str:
    """This process's writer name: host, process id and start time."""
    host = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S")
    return f"{host}-{os.getpid()}-{stamp}"


@dataclasses.dataclass
class Lines:
    """Every readable JSON line of a directory's files, and the damaged lines' count (a
    last line without its newline is in progress and not counted)."""

    entries: list[dict]
    damaged: int = 0


def read_lines(directory: pathlib.Path) -> Lines:
    """Every writer's lines below `directory` (`*.jsonl`), in file then line order."""
    found = Lines([])
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.jsonl")):
        text = path.read_text()
        lines = text.split("\n")
        for number, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                found.entries.append(json.loads(line))
            except json.JSONDecodeError:
                if number < len(lines) - 1:  # a whole line, damaged
                    found.damaged += 1
    return found


class RatingPool:
    """One pool directory (module docstring). writer: this process's file name (a fresh one
    by default)."""

    def __init__(self, root: str | os.PathLike, writer: str | None = None) -> None:
        self.root = pathlib.Path(root)
        record = self.root / POOL_RECORD
        if not record.is_file():
            raise FileNotFoundError(f"{self.root} is not a rating pool: it has no {POOL_RECORD}")
        self.settings = json.loads(record.read_text())
        if self.settings.get("format") != FORMAT:
            raise ValueError(f"{record} is pool format {self.settings.get('format')}, not {FORMAT}")
        self.writer = writer or _writer()

    @classmethod
    def create(
        cls,
        root: str | os.PathLike,
        lineup: tuple[str, ...] = (),
        arena_scale: float | None = None,
        deathmatch_score: str = "kill_share",
    ) -> RatingPool:
        """A new pool at `root` (an existing one is opened, its conditions unchanged when they
        agree): every game in it is played with `lineup` (each slot's ship type by name, both
        teams; empty: the game's own) at `arena_scale` (None: the presets' own maps), and team
        deathmatch is fitted by `deathmatch_score` unless a command asks for the other. The
        scripted agent, the anchor at 1000, is registered."""
        root = pathlib.Path(root)
        settings = dict(
            format=FORMAT,
            anchor=ANCHOR,
            lineup=list(lineup),
            arena_scale=arena_scale,
            deathmatch_score=deathmatch_score,
        )
        record = root / POOL_RECORD
        root.mkdir(parents=True, exist_ok=True)
        try:
            with record.open("x") as file:  # exclusive: two creators cannot both write it
                file.write(json.dumps(settings, indent=1) + "\n")
        except FileExistsError:
            held = json.loads(record.read_text())
            if {key: held.get(key) for key in settings} != settings:
                raise ValueError(f"{root} is a pool with other conditions: {held}") from None
        pool = cls(root)
        if ANCHOR not in pool.players()[0]:
            pool.register([PoolPlayer(ANCHOR, ANCHOR)])
        return pool

    # ---- players

    def players(self) -> tuple[dict[str, PoolPlayer], list[str]]:
        """Every registered player by name (the first registration of a name stands; the
        paths of every registration with its identity are kept), and the conflicts found."""
        found: dict[str, PoolPlayer] = {}
        conflicts = []
        lines = read_lines(self.root / PLAYERS)
        for entry in sorted(lines.entries, key=lambda entry: entry.get("time", "")):
            name, identity = entry["name"], entry["identity"]
            held = found.get(name)
            paths = tuple(entry.get("paths", ()))
            if held is None:
                found[name] = PoolPlayer(
                    name, identity, entry.get("run"), entry.get("step"), entry.get("kind"), paths
                )
            elif held.identity != identity:
                conflicts.append(
                    f"{name} registered by {entry.get('writer')} with {identity}, not "
                    f"{held.identity}: its games under the other weights are not counted"
                )
            else:
                merged = held.paths + tuple(path for path in paths if path not in held.paths)
                found[name] = dataclasses.replace(held, paths=merged)
        if lines.damaged:
            conflicts.append(f"{lines.damaged} damaged registration lines skipped")
        return found, conflicts

    def register(self, players: list[PoolPlayer]) -> list[PoolPlayer]:
        """Register players (one line each in this writer's file); a name already held with
        other weights raises ValueError, and one held with the same weights only adds its
        paths. Returns the players registered."""
        held, _ = self.players()
        for one in players:
            if one.name in held and held[one.name].identity != one.identity:
                raise ValueError(
                    f"{one.name} is registered with {held[one.name].identity}, not "
                    f"{one.identity}: other weights under the same name"
                )
        stamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        for one in players:
            _append(
                self.root / PLAYERS / f"{self.writer}.jsonl",
                dict(
                    dataclasses.asdict(one), paths=list(one.paths), writer=self.writer, time=stamp
                ),
            )
        return players

    def add_policy(self, path: str | os.PathLike, name: str | None = None) -> PoolPlayer:
        """Register the policy at `path` under its pool name (`policy_name`, or `name`),
        checked by its content hash."""
        path = pathlib.Path(path).resolve()
        parsed = policy_name(path)
        if parsed is None and name is None:
            raise ValueError(
                f"{path} is not a run's step directory (<run>/policies/step_<n>/<kind>): name it"
            )
        pool_name, run, step, kind = parsed if parsed is not None else (name, None, None, None)
        one = PoolPlayer(name or pool_name, policy_identity(path), run, step, kind, (str(path),))
        return self.register([one])[0]

    # ---- games

    def batches(self, preset: str) -> tuple[list[dict], int]:
        """Every batch of a preset (each id once), and the damaged lines skipped."""
        lines = read_lines(self.root / GAMES / preset)
        seen, batches = set(), []
        for entry in lines.entries:
            if entry["id"] in seen:
                continue
            seen.add(entry["id"])
            batches.append(entry)
        return batches, lines.damaged

    def append_batch(self, preset: str, entry: dict) -> None:
        """Append one batch (an id is given when it has none) to this writer's file."""
        entry = dict(entry)
        entry.setdefault("id", uuid.uuid4().hex)
        entry.setdefault("writer", self.writer)
        _append(self.root / GAMES / preset / f"{self.writer}.jsonl", entry)


@dataclasses.dataclass(frozen=True)
class Tally:
    """A preset's games among `names` (K): wins and pair games (K, K), the games counted, and
    the blocks left out by reason."""

    names: list[str]
    wins: torch.Tensor
    pair_games: torch.Tensor
    games: int
    skipped: dict[str, int]


def tally(batches: list[dict], players: dict[str, PoolPlayer], scoring: tuple[str, str]) -> Tally:
    """Every block of `batches` between two registered players whose recorded identities
    are theirs, of this game version and readable under `scoring` (`recorded_results`):
    the wins and games over the registered players (the anchor first, then by name)."""
    names = [ANCHOR] + sorted(name for name in players if name != ANCHOR)
    index = {name: number for number, name in enumerate(names)}
    count = len(names)
    wins = torch.zeros((count, count), dtype=torch.float64)
    pair_games = torch.zeros((count, count), dtype=torch.float64)
    skipped: dict[str, int] = {}
    counted = 0

    def skip(reason: str, blocks: int) -> None:
        skipped[reason] = skipped.get(reason, 0) + blocks

    for entry in batches:
        blocks = len(entry["matchups"])
        if entry.get("game_version", 1) != GAME_VERSION:
            skip(f"game version {entry.get('game_version', 1)}", blocks)
            continue
        results = recorded_results(entry, scoring)
        if results is None:
            skip("scored by the other deathmatch scoring", blocks)
            continue
        identities = entry.get("identities", {})
        for (first, second), (won, tied, lost) in zip(entry["matchups"], results, strict=True):
            if first not in index or second not in index:
                skip("a player not registered", 1)
                continue
            if any(identities.get(name) != players[name].identity for name in (first, second)):
                skip("other weights than the registered ones", 1)
                continue
            one, other = index[first], index[second]
            wins[one, other] += won + tied / 2
            wins[other, one] += lost + tied / 2
            games = round(won + tied + lost)
            pair_games[one, other] += games
            pair_games[other, one] += games
            counted += games
    return Tally(names, wins, pair_games, counted, skipped)


def fit(
    pool: RatingPool, preset: str, scoring: tuple[str, str], prior_stderr: float
) -> tuple[Tally, Posterior, dict[str, PoolPlayer], list[str]]:
    """A preset's tally and posterior over every registered player, the players, and the
    notes (conflicts, damaged lines, blocks left out)."""
    players, notes = pool.players()
    batches, damaged = pool.batches(preset)
    counted = tally(batches, players, scoring)
    if damaged:
        notes.append(f"{damaged} damaged batch lines skipped")
    notes += [f"{blocks} blocks left out: {reason}" for reason, blocks in counted.skipped.items()]
    return counted, fit_posterior(counted.wins, 0, prior_stderr), players, notes


def table(
    pool: RatingPool, preset: str, scoring: tuple[str, str], prior_stderr: float
) -> tuple[list[dict], dict, list[str]]:
    """The preset's leaderboard over the players with games (each row: name, run, steps,
    kind, calibrated_elo, stderr, games), the goodness of fit (`goodness_of_fit.report`)
    and the notes."""
    counted, posterior, players, notes = fit(pool, preset, scoring, prior_stderr)
    rows = []
    for number, name in enumerate(counted.names):
        games = int(counted.pair_games[number].sum())
        if games == 0 and name != ANCHOR:
            continue
        one = players[name]
        rows.append(
            dict(
                preset=preset,
                player=name,
                run=one.run,
                step=one.step,
                kind=one.kind,
                calibrated_elo=float(posterior.ratings[number]) + GAUGE_TOP,
                stderr=float(posterior.stderr[number]),
                games=games,
            )
        )
    rows.sort(key=lambda row: -row["calibrated_elo"])
    check = goodness_of_fit.report(counted.wins, posterior, counted.names)
    return rows, check, notes


def _session(
    posterior: Posterior,
    names: list[str],
    playable: list[int],
    config: TournamentConfig,
    session_players: int,
) -> list[int]:
    """The session's players (indices, the anchor first): every playable one when they are
    few enough, else the anchor and the playable players the design over every playable
    pair, for the session's budget, weighs most (by their pairs' summed weights)."""
    if len(playable) <= session_players:
        return [0] + sorted(set(playable) - {0})
    count = len(names)
    allowed = torch.zeros((count, count), dtype=torch.bool)
    index = torch.tensor(playable)
    allowed[index[:, None], index[None, :]] = True
    allowed.fill_diagonal_(False)
    weights = pair_weights(
        posterior, config.targets(names), allowed, config.budget, config.information_margin
    )
    mass = torch.zeros(count, dtype=torch.float64)
    for (first, second), weight in weights.items():
        mass[first] += weight
        mass[second] += weight
    ranked = [int(number) for number in torch.argsort(mass, descending=True)]
    chosen = [number for number in ranked if mass[number] > 0 and number != 0]
    return [0] + sorted(chosen[: session_players - 1])


def rate(
    pool: RatingPool,
    preset: str,
    make_player: Callable[[PoolPlayer], object | None],
    play_batch: Callable[[list, list[tuple[int, int]]], tuple[torch.Tensor, dict]],
    config: TournamentConfig,
    scoring: tuple[str, str],
    session_players: int = SESSION_PLAYERS,
    log: Callable[[str], None] = print,
    extra: dict | None = None,
) -> int:
    """One rating session of a preset (module docstring): returns the batches played.

    make_player: a pool player as the engine plays it on this machine (a
    `bnb.league.matches.Player`), or None when it cannot be played here (no path with its
    weights). play_batch(players, matchups): one batch among the session's players (indices
    into them), its outcome and the record's further columns (`tournament.outcomes`); the
    engine is the caller's, built on the players with the default lanes
    (`bnb.league.matches.default_lanes`), which the plan keeps to. config: the targets, the
    layout (its players per batch too), the prior and the budget (the
    games this session may play); its decision players and contrasts name pool players.
    extra: columns every batch line records (the seed, for one)."""
    counted, posterior, players, notes = fit(pool, preset, scoring, config.prior_stderr)
    for note in notes:
        log(f"{preset}: {note}")
    names = counted.names
    made = {name: make_player(players[name]) for name in names}
    playable = [number for number, name in enumerate(names) if made[name] is not None]
    if ANCHOR not in [names[number] for number in playable]:
        raise ValueError("the scripted agent must be playable: it anchors the pool")
    unknown = [
        name
        for name in (*config.decision, *(name for pair in config.contrasts for name in pair))
        if name not in names
    ]
    if unknown:
        raise ValueError(f"decision players and contrasts {unknown} are not in the pool")
    session = _session(posterior, names, playable, config, session_players)
    field = [made[names[number]] for number in session]
    groups = network_groups(field)
    lanes = default_lanes(groups, 2 * config.arena.blocks)
    session_names = [names[number] for number in session]
    log(f"{preset}: a session of {len(session)} players: {session_names}")
    decision = tuple(session_names.index(name) for name in config.decision if name in session_names)
    session_config = dataclasses.replace(
        config,
        decision=tuple(name for name in config.decision if name in session_names),
        contrasts=tuple(
            pair for pair in config.contrasts if all(name in session_names for name in pair)
        ),
    )
    targets = session_config.targets(session_names)
    labels = targets.labels(session_names, 0)
    allowed = ~torch.eye(len(session), dtype=torch.bool)
    played = batches = 0
    seconds: list[float] = []
    while played + config.arena.games <= config.budget:
        counted, posterior, players, _ = fit(pool, preset, scoring, config.prior_stderr)
        index = {name: number for number, name in enumerate(counted.names)}
        marginal = posterior.restrict([index[name] for name in session_names])
        if targets.met(marginal):
            log(f"{preset}: every target of the session is met")
            break
        left = (config.budget - played) // config.arena.games
        plan = plan_batch(
            marginal,
            targets,
            allowed,
            session_config.layout((groups, lanes)),
            left,
            config.information_margin,
            config.triangle_share,
            decision,
            config.hopeless,
        )
        if plan.criterion <= 0.0:
            log(
                f"{preset}: stopped: every unmet target is out of reach within the budget's "
                f"{left} batches ({_reach_note(labels, plan.out_of_reach)})"
            )
            break
        started = time.perf_counter()
        results, columns = play_batch(field, plan.blocks)
        seconds.append(time.perf_counter() - started)
        pool.append_batch(
            preset,
            dict(
                game_version=GAME_VERSION,
                matchups=[[session_names[a], session_names[b]] for a, b in plan.blocks],
                identities={name: players[name].identity for name in session_names},
                results=results.tolist(),
                **columns,
                layout=dict(
                    blocks=config.arena.blocks, block_environments=config.arena.block_environments
                ),
                seconds=seconds[-1],
                wall_time=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                **(extra or {}),
            ),
        )
        played += config.arena.games
        batches += 1
        errors, goals = targets.standard_errors(marginal)
        ratio = float((errors / goals).max()) if len(goals) else 1.0
        mean = sum(seconds) / len(seconds)
        later = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=mean * (left - 1))
        log(
            f"{preset}: batch {batches}, {played} games; before it the worst standard error "
            f"was {ratio:.2f} of its target (the batch took {seconds[-1]:.0f} s; the budget "
            f"ends by {later.strftime('%H:%M')} at the latest)"
        )
    return batches


def curves(rows: list[dict]) -> list[dict]:
    """The policies' rows by run and steps (`table`'s rows): Elo against steps per run and
    weights kind, each with its standard error."""
    return sorted(
        (row for row in rows if row["run"] is not None),
        key=lambda row: (row["preset"], row["run"], row["kind"], row["step"]),
    )


# ---- importing `bnb ratings` records


def import_records(
    pool: RatingPool,
    records: list[pathlib.Path],
    default_scoring: Callable[[str], str],
    resolve: Callable[[str], object] | None = None,
) -> dict:
    """Bring `bnb ratings` records (`<preset>.batches.jsonl`) into the pool: each recorded
    policy registered under its pool name (`policy_name` of its recorded path; one whose
    identity the pool holds keeps the pool's name), each batch appended with an id from its
    content, so importing a record twice adds nothing. A record without identities (a legacy
    one) is matched by its paths as `bnb ratings --reuse` does (`resolve`; its players left
    out when no longer found). default_scoring(preset): the scoring a batch that names none
    was fitted under. Returns per record its batches added and the names left out."""
    held, _ = pool.players()
    by_identity = {one.identity: name for name, one in held.items()}
    report: dict[str, dict] = {}
    for record in records:
        preset = record.name.removesuffix(".batches.jsonl")
        existing = {entry["id"] for entry in pool.batches(preset)[0]}
        lines = [line for line in record.read_text().splitlines() if line.strip()]
        if not lines:
            continue
        first = json.loads(lines[0])
        recorded = [row["player"] for row in first["ratings"]]
        if "identities" in first:
            identities, left_out = first["identities"], {}
        elif resolve is not None:
            ended = datetime.datetime.fromisoformat(first["wall_time"])
            identities, left_out = _legacy_identities(recorded, ended, resolve)
        else:
            identities, left_out = {}, dict.fromkeys(recorded, "no identities recorded")
        names: dict[str, str] = {}
        new_players = []
        for name in recorded:
            identity = identities.get(name)
            if identity is None:
                continue
            if identity in by_identity:
                names[name] = by_identity[identity]
                continue
            if not identity.startswith("sha256:"):  # a scripted player: its own name
                names[name] = name
                one = PoolPlayer(name, identity)
            else:
                parsed = policy_name(name)
                if parsed is None:
                    left_out[name] = "not a run's step directory"
                    continue
                pool_name, run, step, kind = parsed
                paths = (name,) if pathlib.Path(name).is_dir() else ()
                one = PoolPlayer(pool_name, identity, run, step, kind, paths)
                names[name] = pool_name
            if one.name in held and held[one.name].identity != one.identity:
                left_out[name] = f"{one.name} is registered with other weights"
                del names[name]
                continue
            by_identity[identity] = one.name
            held[one.name] = one
            new_players.append(one)
        if new_players:
            pool.register(new_players)
        added = 0
        for number, line in enumerate(lines):
            entry = json.loads(line)
            batch_id = "import-" + hashlib.sha256(line.encode()).hexdigest()[:32]
            if batch_id in existing:
                continue
            kept = [
                index
                for index, pair in enumerate(entry["matchups"])
                if all(name in names for name in pair)
            ]
            if not kept:
                continue
            batch = dict(
                id=batch_id,
                game_version=entry.get("game_version", 1),
                matchups=[[names[n] for n in entry["matchups"][index]] for index in kept],
                identities={names[n]: identities[n] for n in names},
                results=[entry["results"][index] for index in kept],
                seconds=entry.get("seconds"),
                wall_time=entry.get("wall_time"),
                source=dict(record=str(record.resolve()), batch=entry.get("batch", number)),
            )
            for column in ("kill_share", "kills", "statistics"):
                if column in entry:
                    batch[column] = [entry[column][index] for index in kept]
            if "kill_share" not in entry:  # the results are of one scoring only
                batch["scoring"] = entry.get("scoring", default_scoring(preset))
            pool.append_batch(preset, batch)
            added += 1
        report[str(record)] = dict(preset=preset, batches=added, left_out=left_out)
    return report
