"""`bnb unified SPEC --out DIR`: the unified evaluator's run (docs/guides/unified-evaluation.md).

The spec (TOML) names the players and the targets:

    lineup_types = ["Yongma", "Jujak"]   # each game's lineup, every slot one of these
    buckets = [5, 10, 20]                # the engines' sizes
    seed = 0

    [players]                            # "scripted" (Imugi, the 1000 anchor) is always in
    champion = "RUN/policies/step_N/exponential_average"
    early = "RUN/policies/step_M/exponential_average"

    [[elo]]                              # ratings of a field in one condition
    name = "deathmatch_5v5"
    preset = "real_scale_deathmatch"
    players = ["scripted", "champion", "early"]   # the anchor first
    stderr = 60.0                        # Elo, every player
    contrasts = [["champion", "early", 30.0]]
    pairs = "all"                        # or "anchor": each player against the anchor only
    lineup = [5, 5]                      # optional; arena_scale = 1.0 optional

    [[win_rate]]                         # a cell's score (team sizes)
    preset = "real_scale_elimination"
    first = "champion"
    second = "scripted"
    lineups = [[1, 1], [2, 3], [2, 4]]   # one target per lineup
    stderr = 0.05

    [[break_even]]
    preset = "real_scale_frontline"
    policy = "champion"
    opponent = "scripted"
    stderr = 0.1
    budget = 2048

    [[decision]]                         # A against B, the sequential test
    preset = "real_scale_deathmatch"
    first = "champion"
    second = "early"
    delta = 0.02                         # looks = 4, alpha = 0.05, beta = 0.2 by default

Writes DIR/games.jsonl (every finished game, flushed every `FLUSH_SECONDS`; a rerun with
the same DIR resumes from it), DIR/progress.log (a line per `PROGRESS_SECONDS` with the
games, games per hour, each bucket's environments and the ETA as a clock time),
DIR/report.json (every target's estimate, rewritten with each progress line) and
DIR/sizing.json (each bucket's measured footprint and chosen size).
"""

import copy
import dataclasses
import datetime
import json
import math
import pathlib
import subprocess
import time
import tomllib

import torch

from bnb.game import api
from bnb.game.modes.rules import PRESETS
from bnb.league.matches import Player, player
from bnb.league.sequential import SequentialDesign

from .cells import BUCKETS, Cell, Condition
from .engine import (
    MEMORY_SHARE,
    BucketEngine,
    GameLog,
    bucket_cells,
    footprint,
    free_memory,
    scaled_presets,
    size_environments,
)
from .planner import Planner
from .targets import BreakEvenTarget, DecisionTarget, EloTarget, Target, WinRateTarget

ANCHOR = "scripted"
FLUSH_SECONDS = 10.0
PROGRESS_SECONDS = 30.0
MINIMUM_CONCURRENT = 512  # environments each bucket needs to share the device


@dataclasses.dataclass
class Spec:
    players: list[Player]
    targets: list[Target]
    lineup_types: tuple[int, ...]
    buckets: tuple[int, ...]
    seed: int
    raw: dict


def load_spec(path: pathlib.Path, runs_directory: pathlib.Path | None = None) -> Spec:
    """The players and targets a spec names (the module docstring)."""
    raw = tomllib.loads(pathlib.Path(path).read_text())
    players = [dataclasses.replace(player("imugi"), name=ANCHOR)]
    for name, where in raw.get("players", {}).items():
        if name == ANCHOR:
            continue
        players.append(dataclasses.replace(player(where, runs_directory), name=name))
    known = {one.name for one in players}

    def check(*names):
        unknown = [n for n in names if n not in known]
        if unknown:
            raise ValueError(f"{path}: players {unknown} are not in [players]")

    targets: list[Target] = []
    for entry in raw.get("elo", []):
        field = entry["players"]
        check(*field)
        if field[0] != ANCHOR:
            field = [ANCHOR] + [one for one in field if one != ANCHOR]
        first, second = entry.get("lineup", [5, 5])
        condition = Condition(entry["preset"], first, second, entry.get("arena_scale", 1.0))
        pairs = None
        if entry.get("pairs", "all") == "anchor":
            pairs = [(ANCHOR, one) for one in field[1:]]
        contrasts = tuple(tuple(c) for c in entry.get("contrasts", ()))
        check(*(n for c in contrasts for n in c[:2]))
        targets.append(
            EloTarget(
                entry.get("name", f"elo/{condition.key}"),
                condition,
                field,
                entry.get("stderr", 60.0),
                contrasts,
                pairs,
                kill_share=entry.get("kill_share", False),
                budget=entry.get("budget", 1 << 30),
            )
        )
    for entry in raw.get("win_rate", []):
        check(entry["first"], entry["second"])
        for first, second in entry.get("lineups", [entry.get("lineup", [5, 5])]):
            condition = Condition(entry["preset"], first, second, entry.get("arena_scale", 1.0))
            cell = Cell(condition, entry["first"], entry["second"])
            targets.append(
                WinRateTarget(
                    f"{entry.get('name', 'win_rate')}/{cell.key}",
                    cell,
                    entry.get("stderr", 0.05),
                    budget=entry.get("budget", 1 << 30),
                )
            )
    for entry in raw.get("break_even", []):
        check(entry["policy"], entry.get("opponent", ANCHOR))
        targets.append(
            BreakEvenTarget(
                entry.get("name", f"break_even/{entry['preset']}/{entry['policy']}"),
                entry["preset"],
                entry["policy"],
                entry.get("opponent", ANCHOR),
                entry.get("stderr", 0.1),
                entry.get("fixed_count", 20),
                entry.get("budget", 2048),
            )
        )
    for entry in raw.get("decision", []):
        check(entry["first"], entry["second"])
        first, second = entry.get("lineup", [5, 5])
        cell = Cell(Condition(entry["preset"], first, second), entry["first"], entry["second"])
        design = SequentialDesign(
            entry.get("looks", 4),
            entry.get("alpha", 0.05),
            entry.get("beta", 0.2),
            entry.get("delta", 0.02),
        )
        targets.append(DecisionTarget(entry.get("name", f"decision/{cell.key}"), cell, design))
    if not targets:
        raise ValueError(f"{path}: no targets")
    presets = {_preset(target) for target in targets}
    ship_sets = {PRESETS[preset].ship_set for preset in presets}
    if len(ship_sets) != 1:
        raise ValueError(f"{path}: the presets play several ship sets {ship_sets}: one card")
    ship_set = ship_sets.pop()
    names = api.type_names(ship_set)
    lineup_types = tuple(names.index(n) for n in raw.get("lineup_types", ["Yongma", "Jujak"]))
    return Spec(
        players,
        targets,
        lineup_types,
        tuple(raw.get("buckets", BUCKETS)),
        raw.get("seed", 0),
        raw,
    )


def _preset(target: Target) -> str:
    if isinstance(target, BreakEvenTarget):
        return target.preset
    if isinstance(target, EloTarget):
        return target.condition.preset
    return target.cell.condition.preset


def gpu_utilisation() -> float | None:
    """The device's utilisation now, in percent (nvidia-smi; None without it)."""
    try:
        found = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return float(found.stdout.split()[0])
    except OSError, ValueError, IndexError, subprocess.SubprocessError:
        return None


def clock(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    when = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=seconds)
    return when.strftime("%H:%M")


def information_fraction(planner: Planner) -> float:
    """The mean over targets of the information held over the information their goals need
    (1 / (1 + remaining), 0 for a target with no estimate yet)."""
    remaining = planner.remaining().values()
    return sum(1.0 / (1.0 + r) if math.isfinite(r) else 0.0 for r in remaining) / max(
        len(remaining), 1
    )


def network_shares(planner: Planner, players: list[Player], groups: list) -> list[float]:
    """Each network group's expected share of a batch's rows: its players' seats in the
    proposed cells, weighted by value."""
    count = 1 + max((g for g in groups if g is not None), default=-1)
    held = [0.0] * count
    total = 0.0
    index = {one.name: n for n, one in enumerate(players)}
    for cell, value in planner.values().items():
        weight = min(value, 1.0)
        for name in (cell.first, cell.second):
            total += weight
            group = groups[index[name]]
            if group is not None:
                held[group] += weight
    return [h / total if total else 1.0 for h in held]


def run(
    spec: Spec,
    out: pathlib.Path,
    device: torch.device,
    environments: int | None = None,
    concurrent: str = "auto",
    memory_share: float = MEMORY_SHARE,
    compiled: bool = True,
    hours: float = 24.0,
    log=print,
) -> dict:
    """The run (the module docstring). environments: each bucket's batch (None: sized from
    the free memory); concurrent: "auto" (the buckets with work share the device when each
    gets `MINIMUM_CONCURRENT` environments, else they run in turn), "yes" or "no"."""
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.log"

    def say(text: str) -> None:
        line = f"{datetime.datetime.now():%F %H:%M:%S} {text}"
        log(line)
        with progress_path.open("a") as file:
            file.write(line + "\n")

    planner = Planner(spec.targets, spec.buckets)
    games_log = GameLog(out)
    resumed = games_log.replay()
    for cell, outcome in resumed:
        planner.finished(cell, outcome, pending=False)
    if resumed:
        say(f"resumed {len(resumed)} games from {games_log.path}")
    from .engine import seat_players

    _, groups = seat_players(spec.players)
    started = time.time()
    total_games = len(resumed)
    sizing = {}
    history: list[tuple[float, float]] = []

    def make(bucket, shares, count, compiled_engine=True, own_planner=None) -> BucketEngine:
        return BucketEngine(
            bucket,
            count,
            shares,
            spec.players,
            planner if own_planner is None else own_planner,
            device,
            spec.lineup_types,
            scaled_presets(planner, bucket),
            network_share=network_shares(planner, spec.players, groups),
            compiled=compiled_engine,
            seed=spec.seed,
            record=games_log.add,
        )

    footprints: dict[int, tuple[float, float]] = {}

    def probe(bucket: int, shares: dict[str, float]) -> tuple[float, float]:
        """The bucket's (fixed, per-environment) bytes, measured once per run."""
        if bucket not in footprints:
            footprints[bucket] = footprint(
                lambda n: make(bucket, shares, n, False, copy.deepcopy(planner)), device
            )
        return footprints[bucket]

    def build(bucket: int, shares: dict[str, float], share_of_free: float) -> BucketEngine:
        count = environments
        if count is None:
            fixed, per = probe(bucket, shares)
            free = free_memory(device)
            count = size_environments(fixed, per, int(free * share_of_free), memory_share)
            sizing[bucket] = dict(fixed=fixed, per_environment=per, free=free, environments=count)
            (out / "sizing.json").write_text(json.dumps(sizing, indent=1))
            say(
                f"bucket {bucket}: {per / 1e6:.2f} MB an environment + {fixed / 1e9:.2f} GB, "
                f"{free / 1e9:.1f} GB free x {share_of_free:.2f}: {count} environments"
            )
        engine = make(bucket, shares, count)
        built = time.time()
        engine.start()
        engine.step()
        torch.cuda.synchronize(device)
        say(
            f"bucket {bucket}: {engine.environments} environments {shares}, capacity "
            f"{engine.capacity}, warm-up {time.time() - built:.0f} s"
        )
        return engine

    def shared(order: list[int], demand: dict, weight: dict) -> bool:
        """Whether the buckets share the device (`concurrent`)."""
        if concurrent in ("yes", "no") or len(order) == 1:
            return concurrent == "yes" and len(order) > 1
        if environments is not None:
            return False
        free, total = free_memory(device), sum(weight.values())
        for bucket in order:
            fixed, per = probe(bucket, demand[bucket])
            share = weight[bucket] / total
            if size_environments(fixed, per, int(free * share), memory_share) < MINIMUM_CONCURRENT:
                return False
        return True

    finished_by = started + hours * 3600.0
    while not planner.done() and time.time() < finished_by:
        demand = bucket_cells(planner)
        if not demand:
            break
        order = sorted(demand, key=lambda b: -sum(demand[b].values()))
        weight = {b: sum(demand[b].values()) for b in order}
        together = shared(order, demand, weight)
        running: list[BucketEngine] = []
        if together:
            total = sum(weight.values())
            for bucket in order:
                running.append(build(bucket, demand[bucket], weight[bucket] / total))
        else:
            running.append(build(order[0], demand[order[0]], 1.0))
        last_flush = last_progress = time.time()
        games_at_progress, time_at_progress = total_games, time.time()
        while running:
            for engine in list(running):
                total_games += engine.step()
            now = time.time()
            if now - last_flush >= FLUSH_SECONDS:
                games_log.flush()
                last_flush = now
            if now - last_progress >= PROGRESS_SECONDS:
                fraction = information_fraction(planner)
                history.append((now, fraction))
                history[:] = history[-20:]
                eta = None
                if len(history) >= 2 and history[-1][1] > history[0][1]:
                    rate = (history[-1][1] - history[0][1]) / (history[-1][0] - history[0][0])
                    eta = (1.0 - fraction) / rate
                rate_hour = (total_games - games_at_progress) / (now - time_at_progress) * 3600
                games_at_progress, time_at_progress = total_games, now
                utilisation = gpu_utilisation()
                say(
                    f"{total_games} games ({rate_hour:,.0f}/h), information {fraction:.3f}, "
                    f"GPU {utilisation}%, buckets "
                    + ", ".join(f"{e.bucket}:{e.environments}" for e in running)
                    + f", ETA {clock(eta)}"
                )
                write_report(out, planner, total_games, started)
                last_progress = now
                for engine in list(running):
                    if not engine.wanted():
                        engine.drain()
                        running.remove(engine)
                        say(f"bucket {engine.bucket}: no target wants its games; stopped")
                if planner.done() or now >= finished_by:
                    break
        for engine in running:
            total_games += engine.drain()
        running.clear()
        torch.cuda.empty_cache()
        games_log.flush()
    games_log.flush()
    report = write_report(out, planner, total_games, started)
    say(f"finished: {total_games} games in {(time.time() - started) / 3600:.2f} h")
    return report


def write_report(out: pathlib.Path, planner: Planner, games: int, started: float) -> dict:
    report = dict(
        games=games,
        hours=(time.time() - started) / 3600.0,
        done=planner.done(),
        targets={target.name: target.report() for target in planner.targets},
    )
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report
