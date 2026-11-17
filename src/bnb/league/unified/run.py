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
    prior_keys = ["deathmatch_5v5"]      # optional: a --prior pool's preset keys for it
    note = "..."                         # optional: a caveat carried into report.json
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
    policy_count = 5                     # optional: the policy at 5 ships, the opponent
    fixed_count = 20                     #   1 to 20 (5v5, 5v6, ...); default: weaker side at 20

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

from .cells import BUCKETS, Cell, Condition, Outcome
from .engine import (
    COMPILED_FACTOR,
    MEMORY_SHARE,
    RESIDENTS,
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
    """A loaded spec: the players, the targets, the lineup types, the buckets and the seed."""

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
                neighbours=field_neighbours([p for p in players if p.name in field]),
                prior_keys=tuple(entry.get("prior_keys", ())),
                note=entry.get("note", ""),
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
                policy_count=entry.get("policy_count"),
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


def field_neighbours(players: list[Player]) -> list[tuple[str, str]]:
    """The planning prior's neighbour pairs of a field (`EloTarget`): a cross of checkpoints
    near in training. Along a run, each checkpoint and the next one by steps (a policy's
    run and steps from its path, `bnb.league.rating_pool.policy_name`; players at one run
    and step, a champion and its snapshot, linked too). Across runs, each checkpoint and the
    other run's checkpoint at the nearest steps, except between two stages of one chain
    (`<chain>-bc` and `<chain>-rl`: RL starts from BC's last weights, so its first
    checkpoint is linked to BC's last). Players without a run (the anchor) have none. Only
    a starting point: the fit's estimates take over as games come in."""
    from bnb.league.rating_pool import policy_name

    runs: dict[str, list[tuple[int, str]]] = {}
    for one in players:
        parsed = policy_name(one.policy) if one.policy is not None else None
        if parsed is not None:
            _, run, step, _ = parsed
            runs.setdefault(run, []).append((step, one.name))
    pairs: list[tuple[str, str]] = []
    for held in runs.values():
        held.sort()
        pairs += [(a, b) for (_, a), (_, b) in zip(held, held[1:])]

    def chain(run: str) -> tuple[str, str] | None:
        for stage in ("-bc", "-rl"):
            if run.endswith(stage):
                return run[: -len(stage)], stage
        return None

    names = sorted(runs)
    for number, first in enumerate(names):
        for second in names[number + 1 :]:
            a, b = chain(first), chain(second)
            if a is not None and b is not None and a[0] == b[0] and a[1] != b[1]:
                bc, rl = (first, second) if a[1] == "-bc" else (second, first)
                pairs.append((runs[bc][-1][1], runs[rl][0][1]))
                continue
            for step, name in runs[first]:
                nearest = min(runs[second], key=lambda other: abs(other[0] - step))
                pairs.append((name, nearest[1]))
    return pairs


def load_prior(planner: Planner, pools: list[pathlib.Path], players: list[Player], say=print):
    """Feed every Elo target the games a rating pool already holds (`bnb.league.rating_pool`
    format: pool.json, players/, games/<preset>/) of its condition (`EloTarget.prior_keys`):
    the fits and the plan start from everything known. A game counts when both players are
    the spec's (matched by identity: a policy's weights hash, so names may differ), it is
    of this game version, and the pool plays the spec's lineup types at the target's map
    scale; everything left out is reported. Returns each target's prior games."""
    from bnb.game import VERSION
    from bnb.league.rating_pool import RatingPool
    from bnb.league.tournament import identity, recorded_results

    names = {identity(one): one.name for one in players}
    loaded: dict[str, int] = {}
    for root in pools:
        pool = RatingPool(root)
        settings = pool.settings
        for target in planner.targets:
            if not isinstance(target, EloTarget):
                continue
            scale = settings.get("arena_scale") or 1.0
            if abs(scale - target.condition.arena_scale) > 1e-9 or settings.get("lineup"):
                say(f"prior {root}: other map scale or a fixed lineup; not used for {target.name}")
                continue
            scoring = (
                "kill_share" if target.kill_share else "win_loss",
                settings.get("deathmatch_score", "win_loss"),
            )
            skipped: dict[str, int] = {}
            counted = 0
            for key in target.prior_keys:
                batches, damaged = pool.batches(key)
                if damaged:
                    skipped["damaged lines"] = skipped.get("damaged lines", 0) + damaged
                for entry in batches:
                    blocks = len(entry["matchups"])
                    if entry.get("game_version", 1) != VERSION:
                        version = entry.get("game_version", 1)
                        reason = f"game version {version} (this tree {VERSION})"
                        skipped[reason] = skipped.get(reason, 0) + blocks
                        continue
                    results = recorded_results(entry, scoring)
                    if results is None:
                        skipped["other scoring"] = skipped.get("other scoring", 0) + blocks
                        continue
                    held = entry.get("identities", {})
                    for (first, second), (won, tied, lost) in zip(
                        entry["matchups"], results, strict=True
                    ):
                        a, b = names.get(held.get(first)), names.get(held.get(second))
                        if a is None or b is None:
                            skipped["a player not in the spec"] = (
                                skipped.get("a player not in the spec", 0) + 1
                            )
                            continue
                        cell = Cell(target.condition, a, b)
                        games = round(won + tied + lost)
                        if target.kill_share:  # a block's summed share, spread evenly
                            outcomes = [Outcome(won / games)] * games if games else []
                        else:
                            outcomes = (
                                [Outcome(1.0)] * round(won)
                                + [Outcome(0.5)] * round(tied)
                                + [Outcome(0.0)] * round(lost)
                            )
                        for outcome in outcomes:
                            target.record(cell, outcome, pending=False)
                        counted += len(outcomes)
            target.prior_games += counted
            loaded[target.name] = loaded.get(target.name, 0) + counted
            say(
                f"prior {root}: {counted} games for {target.name}"
                + (f", left out (blocks): {skipped}" if skipped else "")
            )
    planner._touched_all()
    return loaded


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
    """A clock time `seconds` from now, as HH:MM ("unknown" without one)."""
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    when = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=seconds)
    return when.strftime("%H:%M")


def information_fraction(planner: Planner) -> float:
    """The mean over targets of the work done: the information held over what the goal
    needs (1 / (1 + remaining), 0 with no estimate yet), or the games over the budget,
    whichever is more."""
    fractions = []
    for target in planner.targets:
        remaining = target.remaining()
        fraction = 1.0 / (1.0 + remaining) if math.isfinite(remaining) else 0.0
        games = getattr(target, "games", None)
        if isinstance(target, EloTarget):
            games = target.finished
        budget = getattr(target, "budget", None)
        if isinstance(games, int | float) and budget:
            fraction = max(fraction, min(games / budget, 1.0))
        fractions.append(fraction)
    return sum(fractions) / max(len(fractions), 1)


def player_shares(planner: Planner) -> dict[str, float]:
    """Each player's expected share of a batch's seats: its seats in the proposed cells,
    weighted by value (each at most 1), the lanes' sizing (`engine.lane_runs`)."""
    held: dict[str, float] = {}
    total = 0.0
    for cell, value in planner.values().items():
        weight = min(value, 1.0)
        for name in (cell.first, cell.second):
            held[name] = held.get(name, 0.0) + weight
            total += weight
    return {name: h / total for name, h in held.items()} if total else {}


SIZES = "unified-sizes.json"  # under `bnb.cache.ROOT`: each machine's measured sizes


def machine_key(device, bucket: int, shares: dict, share_of_free: float, memory_share: float):
    """A bucket's sizing key on this machine: the GPU, its memory, the bucket, its slices
    and its share of the memory."""
    properties = torch.cuda.get_device_properties(device)
    slices = ",".join(sorted(shares))
    return (
        f"{properties.name}|{properties.total_memory}|{bucket}|{slices}|"
        f"{share_of_free:.3f}|{memory_share:.3f}"
    )


def remembered_size(key: str) -> int | None:
    """The size a previous run measured for this key (the compiled engine's footprint), so
    a machine plays the same shapes run after run (one compile per bucket per machine)."""
    from bnb.cache import ROOT

    try:
        return json.loads((ROOT / SIZES).read_text()).get(key)
    except OSError, ValueError:
        return None


def remember_size(key: str, count: int) -> None:
    """Store a bucket's measured size for this machine (`remembered_size`)."""
    from bnb.cache import ROOT

    try:
        held = json.loads((ROOT / SIZES).read_text())
    except OSError, ValueError:
        held = {}
    held[key] = count
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / SIZES).write_text(json.dumps(held, indent=1))


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
    pool: pathlib.Path | None = None,
    prior: tuple[pathlib.Path, ...] = (),
    residents: int | None = RESIDENTS,
) -> dict:
    """The run (the module docstring). environments: each bucket's batch (None: sized from
    the free memory); concurrent: "auto" (the buckets with work share the device when each
    gets `MINIMUM_CONCURRENT` environments, else they run in turn), "yes" or "no"; pool: a
    rating pool directory the Elo targets' games are also appended to (`GameLog.export_to`;
    created with the spec's lineup types when new); prior: rating pools whose games of the
    Elo targets' conditions are loaded first (`load_prior`); residents: the network players
    a group seats at once when the field has more (`engine.RESIDENTS`; None: every one)."""
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.log"

    def say(text: str) -> None:
        line = f"{datetime.datetime.now():%F %H:%M:%S} {text}"
        log(line, flush=True)
        with progress_path.open("a") as file:
            file.write(line + "\n")

    planner = Planner(spec.targets, spec.buckets)
    if prior:
        load_prior(planner, list(prior), spec.players, say)
    games_log = GameLog(out)
    resumed = games_log.replay()
    for cell, outcome in resumed:
        planner.finished(cell, outcome, pending=False)
    if resumed:
        say(f"resumed {len(resumed)} games from {games_log.path}")
    if pool is not None:
        from bnb.league.rating_pool import RatingPool

        names = api.type_names(PRESETS[_preset(spec.targets[0])].ship_set)
        rating_pool = RatingPool.create(
            pool, lineup_types=tuple(names[i] for i in sorted(set(spec.lineup_types)))
        )
        elo = {t.condition.key for t in spec.targets if isinstance(t, EloTarget)}
        games_log.export_to(rating_pool, spec.players, elo)
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
            player_share=player_shares(planner) or None,
            compiled=compiled_engine,
            seed=spec.seed,
            record=games_log.add,
            residents=residents,
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
        key = machine_key(device, bucket, shares, share_of_free, memory_share)
        if count is None:
            count = remembered_size(key)
            if count is not None:
                say(f"bucket {bucket}: {count} environments, this machine's measured size")
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
        torch.cuda.reset_peak_memory_stats(device)
        engine = make(bucket, shares, count)
        built = time.time()
        engine.start()
        engine.step()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        record = sizing.setdefault(bucket, dict(environments=count))
        record.update(
            peak_allocated=peak,
            peak_reserved=torch.cuda.max_memory_reserved(device),
            compiled_per_environment=peak / count,
        )
        (out / "sizing.json").write_text(json.dumps(sizing, indent=1))
        if environments is None:  # the next run's size: the compiled footprint measured
            reserved = torch.cuda.max_memory_reserved(device)
            free = free_memory(device) + reserved
            remember_size(
                key,
                size_environments(
                    0.0, reserved / count / COMPILED_FACTOR, int(free * share_of_free), memory_share
                ),
            )
        say(
            f"bucket {bucket}: {engine.environments} environments {shares}, capacity "
            f"{[(len(m), c) for _, m, c in engine.runs]} (players, rows per lane), "
            f"warm-up {time.time() - built:.0f} s, peak {peak / 1e9:.2f} GB allocated"
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
                time_at_progress_before = time_at_progress
                games_at_progress, time_at_progress = total_games, now
                utilisation = gpu_utilisation()
                host = sum(e.host_seconds for e in running)
                for engine in running:
                    engine.host_seconds = 0.0
                parked = sum(e.parked for e in running)
                say(
                    f"{total_games} games ({rate_hour:,.0f}/h), information {fraction:.3f}, "
                    f"GPU {utilisation}%, "
                    f"host planning {host / (now - time_at_progress_before):.0%}, "
                    f"parked {parked}, swaps {sum(e.swaps for e in running)}, buckets "
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
    for line in short_of_goal(planner, out_of_time=time.time() >= finished_by):
        say(line)
    return report


def short_of_goal(planner: Planner, out_of_time: bool) -> list[str]:
    """Loud lines for every Elo target a safety bound stopped short of its goals (its game
    budget, or the run's hours): a stop on a bound is never silent."""
    lines = []
    for target in planner.targets:
        if not isinstance(target, EloTarget) or target.met():
            continue
        why = (
            "its game budget"
            if target.finished >= target.budget
            else ("the run's hours" if out_of_time else "no game it could play")
        )
        report = target.report()
        unmet = report["unmet"]
        status = {n: s for n, s in report["status"].items() if s != "rated" and n != ANCHOR}
        lines.append(
            f"WARNING {target.name}: stopped by {why} after {target.finished} games with "
            f"{len(unmet)} goals unmet ({', '.join(unmet[:8])}{', ...' if len(unmet) > 8 else ''})"
            + (f"; without a rating: {status}" if status else "")
        )
    if out_of_time and not planner.done():
        lines.append("WARNING the run's hours ran out before every target was done")
    return lines


def write_report(out: pathlib.Path, planner: Planner, games: int, started: float) -> dict:
    """Write report.json: the games, the hours and every target's report; returns it."""
    report = dict(
        games=games,
        hours=(time.time() - started) / 3600.0,
        done=planner.done(),
        targets={target.name: target.report() for target in planner.targets},
    )
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report
