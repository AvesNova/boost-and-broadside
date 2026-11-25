"""Screening of the scripted agent's parameter space (docs/rewrite/scripted-screening.md).

Skill of a configuration: its mean score (wins + draws / 2) over a fixed panel of 8 scripted
agents and 18 conditions (arena radius x team sizes x spawn rule), equal weight per (panel
member, condition) cell. Every game goes into one lean.eval store; each job plays only the
games the store lacks, so every step resumes where it stopped.

    screen.py STORE panel                      # write panel.json (once; seeded)
    screen.py STORE reference --games 32       # the yardstick and BFM, games per cell
    screen.py STORE morris --games 4           # Morris trajectories (SALib), r = 10
    screen.py STORE sobol --count 4000 --batch 2000 --games 2
    screen.py STORE local --spread 0.25 --count 256 --games 2
    screen.py STORE skills                     # skills.json: every configuration's skill

Each runner call is one (arena radius, spawn rule) for a batch of configurations against the
whole panel at three team sizes; the store is written after each call (the durable record).
"""

import argparse
import dataclasses
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from lean import game
from lean.eval import Evaluator, ScriptedAgent, Store
from lean.eval.store import SobolStream, conditions_of
from lean.scripted import ScriptedConfig
from lean.scripted.parameters import PARAMETERS, SAMPLED, UPWARD, box_point

HERE = Path(__file__).resolve().parent
RADII = (3000.0, 4200.0, 6000.0)
SPAWNS = ("mixed", "sides")
SIZES = ((3, 3), (4, 4), (5, 5))
PANEL_SEED = 20261124
MORRIS_SEED = 20261124
MORRIS_TRAJECTORIES = 10
MORRIS_LEVELS = 4
# The tests' design-value BFM persona (lean/scripted/tests/test_tactics.py). Its lead-turn and
# yo-yo rows went with those mechanisms (author 2026-11-25); its graded-defence rows went with
# the trim (scripted-reach.md "Trim"), so on today's table it is the yardstick (the panel
# member itself is read from panel.json).
BFM_ROWS = {"relax_angle": 1.5 * math.pi, "defend_range": 1.0}
BFM = ScriptedConfig().replace(**{k: v for k, v in BFM_ROWS.items() if k in PARAMETERS})


def game_configs():
    return [dataclasses.replace(game.GameConfig(), arena_radius=radius, spawn=spawn)
            for radius in RADII for spawn in SPAWNS]


def agent_at(unit_row, spread: float = 1.0, centre: ScriptedConfig | None = None) -> ScriptedAgent:
    """The scripted agent at a point of the unit cube (one value per SAMPLED row), as
    Evaluator.sobol_candidates maps it."""
    centre = centre or ScriptedConfig()
    drawn = box_point(centre, torch.as_tensor(np.asarray(unit_row), dtype=torch.float64), spread)
    values = dict(centre.values)
    values.update({row: float(drawn[row]) for row in SAMPLED})
    return ScriptedAgent.of(ScriptedConfig(values))


# The yardstick's point of the unit cube: the box centre, or its floor for an upward box.
DEFAULT_UNIT = np.array([0.0 if PARAMETERS[name].law == UPWARD else 0.5 for name in SAMPLED])
SWEEP_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sweep_agents(names) -> list[tuple[str, float, ScriptedAgent]]:
    """One row at a time through the yardstick: (row, unit level, agent) at SWEEP_LEVELS of
    its box, the other rows at their defaults (the default level itself is the yardstick)."""
    out = []
    for name in names:
        for level in SWEEP_LEVELS:
            unit = DEFAULT_UNIT.copy()
            unit[SAMPLED.index(name)] = level
            agent = ScriptedAgent.of() if np.array_equal(unit, DEFAULT_UNIT) else agent_at(unit)
            out.append((name, level, agent))
    return out


def panel_path() -> Path:
    return HERE / "panel.json"


def make_panel() -> None:
    if panel_path().exists():
        print("panel.json exists; kept")
        return
    generator = torch.Generator().manual_seed(PANEL_SEED)
    unit = torch.rand((6, len(SAMPLED)), generator=generator, dtype=torch.float64)
    members = [("yardstick", ScriptedAgent.of()), ("bfm", ScriptedAgent.of(BFM))]
    members += [(f"persona{index + 1}", agent_at(unit[index])) for index in range(6)]
    record = {"seed": PANEL_SEED, "rows": list(PARAMETERS),
              "members": [{"name": name, "id": agent.id, "values": list(agent.values)} for name, agent in members]}
    panel_path().write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {panel_path()}")


def panel() -> list[ScriptedAgent]:
    """The panel's members on today's row table: each member's values by row name, rows no
    longer in the table dropped (the lead-turn and yo-yo rows, author 2026-11-25), rows added
    since at the table's default (ghost_cutoff, fire_worth: scripted-ghosts)."""
    record = json.loads(panel_path().read_text())
    members = []
    for member in record["members"]:
        named = dict(zip(record["rows"], member["values"], strict=True))
        members.append(ScriptedAgent.of(ScriptedConfig({name: named.get(name, PARAMETERS[name].default) for name in PARAMETERS})))
    return members


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {text}", file=sys.stderr, flush=True)


def play(store: Store, candidates: list[ScriptedAgent], games: int, label: str) -> None:
    """Every candidate against every panel member, at every condition, `games` per cell."""
    members = panel()
    configs = game_configs()
    for number, game_config in enumerate(configs):
        call_started = time.perf_counter()
        requests = [(candidate, member, sizes, games) for candidate in candidates for member in members
                    for sizes in SIZES]
        last = [0.0]

        def progress(started, busy, seconds, total=len(requests) * games):
            if seconds - last[0] >= 15.0:
                last[0] = seconds
                log(f"  {label} call {number + 1}/{len(configs)}: {started} of <= {total} games started, "
                    f"{busy} playing, {seconds:.0f} s")

        evaluator = Evaluator(store, game_config, envs=4096, seed=zlib.crc32(label.encode()) % 1000 + 7 * number, progress=progress)
        before = store.total_games()
        evaluator.ensure(requests)
        played = store.total_games() - before
        seconds = time.perf_counter() - call_started
        log(f"{label} call {number + 1}/{len(configs)} radius {game_config.arena_radius:.0f} {game_config.spawn}: "
            f"{played} games in {seconds:.1f} s ({played / max(seconds, 1e-9):.0f} games/s); store {store.total_games()}")


def morris_sample() -> np.ndarray:
    from SALib.sample import morris
    path = HERE / "morris_sample.npy"
    if path.exists():
        return np.load(path)
    problem = {"num_vars": len(SAMPLED), "names": list(SAMPLED), "bounds": [[0.0, 1.0]] * len(SAMPLED)}
    sample = morris.sample(problem, MORRIS_TRAJECTORIES, num_levels=MORRIS_LEVELS, seed=MORRIS_SEED)
    np.save(path, sample)
    return sample


def sobol_points(store: Store, name: str, count: int, seed: int) -> np.ndarray:
    """The stream's first `count` points (drawn into the store's stream when it has fewer):
    the same points on every call, so a job resumes."""
    stream = SobolStream(store, name, len(SAMPLED), seed)
    if stream.used < count:
        stream.next(count - stream.used)
    engine = torch.quasirandom.SobolEngine(len(SAMPLED), scramble=True, seed=stream.seed)
    return engine.draw(count, dtype=torch.float64).numpy()


def all_conditions() -> list[tuple[str, tuple]]:
    return [(conditions_of(game_config, sizes)[0], (game_config.arena_radius, game_config.spawn, sizes))
            for game_config in game_configs() for sizes in SIZES]


def cell_table(store: Store) -> dict:
    """{(candidate id, member id, conditions id): (games, score sum, score-squared sum)} from
    both orientations of the counts view."""
    members = {agent.id for agent in panel()}
    conditions = {identity for identity, _ in all_conditions()}
    table = {}
    rows = store.connection.execute(
        "SELECT conditions, agent_a, agent_b, games, wins, draws FROM counts").fetchall()
    for condition, agent_a, agent_b, games, wins, draws in rows:
        if condition not in conditions:
            continue
        losses = games - wins - draws
        credits = []
        if agent_b in members:
            credits.append((agent_a, agent_b, wins + 0.5 * draws, wins + 0.25 * draws))
        if agent_a in members and agent_a != agent_b:
            credits.append((agent_b, agent_a, losses + 0.5 * draws, losses + 0.25 * draws))
        for candidate, member, score, square in credits:
            key = (candidate, member, condition)
            old = table.get(key, (0, 0.0, 0.0))
            table[key] = (old[0] + games, old[1] + score, old[2] + square)
    return table


def skill_of(table: dict, candidate: str, member_ids, condition_ids) -> tuple | None:
    """(skill, games, its standard error, the mean score per condition over the panel) of a
    candidate with games in every cell; the error from each cell's unbiased game variance."""
    scores, games, variance = [], 0, 0.0
    per_condition = []
    for condition in condition_ids:
        row = []
        for member in member_ids:
            cell = table.get((candidate, member, condition))
            if cell is None or cell[0] < 2:
                return None
            count, total, square = cell
            mean = total / count
            row.append(mean)
            games += count
            # The cell mean's variance: the games' unbiased variance over the cell's games.
            variance += max(square / count - mean * mean, 0.0) / (count - 1)
        scores += row
        per_condition.append(float(np.mean(row)))
    cells = len(scores)
    return float(np.mean(scores)), games, math.sqrt(variance) / cells, per_condition


def skills(store: Store) -> None:
    table = cell_table(store)
    member_ids = [agent.id for agent in panel()]
    condition_ids = [identity for identity, _ in all_conditions()]
    lengths = dict(((row[0], row[1]), row[2]) for row in store.connection.execute(
        "SELECT agent_a, conditions, SUM(games * mean_length) / SUM(games) FROM counts GROUP BY agent_a, conditions"))
    out = {}

    def add(group, agents, points=None):
        entries = []
        for index, agent in enumerate(agents):
            value = skill_of(table, agent.id, member_ids, condition_ids)
            if value is None:
                continue
            length = [lengths.get((agent.id, condition)) for condition in condition_ids]
            length = [item for item in length if item is not None]
            entry = {"id": agent.id, "skill": value[0], "games": value[1], "error": value[2],
                     "by_condition": value[3],
                     "mean_length": float(np.mean(length)) if length else None}
            if points is not None:
                entry["unit"] = [float(item) for item in points[index]]
            entries.append(entry)
        out[group] = entries

    add("reference", [ScriptedAgent.of(), ScriptedAgent.of(BFM)])
    morris = morris_sample()
    add("morris", [agent_at(row) for row in morris], morris)
    for name, record in json.loads((HERE / "streams.json").read_text()).items() if (HERE / "streams.json").exists() else []:
        points = sobol_points(store, name, record["count"], record["seed"])
        add(name, [agent_at(row, record["spread"]) for row in points], points)
    record = HERE / "sweep.json"
    if record.exists():
        swept = sweep_agents(json.loads(record.read_text()))
        add("sweep", [agent for _, _, agent in swept], [[SAMPLED.index(name), level] for name, level, _ in swept])
    # Per-cell skill of the references (for the conditions' table).
    cells = {}
    for name, agent in (("yardstick", ScriptedAgent.of()), ("bfm", ScriptedAgent.of(BFM))):
        rows = []
        for identity, (radius, spawn, sizes) in all_conditions():
            per = [table.get((agent.id, member, identity)) for member in member_ids]
            rows.append({"radius": radius, "spawn": spawn, "sizes": list(sizes),
                         "scores": [None if cell is None else cell[1] / cell[0] for cell in per],
                         "games": [0 if cell is None else cell[0] for cell in per]})
        cells[name] = rows
    out["reference_cells"] = cells
    out["conditions"] = [{"radius": radius, "spawn": spawn, "sizes": list(sizes)}
                         for _, (radius, spawn, sizes) in all_conditions()]
    path = HERE / "skills.json"
    path.write_text(json.dumps(out) + "\n")
    for group, entries in out.items():
        if group not in ("reference_cells", "conditions"):
            errors = [entry["error"] for entry in entries]
            print(f"{group}: {len(entries)} configurations, median error {np.median(errors) if errors else math.nan:.4f}")


def register_stream(name: str, count: int, seed: int, spread: float) -> None:
    path = HERE / "streams.json"
    streams = json.loads(path.read_text()) if path.exists() else {}
    old = streams.get(name)
    if old is not None and (old["seed"] != seed or old["spread"] != spread):
        raise SystemExit(f"stream {name} exists with seed {old['seed']} spread {old['spread']}")
    streams[name] = {"count": max(count, old["count"] if old else 0), "seed": seed, "spread": spread}
    path.write_text(json.dumps(streams, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store")
    parser.add_argument("command", choices=("panel", "reference", "morris", "sobol", "local", "sweep", "skills"))
    parser.add_argument("--rows", nargs="*", default=[], help="sweep: the rows swept one at a time")
    parser.add_argument("--games", type=int, default=2, help="games per (panel member, condition) cell")
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--spread", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    store = Store(arguments.store)
    if arguments.command == "panel":
        make_panel()
    elif arguments.command == "reference":
        play(store, [ScriptedAgent.of(), ScriptedAgent.of(BFM)], arguments.games, "reference")
    elif arguments.command == "morris":
        sample = morris_sample()
        log(f"morris: {len(sample)} configurations")
        play(store, [agent_at(row) for row in sample], arguments.games, "morris")
    elif arguments.command in ("sobol", "local"):
        name = "screen" if arguments.command == "sobol" else f"local{round(arguments.spread * 100)}"
        spread = 1.0 if arguments.command == "sobol" else arguments.spread
        register_stream(name, arguments.count, arguments.seed, spread)
        points = sobol_points(store, name, arguments.count, arguments.seed)
        for start in range(0, arguments.count, arguments.batch):
            chunk = points[start:start + arguments.batch]
            log(f"{name}: configurations {start} to {start + len(chunk) - 1}")
            play(store, [agent_at(row, spread) for row in chunk], arguments.games, f"{name}[{start}]")
    elif arguments.command == "sweep":
        record = HERE / "sweep.json"
        rows = sorted(set(json.loads(record.read_text()) if record.exists() else []) | set(arguments.rows))
        record.write_text(json.dumps(rows) + "\n")
        agents = [agent for _, _, agent in sweep_agents(rows)]
        play(store, list(dict.fromkeys(agents)), arguments.games, "sweep")
    else:
        skills(store)


if __name__ == "__main__":
    main()
