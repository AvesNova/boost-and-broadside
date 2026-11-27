"""Round 3 (author 2026-11-28 ~03:00): the top of the skill hill and a curvature-shaped persona
sampler. Agent groups are written once as JSON (DATA/agents-GROUP.json: label -> rows over the
defaults), played against a reference (default the round-2 robust centre) at 1v1, 4v4, 16v16
(fields, fog, the study clocks), and reported as Elo (the reference at 1000), counts and style.

    round3.py make-oat                         # one-at-a-time sweeps from the robust centre
    round3.py play GROUP --games 256 256 128 --minutes 10 [--sizes 1 4 16] [--reference JSON]
    round3.py report GROUP OUT.json [--reference JSON]
    round3.py ladder NAME=JSON ... --target-error 0.05 --minutes 10

play: chunks of CHUNK agents per runner call (each chunk's games stored at once: resumable,
`--minutes` bounds an invocation). Store DATA/round3.sqlite (ladder: DATA/ladder.sqlite).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill2  # noqa: E402

from lean.eval import Evaluator, ScriptedAgent, Store  # noqa: E402
from lean.scripted import ScriptedConfig  # noqa: E402
from lean.scripted.parameters import PARAMETERS  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = Path("/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-3")
CENTRE = HERE / "candidates" / "robust-centre.json"
CHUNK = 16

# One-at-a-time levels (box units: value x e^(2 width delta) for log rows).
WIDE = (-2.0, -1.25, -0.5, 0.5, 1.25, 2.0)
WIDE_ROWS = (
    "threat_weight",
    "collision_weight",
    "horizon",
    "temperature",
    "speed_weight",
    "power_temperature",
    "fire_tolerance",
    "power_horizon",
    "preferred_speed",
    "angles_weight",
    "caution",
    "reply_temperature",
)
NARROW = (-0.75, 0.75)
NARROW_ROWS = (
    "energy_weight",
    "aggression",
    "stickiness",
    "store_weight",
    "power_stickiness",
    "disengaged_energy_weight",
)
CROWD_SWEEP = (
    "aggression_crowd",
    "energy_weight_crowd",
    "reply_temperature_crowd",
    "preferred_speed_crowd",
    "speed_weight_crowd",
)
# Hard ends (the code takes any positive value; these test the limits directly).
EXTREMES = {
    "collision-zero": {"collision_weight": 0.0},
    "temperature-0.001": {"temperature": 0.001},
    "horizon-0.1": {"horizon": 0.1},
    "fire-tolerance-20": {"fire_tolerance": 20.0},
}


def log(text):
    print(f"{time.strftime('%H:%M:%S')} {text}", file=sys.stderr, flush=True)


def rows_of(path) -> dict:
    return json.loads(Path(path).read_text())


def agent_of(rows: dict) -> ScriptedAgent:
    return ScriptedAgent.of(ScriptedConfig().replace(**rows))


def moved(rows: dict, name: str, delta: float) -> dict:
    out = dict(rows)
    base = name[: -len("_crowd")] if name.endswith("_crowd") else name
    row = PARAMETERS[base]
    if row.law == "log":
        out[name] = rows[name] * math.exp(2.0 * row.width * delta)
    else:
        out[name] = rows[name] + 2.0 * row.width * delta
    return out


def make_oat():
    centre = rows_of(CENTRE)
    group = {}
    for name in WIDE_ROWS:
        for delta in WIDE:
            group[f"{name}@{delta:+.2f}"] = moved(centre, name, delta)
    for name in NARROW_ROWS:
        for delta in NARROW:
            group[f"{name}@{delta:+.2f}"] = moved(centre, name, delta)
    for name in CROWD_SWEEP:
        for delta in (-1.0, 1.0):
            group[f"{name}@{delta:+.2f}"] = moved(centre, name, delta)
    for label, change in EXTREMES.items():
        group[label] = {**centre, **change}
    write_group("oat", group)


def write_group(name, group):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"agents-{name}.json").write_text(json.dumps(group, indent=1))
    log(f"group {name}: {len(group)} agents")


def read_group(name) -> dict:
    return json.loads((DATA / f"agents-{name}.json").read_text())


def play(group, sizes, games, minutes, reference, envs):
    started = time.perf_counter()
    store = Store(DATA / "round3.sqlite")
    middle = agent_of(rows_of(reference))
    listed = read_group(group)
    for size, count in zip(sizes, games, strict=True):
        # The crowd rows act only in team games: their sweeps skip the duel.
        agents = [
            agent_of(rows)
            for label, rows in listed.items()
            if not (size == 1 and "_crowd@" in label)
        ]
        evaluator = Evaluator(
            store, skill2.config_of(size), envs=envs, minimum_envs=envs, seed=3300 + size
        )
        for start in range(0, len(agents), CHUNK):
            if (time.perf_counter() - started) / 60.0 > minutes:
                log("time bound")
                sys.exit(3)
            chunk = agents[start : start + CHUNK]
            evaluator.ensure([(middle, a, (size, size), count) for a in chunk])
            log(f"{group} {size}v{size}: agents {start + len(chunk)}/{len(agents)}")
    log(f"{group} done")


def report(group, out, reference, store_path=None):
    store = Store(store_path or DATA / "round3.sqlite")
    middle = agent_of(rows_of(reference))
    listed = {"reference": rows_of(reference), **read_group(group)}
    agents = {label: agent_of(rows) for label, rows in listed.items()}
    result = {"rows": listed, "setups": {}}
    for size in skill2.SIZES:
        evaluator = Evaluator(store, skill2.config_of(size))
        sizes = (size, size)
        elo = {i: (r, e) for i, r, e in evaluator.elo(sizes, anchor=middle)}
        conditions, mirrored = evaluator.conditions(sizes)
        styles = {
            row["agent"]: row
            for row in store.styles(conditions, [a.id for a in agents.values()], mirrored)
        }
        entries = {}
        for label, agent in agents.items():
            entry = {"elo": elo.get(agent.id, (None, None))[0]}
            entry["elo_error"] = elo.get(agent.id, (None, None))[1]
            entry["style"] = styles.get(agent.id)
            if label != "reference":
                entry.update(evaluator.counts(middle, agent, sizes))
            entries[label] = entry
        result["setups"][f"{size}v{size}"] = entries
    Path(out).write_text(json.dumps(result, indent=1))
    log(f"wrote {out}")


def ladder(items, sizes, target, minutes, envs):
    started = time.perf_counter()
    store = Store(DATA / "ladder.sqlite")
    agents = {item.split("=", 1)[0]: agent_of(rows_of(item.split("=", 1)[1])) for item in items}
    names = list(agents)
    for size in sizes:
        evaluator = Evaluator(
            store, skill2.config_of(size), envs=envs, minimum_envs=envs, seed=6600 + size
        )
        pairings = [
            (agents[a], agents[b], (size, size)) for i, a in enumerate(names) for b in names[i + 1 :]
        ]
        while True:
            rows = evaluator.adaptive(pairings, target_error=target, budget=8192, round_games=8192)
            worst = max(row["error"] for row in rows)
            log(f"ladder {size}v{size}: worst {worst:.4f}, {sum(r['games'] for r in rows)} games")
            if worst <= target:
                break
            if (time.perf_counter() - started) / 60.0 > minutes:
                log("time bound")
                sys.exit(3)
    log("ladder done")


def ladder_report(items, out):
    store = Store(DATA / "ladder.sqlite")
    agents = {item.split("=", 1)[0]: agent_of(rows_of(item.split("=", 1)[1])) for item in items}
    result = {}
    for size in skill2.SIZES:
        evaluator = Evaluator(store, skill2.config_of(size))
        anchor = agents.get("yardstick")
        elo = {i: (r, e) for i, r, e in evaluator.elo((size, size), anchor=anchor)}
        conditions, mirrored = evaluator.conditions((size, size))
        styles = {row["agent"]: row for row in store.styles(conditions, None, mirrored)}
        result[f"{size}v{size}"] = {
            name: {
                "elo": elo.get(a.id, (None, None))[0],
                "elo_error": elo.get(a.id, (None, None))[1],
                "style": styles.get(a.id),
            }
            for name, a in agents.items()
        }
    Path(out).write_text(json.dumps(result, indent=1))
    log(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", choices=("make-oat", "play", "report", "ladder", "ladder-report")
    )
    parser.add_argument("items", nargs="*")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--games", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--reference", default=str(CENTRE))
    parser.add_argument("--target-error", type=float, default=0.05)
    a = parser.parse_args()
    if a.command == "make-oat":
        make_oat()
    elif a.command == "play":
        games = [dict(zip(skill2.SIZES, a.games, strict=True))[s] for s in a.sizes]
        play(a.items[0], a.sizes, games, a.minutes, a.reference, a.envs)
    elif a.command == "report":
        report(a.items[0], a.items[1], a.reference)
    elif a.command == "ladder":
        ladder(a.items, a.sizes, a.target_error, a.minutes, a.envs)
    else:
        ladder_report(a.items[:-1], a.items[-1])


if __name__ == "__main__":
    main()
