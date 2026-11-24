"""Confirmation of the CMA-ES round 2 (docs/rewrite/scripted-screening.md "CMA-ES round 2").

Three parameter sets on the current code, in a store of their own (the panel's BFM-design
member and personas fly differently since the lead-turn/yo-yo fix, so older games do not
apply): BFM-default (the code's defaults), BFM-CMA1 (cma_candidate.json) and BFM-CMA2
(cma_candidate_round2.json, round 2's final mean).

    round2.py STORE confirm [--games 64]   # the three, every panel cell, many games
    round2.py STORE local                  # local25 around CMA1 and CMA2 (screening's 256 points)
    round2.py STORE held-out [--games 32]  # the three against round 1's 8 held-out personas
    round2.py STORE styles                 # lean.eval styles over the games against the panel
    round2.py STORE divergence --data DIR  # Jensen-Shannon divergence from BFM-default (states.pt)
    round2.py STORE all --data DIR         # all of the above in order

Each step plays only the games the store lacks and writes its result into
cma_candidate_round2.json under its own key (the durable record).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screen  # noqa: E402
import search  # noqa: E402

from lean.eval import Evaluator, ScriptedAgent, Store  # noqa: E402
from lean.eval.store import conditions_of  # noqa: E402
from lean.eval.style import MEASURES  # noqa: E402
from lean.scripted import ScriptedConfig  # noqa: E402
from lean.scripted.parameters import SAMPLED, vector  # noqa: E402

OUTPUT = search.OUTPUT_ROUND2
NAMES = ("BFM-default", "BFM-CMA1", "BFM-CMA2")


def agents() -> dict[str, ScriptedAgent]:
    first = json.loads(search.OUTPUT.read_text())["values"]
    second = json.loads(OUTPUT.read_text())["values"]
    return {"BFM-default": ScriptedAgent.of(), "BFM-CMA1": ScriptedAgent.of(ScriptedConfig(first)),
            "BFM-CMA2": ScriptedAgent.of(ScriptedConfig(second))}


def update(key: str, value) -> None:
    record = json.loads(OUTPUT.read_text())
    record[key] = value
    OUTPUT.write_text(json.dumps(record, indent=1) + "\n")


def confirm(store: Store, games: int) -> None:
    named = agents()
    screen.play(store, list(named.values()), games, "round2-confirm")
    out = {}
    for name, agent in named.items():
        value, error, per_condition, played = search.skill(store, agent)
        out[name] = {"skill": value, "error": error, "games": played, "by_condition": per_condition,
                     "marginals": search.marginals(per_condition)}
        screen.log(f"{name}: {value:.4f} +- {error:.4f} ({played} games); {out[name]['marginals']}")
    # Each against BFM-default as a panel member (its head to head, over the 18 conditions).
    default = named["BFM-default"]
    for name, agent in named.items():
        scores, variance, games_played = [], 0.0, 0
        for game_config in screen.game_configs():
            for sizes in screen.SIZES:
                count = store.counts(agent.id, default.id, conditions_of(game_config, sizes)[0])
                n = count["games"]
                mean = (count["wins"] + 0.5 * count["draws"]) / n
                variance += max((count["wins"] + 0.25 * count["draws"]) / n - mean * mean, 0.0) / (n - 1)
                scores.append(mean)
                games_played += n
        out[name]["against_default"] = {"score": float(np.mean(scores)), "games": games_played,
                                        "error": math.sqrt(variance) / len(scores)}
    update("confirm", out)


def local(store: Store) -> None:
    named = agents()
    points = screen.sobol_points(store, "local25", 256, 0)
    centres = {name: named[name] for name in ("BFM-CMA1", "BFM-CMA2")}
    around = {name: [screen.agent_at(row, 0.25, agent.config()) for row in points] for name, agent in centres.items()}
    screen.play(store, [agent for group in around.values() for agent in group], 2, "round2-local25")
    record = json.loads(OUTPUT.read_text())
    reference = {name: record["confirm"][name]["skill"] for name in NAMES}
    out = {}
    for name, group in around.items():
        local_skills = [search.skill(store, agent) for agent in group]
        values = np.array([item[0] for item in local_skills])
        out[name] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                     "error_of_mean": float(values.std(ddof=1) / math.sqrt(len(values))),
                     "median_noise": float(np.median([item[1] for item in local_skills])),
                     "quantile5": float(np.quantile(values, 0.05)),
                     "centre_minus_mean": reference[name] - float(values.mean()),
                     "share_above": {other: float(np.mean(values > reference[other])) for other in NAMES},
                     "skills": values.tolist()}
        screen.log(f"{name} local25: mean {values.mean():.4f} sd {values.std(ddof=1):.4f} "
                   f"q5 {np.quantile(values, 0.05):.4f}; share above {out[name]['share_above']}")
    paired = np.array(out["BFM-CMA2"]["skills"]) - np.array(out["BFM-CMA1"]["skills"])
    out["paired_CMA2_minus_CMA1"] = {"mean": float(paired.mean()),
                                     "error": float(paired.std(ddof=1) / math.sqrt(len(paired)))}
    screen.log(f"local25 paired CMA2 - CMA1: {out['paired_CMA2_minus_CMA1']}")
    update("local25", out)


def held_out(store: Store, games: int) -> None:
    """The three against round 1's 8 held-out personas (search.HELD_OUT_SEED), at the 18
    conditions, `games` a cell."""
    named = agents()
    unit = torch.rand((8, len(SAMPLED)), generator=torch.Generator().manual_seed(search.HELD_OUT_SEED),
                      dtype=torch.float64)
    members = [screen.agent_at(row) for row in unit]
    for number, game_config in enumerate(screen.game_configs()):
        requests = [(agent, member, sizes, games) for agent in named.values() for member in members
                    for sizes in screen.SIZES]
        Evaluator(store, game_config, envs=4096, seed=900 + 7 * number).ensure(requests)
        screen.log(f"held-out call {number + 1}/6; store {store.total_games()}")
    out = {"seed": search.HELD_OUT_SEED, "games_per_cell": games}
    for name, agent in named.items():
        scores, variance, per_member = [], 0.0, []
        for member in members:
            row = []
            for game_config in screen.game_configs():
                for sizes in screen.SIZES:
                    count = store.counts(agent.id, member.id, conditions_of(game_config, sizes)[0])
                    n = count["games"]
                    mean = (count["wins"] + 0.5 * count["draws"]) / n
                    variance += max((count["wins"] + 0.25 * count["draws"]) / n - mean * mean, 0.0) / (n - 1)
                    row.append(mean)
            scores += row
            per_member.append(float(np.mean(row)))
        out[name] = {"skill": float(np.mean(scores)), "error": math.sqrt(variance) / len(scores),
                     "per_member": per_member}
        screen.log(f"{name} held-out: {out[name]['skill']:.4f} +- {out[name]['error']:.4f}; "
                   f"per member {[round(item, 3) for item in per_member]}")
    update("held_out", out)


def styles(store: Store) -> None:
    """Each set's lean.eval style over its games against the panel (both seats), per
    condition as the store's `styles` view weights them, then the mean over the 18
    conditions (equal weight). Also the mean game length."""
    named = agents()
    panel_ids = [member.id for member in screen.panel()]
    marks = ",".join("?" * len(panel_ids))
    out = {}
    for name, agent in named.items():
        per_condition = []
        for condition, _ in screen.all_conditions():
            rows = []
            for seat, other in (("a", "b"), ("b", "a")):
                columns = ", ".join(f"{seat}_{measure}" for measure in MEASURES)
                rows += store.connection.execute(
                    f"SELECT {columns}, length FROM games WHERE conditions = ? AND agent_{seat} = ? "
                    f"AND agent_{other} IN ({marks}) AND {seat}_minutes IS NOT NULL",
                    (condition, agent.id, *panel_ids)).fetchall()
            table = {measure: [row[index] for row in rows] for index, measure in enumerate(MEASURES)}
            minutes = np.array(table["minutes"], dtype=float)
            entry = {"games": len(rows), "length": float(np.mean([row[-1] for row in rows]))}
            for measure in MEASURES:
                if measure in ("minutes", "two_circle", "hits", "hit_range"):
                    continue
                entry[measure] = float(np.sum(np.array(table[measure], dtype=float) * minutes) / minutes.sum())
            turning = np.array(table["turning"], dtype=float) * minutes
            two = [(value, weight) for value, weight in zip(table["two_circle"], turning, strict=True) if value is not None]
            entry["two_circle"] = float(sum(v * w for v, w in two) / sum(w for _, w in two))
            hits = np.array(table["hits"], dtype=float)
            ranged = [(value, weight) for value, weight in zip(table["hit_range"], hits, strict=True) if value is not None]
            entry["hit_range"] = float(sum(v * w for v, w in ranged) / sum(w for _, w in ranged))
            entry["hits_per_minute"] = float(hits.sum() / minutes.sum())
            per_condition.append(entry)
        keys = [key for key in per_condition[0] if key != "games"]
        out[name] = {key: float(np.mean([entry[key] for entry in per_condition])) for key in keys}
        out[name]["games"] = int(sum(entry["games"] for entry in per_condition))
        screen.log(f"{name} styles: " + " ".join(f"{key}={value:.3f}" for key, value in out[name].items()))
    update("styles", out)


def divergence(data: Path) -> None:
    """Jensen-Shannon divergence (nats) of each set's action distributions from BFM-default's
    on the fixed state set (divergence.py), per head, with the distribution-style shares."""
    import divergence as module

    named = agents()
    states = module.StateSet(data)
    reference = states.marginals(torch.tensor([vector(ScriptedConfig())]))[0]
    vectors = torch.tensor([list(agent.values) for agent in named.values()]
                           + [list(vector(module.BFM))], dtype=torch.float32)
    result = module.divergence_from(states, reference, vectors, 4)
    out = {"ship_states": states.count, "snapshots": states.snapshots}
    for index, name in enumerate([*named, "BFM-design"]):
        out[name] = {key: float(value[index]) for key, value in result.items()}
        screen.log(f"{name} divergence: " + " ".join(f"{key}={value:.4f}" for key, value in out[name].items()))
    update("divergence", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store")
    parser.add_argument("command", choices=("confirm", "local", "held-out", "styles", "divergence", "all"))
    parser.add_argument("--games", type=int, default=None, help="games per cell (confirm 64, held-out 32)")
    parser.add_argument("--data", type=Path, help="the folder with states.pt (divergence)")
    arguments = parser.parse_args()
    store = Store(arguments.store)
    command = arguments.command
    if command in ("confirm", "all"):
        confirm(store, arguments.games or 64)
    if command in ("local", "all"):
        local(store)
    if command in ("held-out", "all"):
        held_out(store, arguments.games if command == "held-out" and arguments.games else 32)
    if command in ("styles", "all"):
        styles(store)
    if command in ("divergence", "all"):
        divergence(arguments.data)


if __name__ == "__main__":
    main()
