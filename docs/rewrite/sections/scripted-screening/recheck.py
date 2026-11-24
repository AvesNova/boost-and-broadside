"""Re-check of BFM-CMA2 after the stick, engine and S2 changes (docs/rewrite/scripted-screening.md
"Re-check after the stick, engine and S2 changes (2026-11-25)").

The fixed stick thirds, no flaps, the 2,200 hp P-51D and the deleted lead turn and yo-yos
change every game, and the row table lost the deleted rows, so every agent id is new: a fresh
store. Three sets on today's code, the screening's skill (screen.py: the panel of 8, on
today's rows, and the 18 conditions, equal weight per cell):

- BFM-CMA2: today's defaults;
- BFM-default: BFM_DEFAULT_2026_11_24 over the defaults (the panel's yardstick member);
- BFM-CMA1: cma_candidate.json's values, minus the deleted rows.

    recheck.py STORE [--games 64]

Plays only the games the store lacks (screen.play writes after each runner call) and writes
recheck.json beside this file (the durable record).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screen  # noqa: E402

from lean.eval import ScriptedAgent, Store  # noqa: E402
from lean.eval.store import conditions_of  # noqa: E402
from lean.scripted import ScriptedConfig  # noqa: E402
from lean.scripted.parameters import BFM_DEFAULT_2026_11_24, PARAMETERS  # noqa: E402

OUTPUT = HERE / "recheck.json"


def agents() -> dict[str, ScriptedAgent]:
    first = json.loads((HERE / "cma_candidate.json").read_text())["values"]
    return {
        "BFM-CMA2": ScriptedAgent.of(),
        "BFM-default": ScriptedAgent.of(ScriptedConfig().replace(**BFM_DEFAULT_2026_11_24)),
        "BFM-CMA1": ScriptedAgent.of(ScriptedConfig({name: first[name] for name in PARAMETERS})),
    }


def skill(store: Store, agent: ScriptedAgent) -> tuple[float, float, list[float], int]:
    """(skill, its standard error, the mean score per condition, games): search.py's skill
    (search.py itself names the deleted rows, so it no longer imports)."""
    members = screen.panel()
    scores, variance, per_condition, games = [], 0.0, [], 0
    for game_config in screen.game_configs():
        for sizes in screen.SIZES:
            condition = conditions_of(game_config, sizes)[0]
            row = []
            for member in members:
                count = store.counts(agent.id, member.id, condition)
                n = count["games"]
                mean = (count["wins"] + 0.5 * count["draws"]) / n
                variance += max((count["wins"] + 0.25 * count["draws"]) / n - mean * mean, 0.0) / (
                    n - 1
                )
                row.append(mean)
                games += n
            scores += row
            per_condition.append(float(np.mean(row)))
    return float(np.mean(scores)), math.sqrt(variance) / len(scores), per_condition, games


def marginals(per_condition: list[float]) -> dict:
    """Mean score by radius, spawn rule and team size (the 18 conditions in screen's order)."""
    conditions = [
        (game_config.arena_radius, game_config.spawn, sizes)
        for game_config in screen.game_configs()
        for sizes in screen.SIZES
    ]
    out = {}
    for key, position in (("radius", 0), ("spawn", 1), ("sizes", 2)):
        levels = dict.fromkeys(str(condition[position]) for condition in conditions)
        out[key] = {
            level: float(
                np.mean(
                    [
                        score
                        for condition, score in zip(conditions, per_condition, strict=True)
                        if str(condition[position]) == level
                    ]
                )
            )
            for level in levels
        }
    return out


def head_to_head(store: Store, agent: ScriptedAgent, other: ScriptedAgent) -> dict:
    """Mean score of `agent` against `other` over the 18 conditions, equal weight, with its
    standard error from each condition's unbiased game variance."""
    scores, variance, games = [], 0.0, 0
    for game_config in screen.game_configs():
        for sizes in screen.SIZES:
            count = store.counts(agent.id, other.id, conditions_of(game_config, sizes)[0])
            n = count["games"]
            mean = (count["wins"] + 0.5 * count["draws"]) / n
            variance += max((count["wins"] + 0.25 * count["draws"]) / n - mean * mean, 0.0) / (
                n - 1
            )
            scores.append(mean)
            games += n
    return {
        "score": float(np.mean(scores)),
        "error": math.sqrt(variance) / len(scores),
        "games": games,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("store")
    parser.add_argument(
        "--games", type=int, default=64, help="games per (panel member, condition) cell"
    )
    arguments = parser.parse_args()
    store = Store(arguments.store)
    named = agents()
    panel_ids = [member.id for member in screen.panel()]
    if named["BFM-default"].id != panel_ids[0]:
        raise SystemExit("BFM-default is not the panel's yardstick member")
    screen.play(store, list(named.values()), arguments.games, "recheck")
    out = {
        "games_per_cell": arguments.games,
        "ids": {name: agent.id for name, agent in named.items()},
    }
    for name, agent in named.items():
        value, error, per_condition, played = skill(store, agent)
        out[name] = {
            "skill": value,
            "error": error,
            "games": played,
            "by_condition": per_condition,
            "marginals": marginals(per_condition),
        }
        screen.log(f"{name}: {value:.4f} +- {error:.4f} ({played} games); {out[name]['marginals']}")
    for name in ("BFM-CMA2", "BFM-CMA1"):
        out[name]["against_default"] = head_to_head(store, named[name], named["BFM-default"])
        screen.log(f"{name} v BFM-default: {out[name]['against_default']}")
    OUTPUT.write_text(json.dumps(out, indent=1) + "\n")
    screen.log(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
