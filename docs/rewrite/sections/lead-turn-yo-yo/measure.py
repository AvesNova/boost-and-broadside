"""Re-measure the lead-turn and yo-yo sweeps with the screening's skill measure
(docs/rewrite/lead-turn-yo-yo-review.md; the measure: docs/rewrite/scripted-screening.md).

    measure.py STORE play     # the two rows at the sweep's five levels, 8 games a cell
    measure.py STORE skills   # sweep.json here: each point's skill, games and error

STORE must be a fresh store: an agent's identity is its parameter vector, not the code, so a
store played before the fix holds the old behaviour under the same identities (the panel's
BFM member and personas fly both features). The yardstick's behaviour is unchanged by the
fix, so its row here is comparable with the screening's.
"""

import json
import sys
from pathlib import Path

import lean
from lean.eval import Store
from lean.scripted.parameters import SAMPLED

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripted-screening"))
import screen  # noqa: E402

ROWS = ("lead_turn_scale", "yo_yo_time")


def main() -> None:
    store = Store(sys.argv[1])
    print(f"lean from {Path(lean.__file__).parent}", flush=True)
    swept = screen.sweep_agents(ROWS)
    if sys.argv[2] == "play":
        agents = list(dict.fromkeys(agent for _, _, agent in swept))
        screen.play(store, agents, 8, "leadyoyo-sweep")
        return
    table = screen.cell_table(store)
    members = [agent.id for agent in screen.panel()]
    conditions = [identity for identity, _ in screen.all_conditions()]
    out = {}
    for name, level, agent in swept:
        value = screen.skill_of(table, agent.id, members, conditions)
        if value is None:
            continue
        row_value = agent.values[list(screen.PARAMETERS).index(name)]
        point = {"level": level, "value": row_value, "skill": value[0], "games": value[1],
                 "error": value[2]}
        out.setdefault(name, []).append(point)
    (HERE / "sweep.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, points in out.items():
        cells = (f"{p['value']:.3f}: {p['skill']:.3f}+-{p['error']:.3f}" for p in points)
        print(name, " ".join(cells))
    assert set(SAMPLED) >= set(ROWS)


if __name__ == "__main__":
    main()
