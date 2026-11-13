"""r9m2 with the two-circle lead turn on the Yongma only and only in a lone duel, at t = 0.2 /
0.3 / 0.5 s before the merge (the Jujak flies plain r9m2), as maneuvers-2 `sizes.py`
candidates on balance-d2, the frozen card (the ship files) and armour-tank, r9m2 beside
them as the same-run control. Writes `candidates.json` (or `--out`).

    python docs/rewrite/sections/yongma-lead/make.py [--seconds 0.3] [--out confirm.json]

The lead turn is morning item 3's (`morning/lead.py`: full stick toward the bandit's passing
side, crossing gate 0.5 rad, time gate scale 0.1 s). The lone-duel gate is the new
`lead_turn_in_crowd` at r9m2's own `sense_in_crowd` (0.5, 0.1): the soft count of the other
known enemies inside the sense's crowd radius (`sense_crowd_in_gun_range`, 3 gun ranges)
below 1/2.
"""

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "r9m2-maps"))

from make import card  # noqa: E402

ROOT = HERE.parents[3]
MAPS = ("balance-d2", "frozen", "armour-tank")  # frozen: the ship files


def lead(seconds: float, crowd: list[float]) -> dict:
    return {
        "lead_turn": 1.0,
        "lead_turn_in_seconds": [seconds, min(0.1, seconds / 2)],
        "lead_turn_in_crossing": [0.5, 0.2],
        "lead_turn_stick": 1.0,
        "lead_turn_one_circle": 0.0,
        "lead_turn_in_crowd": crowd,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    parser.add_argument("--maps", nargs="+", default=list(MAPS))
    parser.add_argument("--out", default="candidates.json")
    args = parser.parse_args()
    champion = json.loads((HERE.parent / "maneuvers-2" / "champion2.json").read_text())
    parameters = {str(i): v for i, v in enumerate(champion["parameters"])}
    crowd = champion["yongma"]["sense_in_crowd"]
    doctrines = {"r9m2": dict(energy=champion["yongma"], turn=champion["jujak"])}
    for t in args.seconds:
        doctrines[f"r9m2 + Yongma lone lead {t} s"] = dict(
            energy=dict(champion["yongma"], **lead(t, crowd)), turn=champion["jujak"]
        )
    out = []
    for name in args.maps:
        if name == "frozen":
            energy, turn = {}, {}
        else:
            data = json.loads((ROOT / "src/bnb/game/maps" / f"{name}.json").read_text())
            energy, turn = card(data["ships"], "yongma"), card(data["ships"], "jujak")
        for label, doctrine in doctrines.items():
            out.append(
                dict(
                    name=f"{name} | {label}",
                    energy=energy,
                    turn=turn,
                    doctrine=doctrine,
                    parameters=parameters,
                )
            )
    path = HERE / args.out
    path.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} candidates -> {path}")


if __name__ == "__main__":
    main()
