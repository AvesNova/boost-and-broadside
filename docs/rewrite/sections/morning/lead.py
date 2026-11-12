"""Item 3: r9m2 with maneuvers' two-circle lead turn at 0.1 / 0.2 / 0.3 / 0.5 s before the
merge, as maneuvers-2 `sizes.py` candidates on balance-d2, the frozen card (the ship files)
and arch-standard-hit-and-run, r9m2 beside them as the same-run control. Writes
`lead/candidates.json`.

    python docs/rewrite/sections/morning/lead.py

The lead turn is maneuvers' round-3 setting (`maneuvers/round3/candidates.json`, "lead 0.5
s"): on a head-on approach with the merge t s away, full stick toward the bandit's passing
side (two-circle), on both slots; the time gate's scale is round 3's 0.1 s, or t / 2 below
0.2 s (so the gate stays a gate at 0.1 s). Every other doctrine field and scripted parameter is
r9m2's (`maneuvers-2/champion2.json`), the map cards as `r9m2-maps/make.py` builds them.
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "r9m2-maps"))

from make import card  # noqa: E402

ROOT = HERE.parents[3]
MAPS = ("balance-d2", "frozen", "arch-standard-hit-and-run")  # frozen: the ship files
SECONDS = (0.1, 0.2, 0.3, 0.5)


def lead(seconds: float) -> dict:
    return {
        "lead_turn": 1.0,
        "lead_turn_in_seconds": [seconds, min(0.1, seconds / 2)],
        "lead_turn_in_crossing": [0.5, 0.2],
        "lead_turn_stick": 1.0,
        "lead_turn_one_circle": 0.0,
    }


def main() -> None:
    champion = json.loads((HERE.parent / "maneuvers-2" / "champion2.json").read_text())
    parameters = {str(i): v for i, v in enumerate(champion["parameters"])}
    doctrines = {"r9m2": dict(energy=champion["yongma"], turn=champion["jujak"])}
    for t in SECONDS:
        doctrines[f"r9m2 + lead {t} s"] = dict(
            energy=dict(champion["yongma"], **lead(t)), turn=dict(champion["jujak"], **lead(t))
        )
    out = []
    for name in MAPS:
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
    path = HERE / "lead" / "candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} candidates -> {path}")


if __name__ == "__main__":
    main()
