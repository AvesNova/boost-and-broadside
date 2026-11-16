"""confirm1: the best probe3 line (Q4: Yongma store, shot cost, a little less Yongma turn,
Jujak shield up) as handle cards (probe4's form) at 0.7 / 0.8 / 1.0 of the step, with E, for
the 1024-2048-game checks in every mode. Q4x0.7's Yongma turn handle is interpolated from the
measured 0.8 / 1.0 cards (1.112, 1.110 -> 1.113); its card is solved by handle_cards.py from
confirm1-handles-spec.json. Writes candidates/confirm1.json."""

import copy
import json
import pathlib

HERE = pathlib.Path(__file__).parent
PROBE4 = {e["name"]: e for e in json.loads((HERE / "probe4.json").read_text())}
Q4 = {
    "energy": {"max_energy": 1.69, "bullet_energy_cost": 0.571},
    "turn": {"max_shield": 1.234},
}  # make_probe3.py's Q4 without the raw induced drag (the turn is the handle)


def main() -> None:
    out = []
    for name, new in (("p4-E", "g-E"), ("p4-Q4x0.8h", "g-Q4x0.8h"), ("p4-Q4x1.0h", "g-Q4x1.0h")):
        entry = copy.deepcopy(PROBE4[name])
        entry["name"] = new
        out.append(entry)
    solved = json.loads((HERE / "confirm1-handles.json").read_text())[0]
    for side, factors in Q4.items():
        for field, factor in factors.items():
            solved["card"][side][field] = factor**0.7
    solved["probe"] = {s: {f: v**0.7 for f, v in fs.items()} for s, fs in Q4.items()}
    out.insert(1, solved)
    (HERE / "confirm1.json").write_text(json.dumps(out, indent=0) + "\n")


if __name__ == "__main__":
    main()
