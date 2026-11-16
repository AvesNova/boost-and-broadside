"""probe3: the least-squares combinations of a linear kill-share model fitted to probe1 + probe2
(per mode, on the log factors), with Jujak shield up as the Frontline 10v10-over-5v5 handle, at
0.8, 1 and 1.2 of the fitted step, around E. Raw card factors (a probe). Writes
candidates/probe3.json."""

import copy
import json
import pathlib

HERE = pathlib.Path(__file__).parent
E = next(
    e
    for e in json.loads((HERE / "options.json").read_text())
    if e["name"] == "jt0.95-yt1.12-ys0.55"
)

# name -> {side: {field: factor at the fitted step}}
COMBOS = {
    "Q1": {
        "energy": {
            "max_energy": 1.93,
            "vision_range_scale": 1.48,
            "induced_drag_scale": 1.158,
            "bullet_energy_cost": 0.612,
        },
        "turn": {"max_shield": 1.245},
    },
    "Q2": {
        "energy": {
            "vision_range_scale": 1.44,
            "induced_drag_scale": 1.204,
            "bullet_energy_cost": 0.423,
        },
        "turn": {"max_shield": 1.284},
    },
    "Q3": {
        "energy": {"max_energy": 2.1, "induced_drag_scale": 1.222, "bullet_energy_cost": 0.57},
        "turn": {"max_shield": 1.318, "vision_range_scale": 0.77},
    },
    "Q4": {
        "energy": {"max_energy": 1.69, "induced_drag_scale": 1.056, "bullet_energy_cost": 0.571},
        "turn": {"max_shield": 1.234},
    },
    "Q5": {
        "energy": {"max_energy": 3.7, "vision_range_scale": 1.52, "induced_drag_scale": 1.126},
        "turn": {"max_shield": 1.156},
    },
}
SINGLE = {}


def entry(name: str, sides: dict) -> dict:
    out = copy.deepcopy(E)
    out["name"] = f"p3-{name}"
    out["probe"] = sides
    for side, factors in sides.items():
        card = out["card"][side]
        for field, factor in factors.items():
            if isinstance(card.get(field), dict):
                card[field] = {"set": card[field]["set"] * factor}
            else:
                card[field] = factor
    return out


def main() -> None:
    out = [entry("E", {}), entry("E-again", {})]
    for name, sides in COMBOS.items():
        for scale in (0.8, 1.0, 1.2):
            scaled = {
                side: {field: factor**scale for field, factor in factors.items()}
                for side, factors in sides.items()
            }
            out.append(entry(f"{name}x{scale}", scaled))
    out += [entry(name, sides) for name, sides in SINGLE.items()]
    (HERE / "probe3.json").write_text(json.dumps(out, indent=0) + "\n")


if __name__ == "__main__":
    main()
