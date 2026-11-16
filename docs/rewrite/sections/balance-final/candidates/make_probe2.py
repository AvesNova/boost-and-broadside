"""probe2: combinations of the probe1 fields that the kill-share profiles say should hold every
mode near even (a least-squares fit over probe1's dm5, dm10, el5, FL5, FL10 profiles), each at
0.7, 1 and 1.3 of the fitted step, around E. Raw card factors (a probe). Writes
candidates/probe2.json."""

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
    "store-vision-turn": {
        "energy": {"max_energy": 1.53, "vision_range_scale": 1.374, "induced_drag_scale": 1.266}
    },
    "store-shot-Jrange": {
        "energy": {"max_energy": 1.53, "bullet_energy_cost": 0.79},
        "turn": {"bullet_life": 1.21},
    },
    "shot-vision-Jrange": {
        "energy": {"bullet_energy_cost": 0.60, "vision_range_scale": 1.17},
        "turn": {"bullet_life": 1.355},
    },
}
SINGLE = {
    "shot-vision-turn": {
        "energy": {
            "bullet_energy_cost": 0.858,
            "vision_range_scale": 1.712,
            "induced_drag_scale": 1.40,
        }
    },
    "shot-Jrange": {"energy": {"bullet_energy_cost": 0.61}, "turn": {"bullet_life": 1.285}},
    "Y-store1.6": {"energy": {"max_energy": 1.6}},
    "Y-store2": {"energy": {"max_energy": 2.0}},
    "Y-vision1.5": {"energy": {"vision_range_scale": 1.5}},
    "Y-induced1.27": {"energy": {"induced_drag_scale": 1.27}},
}


def entry(name: str, sides: dict) -> dict:
    out = copy.deepcopy(E)
    out["name"] = f"p2-{name}"
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
        for scale in (0.7, 1.0, 1.3):
            scaled = {
                side: {field: factor**scale for field, factor in factors.items()}
                for side, factors in sides.items()
            }
            out.append(entry(f"{name}x{scale}", scaled))
    out += [entry(name, sides) for name, sides in SINGLE.items()]
    (HERE / "probe2.json").write_text(json.dumps(out, indent=0) + "\n")


if __name__ == "__main__":
    main()
