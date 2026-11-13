"""Game v13's balance card round: single levers on Yongma (energy) or Jujak (turn), both
sides flying r9m2, so the author can pick with data. Card factors as
`bench.game_loop.variant_types` (a field times the factor). Levers: Yongma's lift (the duel
lever), Yongma's power and drag (the large-map lever), Yongma's hit damage, Jujak's shield.

    uv run --no-sync python docs/rewrite/sections/lethality/cards.py round3
"""

import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("make", HERE / "make.py")
make = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make)
R9M = make.R9M
R9M2 = make.gates(1.0, 0.4)

CARDS = {
    "round3": [
        ("base", {}, {}),
        ("Y lift x1.1", {"max_lift_coefficient": 1.1}, {}),
        ("Y lift x1.2", {"max_lift_coefficient": 1.2}, {}),
        ("Y lift x1.3", {"max_lift_coefficient": 1.3}, {}),
        ("Y power x1.15", {"engine_power": 1.15, "engine_power_full_store": 1.15}, {}),
        ("Y power x1.3", {"engine_power": 1.3, "engine_power_full_store": 1.3}, {}),
        ("Y drag x0.8", {"zero_lift_drag": 0.8}, {}),
        ("Y damage x1.25", {"hit_damage": 1.25}, {}),
        ("Y damage x1.5", {"hit_damage": 1.5}, {}),
        ("J shield x0.8", {}, {"max_shield": 0.8}),
        ("J shield x0.65", {}, {"max_shield": 0.65}),
    ],
    # round 3's best per lever (none reached 0.45 at 1v1) and one step further each
    "round4": [
        ("base", {}, {}),
        ("Y lift x1.3", {"max_lift_coefficient": 1.3}, {}),
        ("Y lift x1.5", {"max_lift_coefficient": 1.5}, {}),
        ("Y power x1.3", {"engine_power": 1.3, "engine_power_full_store": 1.3}, {}),
        ("Y power x1.5", {"engine_power": 1.5, "engine_power_full_store": 1.5}, {}),
        ("Y damage x1.5", {"hit_damage": 1.5}, {}),
        ("Y damage x2", {"hit_damage": 2.0}, {}),
        ("J shield x0.65", {}, {"max_shield": 0.65}),
        ("J shield x0.5", {}, {"max_shield": 0.5}),
    ],
}


def main() -> None:
    name = sys.argv[1]
    out = [
        dict(
            name=label,
            energy=energy,
            turn=turn,
            doctrine=dict(energy=dict(R9M, **R9M2[0]), turn=dict(R9M, **R9M2[1])),
        )
        for label, energy, turn in CARDS[name]
    ]
    path = HERE / name / "candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} candidates -> {path}")


if __name__ == "__main__":
    main()
