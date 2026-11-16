"""probe4: probe3's cards (Q1-Q5 at 0.8 / 1 / 1.2 of the step) with the raw Yongma induced-drag
factor replaced by a Yongma sustained_turn_rate handle: the turn each raw card measures
(`measure_cards.py candidates/probe3.json`), re-solved with every other flight handle held
(so Yongma's low-speed acceleration stays at its value instead of falling 3-15%). E's handles
(Jujak turn x0.95, Yongma shield x0.55) as before; the game knobs (Yongma store, shot cost,
vision) and Jujak's shield/vision stay raw card factors on top of the solved card.

    python candidates/make_probe4.py spec    # writes candidates/probe4-handles-spec.json
    (handle_cards.py probe4-handles-spec.json probe4-handles.json, on the box)
    python candidates/make_probe4.py merge   # writes candidates/probe4.json
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
PROBE3 = {e["name"]: e for e in json.loads((HERE / "probe3.json").read_text())}
# Yongma's sustained turn of each raw probe3 card over before's (measure_cards.py, 2026-11-16)
MEASURED_TURN = {
    "Q1x0.8": 1.099, "Q1x1.0": 1.094, "Q1x1.2": 1.088,
    "Q2x0.8": 1.093, "Q2x1.0": 1.086, "Q2x1.2": 1.080,
    "Q3x0.8": 1.091, "Q3x1.0": 1.084, "Q3x1.2": 1.076,
    "Q4x0.8": 1.112, "Q4x1.0": 1.110, "Q4x1.2": 1.108,
    "Q5x0.8": 1.103, "Q5x1.0": 1.099, "Q5x1.2": 1.094,
}  # fmt: skip
RAW = {"energy": ("max_energy", "vision_range_scale", "bullet_energy_cost"),
       "turn": ("max_shield", "vision_range_scale")}  # fmt: skip


def spec() -> None:
    out = [{"name": "p4-E", "energy_handles": {"sustained_turn_rate": 1.12, "max_shield": 0.55},
            "turn_handles": {"sustained_turn_rate": 0.95}}]  # fmt: skip
    for name, turn in MEASURED_TURN.items():
        out.append(
            {
                "name": f"p4-{name}h",
                "energy_handles": {"sustained_turn_rate": turn, "max_shield": 0.55},
                "turn_handles": {"sustained_turn_rate": 0.95},
            }
        )
    (HERE / "probe4-handles-spec.json").write_text(json.dumps(out, indent=1) + "\n")


def merge() -> None:
    solved = json.loads((HERE / "probe4-handles.json").read_text())
    out = []
    for entry in solved:
        source = PROBE3["p3-" + entry["name"][3:].removesuffix("h")]
        for side, fields in RAW.items():
            for field in fields:
                factor = source["probe"].get(side, {}).get(field)
                if factor is not None:
                    entry["card"][side][field] = factor
        entry["probe"] = source["probe"]
        out.append(entry)
    e_again = json.loads(json.dumps(out[0]))
    e_again["name"] = "p4-E-again"
    out.insert(1, e_again)
    (HERE / "probe4.json").write_text(json.dumps(out, indent=0) + "\n")


if __name__ == "__main__":
    {"spec": spec, "merge": merge}[sys.argv[1]]()
