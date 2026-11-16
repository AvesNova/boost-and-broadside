"""The handles of candidate cards (sweep.py's form: before/'s card with the candidate's
"card" factors or sets), measured by flying them (`api.measure_handles`), CPU. Prints each
candidate's chosen handles for both ships, and the ratio to the before card's.

    PYTHONPATH=src:. uv run --no-sync python docs/rewrite/sections/balance-final/measure_cards.py \
        candidates/probe3.json [--names p3-Q1x1.0 ...]
"""

import argparse
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

from sweep import before_card  # noqa: E402

from bench.game_loop import base_card, variant_types  # noqa: E402
from bnb.game import api  # noqa: E402

SHOWN = (
    "sustained_turn_rate",
    "low_speed_acceleration",
    "climb_time",
    "cruise_speed",
    "ceiling",
    "shots_per_store",
    "vision_range",
    "max_shield",
    "hit_damage",
    "fire_rate",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=pathlib.Path)
    parser.add_argument("--names", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(2)
    entries = json.loads(args.candidates.read_text())
    if args.names:
        entries = [e for e in entries if e["name"] in args.names]
    base = before_card(base_card("cpu"))
    rules = api.rules_of(
        api.PRESETS["real_scale_deathmatch"], types=api.ship_types(ship_set="real-scale")
    )
    energy = [{}] + [e.get("card", {}).get("energy", {}) for e in entries]
    turn = [{}] + [e.get("card", {}).get("turn", {}) for e in entries]
    types = variant_types(base, energy, turn)
    count = len(energy)
    handles = api.measure_handles(types, rules)
    names = ["before"] + [e["name"] for e in entries]
    for side, offset in (("Yongma", 0), ("Jujak", count)):
        print(f"\n{side}: before values, then ratios to before")
        print("| card | " + " | ".join(SHOWN) + " |")
        print("|---|" + "---|" * len(SHOWN))
        for row, name in enumerate(names):
            cells = []
            for field in SHOWN:
                value = float(getattr(handles, field)[offset + row])
                origin = float(getattr(handles, field)[offset])
                cells.append(f"{value:.4g}" if row == 0 else f"{value / origin:.3f}")
            print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
