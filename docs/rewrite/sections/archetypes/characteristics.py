"""The flight and combat characteristics of card variants (`sizes.py`'s factors): each
ship's sustained turn (full stick), cruise, top and corner speed, shield, hit damage and gun
range, to check what a balance does to the archetypes.

    PYTHONPATH=src:. uv run --no-sync --active python \
        docs/rewrite/sections/archetypes/characteristics.py round5/candidates.json:2 ...
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
from bench.game_loop import PRESET, api, base_card, variant_types  # noqa: E402
from bnb.game.handling.characteristics import characteristics  # noqa: E402

entries = [{"name": "frozen", "energy": {}, "turn": {}}]
for argument in sys.argv[1:]:
    path, index = argument.split(":")
    entries.append(json.loads((HERE / path).read_text())[int(index)])
types = variant_types(
    base_card("cpu"), [e["energy"] for e in entries], [e["turn"] for e in entries]
)
card = characteristics(types, api.rules_of(api.PRESETS[PRESET], types=types))
K = len(entries)
print(
    "| Card | Ship | Sustained turn rad/s | Cruise m/s | Top m/s | Corner m/s | Shield "
    "| Hit damage | Gun range m |"
)
print("|---|---|---|---|---|---|---|---|---|")
for k, entry in enumerate(entries):
    for name, row in (("Yongma", k), ("Jujak", K + k)):
        print(
            f"| {entry['name']} | {name} | {float(card.turn_rate[row, 2]):.3f} "
            f"| {float(card.cruise_speed[row]):.0f} | {float(card.top_speed[row]):.0f} "
            f"| {float(card.corner_speed[row]):.0f} | {float(types.max_shield[row]):.0f} "
            f"| {float(types.hit_damage[row]):.2f} | {float(card.gun_range[row]):.0f} |"
        )
