"""The flight and combat numbers of candidate cards: per ship the sustained turn (full stick,
settled), the instantaneous peak turn (full stick, over speed), cruise, top and corner speed,
shield, hit damage and gun range.

    PYTHONPATH=src:. uv run --no-sync --active python \
        docs/rewrite/sections/ships-r9m2/flight.py round1/candidates.json [names...]
"""

import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).parent
from bench.game_loop import PRESET, api, base_card, variant_types  # noqa: E402
from bnb.game.handling.characteristics import characteristics  # noqa: E402
from bnb.game.handling.manoeuvres import batched  # noqa: E402
from bnb.game.handling.shown import SPEED_GRID, instant  # noqa: E402


def table(entries: list[dict], names: tuple[str, str] = ("Yongma", "Jujak")) -> str:
    types = variant_types(
        base_card("cpu"),
        [e.get("energy", {}) for e in entries],
        [e.get("turn", {}) for e in entries],
    )
    rules = api.rules_of(api.PRESETS[PRESET], types=types)
    card = characteristics(types, rules)
    many = batched(types)
    corner = torch.sqrt(many.max_acceleration / many.max_lift_coefficient)
    stall = torch.sqrt(rules.flight.gravity / many.max_lift_coefficient)
    fraction = torch.linspace(0.0, 1.0, SPEED_GRID, dtype=corner.dtype)
    speeds = (1.01 * stall)[..., None] + (2.0 * corner - 1.01 * stall)[..., None] * fraction
    peak = instant(rules, many, speeds, 1.0).turn_rate.max(-1).values[0]
    K = len(entries)
    lines = [
        "| Card | Ship | Sustained turn rad/s | Peak turn rad/s | Cruise m/s | Top m/s "
        "| Corner m/s | Shield | Hit damage | Gun range m |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, entry in enumerate(entries):
        for name, row in ((names[0], k), (names[1], K + k)):
            lines.append(
                f"| {entry['name']} | {name} | {float(card.turn_rate[row, 2]):.3f} "
                f"| {float(peak[row]):.3f} | {float(card.cruise_speed[row]):.0f} "
                f"| {float(card.top_speed[row]):.0f} | {float(card.corner_speed[row]):.0f} "
                f"| {float(types.max_shield[row]):.0f} | {float(types.hit_damage[row]):.2f} "
                f"| {float(card.gun_range[row]):.0f} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    entries = json.loads((HERE / sys.argv[1]).read_text())
    if sys.argv[2:]:
        entries = [e for e in entries if e["name"] in sys.argv[2:]]
    seen, unique = set(), []
    for e in entries:  # one row a card
        key = json.dumps([e.get("energy", {}), e.get("turn", {})], sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(dict(e, name=e["name"].split(" | ")[0]))
    print(table(unique))
