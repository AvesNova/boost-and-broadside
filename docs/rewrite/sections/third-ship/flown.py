"""The flown characteristics of the release cards and candidates, side by side
(`bnb.game.handling.characteristics`, the game's own flights; CPU, ~12 s a batch).

    PYTHONPATH=$PWD/src:$PWD .venv/bin/python docs/rewrite/sections/third-ship/flown.py \
        candidates/screen4.json [NAME ...] [--jujak-card candidates/jujak-v23-card.json]
"""

import argparse
import dataclasses
import json
import math
import pathlib
import tomllib

import torch

from bnb.game import api, ships
from bnb.game.handling.characteristics import characteristics
from bnb.game.ship_types import ShipTypes

SET = "real-scale"


def release_card(key: str, changes: dict | None = None) -> dict:
    card = tomllib.loads((ships.directory_of(SET) / f"{key}.toml").read_text())["card"]
    for name, change in (changes or {}).items():
        card[name] = change["set"] if isinstance(change, dict) else card[name] * change
    return card


def profile(cards: dict[str, dict]) -> dict[str, dict]:
    names = list(cards)
    fields = [f.name for f in dataclasses.fields(ShipTypes)]
    types = ShipTypes(**{f: torch.tensor([float(cards[n][f]) for n in names]) for f in fields})
    rules = api.rules_of(api.PRESETS["real_scale_deathmatch"], types=types)
    c = characteristics(types, rules)
    out = {}
    for i, n in enumerate(names):
        card = cards[n]
        dps = card["hit_damage"] * card["fire_rate"]
        out[n] = {
            "cruise": float(c.cruise_speed[i]),
            "top": float(c.top_speed[i]),
            "corner": float(c.corner_speed[i]),
            "turn deg/s (fine, sustained, max)": [round(math.degrees(float(x)), 1) for x in c.turn_rate[i]],
            "gun range": float(c.gun_range[i]),
            "bullet speed": float(c.bullet_speed[i]),
            "climb power": float(c.climb_power[i]),
            "store / climb power (s)": float(c.max_energy[i] / c.climb_power[i]),
            "shots per store": float(c.max_energy[i] / c.bullet_energy_cost[i]),
            "dps": dps,
            "damage per bullet": card["hit_damage"],
            "fire rate": card["fire_rate"],
            "shield": card["max_shield"],
            "shield / dps (s)": card["max_shield"] / dps,
            "max lift coefficient": card["max_lift_coefficient"],
            "zero lift drag": card["zero_lift_drag"],
            "explosive": card["explosive_share"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=pathlib.Path, nargs="?")
    parser.add_argument("names", nargs="*")
    parser.add_argument("--jujak-card", type=pathlib.Path)
    args = parser.parse_args()
    jujak = json.loads(args.jujak_card.read_text()) if args.jujak_card else {}
    cards = {"Yongma": release_card("yongma"), "Jujak": release_card("jujak", jujak)}
    if args.candidates:
        for entry in json.loads(args.candidates.read_text()):
            if not args.names or entry["name"] in args.names:
                cards[entry["name"]] = entry["card"]
    table = profile(cards)
    names = list(table)
    keys = list(table[names[0]])
    print("| | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for key in keys:
        cells = []
        for n in names:
            v = table[n][key]
            cells.append(str(v) if isinstance(v, list) else f"{v:.4g}")
        print(f"| {key} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
