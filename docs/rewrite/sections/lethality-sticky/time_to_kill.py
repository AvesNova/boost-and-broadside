"""Time to kill for every real-scale pairing, from the cards alone.

Definition (the shooter's sustained fire on the target, every bullet hitting):
  time to kill = target max_shield / (shooter hit_damage x tail aspect x fire_rate)
with each hit a core hit (closest approach inside the target's 5 m core, closeness 1) by a
bullet at the shooter's reference speed (the card's hit_damage holds at the rules'
damage_reference_distance, 300 m, so the kinetic factor is 1), and the target's head-on
armour at its tail aspect (theta = 180 degrees: 1 - (1 - f0) exp(-ln2 (pi / half)^p)). The
recharge plays no part: every hit restarts its wait. The discrete count is also printed:
the hit that empties the shield only breaks it and the next hit kills, so
hits = ceil(shield / damage) + 1, the first and last (hits - 1) / fire_rate apart.

Usage (from the repository root, PYTHONPATH=$PWD/src):
  uv run --no-sync python docs/rewrite/sections/lethality-sticky/time_to_kill.py
  [--apply SCALE]  multiply both ships' max_shield and shield_recharge_rate (handle and card)
"""

import argparse
import dataclasses
import math

from bnb import game
from bnb.game import ships
from bnb.game.rules import Rules

SET = "real-scale"
KEYS = ("yongma", "jujak")


def tail_aspect(card: dict[str, float], rules: dict[str, float]) -> float:
    fraction = (
        rules.get("rules.head_on_fraction", Rules.head_on_fraction) * card["head_on_fraction_scale"]
    )
    half = rules.get("rules.head_on_angle", Rules.head_on_angle) * card["head_on_angle_scale"]
    ratio = math.pi / half
    return 1.0 - (1.0 - fraction) * math.exp(-math.log(2.0) * ratio ** card["head_on_sharpness"])


def table() -> None:
    by_key = {ship.key: ship for ship in ships.load(SET)}
    rules = ships.rules(SET)
    print(
        f"{'shooter':>8} {'target':>8} {'shield':>8} {'hit':>7} {'hits':>5} "
        f"{'continuous s':>13} {'discrete s':>11}"
    )
    for shooter in KEYS:
        for target in KEYS:
            fire, aim = by_key[shooter].card, by_key[target].card
            damage = fire["hit_damage"] * tail_aspect(aim, rules)
            shield = aim["max_shield"]
            continuous = shield / (damage * fire["fire_rate"])
            hits = math.ceil(shield / damage) + 1
            discrete = (hits - 1) / fire["fire_rate"]
            print(
                f"{shooter:>8} {target:>8} {shield:8.1f} {damage:7.2f} {hits:5d} "
                f"{continuous:13.3f} {discrete:11.3f}"
            )


def apply(scale: float) -> None:
    for ship in ships.load(SET):
        if ship.key not in KEYS:
            continue
        handles = dict(
            ship.handles,
            max_shield=ship.handles["max_shield"] * scale,
            shield_recharge_rate=ship.handles["shield_recharge_rate"] * scale,
        )
        card = dict(
            ship.card,
            max_shield=ship.card["max_shield"] * scale,
            shield_recharge_rate_scale=ship.card["shield_recharge_rate_scale"] * scale,
        )
        note = (
            f" Game v{game.VERSION + 1} (lethality-sticky): x {scale:.4g},"
            " about 1 s of sustained core fire"
        )
        notes = dict(ship.notes)
        for name in ("max_shield", "shield_recharge_rate"):
            notes[name] = notes.get(name, "") + note
        ships.write_ship(dataclasses.replace(ship, handles=handles, card=card, notes=notes), SET)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", type=float)
    arguments = parser.parse_args()
    table()
    if arguments.apply:
        apply(arguments.apply)
        print(f"\nafter x {arguments.apply}:")
        table()
