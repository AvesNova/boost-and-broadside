"""The doctrine arena with game version 4's shield recharge (an input other than a quiet one
pauses the recharge wait instead of restarting it), to measure what version 5's rule does:

    PYTHONPATH=src python docs/rewrite/sections/doctrines/version4_recharge.py play ...

takes `bench.doctrine_arena`'s arguments. Patches `bnb.game.physics.step.apply_damage` in
this process only.
"""

import sys

import torch

from bench import doctrine_arena
from bnb.game.physics import combat, step
from bnb.game.ship_types import per_ship
from bnb.game.state import replace


def version4_apply_damage(rules, types, ships, alive_at_start, total, quiet_input):
    """`combat.apply_damage` as in game version 4: the wait holds on a non-quiet tick."""
    tick_length = rules.tick_length
    shield0, wait0 = ships.shield, ships.shield_recharge_wait
    hurt = total > 0.0
    died = alive_at_start & hurt & (shield0 <= 0.0)
    shield = (shield0 - total).clamp_min(0.0)
    rate = rules.shield_recharge_rate * per_ship(types.shield_recharge_rate_scale, ships.ship_type)
    delay = rules.shield_recharge_delay * per_ship(
        types.shield_recharge_delay_scale, ships.ship_type
    )
    recharge = torch.minimum(
        per_ship(types.max_shield, ships.ship_type) - shield,
        rate * (tick_length - wait0).clamp(0.0, tick_length),
    )
    restored = torch.where(alive_at_start & ~hurt & quiet_input, recharge, 0.0)
    counted = torch.where(quiet_input, (wait0 - tick_length).clamp_min(0.0), wait0)
    wait = torch.where(hurt, delay, counted)
    ships = replace(
        ships,
        shield=torch.where(alive_at_start, shield + restored, shield0),
        shield_recharge_wait=torch.where(alive_at_start, wait, wait0),
    )
    return combat.Damage(
        ships=ships,
        died=died,
        shield_lost=torch.where(alive_at_start, shield0 - shield, 0.0),
        shield_restored=restored,
    )


if __name__ == "__main__":
    step.apply_damage = version4_apply_damage
    sys.argv[0] = "doctrine_arena"
    doctrine_arena.main()
