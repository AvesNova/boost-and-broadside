"""Imugi's plan choices in real-scale deathmatch 5v5 (energy v turn, fields on, the preset's
map), uncompiled: the share of living ship-ticks extending (by kind), the extensions started
and the target changes per ship-minute, the share with a target.

    PYTHONPATH=src:. uv run --no-sync python docs/rewrite/sections/lethality-sticky/probe.py
"""

import sys

import torch

from bench.game_loop import Group, Loop, Player
from bnb.contracts.obs import SHIP
from bnb.controllers.imugi_defaults import MODE_DEFAULTS
from bnb.controllers.imugi_doctrine import stored
from bnb.controllers.imugi_parameters import pack

ENVIRONMENTS, SECONDS = 128, 60.0
lineup = sys.argv[1] if len(sys.argv) > 1 else "energy v turn"
doctrines = stored()["real-scale"]
player = Player(
    "imugi", doctrines["yongma"], doctrines["jujak"],
    parameters=tuple(pack(MODE_DEFAULTS.sets["real_scale_deathmatch"])),
)  # fmt: skip
loop = Loop(ENVIRONMENTS, 1, SECONDS, compiled=False, fields=True, fixed_map=True)
state, observation, carry, tally = loop.start([Group(0, lineup, player, player, ENVIRONMENTS)])
S = loop.S
ENGAGED = 2.0
gun_range = float(loop.card.gun_range.max())
sums = dict(
    alive_y=0.0,
    alive_j=0.0,
    ext_y=0.0,
    ext_j=0.0,
    starts=0.0,
    changes=0.0,
    targeted=0.0,
    engaged_changes=0.0,
)
for tick in range(loop.ticks - 1):
    before = carry
    state, observation, carry, tally = loop._tick(state, observation, carry, tally)
    own = observation.ships.flatten(0, 1)[:, :S]
    alive = own[..., SHIP["alive"]] > 0.5
    yongma = own[..., SHIP["ship_type"]].round() < loop.K  # the variant table: Yongma rows first
    # (a carry without the extension, before it existed: none)
    phase = getattr(carry, "extend_phase", torch.zeros_like(own[..., 0]))
    extending = phase > 0
    sums["alive_y"] += float((alive & yongma).sum())
    sums["alive_j"] += float((alive & ~yongma).sum())
    sums["ext_y"] += float((alive & yongma & extending).sum())
    sums["ext_j"] += float((alive & ~yongma & extending).sum())
    was = getattr(before, "extend_phase", phase)
    sums["starts"] += float((alive & extending & (was <= 0)).sum())
    changed = (before.target >= 0) & (carry.target >= 0) & (before.target != carry.target)
    sums["changes"] += float((alive & changed & ~before.fresh).sum())
    # a change while the old target was within ENGAGED gun ranges (its last seen place)
    held = before.target.clamp_min(0)
    off_x = before.enemy_x.gather(1, held) - own[..., SHIP["x"]]
    off_y = before.enemy_y.gather(1, held) - own[..., SHIP["y"]]
    near = torch.sqrt(off_x * off_x + off_y * off_y) < ENGAGED * gun_range
    sums["engaged_changes"] += float((alive & changed & ~before.fresh & near).sum())
    sums["targeted"] += float((alive & (carry.target >= 0)).sum())
tick = loop.rules.tick_length
alive = sums["alive_y"] + sums["alive_j"]
minutes = alive * tick / 60.0
print(lineup)
print(
    f"extending: Yongma {sums['ext_y'] / max(sums['alive_y'], 1):.3f}, "
    f"Jujak {sums['ext_j'] / max(sums['alive_j'], 1):.3f} of living ticks"
)
print(
    "extensions started per Yongma-minute "
    f"{sums['starts'] / max(sums['alive_y'] * tick / 60, 1e-9):.2f}"
)
print(f"target changes per ship-minute {sums['changes'] / minutes:.2f}")
print(
    f"of which with the old target within {ENGAGED} gun ranges "
    f"{sums['engaged_changes'] / minutes:.2f}"
)
print(f"with a target {sums['targeted'] / alive:.3f}")
