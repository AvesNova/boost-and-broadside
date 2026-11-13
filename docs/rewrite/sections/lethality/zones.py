"""Frontline 2v2 zone statistics by type (game v13, the base card, both sides r9m2): the
share of living time each type spends inside a capture circle (any zone, and the zones its
team attacks or defends), its deaths inside or near one, the meter movement by type, and
the zone radius against each type's settled turn circles. CPU, uncompiled, small batch.

    PYTHONPATH=src:. uv run --no-sync python docs/rewrite/sections/lethality/zones.py B SECONDS
"""

import json
import math
import pathlib
import sys
import time

import torch

sys.path.insert(0, ".")
from bench.game_loop import Group, Loop, player_from  # noqa: E402
from bnb.game.modes.frontline import TEAM0_SPAWN, TEAM1_SPAWN  # noqa: E402

HERE = pathlib.Path(__file__).parent
torch.set_num_threads(2)


def main() -> None:
    B, seconds = int(sys.argv[1]), float(sys.argv[2])
    candidate = json.loads((HERE / "round5" / "candidates.json").read_text())[0]
    league = json.loads((HERE.parent / "game-loop" / "league.json").read_text())
    r9 = player_from([e for e in league if e["name"] == "r9"][0])
    from bnb.controllers.scripted_doctrine import from_named, named

    def with_fields(doctrine, fields):
        return from_named(dict(named(doctrine), **fields))

    player = type(r9)(
        "r9m2",
        with_fields(r9.energy, candidate["doctrine"]["energy"]),
        with_fields(r9.turn, candidate["doctrine"]["turn"]),
        r9.temperature,
        r9.parameters,
    )
    loop = Loop(B, 1, seconds, device="cpu", compiled=False, ships_per_team=2, mode="frontline")
    state, observation, carry, tally = loop.start([Group(0, "energy v turn", player, player, B)])
    kind = (loop.ship_type >= loop.K).long()  # 0 Yongma, 1 Jujak
    inside = torch.zeros(2)
    near = torch.zeros(2)
    alive_ticks = torch.zeros(2)
    started = time.time()
    for tick in range(loop.ticks):
        state, observation, carry, tally = loop._tick(state, observation, carry, tally)
        mode, ships = state.mode, state.ships
        dx = ships.x[:, :, None] - mode.zone_x[:, None, :]
        dy = ships.y[:, :, None] - mode.zone_y[:, None, :]
        distance = torch.sqrt(dx * dx + dy * dy)  # (B, N, Z)
        radius = mode.zone_radius[:, None, :]
        alive = ships.alive.bool()
        # capturable zones only (a spawn zone is neither attacked nor defended)
        capturable = ((mode.role != TEAM0_SPAWN) & (mode.role != TEAM1_SPAWN))[:, None, :]
        any_in = ((distance <= radius) & capturable).any(-1) & alive
        any_near = ((distance <= 1.5 * radius) & capturable).any(-1) & alive
        for k in (0, 1):
            mask = kind == k
            alive_ticks[k] += float((alive & mask).sum())
            inside[k] += float((any_in & mask).sum())
            near[k] += float((any_near & mask).sum())
        if tick % 3000 == 0:
            print(f"tick {tick}/{loop.ticks} {time.time() - started:.0f} s", flush=True)
    a_result = torch.where(loop.a_on_zero, tally["result"], -tally["result"])
    print("Yongma side result: win/draw/loss", [int((a_result == v).sum()) for v in (1, 0, -1)])
    for k, name in enumerate(("Yongma", "Jujak")):
        print(
            f"{name}: inside a zone {inside[k] / alive_ticks[k]:.3f} of living time, "
            f"within 1.5 zone radii {near[k] / alive_ticks[k]:.3f}"
        )
    print("zone radius (m)", float(state.mode.zone_radius[0, 0]))
    deaths_by_kind = tally["kills_kind"].sum((0, 3))  # (killer kind, victim kind)
    print("kills: killer kind x victim kind (Y, J)", deaths_by_kind.tolist())
    for name, speed, rate in (("Yongma", 161.0, 26.1), ("Jujak", 161.0, 39.4)):
        diameter = 2.0 * speed / math.radians(rate)
        print(f"{name} sustained turn circle {diameter:.0f} m across (native settled)")


if __name__ == "__main__":
    main()
