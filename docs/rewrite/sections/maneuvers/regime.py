"""The flight regime of a 1v1 mirror under r9 (or a candidate): speed over the corner speed,
the store fraction, the power chosen, by distance band, sampled every 10 ticks.

    python docs/rewrite/sections/maneuvers/regime.py --card "D2 cone 0.085" --kind 1
"""

import argparse
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "game-loop"))
sys.path.insert(0, str(HERE))

import league  # noqa: E402
from make import CARDS  # noqa: E402
from sizes import overridden  # noqa: E402

from bench.game_loop import Group, Loop  # noqa: E402
from bnb.game import api  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", default="D2 cone 0.085")
    parser.add_argument("--lineup", default="turn mirror")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--environments", type=int, default=512)
    parser.add_argument("--doctrine", default="{}")
    args = parser.parse_args()
    by = {p.name: p for p in league.load()}
    r9 = by["r9"]
    fields = json.loads(args.doctrine)
    import dataclasses

    player = dataclasses.replace(
        r9, name="c", energy=overridden(r9.energy, fields), turn=overridden(r9.turn, fields)
    )
    loop = Loop(args.environments, 1, args.seconds, ships_per_team=1, compiled=False)
    energy, turn = CARDS[args.card]
    loop.set_types([energy], [turn])
    state, observation, carry, tally = loop.start(
        [Group(0, args.lineup, player, player, args.environments)]
    )
    corner = loop.card.corner_speed  # (2K,)
    gun = loop.card.gun_range
    samples = []
    for tick in range(loop.ticks):
        state, observation, carry, tally = loop.tick(state, observation, carry, tally)
        if tick % 10 == 0 and tick > 150:
            shown = api.display(state)
            t = loop.ship_type.long()
            speed = state.ships.proper_speed / corner[t]
            store = state.ships.energy / loop.types.max_energy[t]
            dx = shown["ships.x"][:, 0] - shown["ships.x"][:, 1]
            dy = shown["ships.y"][:, 0] - shown["ships.y"][:, 1]
            d = torch.sqrt(dx * dx + dy * dy) / gun[t[:, 0]]
            alive = shown["ships.alive"].bool().all(1)
            samples.append(torch.stack([speed[:, 0], store[:, 0], d], -1)[alive].cpu())
    s = torch.cat(samples)
    for low, high in ((0, 1), (1, 2), (2, 4), (4, 1e9)):
        m = (s[:, 2] >= low) & (s[:, 2] < high)
        if m.sum() == 0:
            continue
        q = lambda v: [round(float(x), 2) for x in torch.quantile(v, torch.tensor([0.1, 0.5, 0.9]))]  # noqa: E731
        print(
            f"distance {low}-{high} gun ranges: share {float(m.float().mean()):.2f}, "
            f"speed/corner p10/50/90 {q(s[m, 0])}, store {q(s[m, 1])}"
        )


if __name__ == "__main__":
    main()
