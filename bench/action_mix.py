"""Imugi's action mix (2026-11-16): the share of living-ship ticks at each stick action
(straight, then right and left at 0.4, 0.75 and full stick), each power mode (coast, boost,
reverse) and shooting, per ship type (Yongma: side A, Jujak: side B), on one balance
condition's "energy v turn" job with the current card, eager.

    python -m bench.action_mix --condition deathmatch-5v5 --games 128 --out /tmp/mix.json
"""

import argparse
import json
import pathlib
import time

import torch

from bench import balance_sweep as bs
from bench import game_loop as gl
from bnb.game import api

STICKS = ("straight", "R0.4", "L0.4", "R0.75", "L0.75", "R1.0", "L1.0")
POWERS = ("coast", "boost", "reverse")
STASH: dict = {}


def install():
    """Wrap `api.resolve` to keep each tick's state before it and the action."""
    original = api.resolve

    def resolve(*args, **kwargs):
        STASH["state_before"], STASH["action"] = args[2], args[4]
        return original(*args, **kwargs)

    api.resolve = resolve


class MixLoop(gl.Loop):
    """The game loop with the living ships' stick, power and shoot counts per kind."""

    def start(self, groups):
        """Start the loop and zero the counts."""
        out = super().start(groups)
        z = dict(device=self.device)
        self.stick = torch.zeros((2, len(STICKS)), **z)
        self.power = torch.zeros((2, len(POWERS)), **z)
        self.shoot = torch.zeros((2, 2), **z)
        return out

    def tick(self, state, observation, carry, tally):
        """One tick, then count the actions of the ships alive before it."""
        result = super().tick(state, observation, carry, tally)
        action = STASH["action"]
        alive = api.display(STASH["state_before"])["ships.alive"].bool()
        live = alive & ~tally["finished"][:, None]
        kind = (self.ship_type >= self.K).long()
        onehot = torch.nn.functional.one_hot(kind, 2).float() * live[..., None].float()
        for table, values, count in (
            (self.stick, action.stick, len(STICKS)),
            (self.power, action.power, len(POWERS)),
            (self.shoot, action.shoot, 2),
        ):
            hot = torch.nn.functional.one_hot(values.long(), count).float()
            table += torch.einsum("bnk,bnc->kc", onehot, hot)
        return result


def shares(table: torch.Tensor, names) -> dict:
    """Each kind's counts (2, C) as shares by name: {kind: {name: share}}."""
    return {
        kind: dict(zip(names, (row / row.sum()).tolist(), strict=True))
        for kind, row in zip(("Yongma", "Jujak"), table.cpu(), strict=True)
    }


def measure(condition_name: str, games: int, seed: int = 0, entry: dict | None = None) -> dict:
    """The condition's "energy v turn" games on the set's own card, or a candidates entry
    (`bench.balance_sweep.as_pair`): the shares and the balance numbers."""
    condition = bs.CONDITIONS[condition_name]
    loop = bs.balance_loop(condition, games, 1, bs.BALANCE, "cuda", compiled=False)
    loop.__class__ = MixLoop
    loop._tick = loop.tick
    pair = bs.as_pair(entry or {"name": "base"})
    loop.set_table(bs.card_table([pair["a"]], [pair["b"]], loop.device))
    jobs = bs.jobs_of([pair], games, 0, bs.mode_parameters(condition))
    loop.spawn_seed = seed
    torch.manual_seed(seed)
    started = time.time()
    outs = gl.run(loop, jobs)
    summary = bs.balance_summary(outs[0], loop.rules.tick_length)
    return dict(
        condition=condition_name,
        entry=pair["name"],
        games=games,
        seconds=time.time() - started,
        kill_share_a=summary["kill_share_a"],
        point_share_a=summary["point_share_a"],
        stick=shares(loop.stick, STICKS),
        power=shares(loop.power, POWERS),
        shoot=shares(loop.shoot, ("hold", "fire")),
    )


def main():
    """The command line (the module docstring)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="deathmatch-5v5", choices=bs.CONDITIONS)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--entries", type=pathlib.Path, default=None, help="a candidates file")
    parser.add_argument("--entry", default=None, help="its entry's name (default: the set's card)")
    args = parser.parse_args()
    install()
    entry = None
    if args.entry is not None:
        entry = next(e for e in json.loads(args.entries.read_text()) if e["name"] == args.entry)
    result = measure(args.condition, args.games, args.seed, entry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
