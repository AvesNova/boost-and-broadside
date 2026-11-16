"""Why the scripted agent's 1v1 duels rarely end (diagnosis, 2026-11-16).

Runs deathmatch-1v1's "energy v turn" job on a trims.json entry eagerly with
`bench.boundary_deaths`' capture and tallies per ship kind (Yongma, Jujak): quiet flight
(the input that lets the shield recharge) and its longest streak, the shield restored and
lost, merges (the game loop's own count), shots and hits, the time with the opponent on the
nose (the nose line within the opponent's hit radius) and in gun range, and the time both
ships hold a sustained turn (one or two circles) or extend.

    python -m bench.duel_anatomy --entry Y0.46 --games 128 --out /tmp/duel.json
"""

import argparse
import dataclasses
import json
import math
import pathlib
import time

import torch

from bench import balance_sweep as bs
from bench import boundary_deaths as bd
from bench import game_loop as gl
from bnb.game import api
from bnb.game.physics.combat import quiet

SUSTAINED = 3.0  # s: a turn held this long in one sense is a sustained turn
HARD = 0.5  # of the type's sustained turn rate: a turn at least this fast
LOOSE_CONE = math.radians(15.0)


class DuelLoop(bd.DiagLoop):
    def start(self, groups):
        out = super().start(groups)
        B, N = self.B, self.N
        z = dict(device=self.device)
        self.e = dict(
            streak=torch.zeros((B, N), **z),
            longest=torch.zeros((B, N), **z),
            turn_held=torch.zeros((B, N), **z),
            turn_sign=torch.zeros((B, N), **z),
            previous=None,
        )
        names = (
            "alive", "quiet", "restored", "lost", "shots", "hits", "on_nose", "nose_15",
            "in_range", "turn_sustained", "extending", "both_circle_same", "both_circle_opposite",
            "both_extending", "one_extending",
        )  # fmt: skip
        self.names = names
        self.sums = torch.zeros((len(names), 2), **z)  # per kind (Yongma, Jujak); ticks or amounts
        self.pair_ticks = torch.zeros((), **z)
        return out

    def tick(self, state, observation, carry, tally):
        self.tally_ref = tally
        result = super().tick(state, observation, carry, tally)
        B, S, N = self.B, self.S, self.N
        assert S == 1, "a duel"
        events, action = bd.STASH["events"], bd.STASH["action"]
        dt = self.rules.tick_length
        kind = (self.ship_type >= self.K).long()
        e = self.e
        shown = api.display(bd.STASH["state_before"])
        alive = shown["ships.alive"].bool()
        after = api.display(state)
        x, y = after["ships.x"], after["ships.y"]
        cos, sin = after["ships.heading_cos"], after["ships.heading_sin"]
        is_quiet = quiet(action) & alive
        e["streak"] = torch.where(is_quiet, e["streak"] + 1, 0.0)
        e["longest"] = torch.maximum(e["longest"], e["streak"])
        damage = events.damage  # (B, N, N) shooter -> target
        hits = (damage > 0).any(2).float()
        # geometry to the opponent
        dx, dy = x.flip(1) - x, y.flip(1) - y
        distance = torch.sqrt(dx * dx + dy * dy)
        along = cos * dx + sin * dy
        across = (cos * dy - sin * dx).abs()
        hit_radius = gl.per_type(self.card.hit_radius, self.ship_type.flip(1))
        reach = gl.per_type(self.card.gun_range, self.ship_type)
        both = alive & alive.flip(1)
        in_range = both & (distance < reach)
        on_nose = in_range & (along > 0) & (across < hit_radius)
        nose_15 = in_range & (torch.atan2(across, along) < LOOSE_CONE)
        # sustained turns from the heading's change
        if e["previous"] is None:
            e["previous"] = (cos, sin)
        pc, ps = e["previous"]
        rate = torch.atan2(pc * sin - ps * cos, pc * cos + ps * sin) / dt
        e["previous"] = (cos, sin)
        limit = HARD * gl.per_type(self.card.turn_rate[..., 2], self.ship_type)
        sign = torch.where(rate.abs() > limit, torch.sign(rate), 0.0)
        held = (sign != 0) & (sign == e["turn_sign"])
        e["turn_held"] = torch.where(held, e["turn_held"] + 1, torch.where(sign != 0, 1.0, 0.0))
        e["turn_sign"] = sign
        sustained = alive & (e["turn_held"] * dt >= SUSTAINED)
        extending = alive & bd.STASH["extending"].float().reshape(B, N).bool()
        both_turn = both & sustained & sustained.flip(1)
        same = both_turn & (e["turn_sign"] == e["turn_sign"].flip(1))
        rows = [
            alive.float(),
            is_quiet.float(),
            events.shield_restored,
            events.shield_lost,
            events.fired.float(),
            hits,
            on_nose.float(),
            nose_15.float(),
            in_range.float(),
            sustained.float(),
            extending.float(),
            same.float(),
            (both_turn & ~same).float(),
            (both & extending & extending.flip(1)).float(),
            (both & (extending ^ extending.flip(1))).float(),
        ]
        stacked = torch.stack(rows, 0)  # (R, B, N)
        onehot = torch.nn.functional.one_hot(kind, 2).float()  # (B, N, 2)
        self.sums += torch.einsum("rbn,bnk->rk", stacked, onehot)
        self.pair_ticks += both[:, 0].float().sum()
        return result


def install_extra():
    original = api.resolve

    def resolve(*args, **kwargs):
        bd.STASH["state_before"] = args[2]
        bd.STASH["action"] = args[4]
        return original(*args, **kwargs)

    api.resolve = resolve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", default="base")
    parser.add_argument(
        "--entries",
        type=pathlib.Path,
        default=pathlib.Path("docs/rewrite/sections/g-v23/trims.json"),
    )
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    install_extra()
    bd.install()  # wraps api.resolve again, outside ours; both see each tick
    condition = bs.CONDITIONS["deathmatch-1v1"]
    settings = dataclasses.replace(bs.BALANCE, seconds=None)
    entry = (
        {"name": "base"}  # the set's own card (game v24: Y0.46 with head-on damage 0.05)
        if args.entry == "base"
        else next(e for e in json.loads(args.entries.read_text()) if e["name"] == args.entry)
    )
    pair = bs.as_pair(entry)
    loop = bs.balance_loop(condition, args.games, 1, settings, "cuda", compiled=False)
    loop.__class__ = DuelLoop
    loop._tick = loop.tick
    loop.set_table(bs.card_table([pair["a"]], [pair["b"]], loop.device))
    jobs = bs.jobs_of([pair], args.games, 0, bs.mode_parameters(condition))
    loop.spawn_seed = 0
    torch.manual_seed(0)
    started = time.time()
    outs = gl.run(loop, jobs)
    summary = bs.balance_summary(outs[0], loop.rules.tick_length)
    dt = loop.rules.tick_length
    games = args.games
    sums = dict(zip(loop.names, loop.sums.cpu().tolist(), strict=True))
    alive = sums["alive"]
    per_kind = {}
    longest = loop.e["longest"]
    kind = (loop.ship_type >= loop.K).long()
    for k, name in enumerate(("Yongma", "Jujak")):
        mask = kind == k
        per_kind[name] = dict(
            quiet_share=sums["quiet"][k] / alive[k],
            longest_quiet_s_mean=float(longest[mask].mean()) * dt,
            longest_quiet_s_max=float(longest[mask].max()) * dt,
            shield_restored_per_game=sums["restored"][k] / games,
            shield_lost_per_game=sums["lost"][k] / games,
            shots_per_game=sums["shots"][k] / games,
            hits_per_game=sums["hits"][k] / games,
            on_nose_in_range_share=sums["on_nose"][k] / alive[k],
            within_15deg_in_range_share=sums["nose_15"][k] / alive[k],
            in_range_share=sums["in_range"][k] / alive[k],
            sustained_turn_share=sums["turn_sustained"][k] / alive[k],
            extending_share=sums["extending"][k] / alive[k],
        )
    merges = float(loop.tally_ref["merges"].sum()) / games
    circles = loop.tally_ref["circles"].sum(0).cpu().tolist()  # one, two, scissors
    pair = float(loop.pair_ticks)
    result = dict(
        entry=args.entry,
        games=games,
        seconds=time.time() - started,
        kill_share_a=summary.get("kill_share_a"),
        point_share_a=summary.get("point_share_a"),
        deaths_per_game=float(loop.causes.sum()) / games,
        causes=loop.causes.cpu().tolist(),
        merges_per_game=merges,
        post_merge_circles_one_two_scissors=circles,
        both_alive_minutes_per_game=pair * dt / 60 / games,
        # shares of the time both are alive (each pair tick counted once per ship: halve)
        both_circle_same_sense_share=sum(sums["both_circle_same"]) / 2 / pair,
        both_circle_opposite_sense_share=sum(sums["both_circle_opposite"]) / 2 / pair,
        both_extending_share=sum(sums["both_extending"]) / 2 / pair,
        one_extending_share=sum(sums["one_extending"]) / 2 / pair,
        per_kind=per_kind,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
