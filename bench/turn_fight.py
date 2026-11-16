"""The 1v1 turning fight's tally (2026-11-16, the initiative rule's before and after).

`bench.duel_anatomy`'s duel (deathmatch-1v1, "energy v turn", the set's own card, eager)
with the turning fight's numbers added, per ship kind (Yongma, Jujak):

- reversals after a merge: in the game loop's post-merge window (3 s after a merge ends),
  each ship's nose-rate sense flips, a sense counted only at |rate| > SENSE_RATE (15 deg/s;
  a slower turn keeps the last sense: hysteresis); the mean per window and the share of
  windows in which the ship reversed at least twice (and both did: the scissors share at
  this threshold). The game loop's own count (any rate above 0.05 rad/s) is kept as
  post_merge_circles_one_two_scissors;
- the sustained-turn share: the time alive turning faster than SENSE_RATE in one sense held
  at least HELD seconds (and the plain share above SENSE_RATE; the old definition, at least
  half the type's sustained rate for 3 s, is duel_anatomy's sustained_turn_share);
- the commanded stick's sense flips a minute (a nonzero stick's side against the last
  nonzero one), merges per kill, deaths a game, and the stick and power mix
  (`bench.action_mix`'s actions).

    python -m bench.turn_fight --games 128 --out /tmp/turn.json
"""

import argparse
import dataclasses
import json
import math
import pathlib
import time

import torch

from bench import action_mix as am
from bench import balance_sweep as bs
from bench import boundary_deaths as bd
from bench import duel_anatomy as da
from bench import game_loop as gl

SENSE_RATE = math.radians(15.0)  # rad/s
HELD = 1.0  # s


class FightLoop(da.DuelLoop):
    def start(self, groups):
        out = super().start(groups)
        B, N = self.B, self.N
        z = dict(device=self.device)
        self.f = dict(
            sense=torch.zeros((B, N), **z),  # the hysteresis sense, +1 left
            held=torch.zeros((B, N), **z),  # ticks the sense held above SENSE_RATE
            window_sense=torch.zeros((B, N), **z),
            window_flips=torch.zeros((B, N), **z),
            last_window=torch.zeros((B,), **z),
            stick_side=torch.zeros((B, N), **z),
            previous=None,
        )
        names = (
            "windows", "window_flips", "window_two", "fast", "fast_held", "stick_flips",
        )  # fmt: skip
        self.fight_names = names
        self.fight = torch.zeros((len(names), 2), **z)
        self.both_two = torch.zeros((), **z)
        self.stick = torch.zeros((2, len(am.STICKS)), **z)
        self.power = torch.zeros((2, len(am.POWERS)), **z)
        return out

    def tick(self, state, observation, carry, tally):
        result = super().tick(state, observation, carry, tally)
        new_tally = result[3]
        f, dt = self.f, self.rules.tick_length
        kind = (self.ship_type >= self.K).long()
        onehot = torch.nn.functional.one_hot(kind, 2).float()
        shown = gl.api.display(bd.STASH["state_before"])
        alive = shown["ships.alive"].bool() & ~tally["finished"][:, None]
        after = gl.api.display(state)
        cos, sin = after["ships.heading_cos"], after["ships.heading_sin"]
        if f["previous"] is None:
            f["previous"] = (cos, sin)
        pc, ps = f["previous"]
        rate = torch.atan2(pc * sin - ps * cos, pc * cos + ps * sin) / dt
        f["previous"] = (cos, sin)
        fast = rate.abs() > SENSE_RATE
        now = torch.where(fast, torch.sign(rate), 0.0)
        same = fast & (now == f["sense"])
        f["held"] = torch.where(same, f["held"] + 1, torch.where(fast, 1.0, 0.0))
        f["sense"] = torch.where(fast, now, f["sense"])
        fast_held = alive & fast & (f["held"] * dt >= HELD)
        # the post-merge window (`bench.game_loop`): opened when it is set to its length
        window = new_tally["window"][:, 0, 0]
        opened = window == self.window_ticks - 1
        closed = (f["last_window"] > 0) & (window == 0)
        f["last_window"] = window
        flipped = fast & (f["window_sense"] != 0) & (now != f["window_sense"])
        f["window_flips"] = torch.where(opened[:, None], 0.0, f["window_flips"] + flipped.float())
        f["window_sense"] = torch.where(
            opened[:, None], torch.where(fast, now, 0.0), torch.where(fast, now, f["window_sense"])
        )
        closing = closed[:, None].expand_as(alive)
        two = closing & (f["window_flips"] >= 2)
        self.both_two += (two.all(1)).float().sum()
        # the commanded stick's side (STICK_LEVELS: odd right, even left)
        action = bd.STASH["action"]
        stick = action.stick.long()
        side = torch.where(stick == 0, 0.0, torch.where(stick % 2 == 0, 1.0, -1.0))
        stick_flip = alive & (side != 0) & (f["stick_side"] != 0) & (side != f["stick_side"])
        f["stick_side"] = torch.where(side != 0, side, f["stick_side"])
        rows = [
            closing.float(),
            torch.where(closing, f["window_flips"], 0.0),
            two.float(),
            (alive & fast).float(),
            fast_held.float(),
            stick_flip.float(),
        ]
        self.fight += torch.einsum("rbn,bnk->rk", torch.stack(rows, 0), onehot)
        live = onehot * alive[..., None].float()
        for table, values, count in (
            (self.stick, action.stick, len(am.STICKS)),
            (self.power, action.power, len(am.POWERS)),
        ):
            hot = torch.nn.functional.one_hot(values.long(), count).float()
            table += torch.einsum("bnk,bnc->kc", live, hot)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    da.install_extra()
    bd.install()
    condition = bs.CONDITIONS["deathmatch-1v1"]
    settings = dataclasses.replace(bs.BALANCE, seconds=None)
    pair = bs.as_pair({"name": "base"})
    loop = bs.balance_loop(condition, args.games, 1, settings, "cuda", compiled=False)
    loop.__class__ = FightLoop
    loop._tick = loop.tick
    loop.set_table(bs.card_table([pair["a"]], [pair["b"]], loop.device))
    jobs = bs.jobs_of([pair], args.games, 0, bs.mode_parameters(condition))
    loop.spawn_seed = args.seed
    torch.manual_seed(args.seed)
    started = time.time()
    outs = gl.run(loop, jobs)
    summary = bs.balance_summary(outs[0], loop.rules.tick_length)
    dt, games = loop.rules.tick_length, args.games
    sums = dict(zip(loop.names, loop.sums.cpu().tolist(), strict=True))
    fight = dict(zip(loop.fight_names, loop.fight.cpu().tolist(), strict=True))
    alive = sums["alive"]
    kills = float(outs[0]["match_kills"].sum())
    merges = float(loop.tally_ref["merges"].sum())
    per_kind = {}
    for k, name in enumerate(("Yongma", "Jujak")):
        windows = max(fight["windows"][k], 1.0)
        per_kind[name] = dict(
            reversals_per_window=fight["window_flips"][k] / windows,
            windows_two_reversals_share=fight["window_two"][k] / windows,
            turn_above_15_share=fight["fast"][k] / alive[k],
            sustained_turn_share=fight["fast_held"][k] / alive[k],
            sustained_turn_share_old=sums["turn_sustained"][k] / alive[k],
            stick_flips_per_minute=fight["stick_flips"][k] / (alive[k] * dt / 60.0),
            extending_share=sums["extending"][k] / alive[k],
            hits_per_game=sums["hits"][k] / games,
            shots_per_game=sums["shots"][k] / games,
            shield_lost_per_game=sums["lost"][k] / games,
        )
    result = dict(
        games=games,
        seconds=time.time() - started,
        kill_share_a=summary.get("kill_share_a"),
        point_share_a=summary.get("point_share_a"),
        deaths_per_game=float(loop.causes.sum()) / games,
        causes_by_side_kind_cause=loop.causes.cpu().tolist(),
        kills_per_game=kills / games,
        merges_per_game=merges / games,
        merges_per_kill=merges / max(kills, 1.0),
        windows=fight["windows"][0],
        both_two_reversals_share=float(loop.both_two) / max(fight["windows"][0], 1.0),
        post_merge_circles_one_two_scissors=loop.tally_ref["circles"].sum(0).cpu().tolist(),
        per_kind=per_kind,
        stick=am.shares(loop.stick, am.STICKS),
        power=am.shares(loop.power, am.POWERS),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
