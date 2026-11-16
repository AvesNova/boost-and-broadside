"""Deaths by cause and the boundary deaths' context (diagnosis, 2026-11-16).

Runs one balance condition's "energy v turn" job on a trims.json entry eagerly, with the
agent's and the game's internals captured each tick (`api.resolve` and Imugi's `context`,
`extend_and_return`, `choose_target`, `maneuvers` wrapped), and writes per death its
cause (enemy hit, friendly fire, boundary) and, for a boundary death, the ship's state on
the tick it crossed the boundary outward.

    python -m bench.boundary_deaths --condition deathmatch-1v1 --entry Y0.46 --games 256
"""

import argparse
import json
import pathlib
import time

import torch

from bench import balance_sweep as bs
from bench import game_loop as gl
from bnb.controllers import imugi, imugi_combat
from bnb.game import api

STASH: dict = {}


def wrap(module, name, after):
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        after(result, args, kwargs)
        return result

    setattr(module, name, wrapper)


def install():
    def on_context(fleet, args, kwargs):
        STASH["boundary_weight"] = fleet.boundary_weight
        STASH["boundary_radius"] = fleet.boundary_radius.expand_as(fleet.x)
        STASH["speed"] = fleet.speed
        STASH["view_x"] = fleet.x
        STASH["sense_override"] = torch.zeros_like(fleet.x)

    wrap(imugi, "context", on_context)
    wrap(imugi_combat, "extend_and_return", lambda r, a, k: STASH.__setitem__("extending", r[0]))
    wrap(imugi_combat, "choose_target", lambda r, a, k: STASH.__setitem__("has_target", r[0] >= 0))

    def on_maneuvers(result, args, kwargs):
        stick_in, stick_out = args[6], result[0]
        STASH["sense_override"] = 0.5 * (stick_out - stick_in).abs().sum(-1)

    wrap(imugi_combat, "maneuvers", on_maneuvers)

    def on_resolve(result, args, kwargs):
        STASH["events"] = result[1]
        STASH["state"] = result[0]

    wrap(api, "resolve", on_resolve)


class DiagLoop(gl.Loop):
    def start(self, groups):
        out = super().start(groups)
        B, N = self.B, self.N
        z = dict(device=self.device)
        self.d = dict(
            outside=torch.zeros((B, N), dtype=torch.bool, **z),
            cross=torch.zeros((B, N, 8), **z),  # the crossing tick's context
            last_enemy_damage=torch.full((B, N), -1e9, **z),
            ticks_outside=torch.zeros((B, N), **z),
            boundary_taken=torch.zeros((B, N), **z),
        )
        self.records = []
        self.causes = torch.zeros((2, 2, 4), **z)  # victim side (A, B), kind, cause
        self.alive_ticks = torch.zeros((2, 2), **z)
        self.tick_number = 0
        return out

    def tick(self, state, observation, carry, tally):
        STASH.clear()
        result = super().tick(state, observation, carry, tally)
        B, S, N = self.B, self.S, self.N
        events, after = STASH["events"], STASH["state"]
        dt = self.rules.tick_length

        def sim(name):  # (2B, S) view rows -> (B, N) sim ships
            return STASH[name].float().reshape(B, 2 * S)

        d = self.d
        t = self.tick_number
        if t == 1:
            shown = gl.api.display(state)
            print("mapping check |view x - sim x| max:",
                  float((sim("view_x") - shown["ships.x"]).abs().max()), flush=True)
        side_a = torch.cat(
            [self.a_on_zero[:, None].expand(B, S), (~self.a_on_zero)[:, None].expand(B, S)], 1
        )
        kind = (self.ship_type >= self.K).long()
        alive_start = gl.api.display(state)["ships.alive"].bool()
        enemy_hit = events.enemy_damage_by_aspect.sum(-1) > 0
        d["last_enemy_damage"] = torch.where(enemy_hit, float(t), d["last_enemy_damage"])
        outside = (events.boundary_damage > 0) & alive_start
        crossing = outside & ~d["outside"]
        speed = sim("speed")
        accel = gl.per_type(self.card.max_acceleration, self.ship_type)
        turn_radius = speed * speed / accel
        context = torch.stack(
            [
                speed,
                turn_radius,
                sim("boundary_radius"),
                sim("boundary_weight"),
                sim("has_target"),
                sim("extending"),
                sim("sense_override"),
                ((t - d["last_enemy_damage"]) * dt).clamp_max(999.0),
            ],
            -1,
        )
        d["cross"] = torch.where(crossing[..., None], context, d["cross"])
        d["ticks_outside"] = torch.where(
            outside, d["ticks_outside"] + 1, torch.where(alive_start, 0.0, d["ticks_outside"])
        )
        d["boundary_taken"] = torch.where(
            outside, d["boundary_taken"] + events.boundary_damage, d["boundary_taken"]
        )
        d["outside"] = outside
        died = events.died & ~gl_done(events)[:, None]
        killer = events.killer.long()
        index = torch.arange(N, device=self.device)
        same_team = (killer >= 0) & ((killer < S) == (index < S))
        cause = torch.where(killer < 0, 2, torch.where(same_team, 1, 0))
        # boundary deaths with an enemy hit in the last 10 s: cause 3
        recent = ((t - d["last_enemy_damage"]) * dt) < 10.0
        cause = torch.where((cause == 2) & recent, 3, cause)
        victim_side = (~side_a).long()
        flat = (victim_side * 2 + kind) * 4 + cause
        self.causes.view(-1).index_add_(0, flat[died], torch.ones_like(flat[died], dtype=torch.float))
        live = alive_start.float()
        self.alive_ticks.view(-1).index_add_(0, (victim_side * 2 + kind).flatten(), live.flatten())
        boundary_died = died & (killer < 0)
        if bool(boundary_died.any()):
            b, n = boundary_died.nonzero(as_tuple=True)
            for row, (bb, nn) in enumerate(zip(b.tolist(), n.tolist(), strict=True)):
                c = d["cross"][bb, nn].tolist()
                self.records.append(
                    dict(
                        side="A" if bool(side_a[bb, nn]) else "B",
                        kind=int(kind[bb, nn]),
                        tick=t,
                        seconds_outside=float(d["ticks_outside"][bb, nn]) * dt,
                        boundary_taken=float(d["boundary_taken"][bb, nn]),
                        speed_at_death=float(speed[bb, nn]),
                        cross=dict(
                            zip(
                                ["speed", "turn_radius", "boundary_radius", "boundary_weight",
                                 "has_target", "extending", "sense_override",
                                 "since_enemy_damage"],
                                c,
                                strict=True,
                            )
                        ),
                    )
                )
        d["boundary_taken"] = torch.where(events.died, 0.0, d["boundary_taken"])
        d["last_enemy_damage"] = torch.where(events.spawned, -1e9, d["last_enemy_damage"])
        self.tick_number += 1
        return result


def gl_done(events):
    return torch.zeros_like(events.done)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--entry", default="Y0.46")
    parser.add_argument("--entries", type=pathlib.Path,
                        default=pathlib.Path("docs/rewrite/sections/g-v23/trims.json"))
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    install()
    condition = bs.CONDITIONS[args.condition]
    import dataclasses

    settings = dataclasses.replace(bs.BALANCE, seconds=args.seconds)
    entry = next(e for e in json.loads(args.entries.read_text()) if e["name"] == args.entry)
    pair = bs.as_pair(entry)
    loop = bs.balance_loop(condition, args.games, 1, settings, "cuda", compiled=False)
    loop.__class__ = DiagLoop
    loop._tick = loop.tick
    loop.set_table(bs.card_table([pair["a"]], [pair["b"]], loop.device))
    parameters = bs.mode_parameters(condition)
    jobs = bs.jobs_of([pair], args.games, 0, parameters)
    loop.spawn_seed = 0
    torch.manual_seed(0)
    started = time.time()
    outs = gl.run(loop, jobs)
    summary = bs.balance_summary(outs[0], loop.rules.tick_length)
    radius = float(loop.game.arena_radius)
    result = dict(
        condition=args.condition,
        entry=args.entry,
        games=args.games,
        arena_radius=radius,
        seconds=time.time() - started,
        kill_share_a=summary.get("kill_share_a"),
        point_share_a=summary.get("point_share_a"),
        causes=loop.causes.cpu().tolist(),  # [victim side A,B][kind Y,J][hit,ff,boundary,boundary<10s after enemy hit]
        alive_seconds=(loop.alive_ticks * loop.rules.tick_length).cpu().tolist(),
        records=loop.records,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
