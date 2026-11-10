"""The pacing of a real-scale fight: how much of a ship's time is fighting, how much is
recovering the store, and how the fights themselves go, for cards that differ in the climb
(the store's charge), the store and the engine knee (docs/explanation/game/real-scale-pacing.md).

    uv run --no-sync python -m bench.pacing --variants VARIANTS.json --out OUT.jsonl

VARIANTS.json is a list of {"name", "type": the type's name in the set, "card": {field:
value} overrides for that type (a solved card's fields), "engine_knee_scale": optional, set
on every type}. Every variant plays `--games` games of `--preset` (real_scale_deathmatch by
default), every ship the variant's type, flown by the scripted agent (its real-scale set,
T = 1) on both teams, for one episode's ticks (600 s). The card is written in place between
variants (`set_card`), so the rollout compiles once.

Each ship-tick of a living ship is one of:

- engaged: an enemy is alive within ENGAGED_IN_GUN_RANGES of the ship's own gun ranges;
- recovering: not engaged, its store below the type's spawn energy (40% of the store: what
  a fresh ship starts a fight with);
- ready: not engaged, the store at or above it.

Per variant, one JSON line: the shares of living time in each state; the mean store
fraction while engaged and at an engagement's start; the share of engagement starts below
the spawn energy; the mean time from leaving a fight to being ready again (a recovery
that ends by the next fight ending it unfinished is counted in `recoveries_cut`); while
engaged, the mean speed over cruise and the mean flight-path turn rate (deg/s); kills a
minute, and the engaged ship-seconds per kill (a fight's kill time, summed over the
ships in it). Each line is appended as it finishes. Holds the benchmark lock.
"""

import argparse
import dataclasses
import json
import math
import time

import torch

from bench.lock import bench_lock
from bnb.contracts import Pool
from bnb.controllers.scripted import Scripted
from bnb.controllers.scripted_defaults import MODE_DEFAULTS
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.runtime.analytics import RESOLVE_COLUMNS, reset_game_sums
from bnb.runtime.rollout import Rollout

ENGAGED_IN_GUN_RANGES = 2.0
SUMS = (
    "alive",
    "engaged",
    "recovering",
    "ready",
    "engaged_store",
    "engaged_speed",
    "engaged_turn",
    "starts",
    "starts_store",
    "starts_low",
    "recoveries",
    "recovery_seconds",
    "recoveries_cut",
)


def blank_pace(environments: int, ships: int, device) -> dict:
    """The pace stage's carry: SUMS per environment, and per ship its last heading, whether
    it was engaged and ready, and the seconds since it left its last fight (-1: none)."""
    zeros = torch.zeros(environments, device=device)
    pace = {name: zeros.clone() for name in SUMS}
    shape = (environments, ships)
    pace["heading_cos"] = torch.ones(shape, device=device)
    pace["heading_sin"] = torch.zeros(shape, device=device)
    pace["was_engaged"] = torch.zeros(shape, dtype=torch.bool, device=device)
    pace["was_ready"] = torch.ones(shape, dtype=torch.bool, device=device)
    pace["since_fight"] = torch.full(shape, -1.0, device=device)
    return pace


def pace_stage(rules, gun_range: torch.Tensor, cruise: torch.Tensor, ships_per_team: int):
    """The stage appended to the rollout's tick: the pace sums of this tick (see the module)."""
    dt = rules.tick_length

    def stage(carry: dict) -> None:
        ships, types, pace = carry["state"].ships, carry["types"], carry["pace"]
        kind = ships.ship_type.long()
        alive = ships.alive
        team = torch.arange(ships.x.shape[1], device=ships.x.device) // ships_per_team
        enemy = team[None, :, None] != team[None, None, :]
        dx = ships.x[:, None, :] - ships.x[:, :, None]
        dy = ships.y[:, None, :] - ships.y[:, :, None]
        near = (dx * dx + dy * dy) < (ENGAGED_IN_GUN_RANGES * gun_range[kind])[..., None] ** 2
        engaged = alive & (near & enemy & alive[:, None, :]).any(-1)
        store = types.max_energy[kind]
        fraction = ships.energy / store
        ready_level = types.spawn_energy[kind]
        ready = ships.energy >= ready_level
        recovering = alive & ~engaged & ~ready
        cos, sin = ships.heading_cos, ships.heading_sin
        turned = torch.atan2(
            pace["heading_cos"] * sin - pace["heading_sin"] * cos,
            pace["heading_cos"] * cos + pace["heading_sin"] * sin,
        ).abs()
        steady = engaged & ~ships.spawned
        start = engaged & ~pace["was_engaged"] & ~ships.spawned
        # a recovery: the seconds from leaving a fight to being ready again, unengaged
        since = pace["since_fight"]
        left = pace["was_engaged"] & ~engaged & alive
        since = torch.where(left, 0.0, since)
        counting = (since >= 0.0) & alive & ~engaged
        since = torch.where(counting, since + dt, since)
        recovered = counting & ready & ~pace["was_ready"]
        cut = (since >= 0.0) & engaged & ~ready
        sums = dict(
            alive=alive,
            engaged=engaged,
            recovering=recovering,
            ready=alive & ~engaged & ready,
            engaged_store=torch.where(engaged, fraction, 0.0),
            engaged_speed=torch.where(engaged, ships.proper_speed / cruise[kind], 0.0),
            engaged_turn=torch.where(steady, turned / dt, 0.0),
            starts=start,
            starts_store=torch.where(start, fraction, 0.0),
            starts_low=start & ~ready,
            recoveries=recovered,
            recovery_seconds=torch.where(recovered, since, 0.0),
            recoveries_cut=cut,
        )
        for name, value in sums.items():
            pace[name].add_(value.float().sum(-1))
        since = torch.where(recovered | cut | ~alive | (engaged & ready), -1.0, since)
        pace["since_fight"].copy_(since)
        pace["was_engaged"].copy_(engaged)
        pace["was_ready"].copy_(ready | ~alive)
        pace["heading_cos"].copy_(cos)
        pace["heading_sin"].copy_(sin)

    return stage


def card_with(base: api.ShipTypes, index: int, overrides: dict, knee: float) -> api.ShipTypes:
    """The base card with type `index`'s fields overridden and every type's knee scale."""
    fields = {}
    for field in dataclasses.fields(base):
        value = getattr(base, field.name).clone()
        if field.name in overrides:
            value[index] = overrides[field.name]
        if field.name == "engine_knee_scale":
            value[:] = knee
        fields[field.name] = value
    return api.ShipTypes(**fields)


def main() -> None:
    """The variants the module describes, one line each."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--variants", required=True)
    parser.add_argument("--preset", default="real_scale_deathmatch")
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3_000_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hours", type=float, default=1.0, help="the run's bound")
    args = parser.parse_args()
    device = torch.device("cuda")
    game = api.PRESETS[args.preset]
    names = api.type_names(game.ship_set)
    base = api.ship_types(device, ship_set=game.ship_set)
    rules = api.rules_of(game)
    measured = characteristics(base, rules)
    variants = json.loads(open(args.variants).read())
    done = set()
    try:
        done = {json.loads(line)["name"] for line in open(args.out)}
    except FileNotFoundError:
        pass
    S = game.ships_per_team
    agent = Scripted(measured, S, MODE_DEFAULTS)
    rollout = None
    started = time.time()
    with bench_lock(), open(args.out, "a") as out:
        for number, variant in enumerate(variants):
            if variant["name"] in done:
                continue
            index = names.index(variant["type"])
            card = card_with(base, index, variant.get("card", {}), variant.get("knee", 1.0))
            card = api.ShipTypes(**{k: v.to(device) for k, v in vars(card).items()})
            lineup = torch.full((args.games, game.num_ships), index, dtype=torch.int32)
            if rollout is None:
                rollout = Rollout(
                    game,
                    card,
                    (Pool("pace", 1.0, ("a", "b")),),
                    {"a": agent, "b": agent},
                    "a",
                    args.games,
                    32,
                    device,
                    staggered=False,
                    ship_type=lineup,
                )
                rollout.carry["pace"] = blank_pace(args.games, game.num_ships, device)
                cruise = measured.cruise_speed.clone()
                gun = measured.gun_range.clone()
                stage = pace_stage(rollout.rules, gun, cruise, S)
                rollout.stages = rollout.stages + (
                    torch.compile(stage, fullgraph=True, dynamic=False),
                )
                live_gun, live_cruise = gun, cruise
            held = rollout.carry["types"]
            for field in dataclasses.fields(held):
                getattr(held, field.name).copy_(getattr(card, field.name))
            now = characteristics(card, rules)
            for field in dataclasses.fields(now):
                getattr(agent.characteristics, field.name).copy_(getattr(now, field.name))
            live_gun.copy_(now.gun_range)
            live_cruise.copy_(now.cruise_speed)
            rollout.ship_type.copy_(lineup)
            torch.manual_seed(args.seed)
            pace = rollout.carry.pop("pace")  # restart rebuilds the rollout's own carry
            rollout.restart()
            rollout.carry["pace"] = pace
            fresh = blank_pace(args.games, game.num_ships, device)
            for name, value in rollout.carry["pace"].items():  # in place: the graph reads them
                value.copy_(fresh[name])
            reset_game_sums(rollout.carry["game_sums"])
            ticks = int(rollout.episode_ticks.max())
            for _ in range(math.ceil(ticks / rollout.length)):
                rollout.run()
            sums = {
                name: float(v.sum()) for name, v in rollout.carry["pace"].items() if name in SUMS
            }
            resolve = dict(
                zip(RESOLVE_COLUMNS, rollout.carry["game_sums"]["resolve"].sum(0).tolist())
            )
            minutes = resolve["environment_ticks"] * rollout.rules.tick_length / 60.0
            dt = rollout.rules.tick_length
            alive = max(sums["alive"], 1.0)
            engaged = max(sums["engaged"], 1.0)
            starts = max(sums["starts"], 1.0)
            row = dict(
                name=variant["name"],
                type=variant["type"],
                knee=variant.get("knee", 1.0),
                games=args.games,
                engaged_share=sums["engaged"] / alive,
                recovering_share=sums["recovering"] / alive,
                ready_share=sums["ready"] / alive,
                engaged_store=sums["engaged_store"] / engaged,
                start_store=sums["starts_store"] / starts,
                starts_low_share=sums["starts_low"] / starts,
                recovery_seconds=sums["recovery_seconds"] / max(sums["recoveries"], 1.0),
                recoveries=sums["recoveries"],
                recoveries_cut=sums["recoveries_cut"],
                engaged_speed_over_cruise=sums["engaged_speed"] / engaged,
                engaged_turn_degrees=math.degrees(sums["engaged_turn"] / engaged),
                kills_per_minute=resolve["combat_deaths"] / minutes,
                engaged_seconds_per_kill=sums["engaged"] * dt / max(resolve["combat_deaths"], 1.0),
                shots_per_minute=resolve["episode_event/shots_fired"] / minutes
                if "episode_event/shots_fired" in resolve
                else None,
                seconds=time.time() - started,
            )
            out.write(json.dumps(row) + "\n")
            out.flush()
            left = (time.time() - started) / (number + 1) * (len(variants) - number - 1)
            print(
                json.dumps(row),
                f"ETA {time.strftime('%H:%M', time.localtime(time.time() + left))}",
                flush=True,
            )
            if time.time() - started > 3600.0 * args.hours:
                raise SystemExit(f"past the {args.hours} h bound: rerun to resume")


if __name__ == "__main__":
    main()
