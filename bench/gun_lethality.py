"""How lethal the guns are: 1v1 duels of the real-scale set, the scripted agent (its
real-scale deathmatch set, T = 1) on both sides, batched on the CPU, uncompiled.

    PYTHONPATH=src python -m bench.gun_lethality play --record DIR/duels.jsonl
    PYTHONPATH=src python -m bench.gun_lethality report --record DIR/duels.jsonl --out DIR

The duel: real_scale_deathmatch at one ship a side, no fields (the guns alone), on the 1v1
size of the real-scale map by the fleet presets' density rule (7800 sqrt(1/5) = 3488 m).
Each environment counts its first kill only: both ships spawn at full shield, so that
kill is a duel from a neutral start. Half the environments seat the first type on team 0.
An environment ends at its first kill or at `--seconds` (no kill, a timeout).

The merge: the first tick both ships are alive within the pair's longer gun range, each
nose within `MERGE_CONE` (30 degrees) of the line to the other (closing head-on). It ends
when the ships have passed (either nose more than 90 degrees off the line to the other)
or opened beyond 1.2 gun ranges, and its window runs `MERGE_GRACE` (2 s) after that. A
first kill in the window is a first-merge kill; before the merge starts, a kill before any
merge.

`play`: every matchup (the set's mirrors, then the cross pairs) a batch of
`--environments` duels, one line of the record each, resumed by matchup. `report`: the
table per type (Markdown, printed, and `gun_lethality.json` in `--out`), with the card's
closed forms: time to kill from full shield at 100% hits, head-on (the target's head-on
fraction) and from the tail, (hits needed - 1) / fire rate; the 180-degree turn at the
sustained turn rate (the settled full-stick coasting turn) and at the instantaneous one
(the g limit at the corner speed, a_max / v_corner); a head-on pass's time in gun range,
gun range / (2 cruise).
"""

import argparse
import dataclasses
import datetime
import json
import math
import pathlib
import time
import zoneinfo

import torch

from bnb.contracts import Action, Obs
from bnb.controllers import scripted
from bnb.game import api
from bnb.game.handling.characteristics import characteristics

PRESET = "real_scale_deathmatch"
ARENA_RADIUS = round(api.PRESETS[PRESET].arena_radius * (1 / 5) ** 0.5, 1)
MERGE_CONE = math.radians(30.0)
PASSED = math.radians(90.0)
OPEN_FACTOR = 1.2
MERGE_GRACE = 2.0  # s
CROSS_PAIRS = (
    ("vizia", "P-51D"),
    ("vizia", "Spitfire XIV"),
    ("Me 262", "Zero"),
    ("P-47D", "Spitfire IX"),
    ("Fw 190A", "Yak-3"),
    ("P-51D", "Spitfire XIV"),
)
KST = zoneinfo.ZoneInfo("Asia/Seoul")


def _clock(seconds: float) -> str:
    return (datetime.datetime.now(KST) + datetime.timedelta(seconds=seconds)).strftime("%H:%M")


def setup():
    game = dataclasses.replace(
        api.PRESETS[PRESET], ships_per_team=1, num_fields=0, arena_radius=ARENA_RADIUS
    )
    types = api.ship_types(ship_set=game.ship_set)
    rules = api.rules_of(game, types=types)
    return game, types, rules


def duels(rules, types, first: int, second: int, environments: int, seconds: float, seed: int):
    """One batch of duels of type `first` against `second`; per environment arrays."""
    torch.manual_seed(seed)
    B = environments
    draws = torch.Generator().manual_seed(seed + 1)
    seat = torch.arange(B) % 2  # the team the first type sits on
    ship_type = torch.stack(
        [torch.where(seat == 0, first, second), torch.where(seat == 0, second, first)], 1
    ).int()
    state = api.reset(rules, types, api.uniforms(rules, B, generator=draws), ship_type=ship_type)
    state, observation, _ = api.perceive(rules, state)
    agent = scripted.Scripted(characteristics(types, rules), 1)
    carry = agent.init_carry(2 * B, torch.device("cpu"))
    gun_range = characteristics(types, rules).gun_range
    reach = torch.maximum(gun_range[first], gun_range[second])
    full_damage = types.hit_damage[ship_type.long()]  # (B, 2) each shooter's bullet
    ticks = int(round(seconds / rules.tick_length))
    grace = int(round(MERGE_GRACE / rules.tick_length))
    never = torch.full((B,), -1, dtype=torch.long)
    merge_start, merge_end = never.clone(), never.clone()
    kill_tick, killer = never.clone(), never.clone()  # killer: the team, -2 the boundary
    first_hit_tick, first_hitter = never.clone(), never.clone()
    victim_first_hit = torch.full((B, 2), -1, dtype=torch.long)
    shots = torch.zeros((B, 2))
    hits = torch.zeros((B, 2))
    damage_dealt = torch.zeros((B, 2))
    front_damage = torch.zeros((B, 2))  # dealt at aspect below the half point
    half_point = (1.0 + rules.head_on_fraction * types.head_on_fraction_scale) / 2.0
    target_half = half_point[ship_type.long().flip(1)]  # (B, 2) the target's, per shooter
    done = torch.zeros(B, dtype=torch.bool)
    with torch.no_grad():
        for tick in range(ticks):
            view = Obs(
                **{
                    name: getattr(observation, name).flatten(0, 1)
                    for name in ("ships", "fields", "globals", "zones")
                }
            )
            action, carry, _ = agent.step(carry, view, torch.full((2 * B,), tick == 0))
            action = Action.unpack(action.packed().reshape(B, 2, 3).to(torch.int32))
            moved, sweep = api.move(rules, types, state, action, api.spread(rules, B, generator=draws))
            state, events = api.resolve(
                rules, types, moved, sweep, action, api.uniforms(rules, B, generator=draws)
            )
            state, observation, _ = api.perceive(rules, state)
            live = ~done
            shown = api.display(state)
            x, y = shown["ships.x"], shown["ships.y"]
            cos, sin = shown["ships.heading_cos"], shown["ships.heading_sin"]
            alive = shown["ships.alive"].bool()
            dx, dy = x[:, 1] - x[:, 0], y[:, 1] - y[:, 0]
            distance = torch.sqrt(dx * dx + dy * dy)
            off0 = torch.atan2((cos[:, 0] * dy - sin[:, 0] * dx).abs(), cos[:, 0] * dx + sin[:, 0] * dy)
            off1 = torch.atan2((cos[:, 1] * dy - sin[:, 1] * dx).abs(), -(cos[:, 1] * dx + sin[:, 1] * dy))
            both = alive.all(1)
            starting = live & both & (merge_start < 0) & (distance < reach)
            starting &= (off0 < MERGE_CONE) & (off1 < MERGE_CONE)
            merge_start = torch.where(starting, tick, merge_start)
            ending = live & (merge_start >= 0) & (merge_end < 0) & ~starting
            ending &= (off0 > PASSED) | (off1 > PASSED) | (distance > OPEN_FACTOR * reach) | ~both
            merge_end = torch.where(ending, tick, merge_end)
            # hits: damage[b, shooter, target]
            dealt = torch.stack([events.damage[:, 0, 1], events.damage[:, 1, 0]], 1)  # by shooter
            landed = (dealt > 0) & live[:, None]
            hits += landed.float()
            shots += (events.fired & live[:, None]).float()
            damage_dealt += torch.where(live[:, None], dealt, 0.0)
            aspect = dealt / full_damage  # one bullet a tick at these fire rates
            front_damage += torch.where(landed & (aspect < target_half), dealt, 0.0)
            any_hit = landed.any(1)
            new_first = any_hit & (first_hit_tick < 0)
            first_hit_tick = torch.where(new_first, tick, first_hit_tick)
            # an exact tie (both land on the same tick) counts as no first hitter: -3
            hitter = torch.where(landed.all(1), -3, landed[:, 1].long())
            first_hitter = torch.where(new_first, hitter, first_hitter)
            taken = landed.flip(1)  # by victim
            victim_first_hit = torch.where(taken & (victim_first_hit < 0), tick, victim_first_hit)
            died = events.died & live[:, None]
            dying = died.any(1)
            team_killer = torch.where(
                died.all(1),
                -3,  # both died on one tick
                torch.where(
                    died[:, 0], events.killer[:, 0].long(), events.killer[:, 1].long()
                ),
            )
            team_killer = torch.where(team_killer == -1, -2, team_killer)
            kill_tick = torch.where(dying, tick, kill_tick)
            killer = torch.where(dying, team_killer, killer)
            done |= dying
            if bool(done.all()):
                break
    # results in the first type's terms: who is "first" in each environment
    def by_type(values):  # (B, 2) by team -> (B, 2) [first type, second type]
        return torch.where((seat == 0)[:, None], values, values.flip(1))

    def team_to_type(team):  # team index -> 0 first type, 1 second, others kept
        return torch.where(team >= 0, (team != seat).long(), team)

    victim = torch.where(killer >= 0, 1 - killer, -1)
    victim_hit = torch.where(
        victim >= 0, victim_first_hit.gather(1, victim.clamp_min(0)[:, None])[:, 0], -1
    )
    return dict(
        merge_start=merge_start.tolist(),
        merge_end=merge_end.tolist(),
        kill_tick=kill_tick.tolist(),
        killer=team_to_type(killer).tolist(),
        first_hit_tick=first_hit_tick.tolist(),
        first_hitter=team_to_type(first_hitter).tolist(),
        victim_first_hit=victim_hit.tolist(),
        shots=by_type(shots).sum(0).tolist(),
        hits=by_type(hits).sum(0).tolist(),
        damage=by_type(damage_dealt).sum(0).tolist(),
        front_damage=by_type(front_damage).sum(0).tolist(),
        ticks=tick + 1,
        grace=grace,
    )


def play(args) -> None:
    game, types, rules = setup()
    names = api.type_names(game.ship_set)
    matchups = [(name, name) for name in names] + list(CROSS_PAIRS)
    record = pathlib.Path(args.record)
    record.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if record.exists():
        done = {tuple(json.loads(line)["pair"]) for line in record.read_text().splitlines()}
    index, count = (int(part) for part in args.shard.split("/"))
    todo = [pair for pair in matchups[index::count] if pair not in done]
    started = time.time()
    deadline = started + 3600.0 * args.hours
    print(
        f"{len(todo)} of {len(matchups)} matchups to play, {args.environments} duels each, "
        f"arena {ARENA_RADIUS} m, shard {args.shard}; bound {_clock(deadline - started)} KST",
        flush=True,
    )
    for number, (a, b) in enumerate(todo):
        if time.time() > deadline:
            raise SystemExit(f"past the {args.hours} h bound: resume from {record}")
        batch_started = time.time()
        result = duels(
            rules, types, names.index(a), names.index(b), args.environments, args.seconds,
            args.seed + matchups.index((a, b)),
        )
        result["pair"] = [a, b]
        result["environments"] = args.environments
        with record.open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        elapsed = time.time() - started
        left = elapsed / (number + 1) * (len(todo) - number - 1)
        print(
            f"{datetime.datetime.now(KST):%H:%M:%S} {a} v {b}: {time.time() - batch_started:.0f} s, "
            f"{result['ticks']} ticks; {number + 1}/{len(todo)}, ETA {_clock(left)} KST",
            flush=True,
        )


def card_table(types, rules):
    c = characteristics(types, rules)
    names = api.type_names("real-scale")
    f0 = rules.head_on_fraction * types.head_on_fraction_scale
    rows = {}
    for i, name in enumerate(names):
        damage, rate = float(types.hit_damage[i]), float(types.fire_rate[i])
        shield = float(types.max_shield[i])

        def ttk(aspect):  # a mirror: the type shooting itself
            return (math.ceil(shield / (damage * aspect) - 1e-9) - 1) / rate

        sustained = float(c.turn_rate[i, 2])
        instantaneous = float(c.max_acceleration[i] / c.corner_speed[i])
        rows[name] = dict(
            damage_per_second=damage * rate,
            max_shield=shield,
            head_on_fraction=float(f0[i]),
            ttk_head_on=ttk(float(f0[i])),
            ttk_tail=ttk(1.0),
            turn_180_sustained=math.pi / sustained,
            turn_180_instantaneous=math.pi / instantaneous,
            pass_time=float(c.gun_range[i] / (2.0 * c.cruise_speed[i])),
            gun_range=float(c.gun_range[i]),
            cruise_speed=float(c.cruise_speed[i]),
            sustained_deg=math.degrees(sustained),
            instantaneous_deg=math.degrees(instantaneous),
        )
    return rows


def share(count, total):
    p = count / max(total, 1)
    return p, math.sqrt(max(p * (1 - p), 1e-12) / max(total, 1))


def summarise(line, tick_length):
    k = torch.tensor(line["kill_tick"])
    ms, me = torch.tensor(line["merge_start"]), torch.tensor(line["merge_end"])
    killer = torch.tensor(line["killer"])
    first = torch.tensor(line["first_hitter"])
    victim_hit = torch.tensor(line["victim_first_hit"])
    B = len(k)
    shot = killer >= 0
    traded = killer == -3  # both died on one tick
    fought = shot | traded
    window_end = torch.where(me >= 0, me + line["grace"], torch.iinfo(torch.long).max)
    in_merge = fought & (ms >= 0) & (k >= ms) & (k <= window_end)
    before = fought & ((ms < 0) | (k < ms))
    decided = shot & (first >= 0)
    first_wins = decided & (first == killer)
    ttk = ((k - victim_hit)[shot & (victim_hit >= 0)].float() * tick_length)
    shots, hits = line["shots"], line["hits"]
    damage, front = line["damage"], line["front_damage"]
    return dict(
        pair=line["pair"],
        environments=B,
        shot_kills=int(shot.sum()),
        trades=int(traded.sum()),
        trade_share=share(int(traded.sum()), int(fought.sum())),
        boundary=int((killer == -2).sum()),
        timeouts=int((k < 0).sum()),
        merges=int((ms >= 0).sum()),
        first_merge_share=share(int(in_merge.sum()), int(fought.sum())),
        before_merge_share=share(int(before.sum()), int(fought.sum())),
        first_kill_seconds=float(k[fought].float().median() * tick_length) if fought.any() else None,
        first_hitter_wins=share(int(first_wins.sum()), int(decided.sum())),
        first_type_wins=share(int((killer[shot] == 0).sum()), int(shot.sum())),
        ttk_median=float(ttk.median()) if len(ttk) else None,
        ttk_quartiles=[float(ttk.quantile(0.25)), float(ttk.quantile(0.75))] if len(ttk) else None,
        hit_fraction=sum(hits) / max(sum(shots), 1),
        front_damage_share=sum(front) / max(sum(damage), 1e-9),
    )


def report(args) -> None:
    game, types, rules = setup()
    card = card_table(types, rules)
    lines = [
        json.loads(line)
        for record in args.record
        for line in pathlib.Path(record).read_text().splitlines()
    ]
    order = {name: number for number, name in enumerate(api.type_names(game.ship_set))}
    lines.sort(key=lambda line: (line["pair"][0] != line["pair"][1], order[line["pair"][0]]))
    rows = [summarise(line, rules.tick_length) for line in lines]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gun_lethality.json").write_text(
        json.dumps(dict(arena_radius=ARENA_RADIUS, card=card, duels=rows), indent=1)
    )
    print(
        "| type | TTK head-on (s) | TTK tail (s) | 180° sustained (s) | 180° instant (s) | "
        "TTK head-on / 180° instant | pass in range (s) | first-merge kills | trades | kills before a merge "
        "| P(win \\| first hit) | measured TTK median (s) | hit fraction | front damage share |"
    )
    print("|---" * 14 + "|")
    for row in rows:
        a, b = row["pair"]
        c = card[a]
        name = a if a == b else f"{a} v {b}"
        if a == "vizia" and a == b:
            name = "**vizia**"

        def pm(value):
            return f"{value[0]:.2f} ± {value[1]:.2f}"

        print(
            f"| {name} | {c['ttk_head_on']:.2f} | {c['ttk_tail']:.2f} | "
            f"{c['turn_180_sustained']:.1f} | {c['turn_180_instantaneous']:.1f} | "
            f"{c['ttk_head_on'] / c['turn_180_instantaneous']:.2f} | {c['pass_time']:.1f} | "
            f"{pm(row['first_merge_share'])} | {pm(row['trade_share'])} | "
            f"{pm(row['before_merge_share'])} | "
            f"{pm(row['first_hitter_wins'])} | {row['ttk_median']:.2f} | "
            f"{row['hit_fraction']:.2f} | {row['front_damage_share']:.2f} |"
        )
    for row in rows:
        if row["pair"][0] != row["pair"][1]:
            print(f"{row['pair']}: first type wins {row['first_type_wins'][0]:.2f} ± "
                  f"{row['first_type_wins'][1]:.2f}; timeouts {row['timeouts']}, boundary {row['boundary']}")
    timeouts = sum(row["timeouts"] for row in rows)
    boundary = sum(row["boundary"] for row in rows)
    print(f"timeouts {timeouts}, boundary deaths {boundary} of {sum(r['environments'] for r in rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("play")
    one.add_argument("--record", required=True)
    one.add_argument("--environments", type=int, default=512)
    one.add_argument("--seconds", type=float, default=120.0)
    one.add_argument("--seed", type=int, default=11)
    one.add_argument("--hours", type=float, default=3.0)
    one.add_argument("--shard", default="0/1", help="i/n: play every n-th matchup from the i-th")
    two = sub.add_parser("report")
    two.add_argument("--record", required=True, nargs="+")
    two.add_argument("--out", required=True)
    args = parser.parse_args()
    {"play": play, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
