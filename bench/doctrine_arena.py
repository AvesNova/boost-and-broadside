"""The doctrine arena: 5v5 games of the real-scale set's Yongma and Jujak, the scripted agent
on both sides with each side's per-type doctrines (`bnb.controllers.scripted_doctrine`),
batched on the CPU, uncompiled; and a search of one type's doctrine for strength.

    PYTHONPATH=src python -m bench.doctrine_arena play --record DIR/arena.jsonl \
        --label round1 [--doctrines FILE]
    PYTHONPATH=src python -m bench.doctrine_arena report --record DIR/arena.jsonl
    PYTHONPATH=src python -m bench.doctrine_arena search --type yongma --log DIR/search.jsonl

The game: elimination (`real_scale_elimination`: no respawn, a team with no ship left loses,
at the 180 s clock more ships alive wins, then more total shield) at 5v5 with no fields
(the scripted agent ignores fields), on the real-scale map (7800 m, the 5v5 size of the
fleet presets' density rule, as in real_scale_deathmatch). Elimination rather than the
respawn deathmatch: every game is one fight from full shields to a resolution, so a
timeout is a fight that did not resolve, and a game costs a third of a deathmatch's.

The lineups (`LINEUPS`): energy v turn (all Yongma against all Jujak), the Jujak mirror, the
Yongma mirror, and the mixed lineup of training (types alternating by slot, both teams the
same: Yongma, Jujak, Yongma, Jujak, Yongma in half the games and the reverse in the
other). Side A and side B each fly their own doctrine table; A sits on team 0 in half the
games. A record line is one batch of one lineup: the games' results for A, their lengths
and timeouts, the enemy damage by aspect class (`bnb.contracts.ASPECTS`) by victim type,
and every kill: its killer's side and type, the victim's type, the killing hit's aspect
class, the head-on passes (merges) of the pair before it, whether it fell in the pair's first
merge, whether the killer landed the pair's first hit, the time from the victim's first
hit taken, and the victim's shield recharges since then.

The merge (as in `bench.gun_lethality`): for an enemy pair, the first tick both are alive
within the longer gun range of the two with each nose within 30 degrees of the line to the
other; it ends when either nose is more than 90 degrees off that line or the range opens
past 1.2 gun ranges. A kill within the pair's first merge or 2 s after it is a first-merge
kill. A pair's merges count from when both last spawned.

`search`: one type's doctrine against the champion's (the doctrines file's, both types), a
(1 + lambda) evolution: each generation the parent and `--children` perturbations of it
play `--games` games in each lineup the type flies (energy v turn, its mirror, mixed), as
side A against the champion as side B; a candidate's score is its mean over the three.
The parent replays every generation, so its score accumulates; a child whose score beats
the parent's becomes the parent. Every candidate is a line of the log, which resumes.
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

from bnb.contracts import ASPECTS, Action, Obs
from bnb.controllers import scripted
from bnb.controllers.scripted_doctrine import (
    DOCTRINES,
    EXPONENTS,
    NAMES,
    SCALE_FLOORS,
    WEIGHTS,
    Doctrine,
    from_named,
    from_row,
    named,
    pack,
    table,
)
from bnb.controllers.scripted_parameters import Logistic
from bnb.game import api, ships
from bnb.game.handling.characteristics import characteristics

PRESET = "real_scale_elimination"
SHIP_SET = "real-scale"
ENERGY, TURN = "yongma", "jujak"
KEYS = (ENERGY, TURN)
LINEUPS = {
    "energy v turn": ([ENERGY] * 5, [TURN] * 5),
    "turn mirror": ([TURN] * 5, [TURN] * 5),
    "energy mirror": ([ENERGY] * 5, [ENERGY] * 5),
    "mixed": ([ENERGY, TURN, ENERGY, TURN, ENERGY], [ENERGY, TURN, ENERGY, TURN, ENERGY]),
}
MERGE_CONE = math.radians(30.0)
PASSED = math.radians(90.0)
OPEN_FACTOR = 1.2
MERGE_GRACE = 2.0  # s
KST = zoneinfo.ZoneInfo("Asia/Seoul")


def clock(seconds: float) -> str:
    """The KST clock time `seconds` from now."""
    return (datetime.datetime.now(KST) + datetime.timedelta(seconds=seconds)).strftime("%H:%M")


def setup(preset: str = PRESET):
    """The arena's game (5v5, no fields), card and rules, and the set's type keys."""
    game = dataclasses.replace(api.PRESETS[preset], num_fields=0)
    types = api.ship_types(ship_set=SHIP_SET)
    rules = api.rules_of(game, types=types)
    keys = [ship.key for ship in ships.load(SHIP_SET)]
    return game, types, rules, keys


def read_doctrines(path: pathlib.Path, keys: list[str]) -> torch.Tensor:
    """(T, WIDTH) a doctrines file's set (`scripted_doctrines.json`'s shape) as a table."""
    stored = json.loads(path.read_text()).get(SHIP_SET, {})
    by_key = {key: from_named(entry["doctrine"]) for key, entry in stored.items()}
    return table([by_key.get(key, Doctrine()) for key in keys])


def lineup_types(lineup: str, keys: list[str], environments: int) -> tuple:
    """((B, N) int32 each environment's ship types, (B,) bool side A on team 0): A on team 0
    in the even environments; the mixed lineup reversed in every other pair."""
    first, second = ([keys.index(key) for key in team] for team in LINEUPS[lineup])
    index = torch.arange(environments)
    a_on_zero = index % 2 == 0
    first_t, second_t = torch.tensor(first), torch.tensor(second)
    if lineup == "mixed":
        reverse = (index // 2) % 2 == 1
        swap = {keys.index(ENERGY): keys.index(TURN), keys.index(TURN): keys.index(ENERGY)}
        flipped = torch.tensor([swap[t] for t in first])
        a_types = torch.where(reverse[:, None], flipped, first_t)
        b_types = a_types
    else:
        a_types = first_t.expand(environments, -1)
        b_types = second_t.expand(environments, -1)
    team0 = torch.where(a_on_zero[:, None], a_types, b_types)
    team1 = torch.where(a_on_zero[:, None], b_types, a_types)
    return torch.cat([team0, team1], 1).to(torch.int32), a_on_zero


COMPACT_EVERY = 60  # ticks: how often the finished games leave the batch


def selected(value, keep: torch.Tensor, size: int):
    """A batch's tree (dataclasses, tuples, tensors) with the rows `keep` (bool, (size,)) of
    every tensor whose first axis is the batch's."""
    if isinstance(value, torch.Tensor):
        return value[keep] if value.dim() > 0 and value.shape[0] == size else value
    if dataclasses.is_dataclass(value):
        fields = dataclasses.fields(value)
        return type(value)(**{f.name: selected(getattr(value, f.name), keep, size) for f in fields})
    if isinstance(value, tuple):
        return type(value)(*(selected(part, keep, size) for part in value))
    return value


def games(
    rules,
    types,
    keys: list[str],
    ship_type: torch.Tensor,
    a_on_zero: torch.Tensor,
    doctrines_a: torch.Tensor,
    doctrines_b: torch.Tensor,
    episode_ticks: int,
    seed: int,
) -> dict:
    """One batch of games to their first end, A's doctrines (T, W) or per environment
    (B, T, W) against B's; per environment arrays and the kills. Every COMPACT_EVERY ticks
    the finished games leave the batch (`selected`), so a batch costs its games' lengths,
    not its longest's."""
    torch.manual_seed(seed)
    B, N = ship_type.shape
    S = N // 2
    draws = torch.Generator().manual_seed(seed + 1)
    state = api.reset(rules, types, api.uniforms(rules, B, generator=draws), ship_type=ship_type)
    state, observation, _ = api.perceive(rules, state)
    card = characteristics(types, rules)

    def per_environment(table_: torch.Tensor) -> torch.Tensor:
        return table_ if table_.dim() == 3 else table_[None].expand(B, -1, -1)

    a_rows, b_rows = per_environment(doctrines_a), per_environment(doctrines_b)
    team_is_a = torch.stack([a_on_zero, ~a_on_zero], 1)  # (B, 2)
    view_doctrines = torch.where(
        team_is_a[..., None, None], a_rows[:, None], b_rows[:, None]
    ).flatten(0, 1)  # (2B, T, W)
    agent = scripted.Scripted(card, S, doctrines=view_doctrines)
    carry = agent.init_carry(2 * B, torch.device("cpu"))
    ship_side_a = torch.cat(
        [a_on_zero[:, None].expand(B, S), (~a_on_zero)[:, None].expand(B, S)], 1
    )  # (B, N) the ship flies for A
    reach_by_type = card.gun_range
    grace = int(round(MERGE_GRACE / rules.tick_length))
    team0, team1 = torch.arange(S), torch.arange(S, N)
    pair_reach = torch.maximum(
        reach_by_type[ship_type[:, :S].long()][:, :, None],
        reach_by_type[ship_type[:, S:].long()][:, None, :],
    )  # (B, S, S): team 0 ship i against team 1 ship j
    in_merge = torch.zeros((B, S, S), dtype=torch.bool)
    merges = torch.zeros((B, S, S), dtype=torch.long)
    first_merge_end = torch.full((B, S, S), -1, dtype=torch.long)
    first_hit = torch.full((B, S, S), -1, dtype=torch.long)  # 0: i first, 1: j first, 2: tie
    victim_hit_tick = torch.full((B, N), -1, dtype=torch.long)
    recharges = torch.zeros((B, N), dtype=torch.long)
    restoring = torch.zeros((B, N), dtype=torch.bool)
    done = torch.zeros(B, dtype=torch.bool)
    environment = torch.arange(B)  # each row's game
    result = torch.zeros(B, dtype=torch.long)
    length = torch.zeros(B, dtype=torch.long)
    first_game_hit = torch.full((B,), -1, dtype=torch.long)  # the side (0 A, 1 B) to hit first
    games_first_hit = first_game_hit.clone()  # by game, kept as rows leave
    damage_by_victim_type = torch.zeros((len(KEYS), len(ASPECTS)), dtype=torch.float64)
    damage_by_shooter_side = torch.zeros((2, len(ASPECTS)), dtype=torch.float64)
    restored = torch.zeros((), dtype=torch.float64)  # the shield every ship recharged
    type_slot = torch.full((len(keys),), -1, dtype=torch.long)
    for slot, key in enumerate(KEYS):
        type_slot[keys.index(key)] = slot
    victim_slot = type_slot[ship_type.long()]  # (B, N)
    kills = []
    with torch.no_grad():
        for tick in range(episode_ticks):
            view = Obs(
                **{
                    name: getattr(observation, name).flatten(0, 1)
                    for name in ("ships", "fields", "globals", "zones")
                }
            )
            rows = done.shape[0]
            action, carry, _ = agent.step(carry, view, torch.full((2 * rows,), tick == 0))
            action = Action.unpack(action.packed().reshape(rows, N, 3).to(torch.int32))
            moved, sweep = api.move(
                rules, types, state, action, api.spread(rules, rows, generator=draws)
            )
            state, events = api.resolve(
                rules, types, moved, sweep, action, api.uniforms(rules, rows, generator=draws)
            )
            live = ~done
            shown = api.display(state)
            x, y = shown["ships.x"], shown["ships.y"]
            cos, sin = shown["ships.heading_cos"], shown["ships.heading_sin"]
            alive = shown["ships.alive"].bool() & ~events.done[:, None]

            # merges of every enemy pair
            dx = x[:, team1][:, None, :] - x[:, team0][:, :, None]  # (B, S, S) i -> j
            dy = y[:, team1][:, None, :] - y[:, team0][:, :, None]
            distance = torch.sqrt(dx * dx + dy * dy)
            ci, si = cos[:, team0][:, :, None], sin[:, team0][:, :, None]
            cj, sj = cos[:, team1][:, None, :], sin[:, team1][:, None, :]
            off_i = torch.atan2((ci * dy - si * dx).abs(), ci * dx + si * dy)
            off_j = torch.atan2((cj * dy - sj * dx).abs(), -(cj * dx + sj * dy))
            both = alive[:, team0][:, :, None] & alive[:, team1][:, None, :]
            starting = both & ~in_merge & (distance < pair_reach)
            starting &= (off_i < MERGE_CONE) & (off_j < MERGE_CONE)
            ending = in_merge & (
                (off_i > PASSED) | (off_j > PASSED) | (distance > OPEN_FACTOR * pair_reach) | ~both
            )
            first_end = ending & (merges == 1) & (first_merge_end < 0)
            first_merge_end = torch.where(first_end, tick, first_merge_end)
            merges = merges + starting.long()
            in_merge = (in_merge | starting) & ~ending

            # hits
            damage = events.damage
            hit_ij = damage[:, team0][:, :, team1] > 0.0
            hit_ji = (damage[:, team1][:, :, team0] > 0.0).transpose(1, 2)
            new_first = (first_hit < 0) & (hit_ij | hit_ji)
            first_hit = torch.where(
                new_first, torch.where(hit_ij & hit_ji, 2, hit_ji.long()), first_hit
            )
            taken = damage.sum(1) > 0.0  # (B, N) hit by anyone this tick
            enemy_taken = events.enemy_damage_by_aspect.sum(-1) > 0.0
            victim_hit_tick = torch.where(
                enemy_taken & (victim_hit_tick < 0) & live[:, None], tick, victim_hit_tick
            )
            hitter_side_a = (hit_ij.any(2) & ship_side_a[:, team0]).any(1) | (
                hit_ji.any(1) & ship_side_a[:, team1]
            ).any(1)
            hitter_side_b = (hit_ij.any(2) & ~ship_side_a[:, team0]).any(1) | (
                hit_ji.any(1) & ~ship_side_a[:, team1]
            ).any(1)
            first_game = live & (first_game_hit < 0) & (hitter_side_a | hitter_side_b)
            first_game_hit = torch.where(
                first_game,
                torch.where(hitter_side_a & hitter_side_b, 2, hitter_side_b.long()),
                first_game_hit,
            )
            del taken

            # damage by aspect: by the victim's type, and by the shooter's side (the
            # victim's other side)
            by_aspect = torch.where(live[:, None, None], events.enemy_damage_by_aspect, 0.0)
            for slot in range(len(KEYS)):
                mine = (victim_slot == slot)[..., None]
                damage_by_victim_type[slot] += torch.where(mine, by_aspect, 0.0).sum((0, 1))
            damage_by_shooter_side[0] += torch.where(~ship_side_a[..., None], by_aspect, 0.0).sum(
                (0, 1)
            )
            damage_by_shooter_side[1] += torch.where(ship_side_a[..., None], by_aspect, 0.0).sum(
                (0, 1)
            )

            # shield recharges: each spell's start
            spell = events.shield_restored > 0.0
            restored += torch.where(live[:, None], events.shield_restored, 0.0).sum()
            recharges = recharges + (spell & ~restoring & (victim_hit_tick >= 0)).long()
            restoring = spell

            # kills
            died = events.died & live[:, None]
            killer = events.killer.long()
            for b, v in died.nonzero().tolist():
                k = int(killer[b, v])
                record = dict(
                    environment=int(environment[b]),
                    victim_type=KEYS[int(victim_slot[b, v])],
                    victim_side="A" if bool(ship_side_a[b, v]) else "B",
                    aspect=int(events.kill_aspect[b, v]),
                    tick=tick,
                    seconds_from_first_hit=(
                        (tick - int(victim_hit_tick[b, v])) * rules.tick_length
                        if int(victim_hit_tick[b, v]) >= 0
                        else None
                    ),
                    recharges=int(recharges[b, v]),
                )
                if k >= 0 and (k < S) != (v < S):
                    i, j = (k, v - S) if k < S else (v, k - S)
                    count = int(merges[b, i, j])
                    end = int(first_merge_end[b, i, j])
                    in_first = count == 1 and (end < 0 or tick <= end + grace)
                    first = int(first_hit[b, i, j])
                    killer_first = first == 2 or first == (0 if k < S else 1)
                    record.update(
                        killer_type=KEYS[int(victim_slot[b, k])],
                        killer_side="A" if bool(ship_side_a[b, k]) else "B",
                        passes=count,
                        first_merge=in_first,
                        killer_first_hit=killer_first,
                    )
                kills.append(record)
            # a dead ship's pairs start again when it spawns
            gone = (died | events.spawned) | ~shown["ships.alive"].bool()
            gone_i, gone_j = gone[:, team0][:, :, None], gone[:, team1][:, None, :]
            reset_pair = gone_i | gone_j
            merges = torch.where(reset_pair, 0, merges)
            in_merge = in_merge & ~reset_pair
            first_merge_end = torch.where(reset_pair, -1, first_merge_end)
            first_hit = torch.where(reset_pair, -1, first_hit)
            victim_hit_tick = torch.where(gone, -1, victim_hit_tick)
            recharges = torch.where(gone, 0, recharges)

            ending_now = events.done & live
            team0_result = events.result.long()
            outcome = torch.where(a_on_zero, team0_result, -team0_result)
            ended = environment[ending_now]
            result[ended] = outcome[ending_now]
            length[ended] = tick + 1
            games_first_hit[ended] = first_game_hit[ending_now]
            done |= events.done
            state, observation, _ = api.perceive(rules, state)
            if bool(done.all()):
                break
            if tick % COMPACT_EVERY == 0 and bool(done.any()):
                keep, rows = ~done, done.shape[0]
                views = keep.repeat_interleave(2)
                state, observation = selected(state, keep, rows), selected(observation, keep, rows)
                carry = selected(carry, views, 2 * rows)
                agent.doctrines = agent.doctrines[views]
                (
                    environment,
                    ship_side_a,
                    a_on_zero,
                    pair_reach,
                    in_merge,
                    merges,
                    first_merge_end,
                    first_hit,
                    victim_hit_tick,
                    recharges,
                    restoring,
                    victim_slot,
                    first_game_hit,
                    done,
                ) = (
                    tensor[keep]
                    for tensor in (
                        environment,
                        ship_side_a,
                        a_on_zero,
                        pair_reach,
                        in_merge,
                        merges,
                        first_merge_end,
                        first_hit,
                        victim_hit_tick,
                        recharges,
                        restoring,
                        victim_slot,
                        first_game_hit,
                        done,
                    )
                )
    return dict(
        result=result.tolist(),
        length=length.tolist(),
        timeout=(length >= episode_ticks).tolist(),
        first_game_hit=games_first_hit.tolist(),
        damage_by_victim_type=damage_by_victim_type.tolist(),
        damage_by_shooter_side=damage_by_shooter_side.tolist(),
        restored=float(restored),
        kills=kills,
    )


def play(args) -> None:
    """Every lineup not yet in the record for this label, one line each."""
    game, types, rules, keys = setup(args.preset)
    doctrines = pathlib.Path(args.doctrines)
    side_a = read_doctrines(doctrines, keys)
    side_b = read_doctrines(pathlib.Path(args.doctrines_b or doctrines), keys)
    if args.off_a:
        side_a = table([Doctrine()] * len(keys))
    record = pathlib.Path(args.record)
    record.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if record.exists():
        done = {
            (line["label"], line["lineup"], line["batch"])
            for line in map(json.loads, record.read_text().splitlines())
        }
    todo = [
        (lineup, batch)
        for lineup in (args.lineups or LINEUPS)
        for batch in range(args.batches)
        if (args.label, lineup, batch) not in done
    ]
    started = time.time()
    deadline = started + 3600.0 * args.hours
    episode_ticks = int(round(game.episode_seconds / rules.tick_length))
    print(
        f"{len(todo)} batches of {args.environments} games ({args.label}); bound "
        f"{clock(deadline - started)} KST",
        flush=True,
    )
    for number, (lineup, batch) in enumerate(todo):
        if time.time() > deadline:
            raise SystemExit(f"past the {args.hours} h bound: resume from {record}")
        batch_started = time.time()
        ship_type, a_on_zero = lineup_types(lineup, keys, args.environments)
        seed = args.seed + 1000 * batch + list(LINEUPS).index(lineup)
        result = games(
            rules, types, keys, ship_type, a_on_zero, side_a, side_b, episode_ticks, seed
        )
        result.update(label=args.label, lineup=lineup, batch=batch, environments=args.environments)
        with record.open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        left = (time.time() - started) / (number + 1) * (len(todo) - number - 1)
        score = summary([result])["score"]
        print(
            f"{datetime.datetime.now(KST):%H:%M:%S} {lineup} batch {batch}: A scores "
            f"{score[0]:.3f} +- {score[1]:.3f}, {time.time() - batch_started:.0f} s; "
            f"{number + 1}/{len(todo)}, ETA {clock(left)} KST (worst {clock(1.3 * left)})",
            flush=True,
        )


def share(count: float, total: float) -> tuple[float, float]:
    """A share and its binomial standard error."""
    p = count / max(total, 1)
    return p, math.sqrt(max(p * (1 - p), 1e-12) / max(total, 1))


def summary(lines: list[dict]) -> dict:
    """The statistics of one lineup's lines."""
    results = [r for line in lines for r in line["result"]]
    n = len(results)
    points = [1.0 if r > 0 else 0.5 if r == 0 else 0.0 for r in results]
    mean = sum(points) / max(n, 1)
    variance = sum((p - mean) ** 2 for p in points) / max(n - 1, 1)
    kills = [k for line in lines for k in line["kills"]]
    shot = [k for k in kills if "passes" in k]
    damage = torch.tensor([line["damage_by_shooter_side"] for line in lines]).sum(0)
    by_victim = torch.tensor([line["damage_by_victim_type"] for line in lines]).sum(0)
    kill_counts = torch.zeros(len(ASPECTS))
    for k in shot:
        if k["aspect"] >= 0:
            kill_counts[k["aspect"]] += 1
    timeouts = sum(t for line in lines for t in line["timeout"])
    hits = [g for line in lines for g in line["first_game_hit"]]
    decided = [(g, r) for g, r in zip(hits, results, strict=True) if g in (0, 1) and r != 0]
    first_wins = sum(1 for g, r in decided if (g == 0) == (r > 0))
    lengths = sorted(x for line in lines for x in line["length"])
    ttk = sorted(k["seconds_from_first_hit"] for k in shot if k["seconds_from_first_hit"])
    total = damage.sum().clamp_min(1e-9)
    return dict(
        games=n,
        score=(mean, math.sqrt(variance / max(n, 1))),
        timeouts=share(timeouts, n),
        median_seconds=lengths[len(lengths) // 2] / 30.0 if lengths else None,
        kills_per_game=len(shot) / max(n, 1),
        damage_share=(damage.sum(0) / total).tolist(),
        damage_share_by_side=(damage / damage.sum(1, keepdim=True).clamp_min(1e-9)).tolist(),
        damage_share_by_victim_type={
            key: (row / row.sum().clamp_min(1e-9)).tolist()
            for key, row in zip(KEYS, by_victim, strict=True)
        },
        kill_share=(kill_counts / kill_counts.sum().clamp_min(1.0)).tolist(),
        kills=len(shot),
        passes_per_kill=sum(k["passes"] for k in shot) / max(len(shot), 1),
        no_merge_kills=share(sum(1 for k in shot if k["passes"] == 0), len(shot)),
        first_merge_kills=share(sum(1 for k in shot if k["first_merge"]), len(shot)),
        killer_first_hit=share(sum(1 for k in shot if k["killer_first_hit"]), len(shot)),
        first_hit_wins_game=share(first_wins, len(decided)),
        recharges_per_kill=sum(k["recharges"] for k in shot) / max(len(shot), 1),
        restored_per_game=sum(line.get("restored", 0.0) for line in lines) / max(n, 1),
        ttk_median=ttk[len(ttk) // 2] if ttk else None,
        kills_by_side={
            side: sum(1 for k in shot if k["killer_side"] == side) for side in ("A", "B")
        },
    )


def report(args) -> None:
    """A table per label: each lineup's statistics, printed (Markdown) and as JSON."""
    lines = [
        json.loads(line)
        for record in args.record
        for line in pathlib.Path(record).read_text().splitlines()
    ]
    labels = list(dict.fromkeys(line["label"] for line in lines))
    out = {}
    for label in labels:
        print(f"\n### {label}\n")
        print(
            "| Lineup | Games | A scores | Timeouts | Median s | Kills a game | Damage "
            "mutual / offset / flank / tail | Kills mutual / offset / flank / tail | Passes "
            "a kill | No-merge kills | First-merge kills | Killer hit first | First hit wins "
            "game | Recharges a kill | Shield recharged a game | TTK median s |"
        )
        print("|---" * 16 + "|")
        out[label] = {}
        for lineup in LINEUPS:
            mine = [line for line in lines if line["label"] == label and line["lineup"] == lineup]
            if not mine:
                continue
            row = summary(mine)
            out[label][lineup] = row

            def pm(value):
                return f"{value[0]:.2f} ± {value[1]:.2f}"

            def shares(values):
                return " / ".join(f"{v:.2f}" for v in values)

            ttk = row["ttk_median"]
            print(
                f"| {lineup} | {row['games']} | {pm(row['score'])} | {pm(row['timeouts'])} "
                f"| {row['median_seconds']:.0f} | {row['kills_per_game']:.1f} "
                f"| {shares(row['damage_share'])} | {shares(row['kill_share'])} "
                f"| {row['passes_per_kill']:.2f} | {pm(row['no_merge_kills'])} "
                f"| {pm(row['first_merge_kills'])} | {pm(row['killer_first_hit'])} "
                f"| {pm(row['first_hit_wins_game'])} | {row['recharges_per_kill']:.2f} "
                f"| {row['restored_per_game']:.0f} "
                f"| {ttk if ttk is None else round(ttk, 1)} |"
            )
        print()
        for lineup, row in out[label].items():
            by_side = " ; ".join(
                f"{side} {' / '.join(f'{v:.2f}' for v in values)}"
                for side, values in zip("AB", row["damage_share_by_side"], strict=True)
            )
            print(f"- {lineup}: damage dealt by side {by_side}; kills {row['kills_by_side']}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


def card(args) -> None:
    """The two cards' time-to-kill budget (card, 100% hits) for each shooter and target: the
    head-on time to kill (the target's head-on fraction) in seconds and in head-on passes
    (the pair's longer gun range over the two cruises), and the tail's; with the armour's
    half-strength angle and the shield's recharge."""
    game, types, rules, keys = setup(args.preset)
    c = characteristics(types, rules)
    fraction = rules.head_on_fraction * types.head_on_fraction_scale
    angle = rules.head_on_angle * types.head_on_angle_scale
    print("| Shooter on target | Head-on TTK (s) | Head-on passes | Tail TTK (s) |")
    print("|---|---|---|---|")
    for shooter in KEYS:
        for target in KEYS:
            i, j = keys.index(shooter), keys.index(target)
            damage, rate = float(types.hit_damage[i]), float(types.fire_rate[i])
            shield = float(types.max_shield[j])

            def ttk(aspect: float) -> float:
                return (math.ceil(shield / (damage * aspect) - 1e-9) - 1) / rate

            reach = max(float(c.gun_range[i]), float(c.gun_range[j]))
            passing = reach / float(c.cruise_speed[i] + c.cruise_speed[j])
            head_on = ttk(float(fraction[j]))
            print(
                f"| {shooter} on {target} | {head_on:.2f} | {head_on / passing:.2f} "
                f"| {ttk(1.0):.2f} |"
            )
    for key in KEYS:
        i = keys.index(key)
        rate = float(rules.shield_recharge_rate * types.shield_recharge_rate_scale[i])
        delay = float(rules.shield_recharge_delay * types.shield_recharge_delay_scale[i])
        print(
            f"{key}: head-on fraction {float(fraction[i]):.3f}, half-strength angle "
            f"{math.degrees(float(angle[i])):.1f} degrees, recharge "
            f"{rate:.1f}/s after {delay:.1f} s"
        )


# The search: what each field draws and its bounds. Weights and exponents step additively;
# logistics, the deny angle and the lag multiplicatively.
FROZEN = ("climb_exponent", "speed_exponent", "range_exponent")
MULTIPLICATIVE_BOUNDS = {"deny_angle": (0.05, 1.5), "lag_in_gun_range": (0.05, 3.0)}
SCALE_CEILING = 10.0


def clamp(doctrine: Doctrine) -> Doctrine:
    """Inside the search's bounds: weights in [0, 1], exponents in [-2, 2], each logistic's
    scale between its floor and SCALE_CEILING floors."""
    values = {}
    for name in NAMES:
        value = getattr(doctrine, name)
        if name in WEIGHTS:
            value = min(max(value, 0.0), 1.0)
        elif name in EXPONENTS:
            value = min(max(value, -2.0), 2.0)
        elif name in MULTIPLICATIVE_BOUNDS:
            low, high = MULTIPLICATIVE_BOUNDS[name]
            value = min(max(value, low), high)
        elif isinstance(value, Logistic):
            floor = SCALE_FLOORS[name]
            value = Logistic(
                value.centre, min(max(value.scale, floor), SCALE_CEILING * floor), value.rising
            )
        values[name] = value
    return Doctrine(**values)


def perturb(doctrine: Doctrine, sigma: float, generator: torch.Generator) -> Doctrine:
    """Every searched number moved: a weight or exponent plus sigma N(0, 1) times its
    bounds' width (a weight at 0 stays there with probability 1/2, so an element can stay
    off), a logistic's centre and scale and the other numbers times exp(sigma N(0, 1))."""

    def normal() -> float:
        return float(torch.randn(1, generator=generator))

    values = {}
    for name in NAMES:
        value = getattr(doctrine, name)
        if name in FROZEN:
            values[name] = value
        elif name in WEIGHTS:
            values[name] = value + sigma * normal()
        elif name in EXPONENTS:
            values[name] = value + 4.0 * sigma * normal()
        elif isinstance(value, Logistic):
            values[name] = Logistic(
                value.centre * math.exp(sigma * normal()),
                value.scale * math.exp(sigma * normal()),
                value.rising,
            )
        else:
            values[name] = value * math.exp(sigma * normal())
    return clamp(Doctrine(**values))


def search(args) -> None:
    """The (1 + lambda) search of one type's doctrine (the module docstring)."""
    game, types, rules, keys = setup(args.preset)
    path = pathlib.Path(args.doctrines)
    champion = read_doctrines(path, keys)
    index = keys.index(args.type)
    lineups = ["energy v turn", "mixed", "energy mirror" if args.type == ENERGY else "turn mirror"]
    log = pathlib.Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    parent = from_row(champion[index].tolist())
    parent_points, parent_games, start = [], 0, 0
    if rows:
        start = max(row["generation"] for row in rows) + 1
        last = [row for row in rows if row["generation"] == start - 1]
        best = max(last, key=lambda row: row["parent_score"] if row["parent"] else -1)
        parent = from_row(best["doctrine"])
        parent_points = [
            row["points"] for row in rows if row["parent"] and row["doctrine"] == best["doctrine"]
        ]
        parent_points = [p for chunk in parent_points for p in chunk]
        parent_games = len(parent_points)
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(start * args.children):  # the draws of the generations already played
        perturb(parent, args.sigma, generator)
    episode_ticks = int(round(game.episode_seconds / rules.tick_length))
    started = time.time()
    deadline = started + 3600.0 * args.hours
    for generation in range(start, args.generations):
        if time.time() > deadline:
            raise SystemExit(f"past the {args.hours} h bound: resume from {log}")
        sigma = max(args.sigma * args.decay**generation, args.minimum_sigma)
        candidates = [parent] + [perturb(parent, sigma, generator) for _ in range(args.children)]
        K, G = len(candidates), args.games
        points = [[] for _ in candidates]
        generation_started = time.time()
        for lineup in lineups:
            ship_type, a_on_zero = lineup_types(lineup, keys, K * G)
            side_a = champion[None].repeat(K * G, 1, 1)
            for number, candidate in enumerate(candidates):
                side_a[number * G : (number + 1) * G, index] = torch.tensor(pack(candidate))
            if lineup == "energy v turn" and args.type == TURN:
                # Jujak flies the lineup's second team: the candidates' side takes it
                a_on_zero = ~a_on_zero
            seed = args.seed * 100_000 + generation * 10 + lineups.index(lineup)
            result = games(
                rules,
                types,
                keys,
                ship_type,
                a_on_zero,
                side_a,
                champion,
                episode_ticks,
                seed,
            )
            outcome = torch.tensor(result["result"])  # the candidates' side's
            for number in range(K):
                part = outcome[number * G : (number + 1) * G]
                points[number].extend(
                    (1.0 if r > 0 else 0.5 if r == 0 else 0.0) for r in part.tolist()
                )
        scores = [sum(p) / len(p) for p in points]
        parent_points = parent_points + points[0]
        parent_score = sum(parent_points) / len(parent_points)
        parent_games = len(parent_points)
        with log.open("a") as handle:
            for number, candidate in enumerate(candidates):
                handle.write(
                    json.dumps(
                        dict(
                            generation=generation,
                            index=number,
                            parent=number == 0,
                            parent_score=parent_score if number == 0 else None,
                            score=scores[number],
                            games=len(points[number]),
                            points=points[number],
                            sigma=sigma,
                            doctrine=pack(candidate),
                            named=named(candidate),
                        )
                    )
                    + "\n"
                )
        best = max(range(1, K), key=lambda number: scores[number])
        error = math.sqrt(0.25 / len(points[best]))
        promoted = scores[best] > parent_score + args.margin * error
        elapsed = time.time() - started
        left = elapsed / (generation - start + 1) * (args.generations - generation - 1)
        print(
            f"{datetime.datetime.now(KST):%H:%M:%S} generation {generation}: parent "
            f"{parent_score:.3f} ({parent_games} games), best child {best} {scores[best]:.3f}"
            f"{' -> parent' if promoted else ''}; {time.time() - generation_started:.0f} s; "
            f"ETA {clock(left)} KST (worst {clock(1.3 * left)})",
            flush=True,
        )
        if promoted:
            parent, parent_points = candidates[best], list(points[best])
    print("parent", json.dumps(named(parent)))


def main() -> None:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("play")
    one.add_argument("--record", required=True)
    one.add_argument("--label", required=True)
    one.add_argument("--doctrines", default=str(DOCTRINES))
    one.add_argument("--doctrines-b", help="side B's doctrines file (default: side A's)")
    one.add_argument("--off-a", action="store_true", help="side A flies no doctrine")
    one.add_argument("--lineups", nargs="+", choices=list(LINEUPS))
    one.add_argument("--environments", type=int, default=256)
    one.add_argument("--batches", type=int, default=2)
    one.add_argument("--seed", type=int, default=11)
    one.add_argument("--hours", type=float, default=3.0)
    one.add_argument("--preset", default=PRESET)
    sub.add_parser("card").add_argument("--preset", default=PRESET)
    two = sub.add_parser("report")
    two.add_argument("--record", required=True, nargs="+")
    two.add_argument("--out")
    three = sub.add_parser("search")
    three.add_argument("--type", required=True, choices=KEYS)
    three.add_argument("--log", required=True)
    three.add_argument("--doctrines", default=str(DOCTRINES))
    three.add_argument("--generations", type=int, default=8)
    three.add_argument("--children", type=int, default=7)
    three.add_argument("--games", type=int, default=48, help="per candidate and lineup")
    three.add_argument("--sigma", type=float, default=0.3)
    three.add_argument("--decay", type=float, default=0.85)
    three.add_argument("--minimum-sigma", type=float, default=0.08)
    three.add_argument("--margin", type=float, default=1.0, help="standard errors to promote")
    three.add_argument("--seed", type=int, default=5)
    three.add_argument("--hours", type=float, default=3.0)
    three.add_argument("--preset", default=PRESET)
    for command in (one, three):
        command.add_argument("--threads", type=int, default=4, help="torch's CPU threads")
    args = parser.parse_args()
    if args.command in ("play", "search"):
        torch.set_num_threads(args.threads)
    {"play": play, "report": report, "search": search, "card": card}[args.command](args)


if __name__ == "__main__":
    main()
