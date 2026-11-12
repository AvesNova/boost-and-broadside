"""The game loop's GPU harness: 5v5 team deathmatch with respawns of the real-scale set's
Yongma (energy) and Jujak (turn), the scripted agent on both sides, many games at once in
one compiled tick, with the ship cards and the doctrines as tensors that vary by group of
environments, so a whole sweep (card variants x doctrine pairs x lineups) runs in one graph
and a new sweep of the same sizes never recompiles.

    PYTHONPATH=src python -m bench.game_loop speed --environments 512 1024 2048

The layout (`Layout`): B environments in groups. Every group names a card variant (a
Yongma row and a Jujak row of the variant table: `variant_types`, 2K rows, the first K
Yongma's, the last K Jujak's), a lineup (`LINEUPS`) and two players (each a doctrine per
type: `Player`). A side's doctrine table per view is (2B, 2K, WIDTH): every Yongma row
carries the side's Yongma doctrine, every Jujak row its Jujak doctrine. Side A sits on team
0 in the even environments of a group. The game: `real_scale_deathmatch` with no fields and
no zones, the episode `--seconds` long (every environment ends on the same tick; the game
resets itself), a dead ship respawning after the mode's 10 s. A game's result for A is the
sign of its kills minus B's.

Statistics per environment (`Tally`), accumulated in the tick: kills by killer side, by
killer and victim type and by the killing hit's aspect class (`bnb.contracts.ASPECTS`);
enemy damage taken by victim type and side and aspect; the head-on passes (merges) of the
killer's pair before each kill (a histogram), first-merge kills, kills by the pair's
first hitter, the time from the victim's first hit taken (a histogram); the side landing
the game's first hit; and the post-merge turning fights (`CIRCLES`: one-circle,
two-circle, scissors) from the pair's turn directions in the seconds after a merge.

A merge: a pair inside the longer of their gun ranges with both noses within MERGE_CONE of
the line between them; it ends when either nose is more than 90 degrees off that line or
the range opens past 1.2 gun ranges. A kill within the pair's first merge or 2 s after it is
a first-merge kill. A pair's merges count from when both last spawned.
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
from bnb.controllers.scripted_doctrine import WIDTH, Doctrine, from_named, pack
from bnb.game import api, ships
from bnb.game.handling.characteristics import characteristics
from bnb.game.ship_types import ShipTypes

PRESET = "real_scale_deathmatch"
SHIP_SET = "real-scale"
ENERGY, TURN = "yongma", "jujak"
KINDS = (ENERGY, TURN)
# each lineup's team (slot kinds: 0 Yongma, 1 Jujak) for side A and side B
LINEUPS = {
    "energy v turn": ((0,) * 5, (1,) * 5),
    "turn v energy": ((1,) * 5, (0,) * 5),
    "turn mirror": ((1,) * 5, (1,) * 5),
    "energy mirror": ((0,) * 5, (0,) * 5),
    "mixed": ((0, 1, 0, 1, 0), (0, 1, 0, 1, 0)),
    "mixed reversed": ((1, 0, 1, 0, 1), (1, 0, 1, 0, 1)),
}
MERGE_CONE = math.radians(30.0)
PASSED = math.radians(90.0)
OPEN_FACTOR = 1.2
MERGE_GRACE = 2.0  # s
PASS_BINS = 8  # passes 0..6, 7 and more
TTK_BINS = 32  # seconds from the first hit taken: [0, 1), ..., [31, inf)
CIRCLE_WINDOW = 3.0  # s after a merge's end
TURNING = 0.05  # rad/s: a nose rate below this is not a turn
CIRCLES = ("one circle", "two circle", "scissors")
# The base card: the ship files' with these factors (the author's 2026-11-12 decision:
# hit damage x 0.65 restores v5's damage per landed hit).
BASE_FACTORS = {"hit_damage": 0.65}
KST = zoneinfo.ZoneInfo("Asia/Seoul")


def clock(seconds: float) -> str:
    """The KST clock time `seconds` from now."""
    return (datetime.datetime.now(KST) + datetime.timedelta(seconds=seconds)).strftime("%H:%M")


@dataclasses.dataclass(frozen=True)
class Player:
    """A scripted player: a doctrine for each type."""

    name: str
    energy: Doctrine = Doctrine()
    turn: Doctrine = Doctrine()
    temperature: float = 1.0  # the scripted agent's (inf: uniform random play)

    def rows(self) -> tuple[list[float], list[float]]:
        return pack(self.energy), pack(self.turn)

    def named(self) -> dict:
        from bnb.controllers.scripted_doctrine import named

        return {
            "name": self.name,
            "temperature": self.temperature,
            ENERGY: named(self.energy),
            TURN: named(self.turn),
        }


def player_from(entry: dict) -> Player:
    """A player from `Player.named`'s form."""
    return Player(
        entry["name"],
        from_named(entry[ENERGY]),
        from_named(entry[TURN]),
        float(entry.get("temperature", 1.0)),
    )


def base_card(device) -> tuple[ShipTypes, dict]:
    """(2,) rows the set's Yongma and Jujak cards with BASE_FACTORS, and the rules' game."""
    keys = [ship.key for ship in ships.load(SHIP_SET)]
    full = api.ship_types(ship_set=SHIP_SET)
    index = torch.tensor([keys.index(ENERGY), keys.index(TURN)])
    values = {
        field.name: getattr(full, field.name)[index].clone() for field in dataclasses.fields(full)
    }
    for name, factor in BASE_FACTORS.items():
        values[name] = values[name] * factor
    return ShipTypes(**{k: v.to(device) for k, v in values.items()})


def variant_types(base: ShipTypes, energy: list[dict], turn: list[dict]) -> ShipTypes:
    """The variant table (2K,): K Yongma rows then K Jujak rows, each the base row with its
    dict's fields multiplied (a field name -> factor, or {"set": value})."""
    if len(energy) != len(turn):
        raise ValueError("as many Yongma variants as Jujak variants")
    rows = []
    for kind, variants in enumerate((energy, turn)):
        for factors in variants:
            row = {f.name: getattr(base, f.name)[kind].clone() for f in dataclasses.fields(base)}
            for name, factor in factors.items():
                if isinstance(factor, dict):  # {"set": value}: an absolute value
                    row[name] = torch.full_like(row[name], factor["set"])
                else:
                    row[name] = row[name] * factor
            rows.append(row)
    return ShipTypes(
        **{f.name: torch.stack([row[f.name] for row in rows]) for f in dataclasses.fields(base)}
    )


@dataclasses.dataclass
class Group:
    """One group of environments: a card variant, a lineup and two players."""

    variant: int
    lineup: str
    a: Player
    b: Player
    environments: int

    def key(self) -> dict:
        return dict(variant=self.variant, lineup=self.lineup, a=self.a.name, b=self.b.name)


def layout(groups: list[Group], variants: int, device) -> dict:
    """The per-environment tensors of a list of groups: ship types (B, N) int32, side A on
    team 0 (B,), the group index (B,), and the view doctrine table (2B, 2K, WIDTH)."""
    ship_type, a_on_zero, group_index, view_rows, temperature = [], [], [], [], []
    for number, group in enumerate(groups):
        first, second = LINEUPS[group.lineup]
        rows = {side: player.rows() for side, player in (("a", group.a), ("b", group.b))}
        heat = {"a": group.a.temperature, "b": group.b.temperature}

        def types_of(team):
            return [group.variant + kind * variants for kind in team]

        for e in range(group.environments):
            on_zero = e % 2 == 0
            team0, team1 = (first, second) if on_zero else (second, first)
            ship_type.append(types_of(team0) + types_of(team1))
            a_on_zero.append(on_zero)
            group_index.append(number)
            for team_side in ("a", "b") if on_zero else ("b", "a"):
                energy_row, turn_row = rows[team_side]
                view_rows.append([energy_row] * variants + [turn_row] * variants)
                temperature.append(heat[team_side])
    return dict(
        ship_type=torch.tensor(ship_type, dtype=torch.int32, device=device),
        a_on_zero=torch.tensor(a_on_zero, device=device),
        group=torch.tensor(group_index, device=device),
        doctrines=torch.tensor(view_rows, dtype=torch.float32, device=device),
        temperature=torch.tensor(temperature, dtype=torch.float32, device=device),
    )


def tally_like(B: int, S: int, device) -> dict:
    """The zeroed per-environment statistics and the trackers."""
    zeros = dict(device=device)
    return dict(
        kills_side=torch.zeros((B, 2), **zeros),  # by killer side A, B
        kills_kind=torch.zeros((B, 2, 2, len(ASPECTS)), **zeros),  # killer, victim kind
        damage_kind=torch.zeros((B, 2, 2, len(ASPECTS)), **zeros),  # victim side, kind
        passes=torch.zeros((B, PASS_BINS), **zeros),
        first_merge=torch.zeros((B,), **zeros),
        killer_first=torch.zeros((B,), **zeros),
        ttk=torch.zeros((B, TTK_BINS), **zeros),
        circles=torch.zeros((B, len(CIRCLES)), **zeros),
        first_game_hit=torch.full((B,), -1.0, **zeros),  # 0 A, 1 B, 2 both
        # trackers
        in_merge=torch.zeros((B, S, S), dtype=torch.bool, device=device),
        merges=torch.zeros((B, S, S), **zeros),
        first_merge_end=torch.full((B, S, S), -1.0, **zeros),
        first_hit=torch.full((B, S, S), -1.0, **zeros),
        victim_hit_tick=torch.full((B, 2 * S), -1.0, **zeros),
        window=torch.zeros((B, S, S), **zeros),  # ticks left of the post-merge window
        same=torch.zeros((B, S, S), **zeros),
        opposite=torch.zeros((B, S, S), **zeros),
        reversals=torch.zeros((B, S, S, 2), **zeros),
        last_sign=torch.zeros((B, 2 * S), **zeros),
        heading=torch.zeros((B, 2 * S, 2), **zeros),
        tick=torch.zeros((), **zeros),
    )


STATISTICS = (
    "kills_side",
    "kills_kind",
    "damage_kind",
    "passes",
    "first_merge",
    "killer_first",
    "ttk",
    "circles",
    "first_game_hit",
)


class Loop:
    """A compiled sweep of one size: B environments, K variants per type."""

    def __init__(
        self, environments: int, variants: int, seconds: float, device="cuda", compiled: bool = True
    ) -> None:
        self.device = torch.device(device)
        game = dataclasses.replace(api.PRESETS[PRESET], num_fields=0, episode_seconds=seconds)
        self.game = game
        self.base = base_card(self.device)
        self.K = variants
        self.types = variant_types(self.base, [{}] * variants, [{}] * variants)
        self.rules = api.rules_of(game, types=self.types)
        self.B, self.S = environments, game.ships_per_team
        self.N = 2 * self.S
        self.ticks = int(round(seconds / self.rules.tick_length))
        self.card = characteristics(self.types, self.rules)
        self.doctrines = torch.zeros((2 * self.B, 2 * variants, WIDTH), device=self.device)
        self.temperature = torch.ones((2 * self.B,), device=self.device)
        self.agent = scripted.Scripted(
            self.card, self.S, temperature=self.temperature, doctrines=self.doctrines
        )
        self.ship_type = torch.zeros((self.B, self.N), dtype=torch.int32, device=self.device)
        self.a_on_zero = torch.zeros((self.B,), dtype=torch.bool, device=self.device)
        self.grace = float(round(MERGE_GRACE / self.rules.tick_length))
        self.window_ticks = float(round(CIRCLE_WINDOW / self.rules.tick_length))
        self._tick = (
            torch.compile(self.tick, fullgraph=True, dynamic=False) if compiled else self.tick
        )

    def set_types(self, energy: list[dict], turn: list[dict]) -> None:
        """New card variants, written in place (no recompile)."""
        fresh = variant_types(self.base, energy, turn)
        for field in dataclasses.fields(fresh):
            getattr(self.types, field.name).copy_(getattr(fresh, field.name))
        card = characteristics(self.types, self.rules)
        for field in dataclasses.fields(card):
            value = getattr(self.card, field.name)
            if isinstance(value, torch.Tensor):
                value.copy_(getattr(card, field.name))

    def start(self, groups: list[Group]):
        """A fresh game in every environment of the layout."""
        if sum(g.environments for g in groups) != self.B:
            raise ValueError("the groups' environments must sum to B")
        placed = layout(groups, self.K, self.device)
        self.ship_type.copy_(placed["ship_type"])
        self.a_on_zero.copy_(placed["a_on_zero"])
        self.doctrines.copy_(placed["doctrines"])
        self.temperature.copy_(placed["temperature"])
        self.group = placed["group"]
        state = api.reset(
            self.rules,
            self.types,
            api.uniforms(self.rules, self.B, self.device),
            ship_type=self.ship_type,
        )
        state, observation, _ = api.perceive(self.rules, state)
        carry = self.agent.init_carry(2 * self.B, self.device)
        tally = tally_like(self.B, self.S, self.device)
        shown = api.display(state)
        tally["heading"] = torch.stack([shown["ships.heading_cos"], shown["ships.heading_sin"]], -1)
        return state, observation, carry, tally

    def tick(self, state, observation, carry, tally):
        """One tick of every game and the statistics."""
        B, S, N = self.B, self.S, self.N
        rules, types = self.rules, self.types
        first_tick = tally["tick"] == 0
        view = Obs(
            **{
                name: getattr(observation, name).flatten(0, 1)
                for name in ("ships", "fields", "globals", "zones")
            }
        )
        reset = first_tick.expand(2 * B)
        action, carry, _ = self.agent.step(carry, view, reset)
        action = Action.unpack(action.packed().reshape(B, N, 3).to(torch.int32))
        moved, sweep = api.move(rules, types, state, action, api.spread(rules, B, self.device))
        state, events = api.resolve(
            rules, types, moved, sweep, action, api.uniforms(rules, B, self.device)
        )
        tally = dict(tally)
        tick = tally["tick"]
        shown = api.display(state)
        x, y = shown["ships.x"], shown["ships.y"]
        cos, sin = shown["ships.heading_cos"], shown["ships.heading_sin"]
        alive = shown["ships.alive"].bool() & ~events.done[:, None]
        a_on_zero = self.a_on_zero
        side_a = torch.cat([a_on_zero[:, None].expand(B, S), (~a_on_zero)[:, None].expand(B, S)], 1)
        kind = (self.ship_type >= self.K).long()  # (B, N) 0 Yongma, 1 Jujak
        team0, team1 = slice(0, S), slice(S, N)

        # nose rates (the heading's turn this tick)
        previous = tally["heading"]
        turn = (
            torch.atan2(
                previous[..., 0] * sin - previous[..., 1] * cos,
                previous[..., 0] * cos + previous[..., 1] * sin,
            )
            / rules.tick_length
        )
        sign = torch.where(turn.abs() > TURNING, torch.sign(turn), 0.0)
        tally["heading"] = torch.stack([cos, sin], -1)

        # merges of every enemy pair
        reach = per_type(self.card.gun_range, self.ship_type)
        pair_reach = torch.maximum(reach[:, team0][:, :, None], reach[:, team1][:, None, :])
        dx = x[:, team1][:, None, :] - x[:, team0][:, :, None]
        dy = y[:, team1][:, None, :] - y[:, team0][:, :, None]
        distance = torch.sqrt(dx * dx + dy * dy)
        ci, si = cos[:, team0][:, :, None], sin[:, team0][:, :, None]
        cj, sj = cos[:, team1][:, None, :], sin[:, team1][:, None, :]
        off_i = torch.atan2((ci * dy - si * dx).abs(), ci * dx + si * dy)
        off_j = torch.atan2((cj * dy - sj * dx).abs(), -(cj * dx + sj * dy))
        both = alive[:, team0][:, :, None] & alive[:, team1][:, None, :]
        in_merge = tally["in_merge"]
        merges = tally["merges"]
        starting = both & ~in_merge & (distance < pair_reach)
        starting &= (off_i < MERGE_CONE) & (off_j < MERGE_CONE)
        ending = in_merge & (
            (off_i > PASSED) | (off_j > PASSED) | (distance > OPEN_FACTOR * pair_reach) | ~both
        )
        first_merge_end = tally["first_merge_end"]
        first_end = ending & (merges == 1) & (first_merge_end < 0)
        first_merge_end = torch.where(first_end, tick, first_merge_end)
        merges = merges + starting.float()
        in_merge = (in_merge | starting) & ~ending

        # post-merge turning: each pair's turn senses for CIRCLE_WINDOW after a merge ends
        window = tally["window"]
        opened = ending & both
        same, opposite, reversals = tally["same"], tally["opposite"], tally["reversals"]
        same = torch.where(opened, 0.0, same)
        opposite = torch.where(opened, 0.0, opposite)
        reversals = torch.where(opened[..., None], 0.0, reversals)
        window = torch.where(opened, self.window_ticks, window)
        active = (window > 0) & both
        si_, sj_ = sign[:, team0][:, :, None], sign[:, team1][:, None, :]
        product = si_ * sj_
        same = same + (active & (product > 0)).float()
        opposite = opposite + (active & (product < 0)).float()
        last = tally["last_sign"]
        flipped = (sign * last < 0).float()  # (B, N)
        reversals = reversals + active[..., None].float() * torch.stack(
            [
                flipped[:, team0][:, :, None].expand(B, S, S),
                flipped[:, team1][:, None, :].expand(B, S, S),
            ],
            -1,
        )
        tally["last_sign"] = torch.where(sign != 0, sign, last)
        closing = active & (window == 1)
        turned = closing & ((same + opposite) > 0.5 * self.window_ticks)
        scissors = turned & (reversals.amin(-1) >= 2)
        two = turned & ~scissors & (same > opposite)
        one = turned & ~scissors & ~(same > opposite)
        tally["circles"] = (
            tally["circles"]
            + torch.stack([one.sum((1, 2)), two.sum((1, 2)), scissors.sum((1, 2))], -1).float()
        )
        window = torch.where(both, (window - 1).clamp_min(0), 0.0)

        # hits
        damage = events.damage
        hit_ij = damage[:, team0][:, :, team1] > 0.0
        hit_ji = (damage[:, team1][:, :, team0] > 0.0).transpose(1, 2)
        first_hit = tally["first_hit"]
        new_first = (first_hit < 0) & (hit_ij | hit_ji)
        first_hit = torch.where(
            new_first, torch.where(hit_ij & hit_ji, 2.0, hit_ji.float()), first_hit
        )
        enemy_taken = events.enemy_damage_by_aspect.sum(-1) > 0.0
        victim_hit_tick = tally["victim_hit_tick"]
        victim_hit_tick = torch.where(enemy_taken & (victim_hit_tick < 0), tick, victim_hit_tick)
        hits_by = damage.sum(2) > 0.0  # (B, N) the shooter hit someone (enemy or not)
        hit_a = (hits_by & side_a).any(1)
        hit_b = (hits_by & ~side_a).any(1)
        fgh = tally["first_game_hit"]
        tally["first_game_hit"] = torch.where(
            (fgh < 0) & (hit_a | hit_b), torch.where(hit_a & hit_b, 2.0, hit_b.float()), fgh
        )

        # damage taken by aspect, by the victim's side and kind
        by_aspect = events.enemy_damage_by_aspect  # (B, N, 4)
        victim_side = (~side_a).long()  # 0 A, 1 B
        onehot = torch.nn.functional.one_hot(victim_side * 2 + kind, 4).float()  # (B, N, 4)
        tally["damage_kind"] = tally["damage_kind"] + torch.einsum(
            "bnc,bna->bca", onehot, by_aspect
        ).reshape(B, 2, 2, len(ASPECTS))

        # kills
        died = events.died
        killer = events.killer.long().clamp_min(0)
        has_killer = events.killer >= 0
        enemy_kill = died & has_killer & ((killer < S) != (torch.arange(N, device=x.device) < S))
        killer_side_a = torch.gather(side_a, 1, killer)
        killer_kind = torch.gather(kind, 1, killer)
        kill = enemy_kill.float()
        tally["kills_side"] = tally["kills_side"] + torch.stack(
            [(kill * killer_side_a).sum(1), (kill * ~killer_side_a).sum(1)], -1
        )
        aspect = torch.nn.functional.one_hot(
            events.kill_aspect.long().clamp_min(0), len(ASPECTS)
        ).float()
        pair_kind = torch.nn.functional.one_hot(killer_kind * 2 + kind, 4).float()  # (B, N, 4)
        tally["kills_kind"] = tally["kills_kind"] + torch.einsum(
            "bn,bnc,bna->bca", kill, pair_kind, aspect
        ).reshape(B, 2, 2, len(ASPECTS))
        # the killer's pair: (i, j) team 0 ship i, team 1 ship j
        victim_index = torch.arange(N, device=x.device).expand(B, N)
        i = torch.where(victim_index < S, victim_index, killer)
        j = torch.where(victim_index < S, killer, victim_index) - S
        i, j = i.clamp(0, S - 1), j.clamp(0, S - 1)
        flat = i * S + j
        count = torch.gather(merges.flatten(1), 1, flat)
        end = torch.gather(first_merge_end.flatten(1), 1, flat)
        first = torch.gather(first_hit.flatten(1), 1, flat)
        in_first = (count == 1) & ((end < 0) | (tick <= end + self.grace))
        killer_on_zero = killer < S
        killer_first = (
            (first == 2) | ((first == 0) & killer_on_zero) | ((first == 1) & ~killer_on_zero)
        )
        tally["passes"] = tally["passes"] + torch.einsum(
            "bn,bnp->bp",
            kill,
            torch.nn.functional.one_hot(count.long().clamp_max(PASS_BINS - 1), PASS_BINS).float(),
        )
        tally["first_merge"] = tally["first_merge"] + (kill * in_first).sum(1)
        tally["killer_first"] = tally["killer_first"] + (kill * killer_first).sum(1)
        seconds = (tick - victim_hit_tick) * rules.tick_length
        ttk_bin = seconds.clamp(0, TTK_BINS - 1).long()
        tally["ttk"] = tally["ttk"] + torch.einsum(
            "bn,bnt->bt",
            kill * (victim_hit_tick >= 0),
            torch.nn.functional.one_hot(ttk_bin, TTK_BINS).float(),
        )

        # a dead ship's pairs start again when it spawns
        gone = died | events.spawned | ~shown["ships.alive"].bool()
        reset_pair = gone[:, team0][:, :, None] | gone[:, team1][:, None, :]
        tally["merges"] = torch.where(reset_pair, 0.0, merges)
        tally["in_merge"] = in_merge & ~reset_pair
        tally["first_merge_end"] = torch.where(reset_pair, -1.0, first_merge_end)
        tally["first_hit"] = torch.where(reset_pair, -1.0, first_hit)
        tally["victim_hit_tick"] = torch.where(gone, -1.0, victim_hit_tick)
        tally["window"] = window
        tally["same"], tally["opposite"], tally["reversals"] = same, opposite, reversals
        tally["tick"] = tick + 1
        state, observation, _ = api.perceive(rules, state)
        return state, observation, carry, tally

    def play(self, groups: list[Group]) -> dict:
        """One game in every environment; the statistics per environment, on the host."""
        state, observation, carry, tally = self.start(groups)
        for _ in range(self.ticks - 1):  # the game ends (and resets) on its last tick
            state, observation, carry, tally = self._tick(state, observation, carry, tally)
        torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
        return {name: tally[name].cpu() for name in STATISTICS} | {"group": self.group.cpu()}


def per_type(table: torch.Tensor, ship_type: torch.Tensor) -> torch.Tensor:
    """A per-type value for each ship (a gather: the variant table is small)."""
    return table[ship_type.long()]


def speed(args) -> None:
    """Games per second at a few batch sizes."""
    for environments in args.environments:
        loop = Loop(environments, args.variants, args.seconds)
        groups = [Group(0, "mixed", Player("off"), Player("off"), environments)]
        started = time.time()
        loop.play(groups)
        first = time.time() - started
        started = time.time()
        out = loop.play(groups)
        steady = time.time() - started
        kills = float(out["kills_side"].sum())
        print(
            f"B={environments}: first {first:.0f} s (compile), steady {steady:.1f} s for "
            f"{environments} games of {args.seconds:.0f} s: {environments / steady:.1f} games/s, "
            f"{environments * loop.ticks / steady / 1e3:.0f}k env-ticks/s; kills/min "
            f"{kills / environments / (args.seconds / 60):.2f}; "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
            flush=True,
        )


def share(count: float, total: float) -> tuple[float, float]:
    """A share and its binomial standard error."""
    p = count / max(total, 1e-9)
    return p, math.sqrt(max(p * (1 - p), 1e-12) / max(total, 1))


def median_of(histogram: torch.Tensor) -> float | None:
    """The median bin's lower edge (+ half a bin) of a histogram of whole-second bins."""
    total = float(histogram.sum())
    if total <= 0:
        return None
    cumulative = histogram.cumsum(0)
    return float((cumulative < total / 2).sum()) + 0.5


def summary(out: dict, mask: torch.Tensor, seconds: float) -> dict:
    """A set of environments' statistics (mask (B,) bool)."""
    kills = out["kills_side"][mask]
    n = int(mask.sum())
    difference = kills[:, 0] - kills[:, 1]
    points = torch.where(difference > 0, 1.0, torch.where(difference < 0, 0.0, 0.5))
    score = float(points.mean())
    score_se = float(points.std() / math.sqrt(max(n, 1))) if n > 1 else 0.5
    per_minute = kills.sum(1) / (seconds / 60.0)
    kills_kind = out["kills_kind"][mask].sum(0)  # (killer kind, victim kind, aspect)
    damage_kind = out["damage_kind"][mask].sum(0)  # (victim side, victim kind, aspect)
    total_kills = float(kills_kind.sum())
    kill_aspect = kills_kind.sum((0, 1))
    damage_aspect = damage_kind.sum((0, 1))
    passes = out["passes"][mask].sum(0)
    weights = torch.arange(PASS_BINS, dtype=torch.float32)
    first_hit = out["first_game_hit"][mask]
    decided = (first_hit >= 0) & (first_hit < 2) & (difference != 0)
    first_wins = ((first_hit == 0) & (difference > 0)) | ((first_hit == 1) & (difference < 0))
    circles = out["circles"][mask].sum(0)
    return dict(
        games=n,
        score=(score, score_se),
        kills_per_minute_mean=float(per_minute.mean()),
        kills_per_minute_median=float(per_minute.median()),
        kills_per_game=float(kills.sum(1).mean()),
        kill_share=(kill_aspect / kill_aspect.sum().clamp_min(1)).tolist(),
        damage_share=(damage_aspect / damage_aspect.sum().clamp_min(1e-9)).tolist(),
        kill_share_by_pair={
            f"{KINDS[k][0].upper()} on {KINDS[v][0].upper()}": (
                (kills_kind[k, v] / kills_kind[k, v].sum().clamp_min(1)).tolist(),
                float(kills_kind[k, v].sum()),
            )
            for k in range(2)
            for v in range(2)
        },
        damage_share_by_victim={
            KINDS[v]: (damage_kind[:, v].sum(0) / damage_kind[:, v].sum().clamp_min(1e-9)).tolist()
            for v in range(2)
        },
        passes_per_kill=float((passes * weights).sum() / passes.sum().clamp_min(1)),
        no_merge_kills=share(float(passes[0]), float(passes.sum())),
        first_merge_kills=share(float(out["first_merge"][mask].sum()), total_kills),
        killer_first_hit=share(float(out["killer_first"][mask].sum()), total_kills),
        first_hit_wins=share(float(first_wins[decided].sum()), float(decided.sum())),
        ttk_median=median_of(out["ttk"][mask].sum(0)),
        circles=(circles / circles.sum().clamp_min(1)).tolist(),
        circle_count=float(circles.sum()),
    )


def distinctness(
    loop: Loop, groups: list[Group], energy: Doctrine, turn: Doctrine, every: int = 150
) -> dict:
    """The energy and turn doctrines on the same situations: play the layout's games (compiled)
    and every `every` ticks give each view's observation to the memoryless agent twice, once
    with the energy doctrine on every own ship and once with the turn doctrine; per living
    own ship, the symmetric KL of the power and stick marginals (nats, summed over the two)
    and the share whose most likely (power, stick) agree. By own kind."""
    state, observation, carry, tally = loop.start(groups)
    tables = {
        name: torch.tensor([pack(d)] * (2 * loop.K), device=loop.device)
        for name, d in (("energy", energy), ("turn", turn))
    }
    kl = {kind: [] for kind in KINDS}
    agree = {kind: [] for kind in KINDS}
    for number in range(loop.ticks - 1):
        state, observation, carry, tally = loop._tick(state, observation, carry, tally)
        if number % every != every - 1:
            continue
        view = Obs(
            **{
                n: getattr(observation, n).flatten(0, 1)
                for n in ("ships", "fields", "globals", "zones")
            }
        )
        with torch.no_grad():
            parts = {
                name: scripted.marginals(
                    view, loop.card, loop.agent.parameters, 1.0, None, doctrines=table
                )
                for name, table in tables.items()
            }
        own = view.ships[:, : loop.S]
        from bnb.contracts.obs import SHIP

        alive = own[..., SHIP["alive"]] > 0.5
        own_kind = own[..., SHIP["ship_type"]].round().long() >= loop.K
        total = 0.0
        same = None
        for part in ("power", "stick"):
            p = getattr(parts["energy"], part).clamp_min(1e-6)
            q = getattr(parts["turn"], part).clamp_min(1e-6)
            total = total + ((p - q) * (p.log() - q.log())).sum(-1)
            match = p.argmax(-1) == q.argmax(-1)
            same = match if same is None else same & match
        for k, kind in enumerate(KINDS):
            mine = alive & (own_kind == bool(k))
            kl[kind].append(total[mine].float().cpu())
            agree[kind].append(same[mine].float().cpu())
    out = {}
    for kind in KINDS:
        values, matches = torch.cat(kl[kind]), torch.cat(agree[kind])
        out[kind] = dict(
            symmetric_kl=float(values.mean()) if len(values) else None,
            agreement=float(matches.mean()) if len(matches) else None,
            samples=len(values),
        )
    return out


@dataclasses.dataclass
class Job:
    """Games of one matchup: a card variant, a lineup, players A and B."""

    variant: int
    lineup: str
    a: Player
    b: Player
    games: int


FILLER = Player("filler")


def batches(jobs: list[Job], environments: int) -> list[list[tuple[int, Group]]]:
    """The jobs cut into batches of exactly `environments` games (even group sizes, so each
    side sits on each team equally), the last padded with filler games: each batch a list of
    (job index or -1, group)."""
    out, current, room = [], [], environments
    for number, job in enumerate(jobs):
        left = job.games + job.games % 2
        while left > 0:
            take = min(left, room)
            current.append((number, Group(job.variant, job.lineup, job.a, job.b, take)))
            left -= take
            room -= take
            if room == 0:
                out.append(current)
                current, room = [], environments
    if current:
        current.append((-1, Group(0, "mixed", FILLER, FILLER, room)))
        out.append(current)
    return out


def run(loop: Loop, jobs: list[Job], log=print) -> list[dict]:
    """Every job's games; per job the per-environment statistics, concatenated."""
    parts: list[list[dict]] = [[] for _ in jobs]
    plan = batches(jobs, loop.B)
    started = time.time()
    for number, batch in enumerate(plan):
        out = loop.play([group for _, group in batch])
        offset = 0
        for job_index, group in batch:
            rows = slice(offset, offset + group.environments)
            offset += group.environments
            if job_index >= 0:
                parts[job_index].append({k: v[rows] for k, v in out.items()})
        elapsed = time.time() - started
        left = elapsed / (number + 1) * (len(plan) - number - 1)
        if len(plan) > 1:
            log(f"  batch {number + 1}/{len(plan)} {elapsed:.0f} s; ETA {clock(left)} KST")
    return [{k: torch.cat([p[k] for p in part]) for k in part[0]} for part in parts]


def job_summary(job: Job, out: dict, seconds: float) -> dict:
    """A job's statistics and its key."""
    mask = torch.ones(out["kills_side"].shape[0], dtype=torch.bool)
    return dict(
        variant=job.variant,
        lineup=job.lineup,
        a=job.a.name,
        b=job.b.name,
        **summary(out, mask, seconds),
        wins=float((out["kills_side"][:, 0] > out["kills_side"][:, 1]).sum()),
        draws=float((out["kills_side"][:, 0] == out["kills_side"][:, 1]).sum()),
    )


def ratings(
    rows: list[dict], anchor: str = "off", anchor_rating: float = 1000.0, prior_games: float = 2.0
) -> dict:
    """Bradley-Terry ratings (Elo scale, `bnb.league.ratings.fit_ratings`) of the players in
    job summaries (wins, draws as half), `anchor` pinned at `anchor_rating`."""
    from bnb.league.ratings import fit_ratings

    names = sorted({r["a"] for r in rows} | {r["b"] for r in rows})
    if anchor not in names:
        raise ValueError(f"the anchor {anchor} played no game")
    index = {n: i for i, n in enumerate(names)}
    wins = torch.zeros((len(names), len(names)), dtype=torch.float64)
    for r in rows:
        a, b = index[r["a"]], index[r["b"]]
        if a == b:
            continue
        losses = r["games"] - r["wins"] - r["draws"]
        wins[a, b] += r["wins"] + r["draws"] / 2
        wins[b, a] += losses + r["draws"] / 2
    rating, error = fit_ratings(wins, index[anchor], prior_games)
    return {n: (float(rating[i]) + anchor_rating, float(error[i])) for n, i in index.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("speed")
    one.add_argument("--environments", type=int, nargs="+", default=[512])
    one.add_argument("--variants", type=int, default=4)
    one.add_argument("--seconds", type=float, default=120.0)
    args = parser.parse_args()
    {"speed": speed}[args.command](args)


if __name__ == "__main__":
    main()


def markdown(rows: list[dict]) -> str:
    """Job summaries as a Markdown table."""

    def shares(values):
        return " / ".join(f"{v:.2f}" for v in values)

    lines = [
        "| Variant | Lineup | A v B | Games | A scores | Kills/min mean (median) "
        "| Damage m / o / f / t | Kills m / o / f / t | Passes a kill | First-merge kills "
        "| Killer hit first | P(win \\| first hit) | TTK median s | Circles 1 / 2 / scissors (n) |",
        "|---" * 14 + "|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['lineup']} | {r['a']} v {r['b']} | {r['games']} "
            f"| {r['score'][0]:.3f} ± {r['score'][1]:.3f} "
            f"| {r['kills_per_minute_mean']:.2f} ({r['kills_per_minute_median']:.2f}) "
            f"| {shares(r['damage_share'])} | {shares(r['kill_share'])} "
            f"| {r['passes_per_kill']:.2f} | {r['first_merge_kills'][0]:.2f} "
            f"| {r['killer_first_hit'][0]:.2f} | {r['first_hit_wins'][0]:.2f} "
            f"| {r['ttk_median']} | {shares(r['circles'])} ({r['circle_count']:.0f}) |"
        )
    return "\n".join(lines)


SEARCH_LINEUPS = (
    "mixed",
    "mixed reversed",
    "energy v turn",
    "turn v energy",
    "energy mirror",
    "turn mirror",
)
# a candidate's score: the mixed lineups (the training lineup, the Elo) half, the four
# single-type lineups the other half
LINEUP_WEIGHTS = {"mixed": 0.25, "mixed reversed": 0.25} | {
    lineup: 0.125 for lineup in SEARCH_LINEUPS[2:]
}


def weighted_score(rows: list[dict]) -> tuple[float, float]:
    """A player's weighted mean score over job summaries (its side A), and the standard
    error."""
    total, variance, weight = 0.0, 0.0, 0.0
    by_lineup: dict[str, list[dict]] = {}
    for r in rows:
        by_lineup.setdefault(r["lineup"], []).append(r)
    for lineup, mine in by_lineup.items():
        w = LINEUP_WEIGHTS.get(lineup, 0.0)
        games = sum(r["games"] for r in mine)
        mean = sum(r["score"][0] * r["games"] for r in mine) / games
        se2 = sum((r["score"][1] * r["games"]) ** 2 for r in mine) / games**2
        total += w * mean
        variance += w * w * se2
        weight += w
    return total / weight, math.sqrt(variance) / weight


def search(
    loop: Loop,
    parent: Player,
    opponents: list[Player],
    log_path: pathlib.Path,
    generations: int,
    children: int,
    games: int,
    sigma: float,
    decay: float = 0.85,
    minimum_sigma: float = 0.06,
    margin: float = 1.0,
    seed: int = 5,
    variant: int = 0,
    deadline: float | None = None,
    log=print,
) -> Player:
    """A (1 + lambda) search of both types' doctrines at once against a fixed league: each
    generation the parent, `children` perturbations of its Yongma doctrine and `children`
    of its Jujak doctrine each play `games` games per league opponent and lineup
    (SEARCH_LINEUPS) as side A; a candidate's score is its weighted mean
    (`weighted_score`). The parent's games accumulate over generations; the best child of
    each type that beats the parent by `margin` standard errors replaces that type's
    doctrine (both may change in one generation). Every candidate is a line of the log."""
    from bench.doctrine_arena import perturb

    generator = torch.Generator().manual_seed(seed)
    parent_rows: list[dict] = []
    started = time.time()
    for generation in range(generations):
        if deadline is not None and time.time() > deadline:
            log("past the deadline: stopping the search")
            break
        s = max(sigma * decay**generation, minimum_sigma)
        candidates = [dataclasses.replace(parent, name="parent")]
        for number in range(children):
            candidates.append(
                dataclasses.replace(
                    parent, name=f"e{number}", energy=perturb(parent.energy, s, generator)
                )
            )
        for number in range(children):
            candidates.append(
                dataclasses.replace(
                    parent, name=f"t{number}", turn=perturb(parent.turn, s, generator)
                )
            )
        jobs = [
            Job(variant, lineup, candidate, opponent, games)
            for candidate in candidates
            for opponent in opponents
            for lineup in SEARCH_LINEUPS
        ]
        generation_started = time.time()
        outs = run(loop, jobs, log=lambda line: None)
        rows = [
            job_summary(job, out, loop.ticks * loop.rules.tick_length)
            for job, out in zip(jobs, outs, strict=True)
        ]
        by_candidate: dict[str, list[dict]] = {}
        for r in rows:
            by_candidate.setdefault(r["a"], []).append(r)
        parent_rows += by_candidate["parent"]
        parent_score = weighted_score(parent_rows)
        scores = {name: weighted_score(mine) for name, mine in by_candidate.items()}
        with log_path.open("a") as handle:
            for candidate in candidates:
                handle.write(
                    json.dumps(
                        dict(
                            generation=generation,
                            name=candidate.name,
                            sigma=s,
                            score=scores[candidate.name],
                            parent_score=parent_score if candidate.name == "parent" else None,
                            player=candidate.named(),
                            by_opponent={
                                o.name: weighted_score(
                                    [r for r in by_candidate[candidate.name] if r["b"] == o.name]
                                )[0]
                                for o in opponents
                            },
                        )
                    )
                    + "\n"
                )
        promoted = []
        new_energy, new_turn = parent.energy, parent.turn
        for prefix in ("e", "t"):
            best = max((n for n in scores if n.startswith(prefix)), key=lambda n: scores[n][0])
            score, error = scores[best]
            if score > parent_score[0] + margin * math.hypot(error, parent_score[1]):
                promoted.append(f"{best} {score:.3f}")
                chosen = candidates[[c.name for c in candidates].index(best)]
                if prefix == "e":
                    new_energy = chosen.energy
                else:
                    new_turn = chosen.turn
        elapsed = time.time() - started
        left = elapsed / (generation + 1) * (generations - generation - 1)
        log(
            f"{datetime.datetime.now(KST):%H:%M:%S} generation {generation}: parent "
            f"{parent_score[0]:.3f} ± {parent_score[1]:.3f}, best e "
            f"{max(v[0] for n, v in scores.items() if n.startswith('e')):.3f}, best t "
            f"{max(v[0] for n, v in scores.items() if n.startswith('t')):.3f}"
            f"{' -> ' + ', '.join(promoted) if promoted else ''}; "
            f"{time.time() - generation_started:.0f} s; ETA {clock(left)} KST"
        )
        if promoted:
            parent = dataclasses.replace(parent, energy=new_energy, turn=new_turn)
            parent_rows = []
    return parent


def card_passes(loop: Loop) -> list[dict]:
    """Each variant's closed-form head-on time to kill in passes (100% hits, the expected
    hit D x 8/15 x the kinetic factor at the reference distance, 1), shooter on target, and
    its tail TTK in seconds: (ceil(shield / (hit x aspect)) - 1) / fire rate over the pass
    time (the pair's longer gun range over the two cruises)."""
    from bnb.game.physics.combat import MEAN_CLOSENESS

    types, rules, card = loop.types, loop.rules, characteristics(loop.types, loop.rules)
    fraction = rules.head_on_fraction * types.head_on_fraction_scale
    out = []
    for v in range(loop.K):
        row = {}
        for shooter, i in ((ENERGY, v), (TURN, v + loop.K)):
            for target, j in ((ENERGY, v), (TURN, v + loop.K)):
                damage = float(types.hit_damage[i]) * MEAN_CLOSENESS
                rate, shield = float(types.fire_rate[i]), float(types.max_shield[j])

                def ttk(aspect: float) -> float:
                    return (math.ceil(shield / (damage * aspect) - 1e-9) - 1) / rate

                reach = max(float(card.gun_range[i]), float(card.gun_range[j]))
                passing = reach / float(card.cruise_speed[i] + card.cruise_speed[j])
                row[f"{shooter[0].upper()} on {target[0].upper()}"] = dict(
                    head_on_passes=ttk(float(fraction[j])) / passing, tail_seconds=ttk(1.0)
                )
        row["gun_range"] = [float(card.gun_range[v]), float(card.gun_range[v + loop.K])]
        row["head_on_angle_degrees"] = [
            math.degrees(rules.head_on_angle * float(types.head_on_angle_scale[k]))
            for k in (v, v + loop.K)
        ]
        out.append(row)
    return out
