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
# The harness's games (game v13: the rating games are elimination and frontline): each the
# preset a Loop plays and how a game's result for A is read. "deathmatch" (the default, as
# every round before v13 ran): the whole clock, the result the sign of A's kills less B's.
# "elimination" and "frontline": the mode's own result for the environment's first match
# (a wipe or the front's threshold, else the clock's rule: the health left, the front),
# the environment's later matches (after an early win) played but not scored.
MODES = {
    "deathmatch": "real_scale_deathmatch",
    "elimination": "real_scale_elimination",
    "frontline": "real_scale_frontline",
}
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
# A two-circle fight's turn circles (each ship's mean turn centre and radius over the
# post-merge window): overlapping when the centres are nearer than the radii's sum, side by
# side otherwise.
TWO_CIRCLES = ("overlapping", "side by side")
# Each ship's choice after a merge (over the post-merge window): turning in the bandit's
# rotation sense (the two-circle choice), the opposite sense (one-circle), extending (turning
# less than half the window), or turning while the bandit extends.
CHOICES = ("same sense", "opposite sense", "extend", "turn v extender")
# A choice's outcome, until the pair's next merge ends, either ship dies or respawns: the
# chooser kills the bandit, the bandit kills the chooser, neither.
OUTCOMES = ("kills", "killed", "neither")
# Engagements (the author's definition): a pair of enemy ships within the threshold (two
# definitions: the pair's longer gun range; two of the pair's larger full-stick turn
# diameters) is engaged until it stays beyond it ENGAGE_APART seconds or either dies. A
# merge: a closing -> least distance -> opening cycle (MERGE_HYSTERESIS metres each way).
ENGAGEMENTS = ("gun range", "two turn diameters")
ENGAGE_APART = 3.0  # s
MERGE_HYSTERESIS = 30.0  # m
MERGE_BINS = 16  # merges in an engagement: 0..14, 15 and more
TURN_BINS = 32  # circles turned by a ship in an engagement, quarter circles: 0..7.75, 8+
PACE_BINS = 121  # whole seconds 0..119, and 120 or more
TRADE_WINDOW = 0.5  # s: a death with one on the other team this close is a trade
# The base card: the ship files' with these factors. The tuning ran on the files before
# the freeze with hit damage x 0.65 (the author's 2026-11-12 decision: v5's damage per
# landed hit); the frozen files carry it, so none now.
BASE_FACTORS: dict[str, float] = {}
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
    # the scripted agent's parameters (`scripted_parameters.pack`), None: the mode's default
    parameters: tuple[float, ...] | None = None

    def rows(self) -> tuple[list[float], list[float]]:
        """The energy and turn doctrines packed as parameter rows."""
        return pack(self.energy), pack(self.turn)

    def named(self) -> dict:
        """The player as a name-keyed dict (`player_from` reads it back)."""
        from bnb.controllers.scripted_doctrine import named

        out = {
            "name": self.name,
            "temperature": self.temperature,
            ENERGY: named(self.energy),
            TURN: named(self.turn),
        }
        if self.parameters is not None:
            out["parameters"] = list(self.parameters)
        return out


def player_from(entry: dict) -> Player:
    """A player from `Player.named`'s form."""
    return Player(
        entry["name"],
        from_named(entry[ENERGY]),
        from_named(entry[TURN]),
        float(entry.get("temperature", 1.0)),
        tuple(entry["parameters"]) if "parameters" in entry else None,
    )


def default_parameters() -> tuple[float, ...]:
    """The scripted agent's real-scale deathmatch set, packed."""
    from bnb.controllers.scripted_defaults import MODE_DEFAULTS
    from bnb.controllers.scripted_parameters import pack as pack_parameters

    return tuple(pack_parameters(MODE_DEFAULTS.sets[PRESET]))


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
        """The group's identity as a dict: variant, lineup and the players' names."""
        return dict(variant=self.variant, lineup=self.lineup, a=self.a.name, b=self.b.name)


def teams_of(lineup: str, ships_per_team: int = 5) -> tuple[tuple[int, ...], ...]:
    """A lineup's two teams of slot kinds at a team size (`LINEUPS` at 5; the patterns
    repeated or cut at others)."""
    first, second = LINEUPS[lineup]
    if ships_per_team == 5:
        return first, second
    S = ships_per_team
    return tuple(first[i % 2 if len(set(first)) > 1 else 0] for i in range(S)), tuple(
        second[i % 2 if len(set(second)) > 1 else 0] for i in range(S)
    )


def layout(groups: list[Group], variants: int, device, ships_per_team: int = 5) -> dict:
    """The per-environment tensors of a list of groups: ship types (B, N) int32, side A on
    team 0 (B,), the group index (B,), and the view doctrine table (2B, 2K, WIDTH)."""
    ship_type, a_on_zero, group_index, view_rows, temperature = [], [], [], [], []
    parameter_rows = []
    default = default_parameters()
    for number, group in enumerate(groups):
        first, second = teams_of(group.lineup, ships_per_team)
        rows = {side: player.rows() for side, player in (("a", group.a), ("b", group.b))}
        heat = {"a": group.a.temperature, "b": group.b.temperature}
        sets = {
            side: list(player.parameters or default)
            for side, player in (("a", group.a), ("b", group.b))
        }

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
                parameter_rows.append(sets[team_side])
    return dict(
        ship_type=torch.tensor(ship_type, dtype=torch.int32, device=device),
        a_on_zero=torch.tensor(a_on_zero, device=device),
        group=torch.tensor(group_index, device=device),
        doctrines=torch.tensor(view_rows, dtype=torch.float32, device=device),
        temperature=torch.tensor(temperature, dtype=torch.float32, device=device),
        parameters=torch.tensor(parameter_rows, dtype=torch.float32, device=device),
    )


def tally_like(B: int, S: int, device) -> dict:
    """The zeroed per-environment statistics and the trackers."""
    zeros = dict(device=device)
    return dict(
        kills_side=torch.zeros((B, 2), **zeros),  # by killer side A, B
        # the first match's result for team 0 (+1, 0, -1) and whether it has ended
        result=torch.zeros((B,), dtype=torch.int32, device=device),
        finished=torch.zeros((B,), dtype=torch.bool, device=device),
        kills_kind=torch.zeros((B, 2, 2, len(ASPECTS)), **zeros),  # killer, victim kind
        damage_kind=torch.zeros((B, 2, 2, len(ASPECTS)), **zeros),  # victim side, kind
        passes=torch.zeros((B, PASS_BINS), **zeros),
        first_merge=torch.zeros((B,), **zeros),
        killer_first=torch.zeros((B,), **zeros),
        ttk=torch.zeros((B, TTK_BINS), **zeros),
        circles=torch.zeros((B, len(CIRCLES)), **zeros),
        # two-circle fights by kind and their outcome (a kill in the pair, none)
        two_circle_outcome=torch.zeros((B, len(TWO_CIRCLES), 2), **zeros),
        # post-merge choices by the chooser's kind (0 Yongma, 1 Jujak) and outcome
        choice_outcome=torch.zeros((B, 2, len(CHOICES), len(OUTCOMES)), **zeros),
        post_turn=torch.zeros((B, S, S, 2), **zeros),  # ticks turning, i and j
        post_sign=torch.zeros((B, S, S, 2), **zeros),  # summed turn signs, i and j
        post_both=torch.zeros((B, S, S), **zeros),  # ticks both turning
        post_centre=torch.zeros((B, S, S, 4), **zeros),  # summed turn centres x_i y_i x_j y_j
        post_radius=torch.zeros((B, S, S, 2), **zeros),  # summed turn radii
        pending_choice=torch.full((B, S, S, 2), -1, dtype=torch.long, device=device),
        pending_circle=torch.full((B, S, S), -1, dtype=torch.long, device=device),
        position=torch.zeros((B, 2 * S, 2), **zeros),
        first_game_hit=torch.full((B,), -1.0, **zeros),  # 0 A, 1 B, 2 both
        deaths=torch.zeros((B,), **zeros),
        # engagements, per definition (ENGAGEMENTS)
        engagements=torch.zeros((B, 2), **zeros),
        engagement_kills=torch.zeros((B, 2), **zeros),
        merges_per_kill=torch.zeros((B, 2, MERGE_BINS), **zeros),
        merges_per_engagement=torch.zeros((B, 2, MERGE_BINS), **zeros),
        to_damage=torch.zeros((B, 2, PACE_BINS), **zeros),
        to_death=torch.zeros((B, 2, PACE_BINS), **zeros),
        engagement_length=torch.zeros((B, 2, PACE_BINS), **zeros),
        engagement_turning=torch.zeros((B, 2, TURN_BINS), **zeros),
        merge_kinds=torch.zeros((B, 2, 3), **zeros),  # one circle, two circle, scissors
        eng_on=torch.zeros((B, 2, S, S), dtype=torch.bool, device=device),
        eng_apart=torch.zeros((B, 2, S, S), **zeros),
        eng_start=torch.zeros((B, 2, S, S), **zeros),
        eng_damage=torch.full((B, 2, S, S), -1.0, **zeros),
        eng_merges=torch.zeros((B, 2, S, S), **zeros),
        eng_closing=torch.ones((B, 2, S, S), dtype=torch.bool, device=device),
        eng_extreme=torch.zeros((B, 2, S, S), **zeros),
        eng_turn_i=torch.zeros((B, 2, S, S), **zeros),
        eng_turn_j=torch.zeros((B, 2, S, S), **zeros),
        eng_sign_i=torch.zeros((B, 2, S, S), **zeros),
        eng_sign_j=torch.zeros((B, 2, S, S), **zeros),
        gaps=torch.zeros((B, PACE_BINS), **zeros),  # seconds between deaths (any team)
        engage=torch.zeros((B, PACE_BINS), **zeros),  # a life's spawn to its first hit
        alive_ticks=torch.zeros((B,), **zeros),
        first_kill_tick=torch.full((B,), -1.0, **zeros),
        first_hit_tick=torch.full((B,), -1.0, **zeros),
        last_any_death=torch.full((B,), -1.0, **zeros),
        spawn_tick=torch.zeros((B, 2 * S), **zeros),
        engaged=torch.zeros((B, 2 * S), dtype=torch.bool, device=device),
        traded=torch.zeros((B,), **zeros),  # deaths with an enemy death within TRADE_WINDOW
        last_death=torch.full((B, 2), -1e9, **zeros),  # by team: its last death's tick
        last_deaths=torch.zeros((B, 2), **zeros),  # how many died then
        last_traded=torch.ones((B, 2), dtype=torch.bool, device=device),  # counted already
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
    "engagements",
    "engagement_kills",
    "merges_per_kill",
    "merges_per_engagement",
    "to_damage",
    "to_death",
    "engagement_length",
    "engagement_turning",
    "merge_kinds",
    "gaps",
    "engage",
    "alive_ticks",
    "first_kill_tick",
    "first_hit_tick",
    "deaths",
    "traded",
    "kills_side",
    "result",
    "finished",
    "kills_kind",
    "damage_kind",
    "passes",
    "first_merge",
    "killer_first",
    "ttk",
    "circles",
    "two_circle_outcome",
    "choice_outcome",
    "first_game_hit",
)


class Loop:
    """A compiled sweep of one size: B environments, K variants per type."""

    def __init__(
        self,
        environments: int,
        variants: int,
        seconds: float,
        device="cuda",
        compiled: bool = True,
        ships_per_team: int = 5,
        fog: bool = True,
        arena_scale: float = 1.0,
        mode: str = "deathmatch",
    ) -> None:
        self.device = torch.device(device)
        self.mode = mode
        base_game = api.PRESETS[MODES[mode]]
        # the game's per-size radius (the density rule, the scale, then the mode's floor:
        # deathmatch at least 3000 m at real scale, game v16)
        radius = api.arena_radius_for(base_game, ships_per_team, arena_scale)
        game = dataclasses.replace(
            base_game,
            ships_per_team=ships_per_team,
            arena_radius=radius,
            num_fields=0,
            episode_seconds=seconds,
            # deathmatch: no early win, every game runs the whole clock
            **(dict(kills_per_ship=1000) if mode == "deathmatch" else {}),
        )
        self.fog = fog
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
        from bnb.controllers.scripted_parameters import unpack

        self.parameters = torch.tensor(
            [default_parameters()] * (2 * self.B), dtype=torch.float32, device=self.device
        )
        self.agent = scripted.Scripted(
            self.card,
            self.S,
            parameters=unpack(self.parameters),
            temperature=self.temperature,
            doctrines=self.doctrines,
        )
        self.ship_type = torch.zeros((self.B, self.N), dtype=torch.int32, device=self.device)
        self.a_on_zero = torch.zeros((self.B,), dtype=torch.bool, device=self.device)
        self.grace = float(round(MERGE_GRACE / self.rules.tick_length))
        self.apart_ticks = float(round(ENGAGE_APART / self.rules.tick_length))
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
        placed = layout(groups, self.K, self.device, self.S)
        self.ship_type.copy_(placed["ship_type"])
        self.a_on_zero.copy_(placed["a_on_zero"])
        self.doctrines.copy_(placed["doctrines"])
        self.temperature.copy_(placed["temperature"])
        self.parameters.copy_(placed["parameters"])
        self.group = placed["group"]
        state = api.reset(
            self.rules,
            self.types,
            api.uniforms(self.rules, self.B, self.device),
            torch.full((self.B,), self.fog, dtype=torch.bool, device=self.device),
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
        alive_start = api.display(state)["ships.alive"].bool()
        action, carry, _ = self.agent.step(carry, view, reset)
        action = Action.unpack(action.packed().reshape(B, N, 3).to(torch.int32))
        moved, sweep = api.move(rules, types, state, action, api.spread(rules, B, self.device))
        state, events = api.resolve(
            rules, types, moved, sweep, action, api.uniforms(rules, B, self.device)
        )
        tally = dict(tally)
        tick = tally["tick"]
        first_end = events.done & ~tally["finished"]
        tally["result"] = torch.where(first_end, events.result.to(torch.int32), tally["result"])
        tally["finished"] = tally["finished"] | events.done
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
        # the flight path's velocity this tick (from the move), for the turn circles
        moved_x = (x - tally["position"][..., 0]) / rules.tick_length
        moved_y = (y - tally["position"][..., 1]) / rules.tick_length
        tally["position"] = torch.stack([x, y], -1)

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
        post_close, two_circle = closing, two
        # the window's turning and turn circles, per pair (reset when a merge opens one)
        post_turn = torch.where(opened[..., None], 0.0, tally["post_turn"])
        post_sign = torch.where(opened[..., None], 0.0, tally["post_sign"])
        post_both = torch.where(opened, 0.0, tally["post_both"])
        post_centre = torch.where(opened[..., None], 0.0, tally["post_centre"])
        post_radius = torch.where(opened[..., None], 0.0, tally["post_radius"])
        turning_ship = sign != 0  # (B, N)
        turning_pair = torch.stack(
            [
                turning_ship[:, team0][:, :, None].expand(B, S, S),
                turning_ship[:, team1][:, None, :].expand(B, S, S),
            ],
            -1,
        )
        post_turn = post_turn + (active[..., None] & turning_pair).float()
        post_sign = post_sign + active[..., None].float() * torch.stack(
            [si_.expand(B, S, S), sj_.expand(B, S, S)], -1
        )
        # a turning ship's centre: its position plus speed / rate along its left normal
        speed_now = torch.sqrt(moved_x * moved_x + moved_y * moved_y)
        safe_turn = torch.where(turning_ship, turn, 1.0)
        radius_now = torch.where(turning_ship, speed_now / safe_turn.abs(), 0.0)
        reach_now = torch.where(turning_ship, speed_now / safe_turn, 0.0)  # signed
        unit_x = moved_x / speed_now.clamp_min(1e-3)
        unit_y = moved_y / speed_now.clamp_min(1e-3)
        centre_x, centre_y = x - reach_now * unit_y, y + reach_now * unit_x
        both_turning = active & turning_pair.all(-1)
        post_both = post_both + both_turning.float()
        centres = torch.stack(
            [
                centre_x[:, team0][:, :, None].expand(B, S, S),
                centre_y[:, team0][:, :, None].expand(B, S, S),
                centre_x[:, team1][:, None, :].expand(B, S, S),
                centre_y[:, team1][:, None, :].expand(B, S, S),
            ],
            -1,
        )
        radii = torch.stack(
            [
                radius_now[:, team0][:, :, None].expand(B, S, S),
                radius_now[:, team1][:, None, :].expand(B, S, S),
            ],
            -1,
        )
        post_centre = post_centre + both_turning[..., None].float() * centres
        post_radius = post_radius + both_turning[..., None].float() * radii
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
        # trades: a team's deaths this tick with the other team's last deaths within the window
        window_ticks = round(TRADE_WINDOW / rules.tick_length)
        team_deaths = torch.stack([died[:, :S].sum(1), died[:, S:].sum(1)], -1).float()  # (B, 2)
        other_last = tally["last_death"].flip(-1)
        other_before = (tick - other_last) <= window_ticks
        other_recent = other_before | (team_deaths.flip(-1) > 0)
        traded_now = torch.where(other_recent, team_deaths, 0.0).sum(-1)
        # the other team's earlier deaths, now traded, counted once
        late = other_before & (team_deaths > 0) & ~tally["last_traded"].flip(-1)
        traded_now = traded_now + torch.where(late, tally["last_deaths"].flip(-1), 0.0).sum(-1)
        marked = tally["last_traded"] | late.flip(-1)
        dying = team_deaths > 0
        tally["last_traded"] = torch.where(dying, other_recent, marked)
        tally["last_death"] = torch.where(dying, tick, tally["last_death"])
        tally["last_deaths"] = torch.where(dying, team_deaths, tally["last_deaths"])
        tally["deaths"] = tally["deaths"] + team_deaths.sum(-1)
        # pace: the gaps between deaths (several on one tick: gaps of 0), each life's spawn
        # to its first hit dealt or taken, the game's first hit and first kill
        count = team_deaths.sum(-1)
        last_any = tally["last_any_death"]
        gap = ((tick - last_any) * rules.tick_length).clamp(0, PACE_BINS - 1).long()
        bins = torch.nn.functional.one_hot(gap, PACE_BINS).float()
        zero = torch.nn.functional.one_hot(torch.zeros_like(gap), PACE_BINS).float()
        first_gap = ((count > 0) & (last_any >= 0)).float()
        tally["gaps"] = (
            tally["gaps"]
            + first_gap[:, None] * bins
            + (count - (count > 0).float()).clamp_min(0)[:, None] * zero
        )
        tally["last_any_death"] = torch.where(count > 0, tick, last_any)
        involved = (damage.sum(2) > 0) | (damage.sum(1) > 0)  # dealt or taken
        fresh = involved & ~tally["engaged"] & alive_start
        since = ((tick - tally["spawn_tick"]) * rules.tick_length).clamp(0, PACE_BINS - 1).long()
        tally["engage"] = tally["engage"] + torch.einsum(
            "bn,bnk->bk", fresh.float(), torch.nn.functional.one_hot(since, PACE_BINS).float()
        )
        tally["engaged"] = tally["engaged"] | fresh
        tally["first_hit_tick"] = torch.where(
            (tally["first_hit_tick"] < 0) & involved.any(1), tick, tally["first_hit_tick"]
        )
        tally["first_kill_tick"] = torch.where(
            (tally["first_kill_tick"] < 0) & enemy_kill.any(1), tick, tally["first_kill_tick"]
        )
        tally["alive_ticks"] = tally["alive_ticks"] + alive_start.float().sum(1)
        tally["traded"] = tally["traded"] + traded_now
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

        # engagements (two definitions), their merges, turning and ends
        diameter = 2.0 * per_type(
            self.card.cruise_speed / self.card.turn_rate[..., 2], self.ship_type
        )
        pair_diameter = torch.maximum(
            diameter[:, team0][:, :, None], diameter[:, team1][:, None, :]
        )
        threshold = torch.stack([pair_reach, 2.0 * pair_diameter], 1)  # (B, 2, S, S)
        d = distance[:, None].expand_as(threshold)
        pair_alive = both[:, None].expand_as(threshold)
        inside = pair_alive & (d < threshold)
        on = tally["eng_on"]
        starting = inside & ~on
        on = on | starting
        apart = torch.where(on & ~inside, tally["eng_apart"] + 1, 0.0)
        start = torch.where(starting, tick, tally["eng_start"])
        damaged_pair = (hit_ij | hit_ji)[:, None]
        eng_damage = torch.where(starting, -1.0, tally["eng_damage"])
        eng_damage = torch.where(on & damaged_pair & (eng_damage < 0), tick, eng_damage)
        merges_n = torch.where(starting, 0.0, tally["eng_merges"])
        closing = torch.where(starting, True, tally["eng_closing"])
        extreme = torch.where(starting, d, tally["eng_extreme"])
        turn_i = torch.where(starting, 0.0, tally["eng_turn_i"])
        turn_j = torch.where(starting, 0.0, tally["eng_turn_j"])
        sign_i = torch.where(starting, 0.0, tally["eng_sign_i"])
        sign_j = torch.where(starting, 0.0, tally["eng_sign_j"])
        dt = rules.tick_length
        turn_i = turn_i + on * turn.abs()[:, team0][:, None, :, None] * dt
        turn_j = turn_j + on * turn.abs()[:, team1][:, None, None, :] * dt
        # merges: closing to a least distance, then opening by the hysteresis
        least = torch.minimum(extreme, d)
        most = torch.maximum(extreme, d)
        merged = on & closing & (d > least + MERGE_HYSTERESIS)
        reclosing = on & ~closing & (d < most - MERGE_HYSTERESIS)
        extreme = torch.where(closing, least, most)
        extreme = torch.where(merged | reclosing, d, extreme)
        closing = torch.where(merged, False, torch.where(reclosing, True, closing))
        merges_n = merges_n + merged.float()
        now_i = sign[:, team0][:, None, :, None].expand_as(d)
        now_j = sign[:, team1][:, None, None, :].expand_as(d)
        product = now_i * now_j
        scissor = merged & (sign_i * now_i < 0) & (sign_j * now_j < 0)
        kinds = torch.stack(
            [merged & ~scissor & (product < 0), merged & ~scissor & (product > 0), scissor], -1
        ).float()
        tally["merge_kinds"] = tally["merge_kinds"] + kinds.sum((2, 3))
        sign_i = torch.where(merged & (now_i != 0), now_i, sign_i)
        sign_j = torch.where(merged & (now_j != 0), now_j, sign_j)
        # ends: apart too long, or either ship died; a kill of one by the other
        died_i = died[:, team0][:, None, :, None]
        died_j = died[:, team1][:, None, None, :]
        killer_of = events.killer.long()
        j_index = torch.arange(S, device=x.device)
        killed_j = died[:, team1][:, None, :] & (
            killer_of[:, team1][:, None, :] == j_index[:, None]
        )  # (B, S_i, S_j): j killed by i
        killed_i = died[:, team0][:, :, None] & (
            killer_of[:, team0][:, :, None] == (S + j_index)[None, None, :]
        )  # (B, S_i, S_j): i killed by j
        kill_pair = (killed_i | killed_j)[:, None] & on
        ending = on & ((apart > self.apart_ticks) | died_i | died_j | ~pair_alive)
        length = ((tick - start) * dt).clamp(0, PACE_BINS - 1).long()
        tally["engagements"] = tally["engagements"] + ending.float().sum((2, 3))
        tally["engagement_kills"] = tally["engagement_kills"] + kill_pair.float().sum((2, 3))

        def histogram(mask: torch.Tensor, index: torch.Tensor, bins: int) -> torch.Tensor:
            onehot = torch.nn.functional.one_hot(index.clamp(0, bins - 1), bins).float()
            return (mask.float()[..., None] * onehot).sum((2, 3))

        merge_index = merges_n.long()
        tally["merges_per_kill"] = tally["merges_per_kill"] + histogram(
            kill_pair, merge_index, MERGE_BINS
        )
        tally["merges_per_engagement"] = tally["merges_per_engagement"] + histogram(
            ending, merge_index, MERGE_BINS
        )
        tally["to_death"] = tally["to_death"] + histogram(kill_pair, length, PACE_BINS)
        tally["engagement_length"] = tally["engagement_length"] + histogram(
            ending, length, PACE_BINS
        )
        to_damage = ((eng_damage - start) * dt).clamp(0, PACE_BINS - 1).long()
        tally["to_damage"] = tally["to_damage"] + histogram(
            ending & (eng_damage >= 0), to_damage, PACE_BINS
        )
        quarter = lambda radians: (radians / (2 * math.pi) * 4).long()  # noqa: E731
        tally["engagement_turning"] = (
            tally["engagement_turning"]
            + histogram(ending, quarter(turn_i), TURN_BINS)
            + histogram(ending, quarter(turn_j), TURN_BINS)
        )
        on = on & ~ending
        tally["eng_on"], tally["eng_apart"], tally["eng_start"] = on, apart, start
        tally["eng_damage"], tally["eng_merges"] = eng_damage, merges_n
        tally["eng_closing"], tally["eng_extreme"] = closing, extreme
        tally["eng_turn_i"], tally["eng_turn_j"] = turn_i, turn_j
        tally["eng_sign_i"], tally["eng_sign_j"] = sign_i, sign_j

        # a dead ship's pairs start again when it spawns
        gone = died | events.spawned | ~shown["ships.alive"].bool()
        reset_pair = gone[:, team0][:, :, None] | gone[:, team1][:, None, :]

        # post-merge choices and two-circle fights: their outcomes, then the new ones
        pending_choice, pending_circle = tally["pending_choice"], tally["pending_circle"]
        j_killed_by_i = killed_j  # (B, S_i, S_j)
        i_killed_by_j = killed_i
        finished = j_killed_by_i | i_killed_by_j | opened | reset_pair
        outcome_i = torch.where(j_killed_by_i, 0, torch.where(i_killed_by_j, 1, 2))
        outcome_j = torch.where(i_killed_by_j, 0, torch.where(j_killed_by_i, 1, 2))
        kind_i = kind[:, team0][:, :, None].expand(B, S, S)
        kind_j = kind[:, team1][:, None, :].expand(B, S, S)
        cells = 2 * len(CHOICES) * len(OUTCOMES)

        def counted(mask, index, bins):
            onehot = torch.nn.functional.one_hot(index.clamp(0, bins - 1), bins).float()
            return (mask.float()[..., None] * onehot).sum((1, 2))

        choice_count = torch.zeros((B, cells), device=x.device)
        for side, (chooser_kind, outcome) in enumerate(((kind_i, outcome_i), (kind_j, outcome_j))):
            choice = pending_choice[..., side]
            index = (chooser_kind * len(CHOICES) + choice) * len(OUTCOMES) + outcome
            choice_count = choice_count + counted(finished & (choice >= 0), index, cells)
        tally["choice_outcome"] = tally["choice_outcome"] + choice_count.reshape(
            B, 2, len(CHOICES), len(OUTCOMES)
        )
        a_kill = (j_killed_by_i | i_killed_by_j).long()
        tally["two_circle_outcome"] = tally["two_circle_outcome"] + counted(
            finished & (pending_circle >= 0), pending_circle * 2 + (1 - a_kill), 4
        ).reshape(B, 2, 2)
        pending_choice = torch.where(finished[..., None], -1, pending_choice)
        pending_circle = torch.where(finished, -1, pending_circle)
        # the window just closed: each ship's choice, and a two-circle fight's circles
        half_window = 0.5 * self.window_ticks
        turned_side = post_turn > half_window  # (B, S, S, 2)
        sense = torch.sign(post_sign)
        same_sense = sense[..., 0] == sense[..., 1]
        choices = []
        for side in (0, 1):
            other = 1 - side
            choice = torch.where(
                ~turned_side[..., side],
                2,
                torch.where(~turned_side[..., other], 3, torch.where(same_sense, 0, 1)),
            )
            choices.append(choice)
        new_choice = torch.stack(choices, -1)
        pending_choice = torch.where(
            (post_close & ~reset_pair)[..., None], new_choice, pending_choice
        )
        samples = post_both.clamp_min(1.0)
        mean_centre = post_centre / samples[..., None]
        centre_gap = torch.sqrt(
            (mean_centre[..., 0] - mean_centre[..., 2]) ** 2
            + (mean_centre[..., 1] - mean_centre[..., 3]) ** 2
        )
        radius_sum = post_radius.sum(-1) / samples
        circle_kind = torch.where(centre_gap < radius_sum, 0, 1)
        new_fight = post_close & two_circle & (post_both > 0) & ~reset_pair
        pending_circle = torch.where(new_fight, circle_kind, pending_circle)
        tally["pending_choice"], tally["pending_circle"] = pending_choice, pending_circle
        tally["post_turn"], tally["post_sign"], tally["post_both"] = post_turn, post_sign, post_both
        tally["post_centre"], tally["post_radius"] = post_centre, post_radius
        tally["merges"] = torch.where(reset_pair, 0.0, merges)
        tally["in_merge"] = in_merge & ~reset_pair
        tally["first_merge_end"] = torch.where(reset_pair, -1.0, first_merge_end)
        tally["first_hit"] = torch.where(reset_pair, -1.0, first_hit)
        tally["victim_hit_tick"] = torch.where(gone, -1.0, victim_hit_tick)
        tally["spawn_tick"] = torch.where(events.spawned, tick, tally["spawn_tick"])
        tally["engaged"] = tally["engaged"] & ~events.spawned
        tally["window"] = window
        tally["same"], tally["opposite"], tally["reversals"] = same, opposite, reversals
        tally["tick"] = tick + 1
        state, observation, _ = api.perceive(rules, state)
        return state, observation, carry, tally

    def play(self, groups: list[Group]) -> dict:
        """One game in every environment; the statistics per environment, on the host."""
        state, observation, carry, tally = self.start(groups)
        # deathmatch: the game ends (and resets) on its last tick, which is not played; a
        # scored mode plays it, so the clock's result is read
        ticks = self.ticks - 1 if self.mode == "deathmatch" else self.ticks
        for _ in range(ticks):
            state, observation, carry, tally = self._tick(state, observation, carry, tally)
        torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
        ships = torch.full((self.B,), float(self.N))  # ships a game, for the per-ship pace
        out = {name: tally[name].cpu() for name in STATISTICS} | {
            "group": self.group.cpu(),
            "ships": ships,
        }
        if self.mode != "deathmatch":  # A's first-match result, +1, 0 or -1
            a_result = torch.where(self.a_on_zero, tally["result"], -tally["result"])
            out["a_result"] = a_result.cpu()
        return out


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


def quantile_of(histogram: torch.Tensor, q: float) -> float | None:
    """The bin index at quantile q of a histogram of whole bins."""
    total = float(histogram.sum())
    if total <= 0:
        return None
    return float((histogram.cumsum(0) < q * total).sum())


def engagements_summary(out: dict, mask: torch.Tensor) -> dict:
    """Per engagement definition (ENGAGEMENTS): engagements and kills in them, merges per
    kill (mean, median, first-pass share, 90th percentile), merges per engagement, the
    medians of start to first damage, start to death and the length (s, whole-second bins),
    circles turned per ship and engagement (mean), and the merges' kinds."""
    if "engagements" not in out:
        return {}
    result = {}
    for k, name in enumerate(ENGAGEMENTS):
        per_kill = out["merges_per_kill"][mask][:, k].sum(0)
        per_engagement = out["merges_per_engagement"][mask][:, k].sum(0)
        counts = torch.arange(MERGE_BINS, dtype=torch.float32)
        turning = out["engagement_turning"][mask][:, k].sum(0)
        quarters = (torch.arange(TURN_BINS, dtype=torch.float32) + 0.5) / 4
        kinds = out["merge_kinds"][mask][:, k].sum(0)
        result[name] = dict(
            engagements=float(out["engagements"][mask][:, k].sum()),
            kills=float(out["engagement_kills"][mask][:, k].sum()),
            merges_per_kill_mean=float((per_kill * counts).sum() / per_kill.sum().clamp_min(1)),
            merges_per_kill_median=quantile_of(per_kill, 0.5),
            first_pass_kill_share=float(per_kill[:2].sum() / per_kill.sum().clamp_min(1)),
            merges_per_kill_p90=quantile_of(per_kill, 0.9),
            merges_per_engagement_mean=float(
                (per_engagement * counts).sum() / per_engagement.sum().clamp_min(1)
            ),
            to_damage_median=median_of(out["to_damage"][mask][:, k].sum(0)),
            to_death_median=median_of(out["to_death"][mask][:, k].sum(0)),
            length_median=median_of(out["engagement_length"][mask][:, k].sum(0)),
            circles_turned_mean=float((turning * quarters).sum() / turning.sum().clamp_min(1)),
            one_two_scissors=(kinds / kinds.sum().clamp_min(1)).tolist(),
        )
    return result


def pace(out: dict, mask: torch.Tensor, seconds: float) -> dict:
    """The pacing numbers: the gap between deaths (any team; mean and median, s), the life
    length (ship-seconds alive per death), kills per minute per ship, a life's spawn to its
    first hit (median, s), the game's first hit and first kill (median, s)."""
    if "gaps" not in out:
        return {}
    gaps = out["gaps"][mask].sum(0)
    centres = torch.arange(PACE_BINS, dtype=torch.float32) + 0.5
    deaths = float(out["deaths"][mask].sum())
    ships_alive_seconds = float(out["alive_ticks"][mask].sum()) / 30.0
    # ships a game (`Loop.play` records it; a 5v5 record without it)
    ships = int(out["ships"][mask][0]) if "ships" in out and bool(mask.any()) else 10

    def median_tick(values: torch.Tensor) -> float | None:
        values = values[values >= 0]
        return float(values.median()) / 30.0 if len(values) else None

    return dict(
        death_gap_mean=float((gaps * centres).sum() / gaps.sum().clamp_min(1)),
        death_gap_median=median_of(gaps),
        life_seconds=ships_alive_seconds / max(deaths, 1),
        kills_per_minute_per_ship=deaths / (mask.sum().item() * ships * seconds / 60.0),
        spawn_to_engagement_median=median_of(out["engage"][mask].sum(0)),
        first_hit_median=median_tick(out["first_hit_tick"][mask]),
        first_kill_median=median_tick(out["first_kill_tick"][mask]),
    )


def summary(out: dict, mask: torch.Tensor, seconds: float) -> dict:
    """A set of environments' statistics (mask (B,) bool)."""
    kills = out["kills_side"][mask]
    n = int(mask.sum())
    difference = kills[:, 0] - kills[:, 1]
    if "a_result" in out:  # a scored mode: the first match's own result
        points = (out["a_result"][mask].float() + 1.0) / 2.0
    else:
        points = torch.where(difference > 0, 1.0, torch.where(difference < 0, 0.0, 0.5))
    score = float(points.mean())
    score_se = float(points.std() / math.sqrt(max(n, 1))) if n > 1 else 0.5
    # A's share of the game's kills (0.5 in a game without one): the search's finer signal
    kill_points = torch.where(
        kills.sum(1) > 0, kills[:, 0] / kills.sum(1).clamp_min(1.0), torch.tensor(0.5)
    )
    kill_se = float(kill_points.std() / math.sqrt(max(n, 1))) if n > 1 else 0.5
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
        kill_score=(float(kill_points.mean()), kill_se),
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
        damage_by_side=[  # enemy damage taken a game by side A, B (side, aspect shares)
            (
                float(damage_kind[v].sum()) / max(n, 1),
                (damage_kind[v].sum(0) / damage_kind[v].sum().clamp_min(1e-9)).tolist(),
            )
            for v in range(2)
        ],
        circles=(circles / circles.sum().clamp_min(1)).tolist(),
        circle_count=float(circles.sum()),
        trade_fraction=float(out["traded"][mask].sum() / out["deaths"][mask].sum().clamp_min(1)),
        **pace(out, mask, seconds),
        engagement=engagements_summary(out, mask),
        **fight_choices(out, mask),
    )


def fight_choices(out: dict, mask: torch.Tensor) -> dict:
    """Two-circle fights split by their circles (`TWO_CIRCLES`: count, share, the share
    ending in a kill), and the post-merge choices by the chooser's kind (`CHOICES`: count,
    share of the kind's choices, outcome shares `OUTCOMES`)."""
    if "choice_outcome" not in out:
        return {}
    circles = out["two_circle_outcome"][mask].sum(0)  # (kind of fight, kill / none)
    fights = circles.sum(-1)
    choices = out["choice_outcome"][mask].sum(0)  # (chooser kind, choice, outcome)
    by_kind = {}
    for k, kind_name in enumerate(KINDS):
        made = choices[k].sum(-1)
        if float(made.sum()) == 0:
            continue
        by_kind[kind_name] = {
            name: dict(
                count=float(made[c]),
                share=float(made[c] / made.sum()),
                outcome=(choices[k, c] / made[c].clamp_min(1)).tolist(),
            )
            for c, name in enumerate(CHOICES)
        }
    return dict(
        two_circles={
            name: dict(
                count=float(fights[c]),
                share=float(fights[c] / fights.sum().clamp_min(1)),
                kill=float(circles[c, 0] / fights[c].clamp_min(1)),
            )
            for c, name in enumerate(TWO_CIRCLES)
        },
        post_merge_choices=by_kind,
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
        # regimes (the author's split): against the nearest seen living enemy, own
        # specific energy (v^2 / 2 + store) above or below its, and inside or outside two
        # own full-stick turn radii of it
        enemy = view.ships[:, loop.S :]
        seen = (enemy[..., SHIP["visible"]] > 0.5) & (enemy[..., SHIP["alive"]] > 0.5)
        dx = enemy[:, None, :, SHIP["x"]] - own[:, :, None, SHIP["x"]]
        dy = enemy[:, None, :, SHIP["y"]] - own[:, :, None, SHIP["y"]]
        squared = torch.where(seen[:, None, :], dx * dx + dy * dy, float("inf"))
        nearest = squared.argmin(-1)
        has = torch.isfinite(squared.min(-1).values)
        distance = squared.min(-1).values.clamp_max(1e12).sqrt()

        def energy_of(ships):
            speed = ships[..., SHIP["proper_speed"]]
            return 0.5 * speed * speed + ships[..., SHIP["energy"]]

        theirs = torch.gather(energy_of(enemy), 1, nearest)
        ahead = energy_of(own) > theirs
        own_type = own[..., SHIP["ship_type"]].round().long().clamp(0, 2 * loop.K - 1)
        radius = (loop.card.cruise_speed / loop.card.turn_rate[..., 2])[own_type]
        inside = distance < 2.0 * radius
        for k, kind in enumerate(KINDS):
            mine = alive & (own_kind == bool(k))
            kl[kind].append(total[mine].float().cpu())
            agree[kind].append(same[mine].float().cpu())
            for regime, where in (
                ("energy ahead, inside", ahead & inside),
                ("energy ahead, outside", ahead & ~inside),
                ("energy behind, inside", ~ahead & inside),
                ("energy behind, outside", ~ahead & ~inside),
            ):
                key = f"{kind}: {regime}"
                chosen = mine & has & where
                kl.setdefault(key, []).append(total[chosen].float().cpu())
                agree.setdefault(key, []).append(same[chosen].float().cpu())
    out = {}
    for kind in kl:
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


def outcome(out: dict) -> torch.Tensor:
    """(B,) each game's result for A: the scored mode's first-match result, else the sign of
    A's kills less B's (deathmatch)."""
    if "a_result" in out:
        return out["a_result"]
    return torch.sign(out["kills_side"][:, 0] - out["kills_side"][:, 1])


def job_summary(job: Job, out: dict, seconds: float) -> dict:
    """A job's statistics and its key."""
    mask = torch.ones(out["kills_side"].shape[0], dtype=torch.bool)
    return dict(
        variant=job.variant,
        lineup=job.lineup,
        a=job.a.name,
        b=job.b.name,
        **summary(out, mask, seconds),
        wins=float((outcome(out) > 0).sum()),
        draws=float((outcome(out) == 0).sum()),
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
    """The command line: its subcommands run the game-loop measurements."""
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
        "| Killer hit first | P(win \\| first hit) | TTK median s | Circles 1 / 2 / scissors (n) "
        "| Trades | Life s | Death gap mean (median) s | Spawn to hit s |",
        "|---" * 18 + "|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['lineup']} | {r['a']} v {r['b']} | {r['games']} "
            f"| {r['score'][0]:.3f} ± {r['score'][1]:.3f} "
            f"| {r['kills_per_minute_mean']:.2f} ({r['kills_per_minute_median']:.2f}) "
            f"| {shares(r['damage_share'])} | {shares(r['kill_share'])} "
            f"| {r['passes_per_kill']:.2f} | {r['first_merge_kills'][0]:.2f} "
            f"| {r['killer_first_hit'][0]:.2f} | {r['first_hit_wins'][0]:.2f} "
            f"| {r['ttk_median']} | {shares(r['circles'])} ({r['circle_count']:.0f}) "
            f"| {r.get('trade_fraction', float('nan')):.2f} | {r.get('life_seconds', 0):.0f} "
            f"| {r.get('death_gap_mean', 0):.1f} ({r.get('death_gap_median')}) "
            f"| {r.get('spawn_to_engagement_median')} |"
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


def weighted_score(rows: list[dict], key: str = "score") -> tuple[float, float]:
    """A player's weighted mean score (`key`: the game score, or `kill_score` A's share of
    the kills) over job summaries (its side A), and the standard error."""
    total, variance, weight = 0.0, 0.0, 0.0
    by_lineup: dict[str, list[dict]] = {}
    for r in rows:
        by_lineup.setdefault(r["lineup"], []).append(r)
    for lineup, mine in by_lineup.items():
        w = LINEUP_WEIGHTS.get(lineup, 0.0)
        games = sum(r["games"] for r in mine)
        mean = sum(r[key][0] * r["games"] for r in mine) / games
        se2 = sum((r[key][1] * r["games"]) ** 2 for r in mine) / games**2
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
    objective: str = "kill_score",
) -> Player:
    """A (1 + lambda) search of both types' doctrines at once against a fixed league: each
    generation the parent, `children` perturbations of its Yongma doctrine and `children`
    of its Jujak doctrine each play `games` games per league opponent and lineup
    (SEARCH_LINEUPS) as side A; a candidate's score is its weighted mean
    (`weighted_score` of `objective`). The parent's games accumulate over generations. The
    best child (of either type) that beats the parent by `margin` standard errors becomes
    the parent on probation: the next generation replays it on fresh games beside the old
    parent (the incumbent, its games still accumulating), and it stays only if its fresh
    score beats the incumbent's (the best-of-children's selection bias removed); otherwise
    the incumbent returns. A search ending on probation returns the incumbent. Every
    candidate is a line of the log."""
    from bench.doctrine_arena import perturb

    generator = torch.Generator().manual_seed(seed)
    parent_rows: list[dict] = []
    incumbent, incumbent_rows = None, []
    started = time.time()
    for generation in range(generations):
        if deadline is not None and time.time() > deadline:
            log("past the deadline: stopping the search")
            break
        s = max(sigma * decay**generation, minimum_sigma)
        candidates = [dataclasses.replace(parent, name="parent")]
        if incumbent is not None:
            candidates.append(dataclasses.replace(incumbent, name="incumbent"))
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
        parent_score = weighted_score(parent_rows, objective)
        verdict = ""
        if incumbent is not None:
            incumbent_rows += by_candidate["incumbent"]
            kept = weighted_score(incumbent_rows, objective)
            if parent_score[0] < kept[0]:
                verdict = f" (probation failed: {parent_score[0]:.3f} < {kept[0]:.3f})"
                parent, parent_rows, parent_score = incumbent, incumbent_rows, kept
            else:
                verdict = f" (probation passed: {parent_score[0]:.3f} >= {kept[0]:.3f})"
            incumbent, incumbent_rows = None, []
        scores = {name: weighted_score(mine, objective) for name, mine in by_candidate.items()}
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
                                    [r for r in by_candidate[candidate.name] if r["b"] == o.name],
                                    objective,
                                )[0]
                                for o in opponents
                            },
                        )
                    )
                    + "\n"
                )
        children_scores = {n: v for n, v in scores.items() if n[0] in "et"}
        best = max(children_scores, key=lambda n: children_scores[n][0])
        score, error = children_scores[best]
        promoted = score > parent_score[0] + margin * math.hypot(error, parent_score[1])
        elapsed = time.time() - started
        left = elapsed / (generation + 1) * (generations - generation - 1)
        log(
            f"{datetime.datetime.now(KST):%H:%M:%S} generation {generation}: parent "
            f"{parent_score[0]:.3f} ± {parent_score[1]:.3f}{verdict}, best e "
            f"{max(v[0] for n, v in scores.items() if n.startswith('e')):.3f}, best t "
            f"{max(v[0] for n, v in scores.items() if n.startswith('t')):.3f}"
            f"{f' -> {best} {score:.3f} on probation' if promoted else ''}; "
            f"{time.time() - generation_started:.0f} s; ETA {clock(left)} KST"
        )
        if promoted:
            incumbent, incumbent_rows = parent, parent_rows
            chosen = candidates[[c.name for c in candidates].index(best)]
            parent, parent_rows = dataclasses.replace(chosen, name=parent.name), []
    if incumbent is not None:
        parent = incumbent
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
                    if damage * aspect <= 0.0:  # no damage at this aspect (a head-on fraction 0)
                        return math.inf
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


def encode(doctrine: Doctrine) -> list[float]:
    """A doctrine's searched numbers in the evolution strategy's space: a weight as itself,
    an exponent over 4, a logistic's centre and scale and the other positive numbers as
    logs (the fields `bench.doctrine_arena.FROZEN` left out)."""
    from bench.doctrine_arena import FROZEN
    from bnb.controllers.scripted_doctrine import EXPONENTS, FRACTIONS, NAMES, WEIGHTS
    from bnb.controllers.scripted_parameters import Logistic

    out = []
    for name in NAMES:
        value = getattr(doctrine, name)
        if name in FROZEN:
            continue
        if name in WEIGHTS or name in FRACTIONS:
            out.append(value)
        elif name in EXPONENTS:
            out.append(value / 4.0)
        elif isinstance(value, Logistic):
            out.extend((math.log(value.centre), math.log(value.scale)))
        else:
            out.append(math.log(value))
    return out


def decode(vector: list[float], like: Doctrine) -> Doctrine:
    """`encode`'s inverse (the frozen fields from `like`), clamped to the search's bounds."""
    from bench.doctrine_arena import FROZEN, clamp
    from bnb.controllers.scripted_doctrine import EXPONENTS, FRACTIONS, NAMES, WEIGHTS
    from bnb.controllers.scripted_parameters import Logistic

    values, column = {}, 0
    for name in NAMES:
        value = getattr(like, name)
        if name in FROZEN:
            values[name] = value
        elif name in WEIGHTS or name in FRACTIONS:
            values[name], column = vector[column], column + 1
        elif name in EXPONENTS:
            values[name], column = 4.0 * vector[column], column + 1
        elif isinstance(value, Logistic):
            values[name] = Logistic(
                math.exp(vector[column]), math.exp(vector[column + 1]), value.rising
            )
            column += 2
        else:
            values[name], column = math.exp(vector[column]), column + 1
    return clamp(Doctrine(**values))


# The scripted parameters the ES searches (the combat layer's and the memory's; Frontline's
# and the boundary's are left as they are).
SEARCHED_PARAMETERS = (
    "turn_in_stall_angle",
    "level_two_in_stall_angle",
    "level_three_in_stall_angle",
    "boost_slow_in_corner_speed",
    "boost_far_in_store_gun_ranges",
    "reverse_in_gun_range",
    "aim_in_target_size",
    "shoot_in_gun_range",
    "lead_cap_in_gun_range",
    "separation_radius_in_gun_range",
    "separation_weight",
    "target_switch_in_log_distance",
    "side_commit_in_stall_angle",
    "side_switch_in_radians",
    "memory_fade_in_seconds",
    "boundary_margin_in_turn_radius",
    "centre_hold_in_arena_radius",
)


def encode_parameters(row: tuple[float, ...]) -> list[float]:
    """The searched scripted parameters of a packed row, as logs (a logistic's centre and
    scale, a switch's rate, centre and scale, a float)."""
    from bnb.controllers.scripted_parameters import from_row

    parameters = from_row(list(row))
    out = []
    for name in SEARCHED_PARAMETERS:
        value = getattr(parameters, name)
        numbers = (
            (value.centre, value.scale)
            if hasattr(value, "rising")
            else tuple(value)
            if isinstance(value, tuple)
            else (value,)
        )
        out.extend(math.log(max(n, 1e-6)) for n in numbers)
    return out


def decode_parameters(vector: list[float], like: tuple[float, ...]) -> tuple[float, ...]:
    """`encode_parameters`' inverse into the packed row `like`, each logistic's and
    switch's scale held at its floor or above."""
    from bnb.controllers.scripted_parameters import (
        SCALE_FLOORS as PARAMETER_FLOORS,
    )
    from bnb.controllers.scripted_parameters import (
        SWITCH_SCALE_FLOORS,
        Logistic,
        Switch,
        from_row,
    )
    from bnb.controllers.scripted_parameters import pack as pack_parameters

    parameters = from_row(list(like))
    values, column = {}, 0
    for name in SEARCHED_PARAMETERS:
        value = getattr(parameters, name)
        if isinstance(value, Logistic):
            centre, scale = (math.exp(v) for v in vector[column : column + 2])
            scale = max(scale, PARAMETER_FLOORS.get(name, 0.0))
            values[name] = Logistic(centre, scale, value.rising)
            column += 2
        elif isinstance(value, Switch):
            rate, centre, scale = (math.exp(v) for v in vector[column : column + 3])
            values[name] = Switch(rate, centre, max(scale, SWITCH_SCALE_FLOORS.get(name, 0.0)))
            column += 3
        else:
            values[name] = math.exp(vector[column])
            column += 1
    return tuple(pack_parameters(dataclasses.replace(parameters, **values)))


PARTS_READ = ("ships", "fields", "globals", "zones")


def snapshots(loop: Loop, player: Player, every: int = 450, keep: int = 4096) -> Obs:
    """Observations of `player`'s mixed self-play: every `every` ticks, `keep` views
    (living or not) drawn at random, concatenated (the distinctness penalty's situations)."""
    state, observation, carry, tally = loop.start([Group(0, "mixed", player, player, loop.B)])
    taken = []
    for number in range(loop.ticks - 1):
        state, observation, carry, tally = loop._tick(state, observation, carry, tally)
        if number % every == every - 1:
            view = Obs(
                **{
                    n: getattr(observation, n).flatten(0, 1)
                    for n in ("ships", "fields", "globals", "zones")
                }
            )
            index = torch.randperm(view.ships.shape[0], device=loop.device)[:keep]
            taken.append(Obs(**{n: getattr(view, n)[index].clone() for n in PARTS_READ}))
    return Obs(**{n: torch.cat([getattr(t, n) for t in taken]) for n in PARTS_READ})


def doctrine_kl(loop: Loop, views: Obs, player: Player) -> float:
    """The mean symmetric KL (power + stick marginals, nats) between the player's energy and
    turn doctrines flown by every living own ship of `views` (the memoryless agent, the
    player's parameters)."""
    from bnb.contracts.obs import SHIP
    from bnb.controllers.scripted_parameters import unpack

    rows = views.ships.shape[0]
    parameters = unpack(
        torch.tensor(
            [list(player.parameters or default_parameters())] * rows,
            dtype=torch.float32,
            device=loop.device,
        )
    )
    with torch.no_grad():
        parts = [
            scripted.marginals(
                views,
                loop.card,
                parameters,
                1.0,
                None,
                doctrines=torch.tensor([pack(d)] * (2 * loop.K), device=loop.device),
            )
            for d in (player.energy, player.turn)
        ]
    alive = views.ships[:, : loop.S, SHIP["alive"]] > 0.5
    total = 0.0
    for part in ("power", "stick"):
        p = getattr(parts[0], part).clamp_min(1e-6)
        q = getattr(parts[1], part).clamp_min(1e-6)
        total = total + ((p - q) * (p.log() - q.log())).sum(-1)
    return float(total[alive].mean())


def es_search(
    loop: Loop,
    parent: Player,
    opponents: list[Player],
    log_path: pathlib.Path,
    generations: int,
    pairs: int,
    games: int,
    sigma: float,
    step: float = 1.0,
    seed: int = 5,
    variant: int = 0,
    deadline: float | None = None,
    log=print,
    objective: str = "kill_score",
    lineups: tuple[str, ...] = SEARCH_LINEUPS,
    parts: tuple[str, ...] = ("energy", "turn"),
    distinct_floor: float = 0.0,
    distinct_weight: float = 0.1,
) -> Player:
    """An evolution strategy over `parts` of a player at once (the two types' doctrines in
    `encode`'s space, the scripted parameters in `encode_parameters`'): each
    generation `pairs` antithetic pairs of perturbations (sigma N(0, 1) per number) play
    `games` games per opponent and lineup as side A, beside the parent; the gradient is the
    pairs' score differences times their directions; the parent on probation moves `step`
    sigma along the normalised gradient. The moved parent stays only if its fresh score in
    the next generation beats the incumbent's (whose games accumulate), else the incumbent
    returns and the step halves (a pass grows it 1.5x, up to 1). With `distinct_floor` > 0,
    each candidate's score loses distinct_weight x max(0, 1 - KL / distinct_floor), KL its
    two doctrines' `doctrine_kl`
    on the parent's situations (`snapshots`, taken once): the doctrines may not collapse
    into one. Every candidate is a line of the log."""
    generator = torch.Generator().manual_seed(seed)
    base_parameters = parent.parameters or default_parameters()
    encoded = {
        "energy": encode(parent.energy),
        "turn": encode(parent.turn),
        "parameters": encode_parameters(base_parameters),
    }
    sizes = {part: len(encoded[part]) for part in parts}
    centre = torch.tensor([v for part in parts for v in encoded[part]], dtype=torch.float64)
    parent_rows: list[dict] = []
    incumbent = None  # (centre, rows)
    started = time.time()
    views = snapshots(loop, parent) if distinct_floor > 0 else None

    def penalty(player: Player) -> tuple[float, float]:
        if views is None:
            return 0.0, 0.0
        kl = doctrine_kl(loop, views, player)
        return distinct_weight * max(0.0, 1.0 - kl / distinct_floor), kl

    def player_at(vector: torch.Tensor, name: str) -> Player:
        values, column, changes = vector.tolist(), 0, {}
        for part in parts:
            piece = values[column : column + sizes[part]]
            column += sizes[part]
            if part == "parameters":
                changes[part] = decode_parameters(piece, base_parameters)
            else:
                changes[part] = decode(piece, getattr(parent, part))
        return dataclasses.replace(parent, name=name, **changes)

    for generation in range(generations):
        if deadline is not None and time.time() > deadline:
            log("past the deadline: stopping the search")
            break
        noise = torch.randn((pairs, centre.shape[0]), generator=generator, dtype=torch.float64)
        candidates = [player_at(centre, "parent")]
        if incumbent is not None:
            candidates.append(player_at(incumbent[0], "incumbent"))
        for number in range(pairs):
            candidates.append(player_at(centre + sigma * noise[number], f"p{number}"))
            candidates.append(player_at(centre - sigma * noise[number], f"m{number}"))
        jobs = [
            Job(variant, lineup, candidate, opponent, games)
            for candidate in candidates
            for opponent in opponents
            for lineup in lineups
        ]
        generation_started = time.time()
        outs = run(loop, jobs, log=lambda line: None)
        seconds = loop.ticks * loop.rules.tick_length
        rows = [job_summary(j, o, seconds) for j, o in zip(jobs, outs, strict=True)]
        by: dict[str, list[dict]] = {}
        for r in rows:
            by.setdefault(r["a"], []).append(r)
        penalties = {c.name: penalty(c) for c in candidates}

        def scored(name: str, mine: list[dict]) -> tuple[float, float]:
            mean, error = weighted_score(mine, objective)
            return mean - penalties[name][0], error

        parent_rows += by["parent"]
        parent_score = scored("parent", parent_rows)
        verdict = ""
        if incumbent is not None:
            incumbent_rows = incumbent[1] + by["incumbent"]
            kept = scored("incumbent", incumbent_rows)
            if parent_score[0] < kept[0]:
                verdict = f" (probation failed: {parent_score[0]:.3f} < {kept[0]:.3f})"
                centre, parent_rows, parent_score = incumbent[0], incumbent_rows, kept
                step *= 0.5
            else:
                verdict = f" (probation passed: {parent_score[0]:.3f} >= {kept[0]:.3f})"
                step = min(1.5 * step, 1.0)
            incumbent = None
        scores = {name: scored(name, mine)[0] for name, mine in by.items()}
        difference = torch.tensor(
            [scores[f"p{n}"] - scores[f"m{n}"] for n in range(pairs)], dtype=torch.float64
        )
        gradient = (difference[:, None] * noise).sum(0) / (2 * pairs * sigma)
        with log_path.open("a") as handle:
            handle.write(
                json.dumps(
                    dict(
                        generation=generation,
                        parent_score=parent_score,
                        verdict=verdict,
                        sigma=sigma,
                        step=step,
                        gradient_norm=float(gradient.norm()),
                        parent_kl=penalties["parent"][1],
                        mean_child=float(sum(scores.values()) / len(scores)),
                        best_child=max(v for n, v in scores.items() if n[0] in "pm"),
                        parent=player_at(centre, parent.name).named(),
                        by_opponent={
                            o.name: weighted_score(
                                [r for r in by["parent"] if r["b"] == o.name], objective
                            )[0]
                            for o in opponents
                        },
                    )
                )
                + "\n"
            )
        elapsed = time.time() - started
        left = elapsed / (generation + 1) * (generations - generation - 1)
        log(
            f"{datetime.datetime.now(KST):%H:%M:%S} generation {generation}: parent "
            f"{parent_score[0]:.3f} ± {parent_score[1]:.3f}{verdict}; children mean "
            f"{float(sum(scores.values()) / len(scores)):.3f} best "
            f"{max(v for n, v in scores.items() if n[0] in 'pm'):.3f}; |g| "
            f"{float(gradient.norm()):.3f}, step {step:.2f}, KL {penalties['parent'][1]:.3f}; "
            f"{time.time() - generation_started:.0f} s; ETA {clock(left)} KST"
        )
        if generation < generations - 1 and float(gradient.norm()) > 0:
            incumbent = (centre, parent_rows)
            centre = (
                centre
                + step * sigma * gradient / gradient.norm() * math.sqrt(centre.shape[0]) ** 0.5
            )
            parent_rows = []
    if incumbent is not None:
        centre = incumbent[0]
    return player_at(centre, parent.name)
