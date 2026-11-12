"""Mode dynamics of the scripted agent (T = 1, both sides, the game's lineup or `--lineup`): the
deathmatch's kill rate and the Frontline's captures, on a preset with no fields (the agent
ignores fields; Frontline keeps its zones).

    PYTHONPATH=src python -m bench.mode_dynamics deathmatch --out FILE [--games 256]
    PYTHONPATH=src python -m bench.mode_dynamics frontline --out FILE [--games 256]

deathmatch (`--preset real_scale_deathmatch`): the win by tickets is switched off
(kills_per_ship 10^6), so every game plays its whole clock; per game, every point's tick and
scoring team (a death scores for the other team, whatever its cause), from which any
tickets and clock can be replayed (`report`).

frontline (`--preset real_scale_frontline`): the mode's own rules; per game, every capture's
tick and team (the front's step), and how the game ended: by the front reaching its
threshold, or by the clock.

Each environment's first game is recorded; every `--batch` games are one JSON line of FILE
(appended; a line per batch, so a stopped run keeps its batches). On the compiled rollout
(`bnb.runtime.rollout.Rollout`, the scripted agent on both seats) with a stage that traces
each environment's score, front and clock every tick, on `--device` (cuda if
available). Overrides: `--ships`, `--radius` (m), `--zone-radius` (m at map scale 1),
`--capture-seconds`, `--front-threshold`, `--episode-seconds`, `--temperature` (the
agent's, both teams).
"""

import argparse
import dataclasses
import datetime
import json
import math
import pathlib
import time
import types as types_module
import zoneinfo

import torch

from bnb.contracts import Pool
from bnb.controllers import scripted
from bnb.game import api, ships
from bnb.game.handling.characteristics import characteristics
from bnb.game.modes import frontline
from bnb.runtime import rollout as rollout_module
from bnb.runtime.rollout import Rollout

KST = zoneinfo.ZoneInfo("Asia/Seoul")
NO_TICKETS = 1_000_000


def clock(seconds: float) -> str:
    """The KST clock time `seconds` from now."""
    return (datetime.datetime.now(KST) + datetime.timedelta(seconds=seconds)).strftime("%H:%M")


def game_of(arguments) -> tuple:
    """The game config and rules (and the card) of the run."""
    game = dataclasses.replace(api.PRESETS[arguments.preset], num_fields=0)
    changes = {}
    if arguments.ships:
        changes["ships_per_team"] = arguments.ships
    if arguments.radius:
        changes["arena_radius"] = arguments.radius
    if arguments.episode_seconds:
        changes["episode_seconds"] = arguments.episode_seconds
    if arguments.kind == "deathmatch":
        changes["kills_per_ship"] = NO_TICKETS
    game = dataclasses.replace(game, **changes)
    types = api.ship_types(ship_set=game.ship_set)
    rules = api.rules_of(game, types=types)
    return game, types, rules


def lineup_of(lineup: str | None, game, environments: int) -> torch.Tensor | None:
    """(environments, N) int32 ship types: the keys cycled over each team's slots, the same
    for both teams; None for the game's lineup."""
    if lineup is None:
        return None
    keys = [ship.key for ship in ships.load(game.ship_set)]
    chosen = [keys.index(key) for key in lineup.split(",")]
    team = [chosen[slot % len(chosen)] for slot in range(game.ships_per_team)]
    return torch.tensor(team + team, dtype=torch.int32).repeat(environments, 1)


def mode_overrides(arguments) -> dict:
    """The mode's live settings the run changes (`bnb.game.tuning`)."""
    changes = {}
    if arguments.zone_radius:
        changes["mode.zone_radius"] = arguments.zone_radius
    if arguments.capture_seconds:
        changes["mode.capture_seconds"] = arguments.capture_seconds
    if arguments.front_threshold:
        changes["mode.front_threshold"] = arguments.front_threshold
    return changes


BINS = 900  # 1 s histogram bins (the last one holds every longer time)
HISTOGRAMS = ("engage", "life", "objective", "zone_to_zone", "contest")


class Probe:
    """Per-ship timings inside the resolve stage (`api.resolve` wrapped while the rollout
    compiles, so the tick's events are at hand), as 1 s histograms over every ship of the
    batch's first games: engage, from a spawn to the ship's first hit dealt or taken (enemy
    hits only); life, from a spawn to the ship's death. Frontline: objective, from a spawn to
    the ship's first entry into a zone other than its spawn zone; zone_to_zone, from leaving
    a zone to entering another one; contest, from the last tick every meter was at zero to
    a capture. Per environment: the first game's first enemy hit tick (-1 none)."""

    def __init__(self, environments: int, ships_per_team: int, device) -> None:
        N = 2 * ships_per_team
        self.S = ships_per_team
        self.buffers = dict(
            tick=torch.zeros(1, dtype=torch.long, device=device),
            live=torch.ones(environments, dtype=torch.bool, device=device),
            since_spawn=torch.zeros((environments, N), dtype=torch.long, device=device),
            engaged=torch.zeros((environments, N), dtype=torch.bool, device=device),
            reached=torch.zeros((environments, N), dtype=torch.bool, device=device),
            last_zone=torch.full((environments, N), -1, dtype=torch.long, device=device),
            since_zone=torch.zeros((environments, N), dtype=torch.long, device=device),
            quiet_tick=torch.zeros(environments, dtype=torch.long, device=device),
            front=torch.zeros(environments, dtype=torch.long, device=device),
            first_hit=torch.full((environments,), -1, dtype=torch.long, device=device),
            **{name: torch.zeros(BINS, dtype=torch.long, device=device) for name in HISTOGRAMS},
        )
        self.ticks_per_bin = 30

    def clear(self) -> None:
        for name, value in self.buffers.items():
            if name in ("first_hit", "last_zone"):
                value.fill_(-1)
            elif name == "live":
                value.fill_(True)
            else:
                value.zero_()

    def _count(self, name: str, ticks: torch.Tensor, mask: torch.Tensor) -> None:
        bins = (ticks // self.ticks_per_bin).clamp(0, BINS - 1)
        self.buffers[name].index_add_(0, bins.flatten(), mask.flatten().long())

    def observe(self, state, events, frontline_mode: bool) -> None:
        b = self.buffers
        live = b["live"][:, None]
        S = self.S
        N = 2 * S
        team = torch.arange(N, device=events.died.device) // S
        enemy = team[:, None] != team[None, :]
        dealt = (torch.where(enemy, events.damage, 0.0) > 0).any(-1)
        taken = events.enemy_damage_by_aspect.sum(-1) > 0
        hit = dealt | taken
        alive = state.ships.alive
        since = b["since_spawn"] + 1
        first_engage = hit & ~b["engaged"] & live
        self._count("engage", since, first_engage)
        self._count("life", since, events.died & live)
        game_hit = (hit.any(-1) & (b["first_hit"] < 0)) & b["live"]
        b["first_hit"].copy_(torch.where(game_hit, b["tick"], b["first_hit"]))
        engaged = b["engaged"] | hit
        if frontline_mode:
            mode = state.mode
            position = torch.stack([state.ships.x, state.ships.y], -1)
            zones = torch.stack([mode.zone_x, mode.zone_y], -1)
            inside = frontline.zone_membership(position, zones, mode.zone_radius)  # (B, N, Z)
            spawn_role = torch.where(team == 0, frontline.TEAM0_SPAWN, frontline.TEAM1_SPAWN)
            own_spawn = mode.role[:, None, :] == spawn_role[None, :, None]
            objective = (inside & ~own_spawn).any(-1) & alive
            first_objective = objective & ~b["reached"] & live
            self._count("objective", since, first_objective)
            b["reached"].copy_((b["reached"] | objective) & ~events.spawned)
            zone = torch.where(
                inside.any(-1), inside.long().argmax(-1), torch.full_like(b["last_zone"], -1)
            )
            entering = (zone >= 0) & (b["last_zone"] >= 0) & (zone != b["last_zone"]) & alive
            outside = zone < 0
            self._count("zone_to_zone", b["since_zone"], entering & live & (b["since_zone"] > 0))
            b["since_zone"].copy_(torch.where(outside, b["since_zone"] + 1, 0))
            last = torch.where(zone >= 0, zone, b["last_zone"])
            b["last_zone"].copy_(torch.where(events.spawned | ~alive, -1, last))
            # a capture resets the meters, so the contest is timed from the last all-zero
            # tick before this one
            captured = (mode.front.long() != b["front"]) & b["live"] & ~events.done
            self._count("contest", b["tick"] - b["quiet_tick"], captured)
            quiet = (mode.progress.abs() == 0).all(-1)
            b["quiet_tick"].copy_(torch.where(quiet, b["tick"], b["quiet_tick"]))
            b["front"].copy_(mode.front.long())
        b["engaged"].copy_(engaged & ~events.spawned & alive)
        b["since_spawn"].copy_(torch.where(events.spawned, 0, since))
        b["live"].copy_(b["live"] & ~events.done)
        b["tick"].add_(1)


def install(probe: Probe, frontline_mode: bool) -> None:
    """Wrap the rollout's `api.resolve` so every resolve feeds the probe."""
    original = rollout_module.api.resolve

    def resolve(*arguments, **keywords):
        state, events = original(*arguments, **keywords)
        probe.observe(state, events, frontline_mode)
        return state, events

    rollout_module.api = types_module.SimpleNamespace(
        **{name: getattr(api, name) for name in dir(api) if not name.startswith("__")}
    )
    rollout_module.api.resolve = resolve


def trace_stage(length: int):
    """The stage appended to the rollout's tick: each environment's score, front and episode
    tick after the tick, at the trace's tick (the last slot holds every tick past it)."""

    def stage(carry: dict) -> None:
        trace, state = carry["trace"], carry["state"]
        index = trace["tick"].clamp(max=length - 1)
        trace["score"].index_copy_(1, index, state.mode.score[:, None].to(torch.int16))
        trace["front"].index_copy_(1, index, state.mode.front[:, None].to(torch.int8))
        trace["clock"].index_copy_(1, index, state.match.episode_tick[:, None].to(torch.int32))
        trace["tick"].add_(1)

    return stage


def blank_trace(environments: int, length: int, device) -> dict:
    return dict(
        tick=torch.zeros(1, dtype=torch.long, device=device),
        score=torch.zeros((environments, length, 2), dtype=torch.int16, device=device),
        front=torch.zeros((environments, length), dtype=torch.int8, device=device),
        clock=torch.zeros((environments, length), dtype=torch.int32, device=device),
    )


def first_games(trace: dict, episode: int, threshold: int, kind: str) -> list[dict]:
    """Each environment's first game from its trace (host tensors): its end tick, result,
    whether it ended before the clock, and its events (tick, team): a point (deathmatch) or
    a capture (frontline)."""
    score, front, clock = (trace[name].long() for name in ("score", "front", "clock"))
    games = []
    for environment in range(clock.shape[0]):
        ticks = clock[environment]
        restart = ((ticks[1:] < ticks[:-1]).nonzero()[:, 0] + 1).tolist()
        end = restart[0] - 1 if restart else ticks.shape[0] - 1  # the game's last state
        events = []
        if kind == "deathmatch":
            points = score[environment, : end + 1]
            steps = torch.diff(points, dim=0, prepend=torch.zeros_like(points[:1]))
            for at, team in steps.nonzero().tolist():
                events += [[at, team]] * int(steps[at, team])
            final = points[-1]
            result = int(torch.sign(final[0] - final[1]))
        else:
            fronts = front[environment, : end + 1]
            steps = torch.diff(fronts, prepend=torch.zeros_like(fronts[:1]))
            for at in steps.nonzero()[:, 0].tolist():
                events.append([at, 0 if int(steps[at]) > 0 else 1])
            last = int(fronts[-1])
            result = int(math.copysign(1, last)) if last else 0
            if restart and end < episode - 1 and abs(last) == threshold - 1:
                events.append([end + 1, 0 if last > 0 else 1])  # the winning capture
        games.append(
            dict(
                end_tick=end + 1 if restart else end,
                result=result,
                by_rule=bool(restart) and end + 1 < episode - 1,
                events=events,
            )
        )
    return games


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("deathmatch", "frontline"))
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--ships", type=int, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--zone-radius", type=float, default=None)
    parser.add_argument("--capture-seconds", type=float, default=None)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--front-threshold", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lineup", default=None, help="type keys cycled over each team's slots (both teams)"
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()
    if arguments.preset is None:
        arguments.preset = f"real_scale_{arguments.kind}"
    device = torch.device(arguments.device)
    game, types, rules = game_of(arguments)
    types = api.ship_types(ship_set=game.ship_set, device=device)
    done = 0
    if arguments.out.exists():
        for line in arguments.out.read_text().splitlines():
            entry = json.loads(line)
            if entry["label"] == arguments.label:
                done += len(entry["games"])
    started = time.time()
    index = done // arguments.batch
    episode = int(rules.pools[0].episode_ticks)
    length = episode + 2
    card = characteristics(types, rules)
    agent = scripted.Scripted(card, rules.ships_per_team, temperature=arguments.temperature)
    environments = arguments.batch
    probe = Probe(environments, rules.ships_per_team, device)
    install(probe, arguments.kind == "frontline")
    rollout = Rollout(
        dataclasses.replace(game),
        types,
        (Pool("dynamics", 1.0, ("a", "b")),),
        {"a": agent, "b": agent},
        "a",
        environments,
        32,
        device,
        staggered=False,
        ship_type=lineup_of(arguments.lineup, game, environments),
    )
    if mode_overrides(arguments):
        rollout.rules = api.tuning.tuned(rollout.rules, mode_overrides(arguments))
    rollout.carry["trace"] = blank_trace(environments, length, device)
    rollout.stages = rollout.stages + (
        torch.compile(trace_stage(length), fullgraph=True, dynamic=False),
    )
    threshold = rollout.rules.pools[0].mode.front_threshold
    while done < arguments.games:
        began = time.time()
        torch.manual_seed(arguments.seed + 1000 * index)
        trace = rollout.carry.pop("trace")
        rollout.restart()
        rollout.carry["trace"] = trace
        for name, value in blank_trace(environments, length, device).items():
            trace[name].copy_(value)
        probe.clear()
        for _ in range(math.ceil(length / rollout.length)):
            rollout.run()
        host = {name: value.cpu() for name, value in trace.items()}
        games = first_games(host, episode, threshold, arguments.kind)
        first_hit = probe.buffers["first_hit"].cpu().tolist()
        for game_record, hit_tick in zip(games, first_hit):
            game_record["first_hit"] = hit_tick
        histograms = {name: probe.buffers[name].cpu().tolist() for name in HISTOGRAMS}
        entry = dict(
            label=arguments.label,
            kind=arguments.kind,
            preset=arguments.preset,
            ships=rules.ships_per_team,
            radius=game.arena_radius,
            tick_length=rules.tick_length,
            episode_ticks=int(rules.pools[0].episode_ticks),
            zone_radius=rollout.rules.pools[0].mode.zone_radius,
            capture_seconds=rollout.rules.pools[0].mode.capture_seconds,
            front_threshold=threshold,
            temperature=arguments.temperature,
            lineup=arguments.lineup,
            histograms=histograms,
            games=games,
        )
        with arguments.out.open("a") as file:
            file.write(json.dumps(entry) + "\n")
        done += len(games)
        index += 1
        seconds = time.time() - began
        left = max(0, arguments.games - done) / arguments.batch * seconds
        print(
            f"{arguments.label or arguments.kind}: {done}/{arguments.games} games, batch "
            f"{seconds:.0f} s, ETA {clock(left)} (elapsed {time.time() - started:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
