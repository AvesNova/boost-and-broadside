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
`--capture-seconds`, `--front-threshold`, `--episode-seconds`, `--temperature` (the agent's, both teams).
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

from bnb.contracts import Pool
from bnb.controllers import scripted
from bnb.game import api, ships
from bnb.game.handling.characteristics import characteristics
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
        for _ in range(math.ceil(length / rollout.length)):
            rollout.run()
        host = {name: value.cpu() for name, value in trace.items()}
        games = first_games(host, episode, threshold, arguments.kind)
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
