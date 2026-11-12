"""Mode dynamics of the scripted agent (T = 1, both sides, the game's default lineup): the
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
(appended; a line per batch, so a stopped run keeps its batches). Eager, on `--device`
(cuda if available). Overrides: `--ships`, `--radius` (m), `--zone-radius` (m at map scale
1), `--capture-seconds`, `--episode-seconds`, `--temperature` (the agent's; a pair a,b plays
team 0 at a and team 1 at b... per environment the pair is swapped on odd environments).
"""

import argparse
import dataclasses
import datetime
import json
import pathlib
import time
import zoneinfo

import torch

from bnb.contracts import Action, Obs
from bnb.controllers import scripted
from bnb.game import api
from bnb.game.handling.characteristics import characteristics

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
    mode_changes = {}
    if arguments.zone_radius:
        mode_changes["zone_radius"] = arguments.zone_radius
    if arguments.capture_seconds:
        mode_changes["capture_seconds"] = arguments.capture_seconds
    if mode_changes:
        rules = api.tuning.tuned(
            rules, {f"mode.{key}": value for key, value in mode_changes.items()}
        )
    return game, types, rules


def batch(arguments, game, types, rules, device, seed: int) -> list[dict]:
    """One batch of first games to their ends: a dict per game."""
    torch.manual_seed(seed)
    B, S, N = arguments.batch, rules.ships_per_team, rules.num_ships
    draws = torch.Generator(device=device).manual_seed(seed + 1)
    state = api.reset(rules, types, api.uniforms(rules, B, generator=draws, device=device))
    state, observation, _ = api.perceive(rules, state)
    card = characteristics(types, rules)
    temperatures = [float(value) for value in arguments.temperature.split(",")]
    if len(temperatures) == 1:
        temperature = temperatures[0]
    else:
        swap = torch.arange(B, device=device) % 2 == 1
        first = torch.where(swap, temperatures[1], temperatures[0])
        second = torch.where(swap, temperatures[0], temperatures[1])
        temperature = torch.stack([first, second], 1).flatten()  # views: env-major, team
    agent = scripted.Scripted(card, S, temperature=temperature)
    carry = agent.init_carry(2 * B, device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    end_tick = torch.full((B,), -1, dtype=torch.long, device=device)
    result = torch.zeros(B, dtype=torch.long, device=device)
    front = state.mode.front.clone()
    events_list: list[torch.Tensor] = []  # rows (environment, tick, team)
    episode = int(rules.pools[0].episode_ticks)
    team_of = torch.arange(N, device=device) // S
    with torch.no_grad():
        for tick in range(episode + 2):
            view = Obs(
                **{
                    name: getattr(observation, name).flatten(0, 1)
                    for name in ("ships", "fields", "globals", "zones")
                }
            )
            action, carry, _ = agent.step(
                carry, view, torch.full((2 * B,), tick == 0, device=device)
            )
            action = Action.unpack(action.packed().reshape(B, N, 3).to(torch.int32))
            moved, sweep = api.move(
                rules, types, state, action, api.spread(rules, B, generator=draws, device=device)
            )
            state, events = api.resolve(
                rules,
                types,
                moved,
                sweep,
                action,
                api.uniforms(rules, B, generator=draws, device=device),
            )
            live = ~finished
            if arguments.kind == "deathmatch":
                died = events.died & live[:, None]
                where = died.nonzero()
                if len(where):
                    scorer = 1 - team_of[where[:, 1]]
                    events_list.append(
                        torch.stack([where[:, 0], torch.full_like(scorer, tick), scorer], 1)
                    )
            else:
                step = (state.mode.front - front).long()
                changed = (step != 0) & live & ~events.done
                where = changed.nonzero()[:, 0]
                if len(where):
                    team = (step[where] < 0).long()  # +1: team 0 captured
                    events_list.append(torch.stack([where, torch.full_like(where, tick), team], 1))
                front = state.mode.front.clone()
            ending = events.done & live
            end_tick = torch.where(ending, tick, end_tick)
            result = torch.where(ending, events.result.long(), result)
            finished |= events.done
            state, observation, _ = api.perceive(rules, state)
            if bool(finished.all()):
                break
    rows = torch.cat(events_list).cpu() if events_list else torch.zeros((0, 3), dtype=torch.long)
    games = []
    for environment in range(B):
        mine = rows[rows[:, 0] == environment]
        games.append(
            dict(
                end_tick=int(end_tick[environment]),
                result=int(result[environment]),
                by_rule=0 <= int(end_tick[environment]) < episode - 1,  # before the clock
                events=mine[:, 1:].tolist(),
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
    parser.add_argument("--temperature", default="1")
    parser.add_argument("--seed", type=int, default=0)
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
    while done < arguments.games:
        began = time.time()
        games = batch(arguments, game, types, rules, device, arguments.seed + 1000 * index)
        entry = dict(
            label=arguments.label,
            kind=arguments.kind,
            preset=arguments.preset,
            ships=rules.ships_per_team,
            radius=game.arena_radius,
            tick_length=rules.tick_length,
            episode_ticks=int(rules.pools[0].episode_ticks),
            zone_radius=rules.pools[0].mode.zone_radius,
            capture_seconds=rules.pools[0].mode.capture_seconds,
            temperature=arguments.temperature,
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
