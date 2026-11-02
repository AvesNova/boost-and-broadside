"""Self-play per-seat scores with exactly symmetric controllers (chaser, random).

usage: selfplay.py PRESET KIND GAMES SEED [fog|clear] [compile]
Score for team T: win 1, draw 0.5, loss 0 in each environment's first finished episode.
"""
import sys
import time

import torch

sys.path.insert(0, "/home/vizia/avesnova/boost-and-broadside-v2")
from bench.lock import bench_lock
from bnb.contracts import Action, Obs
from bnb.controllers.scripted import Chaser, Random, Scripted
from bnb.game.handling.characteristics import characteristics
from bnb.game import api
from bnb.game.modes.rules import PRESETS


def view(obs: Obs, v: int) -> Obs:
    return Obs(ships=obs.ships[:, v], fields=obs.fields[:, v], globals=obs.globals[:, v],
               zones=obs.zones[:, v])


def play(preset, kind, games, seed, fog=True, compiled=False, device="cuda", patch=None):
    torch.manual_seed(seed)
    config = PRESETS[preset]
    rules = api.rules_of(config)
    types = api.ship_types(device)
    S = rules.ships_per_team
    if kind == "scripted":
        measured = characteristics(types, rules)
        make = lambda: Scripted(measured)  # noqa: E731
    else:
        make = (lambda: Chaser(S)) if kind == "chaser" else (lambda: Random(S))
    controllers = [make(), make()]
    carries = [c.init_carry(games, device) for c in controllers]
    state = api.reset(rules, types, api.uniforms(rules, games, device),
                      torch.full((games,), fog, device=device))
    state, obs, _ = api.perceive(rules, state)
    stepper = api.step
    perceiver = api.perceive
    if compiled:
        stepper = torch.compile(api.step)
        perceiver = torch.compile(api.perceive)
    result = torch.zeros(games, dtype=torch.int32, device=device)
    finished = torch.zeros(games, dtype=torch.bool, device=device)
    done = torch.zeros(games, dtype=torch.bool, device=device)
    stats = dict(ticks=0)
    team_of = torch.arange(rules.num_ships, device=device) // S
    deaths = torch.zeros(2, device=device)
    friendly = torch.zeros(2, device=device)  # deaths credited to a shooter of the own team
    start = time.time()
    while not bool(finished.all()):
        actions = []
        for v in (0, 1):
            a, carries[v], _ = controllers[v].step(carries[v], view(obs, v), done)
            actions.append(a.packed())
        packed = torch.cat(actions, 1)
        state, events = stepper(rules, types, state, Action.unpack(packed),
                                api.spread(rules, games, device),
                                api.uniforms(rules, games, device))
        state, obs, _ = perceiver(rules, state)
        done = events.done
        live = (~finished)[:, None]
        died = events.died & live
        own = died & (events.killer >= 0) & (events.killer.clamp_min(0) // S == team_of)
        deaths += torch.stack([died[:, :S].sum(), died[:, S:].sum()]).float()
        friendly += torch.stack([own[:, :S].sum(), own[:, S:].sum()]).float()
        first = done & ~finished
        result = torch.where(first, events.result, result)
        finished |= done
        stats["ticks"] += 1
    team0 = (result > 0).float() + 0.5 * (result == 0).float()
    mean0 = float(team0.mean())
    se = float(team0.std()) / games ** 0.5
    draws = float((result == 0).float().mean())
    return dict(preset=preset, kind=kind, games=games, seed=seed, fog=fog, team0=mean0,
                team1=1 - mean0, se=se, draws=draws,
                deaths_per_game=[round(float(d) / games, 3) for d in deaths],
                friendly_kills_per_game=[round(float(d) / games, 3) for d in friendly], ticks=stats["ticks"],
                seconds=round(time.time() - start, 1))


if __name__ == "__main__":
    preset, kind, games, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    fog = (sys.argv[5] if len(sys.argv) > 5 else "fog") == "fog"
    compiled = len(sys.argv) > 6 and sys.argv[6] == "compile"
    with bench_lock():
        print(play(preset, kind, games, seed, fog, compiled), flush=True)
