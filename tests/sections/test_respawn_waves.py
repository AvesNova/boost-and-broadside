"""Respawn waves of the frontline design bench (`bench.frontline_design.Design.waves`): a
synthetic tick loop (the step's death and respawn counting, as `bnb.game.physics.step`)
with deaths and captures at chosen ticks, at tick length 1 s so ticks read as seconds."""

import dataclasses
import types

import torch

from bench.frontline_design import DELAY, RESET, SKIP, WAIT_BIN, Design

S = 2  # ships a team: ships 0 and 1 on team 0, 2 and 3 on team 1
GAME_WAIT = 16  # the game's fixed wait (ticks)


@dataclasses.dataclass
class Ships:
    alive: torch.Tensor
    respawn_wait: torch.Tensor


@dataclasses.dataclass
class Mode:
    front: torch.Tensor


@dataclasses.dataclass
class State:
    ships: Ships
    mode: Mode


def run(variants: list[dict], deaths: dict, fronts: dict, phases, ticks: int = 60):
    """Spawn ticks per (environment, ship) of a synthetic loop (`Design.clear` draws the
    wave phases from the global generator; `phases` overrides them). deaths: {(env, ship): [tick]};
    fronts: {(env, tick): front}; phases (B, 2): ticks to each team's first wave."""
    B = len(variants)
    design = Design(B, S, 1.0, torch.device("cpu"), slots=B)
    design.set(variants)
    design.clear()
    design.buffers["wave_left"].copy_(torch.tensor(phases))
    alive = torch.ones((B, 2 * S), dtype=torch.bool)
    respawn_wait = torch.zeros((B, 2 * S), dtype=torch.int32)
    front = torch.zeros(B, dtype=torch.int32)
    spawns = {}
    for tick in range(ticks):
        was_dead = ~alive
        respawn_wait = torch.where(was_dead, respawn_wait - 1, respawn_wait)
        due = was_dead & (respawn_wait <= 0)
        died = torch.zeros_like(alive)
        for (environment, ship), at in deaths.items():
            if tick in at:
                died[environment, ship] = True
        died &= alive
        alive = (alive & ~died) | due
        respawn_wait = torch.where(died, GAME_WAIT, respawn_wait).int()
        for environment in range(B):
            front[environment] = fronts.get((environment, tick), front[environment])
        for environment, ship in due.nonzero().tolist():
            spawns.setdefault((environment, ship), []).append(tick)
        events = types.SimpleNamespace(
            done=torch.zeros(B, dtype=torch.bool), spawned=due, died=died
        )
        state = design.waves(State(Ships(alive, respawn_wait), Mode(front.clone())), events, S)
        alive, respawn_wait = state.ships.alive, state.ships.respawn_wait
        design.buffers["front"].copy_(front.long())
    return spawns, design


def test_respawn_on_the_first_wave_after_the_minimum(global_generator):
    """Waves every 10 s (team 0 at 3, 13, 23, 33; team 1 at 7, 17), minimum 4 s; and the
    minimum alone (no waves), and the game's fixed wait (both off)."""
    variants = [
        dict(label="waves", respawn_minimum=4, wave_interval=10),
        dict(label="minimum", respawn_minimum=4),
        dict(label="game"),
    ]
    deaths = {(0, 0): [5, 20], (0, 2): [2], (1, 0): [5], (2, 0): [5]}
    spawns, design = run(variants, deaths, {}, [[3, 7], [1, 1], [1, 1]])
    assert spawns[(0, 0)] == [13, 33]  # dead at 5: 9 is the minimum, 13 the wave
    assert spawns[(0, 2)] == [7]  # dead at 2: 6 is the minimum, 7 team 1's wave
    assert spawns[(1, 0)] == [9]
    assert spawns[(2, 0)] == [5 + GAME_WAIT]
    waits = [8, 13, 5]  # the waves environment's respawns
    histogram = design.wait_histogram(0)
    assert sum(histogram) == len(waits)
    for wait in waits:
        assert histogram[int(wait / WAIT_BIN)] >= 1
    assert abs(float(design.buffers["wait_sum"][0]) - sum(waits)) < 1e-9
    assert abs(float(design.buffers["wait_sum"][1]) - 4) < 1e-9


def test_capture_effects_on_the_losing_team(global_generator):
    """Team 1 loses a zone at 10 (the front rises): its waves (7, 17, 27 untouched) move to
    20 (reset), 22 (delayed 5 s) or 27 (one skipped); a ship of team 1 dead at 12 respawns
    there. Team 0's waves never move; nor do team 1's when team 0 loses the zone."""
    variants = [
        dict(label="none", respawn_minimum=4, wave_interval=10),
        dict(label="reset", respawn_minimum=4, wave_interval=10, wave_effect=RESET),
        dict(label="delay", respawn_minimum=4, wave_interval=10, wave_effect=DELAY, wave_delay=5),
        dict(label="skip", respawn_minimum=4, wave_interval=10, wave_effect=SKIP),
        dict(label="other", respawn_minimum=4, wave_interval=10, wave_effect=RESET),
    ]
    deaths = {(e, 2): [12] for e in range(5)} | {(e, 0): [12] for e in range(5)}
    fronts = {(e, 10): 1 for e in range(4)} | {(4, 10): -1}
    spawns, _ = run(variants, deaths, fronts, [[3, 7]] * 5)
    assert [spawns[(e, 2)] for e in range(5)] == [[17], [20], [22], [27], [17]]
    assert [spawns[(e, 0)] for e in range(4)] == [[23]] * 4
    assert spawns[(4, 0)] == [20]  # team 0 lost the zone at 10: its clock reset
