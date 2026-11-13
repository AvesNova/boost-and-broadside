"""The scripted agent's built step (`bnb.runtime.scripted_step`, played by
`bnb.play.sides.ScriptedAgent`): it plays as the uncompiled step does, draws included, is
kept on disk by the shapes it was built for, and the agent steps uncompiled until it is
in."""

import dataclasses
import threading

import pytest
import torch
from play_cases import SMALL

from bnb import cache
from bnb.contracts import GameConfig
from bnb.contracts.obs import GLOBAL
from bnb.controllers import scripted_doctrine
from bnb.controllers.scripted_defaults import MODE_DEFAULTS
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.play.loop import MatchLoop, MatchSetup, SideSetup, team_view
from bnb.play.sides import DOCTRINES, ScriptedAgent
from bnb.runtime import built
from bnb.runtime.scripted_step import ScriptedStep, library_path

SCRIPTED_AGENTS = (SideSetup("scripted_agent"), SideSetup("scripted_agent"))
AGENTS = MatchSetup(SMALL, seed=3, sides=SCRIPTED_AGENTS)
FLOAT_MEMORY = ("enemy_x", "enemy_y", "enemy_velocity_x", "enemy_velocity_y")
# A small real-scale match of the Yongma and the Jujak, the types with stored doctrines: the
# scripted agent flies the play shell's default doctrine (r9m2) there.
REAL_SCALE = dataclasses.replace(
    api.PRESETS["real_scale_deathmatch"], ships_per_team=2, num_fields=4, arena_radius=2500.0
)
REAL_SCALE_AGENTS = MatchSetup(REAL_SCALE, lineup=(0, 1), seed=3, sides=SCRIPTED_AGENTS)


def _agents(setup: MatchSetup = AGENTS, **settings) -> list[ScriptedAgent]:
    """Both sides' scripted agents, seeded as a match of `setup` seeds them."""
    loop = MatchLoop(dataclasses.replace(setup, sides=(SideSetup("scripted"),) * 2), compiled=False)
    return [ScriptedAgent(loop.rules, loop.types, setup.seed + team, **settings) for team in (0, 1)]


def _beside_uncompiled(sides: list, setup: MatchSetup = AGENTS) -> tuple[MatchLoop, MatchLoop]:
    """A match played by `sides` and, beside it, one by uncompiled scripted agents of the
    same seeds."""
    return MatchLoop(setup, sides=sides, compiled=False), MatchLoop(setup, compiled=False)


def _play(loop: MatchLoop, twin: MatchLoop, ticks: int) -> int:
    """Play both matches on: every tick's actions agree exactly (the draws with them), the
    memories' floats to fusion rounding, their flags exactly. Returns the shots fired."""
    shots = 0
    for _ in range(ticks):
        actions = loop.actions()
        assert torch.equal(actions, twin.actions())
        for mine, theirs in zip(loop.sides, twin.sides, strict=True):
            for name, value in vars(mine.carry).items():
                if name in FLOAT_MEMORY:
                    torch.testing.assert_close(value, getattr(theirs.carry, name))
                else:
                    assert torch.equal(value, getattr(theirs.carry, name)), name
        shots += int(actions[:, 2].sum())
        loop.tick(actions)
        twin.tick(actions)
    return shots


def _arguments(game: GameConfig, step: ScriptedStep, traits=None) -> tuple:
    """A scripted agent's step arguments on team 0's first view of a match of `game`."""
    rules, types = api.rules_of(game), api.ship_types()
    loop = MatchLoop(MatchSetup(game, sides=(SideSetup("scripted"),) * 2), compiled=False)
    observation = loop.realtime.obs
    reset = observation.globals[:, 0, GLOBAL["reset"]] > 0.5
    carry = ScriptedAgent(rules, types).carry
    traits = characteristics(types, rules) if traits is None else traits
    return carry, team_view(observation, 0), reset, traits


@pytest.mark.slow
@pytest.mark.parametrize(
    "setup, ticks", [(AGENTS, 200), (REAL_SCALE_AGENTS, 900)], ids=["small", "real-scale r9m2"]
)
def test_the_built_step_plays_as_the_uncompiled_one_and_is_kept_on_disk(
    setup, ticks, tmp_path, monkeypatch
):
    """Built in a child process (once for both sides, then read from disk by a later
    process), the step gives the uncompiled step's actions exactly, its draws included, over
    a match with fights; every step after the first runs built. On the real-scale card the
    Yongma and the Jujak fly r9m2's doctrines (the table is the built step's constant)."""
    sides = _agents(setup, compiled=True)  # the card measured before the cache moves
    monkeypatch.setattr(cache, "ROOT", tmp_path)
    monkeypatch.setattr(built, "_LOADED", {})
    if setup is REAL_SCALE_AGENTS:
        assert sides[0].controller.doctrines is not None
    assert _play(*_beside_uncompiled(sides, setup), ticks) > 0
    assert all(side.uses_compiled for side in sides)
    libraries = list(tmp_path.glob("scripted_steps/*.pt2"))
    assert len(libraries) == 1

    monkeypatch.setattr(built, "_LOADED", {})
    stamp = libraries[0].stat().st_mtime_ns
    again = _agents(setup, compiled=True)
    _play(*_beside_uncompiled(again, setup), 3)
    assert all(side.uses_compiled for side in again)
    assert list(tmp_path.glob("scripted_steps/*.pt2")) == libraries
    assert libraries[0].stat().st_mtime_ns == stamp


def test_the_step_is_keyed_by_its_shapes_and_parameters_not_the_cards_values():
    """A tuned card (other values, the same shapes) plays on the same built step; another
    fleet size or parameter set has its own."""
    step = ScriptedStep(SMALL.ships_per_team)
    arguments = _arguments(SMALL, step)
    traits = arguments[-1]
    tuned = type(traits)(**{name: value * 1.1 for name, value in vars(traits).items()})
    assert library_path(step, arguments) == library_path(step, arguments[:-1] + (tuned,))

    bigger = GameConfig(4, 4, 900.0, episode_seconds=20.0)
    larger = ScriptedStep(bigger.ships_per_team)
    assert library_path(larger, _arguments(bigger, larger)) != library_path(step, arguments)
    deathmatch = ScriptedStep(SMALL.ships_per_team, MODE_DEFAULTS.sets["deathmatch"])
    assert library_path(deathmatch, arguments) != library_path(step, arguments)


def test_the_step_is_keyed_by_its_doctrines_and_their_width(monkeypatch):
    """r9m2 and r9 have their own built steps, and a wider doctrine table another (so a
    grown table rebuilds instead of loading a library built for the old one)."""
    arguments = _arguments(SMALL, ScriptedStep(SMALL.ships_per_team))
    r9m2 = ScriptedStep(SMALL.ships_per_team, doctrines=DOCTRINES["r9m2"])
    r9 = ScriptedStep(SMALL.ships_per_team, doctrines=DOCTRINES["r9"])
    assert library_path(r9m2, arguments) != library_path(r9, arguments)
    before = library_path(r9m2, arguments)
    monkeypatch.setattr(scripted_doctrine, "WIDTH", scripted_doctrine.WIDTH + 1)
    assert library_path(r9m2, arguments) != before


def test_the_agent_steps_uncompiled_until_its_built_step_is_in(monkeypatch):
    """With `background`, the agent plays its uncompiled step while its build runs (here a
    build held open, whose stand-in library is the uncompiled step itself), then the built
    one, and the window's thread never waits for it; its actions are the uncompiled agent's
    throughout."""
    release, started = threading.Event(), threading.Event()

    def held_library(path, function, arguments, outputs, name):
        started.set()
        release.wait()
        built._LOADED[path] = function
        return function

    monkeypatch.setattr(built, "_LOADED", {})
    monkeypatch.setattr(built, "library", held_library)
    sides = _agents(compiled=True, background=True)
    loop, twin = _beside_uncompiled(sides)
    _play(loop, twin, 10)
    assert started.wait(5.0)
    assert all(side.compiling and not side.uses_compiled for side in sides)
    release.set()
    for side in sides:
        side.wait_compiled()
    _play(loop, twin, 10)
    assert all(side.uses_compiled for side in sides)
