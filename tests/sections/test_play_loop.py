"""The match loop (`bnb.play.loop`): the same states as the realtime sim alone, and the
fixed timestep, pause, single step and time scale."""

import pytest
import torch
from torch.utils import _pytree

from bnb.contracts import GameConfig
from bnb.contracts.obs import GLOBAL
from bnb.game.modes.rules import PRESETS
from bnb.play.loop import TIME_SCALES, MatchLoop, MatchSetup, SideSetup, team_view
from bnb.runtime.realtime import Realtime

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """One CPU thread: B = 1 ticks gain nothing from more, and lose much on a busy machine."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _leaves(state):
    return [leaf.clone() for leaf in _pytree.tree_leaves(state)]


def test_the_loop_gives_the_same_states_as_realtime_alone():
    """Drawing (snapshots, interpolation) between ticks draws no random numbers, so the
    sim matches the realtime sim given the same seed and actions."""
    config = GameConfig(3, 6, 1800.0, mode="team_deathmatch")
    loop = MatchLoop(MatchSetup(config, seed=11), compiled=False, pilot=0)
    loop.inputs[:] = torch.tensor([True, False, True, False, True, False, True])
    actions, states = [], []
    original = loop.tick

    def recording_tick(chosen):
        actions.append(chosen.clone())
        original(chosen)
        states.append(_leaves(loop.realtime.state))

    loop.tick = recording_tick
    for _ in range(40):
        loop.advance(1.0 / 60.0)
        loop.frame()
    assert len(actions) == 20

    alone = Realtime(config, compiled=False, generator=torch.Generator().manual_seed(11))
    for chosen, want in zip(actions, states):
        alone.step(chosen)
        for got, expected in zip(_leaves(alone.state), want):
            if got.dtype.is_floating_point:
                torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
            else:
                assert torch.equal(got, expected)


def test_fixed_timestep_pause_step_and_time_scale():
    loop = MatchLoop(MatchSetup(PRESETS["deathmatch"]), compiled=False)
    tick = loop.tick_length
    assert loop.advance(0.5 * tick) == 0 and 0.49 < loop.alpha < 0.51
    assert loop.advance(0.5 * tick) == 1 and loop.alpha < 0.01
    loop.paused = True
    assert loop.advance(10 * tick) == 0
    loop.single_step()
    assert loop.ticks == 2 and loop.alpha == 1.0
    loop.paused = False
    loop.accumulator = 0.0
    assert loop.time_scale == 1.0
    loop.faster()
    assert loop.time_scale == TIME_SCALES[TIME_SCALES.index(1.0) + 1]
    assert loop.advance(tick) == 2  # x2: two ticks per tick of display time
    for _ in TIME_SCALES:
        loop.faster()
    assert loop.time_scale == TIME_SCALES[-1]
    for _ in TIME_SCALES:
        loop.slower()
    assert loop.time_scale == TIME_SCALES[0] == 0.25
    loop.accumulator = 0.0
    assert loop.advance(4 * tick) == 1  # x1/4: a tick per four ticks of display time
    loop.time_scale = 1.0
    # A long stall is not made up all at once.
    assert loop.advance(10.0) <= 5


def test_the_pilot_flies_and_hands_over():
    chasers = (SideSetup("scripted"), SideSetup("scripted"))  # the released ship plays its side
    loop = MatchLoop(MatchSetup(PRESETS["deathmatch"], sides=chasers), compiled=False, pilot=0)
    loop.inputs[:] = torch.tensor([True, False, False, True, True, False, True])
    actions = loop.actions()
    assert actions[0].tolist() == [1, 5, 1]  # boost, full right, shoot
    loop.next_pilot()
    assert loop.pilot == 1
    assert loop.actions()[1].tolist() == [1, 5, 1]
    chaser = loop.sides[0].chaser.step(None, *_first_view(loop))[0].packed()[0]
    assert torch.equal(loop.actions()[0], chaser[0])  # the released ship plays the chaser


def _first_view(loop):
    observation = loop.realtime.obs
    return team_view(observation, 0), observation.globals[:, 0, GLOBAL["reset"]] > 0.5
