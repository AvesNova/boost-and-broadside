"""Replays (`bnb.play.replay`): a recorded match plays back as recorded, a seek equals
playing to the same tick, the markers and the kill feed's kills are the match's events,
and the replay directory keeps the newest ones."""

import numpy as np
import pytest
import torch
from play_cases import SMALL
from torch.utils import _pytree

from bnb.play.loop import MatchLoop, MatchSetup
from bnb.play.replay import KEYFRAME_TICKS, Recorder, Replay, ReplayLoop, prune, replays, save

TICKS = 2 * KEYFRAME_TICKS + 37


def _leaves(tree) -> list[torch.Tensor]:
    return [leaf.clone() for leaf in _pytree.tree_leaves(tree)]


def _equal(a, b) -> bool:
    """Same states: integers and flags exactly, floats to float32's tolerance (no test asks
    for bitwise floats)."""
    return all(_same(x, y) for x, y in zip(a, b, strict=True))


def _same(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.dtype.is_floating_point:
        return torch.allclose(x, y, rtol=1e-5, atol=1e-5)
    return torch.equal(x, y)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """A live scripted match (the human flying a ship part of the time), recorded for
    TICKS ticks: its saved replay, the states after every tick, and the raw events."""
    loop = MatchLoop(MatchSetup(SMALL, seed=7), compiled=False)
    recorder = Recorder(loop.setup, loop.realtime, "test")
    states, raw = [_leaves(loop.realtime.state)], []

    def on_tick(actions, realtime, events):
        recorder.record(actions, realtime, events)
        states.append(_leaves(realtime.state))
        died, killer = realtime.events.died[0], realtime.events.killer[0]
        raw.append([(int(ship), int(killer[ship])) for ship in torch.nonzero(died)[:, 0]])

    loop.on_tick = on_tick
    for tick in range(TICKS):
        loop.pilot = 1 if 40 <= tick < 120 else None
        loop.inputs[:] = torch.tensor([True, False, tick % 3 == 0, False, False, False, True])
        loop.step()
    directory = save(recorder, tmp_path_factory.mktemp("replays"))
    return directory, states, raw, recorder


def test_a_replay_resimulates_the_recorded_states(recorded):
    directory, states, _, _ = recorded
    loop = ReplayLoop(Replay.load(directory), compiled=False)
    assert _equal(_leaves(loop.realtime.state), states[0])
    for tick in range(1, TICKS + 1):
        loop.step()
        assert loop.position == tick
        assert _equal(_leaves(loop.realtime.state), states[tick]), tick


@pytest.mark.parametrize(
    "tick", [0, 1, KEYFRAME_TICKS - 1, KEYFRAME_TICKS, KEYFRAME_TICKS + 5, TICKS]
)
def test_a_seek_equals_playing_to_the_same_tick(recorded, tick):
    directory, states, _, _ = recorded
    loop = ReplayLoop(Replay.load(directory), compiled=False)
    loop.seek(TICKS)  # from the end, back
    loop.seek(tick)
    assert loop.position == tick
    assert _equal(_leaves(loop.realtime.state), states[tick])
    assert loop.alpha == 1.0  # drawn at the tick itself


def test_the_markers_are_the_matchs_kills(recorded):
    directory, _, raw, _ = recorded
    replay = Replay.load(directory)
    kills = [
        (event.tick, event.ship, event.other) for event in replay.events if event.kind == "kill"
    ]
    want = [(tick + 1, ship, killer) for tick, deaths in enumerate(raw) for ship, killer in deaths]
    assert kills == want
    assert kills, "the scenario has kills"


def test_playback_counts_and_feeds_the_same_kills(recorded):
    """The kill counts and the kills a playback's ticks report (the feed's source) are the
    recording's."""
    directory, _, _, recorder = recorded
    loop = ReplayLoop(Replay.load(directory), compiled=False)
    fed = []
    loop.on_tick = lambda actions, realtime, events: fed.extend(
        (loop.position + 1, event.ship, event.other) for event in events if event.kind == "kill"
    )
    for _ in range(TICKS):
        loop.step()
    want = [
        (event.tick, event.ship, event.other) for event in recorder.events if event.kind == "kill"
    ]
    assert fed == want
    kills = np.zeros(SMALL.num_ships, np.int64)
    for _, _, killer in want:
        if killer >= 0:
            kills[killer] += 1
    assert np.array_equal(loop.kills, kills)
    loop.seek(TICKS // 2)
    assert loop.kills.sum() == sum(
        1 for tick, _, killer in want if tick <= TICKS // 2 and killer >= 0
    )


def test_playback_pauses_at_the_end(recorded):
    directory, _, _, _ = recorded
    loop = ReplayLoop(Replay.load(directory), compiled=False)
    loop.seek(TICKS - 1)
    loop.step()
    loop.step()
    assert loop.position == TICKS and loop.paused


def test_the_replay_directory_keeps_the_newest(tmp_path, recorded):
    _, _, _, recorder = recorded
    paths = [save(recorder, tmp_path, keep=100) for _ in range(4)]
    assert len({path.name for path in paths}) == 4
    prune(tmp_path, keep=2)
    assert replays(tmp_path) == sorted(paths, key=lambda path: path.name, reverse=True)[:2]
