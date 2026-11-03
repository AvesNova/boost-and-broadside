"""The league's seat in the rollout (league spec 2, 3; acceptance 3 and 4): the draws' shares
and repeatability, and the drain: no opponent changes mid-episode, every environment of a
drained slot reaches the new entry at its next episode end, and at most one slot drains."""

import league_cases as cases
import pytest
import torch

from bnb.contracts import Obs
from bnb.contracts.obs import GLOBAL
from bnb.league.pool import sampling_weights
from bnb.runtime.opponents import (
    LADDER_KIND,
    OPPONENTS,
    counter_uniform,
    draw,
    stream_key,
)

pytestmark = pytest.mark.usefixtures("global_generator")

DRAWS = 100_000
SIGMAS = 5.0  # a share's bound in binomial standard deviations


def _assert_shares(counts: torch.Tensor, weights: torch.Tensor) -> None:
    total = float(counts.sum())
    expected = weights / weights.sum()
    bound = SIGMAS * torch.sqrt(total * expected * (1 - expected)) + 1.0
    assert ((counts - total * expected).abs() <= bound).all(), (counts / total, expected)


def test_counter_draws_match_p_times_one_minus_p_plus_the_floor():
    probability = torch.tensor([0.5, 0.7, 0.95, 0.99, 0.2, 0.05], dtype=torch.float64)
    weights = sampling_weights(probability, 0.05)
    uniform = counter_uniform(
        stream_key(0, "test"), torch.arange(DRAWS), torch.zeros(DRAWS, dtype=torch.int64)
    )
    drawn = draw(weights, uniform.double())
    _assert_shares(torch.bincount(drawn, minlength=6).double(), weights)


def test_counter_uniforms_are_uniform_and_depend_on_every_argument():
    key = stream_key(0, "test")
    rows = torch.arange(DRAWS)
    uniform = counter_uniform(key, torch.tensor(7), rows)
    assert 0.0 <= float(uniform.min()) and float(uniform.max()) < 1.0
    histogram = torch.histc(uniform, bins=10, min=0.0, max=1.0)
    _assert_shares(histogram.double(), torch.full((10,), 0.1, dtype=torch.float64))
    assert not torch.equal(uniform, counter_uniform(key, torch.tensor(8), rows))
    assert not torch.equal(uniform, counter_uniform(stream_key(1, "test"), torch.tensor(7), rows))
    assert torch.equal(uniform, counter_uniform(key, torch.tensor(7), rows))


def test_a_slot_draws_entries_by_the_mixed_weights(tmp_path):
    _, league = cases.league(tmp_path)
    names = list(league.entries)
    counts = torch.zeros(len(names), dtype=torch.float64)
    for _ in range(5000):
        counts[names.index(league._draw().name)] += 1
    _assert_shares(counts, league.mixed)


def _view(trainer) -> Obs:
    rollout = trainer.rollout
    return rollout._view(rollout.carry, OPPONENTS, trainer.opponents.reads + ("globals",))


def test_ladder_rows_draw_their_rung_from_their_own_modes_weights(tmp_path):
    trainer, _ = cases.league(tmp_path)
    opponents = trainer.opponents
    opponents.kind.fill_(LADDER_KIND)  # every slot holds the ladder
    rows = opponents.modes.shape[0]
    opponents.modes.copy_(torch.arange(rows) % 2)  # two modes over the rows
    weights = torch.tensor([[8.0, 1.0, 0.0, 1.0] + [0.0] * 8, [0.0, 1.0, 1.0, 2.0] + [0.0] * 8])
    opponents.ladder_weights = weights
    opponents.mode_count = 2
    view = _view(trainer)
    carry = trainer.rollout.carry["controllers"][OPPONENTS]
    reset = torch.ones(rows, dtype=torch.bool)
    counts = torch.zeros((2, 12), dtype=torch.float64)
    for _ in range(600):
        _, carry, _ = opponents.step(carry, view, reset)
        for mode in range(2):
            counts[mode] += torch.bincount(
                carry.ladder_index[opponents.modes == mode], minlength=12
            )
    for mode in range(2):
        _assert_shares(counts[mode], weights[mode].double())


def _run_league(directory, seed: int, updates: int = 3) -> dict:
    """The league's side of `updates` updates: collect, the hooks and the step count,
    without learning (the opponents' draws do not depend on it)."""
    torch.manual_seed(seed)
    trainer, league = cases.league(directory, seed=seed)
    rows = []
    for _ in range(updates):
        trainer.collect()
        league.collected(trainer)
        trainer.ppo.environment_steps += trainer.steps_per_update
        rows.append(league.updated(trainer))
    carry = trainer.rollout.carry["controllers"][OPPONENTS]
    return dict(rows=rows, state=league.state()[0], ladder=carry.ladder_index.tolist())


def test_the_same_seed_draws_the_same_opponents(tmp_path):
    first = _run_league(tmp_path / "first", 0)
    second = _run_league(tmp_path / "second", 0)
    other = _run_league(tmp_path / "other", 1)
    assert first == second
    assert first["state"]["held"] != other["state"]["held"] or first["ladder"] != other["ladder"]


# ---- the drain


def _names(league, carry) -> list[str]:
    slots = league.config.slots
    ladder = [entry.name for entry in league.ladder]
    return [
        league.held[column] if column <= slots else ladder[column - slots - 1]
        for column in carry.column.tolist()
    ]


@pytest.fixture(scope="module")
def drain_history(tmp_path_factory):
    """Tick by tick over 600 ticks (10 episodes), the league's hooks every 8 ticks: per tick
    each row's entry, reset flag and incoming flag, the draining slot and the incoming
    entry."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        trainer, league = cases.league(tmp_path_factory.mktemp("drain"))
        rollout, slots = trainer.rollout, league.config.slots
        carry = rollout.carry["controllers"][OPPONENTS]
        history = []
        for tick in range(600):
            reset = _view(trainer).globals[..., GLOBAL["reset"]] > 0.5
            rollout._tick()
            history.append(
                dict(
                    names=_names(league, carry),
                    reset=reset.tolist(),
                    incoming=carry.incoming.tolist(),
                    draining=trainer.opponents.draining_slot,
                    entering=league.held[slots],
                )
            )
            if tick % 8 == 7:
                league.after_shard(trainer, 0)
        return history, league.swaps


def test_no_opponent_changes_mid_episode(drain_history):
    history, _ = drain_history
    for before, after in zip(history, history[1:]):
        for row, (old, new) in enumerate(zip(before["names"], after["names"])):
            assert old == new or after["reset"][row], (row, old, new)


def test_every_environment_of_a_drained_slot_switches_at_its_next_episode_end(drain_history):
    history, swaps = drain_history
    slots = 4
    drains = 0
    for start in range(1, len(history)):
        slot = history[start]["draining"]
        if slot is None or history[start - 1]["draining"] == slot:
            continue
        drains += 1
        for row in range(slot, len(history[start]["names"]), slots):
            first_reset = next(
                (tick for tick in range(start, len(history)) if history[tick]["reset"][row]),
                None,
            )
            if first_reset is None:
                continue  # the run ended first
            assert history[first_reset]["incoming"][row] or (
                history[first_reset]["draining"] != slot  # already finished and renamed
            ), (slot, row, start, first_reset)
    assert drains >= 3 and swaps > drains


def test_at_most_one_slot_drains(drain_history):
    history, _ = drain_history
    for tick in history:
        switched = {row % 4 for row, flag in enumerate(tick["incoming"]) if flag}
        assert len(switched) <= 1
        assert not switched or switched == {tick["draining"]}


def test_a_drain_ends_once_every_row_has_switched(drain_history):
    history, _ = drain_history
    for before, after in zip(history, history[1:]):
        if before["draining"] is not None and after["draining"] != before["draining"]:
            rows = range(before["draining"], len(before["names"]), 4)
            assert all(before["incoming"][row] for row in rows)


def test_the_opponent_rows_split_evenly_over_the_slots():
    with pytest.raises(ValueError, match="split evenly"):
        cases.Trainer(cases.config(environments=12), torch.device("cpu"))
