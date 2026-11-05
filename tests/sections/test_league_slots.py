"""The league's seat in the rollout (league spec 2, 3; acceptance 3 and 4): the draws' shares
and repeatability, and the drain: no opponent changes mid-episode, every environment of a
drained slot reaches the new entry at its next episode end, and at most one slot drains."""

import league_cases as cases
import pytest
import torch

from bnb.contracts import Obs
from bnb.contracts.obs import GLOBAL
from bnb.league.pool import EXPONENTIAL_AVERAGE, UNIFORM_AVERAGE, sampling_weights
from bnb.runtime.opponents import (
    LADDER_KIND,
    OPPONENTS,
    LeagueConfig,
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
    rungs = len(opponents.config.ladder)
    weights = torch.tensor(
        [[8.0, 1.0, 0.0, 1.0] + [0.0] * (rungs - 4), [0.0, 1.0, 1.0, 2.0] + [0.0] * (rungs - 4)]
    )
    opponents.ladder_weights = weights
    opponents.mode_count = 2
    view = _view(trainer)
    carry = trainer.rollout.carry["controllers"][OPPONENTS]
    reset = torch.ones(rows, dtype=torch.bool)
    counts = torch.zeros((2, rungs), dtype=torch.float64)
    for _ in range(600):
        _, carry, _ = opponents.step(carry, view, reset)
        for mode in range(2):
            counts[mode] += torch.bincount(
                carry.ladder_index[opponents.modes == mode], minlength=rungs
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
    with torch.random.fork_rng(devices=[]):
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


def test_each_training_game_is_credited_to_the_entry_that_played_it(tmp_path):
    """Tick by tick, across drains and swaps: each ended game's column names the entry the
    row played as it ended (`_names`), and the league's record, read once per shard, credits
    every game to that entry; each network set holds its entry's weights throughout."""
    torch.manual_seed(0)
    trainer, league = cases.league(tmp_path)
    rollout, slots = trainer.rollout, league.config.slots
    carry = rollout.carry["controllers"][OPPONENTS]
    tally = rollout.carry["tallies"][OPPONENTS]
    columns = trainer.opponents.columns
    expected: dict[str, list[float]] = {}
    seen = torch.zeros_like(tally)
    drains = 0
    for tick in range(600):
        rollout._tick()
        ended = (tally - seen).reshape(columns, len(league.modes), 3)[:, 0]
        seen = tally.clone()
        ladder = [entry.name for entry in league.ladder]
        for column, (wins, ties, losses) in enumerate(ended.tolist()):
            if wins + ties + losses:
                name = league.held[column] if column <= slots else ladder[column - slots - 1]
                assert name in _names(league, carry) or column > slots
                total = expected.setdefault(name, [0.0, 0.0, 0.0])
                for index, count in enumerate((wins, losses, ties)):
                    total[index] += count
        if tick % 8 == 7:
            draining = trainer.opponents.draining_slot
            league.after_shard(trainer, 0)
            seen.zero_()
            drains += draining is not None and trainer.opponents.draining_slot != draining
            for set_index, name in enumerate(league.held):
                entry = league.entries.get(name) if name is not None else None
                if entry is None or not entry.network:
                    continue
                weights = league._weights_of(entry)
                key = next(key for key, value in weights.items() if value.is_floating_point())
                held = trainer.opponents.parameters[f"model.{key}"][set_index]
                assert torch.equal(held, weights[key].to(held.device)), (set_index, name)
    recorded = {
        name: [record.wins, record.losses, record.ties]
        for name, record in league.record[league.modes[0]].items()
    }
    assert drains >= 3 and recorded == expected and sum(map(sum, expected.values())) > 20


def test_a_slot_holding_an_average_follows_it_after_every_update(tmp_path):
    """The averages move: the slots holding one (and the incoming set) take its weights after
    each update, as the lanes' averages do, so its training games are the average's."""
    trainer, league = cases.league(tmp_path)
    for name, set_index in ((EXPONENTIAL_AVERAGE, 0), (UNIFORM_AVERAGE, 1)):
        league.held[set_index] = name
        league._load(set_index, league.entries[name])
    for update in range(2):
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.add_(1.0)
        trainer.collect()
        league.collected(trainer)
        trainer.ppo.environment_steps += trainer.steps_per_update
        league.updated(trainer)
        for name, set_index in ((EXPONENTIAL_AVERAGE, 0), (UNIFORM_AVERAGE, 1)):
            if league.held[set_index] != name:
                continue  # drained onto another entry meanwhile
            weights = league.averages.weights[name]
            key = next(key for key, value in weights.items() if value.is_floating_point())
            held = trainer.opponents.parameters[f"model.{key}"][set_index]
            assert torch.equal(held, weights[key]), (update, name)


def test_the_leagues_games_join_the_runs_fit(tmp_path):
    """Each update's training games enter the run's ratings as the live policy's against
    each entry, and a snapshot taken at the update is credited with them; the uniform
    average's are left out before it has started (it is the live weights until then)."""
    torch.manual_seed(0)
    trainer, league = cases.league(tmp_path, league=LeagueConfig(snapshot_every=1))
    mode, total = league.modes[0], 0.0
    for update in range(3):
        trainer.collect()
        league.collected(trainer)
        played = {
            name: [record.wins, record.ties, record.losses]
            for name, record in league.record[mode].items()
            if name != UNIFORM_AVERAGE
        }
        total += sum(map(sum, played.values()))
        trainer.ppo.environment_steps += trainer.steps_per_update
        league.updated(trainer)
        games = league.ratings.recent[-1]
        snapshot = f"snapshot_{trainer.ppo.environment_steps}"
        for name, outcomes in played.items():
            assert games[f"{mode}|live|{name}"] == outcomes, (update, name)
            stationary = name in {entry.name for entry in league.ladder} or name.startswith(
                "snapshot_"
            )
            credited = league.ratings.whole if stationary else games
            if name != snapshot:
                assert credited[f"{mode}|{snapshot}|{name}"][0] >= outcomes[0]
        assert not any(UNIFORM_AVERAGE in key for key in games)
    assert total > 0 and not league.averages.uniform_started


def test_the_opponent_rows_split_evenly_over_the_slots():
    with pytest.raises(ValueError, match="split evenly"):
        cases.Trainer(cases.config(environments=12), torch.device("cpu"))


def test_the_tally_counts_each_result_from_the_learners_side(tmp_path):
    trainer, _ = cases.league(tmp_path)
    opponents = trainer.opponents
    carry = trainer.rollout.carry["controllers"][OPPONENTS]
    rows = opponents.modes.shape[0]
    own = torch.tensor([1, 0, -1, 1, 1, 0, -1, -1], dtype=torch.int32)  # the opponents' results
    done = torch.tensor([True] * 7 + [False])
    counts = opponents.tally(carry, done, own).reshape(opponents.columns, 1, 3)
    per_column = counts[:, 0]  # (columns, wins/ties/losses of the learner)
    assert per_column.sum(0).tolist() == [2, 2, 3]  # three opponent wins are learner losses
    assert per_column[carry.column[0]].tolist()[2] >= 1 and rows == 8


# ---- the averages


def test_the_uniform_average_starts_after_its_warm_up(tmp_path):
    """Before `uniform_from` the uniform average is the live weights; after it, the mean of
    the updates since; the exponential average folds in every update."""
    steps = cases.config().environments * cases.config().length
    trainer, league = cases.league(
        tmp_path, league=LeagueConfig(snapshot_every=16 * 16, uniform_from=2 * steps)
    )
    seen = []
    for update in range(4):
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.fill_(float(update))
        trainer.collect()
        league.collected(trainer)
        trainer.ppo.environment_steps += trainer.steps_per_update
        league.updated(trainer)
        name, value = next(iter(league.averages.weights[UNIFORM_AVERAGE].items()))
        assert trainer.model.state_dict()[name].is_floating_point()
        seen.append(float(value.flatten()[0]))
    # update 0 ends below 2 x steps: the live weights; from update 1 on, the mean since
    assert seen == [0.0, 1.0, 1.5, 2.0]
