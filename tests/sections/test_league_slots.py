"""The league's seat in the rollout (league spec 2, 3; acceptance 3 and 4): the draws' shares
and repeatability, the ladder in or out of the pool, and the clean cut: it swaps the right
slot at the right update and nothing else, leaves the games it cuts uncredited, and resets
or keeps the opponent's memory."""

import pathlib
import sys

import league_cases as cases
import pytest
import torch
from torch.utils import _pytree

from bnb.contracts import Obs
from bnb.contracts.obs import GLOBAL
from bnb.league.league import same_build
from bnb.league.pool import EXPONENTIAL_AVERAGE, UNIFORM_AVERAGE, Entry, sampling_weights
from bnb.runtime.opponents import (
    LADDER_KIND,
    NETWORK,
    OPPONENTS,
    LeagueConfig,
    Rung,
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
    trainer, _ = cases.league(
        tmp_path, league=LeagueConfig(snapshot_every=256, ladder_in_pool=True)
    )
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


# ---- the ladder in or out of the pool


def test_the_ladder_is_out_of_the_training_pool_by_default(tmp_path, monkeypatch):
    """Without `ladder_in_pool` the pool holds no rung, no slot holds the ladder and the seat
    runs no scripted pass; with it, every rung is an entry."""
    trainer, league = cases.league(tmp_path)
    assert not any(entry.kind == "ladder" for entry in league.entries.values())
    assert all(gauge.names() == list(league.entries) for gauge in league.gauges.values())
    assert (trainer.opponents.kind == NETWORK).all()

    def scripted_pass(*arguments):
        raise AssertionError("the seat ran the ladder's scripted pass")

    monkeypatch.setattr(trainer.opponents.ladder_agent, "step", scripted_pass)
    for _ in range(4):
        trainer.collect()
        league.collected(trainer)
        trainer.ppo.environment_steps += trainer.steps_per_update
        league.updated(trainer)
    carry = trainer.rollout.carry["controllers"][OPPONENTS]
    assert (carry.column < league.config.slots).all()
    _, inside = cases.league(
        tmp_path / "inside", league=LeagueConfig(snapshot_every=256, ladder_in_pool=True)
    )
    ladder = {entry.name for entry in inside.entries.values() if entry.kind == "ladder"}
    assert len(ladder) == len(inside.config.ladder) and "scripted" in ladder


# ---- the clean cut


def _names(league, carry) -> list[str]:
    slots = league.config.slots
    ladder = [entry.name for entry in league.ladder]
    return [
        league.held[column] if column < slots else ladder[column - slots]
        for column in carry.column.tolist()
    ]


def _first_weight(weights: dict[str, torch.Tensor]) -> str:
    return next(key for key, value in weights.items() if value.is_floating_point())


def _cycling_draws(league, monkeypatch) -> None:
    """Each draw returns an entry other than the one the slot being cut holds, so every cut swaps
    (the draws' shares are tested above)."""

    def draw():
        league.swaps += 1
        held = league.held[(league.next_slot - 1) % league.config.slots]  # the slot cut now
        names = [name for name in league.entries if name != held]
        return league.entries[names[league.swaps % len(names)]]

    monkeypatch.setattr(league, "_draw", draw)


def test_a_cut_swaps_the_next_slot_every_cut_every_updates_and_nothing_else(tmp_path, monkeypatch):
    """With cut_every 2 and cut_slots 1, updates 2, 4, 6 and 8 each cut one slot, round robin;
    the other slots keep their entries and weights, and only the cut slot's rows lose the
    tally of their game in progress."""
    trainer, league = cases.league(
        tmp_path, league=LeagueConfig(snapshot_every=10**9, cut_every=2, cut_slots=1)
    )
    _cycling_draws(league, monkeypatch)
    opponents = trainer.opponents
    carry = trainer.rollout.carry["controllers"][OPPONENTS]
    key = _first_weight(trainer.model.state_dict())
    cut_slots = []
    for update in range(1, 9):
        trainer.collect()
        league.collected(trainer)
        trainer.ppo.environment_steps += trainer.steps_per_update
        held = list(league.held)
        whole = carry.whole.clone()
        stacked = {slot: opponents.parameters[f"model.{key}"][slot].clone() for slot in range(4)}
        row = league.updated(trainer)
        changed = [slot for slot in range(4) if league.held[slot] != held[slot]]
        assert row["pool/cut/cuts"] == len(changed)
        if update % 2:
            assert changed == [] and torch.equal(carry.whole, whole)
            continue
        assert len(changed) == 1
        slot = changed[0]
        cut_slots.append(slot)
        rows = opponents.slot_of_row == slot
        assert not carry.whole[rows].any()  # the games in progress are not credited
        assert torch.equal(carry.whole[~rows], whole[~rows])
        assert row["pool/cut/uncredited_games"] == int(whole[rows].sum())
        entry = league.entries[league.held[slot]]
        assert torch.equal(
            opponents.parameters[f"model.{key}"][slot], league._weights_of(entry)[key]
        )
        for other in set(range(4)) - {slot}:
            if league.entries[held[other]].kind != "average":  # an average follows its weights
                assert torch.equal(opponents.parameters[f"model.{key}"][other], stacked[other])
    assert cut_slots == [0, 1, 2, 3]


def test_a_cut_game_is_credited_to_no_one(tmp_path):
    """Tick by tick across cuts: a cut slot's column tallies nothing until one of its rows has
    begun a game after the cut (the games in progress are credited to no one), and the
    league's record, read once per shard, credits every tallied game to the entry its slot
    held."""
    torch.manual_seed(0)
    trainer, league = cases.league(
        tmp_path, league=LeagueConfig(snapshot_every=10**9, cut_every=1, cut_slots=1)
    )
    rollout, slots = trainer.rollout, league.config.slots
    opponents = trainer.opponents
    tally = rollout.carry["tallies"][OPPONENTS]
    columns = opponents.columns
    expected: dict[str, list[float]] = {}
    seen = torch.zeros_like(tally)
    begun = [True] * slots  # a row of the slot has begun a game since the slot's last cut
    uncredited = 0
    for tick in range(600):
        reset = _view(trainer).globals[..., GLOBAL["reset"]] > 0.5
        for slot in range(slots):
            begun[slot] |= bool(reset[opponents.slot_of_row == slot].any())
        rollout._tick()
        ended = (tally - seen).reshape(columns, len(league.modes), 3)[:, 0]
        seen = tally.clone()
        for column, (wins, ties, losses) in enumerate(ended.tolist()):
            if wins + ties + losses:
                assert begun[column], (tick, column)
                total = expected.setdefault(league.held[column], [0.0, 0.0, 0.0])
                for index, count in enumerate((wins, losses, ties)):
                    total[index] += count
        if tick % 8 == 7:
            league.after_shard(trainer, 0)
            seen.zero_()
        if tick % 32 == 31:  # a cut of one slot every 4 shards: some games whole, some cut
            before = list(league.held)
            league._cut(trainer)
            uncredited += league.last_cut["uncredited_games"]
            for slot in range(slots):
                if league.held[slot] != before[slot]:
                    begun[slot] = False
    recorded = {
        name: [record.wins, record.losses, record.ties]
        for name, record in league.record[league.modes[0]].items()
    }
    assert recorded == expected and sum(map(sum, expected.values())) > 20
    assert uncredited > 0


def _yemong_league(directory, carry_at_cut: str):
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "runs"))
    from runs_cases import tiny_yemong_config  # noqa: PLC0415

    config = cases.config(
        model="yemong",
        yemong=tiny_yemong_config(),
        league=LeagueConfig(snapshot_every=10**9, cut_every=1, carry_at_cut=carry_at_cut),
    )
    trainer = cases.Trainer(config, torch.device("cpu"))
    return trainer, cases.League(trainer, directory)


@pytest.mark.parametrize("carry_at_cut", ["reset", "keep"])
def test_a_cut_resets_the_opponents_memory_or_keeps_it(tmp_path, monkeypatch, carry_at_cut):
    """ "reset" zeroes the trunk's carry (state and convolution buffer) on the cut slot's rows
    and keeps the belief, the previous actions and the offset; "keep" changes nothing; the
    other slots' rows are untouched either way."""
    torch.manual_seed(0)
    trainer, league = _yemong_league(tmp_path, carry_at_cut)
    _cycling_draws(league, monkeypatch)
    trainer.collect()
    carry = trainer.rollout.carry["controllers"][OPPONENTS]
    before = _pytree.tree_map_only(torch.Tensor, torch.clone, carry.model)
    assert before.state.abs().sum() > 0
    held = list(league.held)
    league.collected(trainer)
    trainer.ppo.environment_steps += trainer.steps_per_update
    row = league.updated(trainer)
    slot = next(slot for slot in range(4) if league.held[slot] != held[slot])
    rows = trainer.opponents.slot_of_row == slot
    after = carry.model
    for name in ("state", "convolution_buffer"):
        new, old = getattr(after, name), getattr(before, name)
        assert torch.equal(new[~rows], old[~rows])
        if carry_at_cut == "reset":
            assert not new[rows].any()
        else:
            assert torch.equal(new[rows], old[rows])
    for new, old in zip(_pytree.tree_leaves(after.belief), _pytree.tree_leaves(before.belief)):
        assert torch.equal(new, old)
    assert torch.equal(after.previous_action, before.previous_action)
    assert torch.equal(after.offset, before.offset)
    reset_rows = 0 if carry_at_cut == "keep" else int(rows.sum())
    assert row["pool/cut/carries_reset"] == reset_rows


def test_a_cut_keeps_the_memory_only_between_networks():
    snapshot = Entry("snapshot_1", "checkpoint", steps=1, ordinal=0)
    average = Entry(EXPONENTIAL_AVERAGE, "average")
    rung = Entry("scripted", "ladder", rung=Rung(1.0))
    assert same_build(snapshot, average) and same_build(average, snapshot)
    assert not same_build(rung, snapshot) and not same_build(snapshot, rung)
    assert not same_build(None, snapshot)


def test_a_slot_holding_an_average_follows_it_after_every_update(tmp_path):
    """The averages move: the slots holding one take its weights after
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
                continue  # cut to another entry meanwhile
            weights = league.averages.weights[name]
            key = next(key for key, value in weights.items() if value.is_floating_point())
            held = trainer.opponents.parameters[f"model.{key}"][set_index]
            assert torch.equal(held, weights[key]), (update, name)


def test_the_leagues_games_join_the_runs_fit(tmp_path):
    """Each update's training games enter the run's ratings as the live policy's against
    each entry, and a snapshot taken at the update is credited with them; the uniform
    average's are left out before it has started (it is the live weights until then)."""
    torch.manual_seed(0)
    trainer, league = cases.league(
        tmp_path, league=LeagueConfig(snapshot_every=1, uniform_from=10**9)
    )
    mode, total = league.modes[0], 0.0
    for update in range(8):
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
    carry.whole.fill_(True)  # past each row's first game
    counts = opponents.tally(carry, done, own).reshape(opponents.columns, 1, 3)
    per_column = counts[:, 0]  # (columns, wins/ties/losses of the learner)
    assert per_column.sum(0).tolist() == [2, 2, 3]  # three opponent wins are learner losses
    assert per_column[carry.column[0]].tolist()[2] >= 1 and rows == 8


def test_a_rows_first_game_is_never_tallied(tmp_path):
    """Each row's first game is cut short by the staggered clocks (and leans to ties), so the
    league leaves it out as the lanes do: a row's games count from its first reset after its
    first tick."""
    trainer, _ = cases.league(tmp_path)
    opponents = trainer.opponents
    view = _view(trainer)
    rows = opponents.modes.shape[0]
    carry = opponents.init_carry(rows, torch.device("cpu"))
    every, none = torch.ones(rows, dtype=torch.bool), torch.zeros(rows, dtype=torch.bool)
    result = torch.zeros(rows, dtype=torch.int32)
    for reset in (every, none):  # the first game: begun at the first tick, then ended
        _, carry, _ = opponents.step(carry, view, reset)
        assert int(opponents.tally(carry, every, result).sum()) == 0
    _, carry, _ = opponents.step(carry, view, every)  # the second game begins
    assert int(opponents.tally(carry, every, result).sum()) == rows


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


def test_the_measured_snapshot_cadence_reads_the_slowest_of_the_second_to_fourth_updates(
    tmp_path, monkeypatch
):
    """One short update (a PPO update that stops early on its KL) must not set the cadence."""
    trainer, league = cases.league(tmp_path, league=LeagueConfig(snapshot_every=None))
    clock = iter([0.0, 100.0, 1100.0, 1110.0, 1111.0])  # updates of 100, 1000, 10 and 1 seconds
    monkeypatch.setattr("bnb.league.league.time.perf_counter", lambda: next(clock))
    league._measure_cadence(trainer)  # the first update compiles: only starts the clock
    for _ in range(2):
        league._measure_cadence(trainer)
        assert league.snapshot_every is None
    league._measure_cadence(trainer)  # durations 100, 1000, 10: the slowest sets it
    assert league.snapshot_every == round(30 * 60 / 1000) * trainer.steps_per_update
