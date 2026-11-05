"""The diagnostics (diagnostics spec 6): the collector, the series against hand-computed
cases, the method rules (names, no smoothing, no effect on the update), the sinks and the
recorder."""

import copy
import json
import math
import re
import sys

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.contracts import REWARD_LEVELS, Action
from bnb.contracts.action import NUM_JOINT
from bnb.contracts.series import Deferred, Mean, Table
from bnb.league.league import entry_kind, normalised_entropy
from bnb.learn import statistics
from bnb.runs import sinks
from bnb.runtime import analytics, metrics

CPU = torch.device("cpu")


# ------------------------------------------------------------------ the collector


def test_the_collector_reads_every_device_series_in_one_transfer(monkeypatch):
    reads = []
    original = metrics.read_back
    monkeypatch.setattr(metrics, "read_back", lambda values: reads.append(1) or original(values))
    collector = metrics.Collector()
    collector.add({"group/count": torch.tensor(3, dtype=torch.int32), "group/value": 1.5})
    collector.add({"group/count": torch.tensor(4, dtype=torch.int32)})
    collector.set(
        {
            "group/mean": Mean(torch.tensor(6.0), torch.tensor(4.0)),
            "group/empty": Mean(torch.tensor(0.0), torch.tensor(0.0)),
            "group/flag": torch.tensor(True),
            "group/table": Table(("a", "b"), ((1, 2),)),
        }
    )
    out = collector.read()
    assert len(reads) == 1  # one transfer for the device's five tensors
    assert out == {
        "group/count": 7,  # summed, and an int
        "group/value": 1.5,
        "group/mean": 1.5,
        "group/flag": 1,
        "group/table": {"columns": ["a", "b"], "rows": [[1, 2]]},
    }  # the empty mean is absent, not 0/0
    assert collector.read() == {}


def test_a_deferred_series_is_evaluated_on_its_copy_after_the_readback():
    tokens = torch.tensor([1.0, 2.0, 3.0])
    deferred = Deferred.of(lambda copy: copy.sum(), tokens)
    tokens.zero_()  # the copy was taken when the series was made
    assert metrics.read({"group/sum": deferred, "group/value": torch.tensor(2.0)}) == {
        "group/sum": 6.0,
        "group/value": 2.0,
    }


def test_a_mean_adds_its_totals_and_its_counts():
    collector = metrics.Collector()
    collector.add({"x/mean": Mean(torch.tensor(1.0), 1.0)})
    collector.add({"x/mean": Mean(torch.tensor(5.0), 3.0)})
    assert collector.read() == {"x/mean": 1.5}


def test_the_cadence_measures_the_first_and_every_tenth_update():
    assert [update for update in range(1, 31) if metrics.due(update)] == [1, 10, 20, 30]


# ------------------------------------------------------------------ the rollout's series


def _stored(views: int = 2, length: int = 3, ships: int = 2) -> dict:
    """A hand-built shard of two views (pools 0 and 1), two own ships and two enemies."""
    reward = torch.zeros((length, views, ships, len(REWARD_LEVELS)))
    reward[0, 0, 0, 0] = 2.0  # the first level: +2 then -1 in pool 0
    reward[1, 0, 1, 0] = -1.0
    reward[2, 1, 0, 1] = 3.0  # the second level: +3 in pool 1
    done = torch.zeros((length, views), dtype=torch.bool)
    done[1, 0] = True  # view 0 wins at step 1
    done[2, 1] = True  # view 1 draws at step 2
    result = torch.zeros((length, views), dtype=torch.int32)
    result[1, 0] = 1
    alive = torch.ones((length, views, 2 * ships), dtype=torch.bool)
    alive[0, 1, 1] = False  # view 1's second ship is dead at step 0
    power = torch.tensor([[[0, 1], [2, 2]], [[1, 1], [2, 2]], [[1, 0], [0, 2]]])
    action = Action(
        power=torch.cat([power, power], -1).int(),
        stick=torch.zeros((length, views, 2 * ships), dtype=torch.int32),
        shoot=torch.ones((length, views, 2 * ships), dtype=torch.int32),
    )
    return dict(reward=reward, done=done, result=result, alive=alive, action=action)


def _analytics() -> analytics.Analytics:
    return analytics.Analytics(
        ("frontline/self_play", "deathmatch/league"),
        torch.tensor([0, 1]),
        torch.tensor([0, 1, 1]),
        tick_length=0.5,
        mode="frontline",
    )


def test_the_rollout_series_match_a_hand_computed_shard():
    out = metrics.read(_analytics().shard(_stored()))
    # results per pool
    assert out["episode/wins/frontline/self_play"] == 1
    assert out["episode/draws/deathmatch/league"] == 1
    assert out["episode/losses/frontline/self_play"] == 0
    # reward per ship and second: level 0 in pool 0 is (2 - 1) / 2 ships / 3 steps / 0.5 s
    assert out["reward/rate/damage/frontline/self_play"] == pytest.approx(1.0 / 2 / 3 / 0.5)
    assert out["reward/rate/kill_death/deathmatch/league"] == pytest.approx(3.0 / 2 / 3 / 0.5)
    # level shares of the absolute reward: 3 of 6 and 3 of 6
    assert out["reward/level_share/damage"] == pytest.approx(0.5)
    assert out["reward/level_share/kill_death"] == pytest.approx(0.5)
    # 11 of the 12 own ship-steps are alive; view 0 has 6, view 1 has 5
    assert out["episode/alive_share"] == pytest.approx(11 / 12)
    assert out["batch/token_share/mode/frontline"] == pytest.approx(6 / 11)
    assert out["batch/token_share/opponent/league"] == pytest.approx(5 / 11)
    # power use over the living own ships: 0 x3, 1 x4, 2 x4 (view 1's dead ship skipped)
    assert out["policy/usage/power/0"] == pytest.approx(3 / 11)
    assert out["policy/usage/power/1"] == pytest.approx(4 / 11)
    assert out["policy/usage/power/2"] == pytest.approx(4 / 11)
    assert out["policy/usage/shoot/1"] == 1.0 and out["policy/usage/stick/0"] == 1.0
    # switches: pairs alive at both steps with no end between; view 0 ends at step 1, so
    # only (0 -> 1) counts there: ship 0 switches (0 -> 1), ship 1 does not (1 -> 1). View 1:
    # (0 -> 1) ship 0 (2 -> 2), ship 1 dead at 0; (1 -> 2) ship 0 (2 -> 0) switches, ship 1
    # (2 -> 2) does not: 2 switches of 5 pairs
    assert out["policy/switch_rate/power"] == pytest.approx(2 / 5)
    assert out["policy/switch_rate/shoot"] == 0.0


def test_the_episode_series_come_from_the_rollouts_sums():
    sums = torch.tensor([[2.0, 1.0, 30.0], [1.0, 0.0, 8.0], [1.0, 1.0, 12.0]])
    out = metrics.read(_analytics().episodes(sums))
    assert out["episode/count/frontline/self_play"] == 2
    assert out["episode/count/deathmatch/league"] == 2
    assert out["episode/length_seconds/frontline/self_play"] == pytest.approx(30 * 0.5 / 2)
    assert out["episode/end/timeout/deathmatch/league"] == pytest.approx(0.5)
    assert out["episode/end/decided/frontline/self_play"] == pytest.approx(0.5)


@pytest.mark.slow
def test_the_rollout_counts_episode_ends_and_timeouts(global_generator):
    """The rollout's sums against its own records: every learner view's end is its
    environment's end, and a game that lasts its whole clock is a timeout."""
    runner = lc.rollout(cases.model())
    runner.take_episodes()
    trajectory = runner.run()
    sums = runner.take_episodes()
    environments = runner.lineup.environments[runner.learner]
    ends = torch.zeros(runner.environments).index_put_(
        (environments,), trajectory.done.float().sum(0), accumulate=True
    )
    # self-play's two views each record their environment's end: count environments once
    seen = torch.zeros(runner.environments).index_put_(
        (environments,), torch.ones(len(environments)), accumulate=True
    )
    played = seen > 0
    assert torch.equal(sums[played, 0], ends[played] / seen[played])
    assert (sums[:, 1] <= sums[:, 0]).all()
    assert torch.equal(runner.take_episodes(), torch.zeros_like(sums))


# ------------------------------------------------------------------ the learner's series


def test_uniform_logits_give_the_factors_uniform_entropies():
    logits = torch.zeros((4, NUM_JOINT))
    marginals = statistics.marginal_entropies(torch.log_softmax(logits, -1))
    torch.testing.assert_close(
        marginals, torch.tensor([[math.log(3), math.log(7), math.log(2)]] * 4)
    )


def test_policy_sums_match_a_hand_computed_case():
    logits = torch.full((3, NUM_JOINT), -100.0)
    logits[0] = 0.0  # uniform: entropy ln 42
    logits[1, 5] = 100.0  # certain
    logits[2, 7] = 100.0  # certain but masked out
    log_probabilities = torch.log_softmax(logits, -1)
    entropy = -(log_probabilities.exp() * log_probabilities).sum(-1)
    ratio = torch.tensor([1.0, 1.5, 3.0])  # 1.5 is beyond twice a clip of 0.2
    mask = torch.tensor([True, True, False])
    sums = statistics.policy_sums(logits, ratio, entropy, 0.2, mask)
    named = dict(zip(statistics.POLICY_SUMS, sums[: len(statistics.POLICY_SUMS)].tolist()))
    assert named["far_ratio_share"] == 1.0
    assert named["entropy/joint"] == pytest.approx(math.log(42), abs=1e-4)
    assert named["entropy/power"] == pytest.approx(math.log(3), abs=1e-4)
    assert named["near_certain_share"] == 1.0
    histogram = sums[len(statistics.POLICY_SUMS) :]
    assert histogram.sum() == 2.0 and histogram[0] == 1.0 and histogram[-1] == 1.0


def test_histogram_quantiles_interpolate_within_a_bin():
    counts = torch.zeros(statistics.ENTROPY_BINS)
    counts[10] = 4.0
    width = statistics.MAXIMUM_ENTROPY / statistics.ENTROPY_BINS
    quantiles = statistics.histogram_quantiles(counts, torch.tensor([0.25, 0.5, 1.0]))
    torch.testing.assert_close(quantiles, torch.tensor([10.25, 10.5, 11.0]) * width)
    assert torch.isnan(statistics.histogram_quantiles(torch.zeros(4), torch.tensor([0.5]))).all()


def test_target_entropy_of_a_two_hot_is_the_binary_entropy():
    target = torch.tensor([[0.25, 0.75, 0.0], [1.0, 0.0, 0.0]])
    expected = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    torch.testing.assert_close(statistics.target_entropy(target), torch.tensor([expected, 0.0]))


def test_parameter_groups_split_the_trunks_blocks(global_generator):
    model = cases.model()
    groups = statistics.parameter_groups(model)
    blocks = len(model.trunk.blocks)
    assert {f"trunk_block_{index}" for index in range(blocks)} <= groups.keys()
    named = [name for names in groups.values() for name in names]
    assert sorted(named) == sorted(name for name, _ in model.named_parameters())
    tensors = {"a": torch.tensor([3.0]), "b": torch.tensor([4.0])}
    norms = statistics.group_norms(tensors, {"both": ["a", "b"]})
    assert norms["both"].item() == pytest.approx(5.0)


@pytest.fixture(scope="module")
def segment(module_global_generator):
    net = cases.model()
    runner = lc.rollout(net)
    return net, runner


@pytest.mark.slow
def test_the_targets_series_match_a_hand_computed_case(segment, global_generator):
    net, runner = segment
    learner = lc.learner(copy.deepcopy(net), runner)
    L, V, S = 2, 1, 2
    levels = len(REWARD_LEVELS)
    level_returns = torch.zeros((L, V, S, levels))
    level_returns[..., 0] = torch.tensor([[[1.0, 2.0]], [[3.0, 1e9]]])  # 1e9: past the bins
    values = level_returns.clone()
    values[..., 0] += 0.5  # +0.5 on level 0, lost to float32 rounding on the 1e9 return
    valued = torch.tensor([[[True, True]], [[True, False]]])  # the 1e9 return is not valued
    valued[1, 0, 1] = True
    advantage = torch.tensor([[[2.0, -2.0]], [[4.0, 0.0]]])
    alive = torch.ones((L, V, S), dtype=torch.bool)
    scale = torch.tensor(2.0)
    out = metrics.read(
        learner._target_series(level_returns, values, valued, advantage, alive, scale)
    )
    first = REWARD_LEVELS[0]
    assert out[f"critic/return_mean/{first}"] == pytest.approx((1 + 2 + 3 + 1e9) / 4)
    assert out[f"critic/value_bias/{first}"] == pytest.approx(3 * 0.5 / 4)
    assert out[f"critic/support_clip/{first}"] == pytest.approx(0.25)
    assert out[f"returns/p50/{first}"] == pytest.approx(2.5)
    assert out["returns/advantage_mean"] == pytest.approx(0.5)  # (1 - 1 + 2 + 0) / 4
    assert out["returns/advantage_rms"] == pytest.approx(math.sqrt((1 + 1 + 4) / 4))
    assert f"critic/support_clip/{REWARD_LEVELS[learner.outcome_level]}" not in out


@pytest.mark.slow
def test_the_learners_series_are_named_and_finite(segment, global_generator):
    net, runner = segment
    learner = lc.learner(copy.deepcopy(net), runner, epochs=2, minibatches=2)
    out = metrics.read(learner.update(runner.store, runner.bootstrap()))
    for name in (
        "policy/kl",
        "policy/kl_first",
        "policy/entropy/joint",
        "policy/entropy_quantile/p50",
        "policy/factor_coupling",
        "critic/value_kl/outcome",
        "train/gradient_norm",
        "returns/advantage_rms",
        "next_state/loss",
        "shot/loss/outcome",
    ):
        assert math.isfinite(out[name]), name
    joint = out["policy/entropy/joint"]
    assert out["policy/factor_coupling"] >= -1e-5  # total correlation is not negative
    assert out["policy/entropy_quantile/p10"] <= out["policy/entropy_quantile/p90"]
    assert 0.0 <= joint <= math.log(NUM_JOINT) + 1e-5
    assert out["policy/epochs_completed"] == 2 and out["policy/minibatch_steps"] == 4
    assert out["train/skipped_steps"] == 0
    assert all(out[name] > 0 for name in out if name.startswith("train/parameter_norm/"))


@pytest.mark.slow
def test_diagnostics_do_not_change_the_update(segment, global_generator):
    """The same seed gives the same update, bit for bit, with diagnostics on and off."""
    net, runner = segment
    results = []
    for diagnostics in (True, False):
        model = copy.deepcopy(net)
        learner = lc.learner(model, runner, epochs=2, minibatches=2)
        learner.diagnostics = diagnostics
        out = metrics.read(learner.update(runner.store, runner.bootstrap()))
        results.append((out, [p.detach().clone() for p in model.parameters()]))
    (on, on_weights), (off, off_weights) = results
    assert all(torch.equal(a, b) for a, b in zip(on_weights, off_weights, strict=True))
    shared = on.keys() & off.keys()
    assert {name: on[name] for name in shared} == {name: off[name] for name in shared}
    assert "policy/entropy/joint" in on and "policy/entropy/joint" not in off


@pytest.mark.slow
def test_a_non_finite_gradient_skips_the_step_and_keeps_the_state(segment, global_generator):
    net, runner = segment
    model = copy.deepcopy(net)
    learner = lc.learner(model, runner, epochs=1, minibatches=2)
    before = [p.detach().clone() for p in model.parameters()]
    hook = next(model.parameters()).register_hook(lambda gradient: gradient * math.nan)
    out = metrics.read(learner.update(runner.store, runner.bootstrap()))
    hook.remove()
    assert out["train/skipped_steps"] == 2
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    assert all(
        torch.equal(value, torch.zeros_like(value))
        for state in learner.optimizer.state.values()
        for value in state.values()
    )  # the first step's state, as Adam starts it
    assert "train/gradient_norm" not in out  # no finite step to average
    out = metrics.read(learner.update(runner.store, runner.bootstrap()))
    assert out["train/skipped_steps"] == 0  # training continues
    assert not all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


# ------------------------------------------------------------------ the league's series


def test_the_sampling_entropy_is_normalised():
    assert normalised_entropy([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert normalised_entropy([1.0, 0.0, 0.0]) == 0.0
    assert normalised_entropy([0.5, 0.25, 0.25]) == pytest.approx(1.5 * math.log(2) / math.log(3))
    assert normalised_entropy([1.0]) == 0.0


def test_entry_kinds_split_the_ladder():
    assert entry_kind("scripted", "ladder") == "scripted"
    assert entry_kind("rung_T2", "ladder") == "rung"
    assert entry_kind("random", "ladder") == "rung"
    assert entry_kind("step_100", "checkpoint") == "checkpoint"


# ------------------------------------------------------------------ the method rules

# Abbreviations a series name may use (code style, "Names"): universal ones and percentiles.
ALLOWED = {"kl", "rms", "x", "y", "id", "of", "p5", "p10", "p50", "p90", "p95"}
# Truncations a series name must not use.
FORBIDDEN = {
    "adv", "avg", "cfg", "env", "envs", "ev", "grad", "grads", "idx", "lr", "num", "obs",
    "pct", "prob", "probs", "repr", "sps", "std", "temp", "val", "vals",
}  # fmt: skip


def check_names(names) -> list[str]:
    """The names that break the rule: lowercase parts of words in full, quantity first."""
    bad = []
    for name in names:
        parts = name.split("/")
        if len(parts) < 2 or not all(re.fullmatch(r"[a-z0-9_.]+", part) for part in parts):
            bad.append(name)
            continue
        words = [word for part in parts for word in re.split(r"[_.]", part) if word]
        for word in words:
            if word in FORBIDDEN or (len(word) < 3 and not word.isdigit() and word not in ALLOWED):
                bad.append(name)
                break
    return bad


def test_the_name_check_catches_truncations():
    assert check_names(["train/grad_norm", "critic/ev/outcome", "Policy/KL", "loss"]) == [
        "train/grad_norm",
        "critic/ev/outcome",
        "Policy/KL",
        "loss",
    ]
    assert (
        check_names(["policy/kl", "returns/p95/damage", "train/update_ratio/trunk_block_0"]) == []
    )


def _tiny_recorded(tmp_path, updates: int = 2):
    sys.path.insert(0, str(lc.__file__).rsplit("/sections/", 1)[0] + "/runs")
    from runs_cases import tiny_train_config, tiny_yemong_config

    from bnb.runs.recording import Recorder
    from bnb.runs.records import Run
    from bnb.runs.triggers import Triggers
    from bnb.runtime.train import Trainer

    config = tiny_train_config("yemong", updates=updates, yemong=tiny_yemong_config())
    trainer = Trainer(config, CPU)
    run = Run.create(tmp_path, "names", config, ["test"], None)
    output = sinks.Sinks(run, {})
    recorder = Recorder(run, trainer, output, Triggers())
    records = [recorder.record(trainer.update()) for _ in range(updates)]
    output.close()
    return run, trainer, records


@pytest.fixture(scope="module")
def recorded(tmp_path_factory, module_global_generator):
    return _tiny_recorded(tmp_path_factory.mktemp("recorded"))


@pytest.mark.slow
def test_every_logged_name_is_quantity_first_in_full_words(recorded):
    _, _, records = recorded
    names = {name for record in records for name in record}
    assert len(names) > 100
    assert check_names(sorted(names)) == []


@pytest.mark.slow
def test_the_jsonl_sink_holds_every_record(recorded):
    run, _, records = recorded
    assert sinks.read_metrics(run.directory / sinks.METRICS) == json.loads(json.dumps(records))


@pytest.mark.slow
def test_the_recorder_adds_throughput_counters_and_alerts(recorded):
    _, trainer, records = recorded
    last = records[-1]
    assert last["counters/updates"] == 2
    assert last["train/environment_steps_per_second"] == pytest.approx(
        trainer.steps_per_update / last["train/time/update_seconds"]
    )
    assert last["train/ship_tokens_per_second"] == pytest.approx(
        last["train/environment_steps_per_second"] * trainer.config.game.num_ships
    )
    assert last["counters/skipped_steps"] == 0 and last["counters/recompiles"] == 0
    assert last["alerts/active"] == sum(
        value
        for name, value in last.items()
        if name.startswith("alerts/") and name != "alerts/active"
    )


@pytest.mark.slow
def test_no_logged_series_is_smoothed(segment, global_generator):
    """A series is a function of the update's state and batch, never of the run's history:
    two learners with the same weights and optimizer state, one of them with a history of
    updates, log the same series on the same batch. The return scale's percentile EMAs are
    the justified exception (the scale is held at its floor here, so nothing else reads
    them)."""
    net, runner = segment
    justified = {"returns/scale_p5", "returns/scale_p95"}
    fresh = lc.learner(copy.deepcopy(net), runner, epochs=1, minibatches=2, return_scale_floor=1e6)
    seasoned = lc.learner(
        copy.deepcopy(net), runner, epochs=1, minibatches=2, return_scale_floor=1e6
    )
    for _ in range(2):
        seasoned.update(runner.store, runner.bootstrap())
    seasoned.model.load_state_dict(fresh.model.state_dict())
    seasoned.optimizer.load_state_dict(fresh.optimizer.state_dict())
    seasoned.environment_steps = fresh.environment_steps
    for learner in (fresh, seasoned):  # the same minibatch order
        learner.order_generator.manual_seed(7)
        learner.spread_generator.manual_seed(7)
    first = metrics.read(fresh.update(runner.store, runner.bootstrap()))
    second = metrics.read(seasoned.update(runner.store, runner.bootstrap()))
    assert first.keys() == second.keys()
    differing = {name for name in first if first[name] != second[name]}
    assert differing <= justified, sorted(differing - justified)


def test_an_update_of_several_shards_counts_each_shards_episodes_once(global_generator):
    """The analytics read one shard at a time, so their pool membership is one shard's
    views: an update of two shards runs and its pools' results are the shards' sum."""
    sys.path.insert(0, str(lc.__file__).rsplit("/sections/", 1)[0] + "/runs")
    from runs_cases import tiny_train_config

    from bnb.runtime.train import Trainer

    trainer = Trainer(tiny_train_config(shards=2, updates=1), CPU)
    row = trainer.update()
    store = trainer.rollout.store
    done = sum(store.shard(("done",), shard)["done"].sum() for shard in range(2))
    results = sum(
        row[f"episode/{name}/{pool}"]
        for name in ("wins", "draws", "losses")
        for pool in trainer.pool_names
    )
    assert results == int(done)
