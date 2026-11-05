"""Behaviour cloning (scripted spec B6): the loss, the labels, the temperature and the stop
rule, on a real stored Yemong segment on the CPU."""

import dataclasses
import functools
import json
import math
import sys

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases
from scripted_cases import CHARACTERISTICS

from bnb.contracts import Obs, Pool, Trajectory
from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Scripted, distribution
from bnb.game import api
from bnb.learn import bc
from bnb.learn.rewards import RewardConfig
from bnb.models.relational import RelationalConfig
from bnb.runtime import metrics
from bnb.runtime.lanes import LaneConfig
from bnb.runtime.rollout import Rollout

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


POOLS = (
    Pool("self_play", 0.5, ("learner", "learner"), 0.5),
    Pool("scripted", 0.5, ("learner", "scripted"), 0.5, mode="frontline"),
)
TEACHER = functools.partial(
    distribution, characteristics=CHARACTERISTICS, parameters=Scripted(CHARACTERISTICS).parameters
)


def _cloning(net, runner: Rollout, teacher_config=bc.TeacherConfig(), **config):
    views = runner.store.views
    config = dict(dict(minibatches=1, micro_batch_rows=views * runner.rules.num_ships), **config)
    return bc.BehaviourCloning(
        net,
        bc.config(**config),
        runner.rules,
        runner.carry["types"],
        RewardConfig().outcome,
        lc.CPU,
        TEACHER,
        teacher_config,
        tuple(pool.name for pool in POOLS),
        runner.lineup.pool_of["learner"],
    )


@pytest.fixture(scope="module")
def segment(module_global_generator):
    net = cases.model()
    controllers = {
        "learner": cases.controller(net, cases.SHORT_SQUAD),
        "scripted": Scripted(CHARACTERISTICS),
    }
    runner = Rollout(
        cases.SHORT_SQUAD,
        api.ship_types(lc.CPU),
        POOLS,
        controllers,
        "learner",
        8,
        32,
        lc.CPU,
        compiled=False,
        extra_reads=Scripted.reads,
    )
    for _ in range(2):
        runner.run()
    learner = _cloning(net, runner)
    prepared = learner.prepare(runner.store, runner.bootstrap())
    return net, runner, learner, prepared


def test_the_bc_loss_matches_an_independent_computation(segment):
    net, runner, learner, prepared = segment
    trajectory = lc.whole(runner)
    # Mark a third of the own rows dead, so the mask matters.
    marked = trajectory.alive.clone()
    own = trajectory.log_probability.shape[-1]
    marked[..., :own] &= (
        torch.rand(marked[..., :own].shape, generator=torch.Generator().manual_seed(1)) > 0.33
    )
    trajectory = dataclasses.replace(trajectory, alive=marked)
    with torch.no_grad():
        unrolled = learner.unroll(trajectory)
    alive = trajectory.alive[..., :own]
    tokens = alive.sum()
    assert 0 < tokens < alive.numel()
    cross, kl = torch.zeros(()), torch.zeros(())
    for tick in range(trajectory.done.shape[0]):  # the teacher tick by tick, on its views
        obs = Obs(
            **{
                name: None if value is None else value[tick]
                for name, value in dataclasses.asdict(trajectory.obs).items()
            }
        )
        q = distribution(obs, CHARACTERISTICS, Scripted(CHARACTERISTICS).parameters)
        assert bc.TeacherConfig() == bc.TeacherConfig(1.0, 0.0)  # the labels are q itself
        log_student = torch.log_softmax(unrolled.policy_logits[tick], -1)
        per_token = -(q * log_student).sum(-1)
        entropy = -torch.special.xlogy(q, q).sum(-1)
        cross = cross + per_token[alive[tick]].sum()
        kl = kl + (per_token - entropy)[alive[tick]].sum()
    derived = {name: value for name, value in prepared["derived"].items()}
    policy_term, _, statistics = learner._policy(
        unrolled, trajectory, derived, dict(policy=tokens.float()), prepared["scale"]
    )
    torch.testing.assert_close(policy_term, cross / tokens, rtol=1e-5, atol=1e-6)
    report = metrics.read(learner.report(statistics, statistics, statistics))
    assert report["bc/kl"] == pytest.approx(float(kl / tokens), rel=1e-5)
    assert set(report) >= {
        "bc/kl_by_pool/self_play",
        "bc/entropy_by_pool/target/scripted",
        "bc/entropy/student",
        "bc/agreement/joint",
    }


def test_micro_batches_accumulate_to_the_unsplit_minibatch(segment):
    net, runner, learner, prepared = segment
    views = torch.arange(runner.store.views)

    def gradient(size: int):
        lc.with_micro_batch(learner, size)
        terms, _, statistics, _, _ = learner.gradient(runner.store, prepared, views)
        return terms, statistics, [parameter.grad.clone() for parameter in net.parameters()]

    whole_terms, whole_statistics, whole = gradient(runner.store.views)
    split_terms, split_statistics, split = gradient(runner.store.views // 4)
    lc.with_micro_batch(learner, runner.store.views)
    torch.testing.assert_close(split_terms, whole_terms, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(split_statistics, whole_statistics, rtol=1e-5, atol=1e-4)
    for a, b in zip(split, whole, strict=True):
        assert (a - b).abs().max().item() <= 1e-5 * max(1e-12, b.abs().max().item()) + 1e-8
    assert sum(g.abs().sum().item() for g in whole) > 0


class Recording(Scripted):
    """The scripted agent as the learner, keeping the distribution of every decision."""

    def __init__(self) -> None:
        super().__init__(CHARACTERISTICS)
        self.decisions: list[torch.Tensor] = []

    def step(self, carry, obs, reset):
        self.decisions.append(distribution(obs, self.characteristics, self.parameters))
        return super().step(carry, obs, reset)


def test_the_update_time_labels_are_the_rollout_time_distribution():
    recording = Recording()
    pools = (Pool("self_play", 1.0, ("learner", "learner"), 0.5, mode="frontline"),)
    runner = Rollout(
        cases.SHORT_FRONTLINE,
        api.ship_types(lc.CPU),
        pools,
        {"learner": recording},
        "learner",
        4,
        16,
        lc.CPU,
        compiled=False,
    )
    trajectory = runner.run()
    decided = torch.stack(recording.decisions[1:])  # the first is the rollout's one-row probe
    for tick in range(trajectory.done.shape[0]):  # the stored views give the same decision
        stored = Obs(
            ships=trajectory.obs.ships[tick],
            fields=None,
            globals=trajectory.obs.globals[tick],
            zones=trajectory.obs.zones[tick],
        )
        assert torch.equal(TEACHER(stored), decided[tick])
    # Per tick the labels are exact (above). The update labels a whole segment as one
    # batch, and the CPU's vectorized kernels round the tail of a batch apart from its body,
    # so a batch of another shape may differ at the ulp level: measured at most 1.2e-7 (on
    # 0.15% of entries). The tolerance is 2.5e-7, about two float32 ulps at 1.
    labels = bc.teacher_distribution(TEACHER, trajectory.obs)
    torch.testing.assert_close(labels, decided, rtol=0, atol=2.5e-7)


def _living_distributions(segment):
    """The stored segment's views and own-alive mask, and T -> the scripted agent's
    distributions at T over the living own ships (tokens, 42)."""
    _, runner, _, _ = segment
    views = bc.flat_views(lc.whole(runner).obs)
    own = views.ships.shape[1] // 2
    alive = views.ships[:, :own, SHIP["alive"]] > 0.5

    def at(temperature: float) -> torch.Tensor:
        return distribution(
            views, CHARACTERISTICS, Scripted(CHARACTERISTICS).parameters, temperature
        )[alive]

    return at


def test_the_teacher_temperature_and_its_calibration(segment):
    """The labels' floor mixes with uniform; the agent's own temperature raises its
    entropy; the calibration finds the temperature for a target entropy."""
    at = _living_distributions(segment)
    q = at(1.0)
    assert torch.equal(bc.labels(q, 0.0), q)
    floored = bc.labels(q, 0.1)
    torch.testing.assert_close(floored, 0.9 * q + 0.1 / 42)
    entropies = [bc.mean_entropy(at(t)) for t in (0.5, 1.0, 2.0, 4.0)]
    assert entropies == sorted(entropies) and entropies[-1] > entropies[0] + 0.5
    for target in (entropies[1] + 0.1, entropies[2]):
        chosen = bc.calibrate(at, target, 0.0, low=0.25, high=16.0)
        assert bc.mean_entropy(at(chosen.temperature)) == pytest.approx(target, abs=0.01)
    with pytest.raises(ValueError):
        bc.calibrate(at, math.log(42) + 0.01, 0.0)  # above uniform


def test_every_label_gives_every_action_a_probability_above_zero(segment):
    """The scripted agent's own floors (an epsilon per binary decision but the shot): every
    living ship's every action has a probability above zero at T = 1, and at a cold and a
    hot T, with no floor on the labels, except a shot at its gates (no target, or a gun that
    cannot fire), where P(shoot) is 0."""
    at = _living_distributions(segment)
    for temperature in (1.0, 0.25, 4.0):
        joint = at(temperature).unflatten(-1, (3, 7, 2))  # power, stick, shoot
        shoot = joint.sum((-3, -2))[:, 1]
        assert bool((joint[..., 0] > 0.0).all()), temperature  # every not-shooting action
        open_gate = shoot > 0.0
        assert bool((joint[open_gate][..., 1] > 0.0).all()), temperature
        assert bool(open_gate.any()) and bool((shoot < 1.0).all())
    assert bc.TeacherConfig().floor == 0.0 and bc.TeacherConfig().temperature == 1.0


def _scripted_seat(
    objective: str, teacher: bc.TeacherConfig = bc.TeacherConfig(), segments: int = 0
):
    """A small CPU trainer with BC's lineup, its scripted seat's controller and that seat's
    view after `segments` segments of 8 ticks (from 2 on, the scripted agent is no longer
    certain of every action)."""
    from bnb.contracts import GameConfig
    from bnb.runtime.cloning import LINEUP
    from bnb.runtime.train import SCRIPTED, TrainConfig, Trainer

    learner = bc.config(minibatches=2) if objective == "bc" else lc.LearnerConfig(minibatches=2)
    cfg = TrainConfig(
        game=GameConfig(2, 2, 1200.0),
        pools=LINEUP,
        environments=4,
        length=8,
        shards=1,
        lookahead=0,
        lanes=LaneConfig(environments=0),
        model="relational",
        relational=RelationalConfig(hidden=16),
        compiled=False,
        host_store=False,
        learner=learner,
        objective=objective,
        teacher=teacher,
    )
    trainer = Trainer(cfg, lc.CPU)
    for _ in range(segments):
        trainer.rollout.run()
    view = trainer.rollout._view(trainer.rollout.carry, SCRIPTED, Scripted.reads)
    return trainer, trainer.controllers[SCRIPTED], view


def _repeated(view: Obs, times: int) -> Obs:
    """The view's rows (R, ...) repeated `times` times: (times R, ...)."""
    parts = {field.name: getattr(view, field.name) for field in dataclasses.fields(view)}
    return Obs(
        **{
            name: None if value is None else value.repeat(times, *[1] * (value.dim() - 1))
            for name, value in parts.items()
        }
    )


def test_the_scored_opponent_plays_the_labels_distribution():
    """BC's scripted seat samples the labels' distribution (the agent at the teacher's
    temperature, the floor applied): its stated distribution is the labels', BC's labels
    come from the agent at that temperature, and its draws' frequencies over 20000 copies
    of each view match it within five standard errors."""
    from bnb.contracts.action import NUM_JOINT, to_joint

    teacher = bc.TeacherConfig(temperature=3.0, floor=0.2)
    trainer, opponent, view = _scripted_seat("bc", teacher, segments=2)
    expected = bc.labels(TEACHER(view, temperature=3.0), teacher.floor)  # (R, S, 42)
    untempered = bc.labels(TEACHER(view), teacher.floor)
    assert float((expected - untempered).abs().max()) > 0.05  # the temperature matters here
    torch.testing.assert_close(opponent.action_distribution(view), expected)
    torch.testing.assert_close(trainer.ppo.teacher(view), TEACHER(view, temperature=3.0))
    assert trainer.ppo.teacher_config == teacher
    assert trainer.rollout.reads == ("ships", "globals", "zones")
    times = 20_000
    torch.manual_seed(0)
    action, _, _ = opponent.step(None, _repeated(view, times), None)
    joint = to_joint(action).long().unflatten(0, (times, -1))  # (times, R, S)
    frequency = torch.nn.functional.one_hot(joint, NUM_JOINT).float().mean(0)
    error = (expected * (1 - expected) / times).sqrt()
    assert bool(((frequency - expected).abs() <= 5 * error + 1e-4).all()), float(
        (frequency - expected).abs().max()
    )
    assert float(expected.min()) >= teacher.floor / NUM_JOINT * 0.999  # the floor is played


def test_floor_0_and_temperature_1_leave_the_scripted_agents_play_unchanged():
    """At floor 0 and T = 1 the scored opponent draws exactly the scripted agent's actions
    from the same random state; outside BC the scripted seat is the scripted agent itself."""
    plain = bc.TeacherConfig(temperature=1.0, floor=0.0)
    _, opponent, view = _scripted_seat("bc", plain, segments=2)
    assert isinstance(opponent, bc.TeacherOpponent)
    views = _repeated(view, 500)
    agent = Scripted(CHARACTERISTICS)
    draws = []
    for player in (opponent, agent):  # two steps each: the second sees the first's draws
        torch.manual_seed(1)
        draws.append([player.step(None, views, None)[0].packed() for _ in range(2)])
    for drawn, scripted in zip(*draws, strict=True):
        assert torch.equal(drawn, scripted)
        assert len(set(map(tuple, drawn.flatten(0, 1).tolist()))) > 1  # not one action
    torch.testing.assert_close(opponent.action_distribution(view), TEACHER(view))
    _, reinforcement_opponent, _ = _scripted_seat("ppo")
    assert type(reinforcement_opponent) is Scripted
    assert reinforcement_opponent.temperature == 1.0


# ------------------------------------------------------------------ the stop rule


def _run(rule: bc.StopRule, updates: int, steps: int, score: float, kl):
    """Feed `updates` updates of `steps` environment steps, 20 games each at this score;
    kl(update) the BC KL. Returns (update, reason) of the first rule to fire."""
    for update in range(updates):
        wins = round(20 * score)
        reason = rule.observe((update + 1) * steps, wins, 0, 20 - wins, kl(update))
        if reason is not None:
            return update, reason
    return None


def test_each_stop_rule_fires_alone():
    config = bc.StopConfig(score_window=100, plateau_steps=1000, budget_steps=10_000)
    falling = lambda update: 2.0 * 0.95**update  # noqa: E731 - still improving 5% per update
    # 0.8 over 100 games: above parity once the window is full
    assert _run(bc.StopRule(config), 200, 100, 0.8, falling) == (4, "score")  # 100 games
    assert _run(bc.StopRule(config), 200, 100, 0.2, falling) == (99, "budget")
    flat = lambda update: 1.0 + 0.5 * 0.5**update  # noqa: E731 - converges, then plateaus
    update, reason = _run(bc.StopRule(config), 200, 100, 0.2, flat)
    assert reason == "kl_plateau" and 10 < update < 99
    rule = bc.StopRule(config)
    _run(rule, update, 100, 0.2, flat)  # one update earlier it had not plateaued
    assert rule.score == pytest.approx(0.2)


def test_the_score_window_trims_whole_updates_without_favouring_a_result():
    """Window 100: an update of 30 wins and 30 losses, then one of 60 draws. The window
    keeps 40/60 of the first (20 points) and the draws (30 points): 0.50, where dropping
    the oldest games one by one, wins first, would leave 0.40."""
    rule = bc.StopRule(bc.StopConfig(score_window=100, score_target=1.1))
    rule.observe(1, 30, 0, 30, 1.0)
    assert rule.score is None  # 60 games: not yet full
    rule.observe(2, 0, 60, 0, 1.0)
    assert rule.score == pytest.approx(0.5)
    rule.observe(3, 0, 0, 0, 1.0)  # an update without a finished game changes nothing
    assert rule.score == pytest.approx(0.5)
    rule.observe(4, 100, 0, 0, 1.0)  # a full window's worth drops both older updates whole
    assert rule.score == pytest.approx(1.0) and len(rule.updates) == 1


def test_the_score_stop_reads_the_window_trimmed_like_the_score():
    """Window 100: 30 wins and 30 losses, then 20 wins, 40 draws and 10 losses. The window
    keeps half the first update, so its 100 games are 35 wins, 40 draws and 25 losses: a plain
    score of 0.55, reported with its standard error (sample deviation / 10)."""
    config = bc.StopConfig(score_window=100, score_target=0.55, budget_steps=10**9)
    rule = bc.StopRule(config)
    assert rule.observe(1, 30, 0, 30, 1.0) is None and rule.score is None
    games = torch.tensor([1.0] * 35 + [0.5] * 40 + [0.0] * 25, dtype=torch.float64)
    assert rule.observe(2, 20, 40, 10, 1.0) == "score"
    assert rule.score == pytest.approx(0.55)
    assert rule.standard_error == pytest.approx(float(games.std() / 10))
    above = bc.StopRule(dataclasses.replace(config, score_target=0.551))
    above.observe(1, 30, 0, 30, 1.0)
    assert above.observe(2, 20, 40, 10, 1.0) is None


def test_the_score_stop_fires_at_parity_by_default():
    config = bc.StopConfig(score_window=100, budget_steps=10**9)
    assert config.score_target == pytest.approx(0.5)  # parity (author)
    below = bc.StopRule(config)
    assert below.observe(1, 49, 0, 51, 1.0) is None
    at = bc.StopRule(config)
    assert at.observe(1, 40, 20, 40, 1.0) == "score"


def test_the_budget_stops_bc_at_v1s_time_to_parity_by_default():
    assert bc.StopConfig().budget_steps == 50_000_000
    rule = bc.StopRule(bc.StopConfig())
    assert rule.observe(49_999_999, 0, 0, 0, 1.0) is None
    assert rule.observe(50_000_000, 0, 0, 0, 1.0) == "budget"


def test_the_stop_rules_state_round_trips_through_json_and_continues_the_window():
    """A resume restores the rule from its JSON state: the window, the KL's EMA and its
    history, so the restored rule decides exactly as the original would."""
    config = bc.StopConfig(score_window=100, plateau_steps=1000, budget_steps=10**9)
    flat = lambda update: 1.0 + 0.5 * 0.5**update  # noqa: E731 - converges, then plateaus
    original = bc.StopRule(config)
    assert _run(original, 6, 100, 0.3, flat) is None
    restored = bc.StopRule(config, json.loads(json.dumps(original.state)))
    assert restored.state == original.state and restored.score == original.score
    for update in range(6, 60):
        kl = flat(update)
        decisions = [
            rule.observe((update + 1) * 100, 6, 0, 14, kl) for rule in (original, restored)
        ]
        assert decisions[0] == decisions[1] and original.state == restored.state
    assert decisions[0] == "kl_plateau"
    # The history keeps only what the plateau still reads: one average at or before
    # plateau_steps back, and the newer ones.
    assert len(original.history) == 1000 // 100 + 1


def test_alive_mask_reads_the_own_rows(segment):
    _, runner, _, _ = segment
    trajectory: Trajectory = lc.whole(runner)
    S = trajectory.log_probability.shape[-1]
    torch.testing.assert_close(
        trajectory.alive[..., :S].float(),
        (trajectory.obs.ships[..., :S, SHIP["alive"]] > 0.5).float(),
    )


# ------------------------------------------------------------------ the hand-off


def _tiny(objective: str, pools):
    from bnb.contracts import GameConfig
    from bnb.learn.ppo import LearnerConfig
    from bnb.runtime.train import TrainConfig

    learner = bc.config(minibatches=2) if objective == "bc" else LearnerConfig(minibatches=2)
    return TrainConfig(
        game=GameConfig(2, 2, 1200.0),
        pools=pools,
        environments=8,
        length=8,
        shards=1,
        lookahead=0,
        lanes=LaneConfig(environments=0),
        model="relational",
        relational=RelationalConfig(hidden=16),
        compiled=False,
        host_store=False,
        learner=learner,
        objective=objective,
    )


def test_reinforcement_learning_starts_from_the_clone():
    from bnb.learn.ppo import Learner
    from bnb.runtime.cloning import LINEUP, warm_start
    from bnb.runtime.train import LINEUP as RL_LINEUP
    from bnb.runtime.train import Trainer

    torch.manual_seed(0)
    student = Trainer(_tiny("bc", LINEUP), lc.CPU)
    student.update()
    trainer = Trainer(_tiny("ppo", RL_LINEUP), lc.CPU)
    warm_start(trainer, student)
    for name, tensor in student.model.state_dict().items():
        assert torch.equal(trainer.model.state_dict()[name], tensor), name
        for set_weights in trainer.opponents.parameters[f"model.{name}"]:  # every slot's set
            assert torch.equal(set_weights, tensor), name
    assert float(trainer.ppo.return_scale.scale()) == float(student.ppo.return_scale.scale())
    assert float(student.ppo.return_scale.high) != 0.0
    assert not trainer.ppo.optimizer.state  # fresh
    assert type(trainer.ppo) is Learner and trainer.config.learner.policy == 27.0  # no cloning
    row = trainer.update()
    assert "bc/kl" not in row and "policy/kl" in row


# The tiny BC run of the command line: 2v2 relational, 16 environments x 16 ticks per
# update and 4 rating lanes, stopped by its budget after two updates.
BC_OPTIONS = (
    "--device",
    "cpu",
    "--model",
    "relational",
    "--ships-per-team",
    "2",
    "--num-fields",
    "2",
    "--arena-radius",
    "1500",
    "--environments",
    "16",
    "--objective",
    "bc",
    "--bc-budget-steps",
    str(2 * 16 * 16),
    "--shards",
    "1",
    "--set",
    "length=16",
    "--set",
    "lookahead=0",
    "--set",
    "lanes.environments=4",
    "--set",
    "compiled=false",
    "--set",
    "learner.minibatches=2",
    "--set",
    "learner.epochs=1",
    "--set",
    "relational.hidden=16",
    "--set",
    "game.episode_seconds=4.0",
)


def _bnb(monkeypatch, *arguments: str) -> None:
    from bnb import cli

    monkeypatch.setattr(sys, "argv", ["bnb", *arguments])
    with torch.random.fork_rng(devices=[]):
        cli.main()


def test_bnb_train_objective_bc_saves_the_clone_as_a_policy_file(monkeypatch, tmp_path):
    """The hand-off (scripted spec B4, runs spec 3): a BC run saves its clone in its run
    directory with the weights, the return scale and the step counts; RL starts from it."""
    from bnb.runs import checkpoint
    from bnb.runs.records import Run

    runs = str(tmp_path / "runs")
    _bnb(monkeypatch, "train", *BC_OPTIONS, "--name", "clone", "--runs-directory", runs)
    run = Run.open(tmp_path / "runs", "clone")
    assert run.record["status"] == "finished" and run.record["stopped_by"] == "budget"
    assert run.record["segments"][0]["end_step"] == 2 * 16 * 16
    assert json.loads((run.directory / "config.json").read_text())["objective"] == "bc"
    record = checkpoint.read_record(run.policy_path)
    assert record["kind"] == "policy" and record["return_scale"]
    assert record["steps"] == dict(environment_steps=2 * 16 * 16, updates=2)
    clone = checkpoint.load_policy(run.policy_path, lc.CPU)
    # Reinforcement learning from it: the clone's weights, a run of its own.
    _bnb(
        monkeypatch,
        "train",
        *BC_OPTIONS[:12],
        "--set",
        "length=16",
        "--set",
        "compiled=false",
        "--set",
        "relational.hidden=16",
        "--set",
        "game.episode_seconds=4.0",
        "--updates",
        "0",
        "--init-from",
        str(run.policy_path),
        "--name",
        "rl",
        "--runs-directory",
        runs,
    )
    rl = Run.open(tmp_path / "runs", "rl")
    assert rl.record["parent"] == dict(policy=str(run.policy_path.resolve()))
    final = checkpoint.load_policy(rl.checkpoint_at(), lc.CPU)
    for name, value in clone.state_dict().items():
        assert torch.equal(final.state_dict()[name], value), name


@pytest.mark.parametrize("start", [("--from", "clone"), ("--init-from", "somewhere")])
def test_a_bc_run_starts_fresh_or_resumes(monkeypatch, tmp_path, start):
    with pytest.raises(SystemExit, match="a BC run starts fresh or resumes"):
        _bnb(monkeypatch, "train", *BC_OPTIONS, *start, "--runs-directory", str(tmp_path))


def test_a_bc_run_writes_training_checkpoints_and_resumes(monkeypatch, tmp_path):
    """A BC run checkpoints on the usual cadence and at its end, and `--resume` continues
    it with its stop rule's window and budget: stopped by its budget after two updates, then
    resumed with a larger budget, it trains two more and replaces its clone."""
    from bnb.runs import checkpoint
    from bnb.runs.records import Run

    runs = str(tmp_path / "runs")
    every = ("--checkpoint-every", str(16 * 16))
    _bnb(monkeypatch, "train", *BC_OPTIONS, *every, "--name", "clone", "--runs-directory", runs)
    run = Run.open(tmp_path / "runs", "clone")
    assert run.checkpoint_steps() == [16 * 16, 2 * 16 * 16]
    assert run.record["final_checkpoint"] == 2 * 16 * 16
    newest = run.checkpoint_at()
    saved = checkpoint.read_record(newest)
    config, _ = checkpoint.training_config(newest)
    assert config.objective == "bc" and config.stop.budget_steps == 2 * 16 * 16
    window = saved["stop_rule"]
    assert len(window["history"]) == 2 and window["kl_average"] is not None
    _bnb(
        monkeypatch,
        "train",
        "--resume",
        "clone",
        "--device",
        "cpu",
        "--bc-budget-steps",
        str(4 * 16 * 16),
        "--runs-directory",
        runs,
    )
    run = Run.open(tmp_path / "runs", "clone")
    assert run.record["status"] == "finished" and run.record["stopped_by"] == "budget"
    first, second = run.record["segments"]
    assert second["start_step"] == 2 * 16 * 16 and second["end_step"] == 4 * 16 * 16
    assert second["overrides"] == {"stop.budget_steps": 4 * 16 * 16}
    resumed = checkpoint.read_record(run.checkpoint_at())["stop_rule"]
    assert resumed["history"][:2] == window["history"]  # the window went on
    assert checkpoint.read_record(run.policy_path)["steps"]["environment_steps"] == 4 * 16 * 16
    assert not run.policy_path.with_name("policy.replaced").exists()
    rows = [json.loads(line) for line in (run.directory / "metrics.jsonl").read_text().splitlines()]
    assert [row["counters/updates"] for row in rows] == [1, 2, 3, 4]
    assert all("alerts/entropy_collapse" not in row for row in rows)  # an RL-only rule


def test_the_gradient_stays_finite_for_a_confident_student(segment):
    """With probabilities that underflow to zero, the student's entropy computed as
    p log p of the probabilities (`xlogy`) would put 0 x infinity into the gradient; PPO's
    formula from the log-probabilities keeps it finite, with or without a coefficient."""
    import types

    _, runner, learner, prepared = segment
    trajectory = lc.whole(runner)
    shape = trajectory.log_probability.shape + (42,)
    logits = torch.zeros(shape)
    logits[..., 3] = 1e4  # every other action's probability underflows to 0
    logits.requires_grad_(True)
    policy_term, entropy_term, _ = learner._policy(
        types.SimpleNamespace(policy_logits=logits),
        trajectory,
        prepared["derived"],
        dict(policy=trajectory.alive[..., : shape[2]].sum().float()),
        prepared["scale"],
    )
    for coefficient in (0.0, -0.0081):  # BC's, and PPO's entropy bonus
        (gradient,) = torch.autograd.grad(
            policy_term + coefficient * entropy_term.sum(), logits, retain_graph=True
        )
        assert torch.isfinite(gradient).all(), coefficient
