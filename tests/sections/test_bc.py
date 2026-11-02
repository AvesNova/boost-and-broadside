"""Behaviour cloning (scripted spec B6): the loss, the labels, the temperature and the stop
rule, on a real stored Yemong segment on the CPU."""

import dataclasses
import functools
import json
import sys

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases
from scripted_cases import CHARACTERISTICS

from bnb.contracts import Obs, Pool, Trajectory
from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Scripted, distribution
from bnb.learn import bc
from bnb.learn.rewards import RewardConfig
from bnb.runtime.rollout import Rollout

POOLS = (Pool("self_play", 0.5, ("learner", "learner"), 0.5),
         Pool("scripted", 0.5, ("learner", "scripted"), 0.5, mode="frontline"))
TEACHER = functools.partial(distribution, characteristics=CHARACTERISTICS,
                            parameters=Scripted(CHARACTERISTICS).parameters)


def _cloning(net, runner: Rollout, teacher_config=bc.TeacherConfig(), **config):
    views = runner.store.views
    config = dict(dict(minibatches=1, micro_batch_rows=views * runner.rules.num_ships), **config)
    return bc.BehaviourCloning(net, bc.config(**config), runner.rules, runner.carry["types"],
                               RewardConfig().outcome, lc.CPU, TEACHER, teacher_config,
                               tuple(pool.name for pool in POOLS),
                               runner.lineup.pool_of["learner"])


@pytest.fixture(scope="module")
def segment():
    net = cases.model()
    controllers = {"learner": cases.controller(net, cases.SHORT_SQUAD),
                   "scripted": Scripted(CHARACTERISTICS)}
    runner = Rollout(cases.SHORT_SQUAD, POOLS, controllers, "learner", 8, 32, lc.CPU,
                     compile=False, extra_reads=Scripted.reads)
    for _ in range(2):
        runner.run()
    learner = _cloning(net, runner)
    prepared = learner.prepare(runner.store, [runner.bootstrap()])
    return net, runner, learner, prepared


def test_the_bc_loss_matches_an_independent_computation(segment):
    net, runner, learner, prepared = segment
    trajectory = lc.whole(runner)
    # Mark a third of the own rows dead, so the mask matters.
    marked = trajectory.alive.clone()
    own = trajectory.logp.shape[-1]
    marked[..., :own] &= torch.rand(marked[..., :own].shape,
                                    generator=torch.Generator().manual_seed(1)) > 0.33
    trajectory = dataclasses.replace(trajectory, alive=marked)
    with torch.no_grad():
        unrolled = learner.unroll(trajectory)
    alive = trajectory.alive[..., :own]
    tokens = alive.sum()
    assert 0 < tokens < alive.numel()
    cross, kl = torch.zeros(()), torch.zeros(())
    for tick in range(trajectory.done.shape[0]):  # the teacher tick by tick, on its views
        obs = Obs(**{name: None if value is None else value[tick]
                     for name, value in dataclasses.asdict(trajectory.obs).items()})
        q = distribution(obs, CHARACTERISTICS, Scripted(CHARACTERISTICS).parameters)
        floor = bc.TeacherConfig().floor  # the default labels: the uniform floor, T = 1
        q = (1.0 - floor) * q + floor / 42
        log_student = torch.log_softmax(unrolled.policy_logits[tick], -1)
        per_token = -(q * log_student).sum(-1)
        entropy = -torch.special.xlogy(q, q).sum(-1)
        cross = cross + per_token[alive[tick]].sum()
        kl = kl + (per_token - entropy)[alive[tick]].sum()
    derived = {name: value for name, value in prepared["derived"].items()}
    policy_term, _, statistics = learner._policy(unrolled, trajectory, derived,
                                                 dict(policy=tokens.float()), prepared["scale"])
    torch.testing.assert_close(policy_term, cross / tokens, rtol=1e-5, atol=1e-6)
    report = learner.report(statistics.tolist())
    assert report["bc_kl"] == pytest.approx(float(kl / tokens), rel=1e-5)
    assert set(report) >= {"bc_kl/self_play", "teacher_entropy/scripted", "student_entropy"}


def test_micro_batches_accumulate_to_the_unsplit_minibatch(segment):
    net, runner, learner, prepared = segment
    views = torch.arange(runner.store.views)

    def gradient(size: int):
        lc.with_micro_batch(learner, size)
        terms, _, statistics = learner.gradient(runner.store, prepared, views)
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

    def step(self, carry, obs, done):
        self.decisions.append(distribution(obs, self.characteristics, self.parameters))
        return super().step(carry, obs, done)


def test_the_update_time_labels_are_the_rollout_time_distribution():
    recording = Recording()
    pools = (Pool("self_play", 1.0, ("learner", "learner"), 0.5, mode="frontline"),)
    runner = Rollout(cases.SHORT_FRONTLINE, pools, {"learner": recording}, "learner", 4, 16,
                     lc.CPU, compile=False)
    trajectory = runner.run()
    decided = torch.stack(recording.decisions[1:])  # the first is the rollout's one-row probe
    for tick in range(trajectory.done.shape[0]):  # the stored views give the same decision
        stored = Obs(ships=trajectory.obs.ships[tick], fields=None,
                     globals=trajectory.obs.globals[tick], zones=trajectory.obs.zones[tick])
        assert torch.equal(TEACHER(stored), decided[tick])
    # Per tick the labels are exact (above). The update labels a whole segment as one
    # batch, and the CPU's vectorized kernels round the tail of a batch apart from its body,
    # so a batch of another shape may differ at the ulp level: measured at most 1.2e-7 (on
    # 0.15% of entries). The tolerance is 2.5e-7, about two float32 ulps at 1.
    labels = bc.teacher_distribution(TEACHER, trajectory.obs)
    torch.testing.assert_close(labels, decided, rtol=0, atol=2.5e-7)


def test_the_teacher_temperature():
    torch.manual_seed(0)
    q = torch.softmax(torch.randn(500, 42) * 3.0, -1)
    q[:, 5] = 0.0  # a structural zero
    q = q / q.sum(-1, keepdim=True)
    assert torch.equal(bc.labels(q, bc.TeacherConfig(1.0, 0.0)), q)
    hotter = bc.labels(q, bc.TeacherConfig(temperature=2.0, floor=0.0))
    assert (hotter[:, 5] == 0.0).all()
    assert (bc.mean_entropy(q, bc.TeacherConfig(2.0, 0.0))
            > bc.mean_entropy(q, bc.TeacherConfig(1.0, 0.0)))
    for target, floor in ((1.0, 0.0), (2.5, 0.0), (3.5, 0.0), (3.7, 0.05)):
        chosen = bc.calibrate(q, target, floor)
        assert bc.mean_entropy(q, chosen) == pytest.approx(target, abs=0.01)
    with pytest.raises(ValueError):
        bc.calibrate(q, 3.73, 0.0)  # above what the temperature alone reaches with a zero


def test_every_label_gives_every_action_a_probability_above_zero(segment):
    """The default floor, on the stored observations (dead ships' certain no-op included),
    at T = 1 and at a cold temperature."""
    _, runner, _, _ = segment
    trajectory = lc.whole(runner)
    distributions = bc.teacher_distribution(TEACHER, trajectory.obs)
    assert (distributions == 0.0).any()  # the teacher itself has zeros
    for temperature in (1.0, 0.25):
        labels = bc.labels(distributions, bc.TeacherConfig(temperature=temperature))
        assert bool((labels > 0.0).all()), temperature
        assert float(labels.min()) >= 0.99 * bc.FLOOR / 42
    assert bc.TeacherConfig().floor == bc.FLOOR > 0.0


def _scripted_seat(objective: str, teacher: bc.TeacherConfig = bc.TeacherConfig(),
                   segments: int = 0):
    """A small CPU trainer with BC's lineup, its scripted seat's controller and that seat's
    view after `segments` segments of 8 ticks (from 2 on, the scripted agent is no longer
    certain of every action)."""
    from bnb.contracts import GameConfig
    from bnb.runtime.cloning import LINEUP
    from bnb.runtime.train import SCRIPTED, TrainConfig, Trainer

    learner = bc.config(minibatches=2) if objective == "bc" else lc.LearnerConfig(minibatches=2)
    cfg = TrainConfig(game=GameConfig(2, 2, 1200.0), pools=LINEUP, num_envs=4, length=8,
                      model="relational", relational_hidden=16, compile=False, host_store=False,
                      learner=learner, objective=objective, teacher=teacher)
    trainer = Trainer(cfg, lc.CPU)
    for _ in range(segments):
        trainer.rollout.run()
    view = trainer.rollout._view(trainer.rollout.carry, SCRIPTED, Scripted.reads)
    return trainer, trainer.controllers[SCRIPTED], view


def _repeated(view: Obs, times: int) -> Obs:
    """The view's rows (R, ...) repeated `times` times: (times R, ...)."""
    parts = {field.name: getattr(view, field.name) for field in dataclasses.fields(view)}
    return Obs(**{name: None if value is None else value.repeat(times, *[1] * (value.dim() - 1))
                  for name, value in parts.items()})


def test_the_scored_opponent_plays_the_labels_distribution():
    """BC's scripted seat samples the labels' distribution (floor and temperature applied):
    its stated distribution is the labels', and its draws' frequencies over 20000 copies of
    each view match it within five standard errors."""
    from bnb.contracts.action import NUM_JOINT, to_joint

    teacher = bc.TeacherConfig(temperature=3.0, floor=0.2)
    trainer, opponent, view = _scripted_seat("bc", teacher, segments=2)
    expected = bc.labels(TEACHER(view), teacher)  # (R, S, 42)
    untempered = bc.labels(TEACHER(view), dataclasses.replace(teacher, temperature=1.0))
    assert float((expected - untempered).abs().max()) > 0.05  # the temperature matters here
    torch.testing.assert_close(opponent.action_distribution(view), expected)
    assert trainer.ppo.teacher_config == teacher
    assert trainer.rollout.reads == ("ships", "globals", "zones")
    times = 20_000
    torch.manual_seed(0)
    action, _, _ = opponent.step(None, _repeated(view, times), None)
    joint = to_joint(action).long().unflatten(0, (times, -1))  # (times, R, S)
    frequency = torch.nn.functional.one_hot(joint, NUM_JOINT).float().mean(0)
    error = (expected * (1 - expected) / times).sqrt()
    assert bool(((frequency - expected).abs() <= 5 * error + 1e-4).all()), \
        float((frequency - expected).abs().max())
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
    assert reinforcement_opponent.scripted_probability == 1.0


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
    falling = lambda update: 2.0 * 0.95 ** update  # noqa: E731 - still improving 5% per update
    # 0.8 over 100 games: a lower bound of 0.72 (0.5 would give 0.40, below the target)
    assert _run(bc.StopRule(config), 200, 100, 0.8, falling) == (4, "score")  # 100 games
    assert _run(bc.StopRule(config), 200, 100, 0.2, falling) == (99, "budget")
    flat = lambda update: 1.0 + 0.5 * 0.5 ** update  # noqa: E731 - converges, then plateaus
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


def test_the_score_stop_reads_the_windows_lower_bound_trimmed_like_the_score():
    """Window 100: 30 wins and 30 losses, then 20 wins, 40 draws and 10 losses. The window
    keeps half the first update, so its 100 games are 35 wins, 40 draws and 25 losses; the
    bound is their mean less two of their standard errors (sample deviation / 10)."""
    config = bc.StopConfig(score_window=100, score_target=0.47, budget_steps=10**9)
    rule = bc.StopRule(config)
    assert rule.observe(1, 30, 0, 30, 1.0) is None and rule.lower_bound is None
    games = torch.tensor([1.0] * 35 + [0.5] * 40 + [0.0] * 25, dtype=torch.float64)
    error = float(games.std() / 10)
    assert rule.observe(2, 20, 40, 10, 1.0) == "score"  # 0.4728 >= 0.47
    assert rule.score == pytest.approx(0.55)
    assert rule.standard_error == pytest.approx(error)
    assert rule.lower_bound == pytest.approx(0.55 - 2 * error)
    assert rule.lower_bound == pytest.approx(0.472802, abs=1e-6)
    tighter = bc.StopRule(dataclasses.replace(config, score_target=0.48))
    tighter.observe(1, 30, 0, 30, 1.0)
    assert tighter.observe(2, 20, 40, 10, 1.0) is None  # the score 0.55 alone would stop


def test_a_window_at_the_target_does_not_stop_and_one_clearly_above_does():
    config = bc.StopConfig(score_window=100, score_target=0.45, budget_steps=10**9)
    at_target = bc.StopRule(config)
    assert at_target.observe(1, 45, 0, 55, 1.0) is None
    assert at_target.score == pytest.approx(0.45) and at_target.lower_bound < 0.45
    above = bc.StopRule(config)
    assert above.observe(1, 70, 0, 30, 1.0) == "score"
    assert above.lower_bound == pytest.approx(0.7 - 2 * (0.21 / 99) ** 0.5)
    certain = bc.StopRule(config)  # no spread: the bound is the score
    assert certain.observe(1, 0, 100, 0, 1.0) == "score" and certain.lower_bound == 0.5


def test_alive_mask_reads_the_own_rows(segment):
    _, runner, _, _ = segment
    trajectory: Trajectory = lc.whole(runner)
    S = trajectory.logp.shape[-1]
    torch.testing.assert_close(trajectory.alive[..., :S].float(),
                               (trajectory.obs.ships[..., :S, SHIP["alive"]] > 0.5).float())


# ------------------------------------------------------------------ the hand-off


def _tiny(objective: str, pools):
    from bnb.contracts import GameConfig
    from bnb.learn.ppo import LearnerConfig
    from bnb.runtime.train import TrainConfig

    learner = bc.config(minibatches=2) if objective == "bc" else LearnerConfig(minibatches=2)
    return TrainConfig(game=GameConfig(2, 2, 1200.0), pools=pools, num_envs=8, length=8,
                       model="relational", relational_hidden=16, compile=False,
                       host_store=False, learner=learner, objective=objective)


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
    for module in (trainer.model, trainer.league):
        for (name, a), b in zip(module.state_dict().items(), student.model.state_dict().values(),
                                strict=True):
            assert torch.equal(a, b), name
    assert float(trainer.ppo.return_scale.scale()) == float(student.ppo.return_scale.scale())
    assert float(student.ppo.return_scale.high) != 0.0
    assert not trainer.ppo.optimizer.state  # fresh
    assert type(trainer.ppo) is Learner and trainer.config.learner.policy == 27.0  # no cloning
    row = trainer.update()
    assert "bc_kl" not in row and "approximate_kl" in row


# The tiny BC run of the command line: 2v2 relational, 16 environments x 16 ticks per
# update, stopped by its budget after two updates.
BC_OPTIONS = ("--device", "cpu", "--model", "relational", "--ships-per-team", "2",
              "--num-fields", "2", "--arena-radius", "1500", "--num-envs", "16",
              "--objective", "bc", "--bc-budget-steps", str(2 * 16 * 16),
              "--set", "length=16", "--set", "compile=false", "--set", "learner.minibatches=2",
              "--set", "learner.epochs=1", "--set", "relational_hidden=16",
              "--set", "game.episode_s=4.0")


def _bnb(monkeypatch, *arguments: str) -> None:
    from bnb import cli

    monkeypatch.setattr(sys, "argv", ["bnb", *arguments])
    with torch.random.fork_rng():
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
    _bnb(monkeypatch, "train", *BC_OPTIONS[:12], "--set", "length=16", "--set", "compile=false",
         "--set", "relational_hidden=16", "--set", "game.episode_s=4.0", "--updates", "0",
         "--init-from", str(run.policy_path), "--name", "rl", "--runs-directory", runs)
    rl = Run.open(tmp_path / "runs", "rl")
    assert rl.record["parent"] == dict(policy=str(run.policy_path.resolve()))
    final = checkpoint.load_policy(rl.checkpoint_at(), lc.CPU)
    for name, value in clone.state_dict().items():
        assert torch.equal(final.state_dict()[name], value), name


@pytest.mark.parametrize("start", [("--resume", "clone"), ("--from", "clone"),
                                   ("--init-from", "somewhere")])
def test_a_bc_run_starts_fresh(monkeypatch, tmp_path, start):
    with pytest.raises(SystemExit, match="a BC run starts fresh"):
        _bnb(monkeypatch, "train", *BC_OPTIONS, *start, "--runs-directory", str(tmp_path))


def test_the_gradient_stays_finite_for_a_confident_student(segment):
    """With probabilities that underflow to zero, the student's entropy computed as
    p log p of the probabilities (`xlogy`) would put 0 x infinity into the gradient; PPO's
    formula from the log-probabilities keeps it finite, with or without a coefficient."""
    import types

    _, runner, learner, prepared = segment
    trajectory = lc.whole(runner)
    shape = trajectory.logp.shape + (42,)
    logits = torch.zeros(shape)
    logits[..., 3] = 1e4  # every other action's probability underflows to 0
    logits.requires_grad_(True)
    policy_term, entropy_term, _ = learner._policy(
        types.SimpleNamespace(policy_logits=logits), trajectory, prepared["derived"],
        dict(policy=trajectory.alive[..., :shape[2]].sum().float()), prepared["scale"])
    for coefficient in (0.0, -0.0081):  # BC's, and PPO's entropy bonus
        gradient, = torch.autograd.grad(policy_term + coefficient * entropy_term, logits,
                                        retain_graph=True)
        assert torch.isfinite(gradient).all(), coefficient
