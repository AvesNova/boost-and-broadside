"""Behaviour cloning (scripted spec B6): the loss, the labels, the temperature and the stop
rule, on a real stored Yemong segment on the CPU."""

import dataclasses
import functools

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


def test_the_opponent_plays_the_natural_distribution_whatever_the_temperature():
    from bnb.contracts import GameConfig
    from bnb.runtime.cloning import LINEUP
    from bnb.runtime.train import SCRIPTED, TrainConfig, Trainer

    cfg = TrainConfig(game=GameConfig(2, 2, 1200.0), pools=LINEUP, num_envs=4, length=8,
                      model="relational", relational_hidden=16, compile=False, host_store=False,
                      learner=bc.config(minibatches=2), objective="bc",
                      teacher=bc.TeacherConfig(temperature=3.0, floor=0.2))
    trainer = Trainer(cfg, lc.CPU)
    opponent = trainer.controllers[SCRIPTED]
    assert opponent.scripted_probability == 1.0
    view = trainer.rollout._view(trainer.rollout.carry, SCRIPTED, Scripted.reads)
    torch.testing.assert_close(opponent.action_distribution(view), TEACHER(view))
    assert trainer.ppo.teacher_config == cfg.teacher
    assert trainer.rollout.reads == ("ships", "globals", "zones")


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
    assert _run(bc.StopRule(config), 200, 100, 0.5, falling) == (4, "score")  # 100 games
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
    assert type(trainer.ppo) is Learner and trainer.cfg.learner.policy == 27.0  # no cloning
    row = trainer.update()
    assert "bc_kl" not in row and "approximate_kl" in row


def test_the_hand_off_round_trips_through_a_policy_file(tmp_path):
    """Section 11's interface (runs spec 3): save_policy and load_policy. Open until that
    section merges."""
    checkpoint = pytest.importorskip("bnb.runs.checkpoint",
                                     reason="section 11 (bnb.runs) has not merged")
    from bnb.runtime.cloning import LINEUP
    from bnb.runtime.train import Trainer

    student = Trainer(_tiny("bc", LINEUP), lc.CPU)
    student.update()
    checkpoint.save_policy(tmp_path / "clone", student.model)
    loaded = checkpoint.load_policy(tmp_path / "clone", lc.CPU)
    for a, b in zip(loaded.state_dict().values(), student.model.state_dict().values(),
                    strict=True):
        assert torch.equal(a, b)


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
