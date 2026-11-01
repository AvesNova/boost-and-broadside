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
    # The update labels a whole segment at once; the CPU's vectorized kernels round the
    # tail of a batch apart from its body, so a batch of another size may differ in the
    # last bit.
    labels = bc.teacher_distribution(TEACHER, trajectory.obs)
    torch.testing.assert_close(labels, decided, rtol=0, atol=2.5e-7)


def test_the_teacher_temperature():
    torch.manual_seed(0)
    q = torch.softmax(torch.randn(500, 42) * 3.0, -1)
    q[:, 5] = 0.0  # a structural zero
    q = q / q.sum(-1, keepdim=True)
    assert torch.equal(bc.labels(q, bc.TeacherConfig()), q)
    hotter = bc.labels(q, bc.TeacherConfig(temperature=2.0))
    assert (hotter[:, 5] == 0.0).all()
    assert bc.mean_entropy(q, bc.TeacherConfig(2.0)) > bc.mean_entropy(q, bc.TeacherConfig())
    for target, floor in ((1.0, 0.0), (2.5, 0.0), (3.5, 0.0), (3.7, 0.05)):
        chosen = bc.calibrate(q, target, floor)
        assert bc.mean_entropy(q, chosen) == pytest.approx(target, abs=0.01)
    with pytest.raises(ValueError):
        bc.calibrate(q, 3.73)  # above what the temperature alone reaches with a zero


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


def test_alive_mask_reads_the_own_rows(segment):
    _, runner, _, _ = segment
    trajectory: Trajectory = lc.whole(runner)
    S = trajectory.logp.shape[-1]
    torch.testing.assert_close(trajectory.alive[..., :S].float(),
                               (trajectory.obs.ships[..., :S, SHIP["alive"]] > 0.5).float())
