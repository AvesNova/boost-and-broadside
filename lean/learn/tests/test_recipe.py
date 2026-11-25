"""The training recipe (docs/rewrite/core-decisions.md, 7.1 to 7.7): the BC stage on the
teacher's full distribution, the shrinking teacher term and its gate, weight averages, the
optimiser and its warm-up, the KL monitor and the advantage scaling."""

import copy
import dataclasses
import math

import pytest
import torch

from lean import model
from lean.contract import JOINT, LEVEL_NAMES, join_action
from lean.learn import (
    LearnConfig,
    TeacherTerm,
    WeightAverages,
    collect,
    learning_rate,
    load_checkpoint,
    make_optimiser,
    policy_weights,
    save_checkpoint,
    update,
)
from lean.learn.returns import advantage_shares, advantage_sums
from lean.learn.rollout import TEACHER_WIDTH, teacher_joint
from lean.learn.tests.setting import small
from lean.learn.training import teacher_cross_entropy


def rollout_of(ticks: int = 8, **overrides):
    config, game_config, policy, opponent, state, carry, generator = small(**overrides)
    rollout = collect(config, game_config, policy, opponent, state, carry, ticks, generator)
    return config, policy, rollout, generator


# ------------------------------------------------------------------ the teacher's distribution


def test_the_teacher_joint_is_the_product_of_its_marginals_in_the_contracts_order():
    generator = torch.Generator().manual_seed(0)
    stick = torch.softmax(torch.randn(5, 7, generator=generator), -1)
    power = torch.softmax(torch.randn(5, 3, generator=generator), -1)
    fire = torch.rand(5, 1, generator=generator)
    joint = teacher_joint(torch.cat([stick, power, fire], -1))
    assert joint.shape == (5, JOINT) and torch.allclose(joint.sum(-1), torch.ones(5))
    for s in range(7):
        for p in range(3):
            for f in range(2):
                index = int(join_action(torch.tensor([s, p, f])))
                expected = stick[:, s] * power[:, p] * (fire[:, 0] if f else 1.0 - fire[:, 0])
                assert torch.allclose(joint[:, index], expected)


def test_the_rollout_records_the_teachers_full_distribution():
    """Rollout.teacher holds the scripted agent's marginals for every ship of the seat's
    environment, each a distribution; on own rows the joint is a proper distribution."""
    _, _, rollout, _ = rollout_of()
    assert rollout.teacher.shape[-1] == TEACHER_WIDTH
    own = rollout.observation.own
    joint = teacher_joint(rollout.teacher[own])
    assert torch.allclose(joint.sum(-1), torch.ones(joint.shape[0]), atol=1e-5)
    assert (joint.max(-1).values < 0.99).all()  # soft targets: the floors leave no action certain


def test_the_bc_loss_is_the_cross_entropy_on_the_soft_targets():
    """teacher_cross_entropy is -sum p_teacher log q over the 42 joint actions; its minimum
    over the policy is the teacher's entropy (KL zero), reached at the teacher itself."""
    generator = torch.Generator().manual_seed(1)
    marginals = torch.cat(
        [
            torch.softmax(torch.randn(6, 7, generator=generator), -1),
            torch.softmax(torch.randn(6, 3, generator=generator), -1),
            torch.rand(6, 1, generator=generator),
        ],
        -1,
    )
    target = teacher_joint(marginals)
    log_policy = torch.log_softmax(torch.randn(6, JOINT, generator=generator), -1)
    cross, entropy = teacher_cross_entropy(log_policy, marginals)
    assert torch.allclose(cross, -(target * log_policy).sum(-1))
    assert torch.allclose(entropy, -(target * target.log()).sum(-1))
    at_teacher, _ = teacher_cross_entropy(target.log(), marginals)
    assert torch.allclose(at_teacher, entropy, atol=1e-5) and (cross > entropy).all()


# ------------------------------------------------------------------ the BC stage


def heads(policy) -> dict[str, list[torch.Tensor]]:
    """Each head's own last layer: policy and value (the head group), next state and hit."""
    return {
        "policy": list(policy.heads.lasts[0].parameters()),
        "value": list(policy.heads.lasts[1].parameters()),
        "next_state": list(policy.auxiliary_heads.lasts[0].parameters()),
        "hit": list(policy.auxiliary_heads.lasts[1].parameters()),
    }


@pytest.mark.parametrize("cloning", [True, False])
def test_the_bc_stage_trains_every_head_and_ignores_the_advantages(cloning, monkeypatch):
    """A cloning update moves the policy, value, next-state and hit heads; with no surrogate
    its step does not depend on the advantages (the same step with them scaled by 100 after
    their normalisation), where an RL step does."""
    from lean.learn import training

    config, policy, rollout, _ = rollout_of(16)
    normalise = training.normalise_advantages
    steps = []
    for scale in (1.0, 100.0):
        monkeypatch.setattr(
            training,
            "normalise_advantages",
            lambda total, mask, scale=scale: scale * normalise(total, mask),
        )
        twin = copy.deepcopy(policy)
        before = {
            name: [p.detach().clone() for p in params] for name, params in heads(twin).items()
        }
        metrics = update(
            config,
            twin,
            make_optimiser(config, twin),
            rollout,
            cloning=cloning,
            generator=torch.Generator().manual_seed(3),
        )
        moved = {
            name: any(not torch.equal(a, b) for a, b in zip(before[name], params, strict=True))
            for name, params in heads(twin).items()
        }
        assert all(moved.values()), moved
        steps.append([p.detach() for p in twin.parameters()])
    same = all(torch.equal(first, second) for first, second in zip(*steps, strict=True))
    assert same == cloning
    if cloning:
        assert metrics["teacher_weight"] == config.bc_coefficient and metrics["bc"] > 0


def test_cloning_updates_reduce_the_kl_to_the_teacher_and_the_value_loss():
    config, game_config, policy, opponent, state, carry, generator = small(
        epochs=2, learning_rate=1e-3
    )
    optimiser = make_optimiser(config, policy)
    history, steps = [], 0
    for _ in range(6):
        rollout = collect(config, game_config, policy, opponent, state, carry, 16, generator)
        state, carry = rollout.state, rollout.carry
        history.append(
            update(
                config,
                policy,
                optimiser,
                rollout,
                cloning=True,
                step_index=steps,
                generator=generator,
            )
        )
        steps += history[-1]["adam_steps"]
    assert history[-1]["bc_kl"] < history[0]["bc_kl"]
    assert history[-1]["value"] < history[0]["value"]
    assert all(metrics["surrogate"] == metrics["surrogate"] for metrics in history)


# ------------------------------------------------------------------ the teacher term in RL


def test_the_teacher_term_shrinks_every_update_and_switches_off_at_the_gate_for_good():
    config = LearnConfig(
        teacher_coefficient=0.5, teacher_decay=0.9, teacher_gate=0.45, teacher_cutoff=0.001
    )
    term = TeacherTerm()
    weights = [term.weight(config, 0.0) for _ in range(3)]
    assert weights == pytest.approx([0.5, 0.45, 0.405])
    assert term.weight(config, 0.44) == pytest.approx(0.5 * 0.9**3)  # below the gate: still on
    assert term.weight(config, 0.45) == 0.0  # the gate reached
    assert (
        term.off and term.weight(config, 0.0) == 0.0
    )  # and off for good, whatever the win rate after


def test_the_teacher_term_ends_by_its_decay_alone():
    config = LearnConfig(teacher_coefficient=0.5, teacher_decay=0.5, teacher_cutoff=0.1)
    term = TeacherTerm()
    assert [term.weight(config, 0.0) for _ in range(4)] == pytest.approx([0.5, 0.25, 0.125, 0.0])
    assert term.off


def test_the_gate_reads_zero_until_a_scripted_game_is_played():
    """The policy is assumed to lose every game at the start: the league's win rate against
    the scripted agent reads 0, not the sampler's 0.5 prior, until a game has ended."""
    _, _, _, opponent, _, _, _ = small(league=True)
    league = opponent.league
    assert league.scripted_win_rate() == 0.0 and float(league.win_rates()[0]) == 0.5
    league.record(
        torch.tensor([0]), torch.tensor([True]), torch.tensor([True]), torch.tensor([False])
    )
    assert league.scripted_win_rate() == 1.0


def test_an_rl_update_weighs_the_teacher_term_as_given():
    config, policy, rollout, _ = rollout_of(
        minibatches=1
    )  # one step: its loss is measured before it
    results = []
    for weight in (0.0, 1.0):
        twin = copy.deepcopy(policy)
        metrics = update(
            config,
            twin,
            make_optimiser(config, twin),
            rollout,
            teacher_weight=weight,
            generator=torch.Generator().manual_seed(4),
        )
        results.append(metrics)
    assert results[0]["teacher_weight"] == 0.0 and results[1]["teacher_weight"] == 1.0
    assert results[1]["loss"] == pytest.approx(results[0]["loss"] + results[1]["bc"], rel=1e-4)


# ------------------------------------------------------------------ weight averages


def test_the_ema_and_the_uniform_average_follow_their_formulas():
    config = LearnConfig(ema_decay=0.8, warmup_steps=10)
    policy = torch.nn.Linear(3, 2)
    start = [p.detach().clone() for p in policy.parameters()]
    averages = WeightAverages(policy, config)
    history = []
    for update_index in range(5):
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.add_(1.0 + update_index)
        history.append([p.detach().clone() for p in policy.parameters()])
        averages.update(policy, adam_steps=5 * (update_index + 1))  # warm-up ends after update 1
    for index, name in enumerate(dict(policy.named_parameters())):
        ema = start[index]
        for weights in history:
            ema = 0.8 * ema + 0.2 * weights[index]
        assert torch.allclose(averages.ema[name], ema, atol=1e-6)
        uniform = torch.stack([weights[index] for weights in history[1:]]).mean(0)  # updates 1..4
        assert torch.allclose(averages.uniform[name], uniform, atol=1e-6)
    assert averages.count == 4


def test_averages_are_never_fed_back_into_training():
    """Two runs of the same updates, one keeping averages after each: the live weights are
    identical; the averages hold their own tensors (changing the live weights leaves them)."""
    config, policy, rollout, _ = rollout_of(epochs=2)
    lives = []
    for keep in (False, True):
        twin = copy.deepcopy(policy)
        optimiser = make_optimiser(config, twin)
        averages = WeightAverages(twin, config) if keep else None
        generator = torch.Generator().manual_seed(5)
        steps = 0
        for _ in range(2):
            steps += update(
                config, twin, optimiser, rollout, step_index=steps, generator=generator
            )["adam_steps"]
            if keep:
                averages.update(twin, steps)
        lives.append([p.detach().clone() for p in twin.parameters()])
    for first, second in zip(*lives, strict=True):
        assert torch.equal(first, second)
    snapshot = {name: value.clone() for name, value in averages.ema.items()}
    with torch.no_grad():
        for parameter in twin.parameters():
            parameter.add_(1.0)
    assert all(torch.equal(snapshot[name], averages.ema[name]) for name in snapshot)


def test_a_checkpoint_holds_the_live_weights_and_both_averages(tmp_path):
    config, policy, _, _ = rollout_of()
    averages = WeightAverages(policy, dataclasses.replace(config, warmup_steps=0))
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(0.5)
    averages.update(policy, adam_steps=1)
    save_checkpoint(tmp_path / "bc.pt", policy, averages, config, stage="bc")
    checkpoint = load_checkpoint(tmp_path / "bc.pt")
    assert (
        checkpoint["stage"] == "bc" and checkpoint["learn_config"]["ema_decay"] == config.ema_decay
    )
    for which in ("live", "ema", "uniform"):
        restored = model.Policy(checkpoint["model_config"])
        restored.load_state_dict(policy_weights(checkpoint, which))
        expected = averages.weights(which, policy)
        assert all(
            torch.equal(value, expected[name]) for name, value in restored.state_dict().items()
        ), which
    name = next(iter(averages.ema))
    assert not torch.equal(
        policy_weights(checkpoint, "ema")[name], policy_weights(checkpoint, "live")[name]
    )


# ------------------------------------------------------------------ the optimiser


def test_the_optimiser_is_fused_adam_without_weight_decay():
    policy = model.Policy(model.ModelConfig())
    optimiser = make_optimiser(LearnConfig(), policy)
    assert isinstance(optimiser, torch.optim.Adam) and not isinstance(optimiser, torch.optim.AdamW)
    group = optimiser.param_groups[0]
    assert group["fused"] and group["weight_decay"] == 0 and group["eps"] == 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_optimiser_is_fused_on_cuda():
    policy = model.Policy(model.ModelConfig()).cuda()
    optimiser = make_optimiser(LearnConfig(), policy)
    assert optimiser.param_groups[0]["fused"]
    sum(parameter.sum() for parameter in policy.parameters()).backward()
    optimiser.step()


def test_a_step_with_a_non_finite_gradient_is_skipped_and_counted():
    """A NaN in the loss (here through its value coefficient) makes every gradient norm
    non-finite: every step is skipped (weights and Adam's state unchanged) and counted."""
    config, policy, rollout, generator = rollout_of()
    broken = dataclasses.replace(config, value_coefficient=float("nan"))
    optimiser = make_optimiser(broken, policy)
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    metrics = update(broken, policy, optimiser, rollout, generator=generator)
    assert metrics["skipped_steps"] == metrics["adam_steps"] == config.minibatches * config.epochs
    assert all(torch.equal(old, new) for old, new in zip(before, policy.parameters(), strict=True))
    assert all(float(state["step"]) == 0 for state in optimiser.state.values())
    healthy = update(config, policy, optimiser, rollout, generator=generator)
    assert healthy["skipped_steps"] == 0 and math.isfinite(healthy["gradient_norm"])
    assert all(float(state["step"]) == healthy["adam_steps"] for state in optimiser.state.values())


def test_update_refuses_an_optimiser_that_cannot_skip_without_a_host_read():
    config, policy, rollout, generator = rollout_of(4)
    with pytest.raises(ValueError):
        update(config, policy, torch.optim.Adam(policy.parameters()), rollout, generator=generator)


def test_the_learning_rate_warms_up_over_its_adam_steps_then_stays_constant():
    config = LearnConfig()
    warmup, peak = config.warmup_steps, config.learning_rate
    assert warmup % 2 == 0  # the half-way step below
    assert learning_rate(config, 0) == pytest.approx(peak / warmup)
    assert learning_rate(config, warmup // 2 - 1) == pytest.approx(peak / 2)
    assert (
        learning_rate(config, warmup - 1)
        == learning_rate(config, warmup)
        == learning_rate(config, 10**7)
        == peak
    )


def test_an_update_sets_the_rate_of_each_adam_step():
    config, policy, rollout, generator = rollout_of(epochs=2, warmup_steps=100)
    optimiser = make_optimiser(config, policy)
    metrics = update(config, policy, optimiser, rollout, step_index=10, generator=generator)
    assert metrics["adam_steps"] == 4  # 2 epochs x 2 minibatches
    assert (
        optimiser.param_groups[0]["lr"]
        == metrics["learning_rate"]
        == pytest.approx(learning_rate(config, 13))
    )


# ------------------------------------------------------------------ the KL monitor and the advantag


def test_the_kl_monitor_logs_each_epoch_and_warns_without_stopping():
    """Old log-probabilities shifted by 1 nat put every epoch's approximate KL near e - 2: the
    warning is raised and every epoch still runs all its steps (no early stop)."""
    config, policy, rollout, generator = rollout_of(epochs=3)
    calm = update(
        config, copy.deepcopy(policy), make_optimiser(config, policy), rollout, generator=generator
    )
    assert [name for name in calm if name.startswith("approximate_kl_epoch_")] == [
        f"approximate_kl_epoch_{k}" for k in range(3)
    ]
    assert calm["approximate_kl_epoch_0"] < 1e-6 and calm["kl_warning"] == 0.0
    rollout.log_probability = rollout.log_probability - 1.0
    shifted = update(config, policy, make_optimiser(config, policy), rollout, generator=generator)
    assert shifted["approximate_kl_epoch_0"] == pytest.approx(math.e - 2.0, rel=0.05)
    assert (
        shifted["kl_warning"] == 1.0 and shifted["adam_steps"] == config.epochs * config.minibatches
    )


def test_each_levels_share_of_the_advantage_variance_adds_up_to_one():
    generator = torch.Generator().manual_seed(6)
    advantage = torch.randn(40, 6, 4, 3, generator=generator) * torch.tensor([1.0, 2.0, 0.5])
    advantage[..., 2] += 0.3 * advantage[..., 0]  # correlated levels
    mask = torch.rand(40, 6, 4, generator=generator) < 0.8
    halves = advantage_sums(advantage[:20], mask[:20]) + advantage_sums(advantage[20:], mask[20:])
    shares, deviation = advantage_shares(halves)
    chosen, total = advantage[mask], advantage.sum(-1)[mask]
    centred = total - total.mean()
    expected = ((chosen - chosen.mean(0)) * centred[:, None]).mean(0) / centred.pow(2).mean()
    assert torch.allclose(shares, expected, atol=1e-4) and shares.sum() == pytest.approx(
        1.0, abs=1e-5
    )
    assert deviation == pytest.approx(float(total.std()), rel=1e-4)


def test_an_update_logs_the_advantage_shares_and_the_entropy_fraction():
    config, policy, rollout, generator = rollout_of(16)
    metrics = update(config, policy, make_optimiser(config, policy), rollout, generator=generator)
    shares = [metrics[f"advantage_share_{name}"] for name in LEVEL_NAMES]
    assert sum(shares) == pytest.approx(1.0, abs=1e-4) and metrics["advantage_std"] > 0
    assert metrics["entropy_fraction"] == pytest.approx(metrics["entropy"] / math.log(JOINT))
    assert 0.95 < metrics["entropy_fraction"] <= 1.0  # a fresh policy is near uniform
