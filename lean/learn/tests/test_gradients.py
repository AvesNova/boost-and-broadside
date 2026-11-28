"""The gradient decomposition (lean.learn.gradients) and the entropy bonus in the BC stage."""

import copy

import pytest
import torch

from lean.learn import RolloutStore, collect, make_optimiser, update
from lean.learn.gradients import SPLIT, TRUNK_MODULES, trunk_first
from lean.learn.returns import value_support
from lean.learn.tests.setting import small
from lean.learn.training import LOSS_NAMES, loss_inputs, minibatch_loss

torch.set_num_threads(2)


def rollout_of(ticks: int = 16, **overrides):
    config, game_config, policy, opponent, state, carry, generator = small(**overrides)
    rollout = collect(config, game_config, policy, opponent, state, carry, ticks, generator)
    return config, policy, rollout


def diagnosed(config, policy, rollout, cloning: bool, gradients: bool = True):
    twin = copy.deepcopy(policy)
    metrics = update(
        config,
        twin,
        make_optimiser(config, twin),
        rollout,
        cloning=cloning,
        teacher_weight=0.0 if cloning else 0.5,
        generator=torch.Generator().manual_seed(5),
        gradients=gradients,
    )
    return twin, metrics


def top_level(metrics: dict, scope: str) -> dict[str, float]:
    prefix = f"gradient/share/{scope}/"
    return {
        name.removeprefix(prefix): value
        for name, value in metrics.items()
        if name.startswith(prefix)
        and not any(name.removeprefix(prefix).startswith(t + "_") for t in SPLIT)
    }


@pytest.mark.parametrize("cloning", [True, False])
def test_the_shares_sum_to_one_and_the_parts_to_their_term(cloning):
    """Catches a decomposition whose top-level terms are not the step (shares not summing to 1
    in either scope) or whose parts (value by level, next state by column, pair by target) do
    not sum to their term."""
    config, policy, rollout = rollout_of()
    _, metrics = diagnosed(config, policy, rollout, cloning)
    for scope in ("model", "trunk"):
        shares = top_level(metrics, scope)
        assert ("bc" if cloning else "teacher") in shares
        assert sum(shares.values()) == pytest.approx(1.0, abs=1e-4)
        for term in ("value", "next_state", "pair"):
            parts = [
                value
                for name, value in metrics.items()
                if name.startswith(f"gradient/share/{scope}/{term}_")
            ]
            assert len(parts) > 1
            assert sum(parts) == pytest.approx(metrics[f"gradient/share/{scope}/{term}"], abs=1e-5)
    assert 0.0 < metrics["gradient/actor_share"] < 1.0


def test_a_term_with_a_zero_coefficient_has_a_zero_share():
    """Catches a term whose coefficient is 0 still entering the step: the hit term at 0 and
    the surrogate in the BC stage have zero norm and zero share."""
    config, policy, rollout = rollout_of(hit_coefficient=0.0)
    _, metrics = diagnosed(config, policy, rollout, cloning=True)
    for term in ("hit", "surrogate"):
        for scope in ("model", "trunk"):
            assert metrics[f"gradient/norm/{scope}/{term}"] == 0.0
            assert metrics[f"gradient/share/{scope}/{term}"] == 0.0
    assert metrics["gradient/norm/trunk/bc"] > 0.0


@pytest.mark.parametrize("cloning", [True, False])
def test_the_diagnostic_leaves_the_training_unchanged(cloning):
    """Catches a diagnostic that changes the update (draws from its generator, leaves
    gradients behind): with it on, the weights and the update's own measures equal those with
    it off."""
    config, policy, rollout = rollout_of()
    off, plain = diagnosed(config, policy, rollout, cloning, gradients=False)
    on, measured = diagnosed(config, policy, rollout, cloning, gradients=True)
    for first, second in zip(off.parameters(), on.parameters(), strict=True):
        assert torch.equal(first, second)
    assert {name: measured[name] for name in plain} == plain
    assert not any(name.startswith("gradient/") for name in plain)


def test_the_weighted_terms_sum_to_the_loss():
    """Catches a split that loses or double-counts a part: minibatch_loss's weighted terms
    (`parts`) sum to its loss."""
    config, policy, rollout = rollout_of()
    store = RolloutStore.of(rollout)
    store.ready()
    store.targets(len(config.level_names))
    index, _ = store.batch_index(torch.arange(store.views), store.views)
    weights = torch.tensor([1.0, config.entropy_coefficient, 0.5])
    support = value_support(config, policy.config.bins, store.device)
    batch = next(iter(store.batches(store.observation_names + LOSS_NAMES, index)))
    arguments = loss_inputs(config, policy, store, batch, None, torch.Generator().manual_seed(1))
    loss, _, terms = minibatch_loss(
        config, policy, *arguments, weights, store.counts().sum(0), support, parts=True
    )
    total = torch.stack(list(terms.values())).sum().item()
    assert total == pytest.approx(loss.item(), rel=1e-5)


def test_the_trunk_is_the_encoder_and_the_blocks():
    """The trunk scope holds exactly the encoder's and the blocks' parameters, first."""
    _, policy, _ = rollout_of(4)
    parameters, size = trunk_first(policy)
    trunk = [p for name, p in policy.named_parameters() if name.split(".")[0] in TRUNK_MODULES]
    assert size == sum(p.numel() for p in trunk) > 0
    assert all(a is b for a, b in zip(parameters, trunk, strict=False))
    assert TRUNK_MODULES == ("encoder", "blocks")


def test_the_bc_stage_applies_the_entropy_bonus():
    """Catches a BC stage without the entropy bonus (author 2026-11-24: on for the whole run):
    a cloning update with entropy_coefficient 0 moves the weights differently from one with
    the default 0.003."""
    config, policy, rollout = rollout_of()
    steps = []
    for coefficient in (0.0, config.entropy_coefficient):
        twin = copy.deepcopy(policy)
        trial = copy.copy(config)
        trial.entropy_coefficient = coefficient
        update(
            trial,
            twin,
            make_optimiser(trial, twin),
            rollout,
            cloning=True,
            generator=torch.Generator().manual_seed(5),
        )
        steps.append([p.detach().clone() for p in twin.heads.lasts[0].parameters()])
    assert config.entropy_coefficient > 0
    assert any(not torch.equal(a, b) for a, b in zip(*steps, strict=True))
