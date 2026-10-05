"""Contracts for the next-state population split: partition, sums, exactness.

The split is only worth reading if its parts add up to what training does, so
the central assertion is a sum: the trained objective's population-and-half
gradients against the ``next_state`` term's gradient. The rest pins what each
reading means -- identical halves read as pure signal, opposed halves as pure
noise, and each population's sums as means over its own tokens.
"""

from __future__ import annotations

import math

import pytest
import torch

from boost_and_broadside.config import EnvConfig
from boost_and_broadside.config.diagnostics import GradientDiagnosticsConfig
from boost_and_broadside.train.rl.grad_diagnostics import TermGradientAccumulator
from boost_and_broadside.train.rl.next_state_populations import (
    ENVIRONMENT_HALVES,
    MOMENT_NAMES,
    NEXT_STATE_LIKELIHOODS,
    NEXT_STATE_POPULATIONS,
    environment_halves,
    gradient_metric_records,
    population_masks,
    population_metric_records,
    population_moments,
)
from boost_and_broadside.train.rl.ship_codes import CODE_GROUP_DIM, CODE_GROUP_NAMES

from .test_grad_diagnostics import (
    _SUM_TOLERANCE,
    _accumulate_one_minibatch,
    _diagnostic_trainer,
    _one_update,
    _summed_gradient,
)

# ----------------------------------------------------------------------
# Pure pieces
# ----------------------------------------------------------------------


def test_populations_partition_exactly_the_supervised_tokens() -> None:
    torch.manual_seed(0)
    shape = (5, 4, 6)
    supervised = torch.rand(shape) > 0.3
    team_id = torch.randint(0, 2, shape)
    visible = torch.rand(shape) > 0.5
    masks = population_masks(supervised, team_id, visible)

    assert masks.shape == (len(NEXT_STATE_POPULATIONS), *shape)
    assert torch.equal(masks.sum(0), supervised.long())
    ally, enemy_visible, enemy_hidden = masks
    assert not (ally & (team_id != 0)).any()
    assert not (enemy_visible & ~visible).any()
    assert not (enemy_hidden & visible).any()


def test_environment_halves_are_disjoint_and_cover_every_env() -> None:
    halves = environment_halves(7, torch.device("cpu"))
    assert halves.shape == (len(ENVIRONMENT_HALVES), 7)
    assert torch.equal(halves.sum(0), torch.ones(7, dtype=torch.long))
    assert halves[0].any() and halves[1].any()


def test_population_sums_are_means_over_each_populations_own_tokens() -> None:
    torch.manual_seed(1)
    shape = (3, 4, 5)
    supervised = torch.rand(shape) > 0.2
    masks = population_masks(supervised, torch.randint(0, 2, shape), torch.rand(shape) > 0.5)
    cross_entropy = torch.rand(*shape, CODE_GROUP_DIM)
    baseline = cross_entropy + 1.0
    sums, counts = population_moments(cross_entropy, baseline, masks)
    assert sums.shape == (len(NEXT_STATE_POPULATIONS), len(MOMENT_NAMES), CODE_GROUP_DIM)
    records = population_metric_records(sums, counts)
    for p, population in enumerate(NEXT_STATE_POPULATIONS):
        expected = cross_entropy[masks[p]].mean(0).double()
        for g, group in enumerate(CODE_GROUP_NAMES):
            assert records[f"next_state_ce/{population}/{group}"] == pytest.approx(
                float(expected[g]), rel=1e-5
            )
            assert records[f"next_state_baseline_ce/{population}/{group}"] == pytest.approx(
                float(expected[g]) + 1.0, rel=1e-5
            )


class _Linear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(4)
        self.weight = torch.nn.Parameter(torch.randn(8))


def _halves_accumulator(pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    """An accumulator whose ns_ce halves have prescribed gradients."""
    model = _Linear()
    accumulator = TermGradientAccumulator([model.weight], [True])
    terms = {}
    for population in NEXT_STATE_POPULATIONS:
        first, second = pairs[population]
        terms[f"ns_ce/{population}/a"] = (model.weight * first).sum()
        terms[f"ns_ce/{population}/b"] = (model.weight * second).sum()
    accumulator.accumulate(terms)
    return accumulator


def test_identical_halves_are_all_signal_and_opposed_halves_all_noise() -> None:
    direction = torch.randn(8)
    other = torch.randn(8)
    records = gradient_metric_records(
        _halves_accumulator(
            {
                "ally": (direction, direction),
                "enemy_visible": (other, -other),
                "enemy_hidden": (torch.zeros(8), torch.zeros(8)),
            }
        )
    )
    group = "trunk_next_state_ce"
    assert records[f"grad_halves_cos/{group}/ally"] == pytest.approx(1.0)
    assert records[f"grad_coherent_fraction/{group}/ally"] == pytest.approx(1.0)
    assert records[f"grad_coherent_norm/{group}/ally"] == pytest.approx(
        float((2 * direction).norm()), rel=1e-6
    )
    assert records[f"grad_halves_cos/{group}/enemy_visible"] == pytest.approx(-1.0)
    assert records[f"grad_coherent_norm/{group}/enemy_visible"] == 0.0
    assert records[f"grad_coherent_share/{group}/ally"] == pytest.approx(1.0)
    # A population with no gradient is zero everywhere rather than NaN.
    for key, value in records.items():
        assert math.isfinite(value), key


def test_the_split_needs_a_level_to_run() -> None:
    with pytest.raises(ValueError):
        GradientDiagnosticsConfig(level="off", next_state_populations=True)
    settings = GradientDiagnosticsConfig(level="top_level", next_state_populations=True)
    assert settings.decomposes_next_state_by_population
    assert not GradientDiagnosticsConfig(level="top_level").decomposes_next_state_by_population


# ----------------------------------------------------------------------
# Trainer integration
# ----------------------------------------------------------------------


# Short sight so a small rollout holds every population, as the profiles' fog does.
_FOGGED = EnvConfig(
    num_ships=4, max_bullets=8, max_episode_steps=50, vision_range=300.0, spawn_reveal=True
)


def _split_trainer(tmp_path):
    torch.manual_seed(3)
    trainer = _diagnostic_trainer(tmp_path, "top_level", env_config=_FOGGED)
    trainer._grad_diag = GradientDiagnosticsConfig(level="top_level", next_state_populations=True)
    runtime = trainer._initialize_rollout_runtime()
    terminated = trainer._collect_rollout(runtime, False)
    trainer._compute_rollout_gae(runtime, terminated)
    return trainer


def test_trained_likelihood_split_sums_to_the_next_state_gradient(tmp_path) -> None:
    trainer = _split_trainer(tmp_path)
    accumulator = _accumulate_one_minibatch(trainer, level="top_level")

    parts = [
        f"ns_ce/{population}/{half}"
        for population in NEXT_STATE_POPULATIONS
        for half in ENVIRONMENT_HALVES
    ]
    assert set(parts) <= set(accumulator.term_names)
    aggregate = _summed_gradient(accumulator, ["next_state"])
    assert aggregate.norm() > 0.0
    torch.testing.assert_close(
        _summed_gradient(accumulator, parts), aggregate, rtol=_SUM_TOLERANCE, atol=_SUM_TOLERANCE
    )


def test_the_split_is_silent_unless_asked_for(tmp_path) -> None:
    trainer = _diagnostic_trainer(tmp_path, "top_level")
    metrics = _one_update(trainer, update=1)
    assert not any(key.startswith("grad_coherent") for key in metrics)
    assert not any("next_state_ce/" in key and key.startswith("grad") for key in metrics)


def test_an_update_reports_every_population_and_likelihood(tmp_path) -> None:
    trainer = _split_trainer(tmp_path)
    metrics = _one_update(trainer, update=1)

    for likelihood in NEXT_STATE_LIKELIHOODS:
        for group in (f"next_state_{likelihood}", f"trunk_next_state_{likelihood}"):
            for population in NEXT_STATE_POPULATIONS:
                for family in (
                    "grad_norm",
                    "grad_share",
                    "grad_halves_cos",
                    "grad_coherent_norm",
                    "grad_coherent_fraction",
                    "grad_coherent_share",
                ):
                    key = f"{family}/{group}/{population}"
                    assert key in metrics, key
                    assert math.isfinite(metrics[key]), key
            assert f"grad_diag/agreement/{group}" in metrics

    shares = [
        metrics[f"next_state_population/{population}/token_share"]
        for population in NEXT_STATE_POPULATIONS
    ]
    assert sum(shares) == pytest.approx(1.0)
    # The fixture's fog leaves enemies both in and out of sight.
    assert all(share > 0.0 for share in shares), shares
    for population in NEXT_STATE_POPULATIONS:
        for moment in MOMENT_NAMES:
            assert math.isfinite(metrics[f"next_state_population/{population}/{moment}"])
        for group in CODE_GROUP_NAMES:
            assert f"next_state_ce/{population}/{group}" in metrics
            assert f"next_state_baseline_ce/{population}/{group}" in metrics


def test_calibration_moments_are_logged_without_any_gradient_diagnostic(tmp_path) -> None:
    """They cost a few masked reductions, so every run carries them."""
    metrics = _one_update(_diagnostic_trainer(tmp_path, "off"), update=1)
    for population in NEXT_STATE_POPULATIONS:
        assert f"next_state_population/{population}/token_share" in metrics
