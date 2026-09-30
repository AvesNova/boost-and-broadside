"""Contracts for the next-state population split: partition, calibration, exactness.

The split is only worth reading if its parts add up to what training does, so
the central assertion is a sum: the trained likelihood's population-and-half
gradients against the ``next_state`` term's gradient. The rest pins what each
reading means -- a calibrated head reads z2 = 1, identical halves read as pure
signal, opposed halves as pure noise.
"""

from __future__ import annotations

import math

import pytest
import torch

from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.config.diagnostics import GradientDiagnosticsConfig
from boost_and_broadside.train.rl.grad_diagnostics import TermGradientAccumulator
from boost_and_broadside.train.rl.next_state_populations import (
    ENVIRONMENT_HALVES,
    NEXT_STATE_LIKELIHOODS,
    NEXT_STATE_POPULATIONS,
    candidate_losses,
    environment_halves,
    gradient_metric_records,
    population_masks,
)
from boost_and_broadside.train.rl.physical_belief import (
    NEXT_STATE_OUTPUT_DIM,
    PHYSICAL_MEAN_DIM,
    PHYSICAL_MEAN_NAMES,
    POSITION_RHO,
    POSITION_SIGMA,
    POSITION_X,
    VELOCITY_RHO,
    VELOCITY_SIGMA,
    VELOCITY_X,
    PhysicalNextState,
)

from .test_grad_diagnostics import (
    _SUM_TOLERANCE,
    _accumulate_one_minibatch,
    _diagnostic_trainer,
    _one_update,
    _summed_gradient,
)


def _model() -> PhysicalNextState:
    return PhysicalNextState.from_ship_config(ShipConfig())


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


def _correlated_prediction(tokens: int) -> torch.Tensor:
    """A prediction with spread and correlation varying per token."""
    prediction = torch.zeros(tokens, NEXT_STATE_OUTPUT_DIM)
    uncertainty = prediction[:, PHYSICAL_MEAN_DIM:]
    uncertainty.uniform_(-1.0, 1.0)
    uncertainty[:, POSITION_RHO].uniform_(-1.5, 1.5)
    uncertainty[:, VELOCITY_RHO].uniform_(-1.5, 1.5)
    prediction[:, :PHYSICAL_MEAN_DIM].uniform_(-0.2, 0.2)
    return prediction


def test_labels_drawn_from_the_predicted_gaussian_read_as_calibrated() -> None:
    """z2 averages 1.0 per channel when the spread is honest, correlation included.

    Attitude is excluded: its residual wraps on the circle, so a draw wide
    enough to wrap is not Gaussian in the wrapped coordinate. The spreads here
    keep it far from wrapping, but the claim being pinned is about the
    covariance, which the other ten channels exercise.
    """
    torch.manual_seed(1)
    model = _model()
    tokens = 200_000
    prediction = _correlated_prediction(tokens)
    uncertainty = prediction[:, PHYSICAL_MEAN_DIM:]
    labels = prediction[:, :PHYSICAL_MEAN_DIM].clone()
    noise = torch.randn(tokens, PHYSICAL_MEAN_DIM)
    sigma = model.variance(prediction).sqrt()
    for mean_x, axes, rho_column in (
        (POSITION_X, POSITION_SIGMA, POSITION_RHO),
        (VELOCITY_X, VELOCITY_SIGMA, VELOCITY_RHO),
    ):
        rho = torch.tanh(uncertainty[:, rho_column])
        first = noise[:, mean_x]
        second = rho * first + (1.0 - rho * rho).sqrt() * noise[:, mean_x + 1]
        labels[:, mean_x] += torch.exp(uncertainty[:, axes[0]]) * first
        labels[:, mean_x + 1] += torch.exp(uncertainty[:, axes[1]]) * second
    for channel in range(PHYSICAL_MEAN_DIM):
        if channel in (POSITION_X, POSITION_X + 1, VELOCITY_X, VELOCITY_X + 1):
            continue
        labels[:, channel] += sigma[:, channel] * noise[:, channel]

    z2 = model.standardized_square(prediction, labels).mean(0)
    attitude = PHYSICAL_MEAN_NAMES.index("attitude")
    for channel, value in enumerate(z2.tolist()):
        if channel == attitude:
            continue
        assert value == pytest.approx(1.0, abs=0.03), PHYSICAL_MEAN_NAMES[channel]


def test_log_sigma_agrees_with_variance() -> None:
    model = _model()
    prediction = _correlated_prediction(32)
    torch.testing.assert_close(
        torch.exp(2.0 * model.log_sigma(prediction)), model.variance(prediction)
    )


def test_the_candidate_likelihoods_differ_only_where_they_should() -> None:
    """beta-NLL is NLL at unit spread; MSE sends nothing to the spread."""
    torch.manual_seed(2)
    model = _model()
    prediction = _correlated_prediction(64)
    labels = torch.randn(64, PHYSICAL_MEAN_DIM) * 0.3

    unit = prediction.clone()
    unit[:, PHYSICAL_MEAN_DIM:] = 0.0  # log sigma 0, rho 0
    losses = candidate_losses(model, unit, labels, model.loss(unit, labels))
    torch.testing.assert_close(losses["beta_nll"], losses["nll"])

    prediction.requires_grad_(True)
    losses = candidate_losses(model, prediction, labels, model.loss(prediction, labels))
    assert set(losses) == set(NEXT_STATE_LIKELIHOODS)
    (grad,) = torch.autograd.grad(losses["mse"].sum(), prediction)
    assert torch.count_nonzero(grad[:, PHYSICAL_MEAN_DIM:]) == 0
    assert torch.count_nonzero(grad[:, :PHYSICAL_MEAN_DIM]) > 0


class _Linear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(4)
        self.weight = torch.nn.Parameter(torch.randn(8))


def _halves_accumulator(pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    """An accumulator whose ns_nll halves have prescribed gradients."""
    model = _Linear()
    accumulator = TermGradientAccumulator([model.weight], [True])
    terms = {}
    for population in NEXT_STATE_POPULATIONS:
        first, second = pairs[population]
        terms[f"ns_nll/{population}/a"] = (model.weight * first).sum()
        terms[f"ns_nll/{population}/b"] = (model.weight * second).sum()
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
    group = "trunk_next_state_nll"
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
    # Likelihoods that were not accumulated are absent rather than zero.
    assert not any("next_state_beta_nll" in key for key in records)


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
        f"ns_nll/{population}/{half}"
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
    assert not any("next_state_nll" in key for key in metrics)


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
        for moment in ("nll", "z2", "log_sigma", "sq_err"):
            assert math.isfinite(metrics[f"next_state_population/{population}/{moment}"])
        for channel in PHYSICAL_MEAN_NAMES:
            assert f"next_state_z2/{population}/{channel}" in metrics
            assert f"next_state_log_sigma/{population}/{channel}" in metrics


def test_calibration_moments_are_logged_without_any_gradient_diagnostic(tmp_path) -> None:
    """They cost a few masked reductions, so every run carries them."""
    metrics = _one_update(_diagnostic_trainer(tmp_path, "off"), update=1)
    for population in NEXT_STATE_POPULATIONS:
        assert f"next_state_population/{population}/token_share" in metrics
