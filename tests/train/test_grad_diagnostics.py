"""Contracts for gradient decomposition: exactness, accumulation, and silence when off.

The whole instrument is only worth reading if the parts add up to the update
being run, so most of what is asserted here is a sum: component policy
gradients against the aggregate policy gradient, component critic gradients
against the aggregate critic gradient, and micro-batch gradients against the
unsplit minibatch.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from boost_and_broadside.config.diagnostics import (
    GRADIENT_DIAGNOSTICS_LEVELS,
    GradientDiagnosticsConfig,
)
from boost_and_broadside.env.rewards import REWARD_COMPONENT_NAMES
from boost_and_broadside.train.rl.grad_diagnostics import (
    TermGradientAccumulator,
    scope_metric_records,
    scope_statistics,
)

from .test_ppo import _make_trainer

# fp32 accumulation over a few thousand tokens; the decomposition is exact in
# exact arithmetic, so anything above roundoff is a real defect.
_SUM_TOLERANCE = 1e-5


# ----------------------------------------------------------------------
# Accumulator mechanics, on synthetic gradients with known relationships
# ----------------------------------------------------------------------


class _ToyModel(torch.nn.Module):
    """Two disjoint heads over one shared trunk parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = torch.nn.Linear(3, 3, bias=False)
        self.left = torch.nn.Linear(3, 1, bias=False)
        self.right = torch.nn.Linear(3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)  # (B, 3)
        return self.left(hidden).sum(), self.right(hidden).sum()


def _toy_accumulator() -> tuple[_ToyModel, TermGradientAccumulator]:
    torch.manual_seed(0)
    model = _ToyModel()
    params = list(model.parameters())
    trunk_ids = {id(parameter) for parameter in model.trunk.parameters()}
    return model, TermGradientAccumulator(params, [id(p) in trunk_ids for p in params])


def test_identical_terms_have_cosine_one() -> None:
    model, accumulator = _toy_accumulator()
    left, _ = model(torch.randn(4, 3))
    accumulator.accumulate({"a": left, "b": left * 1.0})
    stats = accumulator.statistics(["a", "b"], trunk=False)
    assert stats.cosines[("a", "b")] == pytest.approx(1.0, abs=1e-6)


def test_opposed_terms_have_cosine_minus_one_and_cancel() -> None:
    model, accumulator = _toy_accumulator()
    left, _ = model(torch.randn(4, 3))
    accumulator.accumulate({"a": left, "b": -left})
    stats = accumulator.statistics(["a", "b"], trunk=False)
    assert stats.cosines[("a", "b")] == pytest.approx(-1.0, abs=1e-6)
    # Perfect cancellation: the combined gradient is zero even though each term
    # has a large one. This is exactly what a norm-only diagnostic cannot see.
    assert stats.total_norm == pytest.approx(0.0, abs=1e-5)
    assert stats.agreement == pytest.approx(0.0, abs=1e-5)


def test_terms_on_disjoint_heads_are_orthogonal_over_the_whole_model() -> None:
    model, accumulator = _toy_accumulator()
    # A constant input makes the trunk gradients parallel, leaving the disjoint
    # head parameters as the only source of disagreement.
    left, right = model(torch.ones(1, 3))
    accumulator.accumulate({"left": left, "right": right})
    whole = accumulator.statistics(["left", "right"], trunk=False)
    trunk = accumulator.statistics(["left", "right"], trunk=True)
    assert abs(whole.cosines[("left", "right")]) < abs(trunk.cosines[("left", "right")])


def test_a_term_with_no_gradient_reads_as_zero_rather_than_nan() -> None:
    model, accumulator = _toy_accumulator()
    left, _ = model(torch.randn(4, 3))
    accumulator.accumulate({"live": left, "dead": left * 0.0})
    stats = accumulator.statistics(["live", "dead"], trunk=False)
    assert stats.norms["dead"] == pytest.approx(0.0)
    for value in (*stats.norms.values(), *stats.cosines.values(), *stats.shares.values()):
        assert math.isfinite(value)
    assert math.isfinite(stats.agreement)


def test_a_term_outside_the_graph_is_skipped_rather_than_differentiated() -> None:
    model, accumulator = _toy_accumulator()
    left, _ = model(torch.randn(4, 3))
    accumulator.accumulate({"live": left, "disabled": torch.zeros(())})
    assert accumulator.term_names == ("live",)


def test_microbatch_gradients_accumulate_to_the_unsplit_minibatch() -> None:
    """The reason norms are taken at the end and not per micro-batch."""
    torch.manual_seed(0)
    model, accumulator = _toy_accumulator()
    batch = torch.randn(8, 3)

    for half in (batch[:4], batch[4:]):
        left, _ = model(half)
        accumulator.accumulate({"left": left})
    split = accumulator.statistics(["left"], trunk=False).norms["left"]

    # The same parameters, so the two accumulators are measuring one model.
    params = list(model.parameters())
    trunk_ids = {id(parameter) for parameter in model.trunk.parameters()}
    whole_accumulator = TermGradientAccumulator(
        params, [id(parameter) in trunk_ids for parameter in params]
    )
    left, _ = model(batch)
    whole_accumulator.accumulate({"left": left})
    unsplit = whole_accumulator.statistics(["left"], trunk=False).norms["left"]

    assert unsplit > 0.0

    assert accumulator.microbatches == 2
    assert split == pytest.approx(unsplit, rel=1e-6)


def test_summed_cosine_is_not_the_mean_of_microbatch_cosines() -> None:
    """The two quantities genuinely differ, so choosing between them matters."""
    torch.manual_seed(1)
    model, accumulator = _toy_accumulator()
    halves = [torch.randn(4, 3), torch.randn(4, 3)]

    params = list(model.parameters())
    trunk_ids = {id(parameter) for parameter in model.trunk.parameters()}
    trunk_mask = [id(parameter) in trunk_ids for parameter in params]

    per_microbatch = []
    for half in halves:
        left, right = model(half)
        accumulator.accumulate({"left": left, "right": right})
        single = TermGradientAccumulator(params, trunk_mask)
        single.accumulate({"left": left, "right": right})
        per_microbatch.append(single.statistics(["left", "right"], trunk=True).cosines)

    combined = accumulator.statistics(["left", "right"], trunk=True).cosines[("left", "right")]
    mean_of_parts = sum(c[("left", "right")] for c in per_microbatch) / len(per_microbatch)
    assert combined != pytest.approx(mean_of_parts, abs=1e-9)


def _linear_accumulator(width: int) -> tuple[torch.nn.Linear, TermGradientAccumulator]:
    torch.manual_seed(0)
    model = torch.nn.Linear(width, 1, bias=False)
    params = list(model.parameters())
    return model, TermGradientAccumulator(params, [True] * len(params))


def test_a_gradient_every_game_shares_reads_as_pure_signal() -> None:
    model, accumulator = _linear_accumulator(64)
    x = torch.randn(1, 64)
    for _ in range(6):
        accumulator.accumulate({"signal": model(x).sum()})
    agreement = accumulator.half_agreement(["signal"], trunk=True)["signal"]
    assert agreement.cosine == pytest.approx(1.0, abs=1e-5)
    assert agreement.signal_fraction == pytest.approx(1.0, abs=1e-5)


def test_a_zero_mean_gradient_reads_as_noise() -> None:
    """Independent games whose gradients average to nothing agree on nothing."""
    model, accumulator = _linear_accumulator(256)
    generator = torch.Generator().manual_seed(1)
    for _ in range(40):
        accumulator.accumulate({"noise": model(torch.randn(4, 256, generator=generator)).sum()})
    agreement = accumulator.half_agreement(["noise"], trunk=True)["noise"]
    # Two independent 256-d directions: cos ~ N(0, 1/256).
    assert abs(agreement.cosine) < 0.25
    assert abs(agreement.signal_fraction) < 0.25


def test_the_signal_fraction_recovers_a_known_mixture() -> None:
    """E||s||^2 / E||S||^2 for a shared mean plus per-micro-batch noise."""
    model, accumulator = _linear_accumulator(4096)
    generator = torch.Generator().manual_seed(2)
    mean = torch.randn(1, 4096, generator=generator)
    microbatches = 8
    for _ in range(microbatches):
        noise = torch.randn(1, 4096, generator=generator)
        accumulator.accumulate({"mixed": model(mean + noise).sum()})
    # Sum of 8 micro-batches: ||8 m||^2 = 64 d against noise 8 d.
    expected = 64.0 / (64.0 + 8.0)
    agreement = accumulator.half_agreement(["mixed"], trunk=True)["mixed"]
    assert agreement.signal_fraction == pytest.approx(expected, abs=0.05)


def test_one_microbatch_has_no_halves_to_compare() -> None:
    model, accumulator = _linear_accumulator(8)
    accumulator.accumulate({"only": model(torch.randn(2, 8)).sum()})
    assert accumulator.half_agreement(["only"], trunk=True) == {}


def test_half_agreement_does_not_change_the_accumulated_sum() -> None:
    model, accumulator = _linear_accumulator(8)
    batch = torch.randn(6, 8)
    for row in batch.split(2):
        accumulator.accumulate({"term": model(row).sum()})
    before = accumulator.statistics(["term"], trunk=True).norms["term"]
    accumulator.half_agreement(["term"], trunk=True)
    assert accumulator.statistics(["term"], trunk=True).norms["term"] == before
    assert before == pytest.approx(float(batch.sum(0).norm()), rel=1e-6)


def test_scope_statistics_of_no_terms_is_empty_rather_than_undefined() -> None:
    stats = scope_statistics([], torch.zeros(0, 0))
    assert stats.norms == {} and stats.cosines == {}
    assert stats.total_norm == 0.0 and stats.agreement == 0.0


def test_metric_records_are_namespaced_by_group() -> None:
    stats = scope_statistics(["a", "b"], torch.tensor([[4.0, 0.0], [0.0, 9.0]]))
    records = scope_metric_records("top_level", stats)
    assert records["grad_norm/top_level/a"] == pytest.approx(2.0)
    assert records["grad_norm/top_level/b"] == pytest.approx(3.0)
    assert records["grad_cos/top_level/a__b"] == pytest.approx(0.0)
    assert records["grad_share/top_level/a"] == pytest.approx(0.4)
    # Orthogonal terms: the combined gradient is shorter than the sum of parts.
    assert records["grad_diag/total_norm/top_level"] == pytest.approx(math.sqrt(13.0))
    assert records["grad_diag/agreement/top_level"] == pytest.approx(math.sqrt(13.0) / 5.0)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def test_disabled_is_the_default_and_measures_nothing() -> None:
    settings = GradientDiagnosticsConfig()
    assert settings.level == "off"
    assert not settings.enabled
    assert not settings.measures_update(1)


@pytest.mark.parametrize("level", GRADIENT_DIAGNOSTICS_LEVELS)
def test_every_level_declares_what_it_decomposes(level: str) -> None:
    settings = GradientDiagnosticsConfig(level=level)
    assert settings.decomposes_policy_by_reward == (level in ("reward_policy", "reward_full"))
    assert settings.decomposes_value_by_reward == (level == "reward_full")


@pytest.mark.parametrize(
    "kwargs",
    [{"level": "everything"}, {"interval": 0}, {"minibatches": 0}],
)
def test_a_nonsensical_setting_is_rejected_on_construction(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        GradientDiagnosticsConfig(**kwargs)


# ----------------------------------------------------------------------
# Trainer integration
# ----------------------------------------------------------------------


def _diagnostic_trainer(tmp_path, level: str, **kwargs):
    """A trainer whose gradient diagnostics run at ``level``."""
    trainer = _make_trainer(checkpoint_dir=str(tmp_path), **kwargs)
    trainer._grad_diag = GradientDiagnosticsConfig(level=level)
    if trainer._grad_diag.enabled:
        trunk_ids = trainer._policy_module.trunk_parameter_ids()
        trainer._grad_diag_params = [
            parameter
            for parameter in trainer._policy_module.parameters()
            if parameter.requires_grad
        ]
        trainer._grad_diag_trunk = [
            id(parameter) in trunk_ids for parameter in trainer._grad_diag_params
        ]
    return trainer


def _one_update(trainer, **kwargs) -> dict:
    """Collect one rollout and run one full set of PPO epochs over it."""
    runtime = trainer._initialize_rollout_runtime()
    terminated = trainer._collect_rollout(runtime, False)
    trainer._compute_rollout_gae(runtime, terminated)
    return trainer._update_epochs(all_buffers=[trainer.buffer, *trainer.aux_buffers], **kwargs)


def _diagnostic_keys(metrics: dict) -> set[str]:
    return {
        key
        for key in metrics
        if key.startswith(("grad_norm/", "grad_cos/", "grad_share/", "grad_diag/"))
    }


def test_off_emits_no_diagnostic_metric(tmp_path) -> None:
    metrics = _one_update(_diagnostic_trainer(tmp_path, "off"), update=1)
    assert _diagnostic_keys(metrics) == set()


def test_off_never_calls_diagnostic_autograd(tmp_path, monkeypatch) -> None:
    """The disabled path must not merely produce nothing -- it must not run."""
    trainer = _diagnostic_trainer(tmp_path, "off")
    calls: list[int] = []
    original = torch.autograd.grad
    monkeypatch.setattr(
        torch.autograd,
        "grad",
        lambda *args, **kwargs: (calls.append(1), original(*args, **kwargs))[1],
    )
    _one_update(trainer, update=1)
    assert calls == []
    assert trainer._grad_diag_params == []


def test_measuring_does_not_disturb_the_gradient_that_gets_applied(tmp_path) -> None:
    """Diagnostics observe the update; they must not participate in it.

    Compared on one fixed minibatch rather than end to end, because the update
    loop draws its minibatch order from the unseeded global numpy RNG and two
    whole updates are not reproducible even with diagnostics off. Measured at
    the widest setting, next-state population split included.
    """
    trainer = _prepared_trainer(tmp_path, "reward_full")
    trainer._grad_diag = GradientDiagnosticsConfig(level="reward_full", next_state_populations=True)
    trainer._precompute_lambda_aggregates(trainer.buffer, is_primary=True)
    trainer._precompute_ns_labels(trainer.buffer)
    chunks = next(
        trainer.buffer.get_minibatch_iterator(
            trainer.cfg.num_minibatches, trainer.cfg.microbatch_tokens
        )
    )
    denominators = trainer._minibatch_denominators(chunks, trainer.buffer, True)
    minibatch_envs = sum(chunk.alive.shape[1] for chunk in chunks)

    def applied_gradient(measure: bool) -> list[torch.Tensor]:
        trainer.optim.zero_grad()
        accumulator = (
            TermGradientAccumulator(trainer._grad_diag_params, trainer._grad_diag_trunk)
            if measure
            else None
        )
        for source, device_chunk in trainer._iter_device_chunks(chunks, trainer.buffer):
            loss, _ = trainer._compute_minibatch_loss(
                device_chunk,
                True,
                denominators,
                source.alive.shape[1] / minibatch_envs,
                grad_terms=accumulator,
                grad_scale=1.0,
            )
            loss.backward()
        return [
            parameter.grad.detach().clone()
            for parameter in trainer._policy_module.parameters()
            if parameter.grad is not None
        ]

    measured = applied_gradient(True)
    unmeasured = applied_gradient(False)
    assert measured and len(measured) == len(unmeasured)
    for with_diagnostics, without in zip(measured, unmeasured, strict=True):
        torch.testing.assert_close(with_diagnostics, without, rtol=0, atol=0)


@pytest.mark.parametrize("level", ["top_level", "reward_policy", "reward_full"])
def test_every_active_level_emits_finite_norms_and_cosines(tmp_path, level: str) -> None:
    metrics = _one_update(_diagnostic_trainer(tmp_path, level), update=1)
    keys = _diagnostic_keys(metrics)
    assert keys, f"{level} produced no diagnostics"
    for key in keys:
        assert math.isfinite(metrics[key]), f"{key} is not finite: {metrics[key]}"
    for key in keys:
        if key.startswith("grad_cos/"):
            assert -1.0001 <= metrics[key] <= 1.0001, f"{key} is not a cosine: {metrics[key]}"


def test_top_level_names_every_active_loss_term(tmp_path) -> None:
    metrics = _one_update(_diagnostic_trainer(tmp_path, "top_level"), update=1)
    for term in ("policy", "value", "entropy", "next_state", "enemy_action"):
        assert f"grad_norm/top_level/{term}" in metrics
    assert "grad_norm/trunk_top_level/policy" in metrics
    assert "grad_norm/trunk_top_level/enemy_action" in metrics
    assert "grad_cos/trunk_top_level/policy__enemy_action" in metrics


def test_reward_levels_name_components_from_the_live_registry(tmp_path) -> None:
    trainer = _diagnostic_trainer(tmp_path, "reward_full")
    metrics = _one_update(trainer, update=1)
    active = [
        name
        for index, name in enumerate(trainer._active_names)
        if trainer.wrapper.active_components[index].weight != 0.0
    ]
    assert active
    for name in active:
        assert f"grad_norm/reward_policy/{name}" in metrics
    for name in trainer._active_names:
        assert f"grad_norm/reward_value/{name}" in metrics


def test_reward_policy_does_not_decompose_the_critic(tmp_path) -> None:
    metrics = _one_update(_diagnostic_trainer(tmp_path, "reward_policy"), update=1)
    assert any(key.startswith("grad_norm/reward_policy/") for key in metrics)
    assert not any(key.startswith("grad_norm/reward_value/") for key in metrics)


def test_the_diagnostic_reports_how_it_was_measured(tmp_path) -> None:
    trainer = _diagnostic_trainer(tmp_path, "top_level")
    metrics = _one_update(trainer, update=1)
    assert metrics["grad_diag/microbatches"] >= 1.0
    assert metrics["grad_diag/terms"] >= 1.0
    assert metrics["grad_diag/seconds"] > 0.0
    assert metrics["grad_diag/level"] == float(GRADIENT_DIAGNOSTICS_LEVELS.index("top_level"))


def test_signal_and_noise_are_published_beside_every_measured_term(tmp_path) -> None:
    trainer = _diagnostic_trainer(tmp_path, "reward_full")
    # One environment per micro-batch, so every minibatch has halves to compare.
    trainer.cfg = dataclasses.replace(trainer.cfg, microbatch_tokens=1)
    metrics = _one_update(trainer, update=1)
    assert metrics["grad_diag/microbatches"] >= 2.0
    for key in ("grad_half_cos", "grad_signal_fraction"):
        for scope, term in (
            ("trunk_top_level", "value"),
            ("top_level", "value"),
            ("trunk_reward_value", trainer._active_names[0]),
        ):
            assert math.isfinite(metrics[f"{key}/{scope}/{term}"])


def test_a_single_microbatch_publishes_no_signal_estimate(tmp_path) -> None:
    trainer = _diagnostic_trainer(tmp_path, "top_level")
    trainer.cfg = dataclasses.replace(trainer.cfg, microbatch_tokens=None)
    metrics = _one_update(trainer, update=1)
    assert metrics["grad_diag/microbatches"] == 1.0
    assert not any(key.startswith("grad_half_cos/") for key in metrics)


def test_the_cadence_decides_which_updates_are_measured(tmp_path) -> None:
    trainer = _diagnostic_trainer(tmp_path, "top_level")
    trainer._grad_diag = GradientDiagnosticsConfig(level="top_level", interval=4)
    runtime = trainer._initialize_rollout_runtime()
    terminated = trainer._collect_rollout(runtime, False)
    trainer._compute_rollout_gae(runtime, terminated)
    buffers = [trainer.buffer, *trainer.aux_buffers]

    assert _diagnostic_keys(trainer._update_epochs(all_buffers=buffers, update=3)) == set()
    assert _diagnostic_keys(trainer._update_epochs(all_buffers=buffers, update=4))


def test_the_actor_critic_split_comes_from_the_full_minibatch_when_measuring(tmp_path) -> None:
    """The cheap single-micro-batch probe stands down rather than duplicating it."""
    trainer = _diagnostic_trainer(tmp_path, "top_level")
    metrics = _one_update(trainer, update=1, record_histograms=True)
    assert math.isfinite(metrics["train/grad_norm_actor"])
    assert math.isfinite(metrics["train/grad_norm_critic"])
    assert 0.0 <= metrics["train/actor_grad_share"] <= 1.0
    # Measured over every micro-batch of the minibatch, not one of them.
    assert metrics["grad_diag/microbatches"] >= 1.0


# ----------------------------------------------------------------------
# Exactness of the decompositions against the real aggregate gradients
# ----------------------------------------------------------------------


def _accumulate_one_minibatch(trainer, *, level: str) -> TermGradientAccumulator:
    """Run one primary minibatch through the diagnostic and return its gradients."""
    trainer._precompute_lambda_aggregates(trainer.buffer, is_primary=True)
    trainer._precompute_ns_labels(trainer.buffer)
    accumulator = TermGradientAccumulator(trainer._grad_diag_params, trainer._grad_diag_trunk)
    chunks = next(
        trainer.buffer.get_minibatch_iterator(
            trainer.cfg.num_minibatches, trainer.cfg.microbatch_tokens
        )
    )
    denominators = trainer._minibatch_denominators(chunks, trainer.buffer, True)
    minibatch_envs = sum(chunk.alive.shape[1] for chunk in chunks)
    for source, device_chunk in trainer._iter_device_chunks(chunks, trainer.buffer):
        trainer._compute_minibatch_loss(
            device_chunk,
            True,
            denominators,
            source.alive.shape[1] / minibatch_envs,
            grad_terms=accumulator,
            grad_scale=1.0,
        )
    return accumulator


def _summed_gradient(accumulator: TermGradientAccumulator, names: list[str]) -> torch.Tensor:
    """Concatenate sum_{n in names} g_n into one flat vector for comparison."""
    parts = []
    for index in range(len(accumulator._params)):
        total = None
        for name in names:
            grad = accumulator._sums[name][index]
            if grad is None:
                continue
            total = grad.clone() if total is None else total + grad
        parts.append(
            torch.zeros_like(accumulator._params[index]).reshape(-1)
            if total is None
            else total.reshape(-1)
        )
    return torch.cat(parts)


def _prepared_trainer(tmp_path, level: str, *, clip_coef: float | None = None, jitter: float = 0.0):
    """A trainer holding one collected rollout, optionally off-policy."""
    torch.manual_seed(3)
    trainer = _diagnostic_trainer(tmp_path, level)
    runtime = trainer._initialize_rollout_runtime()
    terminated = trainer._collect_rollout(runtime, False)
    trainer._compute_rollout_gae(runtime, terminated)
    if jitter:
        # Move the policy off the one that collected the rollout, so the
        # importance ratio spreads and PPO's clipping branch is reachable.
        with torch.no_grad():
            for parameter in trainer._policy_module.parameters():
                parameter.add_(torch.randn_like(parameter) * jitter)
    if clip_coef is not None:
        trainer.cfg = dataclasses.replace(trainer.cfg, clip_coef=clip_coef)
    return trainer


@pytest.mark.parametrize(
    ("clip_coef", "jitter"),
    [(None, 0.0), (0.02, 0.05)],
    ids=["unclipped", "clipping_active"],
)
def test_reward_policy_gradients_sum_to_the_aggregate_policy_gradient(
    tmp_path, clip_coef: float | None, jitter: float
) -> None:
    """The decomposition is an attribution of the real update, not a model of it.

    The clipping branch belongs to the aggregate objective. Choosing it per
    component instead lets every reward take whichever branch flatters it, and
    the parts stop summing to the gradient being applied -- by orders of
    magnitude, not by roundoff, which is what the clipping case here catches.
    """
    trainer = _prepared_trainer(tmp_path, "reward_policy", clip_coef=clip_coef, jitter=jitter)
    accumulator = _accumulate_one_minibatch(trainer, level="reward_policy")

    components = [name for name in accumulator.term_names if name.startswith("policy/")]
    assert components
    decomposed = _summed_gradient(accumulator, components)
    aggregate = _summed_gradient(accumulator, ["policy"])
    assert aggregate.norm() > 0.0
    torch.testing.assert_close(decomposed, aggregate, rtol=_SUM_TOLERANCE, atol=_SUM_TOLERANCE)


def test_reward_value_gradients_sum_to_the_aggregate_critic_gradient(tmp_path) -> None:
    trainer = _prepared_trainer(tmp_path, "reward_full")
    accumulator = _accumulate_one_minibatch(trainer, level="reward_full")

    components = [name for name in accumulator.term_names if name.startswith("value/")]
    assert components
    decomposed = _summed_gradient(accumulator, components)
    aggregate = _summed_gradient(accumulator, ["value"])
    assert aggregate.norm() > 0.0
    torch.testing.assert_close(decomposed, aggregate, rtol=_SUM_TOLERANCE, atol=_SUM_TOLERANCE)


def test_value_level_weights_reach_the_critic_loss(tmp_path) -> None:
    """The critic loss is a weighted sum over levels, and the weight is the one
    the decomposition applies too: a level at weight zero sends no gradient,
    and the weighted parts still add up to the term being trained."""
    trainer = _prepared_trainer(tmp_path, "reward_full")
    names = list(trainer._active_names)
    weights = torch.ones(len(names), device=trainer._value_weight_t.device)
    weights[0] = 3.0
    weights[-1] = 0.0
    trainer._value_weight_t = weights
    accumulator = _accumulate_one_minibatch(trainer, level="reward_full")

    components = [name for name in accumulator.term_names if name.startswith("value/")]
    decomposed = _summed_gradient(accumulator, components)
    aggregate = _summed_gradient(accumulator, ["value"])
    torch.testing.assert_close(decomposed, aggregate, rtol=_SUM_TOLERANCE, atol=_SUM_TOLERANCE)
    assert _summed_gradient(accumulator, [f"value/{names[-1]}"]).norm() == 0.0
    assert _summed_gradient(accumulator, [f"value/{names[0]}"]).norm() > 0.0


def test_per_component_clipping_would_be_a_different_objective(tmp_path) -> None:
    """Guards the test above from passing vacuously.

    If independently-clipped components happened to agree with the aggregate,
    the exactness test would prove nothing. They do not agree, and the gap is
    not subtle.
    """
    trainer = _prepared_trainer(tmp_path, "reward_policy", clip_coef=0.02, jitter=0.05)
    trainer._precompute_lambda_aggregates(trainer.buffer, is_primary=True)
    chunks = next(trainer.buffer.get_minibatch_iterator(trainer.cfg.num_minibatches, None))
    batch = chunks[0]
    steps, _, num_ships = batch.alive.shape

    lambda_ij = trainer._team_mixing(
        batch.obs["team_id"][:steps, :, :num_ships].long(), batch.alive
    )
    advantage_k = torch.einsum("tbijk,tbjk->tbik", lambda_ij, batch.advantages)  # (T, b, N, K)
    aggregate = advantage_k.sum(-1)  # (T, b, N)

    ratio = torch.full_like(aggregate, 1.5)  # well outside any sane clip band
    clip = trainer.cfg.clip_coef
    clipped = ratio.clamp(1 - clip, 1 + clip)
    shared = torch.where((-aggregate * clipped) > (-aggregate * ratio), clipped, ratio)
    independent = torch.where(
        (-advantage_k * clipped.unsqueeze(-1)) > (-advantage_k * ratio.unsqueeze(-1)),
        clipped.unsqueeze(-1),
        ratio.unsqueeze(-1),
    )

    true_objective = -(aggregate * shared).sum()
    shared_branch = -(advantage_k * shared.unsqueeze(-1)).sum()
    per_component_branch = -(advantage_k * independent).sum()
    torch.testing.assert_close(shared_branch, true_objective, rtol=1e-4, atol=1e-4)
    # Relative: the advantages are in win units, so their scale follows the
    # reward balance and an absolute tolerance would drift with it.
    assert not torch.isclose(per_component_branch, true_objective, rtol=0.1, atol=1e-6)


# ----------------------------------------------------------------------
# Inactive terms
# ----------------------------------------------------------------------


def test_a_disabled_auxiliary_loss_is_left_out_rather_than_logged_as_zero(tmp_path) -> None:
    trainer = _diagnostic_trainer(tmp_path, "top_level")
    trainer.cfg = dataclasses.replace(trainer.cfg, next_state_coef=0.0)
    metrics = _one_update(trainer, update=1)

    assert "grad_norm/top_level/policy" in metrics
    assert "grad_norm/top_level/next_state" not in metrics


def test_behavior_cloning_off_and_sigreg_off_produce_no_terms(tmp_path) -> None:
    trainer = _diagnostic_trainer(tmp_path, "top_level")
    assert trainer._behavior_cloning_coef == 0.0
    assert trainer._schedule_state.sigreg_coef == 0.0
    metrics = _one_update(trainer, update=1)

    assert "grad_norm/top_level/bc" not in metrics
    assert "grad_norm/top_level/sigreg" not in metrics


def test_a_reward_component_scheduled_to_zero_is_not_logged(tmp_path) -> None:
    """A component with no weight contributes no gradient; an empty series is noise."""
    trainer = _diagnostic_trainer(tmp_path, "reward_policy")
    silenced = trainer._active_names[0]
    for component in trainer.wrapper.active_components:
        if component.name == silenced:
            component.weight = 0.0
    trainer.wrapper.refresh_component_weights()

    metrics = _one_update(trainer, update=1)
    assert f"grad_norm/reward_policy/{silenced}" not in metrics
    assert any(key.startswith("grad_norm/reward_policy/") for key in metrics)


def test_no_series_is_logged_for_a_component_the_environment_does_not_have(tmp_path) -> None:
    """Every reward series names a component the live registry actually emits."""
    trainer = _diagnostic_trainer(tmp_path, "reward_full", zone_capture_weight=0.0)
    metrics = _one_update(trainer, update=1)

    active = set(trainer._active_names)
    assert active < set(REWARD_COMPONENT_NAMES), "the fixture should not activate every component"
    logged = {
        key.rsplit("/", 1)[1]
        for key in metrics
        if key.startswith(("grad_norm/reward_policy/", "grad_norm/reward_value/"))
    }
    assert logged
    assert logged <= active
