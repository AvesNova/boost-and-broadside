"""The belief reaches the trunk as moments, not as a decoded point.

A hidden ship's position is a Fourier moment whose magnitude says how sure the
belief is. Routing it through a coordinate and back puts every harmonic on the
unit circle again, which reports maximum confidence whatever the belief actually
said -- so the substitution happens *after* encoding, in the space the head
predicts in. These pin that path and the spatial rotation that reads the same
numbers.
"""

import math

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import ObsKey, YemongObservation
from boost_and_broadside.models.yemong.rope import SpatialRotary, apply_rotary
from boost_and_broadside.train.rl.belief import BeliefTracker
from boost_and_broadside.train.rl.features import (
    Accessor,
    Feature,
    FeatureCoordinator,
    FeatureScope,
    Fourier,
    FourierMomentPredictor,
    Identity,
    build_standard_coordinator,
)
from tests.train.test_belief import _hold, _view


@pytest.fixture
def coordinator():
    return build_standard_coordinator(ShipConfig())


def _encode(coordinator, obs):
    """Ship-scoped encoding.

    The compact fixtures here carry no field channels, and the ship scope is the
    one the substitution matters for anyway -- it is also the path that shifts
    every offset after an omitted feature, so it exercises the harder index map.
    """

    return coordinator.get_scoped_input_vector(obs, FeatureScope.SHIP)


def _position_columns(coordinator):
    """(input columns, target columns) of position_x in the ship input vector."""
    inputs, targets = coordinator._override_columns(FeatureScope.SHIP, torch.device("cpu"))
    spec = {s.name: s for s in coordinator._predictor_specs}["position_x"]
    wanted = range(spec.t_offset, spec.t_offset + spec.t_dim)
    picked = [i for i, t in enumerate(targets.tolist()) if t in wanted]
    return inputs[picked], targets[picked]


def _magnitudes(block: torch.Tensor) -> torch.Tensor:
    harmonics = block.shape[-1] // 2
    sines, cosines = block[..., :harmonics], block[..., harmonics:]
    return (sines * sines + cosines * cosines).sqrt()


def _hidden_belief_obs(coordinator, magnitude: float):
    """Compose an observation whose one hidden ship believes at ``magnitude``."""

    tracker = BeliefTracker(1, 2, 0.1, coordinator, "cpu")
    seen = tracker.compose(_view(visible=True, x=300.0))
    prediction = _hold(coordinator, seen)
    # Scale the position block down: a belief that has gone vague.
    spec = {s.name: s for s in coordinator._predictor_specs}["position_x"]
    columns = slice(spec.p_offset, spec.p_offset + spec.p_dim)
    prediction[..., columns] = prediction[..., columns] * magnitude
    tracker.advance(seen, prediction)
    return tracker.compose(_view(visible=False, x=9999.0))


def test_a_vague_belief_reaches_the_encoder_with_its_magnitude_intact(coordinator) -> None:
    """The whole point of substituting after encoding rather than before."""

    hidden = _hidden_belief_obs(coordinator, magnitude=0.3)
    encoded = _encode(coordinator, hidden)
    input_columns, _ = _position_columns(coordinator)

    believed = encoded[0, 1].index_select(-1, input_columns)
    assert _magnitudes(believed).max().item() == pytest.approx(0.3, abs=1e-3)


def test_decoding_the_same_belief_would_have_claimed_certainty(coordinator) -> None:
    """The regression this design exists to prevent, stated as a measurement.

    A decode reads a phase and throws the magnitude away; re-encoding restores a
    unit vector. The belief says "roughly here"; the round trip says "here".
    """

    hidden = _hidden_belief_obs(coordinator, magnitude=0.3)
    # ObsKey.POS still carries the decoded point, for consumers that need one.
    round_tripped = Fourier(n_freqs=4, periods=1024.0)(hidden[ObsKey.POS][0, 1, 0:1])
    assert _magnitudes(round_tripped).max().item() == pytest.approx(1.0, abs=1e-3)


def test_total_ignorance_encodes_as_zeros_not_as_the_origin(coordinator) -> None:
    """A zero moment is a uniform belief, which no coordinate can express."""

    hidden = _hidden_belief_obs(coordinator, magnitude=0.0)
    encoded = _encode(coordinator, hidden)
    input_columns, _ = _position_columns(coordinator)
    assert encoded[0, 1].index_select(-1, input_columns).abs().max().item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_a_visible_ship_is_untouched_by_the_substitution(coordinator) -> None:
    """Truth encodes to unit magnitude, and the mask must leave it alone."""

    tracker = BeliefTracker(1, 2, 0.1, coordinator, "cpu")
    seen = tracker.compose(_view(visible=True, x=300.0))
    with_belief = _encode(coordinator, seen)

    stripped = YemongObservation(
        data={k: v for k, v in seen.items() if k is not ObsKey.BELIEF_TARGETS}
    )
    assert torch.allclose(with_belief, _encode(coordinator, stripped))


def test_an_observation_without_a_tracker_encodes_unchanged(coordinator) -> None:
    """Every caller without a belief: the raw env view, a fixture, evaluation."""

    view = _view(visible=True, x=300.0)
    assert ObsKey.BELIEF_TARGETS not in view.data
    _encode(coordinator, view)  # must not raise


def test_a_predicted_feature_whose_encoders_disagree_is_refused() -> None:
    """The invariant the copy rests on, caught at construction.

    Target space *is* input space for a predicted feature, which is what lets the
    belief be copied column for column. A mismatch would otherwise surface much
    later as a scatter that looks correct.
    """

    with pytest.raises(ValueError, match="input and target encoders must match"):
        FeatureCoordinator(
            [
                Feature(
                    name="position_x",
                    accessor=Accessor(ObsKey.POS, channels=[0]),
                    input_encoder=Fourier(n_freqs=4, periods=1024.0),
                    target_encoder=Identity(),
                    predictor=FourierMomentPredictor(),
                )
            ]
        )


class TestUnitDisk:
    """A Fourier moment is an expectation of a unit vector; it cannot exceed one."""

    def test_an_impossible_moment_is_projected_and_keeps_its_phase(self, coordinator) -> None:
        slices = coordinator.target_slices()
        targets = torch.zeros(1, 1, coordinator.total_target_dimension)
        block = targets[0, 0, slices["position_x"]]
        harmonics = block.numel() // 2
        targets[0, 0, slices["position_x"]] = torch.cat(
            [torch.full((harmonics,), 3.0), torch.full((harmonics,), 4.0)]
        )

        projected = coordinator.project_targets(targets)[0, 0, slices["position_x"]]
        assert _magnitudes(projected).max().item() == pytest.approx(1.0, abs=1e-5)
        # 3-4-5 triangle: the direction is unchanged, only the length.
        assert (projected[0] / projected[harmonics]).item() == pytest.approx(0.75, abs=1e-5)

    def test_a_legitimately_vague_moment_is_left_alone(self, coordinator) -> None:
        slices = coordinator.target_slices()
        targets = torch.zeros(1, 1, coordinator.total_target_dimension)
        harmonics = (slices["position_x"].stop - slices["position_x"].start) // 2
        targets[0, 0, slices["position_x"]] = torch.cat(
            [torch.full((harmonics,), 0.18), torch.full((harmonics,), 0.24)]
        )
        projected = coordinator.project_targets(targets)[0, 0, slices["position_x"]]
        assert _magnitudes(projected).max().item() == pytest.approx(0.3, abs=1e-5)

    def test_unbounded_channels_keep_the_numerical_ceiling(self, coordinator) -> None:
        """Symlog space has no natural bound, so its guard is still a guard."""

        from boost_and_broadside.train.rl.belief import BELIEF_TARGET_LIMIT

        slices = coordinator.target_slices()
        targets = torch.zeros(1, 1, coordinator.total_target_dimension)
        targets[0, 0, slices["velocity"]] = torch.tensor([1e6, -1e6])
        projected = coordinator.project_targets(targets)[0, 0, slices["velocity"]]
        assert projected.tolist() == [BELIEF_TARGET_LIMIT, -BELIEF_TARGET_LIMIT]


class TestRotaryFromMoments:
    """The rotation reads the same numbers the encoder does."""

    def _rotary(self, ship_config):
        return SpatialRotary(ship_config, head_dim=64)

    def test_truth_moments_reproduce_the_coordinate_tables_exactly(self, coordinator) -> None:
        """Encoder basis and rotary frequencies are one basis, so this is identity.

        Not approximately: both are ``base2_frequencies`` of the same period, so
        a token's moment vector *is* the table a known coordinate would build.
        """

        ship_config = ShipConfig()
        rotary = self._rotary(ship_config)
        encoders = {s.name: s.target_encoder for s in coordinator._predictor_specs}
        position = torch.tensor([[[300.0, 412.0]]])
        attitude = torch.tensor([[[0.6, 0.8]]])

        cos_point, sin_point = rotary.tables(position, attitude)
        cos_moment, sin_moment = rotary.tables_from_moments(
            encoders["position_x"](position[..., 0:1]),
            encoders["position_y"](position[..., 1:2]),
            encoders["attitude"](attitude),
        )
        assert torch.equal(cos_point, cos_moment)
        assert torch.equal(sin_point, sin_moment)

    def test_an_unnormalised_moment_gives_the_expected_logit(self) -> None:
        """Why the moments are fed unnormalised, as a measurement.

        ``apply_rotary`` is linear in its table, so a logit is bilinear in the two
        tokens' tables and its expectation over independent beliefs is the form of
        the expectations. Feeding ``(E cos, E sin)`` therefore yields ``E[logit]``
        exactly; normalising first yields the logit of the mean, which overstates
        how well the geometry is known.
        """

        torch.manual_seed(0)
        heads, head_dim, pairs = 2, 8, 2
        query = torch.randn(1, 1, heads, head_dim)
        key = torch.randn(1, 1, heads, head_dim)
        kappa, mu, key_angle = 2.0, 0.7, 0.3
        resultant = float(
            torch.special.i1e(torch.tensor(kappa)) / torch.special.i0e(torch.tensor(kappa))
        )

        samples = 200_000
        angles = torch.distributions.VonMises(torch.tensor(mu), torch.tensor(kappa)).sample(
            (samples,)
        )

        def table(values, count):
            expanded = values.reshape(-1, 1, 1, 1).expand(-1, 1, 1, pairs)
            return expanded.cos(), expanded.sin()

        cos_k, sin_k = table(torch.tensor([key_angle]), 1)
        rotated_key = apply_rotary(key, cos_k, sin_k)
        cos_q, sin_q = table(angles, samples)
        sampled = (
            (apply_rotary(query.expand(samples, 1, heads, head_dim), cos_q, sin_q) * rotated_key)
            .sum(-1)
            .mean()
            .item()
        )

        def logit_with(radius: float) -> float:
            cos = torch.full((1, 1, 1, pairs), radius * math.cos(mu))
            sin = torch.full((1, 1, 1, pairs), radius * math.sin(mu))
            return (apply_rotary(query, cos, sin) * rotated_key).sum(-1).mean().item()

        assert logit_with(resultant) == pytest.approx(sampled, rel=0.02)
        # And the normalised alternative is materially different, so this is a
        # real choice rather than a distinction without one.
        assert logit_with(1.0) != pytest.approx(sampled, rel=0.10)

    def test_a_vague_belief_weakens_the_positional_term(self, coordinator) -> None:
        """The attenuation is `r_q * r_k`, which is the Bayesian factor itself."""

        torch.manual_seed(0)
        rotary = self._rotary(ShipConfig())
        encoders = {s.name: s.target_encoder for s in coordinator._predictor_specs}
        position = torch.tensor([[[300.0, 412.0]]])
        attitude = torch.tensor([[[1.0, 0.0]]])
        moments = (
            encoders["position_x"](position[..., 0:1]),
            encoders["position_y"](position[..., 1:2]),
            encoders["attitude"](attitude),
        )
        query = torch.randn(1, 1, 2, 64)
        key = torch.randn(1, 1, 2, 64)

        def logit(radius: float) -> float:
            cos, sin = rotary.tables_from_moments(*(m * radius for m in moments))
            certain_cos, certain_sin = rotary.tables_from_moments(*moments)
            return (
                (apply_rotary(query, cos, sin) * apply_rotary(key, certain_cos, certain_sin))
                .sum(-1)
                .mean()
                .item()
            )

        confident, vague, ignorant = logit(1.0), logit(0.4), logit(0.0)
        assert abs(vague - ignorant) < abs(confident - ignorant)
        # Total ignorance contributes nothing through the rotated dimensions, so
        # the logit is whatever the unrotated dimensions say on their own.
        cos_zero, sin_zero = rotary.tables_from_moments(*(m * 0.0 for m in moments))
        assert cos_zero.abs().max().item() == 0.0
