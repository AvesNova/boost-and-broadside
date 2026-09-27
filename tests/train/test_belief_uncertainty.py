"""The belief carries how uncertain it is, not just how long it has been stale.

``time_since_observation`` says only how old an estimate is. The next-state head
states a spread per channel -- eleven log sigmas plus a correlation latent for
position and one for velocity -- and the belief carries the head's latest claim
outright rather than a running sum, because the head saw the current spread as an
input and answered with the next one.
"""

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import ObsKey
from boost_and_broadside.train.rl.features import build_standard_coordinator
from boost_and_broadside.train.rl.physical_belief import (
    CERTAIN_LOG_SIGMA,
    LOG_SIGMA_COLUMNS,
    NEXT_STATE_OUTPUT_DIM,
    PHYSICAL_MEAN_DIM,
    PHYSICAL_UNCERTAINTY_DIM,
    UNKNOWN_LOG_SIGMA,
)
from tests.train.test_belief import _HIDDEN_X, _env, _place, _tracker, _view


@pytest.fixture
def coordinator():
    return build_standard_coordinator(ShipConfig())


def _uncertainty_accessor(coordinator):
    (feature,) = [f for f in coordinator.features if f.name == "belief_uncertainty"]
    return feature.accessor


def _prediction(log_sigma: float) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(1, 2, PHYSICAL_MEAN_DIM),
            torch.full((1, 2, PHYSICAL_UNCERTAINTY_DIM), log_sigma),
        ],
        dim=-1,
    )


class TestLayout:
    def test_the_channel_width_is_the_physical_uncertainty_layout(self, coordinator):
        assert _uncertainty_accessor(coordinator).absent_width == PHYSICAL_UNCERTAINTY_DIM

    def test_the_width_no_longer_follows_the_world_size(self):
        """It is thirteen physical terms, not one per Fourier harmonic.

        The previous layout reported one spread per position harmonic, so the
        width moved with the map. The next-state model predicts physical
        quantities, so it does not -- and the encoder's input width stops being a
        function of the world.
        """
        from dataclasses import replace

        for side in (1024.0, 65536.0):
            config = replace(ShipConfig(), world_size=(side, side))
            coordinator = build_standard_coordinator(config)
            assert _uncertainty_accessor(coordinator).absent_width == PHYSICAL_UNCERTAINTY_DIM

    def test_map_objects_carry_the_observed_spread(self, coordinator):
        """Static geometry is not believed, so nothing about it is in doubt."""
        env = _env()
        view = _view(env, _tracker())
        uncertainty = view[ObsKey.BELIEF_UNCERTAINTY]
        assert uncertainty.shape[-1] == PHYSICAL_UNCERTAINTY_DIM
        assert uncertainty.shape[1] > 2, "the view carries map tokens as well as ships"
        map_spreads = uncertainty[0, 2:][:, list(LOG_SIGMA_COLUMNS)]
        assert (map_spreads == CERTAIN_LOG_SIGMA).all()

    def test_a_view_built_without_a_belief_reports_none(self, coordinator):
        """An omniscient or fixture view has forecast nothing.

        Read through the accessor: such a view carries no channel at all, because
        the environment does not own the belief's representation.
        """
        from boost_and_broadside.env.observation import observation_from_state

        env = _env()
        view = observation_from_state(env.state, ShipConfig())
        assert ObsKey.BELIEF_UNCERTAINTY not in view.data
        supplied = _uncertainty_accessor(coordinator).get(view)
        assert supplied.shape[-1] == PHYSICAL_UNCERTAINTY_DIM
        assert supplied.abs().max() == 0.0


class TestSemantics:
    def test_the_head_restates_the_spread_rather_than_accumulating_it(self):
        """Four steps of the same claim must read the same, not four times it."""
        env = _env()
        tracker = _tracker()
        _view(env, tracker)

        readings = []
        for _ in range(4):
            tracker.advance(_prediction(1.0))
            _place(env, enemy_x=_HIDDEN_X)
            readings.append(_view(env, tracker)[ObsKey.BELIEF_UNCERTAINTY][0, 1].clone())

        for reading in readings:
            spreads = reading[list(LOG_SIGMA_COLUMNS)]
            assert torch.allclose(spreads, torch.ones_like(spreads))

    def test_a_confident_forecast_reports_less_doubt_than_a_vague_one(self):
        def after_one_step(log_sigma: float) -> torch.Tensor:
            env = _env()
            tracker = _tracker()
            _view(env, tracker)
            tracker.advance(_prediction(log_sigma))
            _place(env, enemy_x=_HIDDEN_X)
            return _view(env, tracker)[ObsKey.BELIEF_UNCERTAINTY][0, 1]

        assert (after_one_step(-2.0) <= after_one_step(2.0)).all()

    def test_an_unobserved_slot_reports_the_ceiling_and_an_observed_one_the_floor(self):
        env = _env()
        _place(env, enemy_x=_HIDDEN_X)
        view = _view(env, _tracker())
        never_seen = view[ObsKey.BELIEF_UNCERTAINTY][0, 1][list(LOG_SIGMA_COLUMNS)]
        observed = view[ObsKey.BELIEF_UNCERTAINTY][0, 0][list(LOG_SIGMA_COLUMNS)]
        assert (never_seen == UNKNOWN_LOG_SIGMA).all()
        assert (observed == CERTAIN_LOG_SIGMA).all()

    def test_the_encoder_sees_the_clamped_range_inside_one_unit(self, coordinator):
        """The input encoding divides the log sigmas by their own clamp bound.

        A symlog of a log would compress twice and flatten the difference between
        a ship in sight and one unseen for a minute, which is the whole signal.
        """
        (feature,) = [f for f in coordinator.features if f.name == "belief_uncertainty"]
        extreme = torch.full((1, 1, PHYSICAL_UNCERTAINTY_DIM), UNKNOWN_LOG_SIGMA)
        encoded = feature.input_encoder(extreme)[0, 0]
        spreads = encoded[list(LOG_SIGMA_COLUMNS)]
        assert torch.allclose(spreads, torch.ones_like(spreads))

    def test_the_head_cannot_claim_a_spread_outside_its_clamp(self):
        from boost_and_broadside.models.yemong.policy import NextStateHead

        head = NextStateHead(16)
        with torch.no_grad():
            for layer in head.net:
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.fill_(50.0)
        out = head(torch.randn(8, 16) * 20.0)
        uncertainty = out[..., PHYSICAL_MEAN_DIM:]
        assert uncertainty.shape[-1] == PHYSICAL_UNCERTAINTY_DIM
        assert torch.equal(uncertainty, uncertainty.clamp(-6.0, 6.0))
        assert out.shape[-1] == NEXT_STATE_OUTPUT_DIM
