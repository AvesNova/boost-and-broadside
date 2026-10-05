"""The belief carries how uncertain it is, and the encoder reads it in the code.

``time_since_observation`` says only how old an estimate is. The belief's
fourteen physical spreads say what that cost, and they reach the trunk by
smoothing the hidden ship's code (§8.3): the encoder reads exactly the
categorical input the next-state head takes as its baseline.
"""

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import ObsKey, observation_from_state
from boost_and_broadside.train.rl.features import build_standard_coordinator
from boost_and_broadside.train.rl.physical_belief import (
    PHYSICAL_MEAN_DIM,
    PHYSICAL_UNCERTAINTY_DIM,
    POSITION_SIGMA,
    physical_means_from_state,
)
from tests.train.test_belief import _HIDDEN_X, _SHIP, _env, _hold, _place, _tracker, _view


@pytest.fixture
def coordinator():
    return build_standard_coordinator(ShipConfig())


def _entropy(code: torch.Tensor) -> torch.Tensor:
    return -(code * torch.log(code.clamp_min(1e-12))).sum(-1)


class TestLayout:
    def test_the_channel_width_is_the_physical_spread_layout(self):
        env = _env()
        view = _view(env, _tracker())
        assert view[ObsKey.BELIEF_UNCERTAINTY].shape[-1] == PHYSICAL_UNCERTAINTY_DIM

    def test_map_objects_carry_no_spread(self):
        """Static geometry is not believed, so nothing about it is in doubt."""
        env = _env()
        uncertainty = _view(env, _tracker())[ObsKey.BELIEF_UNCERTAINTY]
        assert uncertainty.shape[1] > 2, "the view carries map tokens as well as ships"
        assert (uncertainty[0, 2:] == 0).all()


class TestTheEncoderReadsTheCode:
    def test_a_visible_ship_reads_the_exact_code_of_its_truth(self, coordinator):
        env = _env()
        view = _view(env, _tracker())
        codes = coordinator.ship_codes(view, 2)
        truth = physical_means_from_state(env.state)[:, :2]
        torch.testing.assert_close(codes, coordinator.ship_codec.sharp(truth))

    def test_a_view_built_without_a_belief_reads_sharp_codes(self, coordinator):
        env = _env()
        view = observation_from_state(env.state, _SHIP)
        assert ObsKey.BELIEF_UNCERTAINTY not in view.data
        truth = physical_means_from_state(env.state)[:, :2]
        torch.testing.assert_close(
            coordinator.ship_codes(view, 2), coordinator.ship_codec.sharp(truth)
        )

    def test_a_vaguer_belief_reads_as_a_smoother_code(self, coordinator):
        def hidden_code(position_sigma: float) -> torch.Tensor:
            env = _env()
            tracker = _tracker()
            _view(env, tracker)
            forecast = _hold(tracker)
            forecast[0, 1, PHYSICAL_MEAN_DIM + POSITION_SIGMA] = position_sigma
            tracker.advance(forecast)
            _place(env, enemy_x=_HIDDEN_X)
            return coordinator.ship_codes(_view(env, tracker), 2)[0, 1, :81]

        sharp, vague = hidden_code(0.0), hidden_code(50.0)
        assert _entropy(vague.view(9, 9)).sum() > _entropy(sharp.view(9, 9)).sum()

    def test_the_input_vector_carries_the_code(self, coordinator):
        env = _env()
        view = _view(env, _tracker())
        vector = coordinator.get_input_vector(view)
        codes = coordinator.ship_codes(view, 2)
        (start, stop), (state_start, state_stop) = coordinator.sparse_code_columns()
        torch.testing.assert_close(vector[0, :2, start:stop], codes[0, :, :81])
        torch.testing.assert_close(vector[0, :2, state_start:state_stop], codes[0, :, 81:])
