"""The belief reaches the trunk through the ordinary encoder, not a substitution.

A hidden ship's belief is physical state in the same units as truth, so the view
composed from it is an ordinary observation and the encoder reads it without
knowing which slots came from where. These pin that -- including that the encoded
input of a believed ship is exactly what truth at the believed state would give,
and that the spatial rotation follows the believed position -- and that the old
encoded-substitution channels never appear.
"""

import torch

from boost_and_broadside.config import ModelConfig
from boost_and_broadside.env.observation import ObsKey, observation_from_state
from boost_and_broadside.models.yemong.encoder import ShipEncoder
from boost_and_broadside.train.rl.features import FeatureScope, build_standard_coordinator
from boost_and_broadside.train.rl.physical_belief import (
    NEXT_STATE_OUTPUT_DIM,
    POSITION_X,
)
from tests.train.test_belief import _HIDDEN_X, _SHIP, _VISIBLE_X, _env, _place, _tracker, _view


def _encoded(coordinator, view):
    """The ship-scoped encoder input for every token in ``view``."""

    return coordinator.get_scoped_input_vector(view, FeatureScope.SHIP)


class TestEncodedInput:
    def test_a_believed_ship_encodes_exactly_as_truth_at_that_state_would(self):
        """No substitution, no decode: the same encoder over the same numbers.

        The forecast moves the remembered ship ten pixels. Truth teleported
        elsewhere, but an omniscient view placed at the *believed* coordinate
        must encode to the same vector -- which is only possible because belief
        and truth are the same physical quantity.
        """
        coordinator = build_standard_coordinator(_SHIP)
        env = _env()
        tracker = _tracker()
        _view(env, tracker)
        prediction = torch.zeros((1, 2, NEXT_STATE_OUTPUT_DIM))
        prediction[0, 1, POSITION_X] = 4.0  # 4 * 2.5 px
        tracker.advance(prediction)

        _place(env, enemy_x=_HIDDEN_X)
        believed = _view(env, tracker)
        assert believed[ObsKey.POS][0, 1, 0] == _VISIBLE_X + 10.0

        # Put truth where the belief thinks the ship is and encode that instead.
        _place(env, enemy_x=_VISIBLE_X + 10.0)
        truthful = observation_from_state(env.state, _SHIP)

        mine = _encoded(coordinator, believed)[0, 1]
        theirs = _encoded(coordinator, truthful)[0, 1]
        position = coordinator.features[0].input_encoder.out_dim(1)
        torch.testing.assert_close(mine[:position], theirs[:position])

    def test_the_encoder_accepts_a_composed_view_unchanged(self):
        """The trunk's first projection is width-stable across the new channels."""
        coordinator = build_standard_coordinator(_SHIP)
        config = ModelConfig(d_model=64, n_heads=4, n_yemong_blocks=2)
        encoder = ShipEncoder(config, coordinator, num_ships=2)
        env = _env()
        tokens = encoder(_view(env, _tracker()))
        assert tokens.shape[0] == 1
        assert tokens.shape[-1] == 64
        assert torch.isfinite(tokens).all()


class TestRotation:
    def test_the_spatial_rotation_follows_the_believed_position(self):
        """It reads ``ObsKey.POS``, which is now the belief's own coordinate.

        There is no separate moment-space rotary override any more: the belief is
        a coordinate, so the ordinary tables are the believed ones.
        """
        from boost_and_broadside.models.yemong.rope import SpatialRotary

        rotary = SpatialRotary(_SHIP, head_dim=32)
        env = _env()
        tracker = _tracker()
        _view(env, tracker)
        prediction = torch.zeros((1, 2, NEXT_STATE_OUTPUT_DIM))
        prediction[0, 1, POSITION_X] = 40.0  # 100 px
        tracker.advance(prediction)

        _place(env, enemy_x=_HIDDEN_X)
        believed = _view(env, tracker)
        _place(env, enemy_x=_VISIBLE_X + 100.0)
        truthful = observation_from_state(env.state, _SHIP)

        believed_tables = rotary.tables(believed[ObsKey.POS], believed[ObsKey.ATT])
        truth_tables = rotary.tables(truthful[ObsKey.POS], truthful[ObsKey.ATT])
        for mine, theirs in zip(believed_tables, truth_tables):
            torch.testing.assert_close(mine[0, 1], theirs[0, 1])


class TestTheOldPathIsInert:
    def test_no_production_view_carries_an_encoded_belief(self):
        """Phase 4 deletes the substitution machinery; nothing may reach it now."""
        env = _env()
        for view in (_view(env, _tracker()), observation_from_state(env.state, _SHIP)):
            assert ObsKey.BELIEF_TARGETS not in view.data
            assert ObsKey.BELIEF_SUBSTITUTE not in view.data

    def test_the_encoder_override_is_a_no_op_without_those_channels(self):
        coordinator = build_standard_coordinator(_SHIP)
        env = _env()
        view = _view(env, _tracker())
        raw = torch.cat(
            [
                feature.get_input(view)
                for feature in coordinator.features
                if feature.scope in (FeatureScope.SHARED, FeatureScope.SHIP)
            ],
            dim=-1,
        )
        torch.testing.assert_close(_encoded(coordinator, view), raw)
