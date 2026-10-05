"""The belief reaches the trunk through the ordinary encoder, not a substitution.

A hidden ship's belief is physical state in the same units as truth, so the view
composed from it is an ordinary observation and the encoder reads it without
knowing which slots came from where. These pin that -- including that the encoded
input of a believed ship is exactly what truth at the believed state would give,
and that the spatial rotation follows the believed position.
"""

import torch

from boost_and_broadside.config import ModelConfig
from boost_and_broadside.env.observation import ObsKey, observation_from_state
from boost_and_broadside.models.yemong.encoder import ShipEncoder
from boost_and_broadside.train.rl.features import FeatureScope, build_standard_coordinator
from boost_and_broadside.train.rl.physical_belief import POSITION_X
from tests.train.test_belief import (
    _HIDDEN_X,
    _SHIP,
    _VISIBLE_X,
    _env,
    _hold,
    _place,
    _tracker,
    _view,
)


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
        prediction = _hold(tracker)
        prediction[0, 1, POSITION_X] += 10.0
        tracker.advance(prediction)

        _place(env, enemy_x=_HIDDEN_X)
        believed = _view(env, tracker)
        assert believed[ObsKey.POS][0, 1, 0] == _VISIBLE_X + 10.0

        # Put truth where the belief thinks the ship is and encode that instead.
        _place(env, enemy_x=_VISIBLE_X + 10.0)
        truthful = observation_from_state(env.state, _SHIP)

        mine = _encoded(coordinator, believed)[0, 1]
        theirs = _encoded(coordinator, truthful)[0, 1]
        (start, stop), _ = coordinator.sparse_code_columns(FeatureScope.SHIP)
        torch.testing.assert_close(mine[start:stop], theirs[start:stop])

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
        prediction = _hold(tracker)
        prediction[0, 1, POSITION_X] += 100.0
        tracker.advance(prediction)

        _place(env, enemy_x=_HIDDEN_X)
        believed = _view(env, tracker)
        _place(env, enemy_x=_VISIBLE_X + 100.0)
        truthful = observation_from_state(env.state, _SHIP)

        believed_tables = rotary.tables(believed[ObsKey.POS], believed[ObsKey.ATT])
        truth_tables = rotary.tables(truthful[ObsKey.POS], truthful[ObsKey.ATT])
        for mine, theirs in zip(believed_tables, truth_tables):
            torch.testing.assert_close(mine[0, 1], theirs[0, 1])


class TestDerivedFeatures:
    """A feature over several ships must read the legal view, not truth.

    ``local_presence`` is the case that matters: it is a sum over every ship a
    token is allowed to see, so computing it from privileged truth and masking
    afterwards would leak a hidden enemy's *position* through an aggregate the
    observer is entitled to have — just not that one.
    """

    def test_presence_follows_the_believed_position(self):
        coordinator = build_standard_coordinator(_SHIP, local_presence=True)
        (presence,) = [f for f in coordinator.features if f.name == "local_presence"]

        env = _env()
        tracker = _tracker()
        _view(env, tracker)
        # Send the remembered enemy a long way off, and truth somewhere else.
        prediction = _hold(tracker)
        prediction[0, 1, POSITION_X] += 400.0
        tracker.advance(prediction)
        _place(env, enemy_x=_HIDDEN_X)
        believed = _view(env, tracker)

        _place(env, enemy_x=_VISIBLE_X + 400.0)
        at_belief = observation_from_state(env.state, _SHIP)
        _place(env, enemy_x=_HIDDEN_X)
        at_truth = observation_from_state(env.state, _SHIP)

        mine = presence.get_input(believed)[0, 0]
        torch.testing.assert_close(mine, presence.get_input(at_belief)[0, 0])
        assert not torch.allclose(mine, presence.get_input(at_truth)[0, 0]), (
            "the aggregate is reading truth, not the composed view"
        )

    def test_a_never_seen_ship_contributes_nothing(self):
        coordinator = build_standard_coordinator(_SHIP, local_presence=True)
        (presence,) = [f for f in coordinator.features if f.name == "local_presence"]

        env = _env()
        _place(env, enemy_x=_HIDDEN_X)
        view = _view(env, _tracker())
        assert not view[ObsKey.BELIEF_VALID][0, 1]
        assert presence.get_input(view)[0, 0, 1] == 0.0, "enemy presence from a ship never seen"
