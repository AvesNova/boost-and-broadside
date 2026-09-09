"""Numeric refractive-field observation and team-flip integration tests."""

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.field_physics import material_tensors, refresh_ship_field_cache
from boost_and_broadside.env.observation import ObsKey, observation_from_state
from boost_and_broadside.train.rl.features import build_standard_coordinator
from tests.conftest import make_state


def _overlapping_observation():
    config = ShipConfig(world_size=(512.0, 512.0), field_radius_max=200.0)
    state = make_state(
        num_envs=1,
        max_ships=2,
        max_bullets=0,
        ship_config=config,
        num_fields=2,
    )
    state.field_pos[:] = torch.tensor([[256.0 + 256.0j, 256.0 + 256.0j]])
    state.field_radius[:] = torch.tensor([[140.0, 50.0]])
    state.field_transition_width[:] = 20.0
    state.field_index_level[:] = torch.tensor([[1, -2]], dtype=torch.int8)
    state.field_damage_level[:] = torch.tensor([[1, 2]], dtype=torch.int8)
    index, damage = material_tensors(
        state.field_index_level,
        state.field_damage_level,
        config,
    )
    state.field_index[:] = index
    state.field_damage[:] = damage
    state.ship_pos[:] = torch.tensor([[256.0 + 256.0j, 10.0 + 10.0j]])
    refresh_ship_field_cache(state, config)
    return config, observation_from_state(state, config)


def test_field_material_features_and_ship_local_index_are_numeric_and_bounded():
    config, obs = _overlapping_observation()
    assert obs.pos.shape == (1, 4, 2)
    assert obs.team_id[0, 2:].tolist() == [2, 2]
    assert obs.alive[0, 2:].tolist() == [True, True]

    # Absolute target log encoding is k/2. At the shared center the HIGH and
    # VERY_LOW targets blend to exponent -0.5, hence normalized value -0.25.
    assert obs[ObsKey.FIELD_TARGET_LOG_INDEX][0, 2:, 0].tolist() == pytest.approx([0.5, -1.0])
    assert obs[ObsKey.FIELD_DAMAGE][0, 2:, 0].tolist() == pytest.approx([0.5, 1.0])
    assert obs.local_log_index[0, :, 0].tolist() == pytest.approx([-0.25, 0.0, 0.0, 0.0])

    coordinator = build_standard_coordinator(config)
    encoded = coordinator.get_input_vector(obs)
    assert encoded.shape[:2] == (1, 4)
    assert torch.isfinite(encoded).all()

    radius_feature = next(feature for feature in coordinator.features if feature.name == "radius")
    encoded_radius = radius_feature.get_input(obs)
    expected_scale = min(config.world_size) / 2.0
    assert encoded_radius[0, :, 0].tolist() == pytest.approx(
        [
            config.collision_radius / expected_scale,
            config.collision_radius / expected_scale,
            140.0 / expected_scale,
            50.0 / expected_scale,
        ]
    )


def test_team_flip_changes_only_ship_team_ids_and_preserves_field_properties():
    _, obs = _overlapping_observation()
    flipped = obs.flip_team(num_ships=2)
    for key, value in obs.items():
        if key == ObsKey.TEAM_ID:
            assert torch.equal(flipped[key][:, 2:], value[:, 2:])
        else:
            assert torch.equal(flipped[key], value)


def test_local_index_is_registered_as_auxiliary_prediction_target():
    config, obs = _overlapping_observation()
    coordinator = build_standard_coordinator(config)
    assert "local_log_index" in coordinator.target_slices()
    targets = coordinator.get_target_vector(obs)
    labels = coordinator.compute_labels(targets, targets)
    local_prediction = coordinator.get_feature_names().index("local_log_index_0")
    assert torch.equal(
        labels[..., local_prediction], torch.zeros_like(labels[..., local_prediction])
    )
