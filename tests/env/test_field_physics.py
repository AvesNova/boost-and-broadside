"""Static refractive-field profile, hierarchy, cache, and reset tests."""

import math

import pytest
import torch

from boost_and_broadside.config import (
    EnvConfig,
    FieldMapConfig,
    InterfaceDamageLevel,
    RefractiveIndexLevel,
    ShipConfig,
)
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.field_cache import FieldMapCache
from boost_and_broadside.env.field_physics import (
    evaluate_field_profiles,
    evaluate_fields,
    material_tensors,
    validate_field_layout,
)


def _single_profile(points: list[complex], *, center: complex = 100.0 + 100.0j):
    p = torch.tensor([points], dtype=torch.complex64)
    c = torch.tensor([[center]], dtype=torch.complex64)
    radius = torch.tensor([[50.0]])
    width = torch.tensor([[20.0]])
    return evaluate_field_profiles(p, c, radius, width, (256.0, 256.0))


def test_quintic_profile_has_flat_core_outside_and_edges():
    alpha, gradient = _single_profile(
        [100.0 + 100.0j, 139.0 + 100.0j, 140.0 + 100.0j, 150.0 + 100.0j, 160.0 + 100.0j]
    )
    assert alpha[0, 0, 0] == 1.0
    assert alpha[0, 2, 0] == 1.0
    assert alpha[0, 3, 0] == pytest.approx(0.5)
    assert alpha[0, 4, 0] == 0.0
    assert gradient[0, 0, 0] == 0.0j
    assert gradient[0, 2, 0] == 0.0j
    assert gradient[0, 4, 0] == 0.0j


def test_analytic_gradient_matches_finite_difference_and_points_inward():
    alpha, gradient = _single_profile([150.0 + 100.0j])
    eps = 1e-2
    plus, _ = _single_profile([150.0 + eps + 100.0j])
    minus, _ = _single_profile([150.0 - eps + 100.0j])
    finite_diff = (plus - minus) / (2.0 * eps)
    assert gradient[0, 0, 0].real < 0.0
    assert gradient[0, 0, 0].imag == pytest.approx(0.0, abs=1e-7)
    assert gradient[0, 0, 0].real.item() == pytest.approx(finite_diff[0, 0, 0].item(), rel=2e-3)
    assert alpha[0, 0, 0] == pytest.approx(0.5)


def test_profile_wraps_across_toroidal_edge_and_center_is_finite():
    alpha, gradient = _single_profile([251.0 + 100.0j, 5.0 + 100.0j], center=251.0 + 100.0j)
    assert alpha[0, 0, 0] == 1.0
    assert alpha[0, 1, 0] == 1.0
    assert torch.isfinite(alpha).all()
    assert torch.isfinite(gradient.real).all()
    assert torch.isfinite(gradient.imag).all()


def _nested_materials(config: ShipConfig):
    center = torch.tensor([[256.0 + 256.0j] * 3], dtype=torch.complex64)
    radius = torch.tensor([[140.0, 80.0, 30.0]])
    width = torch.tensor([[20.0, 20.0, 20.0]])
    levels = torch.tensor([[1, -2, 2]], dtype=torch.int8)
    damage = torch.tensor([[0, 1, 2]], dtype=torch.int8)
    parent = validate_field_layout(center, radius, width, levels, damage, config.world_size)
    index, crossing_damage, delta = material_tensors(levels, damage, parent, config)
    return center, radius, width, parent, index, crossing_damage, delta


def test_direct_parent_and_telescoping_root_child_grandchild():
    config = ShipConfig(world_size=(512.0, 512.0), field_radius_max=200.0)
    center, radius, width, parent, index, crossing_damage, delta = _nested_materials(config)
    assert parent.tolist() == [[-1, 0, 1]]
    assert crossing_damage.tolist() == [[0.0, 10.0, 20.0]]

    points = torch.tensor([[376.0 + 256.0j, 316.0 + 256.0j, 256.0 + 256.0j]])
    evaluation = evaluate_fields(points, center, radius, width, delta, config.world_size)
    expected = torch.stack([index[0, 0], index[0, 1], index[0, 2]])
    assert torch.allclose(evaluation.index[0], expected, atol=1e-6)


def test_partial_overlap_and_overlapping_bands_are_rejected():
    center = torch.tensor([100.0 + 100.0j, 180.0 + 100.0j])
    radius = torch.tensor([50.0, 50.0])
    width = torch.tensor([20.0, 20.0])
    levels = torch.tensor([1, -1], dtype=torch.int8)
    damage = torch.tensor([0, 2], dtype=torch.int8)
    with pytest.raises(ValueError, match="disjoint or strictly nested"):
        validate_field_layout(center, radius, width, levels, damage, (512.0, 512.0))

    # Nominal circles are disjoint, but their complete transition bands overlap.
    center = torch.tensor([100.0 + 100.0j, 210.0 + 100.0j])
    with pytest.raises(ValueError, match="non-overlapping transition bands"):
        validate_field_layout(center, radius, width, levels, damage, (512.0, 512.0))


def test_toroidal_containment_selects_smallest_parent():
    center = torch.tensor([5.0 + 128.0j, 250.0 + 128.0j, 252.0 + 128.0j])
    radius = torch.tensor([100.0, 60.0, 20.0])
    width = torch.tensor([10.0, 10.0, 10.0])
    levels = torch.tensor([1, 2, -1], dtype=torch.int8)
    damage = torch.tensor([0, 1, 2], dtype=torch.int8)
    parent = validate_field_layout(center, radius, width, levels, damage, (256.0, 256.0))
    assert parent.tolist() == [-1, 0, 1]


def test_all_twelve_index_damage_combinations_are_representable():
    config = ShipConfig()
    levels = [-2, -1, 1, 2]
    damages = [0, 1, 2]
    pos = torch.tensor([[100.0 + 100.0j]] * 12)
    radius = torch.full((12, 1), 30.0)
    width = torch.full((12, 1), 20.0)
    index_level = torch.tensor([[level] for level in levels for _ in damages], dtype=torch.int8)
    damage_level = torch.tensor([[damage] for _ in levels for damage in damages], dtype=torch.int8)
    cache = FieldMapCache(pos, radius, width, index_level, damage_level, config)
    sampled = cache.sample(256, config.world_size, torch.device("cpu"))
    assert sampled[3].shape == (256, 1)
    assert set(index_level.flatten().tolist()) == {-2, -1, 1, 2}
    assert set(damage_level.flatten().tolist()) == {0, 1, 2}


def test_generated_maps_are_bounded_static_and_laminar():
    config = ShipConfig()
    env_config = EnvConfig(num_ships=2, max_bullets=0, max_episode_steps=10, num_fields=4)
    map_config = FieldMapConfig(cache_size=8, max_generation_attempts=256)
    cache = FieldMapCache.generate(config, env_config, map_config, torch.device("cpu"), seed=7)
    first = cache.sample(32, config.world_size, torch.device("cpu"))
    assert len(cache) == 8
    assert first[0].shape == (32, 4)
    validate_field_layout(first[0], first[1], first[2], first[3], first[5], config.world_size)


def test_generation_failure_is_bounded_and_clear():
    config = ShipConfig(
        world_size=(256.0, 256.0),
        field_radius_min=30.0,
        field_radius_max=30.0,
        field_transition_width_min=40.0,
        field_transition_width_max=40.0,
    )
    env_config = EnvConfig(num_ships=2, max_bullets=0, max_episode_steps=10, num_fields=10)
    with pytest.raises(RuntimeError, match="after 2 attempts per field"):
        FieldMapCache.generate(
            config,
            env_config,
            FieldMapConfig(
                cache_size=1,
                max_generation_attempts=2,
                nesting_probability=0.0,
            ),
            torch.device("cpu"),
            seed=1,
        )


def test_reset_uses_proper_speed_populates_cache_and_causes_no_damage():
    config = ShipConfig(random_speed=False, default_speed=100.0)
    env_config = EnvConfig(num_ships=2, max_bullets=0, max_episode_steps=10, num_fields=1)
    cache = FieldMapCache(
        torch.tensor([[512.0 + 512.0j]]),
        torch.tensor([[160.0]]),
        torch.tensor([[40.0]]),
        torch.tensor([[int(RefractiveIndexLevel.HIGH)]], dtype=torch.int8),
        torch.tensor([[int(InterfaceDamageLevel.SEVERE)]], dtype=torch.int8),
        config,
    )
    env = TensorEnv(128, config, env_config, "cpu", cache)
    env.reset(seed=3)
    proper_speed = env.state.ship_local_index * env.state.ship_vel.abs()
    assert torch.allclose(proper_speed, torch.full_like(proper_speed, 100.0), atol=2e-5)
    assert not env.state.ship_field_damage.any()
    assert torch.all(env.state.ship_local_index > 0.0)


def test_zero_field_baseline_allocates_empty_field_axis_and_ambient_index():
    config = ShipConfig()
    env_config = EnvConfig(num_ships=2, max_bullets=0, max_episode_steps=10, num_fields=0)
    env = TensorEnv(3, config, env_config, "cpu")
    env.reset(seed=1)
    assert env.state.ship_field_alpha.shape == (3, 2, 0)
    assert torch.equal(env.state.ship_local_index, torch.ones(3, 2))
    assert torch.allclose(env.state.ship_vel.abs(), torch.full((3, 2), config.default_speed))


def test_config_rejects_ambiguous_toroidal_field_extent():
    with pytest.raises(ValueError, match="toroidal limit"):
        ShipConfig(
            world_size=(256.0, 256.0),
            field_radius_max=120.0,
            field_transition_width_min=20.0,
            field_transition_width_max=20.0,
        )


def test_config_and_layout_require_a_nonempty_flat_core():
    with pytest.raises(ValueError, match="non-empty flat core"):
        ShipConfig(field_radius_min=20.0, field_transition_width_max=40.0)

    with pytest.raises(ValueError, match="flat core"):
        validate_field_layout(
            torch.tensor([100.0 + 100.0j]),
            torch.tensor([20.0]),
            torch.tensor([40.0]),
            torch.tensor([1], dtype=torch.int8),
            torch.tensor([0], dtype=torch.int8),
            (512.0, 512.0),
        )


def test_log_symmetric_index_levels_are_reciprocal():
    config = ShipConfig()
    levels = torch.tensor([[-2, -1, 1, 2]], dtype=torch.int8)
    parents = torch.full_like(levels, -1, dtype=torch.long)
    damage = torch.tensor([[0, 1, 1, 2]], dtype=torch.int8)
    index, crossing_damage, _ = material_tensors(levels, damage, parents, config)
    expected_index = torch.tensor([[0.5, 2.0**-0.5, 2.0**0.5, 2.0]])
    assert torch.allclose(index, expected_index)
    assert torch.equal(crossing_damage, torch.tensor([[0.0, 10.0, 10.0, 20.0]]))
    assert index[0, 0] * index[0, 3] == pytest.approx(1.0, rel=1e-6)
    assert index[0, 1] * index[0, 2] == pytest.approx(1.0, rel=1e-6)
    assert math.isclose(index[0, 2].item(), config.field_index_step, rel_tol=1e-6)


class TestFieldMapRefresh:
    """Maps are regenerated per rollout rather than drawn from a fixed bank."""

    @staticmethod
    def _cache(cache_size=64, num_fields=4):
        config = ShipConfig()
        env_config = EnvConfig(
            num_ships=8, max_bullets=0, max_episode_steps=10, num_fields=num_fields
        )
        map_config = FieldMapConfig(
            cache_size=cache_size, max_generation_attempts=64, nesting_probability=0.35
        )
        return config, FieldMapCache.generate(
            config, env_config, map_config, torch.device("cpu"), seed=3
        )

    def test_refresh_replaces_every_map(self):
        _, cache = self._cache()
        before = cache._pos.clone()
        cache.refresh()
        assert not torch.equal(before, cache._pos)
        assert cache.generation_failures.item() == 0.0

    def test_refreshed_maps_stay_strictly_laminar(self):
        """The generator is the only validity guarantee on the hot path, since
        validate_field_layout costs host syncs and raises."""
        config, cache = self._cache()
        for _ in range(10):
            cache.refresh()
            validate_field_layout(
                cache._pos,
                cache._radius,
                cache._transition_width,
                cache._index_level,
                cache._damage_level,
                config.world_size,
            )

    def test_refresh_never_synchronizes_on_a_failure_path(self):
        """generation_failures stays a device tensor so reading it is optional."""
        _, cache = self._cache()
        cache.refresh()
        assert isinstance(cache.generation_failures, torch.Tensor)
        assert cache.generation_failures.ndim == 0

    def test_zero_field_cache_refresh_is_a_noop(self):
        config = ShipConfig()
        env_config = EnvConfig(num_ships=4, max_bullets=0, max_episode_steps=10, num_fields=0)
        cache = FieldMapCache.generate(
            config,
            env_config,
            FieldMapConfig(cache_size=8, max_generation_attempts=8),
            torch.device("cpu"),
        )
        cache.refresh()
        assert cache.num_fields == 0

    def test_material_levels_never_include_ambient(self):
        """Level 0 is ambient and is not a legal field material."""
        _, cache = self._cache(cache_size=256)
        for _ in range(5):
            cache.refresh()
            assert (cache._index_level != 0).all()
            assert ((cache._damage_level >= 0) & (cache._damage_level <= 2)).all()
