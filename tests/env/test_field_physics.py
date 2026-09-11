"""Overlapping refractive-field composition, generation, and reset tests."""

import math
from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config import EnvConfig, InterfaceDamageLevel, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.field_generation import generate_field_layout
from boost_and_broadside.env.field_physics import (
    compose_refractive_index,
    evaluate_field_profiles,
    evaluate_fields,
    material_tensors,
    validate_field_layout,
    wrap_displacement,
)
from boost_and_broadside.env.frontline import (
    FRONTLINE_FIELD_RADIUS_MAX,
    FRONTLINE_WORLD_SIZE,
)
from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG


def _single_profile(points: list[complex], *, center: complex = 100.0 + 100.0j):
    return evaluate_field_profiles(
        torch.tensor([points], dtype=torch.complex64),
        torch.tensor([[center]], dtype=torch.complex64),
        torch.tensor([[50.0]]),
        torch.tensor([[20.0]]),
        (256.0, 256.0),
    )


def test_quintic_profile_has_flat_core_outside_and_edges():
    alpha, gradient = _single_profile(
        [100.0 + 100.0j, 140.0 + 100.0j, 150.0 + 100.0j, 160.0 + 100.0j]
    )
    assert alpha[0, :, 0].tolist() == pytest.approx([1.0, 1.0, 0.5, 0.0])
    assert gradient[0, 0, 0] == 0.0j
    assert gradient[0, 1, 0] == 0.0j
    assert gradient[0, 3, 0] == 0.0j


def test_profile_gradient_matches_finite_difference_and_wraps_toroidally():
    _, gradient = _single_profile([150.0 + 100.0j])
    eps = 1e-2
    plus, _ = _single_profile([150.0 + eps + 100.0j])
    minus, _ = _single_profile([150.0 - eps + 100.0j])
    finite_diff = (plus - minus) / (2.0 * eps)
    assert gradient[0, 0, 0].real.item() == pytest.approx(finite_diff[0, 0, 0].item(), rel=2e-3)
    wrapped, center_gradient = _single_profile(
        [5.0 + 100.0j, 251.0 + 100.0j], center=251.0 + 100.0j
    )
    assert wrapped[0, :, 0].tolist() == [1.0, 1.0]
    assert torch.isfinite(center_gradient.real).all()
    assert torch.isfinite(center_gradient.imag).all()


def test_single_field_uses_log_space_transition_law():
    alpha = torch.tensor([[[0.5]]])
    gradient = torch.zeros_like(alpha, dtype=torch.complex64)
    index, _ = compose_refractive_index(alpha, gradient, torch.tensor([[2.0]]))
    assert index.item() == pytest.approx(math.sqrt(2.0))


def test_identical_overlaps_reinforce_partial_coverage_without_overshoot():
    alpha = torch.tensor([[[0.5, 0.5]]])
    gradient = torch.zeros_like(alpha, dtype=torch.complex64)
    index, _ = compose_refractive_index(alpha, gradient, torch.tensor([[2.0, 2.0]]))
    assert index.item() == pytest.approx(2.0**0.75)
    assert 2.0**0.5 < index.item() < 2.0


def test_equal_reciprocal_fields_cancel_to_ambient_at_any_coverage():
    alpha = torch.tensor([[[0.2, 0.2], [1.0, 1.0]]])
    gradient = torch.tensor([[[0.1 + 0.2j, 0.1 + 0.2j]]]).expand(1, 2, 2)
    index, grad_index = compose_refractive_index(alpha, gradient, torch.tensor([[2.0, 0.5]]))
    assert torch.allclose(index, torch.ones_like(index), atol=1e-6)
    assert torch.allclose(grad_index, torch.zeros_like(grad_index), atol=1e-6)


def test_field_major_scan_matches_the_point_major_reference():
    """The exclusive products are a layout change, not an arithmetic one.

    ``compose_refractive_index`` scans the field axis field-major so CUDA can
    parallelise across points; the reference below is the point-major form it
    replaced. They multiply the same factors in the same order, so they may
    differ only by float32 reassociation inside the scan kernel.
    """
    generator = torch.Generator().manual_seed(7)
    alpha = torch.rand(64, 12, 6, generator=generator)
    # Exercise both saturated ends: a fully covered point makes ``1 - alpha``
    # exactly zero, which is the case the no-division formulation exists for.
    alpha[alpha > 0.85] = 1.0
    alpha[alpha < 0.15] = 0.0
    grad_alpha = torch.complex(
        torch.randn(64, 12, 6, generator=generator),
        torch.randn(64, 12, 6, generator=generator),
    )
    target_index = torch.rand(64, 6, generator=generator) + 0.5

    def reference(alpha, grad_alpha, target_index):
        log_target = torch.log(target_index).unsqueeze(1)
        weight = alpha.sum(dim=2)
        weighted_log = (alpha * log_target).sum(dim=2)
        grad_weight = grad_alpha.sum(dim=2)
        grad_weighted_log = (grad_alpha * log_target).sum(dim=2)
        remaining = 1.0 - alpha
        prefix = torch.cumprod(remaining, dim=2)
        suffix = torch.flip(torch.cumprod(torch.flip(remaining, dims=(2,)), dim=2), dims=(2,))
        ones = torch.ones_like(remaining[:, :, :1])
        product_before = torch.cat((ones, prefix[:, :, :-1]), dim=2)
        product_after = torch.cat((suffix[:, :, 1:], ones), dim=2)
        coverage = 1.0 - prefix[:, :, -1]
        grad_coverage = (grad_alpha * product_before * product_after).sum(dim=2)
        contributes = weight > 1e-8
        safe_weight = weight.clamp(min=1e-8)
        mean_log = torch.where(contributes, weighted_log / safe_weight, 0.0)
        grad_mean_log = torch.where(
            contributes, (grad_weighted_log - mean_log * grad_weight) / safe_weight, 0.0
        )
        local_log_index = coverage * mean_log
        grad_log_index = grad_coverage * mean_log + coverage * grad_mean_log
        index = torch.exp(local_log_index)
        return index.float(), (index * grad_log_index).to(torch.complex64)

    index, grad_index = compose_refractive_index(alpha, grad_alpha, target_index)
    ref_index, ref_grad = reference(alpha, grad_alpha, target_index)
    assert torch.allclose(index, ref_index, rtol=1e-6, atol=1e-6)
    assert torch.allclose(grad_index, ref_grad, rtol=1e-6, atol=1e-6)


def test_arbitrary_overlap_gradient_matches_finite_difference():
    centers = torch.tensor([[95.0 + 128.0j, 155.0 + 128.0j, 128.0 + 165.0j]])
    radii = torch.tensor([[55.0, 60.0, 48.0]])
    widths = torch.tensor([[50.0, 40.0, 36.0]])
    targets = torch.tensor([[2.0, 0.5, math.sqrt(2.0)]])

    def sample(x: float, y: float):
        return evaluate_fields(
            torch.tensor([[complex(x, y)]]), centers, radii, widths, targets, (256.0, 256.0)
        )

    result = sample(128.0, 128.0)
    eps = 1e-2
    dx = (sample(128.0 + eps, 128.0).index - sample(128.0 - eps, 128.0).index) / (2.0 * eps)
    dy = (sample(128.0, 128.0 + eps).index - sample(128.0, 128.0 - eps).index) / (2.0 * eps)
    assert result.grad_index.real.item() == pytest.approx(dx.item(), rel=4e-3, abs=2e-5)
    assert result.grad_index.imag.item() == pytest.approx(dy.item(), rel=4e-3, abs=2e-5)


def test_toroidal_overlap_composes_both_fields():
    result = evaluate_fields(
        torch.tensor([[0.0 + 100.0j]]),
        torch.tensor([[250.0 + 100.0j, 6.0 + 100.0j]]),
        torch.tensor([[20.0, 20.0]]),
        torch.tensor([[10.0, 10.0]]),
        torch.tensor([[2.0, 2.0]]),
        (256.0, 256.0),
    )
    assert result.alpha[0, 0].tolist() == [1.0, 1.0]
    assert result.index.item() == pytest.approx(2.0)


def test_layout_validation_allows_partial_coincident_and_nested_overlaps():
    validate_field_layout(
        torch.tensor([100.0 + 100.0j, 145.0 + 100.0j, 100.0 + 100.0j]),
        torch.tensor([60.0, 50.0, 25.0]),
        torch.tensor([20.0, 20.0, 10.0]),
        torch.tensor([1, -1, 2], dtype=torch.int8),
        torch.tensor([0, 1, 2], dtype=torch.int8),
        (512.0, 512.0),
    )


def test_layout_validation_still_rejects_invalid_individual_fields():
    with pytest.raises(ValueError, match="flat core"):
        validate_field_layout(
            torch.tensor([100.0 + 100.0j]),
            torch.tensor([20.0]),
            torch.tensor([40.0]),
            torch.tensor([1], dtype=torch.int8),
            torch.tensor([0], dtype=torch.int8),
            (512.0, 512.0),
        )
    with pytest.raises(ValueError, match="ambient is invalid"):
        validate_field_layout(
            torch.tensor([100.0 + 100.0j]),
            torch.tensor([30.0]),
            torch.tensor([20.0]),
            torch.tensor([0], dtype=torch.int8),
            torch.tensor([0], dtype=torch.int8),
            (512.0, 512.0),
        )


def test_all_index_and_damage_materials_are_representable():
    config = ShipConfig()
    levels = torch.tensor([[-2, -1, 1, 2]], dtype=torch.int8)
    damages = torch.tensor([[0, 1, 1, 2]], dtype=torch.int8)
    index, crossing_damage = material_tensors(levels, damages, config)
    assert index.tolist()[0] == pytest.approx([0.5, 2.0**-0.5, 2.0**0.5, 2.0])
    assert crossing_damage.tolist() == [[0.0, 10.0, 10.0, 20.0]]


def test_generation_is_direct_bounded_and_allows_overlap():
    config = ShipConfig()
    env_config = EnvConfig(num_ships=2, max_bullets=0, max_episode_steps=10, num_fields=32)
    layout = generate_field_layout(16, config, env_config, torch.device("cpu"))
    pos, radius, width, levels, index, damage_levels, damage = layout
    assert pos.shape == (16, 32)
    validate_field_layout(pos, radius, width, levels, damage_levels, config.world_size)
    assert torch.allclose(index, config.field_index_step ** levels.float())
    assert torch.allclose(damage, damage_levels.float() * config.field_interface_damage)
    displacement = wrap_displacement(pos[:, :, None] - pos[:, None, :], config.world_size).abs()
    outer = radius + 0.5 * width
    overlaps = displacement < outer[:, :, None] + outer[:, None, :]
    diagonal = torch.eye(32, dtype=torch.bool).unsqueeze(0)
    assert (overlaps & ~diagonal).any()


def test_reset_generates_new_maps_only_for_selected_environments():
    config = ShipConfig()
    env_config = EnvConfig(num_ships=2, max_bullets=0, max_episode_steps=10, num_fields=4)
    env = TensorEnv(3, config, env_config, "cpu")
    env.reset(seed=7)
    before = env.state.field_pos.clone()
    env.reset_envs(torch.tensor([False, True, False]))
    assert torch.equal(env.state.field_pos[[0, 2]], before[[0, 2]])
    assert not torch.equal(env.state.field_pos[1], before[1])


def test_frontline_fields_share_map_translation_and_fit_playable_boundary():
    config = ShipConfig(
        world_size=FRONTLINE_WORLD_SIZE,
        field_radius_max=FRONTLINE_FIELD_RADIUS_MAX,
    )
    env = TensorEnv(8, config, replace(PLAY_ENV_CONFIG, num_fields=16), "cpu")
    env.reset(seed=9)
    distance = wrap_displacement(
        env.state.field_pos - env.state.map_center.unsqueeze(1), config.world_size
    ).abs()
    outer = env.state.field_radius + 0.5 * env.state.field_transition_width
    assert torch.all(distance + outer <= env.state.playable_boundary_radius.unsqueeze(1) + 1e-4)
    assert env.state.field_radius.max() > 490.0


def test_frontline_generation_stratifies_area_instead_of_clumping_radially():
    count = 20
    config = ShipConfig(
        world_size=FRONTLINE_WORLD_SIZE,
        field_radius_max=FRONTLINE_FIELD_RADIUS_MAX,
    )
    env = TensorEnv(16, config, replace(PLAY_ENV_CONFIG, num_fields=count), "cpu")
    env.reset(seed=19)
    distance = wrap_displacement(
        env.state.field_pos - env.state.map_center.unsqueeze(1),
        config.world_size,
    ).abs()
    outer = env.state.field_radius + 0.5 * env.state.field_transition_width
    center_limit = env.state.playable_boundary_radius.unsqueeze(1) - outer
    area_fraction = (distance / center_limit).square().sort(dim=1).values
    stratum = torch.arange(count, dtype=torch.float32).view(1, count)
    assert torch.all(area_fraction >= stratum / count - 2e-5)
    assert torch.all(area_fraction <= (stratum + 1.0) / count + 2e-5)


def test_zero_field_fast_path_stays_ambient():
    config = ShipConfig()
    env = TensorEnv(
        3,
        config,
        EnvConfig(num_ships=2, max_bullets=0, max_episode_steps=10, num_fields=0),
        "cpu",
    )
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


def test_damage_level_enum_bounds_remain_zero_through_severe():
    assert [int(level) for level in InterfaceDamageLevel] == [0, 1, 2]
