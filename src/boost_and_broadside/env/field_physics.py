"""Circular refractive-field geometry and GPU-vectorized evaluation.

The nominal radius lies at the middle of the complete interface band: a width
``w`` extends from ``radius - w/2`` through ``radius + w/2``. Fields may overlap
arbitrarily. Runtime evaluation is a fixed-shape ``(B, N, M)`` reduction that
blends their absolute targets in signed log-index space.
"""

from dataclasses import dataclass

import torch

from boost_and_broadside.config import (
    InterfaceDamageLevel,
    RefractiveIndexLevel,
    ShipConfig,
)
from boost_and_broadside.constants import EPS


@dataclass(frozen=True)
class FieldEvaluation:
    """Field values evaluated at a fixed batch of points."""

    alpha: torch.Tensor  # (B, N, M)
    grad_alpha: torch.Tensor  # (B, N, M) complex64
    index: torch.Tensor  # (B, N)
    grad_index: torch.Tensor  # (B, N) complex64


def wrap_displacement(
    displacement: torch.Tensor,
    world_size: tuple[float, float],
) -> torch.Tensor:
    """Return the minimum-image toroidal complex displacement."""

    world_w, world_h = world_size
    return torch.complex(
        (displacement.real + world_w / 2.0) % world_w - world_w / 2.0,
        (displacement.imag + world_h / 2.0) % world_h - world_h / 2.0,
    )


def evaluate_field_profiles(
    points: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    transition_widths: torch.Tensor,
    world_size: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate quintic alpha and its analytic world-space gradient.

    Args:
        points: ``(B, N)`` complex world positions.
        centers/radii/transition_widths: ``(B, M)`` field tensors.

    Returns:
        ``alpha`` and ``grad(alpha)`` shaped ``(B, N, M)``. Gradient is
        represented as a complex vector. It is exactly zero in both flat
        regions and remains finite at a field center.
    """

    if centers.shape[1] == 0:
        shape = (*points.shape, 0)
        return (
            torch.zeros(shape, dtype=torch.float32, device=points.device),
            torch.zeros(shape, dtype=torch.complex64, device=points.device),
        )

    displacement = wrap_displacement(points.unsqueeze(2) - centers.unsqueeze(1), world_size)
    distance = displacement.abs()
    signed_distance = distance - radii.unsqueeze(1)
    width = transition_widths.unsqueeze(1)
    z_unclamped = 0.5 - signed_distance / width
    z = z_unclamped.clamp(0.0, 1.0)

    # Quintic smoothstep: 6z^5 - 15z^4 + 10z^3.
    alpha = z**3 * (z * (z * 6.0 - 15.0) + 10.0)
    dalpha_dz = 30.0 * z**2 * (z - 1.0) ** 2
    in_transition = (z_unclamped > 0.0) & (z_unclamped < 1.0)

    # dz/dd = -1/w and grad(d) points radially outward. At the center the
    # profile derivative is flat, so selecting zero avoids a 0/0 direction.
    inv_distance = torch.where(
        distance > EPS,
        distance.reciprocal(),
        torch.zeros_like(distance),
    )
    factor = torch.where(
        in_transition,
        -dalpha_dz / width * inv_distance,
        torch.zeros_like(distance),
    )
    grad_alpha = torch.complex(displacement.real * factor, displacement.imag * factor)
    return alpha.float(), grad_alpha.to(torch.complex64)


def compose_refractive_index(
    alpha: torch.Tensor,
    grad_alpha: torch.Tensor,
    target_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose overlapping field targets and their analytic gradient.

    With ``L_i = log(n_i)`` and union coverage
    ``A = 1 - product_i(1 - alpha_i)``, the local log-index is
    ``A * sum(alpha_i L_i) / sum(alpha_i)``. The implementation accumulates the
    union gradient recurrently. That avoids
    division by ``1-alpha`` at fully covered points and remains stable when
    fields overlap exactly.
    """

    if alpha.shape[-1] == 0:
        return (
            torch.ones(alpha.shape[:-1], dtype=torch.float32, device=alpha.device),
            torch.zeros(alpha.shape[:-1], dtype=torch.complex64, device=alpha.device),
        )
    log_target = torch.log(target_index).unsqueeze(1)
    weight = alpha.sum(dim=2)
    weighted_log = (alpha * log_target).sum(dim=2)
    grad_weight = grad_alpha.sum(dim=2)
    grad_weighted_log = (grad_alpha * log_target).sum(dim=2)

    coverage = torch.zeros_like(weight)
    grad_coverage = torch.zeros_like(grad_weight)
    for field_idx in range(alpha.shape[2]):
        field_alpha = alpha[:, :, field_idx]
        field_gradient = grad_alpha[:, :, field_idx]
        remaining = 1.0 - coverage
        grad_coverage = grad_coverage * (1.0 - field_alpha) + remaining * field_gradient
        coverage = coverage + remaining * field_alpha

    contributes = weight > EPS
    safe_weight = weight.clamp(min=EPS)
    mean_log = torch.where(contributes, weighted_log / safe_weight, 0.0)
    grad_mean_log = torch.where(
        contributes,
        (grad_weighted_log - mean_log * grad_weight) / safe_weight,
        0.0,
    )
    local_log_index = coverage * mean_log
    grad_log_index = grad_coverage * mean_log + coverage * grad_mean_log
    index = torch.exp(local_log_index)
    grad_index = index * grad_log_index
    return index.float(), grad_index.to(torch.complex64)


def evaluate_fields(
    points: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    transition_widths: torch.Tensor,
    target_index: torch.Tensor,
    world_size: tuple[float, float],
) -> FieldEvaluation:
    """Evaluate all field profiles and the composed medium at ``points``."""

    alpha, grad_alpha = evaluate_field_profiles(
        points, centers, radii, transition_widths, world_size
    )
    index, grad_index = compose_refractive_index(alpha, grad_alpha, target_index)
    return FieldEvaluation(alpha, grad_alpha, index, grad_index)


def refresh_ship_field_cache(state, config: ShipConfig) -> None:
    """Refresh a state's cached ship alpha/index/gradient without damage.

    Environment reset uses the same operation inline with a reset mask. This
    helper is useful for deterministic map setup in tests and tooling.
    """

    evaluation = evaluate_fields(
        state.ship_pos,
        state.field_pos,
        state.field_radius,
        state.field_transition_width,
        state.field_index,
        config.world_size,
    )
    state.ship_field_alpha = evaluation.alpha
    state.ship_local_index = evaluation.index
    state.ship_field_gradient = evaluation.grad_index
    state.ship_field_damage = torch.zeros_like(state.ship_field_damage)
    state.ship_field_death = torch.zeros_like(state.ship_field_death)


def index_from_level(level: torch.Tensor, index_step: float) -> torch.Tensor:
    """Convert integer log-index levels to absolute refractive indices."""

    base = torch.as_tensor(index_step, dtype=torch.float32, device=level.device)
    return torch.pow(base, level.float())


def damage_from_level(level: torch.Tensor, base_damage: float) -> torch.Tensor:
    """Convert independent interface-damage levels to crossing damage."""

    return level.float() * base_damage


def validate_field_layout(
    centers: torch.Tensor,
    radii: torch.Tensor,
    transition_widths: torch.Tensor,
    index_levels: torch.Tensor,
    damage_levels: torch.Tensor,
    world_size: tuple[float, float],
) -> None:
    """Validate per-field toroidal geometry and material values.

    Pairwise geometry is intentionally unrestricted: nominal circles and their
    transition bands may intersect, coincide, or nest in any order.
    """

    if centers.ndim == 1:
        tensors = [centers, radii, transition_widths, index_levels, damage_levels]
        centers, radii, transition_widths, index_levels, damage_levels = [
            tensor.unsqueeze(0) for tensor in tensors
        ]

    shapes = {
        centers.shape,
        radii.shape,
        transition_widths.shape,
        index_levels.shape,
        damage_levels.shape,
    }
    if len(shapes) != 1:
        raise ValueError("all field layout tensors must share shape (B, M) or (M,)")
    if (
        not torch.isfinite(centers).all().item()
        or not torch.isfinite(radii).all().item()
        or not torch.isfinite(transition_widths).all().item()
    ):
        raise ValueError("field centers, radii, and transition widths must be finite")
    if (radii <= 0.0).any().item() or (transition_widths <= 0.0).any().item():
        raise ValueError("field radii and transition widths must be positive")
    if (radii <= 0.5 * transition_widths).any().item():
        raise ValueError("each field radius must exceed half its width to retain a flat core")

    outer = radii + 0.5 * transition_widths
    toroidal_limit = 0.5 * min(world_size)
    if (outer >= toroidal_limit).any().item():
        raise ValueError(
            "field outer extent must be strictly below half the shorter world dimension"
        )

    allowed_index = torch.zeros_like(index_levels, dtype=torch.bool)
    for level in (
        RefractiveIndexLevel.VERY_LOW,
        RefractiveIndexLevel.LOW,
        RefractiveIndexLevel.HIGH,
        RefractiveIndexLevel.VERY_HIGH,
    ):
        allowed_index |= index_levels == int(level)
    if not allowed_index.all().item():
        raise ValueError("field index levels must be one of {-2, -1, +1, +2}; ambient is invalid")
    invalid_damage = (damage_levels < int(InterfaceDamageLevel.NONE)) | (
        damage_levels > int(InterfaceDamageLevel.SEVERE)
    )
    if invalid_damage.any().item():
        raise ValueError("field damage levels must be NONE, STANDARD, or SEVERE")


def material_tensors(
    index_levels: torch.Tensor,
    damage_levels: torch.Tensor,
    config: ShipConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each field's absolute target index and independent damage."""

    absolute_index = index_from_level(index_levels, config.field_index_step)
    damage = damage_from_level(damage_levels, config.field_interface_damage)
    return absolute_index, damage
