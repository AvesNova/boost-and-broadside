"""Static circular refractive-field geometry and GPU-vectorized evaluation.

The nominal radius lies at the middle of the complete interface band: a width
``w`` extends from ``radius - w/2`` through ``radius + w/2``. Field hierarchy is
resolved while maps are built. Runtime evaluation is a fixed-shape
``(B, N, M)`` reduction with telescoping absolute indices.
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
    delta_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose the absolute local index and gradient by telescoping deltas."""

    if alpha.shape[-1] == 0:
        return (
            torch.ones(alpha.shape[:-1], dtype=torch.float32, device=alpha.device),
            torch.zeros(alpha.shape[:-1], dtype=torch.complex64, device=alpha.device),
        )
    delta = delta_index.unsqueeze(1)
    index = 1.0 + (alpha * delta).sum(dim=2)
    grad_index = (grad_alpha * delta).sum(dim=2)
    return index.float(), grad_index.to(torch.complex64)


def evaluate_fields(
    points: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    transition_widths: torch.Tensor,
    delta_index: torch.Tensor,
    world_size: tuple[float, float],
) -> FieldEvaluation:
    """Evaluate all field profiles and the composed medium at ``points``."""

    alpha, grad_alpha = evaluate_field_profiles(
        points, centers, radii, transition_widths, world_size
    )
    index, grad_index = compose_refractive_index(alpha, grad_alpha, delta_index)
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
        state.field_delta_index,
        config.world_size,
    )
    state.ship_field_alpha = evaluation.alpha
    state.ship_local_index = evaluation.index
    state.ship_field_gradient = evaluation.grad_index
    state.ship_field_damage = torch.zeros_like(state.ship_field_damage)


def index_from_level(level: torch.Tensor, index_step: float) -> torch.Tensor:
    """Convert integer log-index levels to absolute refractive indices."""

    base = torch.as_tensor(index_step, dtype=torch.float32, device=level.device)
    return torch.pow(base, level.float())


def damage_from_level(level: torch.Tensor, base_damage: float) -> torch.Tensor:
    """Convert independent interface-damage levels to crossing damage."""

    return level.float() * base_damage


def field_parents(
    centers: torch.Tensor,
    radii: torch.Tensor,
    transition_widths: torch.Tensor,
    world_size: tuple[float, float],
) -> torch.Tensor:
    """Return each field's smallest/direct enclosing parent, or ``-1``.

    This construction-time operation may use an argmin; the resulting indices
    are cached and no categorical region selection occurs during simulation.
    """

    if centers.ndim == 1:
        centers = centers.unsqueeze(0)
        radii = radii.unsqueeze(0)
        transition_widths = transition_widths.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    batch_size, num_fields = centers.shape
    if num_fields == 0:
        result = torch.empty((batch_size, 0), dtype=torch.long, device=centers.device)
        return result.squeeze(0) if squeeze else result

    displacement = wrap_displacement(
        centers.unsqueeze(2) - centers.unsqueeze(1), world_size
    )  # (B, parent, child)
    distance = displacement.abs()
    parent_core = radii - 0.5 * transition_widths
    child_outer = radii + 0.5 * transition_widths
    contains = distance + child_outer.unsqueeze(1) <= parent_core.unsqueeze(2)
    diagonal = torch.eye(num_fields, dtype=torch.bool, device=centers.device).unsqueeze(0)
    contains &= ~diagonal

    parent_size = radii.unsqueeze(2).expand(-1, -1, num_fields)
    candidates = torch.where(contains, parent_size, torch.full_like(parent_size, float("inf")))
    _, parent = candidates.min(dim=1)
    has_parent = contains.any(dim=1)
    parent = torch.where(has_parent, parent, torch.full_like(parent, -1))
    return parent.squeeze(0) if squeeze else parent


def validate_field_layout(
    centers: torch.Tensor,
    radii: torch.Tensor,
    transition_widths: torch.Tensor,
    index_levels: torch.Tensor,
    damage_levels: torch.Tensor,
    world_size: tuple[float, float],
) -> torch.Tensor:
    """Validate laminar toroidal geometry/materials and return direct parents.

    Every pair must be strictly nested (including its complete transition band)
    or disjoint with non-overlapping bands. Partially overlapping transition
    bands are rejected with a clear ``ValueError``.
    """

    if centers.ndim == 1:
        tensors = [centers, radii, transition_widths, index_levels, damage_levels]
        centers, radii, transition_widths, index_levels, damage_levels = [
            tensor.unsqueeze(0) for tensor in tensors
        ]
        squeeze = True
    else:
        squeeze = False

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

    num_fields = centers.shape[1]
    if num_fields > 1:
        displacement = wrap_displacement(centers.unsqueeze(2) - centers.unsqueeze(1), world_size)
        distance = displacement.abs()
        core = radii - 0.5 * transition_widths
        contains_ij = distance + outer.unsqueeze(1) <= core.unsqueeze(2)
        contains_ji = contains_ij.transpose(1, 2)
        disjoint = distance >= outer.unsqueeze(2) + outer.unsqueeze(1)
        valid_pair = contains_ij | contains_ji | disjoint
        upper = torch.triu(
            torch.ones((num_fields, num_fields), dtype=torch.bool, device=centers.device),
            diagonal=1,
        ).unsqueeze(0)
        if (~valid_pair & upper).any().item():
            raise ValueError(
                "fields must be disjoint or strictly nested with non-overlapping transition bands"
            )

    parents = field_parents(centers, radii, transition_widths, world_size)
    return parents.squeeze(0) if squeeze else parents


def material_tensors(
    index_levels: torch.Tensor,
    damage_levels: torch.Tensor,
    parents: torch.Tensor,
    config: ShipConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return absolute index, crossing damage, and telescoping delta index."""

    absolute_index = index_from_level(index_levels, config.field_index_step)
    damage = damage_from_level(damage_levels, config.field_interface_damage)
    parent_clamped = parents.clamp(min=0)
    parent_index = absolute_index.gather(1, parent_clamped)
    parent_index = torch.where(parents >= 0, parent_index, torch.ones_like(parent_index))
    return absolute_index, damage, absolute_index - parent_index
