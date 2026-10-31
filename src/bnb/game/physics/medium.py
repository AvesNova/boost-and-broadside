"""Refractive fields (game spec 4): ln n and its gradient at a batch of points.

Written as reductions over the field axis: each field's profile is elementwise over
(points, fields), and the union composition needs only sums, one product and one
leave-one-out sum, which torch.compile fuses into a single reduction kernel for any M,
without materializing the (points, fields) intermediate. (Unrolling a loop over the fields
instead makes the generated code grow with M until the compiler splits it into many
kernels that store their intermediates.)
"""

import torch

from ..rules import LN_SQRT2
from ..state import Fields


def medium(fields: Fields, x: torch.Tensor, y: torch.Tensor):
    """(ln n, d/dx ln n, d/dy ln n) at points (B, P) for fields (B, M)."""
    def per_field(values):  # (B, M) as (B, 1, M), against points (B, P, 1)
        return values[:, None, :]

    target = per_field(fields.level.to(torch.float32) * LN_SQRT2)  # target log index L_i
    band = per_field(fields.band_width)
    offset_x, offset_y = x[..., None] - per_field(fields.x), y[..., None] - per_field(fields.y)
    distance_squared = offset_x * offset_x + offset_y * offset_y
    # The distance, with a gradient that stays finite at a field's exact centre.
    away = distance_squared > 0.0
    distance = torch.where(away, torch.sqrt(torch.where(away, distance_squared, 1.0)), 0.0)
    depth = 0.5 - (distance - per_field(fields.radius)) / band  # spec: zeta
    inward = depth.clamp(0.0, 1.0)  # spec: z
    outward = 1.0 - inward
    profile = inward * inward * inward * (inward * (6.0 * inward - 15.0) + 10.0)  # alpha_i
    # 1 - alpha in factored form (the profile is symmetric: 1 - alpha(z) = alpha(1 - z)), so
    # it is exactly positive wherever the gradient is non-zero, even where alpha rounds to 1.
    complement = outward * outward * outward * (outward * (6.0 * outward - 15.0) + 10.0)
    in_band = (depth > 0.0) & (depth < 1.0) & (distance > 1e-6)
    slope = torch.where(in_band, -30.0 * (inward * outward) ** 2 / (band * distance.clamp_min(1e-6)), 0.0)
    profile_gradient_x, profile_gradient_y = slope * offset_x, slope * offset_y

    weight = profile.sum(-1)  # W
    weighted_target = (profile * target).sum(-1)
    clear = complement.prod(-1)  # prod (1 - alpha_i): the coverage is A = 1 - clear
    # grad A = sum_i grad alpha_i prod_{j != i} (1 - alpha_j) = clear sum_i grad alpha_i /
    # (1 - alpha_i): the spec's guarded division, only where grad alpha_i != 0.
    leave_one_out = torch.where(in_band, slope / torch.where(in_band, complement, 1.0), 0.0)
    coverage_gradient_x = clear * (leave_one_out * offset_x).sum(-1)
    coverage_gradient_y = clear * (leave_one_out * offset_y).sum(-1)
    weight_gradient_x, weight_gradient_y = profile_gradient_x.sum(-1), profile_gradient_y.sum(-1)
    target_gradient_x = (profile_gradient_x * target).sum(-1)
    target_gradient_y = (profile_gradient_y * target).sum(-1)

    coverage = 1.0 - clear
    covered = weight > 1e-12
    safe_weight = torch.where(covered, weight, 1.0)
    mean = torch.where(covered, weighted_target / safe_weight, 0.0)
    mean_gradient_x = torch.where(covered, (target_gradient_x - mean * weight_gradient_x)
                                  / safe_weight, 0.0)
    mean_gradient_y = torch.where(covered, (target_gradient_y - mean * weight_gradient_y)
                                  / safe_weight, 0.0)
    return (coverage * mean, mean * coverage_gradient_x + coverage * mean_gradient_x,
            mean * coverage_gradient_y + coverage * mean_gradient_y)
