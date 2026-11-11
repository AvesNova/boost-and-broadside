"""The counterfactual shot codes (spec 9): where a bullet is at each age of its flight
(the trajectory code), how its flight ends (the outcome code), and the age features the
trajectory head is queried with."""

import dataclasses
import math

import torch

from . import codes

HIT_NONE, HIT_ENEMY, HIT_ALLY = 0, 1, 2
SIDE_ENEMY, SIDE_ALLY = 1, 2


@dataclasses.dataclass(frozen=True)
class ShotCodeParameters:
    """displacement_period (m, both axes), top_speed (m/s), index_span (the log index range
    is +-index_span), collision_radius (m, the smallest hit radius); near_bins near-miss
    bins spanning near_span_radii collision radii in log distance, encoded with a sigma of
    near_sigma_bins bins."""

    displacement_period: float
    top_speed: float
    collision_radius: float
    displacement_levels: int = 8
    direction_levels: int = 5
    speed_bins: int = 41
    index_span: float = math.log(2.0)
    index_bins: int = 21
    near_bins: int = 8
    near_span_radii: float = 6.0
    near_sigma_bins: float = 0.75

    @property
    def trajectory_width(self) -> int:
        """The trajectory code's width: the displacement, direction, speed and log-index codes."""
        return (
            9 * self.displacement_levels
            + 4 * self.direction_levels
            + self.speed_bins
            + self.index_bins
        )

    @property
    def trajectory_runs(self) -> tuple[tuple[int, int], ...]:
        """The trajectory code's parts as (groups, size), in code order: displacement,
        direction, speed, refractive index (`bnb.models.yemong.codes.Runs`)."""
        return (
            (self.displacement_levels, 9),
            (self.direction_levels, 4),
            (1, self.speed_bins),
            (1, self.index_bins),
        )

    @property
    def outcome_width(self) -> int:
        """The outcome code's width: enemy hit, ally hit, near_bins near-miss bins per side, and
        clear."""
        return 2 + 2 * self.near_bins + 1


def shot_code_parameters(
    fastest_muzzle: float, velocity_reach: float, lifetime: float, smallest_hit_radius: float
) -> ShotCodeParameters:
    """The v2 constants: the top speed is (the fastest muzzle speed + the ship velocity
    reach) x 2 (the lowest index, 1/2, doubles world speed); the displacement period is 2.5
    top speeds over a bullet's lifetime (s)."""
    top = 2.0 * (fastest_muzzle + velocity_reach)
    return ShotCodeParameters(
        displacement_period=2.5 * top * lifetime,
        top_speed=top,
        collision_radius=smallest_hit_radius,
    )


def trajectory_coordinates(parameters: ShotCodeParameters, snapshots: torch.Tensor) -> torch.Tensor:
    """(..., 5) snapshots (displacement x, y; velocity x, y; refractive index) to the
    coordinates of their sharp trajectory code, (..., 46): the displacement's colours per
    level (x's, then y's), their fractions, the direction's colours and fractions per level,
    then the speed's lower bin and fraction and the log index's (`trajectory_code_at` makes
    the code of them; the code is two-hot within every group, so 46 numbers hold its 154)."""
    displacement, velocity, index = snapshots[..., :2], snapshots[..., 2:4], snapshots[..., 4]
    period = parameters.displacement_period
    position = codes.circular_coordinates(
        displacement, period, 3, parameters.displacement_levels
    )  # (..., 2, levels) each
    direction = codes.circular_coordinates(
        torch.atan2(velocity[..., 1], velocity[..., 0]),
        2.0 * math.pi,
        4,
        parameters.direction_levels,
    )
    speed_lower, speed_fraction = codes.scalar_sharp_coordinates(
        torch.hypot(velocity[..., 0], velocity[..., 1]),
        0.0,
        parameters.top_speed,
        parameters.speed_bins,
    )
    index_lower, index_fraction = codes.scalar_sharp_coordinates(
        torch.log(torch.clamp_min(index, 1e-6)),
        -parameters.index_span,
        parameters.index_span,
        parameters.index_bins,
    )
    return torch.cat(
        [
            *(part.flatten(-2) for part in position),
            *direction,
            torch.stack([speed_lower, speed_fraction, index_lower, index_fraction], -1),
        ],
        -1,
    )


def trajectory_code_at(parameters: ShotCodeParameters, coordinates: torch.Tensor) -> torch.Tensor:
    """(..., 46) coordinates (`trajectory_coordinates`) to the sharp trajectory code (...,
    trajectory_width): the displacement's 3 x 3 outer products, the direction, speed and log
    index two-hots."""
    levels, directions = parameters.displacement_levels, parameters.direction_levels
    colour, fraction, direction_colour, direction_fraction, scalars = coordinates.split(
        [2 * levels, 2 * levels, directions, directions, 4], -1
    )
    x, y = codes.circular_sharp(
        colour.unflatten(-1, (2, levels)), fraction.unflatten(-1, (2, levels)), 3
    ).unbind(-3)
    speed_lower, speed_fraction, index_lower, index_fraction = scalars.unbind(-1)
    return torch.cat(
        [
            (x[..., :, None] * y[..., None, :]).flatten(-3),
            codes.circular_sharp(direction_colour, direction_fraction, 4).flatten(-2),
            codes.two_hot_at(speed_lower, speed_fraction, parameters.speed_bins),
            codes.two_hot_at(index_lower, index_fraction, parameters.index_bins),
        ],
        -1,
    )


def trajectory_code(parameters: ShotCodeParameters, snapshots: torch.Tensor) -> torch.Tensor:
    """(..., 5) snapshots (displacement x, y; velocity x, y; refractive index) to
    (..., trajectory_width): the displacement, direction, speed and log-index codes."""
    return trajectory_code_at(parameters, trajectory_coordinates(parameters, snapshots))


def outcome_code(
    parameters: ShotCodeParameters, hit: torch.Tensor, distance: torch.Tensor, side: torch.Tensor
) -> torch.Tensor:
    """(...) hit (HIT_*), closest-pass distance (m) and its side (SIDE_*) to
    (..., outcome_width): [enemy hit, ally hit, enemy near 1..n, ally near 1..n, clear]."""
    bins = parameters.near_bins + 1
    low = math.log(parameters.collision_radius)
    step = math.log(parameters.near_span_radii) / (parameters.near_bins - 1)
    high = low + parameters.near_bins * step
    log_distance = torch.log(torch.clamp_min(distance, parameters.collision_radius))
    sigma = torch.full_like(distance, parameters.near_sigma_bins * step)
    near = codes.scalar_encode(log_distance, sigma, low, high, bins)
    clear = (torch.arange(bins, device=distance.device) == bins - 1).to(distance.dtype)
    near = torch.where((distance >= math.exp(high))[..., None], clear, near)
    ally = (side == SIDE_ALLY)[..., None]
    passes = torch.cat(
        [
            torch.zeros_like(near[..., :2]),
            torch.where(ally, 0.0, near[..., :-1]),
            torch.where(ally, near[..., :-1], 0.0),
            near[..., -1:],
        ],
        -1,
    )
    width = parameters.outcome_width
    hits = (
        torch.where(hit == HIT_ENEMY, 0, 1)[..., None] == torch.arange(width, device=hit.device)
    ).to(distance.dtype)
    return torch.where((hit != HIT_NONE)[..., None], hits, passes)


def time_features(ages: torch.Tensor, horizon: float, harmonics: int) -> torch.Tensor:
    """[a/H, sin(2 pi k a/H) for k = 1..harmonics, cos(...)], (..., 1 + 2 harmonics). Used
    for bullet ages and the respawn wait alike."""
    phase = ages / horizon
    angles = (
        2.0
        * math.pi
        * phase[..., None]
        * torch.arange(1, harmonics + 1, dtype=ages.dtype, device=ages.device)
    )
    return torch.cat([phase[..., None], torch.sin(angles), torch.cos(angles)], -1)
