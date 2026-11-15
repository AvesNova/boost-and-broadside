"""The counterfactual shot codes (spec 9): where a bullet is at each age of its flight
(the trajectory code), how its flight ends (the outcome code), the age features the
trajectory head is queried with, and the shot damage head's code (`YemongConfig.shot_head`
"damage", arm A22: what a shot fired now would deal to each other ship)."""

import dataclasses
import math

import torch

from . import codes

HIT_NONE, HIT_ENEMY, HIT_ALLY = 0, 1, 2
SIDE_ENEMY, SIDE_ALLY = 1, 2


@dataclasses.dataclass(frozen=True)
class ShotCodeParameters:
    """displacement_period (m, both axes), top_speed (m/s), index_span (the log index range
    is +-index_span), collision_radius (m, the smallest hit radius); speed_bins and
    index_bins the real bins of the speed and log-index HL-Gauss codes, each with
    codes.PADDING_BINS more either side; near_bins near-miss
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
    def speed_width(self) -> int:
        """The speed code's width: its real bins and the padding."""
        return self.speed_bins + 2 * codes.PADDING_BINS

    @property
    def index_width(self) -> int:
        """The log-index code's width: its real bins and the padding."""
        return self.index_bins + 2 * codes.PADDING_BINS

    @property
    def trajectory_width(self) -> int:
        """The trajectory code's width: the displacement, direction, speed and log-index codes."""
        return (
            9 * self.displacement_levels
            + 4 * self.direction_levels
            + self.speed_width
            + self.index_width
        )

    @property
    def trajectory_runs(self) -> tuple[tuple[int, int], ...]:
        """The trajectory code's parts as (groups, size), in code order: displacement,
        direction, speed, refractive index (`bnb.models.yemong.codes.Runs`)."""
        return (
            (self.displacement_levels, 9),
            (self.direction_levels, 4),
            (1, self.speed_width),
            (1, self.index_width),
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
    coordinates of their trajectory code, (..., 44): the displacement's colours per level
    (x's, then y's), their fractions, the direction's colours and fractions per level, then
    the speed's and the log index's coordinates in their padded codes
    (`codes.hl_gauss_scalar_coordinate`). `trajectory_code_at` makes the code of them: 44
    numbers hold its 162."""
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
    speed = codes.hl_gauss_scalar_coordinate(
        torch.hypot(velocity[..., 0], velocity[..., 1]),
        0.0,
        parameters.top_speed,
        parameters.speed_bins,
    )
    log_index = codes.hl_gauss_scalar_coordinate(
        torch.log(torch.clamp_min(index, 1e-6)),
        -parameters.index_span,
        parameters.index_span,
        parameters.index_bins,
    )
    return torch.cat(
        [
            *(part.flatten(-2) for part in position),
            *direction,
            torch.stack([speed, log_index], -1),
        ],
        -1,
    )


def trajectory_code_at(parameters: ShotCodeParameters, coordinates: torch.Tensor) -> torch.Tensor:
    """(..., 44) coordinates (`trajectory_coordinates`) to the trajectory code (...,
    trajectory_width): the displacement's 3 x 3 outer products and the direction's colours
    (sharp colour codes, kept: 3-4 colours per level, see `codes`), then the speed's and
    the log index's HL-Gauss codes of an exact value (`codes.hl_gauss_scalar_at`)."""
    levels, directions = parameters.displacement_levels, parameters.direction_levels
    colour, fraction, direction_colour, direction_fraction, scalars = coordinates.split(
        [2 * levels, 2 * levels, directions, directions, 2], -1
    )
    x, y = codes.circular_sharp(
        colour.unflatten(-1, (2, levels)), fraction.unflatten(-1, (2, levels)), 3
    ).unbind(-3)
    speed, log_index = scalars.unbind(-1)
    exact = torch.zeros_like(speed)
    return torch.cat(
        [
            (x[..., :, None] * y[..., None, :]).flatten(-3),
            codes.circular_sharp(direction_colour, direction_fraction, 4).flatten(-2),
            codes.hl_gauss_scalar_at(speed, exact, parameters.speed_width),
            codes.hl_gauss_scalar_at(log_index, exact, parameters.index_width),
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
    # kept: already a Gaussian of near_sigma_bins (0.75) bins, its last bin the clear miss
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


# The shot damage head's code (arm A22): the damage a shot fired now would deal to one ship,
# over the shooter's centre hit at its reference speed (`ShipTypes.hit_damage`), as an
# HL-Gauss code of DAMAGE_BINS real bins from 0 to DAMAGE_HIGH (a tenth of a full hit
# apart, an exact 0 at the first centre: a miss, and every ship but the one hit) and
# codes.PADDING_BINS more either side. A hit can pass a full one where the bullet flies
# faster than its reference speed (`combat.kinetic`); a share past DAMAGE_HIGH is clamped
# to the padding.
DAMAGE_BINS = 21
DAMAGE_HIGH = 2.0
DAMAGE_WIDTH = DAMAGE_BINS + 2 * codes.PADDING_BINS


def damage_targets(target: torch.Tensor, damage: torch.Tensor, shooters: int) -> torch.Tensor:
    """(..., S, N): what the shot each of the first S rows fires would deal to each row,
    from its hit's target (..., N) (a row, -1 for none) and damage (..., N), the share of a
    full hit; 0 for every row it does not hit."""
    rows = target.shape[-1]
    hit = target[..., :shooters, None] == torch.arange(rows, device=target.device)
    return torch.where(hit, damage[..., :shooters, None], 0.0)


def damage_code(share: torch.Tensor) -> torch.Tensor:
    """(...) damage shares to their codes (..., DAMAGE_WIDTH): HL-Gauss of an exact value."""
    return codes.hl_gauss_scalar_encode(
        share, torch.zeros_like(share), 0.0, DAMAGE_HIGH, DAMAGE_BINS
    )
