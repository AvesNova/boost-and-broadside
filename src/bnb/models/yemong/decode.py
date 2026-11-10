"""The decoder layer: Yemong's outputs as world-space predictions with their uncertainty.

One place turns each head's logits into what it predicts, in the units and frame of the
game: the shot path at each age (mean and sigma), the next state and the hidden enemies'
beliefs (positions back in world coordinates), the density field's expected counts, the
enemy action distribution, the values and the outcome. The figures read the model through
these functions; nothing else decodes a head for display.

Every function is elementwise over leading axes and reads no model state beyond the code
parameters it is given.
"""

import dataclasses
import math

import torch

from bnb.contracts.action import NUM_POWER, NUM_SHOOT, NUM_STICK
from bnb.contracts.tree import contract

from . import codes, critic
from .density import LOG_RATE_RANGE, hex_cells
from .inputs import believed
from .shots import ShotCodeParameters
from .state_code import MEANS, POSITION_PERIOD

# --------------------------------------------------------------------------- shots


@contract
class ShotPath:
    """A shot's decoded flight at each age (..., A) leading, in the launch heading's frame
    (forward, left), as the trajectory code holds it: displacement (..., A, 2) m and its
    sigma (m, per axis); direction (rad, of the velocity in that frame) and its sigma;
    speed (m/s) and its sigma; log_index (ln n at the bullet) and its sigma."""

    displacement: torch.Tensor
    displacement_sigma: torch.Tensor
    direction: torch.Tensor
    direction_sigma: torch.Tensor
    speed: torch.Tensor
    speed_sigma: torch.Tensor
    log_index: torch.Tensor
    log_index_sigma: torch.Tensor


def _centred(values: torch.Tensor, period: float) -> torch.Tensor:
    """values on a circle of `period` as the representative in [-period / 2, period / 2)."""
    return torch.remainder(values + 0.5 * period, period) - 0.5 * period


def shot_path(parameters: ShotCodeParameters, logits: torch.Tensor) -> ShotPath:
    """Trajectory code logits (..., A, trajectory_width) (`Yemong.shot_trajectory`) to the
    decoded path. A displacement comes back modulo the code's period; it is read as the
    representative nearest the launch point (a flight is much shorter than the period)."""
    runs = parameters.trajectory_runs
    probabilities = torch.exp(codes.grouped_log_softmax(logits.float(), runs))
    widths = [groups * size for groups, size in runs]
    position, direction, speed, log_index = probabilities.split(widths, -1)
    period = parameters.displacement_period
    displacement, displacement_sigma = codes.position_decode(
        position.unflatten(-1, (parameters.displacement_levels, 9)),
        (period, period),
        parameters.displacement_levels,
    )
    angle, angle_sigma = codes.circular_decode(
        direction.unflatten(-1, (parameters.direction_levels, 4)),
        2.0 * math.pi,
        4,
        parameters.direction_levels,
    )
    speed_mean, speed_sigma = codes.scalar_decode(
        speed, 0.0, parameters.top_speed, parameters.speed_bins
    )
    index_mean, index_sigma = codes.scalar_decode(
        log_index, -parameters.index_span, parameters.index_span, parameters.index_bins
    )
    return ShotPath(
        displacement=_centred(displacement, period),
        displacement_sigma=displacement_sigma,
        direction=_centred(angle, 2.0 * math.pi),
        direction_sigma=angle_sigma,
        speed=speed_mean,
        speed_sigma=speed_sigma,
        log_index=index_mean,
        log_index_sigma=index_sigma,
    )


def launch_frame_to_world(
    displacement: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    heading_cos: torch.Tensor,
    heading_sin: torch.Tensor,
) -> torch.Tensor:
    """(..., A, 2) displacements in a launch heading's frame (forward, left) to world
    positions (m), for launch points x, y and headings (...)."""
    forward, left = displacement[..., 0], displacement[..., 1]
    cos, sin = heading_cos[..., None], heading_sin[..., None]
    return torch.stack(
        [x[..., None] + forward * cos - left * sin, y[..., None] + forward * sin + left * cos], -1
    )


def world_to_launch_frame(
    offset: torch.Tensor, heading_cos: torch.Tensor, heading_sin: torch.Tensor
) -> torch.Tensor:
    """(..., 2) world offsets from a launch point to its heading's frame (forward, left);
    headings (...)."""
    cos, sin = heading_cos, heading_sin
    return torch.stack(
        [offset[..., 0] * cos + offset[..., 1] * sin, offset[..., 1] * cos - offset[..., 0] * sin],
        -1,
    )


def shot_outcome(parameters: ShotCodeParameters, logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Outcome code logits (..., outcome_width) to probabilities (...) by name: enemy_hit,
    ally_hit, near_miss (a pass within the near-miss bins, either side) and clear_miss."""
    probability = torch.softmax(logits.float(), -1)
    near = parameters.near_bins
    return dict(
        enemy_hit=probability[..., 0],
        ally_hit=probability[..., 1],
        near_miss=probability[..., 2 : 2 + 2 * near].sum(-1),
        clear_miss=probability[..., -1],
    )


# --------------------------------------------------------------------------- ship state


@contract
class ShipForecast:
    """A ship-state prediction (..., N) leading, in physical units: position (..., N, 2) m
    in world coordinates and its sigma (m, per axis); velocity (..., N, 2) m/s and its
    packed covariance (..., N, 3) (xx, xy, yy); heading (rad) and its sigma; means
    (..., N, len(MEANS)) and spreads, every quantity as the state code holds it (positions
    in the model's shifted frame)."""

    position: torch.Tensor
    position_sigma: torch.Tensor
    velocity: torch.Tensor
    velocity_covariance: torch.Tensor
    heading: torch.Tensor
    heading_sigma: torch.Tensor
    means: torch.Tensor
    spreads: torch.Tensor


def nearest_representative(position: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """The representative of position (mod POSITION_PERIOD) nearest to reference (..., 2)."""
    return position + POSITION_PERIOD * torch.round((reference - position) / POSITION_PERIOD)


def ship_forecast(
    means: torch.Tensor, spreads: torch.Tensor, offset: torch.Tensor, reference: torch.Tensor
) -> ShipForecast:
    """State-code means (..., N, len(MEANS)) and spreads (`ShipStateCode.decode`, or a
    belief's) to a forecast in world coordinates: positions less the view's offset (..., 2),
    as the representative nearest reference (..., N, 2) m (the code wraps)."""
    shifted = means[..., 0:2] - offset[..., None, :]
    return ShipForecast(
        position=nearest_representative(shifted, reference),
        position_sigma=spreads[..., 0],
        velocity=means[..., 2:4],
        velocity_covariance=spreads[..., 1:4],
        heading=means[..., MEANS.index("heading")],
        heading_sigma=spreads[..., 4],
        means=means,
        spreads=spreads,
    )


def state_at(ships: torch.Tensor, replay) -> tuple[torch.Tensor, torch.Tensor]:
    """(means, spreads) (..., N, ...): every row's state at t as the model read it, from the
    view's ship rows and its replay record (`inputs.believed`): what `next_state` reads."""
    _, _, means, spreads = believed(ships, replay)
    return means, spreads


def next_state(
    model,
    logits: torch.Tensor,
    means: torch.Tensor,
    spreads: torch.Tensor,
    ship_type: torch.Tensor,
    offset,
    reference,
    rates=None,
    correction_logits: torch.Tensor | None = None,
    hidden: torch.Tensor | None = None,
    age: torch.Tensor | None = None,
) -> ShipForecast:
    """Next-state head logits (..., N, width) of a Yemong to the forecast of the state at
    t + 1, in world coordinates (`ship_forecast`); means and spreads (..., N, ...): the rows'
    state at t as the model read it (`inputs.believed`; the delta head's forecast is its
    baseline plus the decoded change); ship_type (..., N) int64. rates: the extrapolation
    baseline's (`Yemong.segment_rates`, on the model's device); with the error head's
    logits and the hidden rows (..., N), the hidden rows take the corrected forecast, as the
    belief stores it (`Yemong.forecast`); age (..., N): the rows' ticks since seen, which
    the faded forecast reads (`YemongConfig.physics_hidden_fade`)."""
    if model.delta_code is None:
        means, spreads = model.state_code.decode_logits(logits.float(), ship_type)
    else:  # on the model's device (the play shell decodes on the host)
        device = model.delta_code.max_shield.device
        moved = lambda part: None if part is None else part.to(device)  # noqa: E731
        means, spreads = (
            part.to(logits.device)
            for part in model.forecast(
                *(part.float().to(device) for part in (logits, means, spreads)),
                ship_type.to(device),
                None if correction_logits is None else correction_logits.float().to(device),
                moved(hidden),
                rates,
                moved(age),
            )
        )
    return ship_forecast(means, spreads, offset, reference)


# --------------------------------------------------------------------------- density


@contract
class DensityField:
    """The density head's expected ship counts per hex cell, (..., C) leading: ally and
    enemy (..., C) expected counts; centres (..., C, 2) m, each cell's centre in world
    coordinates; cell_radius (...) m, the centre-to-corner distance of a cell."""

    ally: torch.Tensor
    enemy: torch.Tensor
    centres: torch.Tensor
    cell_radius: torch.Tensor


def density(log_rates: torch.Tensor, rings: int, arena_radius: torch.Tensor) -> DensityField:
    """Density log-rates (..., 2C) (ally cells, then enemy cells) to expected counts on the
    grid of `rings` rings scaled by arena_radius (...) m. The arena is centred at the
    origin."""
    rates = torch.exp(torch.clamp(log_rates.float(), *LOG_RATE_RANGE))
    cells = rates.shape[-1] // 2
    _, centres = hex_cells(rings, device=rates.device)
    radius = arena_radius.float()
    return DensityField(
        ally=rates[..., :cells],
        enemy=rates[..., cells:],
        centres=centres * radius[..., None, None],
        cell_radius=radius / (math.sqrt(3.0) * max(rings, 1)),
    )


# --------------------------------------------------------------------------- enemy action


@contract
class ActionDistribution:
    """The enemy-action head's distribution, (...) leading: joint (..., NUM_JOINT)
    probabilities, and its factors' marginals power (..., NUM_POWER), stick
    (..., NUM_STICK) and shoot (..., NUM_SHOOT)."""

    joint: torch.Tensor
    power: torch.Tensor
    stick: torch.Tensor
    shoot: torch.Tensor


def action_distribution(logits: torch.Tensor) -> ActionDistribution:
    """Joint action logits (..., NUM_JOINT) to the distribution and its factor marginals."""
    joint = torch.softmax(logits.float(), -1)
    factored = joint.unflatten(-1, (NUM_POWER, NUM_STICK, NUM_SHOOT))
    return ActionDistribution(
        joint=joint,
        power=factored.sum((-2, -1)),
        stick=factored.sum((-3, -1)),
        shoot=factored.sum((-3, -2)),
    )


# --------------------------------------------------------------------------- critic


@dataclasses.dataclass(frozen=True)
class Outcome:
    """The outcome head's probabilities (...) of a win, a tie, a loss and an unresolved
    match, and its value P(win) - P(loss)."""

    win: torch.Tensor
    tie: torch.Tensor
    loss: torch.Tensor
    unresolved: torch.Tensor
    value: torch.Tensor


def outcome(logits: torch.Tensor) -> Outcome:
    """Outcome logits (..., OUTCOMES) to the class probabilities and the value."""
    probability = torch.softmax(logits.float(), -1)
    return Outcome(
        win=probability[..., critic.WIN],
        tie=probability[..., critic.TIE],
        loss=probability[..., critic.LOSS],
        unresolved=probability[..., critic.UNRESOLVED],
        value=critic.outcome_values(probability),
    )


def level_values(value_logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Per-ship value logits (..., levels - 1, bins) to each level's expected value (in the
    critic's units, before the return scale)."""
    return critic.expectation(value_logits.float(), bins)
