"""Physical belief state: its moment layout and the helpers that read it.

The belief plane stores **physical** ship state -- pixels, pixels/second,
radians -- not the policy's encoding of it. Belief and authoritative truth have
the same units, so composing a legal observation is a selection between two
tensors of the same meaning.

A belief is moments, not distributions (``frontline-redesign-plan.md`` §8.3):
the eleven physical means below and fourteen spread terms. The encoder rebuilds
each channel's categorical code from them inside the forward pass
(:mod:`boost_and_broadside.train.rl.ship_codes`), and the next-state head's
categorical prediction is decoded back to them, so the rollout buffer stores
moments only.

The spread terms are physical and zero means certain: a ship in sight carries
all zeros, which is the exact code of its true state. Position has one sigma
(px); velocity a full covariance of raw world velocity, packed ``(xx, xy, yy)``
in px²/s²; attitude, angular velocity and the four bounded scalars one sigma
each in their own units. Cooldown is the one channel kept as a distribution:
its four probabilities minus the exact two-hot of the cooldown mean, so that it
too is zero when certain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.train.rl.physical_deltas import (
    PHYSICAL_DELTA_NAMES,
    PHYSICAL_DELTA_SCALES,
)

#: The eleven physical means, in the Phase-1 calibration's order.
PHYSICAL_MEAN_NAMES = PHYSICAL_DELTA_NAMES
PHYSICAL_MEAN_DIM = len(PHYSICAL_MEAN_NAMES)

#: Cooldown bins: ready, then one tick per bin up to the firing cooldown.
COOLDOWN_BINS = 4

#: The fourteen spread terms. All zero is certainty.
UNCERTAINTY_NAMES = (
    "position_sigma",
    "velocity_covariance_xx",
    "velocity_covariance_xy",
    "velocity_covariance_yy",
    "attitude_sigma",
    "angular_velocity_sigma",
    "shield_delay_sigma",
    "health_sigma",
    "power_sigma",
    "local_log_index_sigma",
) + tuple(f"cooldown_residual_{k}" for k in range(COOLDOWN_BINS))
PHYSICAL_UNCERTAINTY_DIM = len(UNCERTAINTY_NAMES)

#: One stored belief, and one decoded next-state prediction: means, then spreads.
BELIEF_MOMENT_DIM = PHYSICAL_MEAN_DIM + PHYSICAL_UNCERTAINTY_DIM

# Channel indices into the mean vector, by name rather than by counting.
POSITION_X, POSITION_Y = 0, 1
VELOCITY_X, VELOCITY_Y = 2, 3
ATTITUDE = 4
ANGULAR_VELOCITY = 5
SHIELD_DELAY = 6
HEALTH = 7
POWER = 8
COOLDOWN = 9
LOCAL_LOG_INDEX = 10

# Columns of the spread block.
POSITION_SIGMA = 0
VELOCITY_COVARIANCE = slice(1, 4)
ATTITUDE_SIGMA = 4
ANGULAR_VELOCITY_SIGMA = 5
#: ``(mean channel, sigma column)`` for the four bounded scalars, in code order.
SCALAR_SIGMAS = (
    (SHIELD_DELAY, 6),
    (HEALTH, 7),
    (POWER, 8),
    (LOCAL_LOG_INDEX, 9),
)
COOLDOWN_RESIDUAL = slice(10, 10 + COOLDOWN_BINS)

#: Health above which a ship is considered alive, in health units.
ALIVE_HEALTH_EPS = 1.0


def predicted_means(prediction: torch.Tensor) -> torch.Tensor:
    """The ``(..., 11)`` means of a decoded ``(..., 25)`` prediction."""

    return prediction[..., :PHYSICAL_MEAN_DIM]


def predicted_uncertainty(prediction: torch.Tensor) -> torch.Tensor:
    """The ``(..., 14)`` spread terms of a decoded ``(..., 25)`` prediction."""

    return prediction[..., PHYSICAL_MEAN_DIM:]


def wrap_symmetric(x: torch.Tensor, period: float) -> torch.Tensor:
    """Wrap into ``[-period/2, period/2)``."""

    return (x + period / 2.0) % period - period / 2.0


def physical_means_from_state(state: TensorState) -> torch.Tensor:
    """Authoritative ``(B, N, 11)`` physical ship state, in calibration order.

    ``local_log_index`` is the *natural* logarithm of the refractive index, not
    the observation's profile-dependent normalization of it: the belief stores
    physical state and the encoder normalizes afterwards.
    """

    return torch.stack(
        (
            state.ship_pos.real,
            state.ship_pos.imag,
            state.ship_vel.real,
            state.ship_vel.imag,
            torch.angle(state.ship_attitude),
            state.ship_ang_vel,
            state.ship_shield_delay,
            state.ship_health,
            state.ship_power,
            state.ship_cooldown,
            torch.log(state.ship_local_index),
        ),
        dim=-1,
    )


def physical_means_from_observation(
    observation, index_log_scale: float, num_ships: int | None = None
) -> torch.Tensor:
    """The ``(..., N, 11)`` physical state one observation's ship slots carry.

    This is the legal view's own account of the world: truth for the ships the
    observer owns or can see, its belief for the rest. It is what the next-state
    label steps *from*, which is the whole point of composing the belief in
    physical units -- there is no decode, and no second store to keep in step
    with the observation the policy actually read.

    ``index_log_scale`` undoes the observation's normalization of the log index,
    returning the natural log the belief plane stores.
    """

    from boost_and_broadside.env.observation import ObsKey

    def channel(key: ObsKey) -> torch.Tensor:
        value = observation[key]
        return value if num_ships is None else value[..., :num_ships, :]

    attitude = channel(ObsKey.ATT)
    position = channel(ObsKey.POS)
    velocity = channel(ObsKey.VEL)
    return torch.stack(
        (
            position[..., 0],
            position[..., 1],
            velocity[..., 0],
            velocity[..., 1],
            torch.atan2(attitude[..., 1], attitude[..., 0]),
            channel(ObsKey.ANG_VEL)[..., 0],
            channel(ObsKey.SHIELD_DELAY)[..., 0],
            channel(ObsKey.HEALTH)[..., 0],
            channel(ObsKey.POWER)[..., 0],
            channel(ObsKey.COOLDOWN)[..., 0],
            channel(ObsKey.LOCAL_LOG_INDEX)[..., 0] * index_log_scale,
        ),
        dim=-1,
    )


def physical_mean_deltas(
    current: torch.Tensor,
    next_: torch.Tensor,
    world_size: tuple[float, float],
) -> torch.Tensor:
    """One-decision deltas between two ``(..., 11)`` physical mean vectors.

    Position takes the torus's minimum-image displacement and attitude the
    signed shortest rotation, matching
    :func:`boost_and_broadside.train.rl.physical_deltas.physical_ship_deltas`
    channel for channel. Every other difference is ordinary subtraction.
    """

    if current.shape[-1] != PHYSICAL_MEAN_DIM or next_.shape[-1] != PHYSICAL_MEAN_DIM:
        raise ValueError(
            f"physical means must have {PHYSICAL_MEAN_DIM} channels, got "
            f"{current.shape[-1]} and {next_.shape[-1]}"
        )
    delta = next_ - current
    width, height = world_size
    return torch.stack(
        (
            wrap_symmetric(delta[..., POSITION_X], float(width)),
            wrap_symmetric(delta[..., POSITION_Y], float(height)),
            delta[..., VELOCITY_X],
            delta[..., VELOCITY_Y],
            wrap_symmetric(delta[..., ATTITUDE], 2.0 * math.pi),
            delta[..., ANGULAR_VELOCITY],
            delta[..., SHIELD_DELAY],
            delta[..., HEALTH],
            delta[..., POWER],
            delta[..., COOLDOWN],
            delta[..., LOCAL_LOG_INDEX],
        ),
        dim=-1,
    )


@dataclass(frozen=True)
class PhysicalNextState:
    """Fixed per-channel delta scales, for reading one-decision errors.

    The next-state objective is cross-entropy on categorical codes and needs no
    scale. These remain the Phase-1 calibration constants that turn physical
    one-decision deltas into comparable O(1) units for the analysis modes and
    the label diagnostics.
    """

    world_size: tuple[float, float]
    #: Divisors turning physical deltas into O(1) units.
    scales: tuple[float, ...]

    @classmethod
    def from_ship_config(cls, ship_config: ShipConfig) -> PhysicalNextState:
        return cls(
            world_size=(float(ship_config.world_size[0]), float(ship_config.world_size[1])),
            scales=tuple(float(s) for s in PHYSICAL_DELTA_SCALES),
        )

    def scale_vector(self, device: torch.device | str) -> torch.Tensor:
        return _cached_vector(self.scales, device)

    def labels(self, current: torch.Tensor, next_: torch.Tensor) -> torch.Tensor:
        """Normalized ``(..., 11)`` deltas from ``current`` means to ``next_`` means."""

        deltas = physical_mean_deltas(current, next_, self.world_size)
        return deltas / self.scale_vector(deltas.device)


_VECTOR_CACHE: dict[tuple[tuple[float, ...], torch.device], torch.Tensor] = {}


def _cached_vector(values: tuple[float, ...], device: torch.device | str) -> torch.Tensor:
    """A constant vector on ``device``, built once.

    Cached because ``torch.tensor([...], device="cuda")`` is a synchronizing host
    copy. Safe to cache: nothing here is created inside a CUDA-graph capture.
    """

    key = (values, torch.device(device))
    cached = _VECTOR_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(values, dtype=torch.float32, device=key[1])
        _VECTOR_CACHE[key] = cached
    return cached
