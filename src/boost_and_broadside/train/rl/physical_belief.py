"""Physical belief state, its uncertainty layout, and the Gaussian next-state model.

The belief plane stores **physical** ship state -- pixels, pixels/second,
radians -- not the policy's Fourier/symlog encoding of it. Two things follow.

The belief and authoritative truth have the same units, so composing a legal
observation is a selection between two tensors of the same meaning rather than a
substitution inside an encoded vector. And the next-state head predicts eleven
physical deltas against the Phase-1 calibration in
:mod:`boost_and_broadside.train.rl.physical_deltas`, so its labels are
scale-free by measurement rather than by a fitted constant.

Uncertainty is thirteen numbers, kept in log/unconstrained form everywhere --
head output, belief store, and observation channel alike. Position and velocity
each carry a full 2D covariance (two log sigmas and one unconstrained
correlation latent); the seven remaining channels carry one log sigma each.
Nothing accumulates: the head sees the current uncertainty as an input and
states the *next* one directly, so a long-hidden ship's spread is whatever the
model says it is rather than a running sum nobody supervises.
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

#: The thirteen uncertainty outputs. Position and velocity get a full 2D
#: covariance; every other channel gets one spread. ``*_correlation`` is an
#: unconstrained latent mapped to rho in (-1, 1) by ``tanh``.
UNCERTAINTY_NAMES = (
    "position_log_sigma_x",
    "position_log_sigma_y",
    "position_correlation",
    "velocity_log_sigma_x",
    "velocity_log_sigma_y",
    "velocity_correlation",
    "attitude_log_sigma",
    "angular_velocity_log_sigma",
    "shield_delay_log_sigma",
    "health_log_sigma",
    "power_log_sigma",
    "cooldown_log_sigma",
    "local_log_index_log_sigma",
)
PHYSICAL_UNCERTAINTY_DIM = len(UNCERTAINTY_NAMES)

#: Head width: every mean first, then the uncertainty block.
NEXT_STATE_OUTPUT_DIM = PHYSICAL_MEAN_DIM + PHYSICAL_UNCERTAINTY_DIM

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

# Uncertainty columns, grouped so the clamp and the likelihood read one layout.
POSITION_SIGMA = (0, 1)
POSITION_RHO = 2
VELOCITY_SIGMA = (3, 4)
VELOCITY_RHO = 5
#: ``(mean channel, uncertainty column)`` for every channel with a lone spread.
SCALAR_UNCERTAINTY = (
    (ATTITUDE, 6),
    (ANGULAR_VELOCITY, 7),
    (SHIELD_DELAY, 8),
    (HEALTH, 9),
    (POWER, 10),
    (COOLDOWN, 11),
    (LOCAL_LOG_INDEX, 12),
)
CORRELATION_COLUMNS = (POSITION_RHO, VELOCITY_RHO)
#: The eleven log-sigma columns, i.e. every uncertainty column that is a spread.
LOG_SIGMA_COLUMNS = tuple(
    column for column in range(len(UNCERTAINTY_NAMES)) if column not in CORRELATION_COLUMNS
)

# Log-sigma clamp for the head. Labels are normalized to O(1) by the Phase-1
# scales, so ``exp(+/-6)`` spans spreads from 1/400th of a typical delta to 400
# times one -- wider than any residual the objective sees. The *floor* is the
# bound that matters: a Gaussian likelihood is unbounded below as sigma falls,
# so without it the head is paid to claim certainty it does not have. Clamping
# rather than squashing is deliberate; the zero gradient at the bound is what
# stops a collapse continuing.
LOG_SIGMA_MIN = -6.0
LOG_SIGMA_MAX = 6.0
# ``tanh`` of a latent this large is 0.9999 in float32 and exactly 1.0 not much
# further out, which would make ``1 - rho**2`` zero and the bivariate likelihood
# infinite. The bound keeps the correlation strictly inside the unit interval.
CORRELATION_LATENT_LIMIT = 5.0

#: The finite "I can see it" spread assimilated truth is given, standing in for
#: a mathematical ``log(0)``.
CERTAIN_LOG_SIGMA = LOG_SIGMA_MIN
#: The spread of a ship this observer has never seen. Its slot carries no
#: physical value at all, so the honest statement is maximal doubt -- not the
#: zero a masked channel used to leave behind, which reads as ``sigma = 1``.
UNKNOWN_LOG_SIGMA = LOG_SIGMA_MAX

#: Health above which a ship is considered alive, in health units.
ALIVE_HEALTH_EPS = 1.0

_LOG_TWO_PI = math.log(2.0 * math.pi)
_HALF_LOG_TWO_PI = 0.5 * _LOG_TWO_PI


def uncertainty_clamp_bounds() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-column ``(min, max)`` for the thirteen uncertainty outputs."""

    lower = [LOG_SIGMA_MIN] * PHYSICAL_UNCERTAINTY_DIM
    upper = [LOG_SIGMA_MAX] * PHYSICAL_UNCERTAINTY_DIM
    for column in CORRELATION_COLUMNS:
        lower[column] = -CORRELATION_LATENT_LIMIT
        upper[column] = CORRELATION_LATENT_LIMIT
    return tuple(lower), tuple(upper)


def certain_uncertainty() -> tuple[float, ...]:
    """The uncertainty vector assimilated truth carries: floor spreads, zero rho."""

    values = [CERTAIN_LOG_SIGMA] * PHYSICAL_UNCERTAINTY_DIM
    for column in CORRELATION_COLUMNS:
        values[column] = 0.0
    return tuple(values)


def unknown_uncertainty() -> tuple[float, ...]:
    """The uncertainty vector a never-observed slot carries: ceiling, zero rho."""

    values = [UNKNOWN_LOG_SIGMA] * PHYSICAL_UNCERTAINTY_DIM
    for column in CORRELATION_COLUMNS:
        values[column] = 0.0
    return tuple(values)


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
    """Fixed scales, physical bounds, and the Gaussian objective over them.

    One instance per ship configuration. Everything here is either a Phase-1
    calibration constant or derived from ``ShipConfig``; nothing adapts during
    training, by contract -- there is no online delta scaler.
    """

    world_size: tuple[float, float]
    #: Divisors turning physical deltas into O(1) network units.
    scales: tuple[float, ...]
    #: Inclusive physical bounds for the belief recursion, ``-inf``/``inf`` where
    #: the quantity has none. Position and attitude wrap instead and are ``inf``.
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    @classmethod
    def from_ship_config(cls, ship_config: ShipConfig) -> PhysicalNextState:
        inf = float("inf")
        # A generous numerical guard rather than a physical claim. The
        # thrust/drag equilibrium sits at sqrt(boost_thrust / drag), and a
        # collision or a refractive gradient can briefly exceed it, so this is
        # four times that speed on each axis.
        speed_guard = 4.0 * math.sqrt(
            ship_config.boost_thrust / max(ship_config.no_turn_drag_coeff, 1e-12)
        )
        # Exact: a turn command sets the angular velocity to angle/dt.
        turn_rate = ship_config.sharp_turn_angle / ship_config.dt
        # Exact: the configured index ladder spans step**-2 .. step**2.
        index_span = 2.0 * math.log(ship_config.field_index_step)
        lower = [0.0] * PHYSICAL_MEAN_DIM
        upper = [0.0] * PHYSICAL_MEAN_DIM
        for channel, (low, high) in {
            POSITION_X: (-inf, inf),
            POSITION_Y: (-inf, inf),
            VELOCITY_X: (-speed_guard, speed_guard),
            VELOCITY_Y: (-speed_guard, speed_guard),
            ATTITUDE: (-inf, inf),
            ANGULAR_VELOCITY: (-turn_rate, turn_rate),
            # No configured ceiling reaches this module; the recharge delay
            # lives on the frontline config. Non-negativity is the property
            # that matters for a countdown.
            SHIELD_DELAY: (0.0, inf),
            HEALTH: (0.0, ship_config.max_health),
            POWER: (0.0, ship_config.max_power),
            COOLDOWN: (0.0, ship_config.firing_cooldown),
            LOCAL_LOG_INDEX: (-index_span, index_span),
        }.items():
            lower[channel] = low
            upper[channel] = high
        return cls(
            world_size=(float(ship_config.world_size[0]), float(ship_config.world_size[1])),
            scales=tuple(float(s) for s in PHYSICAL_DELTA_SCALES),
            lower=tuple(lower),
            upper=tuple(upper),
        )

    # ------------------------------------------------------------------
    # Cached device vectors
    # ------------------------------------------------------------------

    def scale_vector(self, device: torch.device | str) -> torch.Tensor:
        return _cached_vector(self.scales, device)

    def lower_vector(self, device: torch.device | str) -> torch.Tensor:
        return _cached_vector(self.lower, device)

    def upper_vector(self, device: torch.device | str) -> torch.Tensor:
        return _cached_vector(self.upper, device)

    # ------------------------------------------------------------------
    # Labels and likelihood
    # ------------------------------------------------------------------

    def labels(self, current: torch.Tensor, next_: torch.Tensor) -> torch.Tensor:
        """Normalized ``(..., 11)`` deltas from ``current`` means to ``next_`` means.

        Ordinary division by a fixed positive scale, so a zero physical delta
        maps to a bit-exact zero normalized delta.
        """

        deltas = physical_mean_deltas(current, next_, self.world_size)
        return deltas / self.scale_vector(deltas.device)

    def apply_means(self, means: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """Advance ``(..., 11)`` means by the head's normalized mean block.

        Position wraps onto the torus and attitude onto the circle, which is
        exact rather than a clamp. Every other channel is clamped to its
        physical range, so the autoregressive recursion cannot leave a bounded
        set however wrong the head is.
        """

        delta = prediction[..., :PHYSICAL_MEAN_DIM] * self.scale_vector(prediction.device)
        raw = means + delta
        width, height = self.world_size
        bounded = raw.clamp(min=self.lower_vector(raw.device), max=self.upper_vector(raw.device))
        # ``clamp`` left the wrapping channels alone (their bounds are infinite);
        # write the wrapped values over them.
        return torch.stack(
            (
                raw[..., POSITION_X] % width,
                raw[..., POSITION_Y] % height,
                bounded[..., VELOCITY_X],
                bounded[..., VELOCITY_Y],
                wrap_symmetric(raw[..., ATTITUDE], 2.0 * math.pi),
                bounded[..., ANGULAR_VELOCITY],
                bounded[..., SHIELD_DELAY],
                bounded[..., HEALTH],
                bounded[..., POWER],
                bounded[..., COOLDOWN],
                bounded[..., LOCAL_LOG_INDEX],
            ),
            dim=-1,
        )

    def residual(self, prediction: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """``(..., 11)`` normalized residuals, with attitude wrapped on the circle.

        Both the predicted and the true attitude delta live in ``[-pi, pi]``, so
        their difference can reach ``2*pi`` -- a prediction of ``-pi`` against a
        label of ``+pi`` is the same rotation, not the largest possible error.
        Wrapping the residual is what makes the scalar Gaussian on this channel
        measure the rotation it is meant to.
        """

        residual = prediction[..., :PHYSICAL_MEAN_DIM] - labels
        attitude_scale = self.scales[ATTITUDE]
        wrapped = wrap_symmetric(residual[..., ATTITUDE] * attitude_scale, 2.0 * math.pi)
        return torch.cat(
            [
                residual[..., :ATTITUDE],
                (wrapped / attitude_scale).unsqueeze(-1),
                residual[..., ATTITUDE + 1 :],
            ],
            dim=-1,
        )

    def loss(self, prediction: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-channel negative log likelihood, ``(..., 11)``.

        Position and velocity use the full bivariate normal over their two axes;
        the other seven channels use a scalar normal. The bivariate term is
        split evenly between its two axes so the returned vector still sums to
        the exact joint likelihood while reading per channel -- the gradient is
        unaffected, since every consumer sums it.

        Both forms carry their normalizing constant. That cancels out of the
        gradient but makes the per-channel series nats, so a channel costing
        more of them is genuinely harder to predict than one costing fewer.
        """

        if prediction.shape[-1] != NEXT_STATE_OUTPUT_DIM:
            raise ValueError(
                f"prediction must have {NEXT_STATE_OUTPUT_DIM} channels, got {prediction.shape[-1]}"
            )
        residual = self.residual(prediction, labels)
        uncertainty = prediction[..., PHYSICAL_MEAN_DIM:]
        terms = [None] * PHYSICAL_MEAN_DIM

        # The two paired channels, each as (mean x, sigma columns, rho column).
        # The mean and uncertainty layouts differ -- position's means are (0, 1)
        # and its sigmas (0, 1), velocity's means (2, 3) and its sigmas (3, 4) --
        # so the pairing is stated rather than derived by arithmetic.
        for mean_x, axes, rho_column in (
            (POSITION_X, POSITION_SIGMA, POSITION_RHO),
            (VELOCITY_X, VELOCITY_SIGMA, VELOCITY_RHO),
        ):
            log_sigma_x = uncertainty[..., axes[0]]
            log_sigma_y = uncertainty[..., axes[1]]
            mean_y = mean_x + 1
            a = residual[..., mean_x] * torch.exp(-log_sigma_x)
            b = residual[..., mean_y] * torch.exp(-log_sigma_y)
            rho = torch.tanh(uncertainty[..., rho_column])
            one_minus = (1.0 - rho * rho).clamp_min(1e-6)
            cross = rho * a * b
            terms[mean_x] = (
                _HALF_LOG_TWO_PI
                + log_sigma_x
                + 0.25 * torch.log(one_minus)
                + (a * a - cross) / (2.0 * one_minus)
            )
            terms[mean_y] = (
                _HALF_LOG_TWO_PI
                + log_sigma_y
                + 0.25 * torch.log(one_minus)
                + (b * b - cross) / (2.0 * one_minus)
            )

        for mean_channel, column in SCALAR_UNCERTAINTY:
            log_sigma = uncertainty[..., column]
            standardized = residual[..., mean_channel] * torch.exp(-log_sigma)
            terms[mean_channel] = _HALF_LOG_TWO_PI + log_sigma + 0.5 * standardized * standardized

        return torch.stack(terms, dim=-1)

    def variance(self, prediction: torch.Tensor) -> torch.Tensor:
        """Per-channel ``sigma**2`` implied by the uncertainty block, ``(..., 11)``."""

        uncertainty = prediction[..., PHYSICAL_MEAN_DIM:]
        columns = [0] * PHYSICAL_MEAN_DIM
        columns[POSITION_X], columns[POSITION_Y] = POSITION_SIGMA
        columns[VELOCITY_X], columns[VELOCITY_Y] = VELOCITY_SIGMA
        for mean_channel, column in SCALAR_UNCERTAINTY:
            columns[mean_channel] = column
        index = torch.tensor(columns, dtype=torch.long, device=uncertainty.device)
        return torch.exp(2.0 * uncertainty.index_select(-1, index))

    def log_sigma(self, prediction: torch.Tensor) -> torch.Tensor:
        """Per-channel log spread, ``(..., 11)``, in mean-channel order.

        :meth:`variance` gathers with an index tensor built on the device, a
        synchronizing host copy; this stacks slices instead, so it is safe on the
        per-micro-batch path.
        """

        uncertainty = prediction[..., PHYSICAL_MEAN_DIM:]
        columns = [0] * PHYSICAL_MEAN_DIM
        columns[POSITION_X], columns[POSITION_Y] = POSITION_SIGMA
        columns[VELOCITY_X], columns[VELOCITY_Y] = VELOCITY_SIGMA
        for mean_channel, column in SCALAR_UNCERTAINTY:
            columns[mean_channel] = column
        return torch.stack([uncertainty[..., column] for column in columns], dim=-1)

    def standardized_square(self, prediction: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-channel squared standardized residual, ``(..., 11)``.

        The calibration reading of :meth:`loss`: a head whose spreads are honest
        averages 1.0 on every channel, above 1.0 where it claims more certainty
        than it has, and below where it claims less. Position and velocity use
        the bivariate Mahalanobis distance, split evenly between the two axes
        exactly as the likelihood is, so each axis also averages 1.0 when
        calibrated -- a per-axis ``(r / sigma)**2`` would not, once rho is
        nonzero.
        """

        residual = self.residual(prediction, labels)
        uncertainty = prediction[..., PHYSICAL_MEAN_DIM:]
        terms = [None] * PHYSICAL_MEAN_DIM
        for mean_x, axes, rho_column in (
            (POSITION_X, POSITION_SIGMA, POSITION_RHO),
            (VELOCITY_X, VELOCITY_SIGMA, VELOCITY_RHO),
        ):
            a = residual[..., mean_x] * torch.exp(-uncertainty[..., axes[0]])
            b = residual[..., mean_x + 1] * torch.exp(-uncertainty[..., axes[1]])
            rho = torch.tanh(uncertainty[..., rho_column])
            one_minus = (1.0 - rho * rho).clamp_min(1e-6)
            half = 0.5 * (a * a - 2.0 * rho * a * b + b * b) / one_minus
            terms[mean_x] = half
            terms[mean_x + 1] = half
        for mean_channel, column in SCALAR_UNCERTAINTY:
            standardized = residual[..., mean_channel] * torch.exp(-uncertainty[..., column])
            terms[mean_channel] = standardized * standardized
        return torch.stack(terms, dim=-1)


_VECTOR_CACHE: dict[tuple[tuple[float, ...], torch.device], torch.Tensor] = {}


def _cached_vector(values: tuple[float, ...], device: torch.device | str) -> torch.Tensor:
    """A constant vector on ``device``, built once.

    Cached because these sit in the per-micro-batch loss and per-step belief
    paths, where ``torch.tensor([...], device="cuda")`` is a synchronizing host
    copy. Safe to cache, unlike the encoder's frequency tables: nothing here is
    created inside a CUDA-graph capture -- the objective is called from eager
    Python and the values are read-only.
    """

    key = (values, torch.device(device))
    cached = _VECTOR_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(values, dtype=torch.float32, device=key[1])
        _VECTOR_CACHE[key] = cached
    return cached
