"""GPU-vectorized ship and bullet physics.

All functions operate on TensorState in-place and return the mutated state.
No Python loops over batch or ship dimensions.
"""

from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import EPS, ShootActions
from boost_and_broadside.env import shot_labels
from boost_and_broadside.env.field_physics import FieldEvaluation, evaluate_fields
from boost_and_broadside.env.state import TensorState

# ---------------------------------------------------------------------------
# Lookup table construction
# ---------------------------------------------------------------------------

# Keyed by (config, device string). Building the tables allocates tensors from
# Python lists (host→device copies), so they must not be rebuilt on the
# per-step hot path.
_LOOKUP_TABLE_CACHE: dict = {}

# Turn action → (side, sharp). Left is the negative slip direction.
TURN_SIDE = (0.0, -1.0, 1.0, -1.0, 1.0)
TURN_SHARP = (False, False, False, True, True)


class _ActionTables(NamedTuple):
    thrust: torch.Tensor  # (3,) float32 — by power action
    turn_side: torch.Tensor  # (5,) float32 — by turn action, {-1, 0, +1}
    turn_sharp: torch.Tensor  # (5,) bool — by turn action


def action_tables(config: ShipConfig, device: torch.device) -> _ActionTables:
    """Return cached per-action physics lookup tensors for (config, device)."""
    key = (config, str(device))
    tables = _LOOKUP_TABLE_CACHE.get(key)
    if tables is None:
        tables = _ActionTables(
            torch.tensor(
                [config.base_thrust, config.boost_thrust, config.reverse_thrust],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(TURN_SIDE, device=device, dtype=torch.float32),
            torch.tensor(TURN_SHARP, device=device, dtype=torch.bool),
        )
        _LOOKUP_TABLE_CACHE[key] = tables
    return tables


# ---------------------------------------------------------------------------
# Kinematics update
# ---------------------------------------------------------------------------


def _field_optical_acceleration(
    velocity: torch.Tensor,
    index: torch.Tensor,
    grad_index: torch.Tensor,
) -> torch.Tensor:
    """Passive acceleration for effective mass ``m=n²``.

    ``a = |v|² grad(log n) - 2 (v·grad(log n)) v``. The expression is
    deterministic and smooth; reflection emerges from the same force as
    transmission rather than from a random or hard collision branch.
    """

    grad_log_n = grad_index / index.clamp(min=EPS)
    speed_sq = velocity.real**2 + velocity.imag**2
    directional = (velocity * torch.conj(grad_log_n)).real
    return speed_sq * grad_log_n - 2.0 * directional * velocity


def _wrap_positions(
    position: torch.Tensor,
    world_size: tuple[float, float],
) -> torch.Tensor:
    """Wrap complex positions onto the toroidal world."""
    world_w, world_h = world_size
    return torch.complex(position.real % world_w, position.imag % world_h)


def _quadratic_drag_scale(
    velocity: torch.Tensor,
    coefficient: float | torch.Tensor,
    duration: float,
) -> torch.Tensor:
    """Return the exact multiplicative speed change for quadratic drag."""
    return 1.0 / (1.0 + coefficient * velocity.abs() * duration)


class _FieldTransportResult(NamedTuple):
    position: torch.Tensor
    velocity: torch.Tensor
    index: torch.Tensor
    grad_index: torch.Tensor
    half_tick_position: torch.Tensor


class BulletTrajectory(NamedTuple):
    """Ephemeral points defining two swept bullet-collision segments."""

    start: torch.Tensor
    midpoint: torch.Tensor


def _evaluate_state_fields(
    points: torch.Tensor,
    state: TensorState,
    world_size: tuple[float, float],
) -> FieldEvaluation:
    return evaluate_fields(
        points,
        state.field_pos,
        state.field_radius,
        state.field_transition_width,
        state.field_index,
        world_size,
    )


def _two_step_field_substep(
    position: torch.Tensor,
    velocity: torch.Tensor,
    index: torch.Tensor,
    grad_index: torch.Tensor,
    step_dt: float,
    state: TensorState,
    world_size: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, FieldEvaluation]:
    """Fast explicit optical kick followed by drift and endpoint evaluation."""
    acceleration = _field_optical_acceleration(velocity, index, grad_index)
    direction_velocity = velocity + acceleration * step_dt
    next_position = _wrap_positions(
        position + direction_velocity * step_dt,
        world_size,
    )
    next_eval = _evaluate_state_fields(next_position, state, world_size)
    return next_position, direction_velocity, next_eval


def _midpoint_field_substep(
    position: torch.Tensor,
    velocity: torch.Tensor,
    index: torch.Tensor,
    grad_index: torch.Tensor,
    step_dt: float,
    state: TensorState,
    world_size: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, FieldEvaluation]:
    """Midpoint optical integration."""
    acceleration = _field_optical_acceleration(velocity, index, grad_index)
    midpoint_velocity = velocity + 0.5 * acceleration * step_dt
    midpoint_position = _wrap_positions(
        position + 0.5 * velocity * step_dt + 0.125 * acceleration * step_dt**2,
        world_size,
    )
    midpoint_eval = _evaluate_state_fields(midpoint_position, state, world_size)
    midpoint_acceleration = _field_optical_acceleration(
        midpoint_velocity,
        midpoint_eval.index,
        midpoint_eval.grad_index,
    )
    direction_velocity = velocity + midpoint_acceleration * step_dt
    next_position = _wrap_positions(
        position + 0.5 * (velocity + direction_velocity) * step_dt,
        world_size,
    )
    next_eval = _evaluate_state_fields(next_position, state, world_size)
    return next_position, direction_velocity, next_eval


_FIELD_INTEGRATORS = {
    "two_step": _two_step_field_substep,
    "midpoint": _midpoint_field_substep,
}


def _transport_field_points(
    position: torch.Tensor,
    velocity: torch.Tensor,
    index: torch.Tensor,
    grad_index: torch.Tensor,
    state: TensorState,
    world_size: tuple[float, float],
    integrator_name: str,
    substeps: int,
    duration: float,
    fallback_direction: torch.Tensor | None = None,
) -> _FieldTransportResult:
    """Transport arbitrary batched points using one selectable optical integrator."""
    integrator = _FIELD_INTEGRATORS[integrator_name]
    step_dt = duration / substeps
    half_tick_position = position

    for substep in range(substeps):
        next_position, direction_velocity, next_eval = integrator(
            position,
            velocity,
            index,
            grad_index,
            step_dt,
            state,
            world_size,
        )
        proper_speed = index * velocity.abs()
        direction_speed = direction_velocity.abs()
        direction = direction_velocity / direction_speed.clamp(min=EPS)
        if fallback_direction is not None:
            direction = torch.where(
                direction_speed > EPS,
                direction,
                fallback_direction,
            )
        velocity = direction * (proper_speed / next_eval.index)
        position = next_position
        index = next_eval.index
        grad_index = next_eval.grad_index

        if substep + 1 == max(1, substeps // 2):
            half_tick_position = position

    return _FieldTransportResult(
        position,
        velocity,
        index,
        grad_index,
        half_tick_position,
    )


def _apply_thrust_impulse_with_power(
    state: TensorState,
    thrust_mag: torch.Tensor,
    duration: float,
    config: ShipConfig,
) -> TensorState:
    """Apply a generalized thrust impulse and exchange its exact work with power."""

    mass = state.ship_local_index.square()
    positive = thrust_mag > 0.0
    # Forward/coast thrust cannot spend unavailable power. Reverse remains
    # available at zero power because it converts kinetic energy back to power.
    active_mag = torch.where(
        positive & (state.ship_power <= 0.0),
        torch.zeros_like(thrust_mag),
        thrust_mag,
    )
    dv_full = active_mag * state.ship_attitude * (duration / mass)

    # ΔH(λ) = Aλ + Bλ² for impulse fraction λ.
    a = mass * (state.ship_vel * torch.conj(dv_full)).real
    b = 0.5 * mass * dv_full.abs().square()
    b_safe = b.clamp(min=1e-12)

    # Reverse may decelerate only as far as the minimum-energy point. Continuing
    # through zero velocity would turn reverse into an unpowered backward boost.
    reverse_limit = (-a / (2.0 * b_safe)).clamp(0.0, 1.0)
    impulse_fraction = torch.where(active_mag < 0.0, reverse_limit, torch.ones_like(a))
    impulse_fraction = torch.where(active_mag == 0.0, torch.zeros_like(a), impulse_fraction)
    delta_energy = a * impulse_fraction + b * impulse_fraction.square()

    # Cap positive work by available power using the exact quadratic root.
    spendable = state.ship_power * config.power_speed_constant
    forward_root = (-a + torch.sqrt((a.square() + 4.0 * b * spendable).clamp(min=0.0))) / (
        2.0 * b_safe
    )
    linear_forward_root = spendable / a.clamp(min=1e-12)
    forward_root = torch.where(b > 1e-12, forward_root, linear_forward_root)
    needs_spend_cap = delta_energy > spendable
    impulse_fraction = torch.where(
        needs_spend_cap,
        torch.minimum(impulse_fraction, forward_root.clamp(0.0, 1.0)),
        impulse_fraction,
    )

    # Cap recovered work by remaining storage. On the descending branch the
    # smaller root reaches exactly -recoverable energy.
    recoverable = (config.max_power - state.ship_power) * config.power_speed_constant
    recovery_disc = (a.square() - 4.0 * b * recoverable).clamp(min=0.0)
    recovery_root = (-a - torch.sqrt(recovery_disc)) / (2.0 * b_safe)
    linear_recovery_root = recoverable / (-a).clamp(min=1e-12)
    recovery_root = torch.where(b > 1e-12, recovery_root, linear_recovery_root)
    needs_recovery_cap = -delta_energy > recoverable
    impulse_fraction = torch.where(
        needs_recovery_cap,
        torch.minimum(impulse_fraction, recovery_root.clamp(0.0, 1.0)),
        impulse_fraction,
    )

    dv = dv_full * impulse_fraction
    actual_delta_energy = a * impulse_fraction + b * impulse_fraction.square()
    state.ship_vel = state.ship_vel + dv
    state.ship_power = torch.clamp(
        state.ship_power - actual_delta_energy / config.power_speed_constant,
        0.0,
        config.max_power,
    )
    return state


def _apply_flight_half_step(
    state: TensorState,
    thrust_mag: torch.Tensor,
    drag_coeff: torch.Tensor,
    lift_coeff: torch.Tensor,
    config: ShipConfig,
) -> TensorState:
    """Apply one control/drag/lift half-step at the current local index."""

    duration = 0.5 * config.dt
    state = _apply_thrust_impulse_with_power(state, thrust_mag, duration, config)

    speed = state.ship_vel.abs()
    direction = state.ship_vel / speed.clamp(min=EPS)

    # Proper-speed force magnitude is c*(n|v|)^2. Division by m=n² yields
    # dv/dt=-c|v|v. Integrating its scalar speed ODE exactly guarantees drag
    # dissipates rather than gaining energy through an Euler overshoot.
    dragged_speed = speed * _quadratic_drag_scale(state.ship_vel, drag_coeff, duration)

    # Lift is perpendicular work-free rotation. At a fixed proper speed its
    # world turn rate is reciprocal in n, matching faster/slower local handling.
    lift_angle = lift_coeff * dragged_speed * duration
    rotation = torch.polar(torch.ones_like(lift_angle), lift_angle)
    state.ship_vel = direction * dragged_speed * rotation

    if config.gravity_factor != 0.0:
        world_w, world_h = config.world_size
        proper_speed = state.ship_local_index * state.ship_vel.abs()
        diff = state.ship_pos.unsqueeze(1) - state.ship_pos.unsqueeze(2)
        diff.real = (diff.real + world_w / 2) % world_w - world_w / 2
        diff.imag = (diff.imag + world_h / 2) % world_h - world_h / 2
        dist_sq = diff.real**2 + diff.imag**2
        dist = torch.sqrt(dist_sq)
        speed_i = proper_speed.unsqueeze(2)
        speed_j = proper_speed.unsqueeze(1)
        force_mag = (
            config.gravity_factor
            * config.gravity_eps
            * torch.log1p(speed_i * speed_j)
            / (dist_sq + config.gravity_eps)
        )
        force_dir = diff / dist.clamp(min=EPS)
        force = force_mag * force_dir
        alive = state.ship_alive.unsqueeze(2) & state.ship_alive.unsqueeze(1)
        diagonal = torch.eye(state.max_ships, device=state.device, dtype=torch.bool).unsqueeze(0)
        gravity = torch.where(alive & ~diagonal, force, torch.zeros_like(force)).sum(dim=2)
        state.ship_vel = state.ship_vel + gravity / state.ship_local_index.square() * duration

    return state


def _transport_through_fields(state: TensorState, config: ShipConfig) -> TensorState:
    """Passive optical ship transport with generalized-energy projection."""

    result = _transport_field_points(
        state.ship_pos,
        state.ship_vel,
        state.ship_local_index,
        state.ship_field_gradient,
        state,
        config.world_size,
        config.field_integrator,
        config.field_integration_substeps,
        config.dt,
        state.ship_attitude,
    )
    state.ship_pos = result.position
    state.ship_vel = result.velocity
    state.ship_local_index = result.index
    state.ship_field_gradient = result.grad_index
    return state


def slip_command(
    turn_side: torch.Tensor,
    turn_sharp: torch.Tensor,
    proper_speed: torch.Tensor,
    config: ShipConfig,
) -> torch.Tensor:
    """The slip a turn action commands at a proper speed, in radians.

    The action asks for a lift coefficient, capped so that lateral
    acceleration ``C_L u^2`` stays within ``max_lateral_accel`` (the normal
    level within half of it); the slip is the exact inverse of the lift curve.
    Below ``slip_fade_speed`` the command fades linearly with speed.

    Args:
        turn_side: (B, N) float — -1 left, 0 straight, +1 right.
        turn_sharp: (B, N) bool — sharp rather than normal pull.
        proper_speed: (B, N) float — ``n |v|``.

    Returns:
        (B, N) float — commanded slip.
    """
    speed_sq = proper_speed.square().clamp(min=EPS)
    sharp_lift = (config.max_lateral_accel / speed_sq).clamp(max=config.max_lift_coeff)
    normal_lift = (0.5 * config.max_lateral_accel / speed_sq).clamp(max=config.normal_lift_coeff)
    lift = torch.where(turn_sharp, sharp_lift, normal_lift)  # (B, N)
    fraction = 1.0 - torch.sqrt((1.0 - lift / config.max_lift_coeff).clamp(min=0.0))
    fade = (proper_speed / config.slip_fade_speed).clamp(max=1.0)
    return turn_side * config.stall_angle * fraction * fade


def slip_lift_drag(slip: torch.Tensor, config: ShipConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed lift and drag coefficients at a slip angle.

    Past the stall angle lift falls along the same parabola and is clamped at
    zero. Drag is held at its value at twice the stall angle beyond it: only a
    ship leaving a stall with a large held slip gets there, and the quartic
    would otherwise stop it dead.

    Returns:
        ``(lift, drag)``, each shaped like ``slip``.
    """
    fraction = slip.abs() / config.stall_angle
    lift = torch.sign(slip) * config.max_lift_coeff * (1.0 - (1.0 - fraction).square())
    lift = torch.where(lift * torch.sign(slip) > 0.0, lift, torch.zeros_like(lift))
    quadratic, quartic = config.slip_drag_terms
    drag_fraction_sq = fraction.clamp(max=2.0).square()
    drag = config.zero_slip_drag_coeff + drag_fraction_sq * (quadratic + quartic * drag_fraction_sq)
    return lift, drag


def _damped_error(
    error: torch.Tensor, rate: torch.Tensor, frequency: torch.Tensor, horizon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact critically damped evolution of ``(error, rate)`` over ``horizon``."""
    decay = torch.exp(-frequency * horizon)
    drive = rate + frequency * error
    return (error + drive * horizon) * decay, (rate - frequency * drive * horizon) * decay


def _nose_step(
    slip: torch.Tensor,
    slip_rate: torch.Tensor,
    command: torch.Tensor,
    index: torch.Tensor,
    config: ShipConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the slip one tick toward its command.

    A critically damped response stepped exactly with the command held, so it
    is stable at any ``omega dt`` and never rings. The ship is a rotor of
    inertia ``n^2`` with torques on proper quantities, so in world time its
    frequency is ``omega / n`` and its slip-rate limit ``max_slip_rate / n``.
    The rate limit can leave a residual approach speed the exact step would
    carry past the command; arrival is clamped so the slip never overshoots.

    Args:
        slip, slip_rate, command, index: (B, N) float.

    Returns:
        ``(slip, slip_rate)`` after one tick.
    """
    dt = config.dt
    frequency = config.nose_frequency / index  # (B, N) world rad/s
    rate_limit = config.max_slip_rate / index  # (B, N)
    error = slip - command
    next_error, next_rate = _damped_error(error, slip_rate, frequency, dt)

    step = (next_error - error).clamp(min=-rate_limit * dt, max=rate_limit * dt)
    next_error = error + step
    next_rate = torch.maximum(torch.minimum(next_rate, rate_limit), -rate_limit)
    arrived = next_error * error <= 0.0
    next_error = torch.where(arrived, torch.zeros_like(next_error), next_error)
    next_rate = torch.where(arrived, torch.zeros_like(next_rate), next_rate)

    next_slip = command + next_error
    bound = torch.maximum(slip.abs(), torch.full_like(slip, config.stall_angle))
    next_slip = torch.maximum(torch.minimum(next_slip, bound), -bound)
    return next_slip, next_rate


def _drift(state: TensorState, config: ShipConfig) -> TensorState:
    """Field-free transport: a straight drift over one tick at n = 1."""

    state.ship_pos = _wrap_positions(state.ship_pos + state.ship_vel * config.dt, config.world_size)
    return state


def _update_kinematics(
    state: TensorState,
    actions: torch.Tensor,
    config: ShipConfig,
    tables: _ActionTables,
) -> TensorState:
    """Advance the nose, then fly: half-step, transport, half-step.

    Transport is optical through the field map, or a plain drift when the map
    has no fields. Either way the flight steps are the same exact half-steps,
    so there is one flight model: at n = 1 the two transports coincide.

    GPU kernel: kept together for performance.
    """

    power_action = actions[..., 0].long()  # (B, N)
    turn_action = actions[..., 1].long()  # (B, N)
    thrust_mag = tables.thrust[power_action]  # (B, N)
    turn_side = tables.turn_side[turn_action]  # (B, N)
    turn_sharp = tables.turn_sharp[turn_action]  # (B, N)

    index = state.ship_local_index  # (B, N)
    proper_speed = index * state.ship_vel.abs()
    below_min_speed = proper_speed < config.min_speed
    # Reverse is a kinetic-energy recovery action and has no useful effect at a
    # stall; forward thrust remains able to restart the ship.
    thrust_mag = torch.where(
        below_min_speed & (thrust_mag < 0.0), torch.zeros_like(thrust_mag), thrust_mag
    )

    previous_attitude = state.ship_attitude
    speed = state.ship_vel.abs()
    velocity_direction = state.ship_vel / speed.clamp(min=EPS)  # (B, N) complex

    command = slip_command(turn_side, turn_sharp, proper_speed, config)
    flown_slip, flown_rate = _nose_step(
        state.ship_slip, state.ship_slip_rate, command, index, config
    )
    # A stalled ship holds its absolute attitude; its slip is re-derived from
    # it with zero rate, so leaving the stall does not jump the nose.
    held_slip = torch.angle(previous_attitude * torch.conj(velocity_direction))
    slip = torch.where(below_min_speed, held_slip, flown_slip)
    state.ship_slip_rate = torch.where(below_min_speed, torch.zeros_like(flown_rate), flown_rate)

    slip_rotation = torch.polar(torch.ones_like(slip), slip)
    state.ship_attitude = torch.where(
        below_min_speed, previous_attitude, velocity_direction * slip_rotation
    )
    lift_coeff, drag_coeff = slip_lift_drag(slip, config)
    lift_coeff = torch.where(below_min_speed, torch.zeros_like(lift_coeff), lift_coeff)

    state = _apply_flight_half_step(state, thrust_mag, drag_coeff, lift_coeff, config)
    if state.num_fields == 0:
        state = _drift(state, config)
    else:
        state = _transport_through_fields(state, config)
    state = _apply_flight_half_step(state, thrust_mag, drag_coeff, lift_coeff, config)
    state.ship_power = torch.clamp(
        state.ship_power + config.passive_power_gain * config.dt,
        0.0,
        config.max_power,
    )

    speed = state.ship_vel.abs()
    state.ship_vel = torch.where(speed < EPS, EPS * state.ship_attitude, state.ship_vel)

    # The nose rides the flight path through the tick, so the attitude read
    # after it is the end-of-tick path rotated by the slip, and the turn rate is
    # the path's plus the slip rate. A stalled ship's held attitude is measured
    # against the end-of-tick path instead. Either way slip is exactly attitude
    # minus velocity heading, so the state stays observable.
    end_direction = state.ship_vel / state.ship_vel.abs().clamp(min=EPS)
    state.ship_attitude = torch.where(
        below_min_speed, state.ship_attitude, end_direction * slip_rotation
    )
    end_held_slip = torch.angle(state.ship_attitude * torch.conj(end_direction))
    state.ship_slip = torch.where(below_min_speed, end_held_slip, slip)
    state.ship_ang_vel = (
        torch.angle(state.ship_attitude * torch.conj(previous_attitude)) / config.dt
    )
    return state


# ---------------------------------------------------------------------------
# Shooting
# ---------------------------------------------------------------------------


def _handle_shooting(
    state: TensorState,
    shoot_action: torch.Tensor,
    config: ShipConfig,
    shot_generator: torch.Generator | None = None,
) -> TensorState:
    """Manage cooldowns and spawn bullets for ships that fire.

    Fully branchless: bullet spawns are written through a one-hot mask on the
    ring-buffer cursor instead of dynamic indexing, so no host-device sync occurs
    (this runs every step of every rollout).

    With a ``shot_generator`` (shot labels on) a living ship that does not fire
    launches a ghost whenever its ghost cooldown allows; see env/shot_labels.py.
    A ghost shares the real launch's state and noise draw, is never active, and
    leaves power, the firing cooldown and ``ship_is_shooting`` alone. A real
    shot inside a ghost's cooldown takes over that ghost's slot, which keeps
    every launch one cooldown apart, and every other launch reclaims a slot
    instead of following the ring cursor.
    """
    if state.max_bullets == 0:
        state.ship_is_shooting = torch.zeros_like(state.ship_is_shooting)
        return state

    state.ship_cooldown = (state.ship_cooldown - config.dt).clamp(min=0.0)
    labels = shot_generator is not None
    if labels:
        state.ship_ghost_cooldown = (state.ship_ghost_cooldown - config.dt).clamp(min=0.0)

    can_shoot = (
        (shoot_action == ShootActions.SHOOT)
        & (state.ship_cooldown <= 0)
        & (state.ship_power >= config.bullet_energy_cost)
        & state.ship_alive
    )  # (B, N) bool
    state.ship_is_shooting = can_shoot

    state.ship_power = torch.where(
        can_shoot, state.ship_power - config.bullet_energy_cost, state.ship_power
    )
    state.ship_cooldown = torch.where(
        can_shoot,
        config.firing_cooldown,
        state.ship_cooldown,
    )

    K = state.max_bullets
    if labels:
        flying = state.bullet_flying
        spent = flying & ~state.bullet_active  # a ghost, or a real bullet that hit
        # The most recent flight. While the ghost cooldown runs and a real shot
        # is possible, it is the ghost that started the cooldown -- unless a
        # respawn cleared the firing cooldown early, when it may be the real
        # shot itself and must not be replaced.
        latest = torch.where(flying, state.bullet_age, torch.iinfo(torch.int32).max).argmin(-1)
        latest_spent = spent.gather(-1, latest.unsqueeze(-1)).squeeze(-1)
        replaces_ghost = can_shoot & (state.ship_ghost_cooldown > 0) & latest_spent
        # A ghost never displaces a live real bullet: with every slot live it
        # waits for one to free up.
        ghost = (
            ~can_shoot
            & (state.ship_ghost_cooldown <= 0)
            & state.ship_alive
            & (~state.bullet_active).any(-1)
        )
        launch = can_shoot | ghost
        state.ship_ghost_cooldown = torch.where(
            launch, config.firing_cooldown, state.ship_ghost_cooldown
        )
        # Otherwise reclaim: a free slot, else the oldest spent flight (only a
        # label is lost), else the oldest live bullet -- the one the plain ring
        # would have overwritten, so gameplay never depends on the ghosts.
        # Launches a cooldown apart always leave a free slot; a same-tick
        # respawn can briefly compress them.
        rank = torch.where(~flying, 2, torch.where(spent, 1, 0))
        reclaimed = (rank * (shot_labels.MAX_AGE_KEY + 1) + state.bullet_age).argmax(-1)
        slot = torch.where(replaces_ghost, latest, reclaimed)
        advances_cursor = torch.zeros_like(launch)  # the ring cursor is unused
    else:
        launch = can_shoot
        slot = state.bullet_cursor
        advances_cursor = can_shoot
    slot_onehot = F.one_hot(slot, K).bool() & launch.unsqueeze(-1)  # (B, N, K)

    # bullet_speed is a proper muzzle speed, just like configured ship speeds.
    # Dividing by local n avoids creating energy when firing inside a medium.
    muzzle_vel = config.bullet_speed * state.ship_attitude / state.ship_local_index
    base_vel = state.ship_vel + muzzle_vel  # (B, N)
    noise = torch.complex(
        torch.randn_like(base_vel.real) * config.bullet_spread,
        torch.randn_like(base_vel.real) * config.bullet_spread,
    )
    spawn_vel = base_vel + noise

    state.bullet_pos = torch.where(slot_onehot, state.ship_pos.unsqueeze(-1), state.bullet_pos)
    state.bullet_vel = torch.where(slot_onehot, spawn_vel.unsqueeze(-1), state.bullet_vel)
    state.bullet_time = torch.where(slot_onehot, config.bullet_lifetime, state.bullet_time)
    # A launched slot is active exactly when it is real; without labels every
    # launch is real and this is the plain union.
    state.bullet_active = torch.where(slot_onehot, can_shoot.unsqueeze(-1), state.bullet_active)
    state.bullet_local_index = torch.where(
        slot_onehot,
        state.ship_local_index.unsqueeze(-1),
        state.bullet_local_index,
    )
    state.bullet_field_gradient = torch.where(
        slot_onehot,
        state.ship_field_gradient.unsqueeze(-1),
        state.bullet_field_gradient,
    )
    state.bullet_cursor = torch.where(
        advances_cursor, (state.bullet_cursor + 1) % K, state.bullet_cursor
    )
    if labels:
        shot_labels.record_launch(state, slot_onehot, config, shot_generator)

    return state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def update_ships(
    state: TensorState,
    actions: torch.Tensor,
    config: ShipConfig,
    shot_generator: torch.Generator | None = None,
) -> TensorState:
    """Apply one physics timestep: kinematics + shooting.

    Args:
        state: Current environment state (mutated in-place).
        actions: (B, N, 3) int tensor — [power_action, turn_action, shoot_action].
        config: Physics configuration.
        shot_generator: The shot-label random stream; ``None`` launches no ghosts.

    Returns:
        The mutated state.
    """
    tables = action_tables(config, state.device)
    state = _update_kinematics(state, actions, config, tables)
    state = _handle_shooting(state, actions[..., 2].long(), config, shot_generator)
    return state


def _transport_bullets_through_fields(
    state: TensorState,
    transport_vel: torch.Tensor,
    config: ShipConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuously refract bullets and return final velocity/position and half-tick position."""
    batch_size, num_ships, num_bullets = state.bullet_pos.shape
    num_flat_bullets = num_ships * num_bullets
    result = _transport_field_points(
        state.bullet_pos.view(batch_size, num_flat_bullets),
        transport_vel.view(batch_size, num_flat_bullets),
        state.bullet_local_index.view(batch_size, num_flat_bullets),
        state.bullet_field_gradient.view(batch_size, num_flat_bullets),
        state,
        config.world_size,
        config.bullet_field_integrator,
        config.bullet_field_integration_substeps,
        config.dt,
    )

    state.bullet_local_index = result.index.view(batch_size, num_ships, num_bullets)
    state.bullet_field_gradient = result.grad_index.view(batch_size, num_ships, num_bullets)
    return (
        result.velocity.view(batch_size, num_ships, num_bullets),
        result.position.view(batch_size, num_ships, num_bullets),
        result.half_tick_position.view(batch_size, num_ships, num_bullets),
    )


def advance_bullets(
    state: TensorState,
    config: ShipConfig,
) -> tuple[TensorState, BulletTrajectory]:
    """Advance bullets and return the start and midpoint of each trajectory.

    Args:
        state: Current state (mutated in-place).
        config: Physics configuration.

    Returns:
        The state and ephemeral points defining two swept collision segments.
    """
    start_pos = state.bullet_pos
    state.bullet_time = state.bullet_time - config.dt
    state.bullet_active = state.bullet_active & (state.bullet_time > 0)
    transport_vel = state.bullet_vel
    if config.bullet_drag_coeff != 0.0:
        transport_vel = transport_vel * _quadratic_drag_scale(
            transport_vel,
            config.bullet_drag_coeff,
            0.5 * config.dt,
        )
    if state.num_fields:
        transport_vel, state.bullet_pos, midpoint_pos = _transport_bullets_through_fields(
            state,
            transport_vel,
            config,
        )
    else:
        midpoint_pos = _wrap_positions(
            start_pos + transport_vel * (0.5 * config.dt),
            config.world_size,
        )
        state.bullet_pos = _wrap_positions(
            start_pos + transport_vel * config.dt,
            config.world_size,
        )
    if config.bullet_drag_coeff != 0.0:
        transport_vel = transport_vel * _quadratic_drag_scale(
            transport_vel,
            config.bullet_drag_coeff,
            0.5 * config.dt,
        )
    state.bullet_vel = transport_vel
    return state, BulletTrajectory(start_pos, midpoint_pos)


def resolve_collisions(
    state: TensorState,
    config: ShipConfig,
    combat_damage_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    trajectory: BulletTrajectory | None = None,
    frontline=None,
    combat_proximity_fn: Callable[..., tuple[torch.Tensor, ...]] | None = None,
) -> tuple[TensorState, torch.Tensor]:
    """Detect bullet-ship collisions, apply damage, and check game-over.

    Args:
        state: Current state (mutated in-place).
        config: Physics configuration.
        combat_proximity_fn: Optional compiled
            :func:`_combat_damage_and_proximity_tensors`, used with shot labels.

    Returns:
        (state, dones) where dones is a (B,) bool tensor.
    """
    state = _apply_combat_damage(
        state,
        config,
        combat_damage_fn,
        trajectory,
        frontline,
        combat_proximity_fn,
    )
    dones = _check_game_over(state)
    return state, dones


def _swept_dist_sq(
    ship_pos: torch.Tensor,
    bullet_start_pos: torch.Tensor,
    bullet_midpoint_pos: torch.Tensor,
    bullet_pos: torch.Tensor,
    world_size: tuple[float, float],
) -> torch.Tensor:
    """Least squared distance from every ship to every slot's two swept segments.

    Returns:
        ``(B, N*K, N)`` over flat bullets then target ships, toroidal.
    """
    batch_size, num_ships = ship_pos.shape
    num_flat_bullets = num_ships * bullet_pos.shape[2]
    world_w, world_h = world_size

    flat_bullet_start = bullet_start_pos.view(batch_size, num_flat_bullets)
    flat_bullet_midpoint = bullet_midpoint_pos.view(batch_size, num_flat_bullets)
    flat_bullet_pos = bullet_pos.view(batch_size, num_flat_bullets)

    def segment_dist_sq(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        delta_r = (end.real - start.real + world_w / 2) % world_w - world_w / 2
        delta_i = (end.imag - start.imag + world_h / 2) % world_h - world_h / 2
        rel_r = ship_pos.real.unsqueeze(1) - start.real.unsqueeze(2)
        rel_i = ship_pos.imag.unsqueeze(1) - start.imag.unsqueeze(2)
        rel_r = (rel_r + world_w / 2) % world_w - world_w / 2
        rel_i = (rel_i + world_h / 2) % world_h - world_h / 2
        delta_r = delta_r.unsqueeze(2)
        delta_i = delta_i.unsqueeze(2)
        segment_length_sq = delta_r.square() + delta_i.square()
        along = (rel_r * delta_r + rel_i * delta_i) / segment_length_sq.clamp(min=EPS)
        along = along.clamp(0.0, 1.0)
        closest_r = rel_r - along * delta_r
        closest_i = rel_i - along * delta_i
        return closest_r.square() + closest_i.square()

    first_dist_sq = segment_dist_sq(flat_bullet_start, flat_bullet_midpoint)
    second_dist_sq = segment_dist_sq(flat_bullet_midpoint, flat_bullet_pos)
    return torch.minimum(first_dist_sq, second_dist_sq)


def _not_own_bullet(num_ships: int, num_bullets: int, device: torch.device) -> torch.Tensor:
    """``(N*K, N)``: the flat bullet was not fired by the target ship."""
    owner_idx = torch.arange(num_ships * num_bullets, device=device) // num_bullets
    target_idx = torch.arange(num_ships, device=device)
    return owner_idx.unsqueeze(1) != target_idx.unsqueeze(0)


def _damage_from_dist_sq(
    dist_sq: torch.Tensor,
    ship_attitude: torch.Tensor,
    ship_alive: torch.Tensor,
    bullet_vel: torch.Tensor,
    bullet_active: torch.Tensor,
    bullet_damage: float,
    collision_radius: float,
    bullet_min_damage_frac: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gameplay hits, damage and surviving bullets from the swept distances."""
    batch_size, num_ships, num_bullets = bullet_active.shape
    num_flat_bullets = num_ships * num_bullets
    flat_bullet_active = bullet_active.view(batch_size, num_flat_bullets)
    flat_bullet_vel = bullet_vel.view(batch_size, num_flat_bullets)

    not_own_bullet = _not_own_bullet(num_ships, num_bullets, dist_sq.device)
    valid_hit = (
        (dist_sq < collision_radius**2)
        & ship_alive.unsqueeze(1)
        & flat_bullet_active.unsqueeze(2)
        & not_own_bullet.unsqueeze(0)
    )

    if bullet_min_damage_frac == 1.0:
        damage_per_hit = valid_hit.float() * bullet_damage
    else:
        hit_angles = torch.angle(
            -flat_bullet_vel.unsqueeze(2) * torch.conj(ship_attitude.unsqueeze(1))
        )
        damage_scale = 1.0 - (1.0 - bullet_min_damage_frac) * torch.exp(
            -(hit_angles**2) * 4.0 / torch.pi
        )
        damage_per_hit = damage_scale * valid_hit.float() * bullet_damage

    total_damage = damage_per_hit.sum(dim=1)
    per_shooter = damage_per_hit.view(batch_size, num_ships, num_bullets, num_ships).sum(dim=2)
    hit_any_ship = valid_hit.any(dim=2)
    next_bullet_active = (flat_bullet_active & ~hit_any_ship).view(
        batch_size, num_ships, num_bullets
    )
    return total_damage, per_shooter, next_bullet_active


def _combat_damage_tensors(
    ship_pos: torch.Tensor,
    ship_attitude: torch.Tensor,
    ship_alive: torch.Tensor,
    bullet_start_pos: torch.Tensor,
    bullet_midpoint_pos: torch.Tensor,
    bullet_pos: torch.Tensor,
    bullet_vel: torch.Tensor,
    bullet_active: torch.Tensor,
    bullet_damage: float,
    collision_radius: float,
    bullet_min_damage_frac: float,
    world_size: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure dense bullet-target kernel, suitable for ``torch.compile``."""
    dist_sq = _swept_dist_sq(
        ship_pos, bullet_start_pos, bullet_midpoint_pos, bullet_pos, world_size
    )
    return _damage_from_dist_sq(
        dist_sq,
        ship_attitude,
        ship_alive,
        bullet_vel,
        bullet_active,
        bullet_damage,
        collision_radius,
        bullet_min_damage_frac,
    )


def _combat_damage_and_proximity_tensors(
    ship_pos: torch.Tensor,
    ship_attitude: torch.Tensor,
    ship_alive: torch.Tensor,
    bullet_start_pos: torch.Tensor,
    bullet_midpoint_pos: torch.Tensor,
    bullet_pos: torch.Tensor,
    bullet_vel: torch.Tensor,
    bullet_active: torch.Tensor,
    bullet_damage: float,
    collision_radius: float,
    bullet_min_damage_frac: float,
    world_size: tuple[float, float],
    ship_team_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`_combat_damage_tensors` plus every slot's nearest enemy and ally.

    The shot labels' encounters come from the very distances that decide
    gameplay hits, computed once. Returns the three gameplay outputs, then
    ``(B, N, K)`` least squared swept distances to a living enemy and to a
    living ally of each slot's shooter, regardless of whether the slot is
    active (``NO_ENCOUNTER_DSQ`` when there is none).
    """
    batch_size, num_ships, num_bullets = bullet_active.shape
    dist_sq = _swept_dist_sq(
        ship_pos, bullet_start_pos, bullet_midpoint_pos, bullet_pos, world_size
    )
    gameplay = _damage_from_dist_sq(
        dist_sq,
        ship_attitude,
        ship_alive,
        bullet_vel,
        bullet_active,
        bullet_damage,
        collision_radius,
        bullet_min_damage_frac,
    )
    shooter_team = ship_team_id.repeat_interleave(num_bullets, dim=1)  # (B, N*K)
    same_team = shooter_team.unsqueeze(2) == ship_team_id.unsqueeze(1)  # (B, N*K, N)
    target = ship_alive.unsqueeze(1) & _not_own_bullet(
        num_ships, num_bullets, dist_sq.device
    ).unsqueeze(0)
    far = torch.full_like(dist_sq, shot_labels.NO_ENCOUNTER_DSQ)
    enemy = torch.where(target & ~same_team, dist_sq, far).amin(dim=2)
    ally = torch.where(target & same_team, dist_sq, far).amin(dim=2)
    shape = (batch_size, num_ships, num_bullets)
    return (*gameplay, enemy.view(shape), ally.view(shape))


def _apply_combat_damage(
    state: TensorState,
    config: ShipConfig,
    combat_damage_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    trajectory: BulletTrajectory | None = None,
    frontline=None,
    combat_proximity_fn: Callable[..., tuple[torch.Tensor, ...]] | None = None,
) -> TensorState:
    """Apply vectorized bullet damage and attribution to mutable state.

    Dense over all bullet slots (active and inactive): inactive slots are
    masked out of the hit test rather than compacted with nonzero/gather.
    This keeps every tensor shape static and avoids the host-device syncs
    that dynamic-shape indexing forces on the per-step hot path.

    Also fills state.damage_matrix (B, N_shooter, N_target) for this step and
    accumulates into state.cumulative_damage_matrix for episode-level attribution.

    With shot labels on (a non-empty ``bullet_shot_record``) the same distance
    pass also feeds every flying slot's encounter record.
    """
    num_bullets = state.max_bullets

    # Reset per-step attribution; cumulative is carried forward across steps.
    state.damage_matrix.zero_()
    state.ship_combat_damage.zero_()
    state.ship_combat_death.zero_()

    if num_bullets == 0:
        return state

    if trajectory is None:
        trajectory = BulletTrajectory(state.bullet_pos, state.bullet_pos)

    kernel_args = (
        state.ship_pos,
        state.ship_attitude,
        state.ship_alive,
        trajectory.start,
        trajectory.midpoint,
        state.bullet_pos,
        state.bullet_vel,
        state.bullet_active,
        config.bullet_damage,
        config.collision_radius,
        config.bullet_min_damage_frac,
        config.world_size,
    )
    if state.bullet_shot_record.shape[-1] > 0:
        proximity_fn = combat_proximity_fn or _combat_damage_and_proximity_tensors
        total_damage, per_shooter, next_bullet_active, enemy_dsq, ally_dsq = proximity_fn(
            *kernel_args, state.ship_team_id
        )
        shot_labels.record_encounters(state, enemy_dsq, ally_dsq, config)
    else:
        damage_fn = combat_damage_fn or _combat_damage_tensors
        total_damage, per_shooter, next_bullet_active = damage_fn(*kernel_args)
    if frontline is not None:
        from boost_and_broadside.env.frontline import friendly_spawn_mask

        protected = friendly_spawn_mask(state, config)
        per_shooter = per_shooter * ~protected[:, None, :]
        total_damage = per_shooter.sum(1)
    state.damage_matrix.copy_(per_shooter)

    # Record only applied health loss: simultaneous hits and overkill cannot
    # inflate source-specific damage rewards.
    alive_before = state.ship_alive
    health_before = state.ship_health
    health_after = (health_before - total_damage).clamp(min=0.0)
    applied = health_before - health_after  # (B, N)
    state.ship_combat_damage.copy_(applied)
    # Simultaneous hits share what was actually applied in proportion to impact.
    applied_fraction = applied / total_damage.clamp(min=EPS)  # (B, N_target)
    state.cumulative_damage_matrix += per_shooter * applied_fraction.unsqueeze(1)
    if frontline is None:
        died = health_after <= 0.0
    else:
        enemy = state.ship_team_id[:, :, None] != state.ship_team_id[:, None, :]
        enemy_hit = (per_shooter * enemy).sum(1) > 0
        died = (health_before <= 0.0) & enemy_hit
    state.ship_combat_death.copy_(alive_before & died)
    state.ship_health = health_after
    state.ship_alive = alive_before & ~state.ship_combat_death

    state.bullet_active = next_bullet_active

    return state


def _check_game_over(state: TensorState) -> torch.Tensor:
    """Return (B,) done mask — True when a team that exists is fully eliminated."""
    team0_alive = ((state.ship_team_id == 0) & state.ship_alive).sum(dim=1)  # (B,)
    team1_alive = ((state.ship_team_id == 1) & state.ship_alive).sum(dim=1)  # (B,)
    team0_exists = (state.ship_team_id == 0).any(dim=1)  # (B,)
    team1_exists = (state.ship_team_id == 1).any(dim=1)  # (B,)
    return (team0_exists & (team0_alive == 0)) | (team1_exists & (team1_alive == 0))
