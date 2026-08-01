"""GPU-vectorized ship and bullet physics.

All functions operate on TensorState in-place and return the mutated state.
No Python loops over batch or ship dimensions.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import EPS, ShootActions
from boost_and_broadside.env.field_physics import evaluate_fields
from boost_and_broadside.env.state import TensorState

# ---------------------------------------------------------------------------
# Lookup table construction
# ---------------------------------------------------------------------------

# Keyed by (config, device string). Building the tables allocates four tensors
# from Python lists (host→device copies), so they must not be rebuilt on the
# per-step hot path.
_LOOKUP_TABLE_CACHE: dict = {}


def _get_lookup_tables(
    config: ShipConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cached per-action physics lookup tensors for (config, device)."""
    key = (config, str(device))
    tables = _LOOKUP_TABLE_CACHE.get(key)
    if tables is None:
        tables = _build_lookup_tables(config, device)
        _LOOKUP_TABLE_CACHE[key] = tables
    return tables


def _build_lookup_tables(
    config: ShipConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-action physics lookup tensors from config.

    Returns:
        (thrust, turn_offset, drag_coeff, lift_coeff) — all float32.
    """
    thrust_table = torch.tensor(
        [config.base_thrust, config.boost_thrust, config.reverse_thrust],
        device=device,
        dtype=torch.float32,
    )
    turn_offset_table = torch.tensor(
        [
            0.0,
            -config.normal_turn_angle,
            config.normal_turn_angle,
            -config.sharp_turn_angle,
            config.sharp_turn_angle,
            0.0,
            0.0,
        ],
        device=device,
        dtype=torch.float32,
    )
    drag_coeff_table = torch.tensor(
        [
            config.no_turn_drag_coeff,
            config.normal_turn_drag_coeff,
            config.normal_turn_drag_coeff,
            config.sharp_turn_drag_coeff,
            config.sharp_turn_drag_coeff,
            config.normal_turn_drag_coeff,
            config.sharp_turn_drag_coeff,
        ],
        device=device,
        dtype=torch.float32,
    )
    lift_coeff_table = torch.tensor(
        [
            0.0,
            -config.normal_turn_lift_coeff,
            config.normal_turn_lift_coeff,
            -config.sharp_turn_lift_coeff,
            config.sharp_turn_lift_coeff,
            0.0,
            0.0,
        ],
        device=device,
        dtype=torch.float32,
    )
    return (
        thrust_table,
        turn_offset_table,
        drag_coeff_table,
        lift_coeff_table,
    )


# ---------------------------------------------------------------------------
# Kinematics update
# ---------------------------------------------------------------------------


def _update_kinematics(
    state: TensorState,
    actions: torch.Tensor,
    config: ShipConfig,
    tables: tuple[torch.Tensor, ...],
) -> TensorState:
    """Update power, attitude, velocity, and position for all ships.

    GPU kernel: kept together for performance — splitting would force extra tensor
    allocations and destroy cache locality.
    """
    device = state.device
    (
        thrust_table,
        turn_offset_table,
        drag_coeff_table,
        lift_coeff_table,
    ) = tables

    power_action = actions[..., 0].long()  # (B, N)
    turn_action = actions[..., 1].long()  # (B, N)

    thrust_mag = thrust_table[power_action]  # (B, N)
    turn_offset = turn_offset_table[turn_action]  # (B, N)
    drag_coeff = drag_coeff_table[turn_action]  # (B, N)
    lift_coeff = lift_coeff_table[turn_action]  # (B, N)

    # Speed from current velocity — used for power drain and forces below
    speed = state.ship_vel.abs()  # (B, N)

    # Gates both stall (turn/lift zeroing) and attitude-hold below: a ship this
    # slow has neither turning authority nor a well-defined velocity direction.
    below_min_speed = speed < config.min_speed

    # Stall: below min_speed, lose turning authority and reverse thrust
    turn_offset = torch.where(below_min_speed, torch.zeros_like(turn_offset), turn_offset)
    lift_coeff = torch.where(below_min_speed, torch.zeros_like(lift_coeff), lift_coeff)

    # Ships with no power can't thrust
    thrust_mag = thrust_mag * (state.ship_power > 0).float()

    # Power exchange: forward thrust drains, reverse thrust gains (equal and opposite).
    # Passive regen added on top regardless of action.
    power_delta = (
        -(thrust_mag / config.power_speed_constant) * speed + config.passive_power_gain
    ) * config.dt
    state.ship_power = torch.clamp(state.ship_power + power_delta, 0.0, config.max_power)

    # Attitude — align with velocity direction then apply turn rotation
    speed_safe = torch.clamp(speed, min=EPS)
    vel_dir = state.ship_vel / speed_safe
    base_att = torch.where(below_min_speed, state.ship_attitude, vel_dir)

    rotation = torch.polar(torch.ones_like(turn_offset), turn_offset)  # (B, N)
    state.ship_attitude = base_att * rotation
    state.ship_ang_vel = turn_offset / config.dt

    # Forces
    thrust_force = thrust_mag * state.ship_attitude  # (B, N) complex
    drag_force = -drag_coeff * speed * state.ship_vel  # (B, N) complex
    lift_force = lift_coeff * speed * (state.ship_vel * 1j)  # (B, N) complex  — perpendicular

    # Pairwise gravity (attracts fast ships toward each other)
    _, num_ships = state.ship_pos.shape
    world_w, world_h = config.world_size

    if config.gravity_factor == 0.0:
        gravity = torch.zeros_like(thrust_force)
    else:
        # (B, N_i, N_j) complex — wrapped difference i→j
        diff = state.ship_pos.unsqueeze(1) - state.ship_pos.unsqueeze(2)
        diff.real = (diff.real + world_w / 2) % world_w - world_w / 2
        diff.imag = (diff.imag + world_h / 2) % world_h - world_h / 2

        dist_sq = diff.real**2 + diff.imag**2  # (B, N, N)
        dist = torch.sqrt(dist_sq)

        def _symlog(x: torch.Tensor) -> torch.Tensor:
            return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

        speed_i = speed.unsqueeze(2)  # (B, N, 1)
        speed_j = speed.unsqueeze(1)  # (B, 1, N)
        force_mag = (
            config.gravity_factor
            * config.gravity_eps
            * _symlog(speed_i * speed_j)
            / (dist_sq + config.gravity_eps)
        )  # (B, N, N)
        force_dir = diff / torch.clamp(dist, min=EPS)  # (B, N, N)
        force_vec = force_mag * force_dir  # (B, N, N) complex

        alive_mask = state.ship_alive.unsqueeze(2) & state.ship_alive.unsqueeze(1)  # (B, N, N)
        self_mask = torch.eye(num_ships, device=device, dtype=torch.bool).unsqueeze(0)
        force_vec = torch.where(alive_mask & ~self_mask, force_vec, torch.zeros_like(force_vec))
        gravity = force_vec.sum(dim=2)  # (B, N) complex

    # Integrate
    total_force = thrust_force + drag_force + lift_force + gravity
    state.ship_vel = state.ship_vel + total_force * config.dt
    state.ship_pos = state.ship_pos + state.ship_vel * config.dt

    # Toroidal wrap
    state.ship_pos.real = state.ship_pos.real % world_w
    state.ship_pos.imag = state.ship_pos.imag % world_h

    # Prevent exactly-zero velocity (would break direction computations)
    new_speed = state.ship_vel.abs()
    too_slow = new_speed < EPS
    min_vel = EPS * state.ship_attitude
    state.ship_vel = torch.where(too_slow, min_vel, state.ship_vel)

    return state


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


def _apply_field_flight_half_step(
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
    dragged_speed = speed / (1.0 + drag_coeff * speed * duration)

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
    """Midpoint passive optical transport with generalized-energy projection."""

    world_w, world_h = config.world_size
    step_dt = config.dt / config.field_integration_substeps
    alpha = state.ship_field_alpha
    index = state.ship_local_index
    grad_index = state.ship_field_gradient
    total_damage = torch.zeros_like(state.ship_health)

    for _ in range(config.field_integration_substeps):
        velocity = state.ship_vel
        acceleration = _field_optical_acceleration(velocity, index, grad_index)
        midpoint_velocity = velocity + 0.5 * acceleration * step_dt
        midpoint_pos = state.ship_pos + 0.5 * velocity * step_dt + 0.125 * acceleration * step_dt**2
        midpoint_pos = torch.complex(midpoint_pos.real % world_w, midpoint_pos.imag % world_h)
        midpoint = evaluate_fields(
            midpoint_pos,
            state.field_pos,
            state.field_radius,
            state.field_transition_width,
            state.field_delta_index,
            config.world_size,
        )
        midpoint_acceleration = _field_optical_acceleration(
            midpoint_velocity, midpoint.index, midpoint.grad_index
        )
        direction_velocity = velocity + midpoint_acceleration * step_dt
        next_pos = state.ship_pos + 0.5 * (velocity + direction_velocity) * step_dt
        next_pos = torch.complex(next_pos.real % world_w, next_pos.imag % world_h)
        next_eval = evaluate_fields(
            next_pos,
            state.field_pos,
            state.field_radius,
            state.field_transition_width,
            state.field_delta_index,
            config.world_size,
        )

        # Project only the passive substep: n|v| is held exactly while the
        # midpoint force determines direction. Powered work is applied outside
        # this function and is never erased by the projection.
        proper_speed = index * velocity.abs()
        direction = direction_velocity / direction_velocity.abs().clamp(min=EPS)
        direction = torch.where(
            direction_velocity.abs() > EPS,
            direction,
            state.ship_attitude,
        )
        state.ship_vel = direction * (proper_speed / next_eval.index)
        state.ship_pos = next_pos

        variation = (midpoint.alpha - alpha).abs() + (next_eval.alpha - midpoint.alpha).abs()
        total_damage = total_damage + (variation * state.field_damage.unsqueeze(1)).sum(dim=2)
        alpha = next_eval.alpha
        index = next_eval.index
        grad_index = next_eval.grad_index

    alive_before = state.ship_alive
    state.ship_field_damage = total_damage * alive_before.float()
    state.ship_health = (state.ship_health - state.ship_field_damage).clamp(min=0.0)
    state.ship_alive = state.ship_alive & (state.ship_health > 0.0)
    state.ship_field_alpha = alpha
    state.ship_local_index = index
    state.ship_field_gradient = grad_index
    return state


def _update_kinematics_in_fields(
    state: TensorState,
    actions: torch.Tensor,
    config: ShipConfig,
    tables: tuple[torch.Tensor, ...],
) -> TensorState:
    """Split flight/control around passive effective-mass field transport."""

    thrust_table, turn_offset_table, drag_coeff_table, lift_coeff_table = tables
    power_action = actions[..., 0].long()
    turn_action = actions[..., 1].long()
    thrust_mag = thrust_table[power_action]
    turn_offset = turn_offset_table[turn_action]
    drag_coeff = drag_coeff_table[turn_action]
    lift_coeff = lift_coeff_table[turn_action]

    proper_speed = state.ship_local_index * state.ship_vel.abs()
    below_min_speed = proper_speed < config.min_speed
    turn_offset = torch.where(below_min_speed, torch.zeros_like(turn_offset), turn_offset)
    lift_coeff = torch.where(below_min_speed, torch.zeros_like(lift_coeff), lift_coeff)
    # Reverse is a kinetic-energy recovery action and has no useful effect at a
    # stall; forward thrust remains able to restart the ship.
    thrust_mag = torch.where(
        below_min_speed & (thrust_mag < 0.0), torch.zeros_like(thrust_mag), thrust_mag
    )

    speed = state.ship_vel.abs()
    velocity_direction = state.ship_vel / speed.clamp(min=EPS)
    base_attitude = torch.where(below_min_speed, state.ship_attitude, velocity_direction)
    state.ship_attitude = base_attitude * torch.polar(torch.ones_like(turn_offset), turn_offset)
    state.ship_ang_vel = turn_offset / config.dt

    state.ship_field_damage = torch.zeros_like(state.ship_field_damage)
    state = _apply_field_flight_half_step(state, thrust_mag, drag_coeff, lift_coeff, config)
    state = _transport_through_fields(state, config)
    state = _apply_field_flight_half_step(state, thrust_mag, drag_coeff, lift_coeff, config)
    state.ship_power = torch.clamp(
        state.ship_power + config.passive_power_gain * config.dt,
        0.0,
        config.max_power,
    )

    speed = state.ship_vel.abs()
    state.ship_vel = torch.where(speed < EPS, EPS * state.ship_attitude, state.ship_vel)
    return state


# ---------------------------------------------------------------------------
# Shooting
# ---------------------------------------------------------------------------


def _handle_shooting(
    state: TensorState, shoot_action: torch.Tensor, config: ShipConfig
) -> TensorState:
    """Manage cooldowns and spawn bullets for ships that fire.

    Fully branchless: bullet spawns are written through a one-hot mask on the
    ring-buffer cursor instead of dynamic indexing, so no host-device sync occurs
    (this runs every step of every rollout).
    """
    if state.max_bullets == 0:
        state.ship_is_shooting = torch.zeros_like(state.ship_is_shooting)
        return state

    state.ship_cooldown = (state.ship_cooldown - config.dt).clamp(min=0.0)

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
    slot_onehot = F.one_hot(state.bullet_cursor, K).bool() & can_shoot.unsqueeze(-1)  # (B, N, K)

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
    state.bullet_active = state.bullet_active | slot_onehot
    state.bullet_remaining_damage = torch.where(
        slot_onehot,
        config.bullet_damage,
        state.bullet_remaining_damage,
    )
    state.bullet_field_alpha = torch.where(
        slot_onehot.unsqueeze(-1),
        state.ship_field_alpha.unsqueeze(2),
        state.bullet_field_alpha,
    )
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
    state.bullet_cursor = torch.where(can_shoot, (state.bullet_cursor + 1) % K, state.bullet_cursor)

    return state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def update_ships(state: TensorState, actions: torch.Tensor, config: ShipConfig) -> TensorState:
    """Apply one physics timestep: kinematics + shooting.

    Args:
        state: Current environment state (mutated in-place).
        actions: (B, N, 3) int tensor — [power_action, turn_action, shoot_action].
        config: Physics configuration.

    Returns:
        The mutated state.
    """
    tables = _get_lookup_tables(config, state.device)
    if state.num_fields == 0:
        # Preserve the exact ambient-only baseline and its cheap hot path.
        state = _update_kinematics(state, actions, config, tables)
    else:
        state = _update_kinematics_in_fields(state, actions, config, tables)
    state = _handle_shooting(state, actions[..., 2].long(), config)
    return state


def _wrap_positions(
    position: torch.Tensor,
    world_size: tuple[float, float],
) -> torch.Tensor:
    """Wrap complex positions onto the toroidal world."""
    world_w, world_h = world_size
    return torch.complex(position.real % world_w, position.imag % world_h)


def _transport_bullets_through_fields(
    state: TensorState,
    transport_vel: torch.Tensor,
    config: ShipConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuously refract bullets and return final velocity/position and half-tick position."""
    batch_size, num_ships, num_bullets = state.bullet_pos.shape
    num_flat_bullets = num_ships * num_bullets
    step_dt = config.dt / config.bullet_field_integration_substeps
    position = state.bullet_pos.view(batch_size, num_flat_bullets)
    velocity = transport_vel.view(batch_size, num_flat_bullets)
    alpha = state.bullet_field_alpha.view(batch_size, num_flat_bullets, state.num_fields)
    index = state.bullet_local_index.view(batch_size, num_flat_bullets)
    grad_index = state.bullet_field_gradient.view(batch_size, num_flat_bullets)
    half_tick_position = position
    total_variation = None

    for substep in range(config.bullet_field_integration_substeps):
        acceleration = _field_optical_acceleration(velocity, index, grad_index)
        if config.bullet_field_integrator == "midpoint":
            midpoint_velocity = velocity + 0.5 * acceleration * step_dt
            integration_midpoint = _wrap_positions(
                position + 0.5 * velocity * step_dt + 0.125 * acceleration * step_dt**2,
                config.world_size,
            )
            midpoint_eval = evaluate_fields(
                integration_midpoint,
                state.field_pos,
                state.field_radius,
                state.field_transition_width,
                state.field_delta_index,
                config.world_size,
            )
            midpoint_acceleration = _field_optical_acceleration(
                midpoint_velocity,
                midpoint_eval.index,
                midpoint_eval.grad_index,
            )
            direction_velocity = velocity + midpoint_acceleration * step_dt
            next_position = _wrap_positions(
                position + 0.5 * (velocity + direction_velocity) * step_dt,
                config.world_size,
            )
            midpoint_alpha = midpoint_eval.alpha
        else:
            direction_velocity = velocity + acceleration * step_dt
            next_position = _wrap_positions(
                position + direction_velocity * step_dt,
                config.world_size,
            )

        next_eval = evaluate_fields(
            next_position,
            state.field_pos,
            state.field_radius,
            state.field_transition_width,
            state.field_delta_index,
            config.world_size,
        )
        proper_speed = index * velocity.abs()
        direction = direction_velocity / direction_velocity.abs().clamp(min=EPS)
        velocity = direction * (proper_speed / next_eval.index)
        position = next_position
        if config.bullet_field_damage_scale != 0.0:
            if config.bullet_field_integrator == "midpoint":
                variation = (midpoint_alpha - alpha).abs() + (
                    next_eval.alpha - midpoint_alpha
                ).abs()
            else:
                variation = (next_eval.alpha - alpha).abs()
            total_variation = variation if total_variation is None else total_variation + variation
        alpha = next_eval.alpha
        index = next_eval.index
        grad_index = next_eval.grad_index

        if substep + 1 == config.bullet_field_integration_substeps // 2:
            half_tick_position = position

    state.bullet_field_alpha = alpha.view(
        batch_size,
        num_ships,
        num_bullets,
        state.num_fields,
    )
    state.bullet_local_index = index.view(batch_size, num_ships, num_bullets)
    state.bullet_field_gradient = grad_index.view(batch_size, num_ships, num_bullets)
    if config.bullet_field_damage_scale != 0.0:
        damage_loss = (total_variation * state.field_damage.unsqueeze(1)).sum(dim=2)
        remaining_damage = state.bullet_remaining_damage.view(batch_size, num_flat_bullets)
        active = state.bullet_active.view(batch_size, num_flat_bullets)
        next_remaining_damage = (
            remaining_damage - config.bullet_field_damage_scale * damage_loss
        ).clamp(min=0.0)
        depleted = (remaining_damage > 0.0) & (next_remaining_damage <= 0.0)
        state.bullet_remaining_damage = torch.where(
            active,
            next_remaining_damage,
            remaining_damage,
        ).view(batch_size, num_ships, num_bullets)
        state.bullet_active = (active & ~depleted).view(batch_size, num_ships, num_bullets)
    return (
        velocity.view(batch_size, num_ships, num_bullets),
        position.view(batch_size, num_ships, num_bullets),
        half_tick_position.view(batch_size, num_ships, num_bullets),
    )


def advance_bullets(
    state: TensorState,
    config: ShipConfig,
) -> tuple[TensorState, torch.Tensor, torch.Tensor]:
    """Advance bullets and return the start and midpoint of each trajectory.

    Args:
        state: Current state (mutated in-place).
        config: Physics configuration.

    Returns:
        ``(state, start_pos, midpoint_pos)``. Together with the final
        ``state.bullet_pos``, these define two swept collision segments without
        retaining another full bullet tensor in persistent environment state.
    """
    start_pos = state.bullet_pos
    state.bullet_time = state.bullet_time - config.dt
    state.bullet_active = state.bullet_active & (state.bullet_time > 0)
    transport_vel = state.bullet_vel
    if config.bullet_drag_coeff != 0.0:
        speed = transport_vel.abs()
        half_drag_scale = 1.0 / (1.0 + config.bullet_drag_coeff * speed * (0.5 * config.dt))
        transport_vel = transport_vel * half_drag_scale
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
        speed = transport_vel.abs()
        half_drag_scale = 1.0 / (1.0 + config.bullet_drag_coeff * speed * (0.5 * config.dt))
        transport_vel = transport_vel * half_drag_scale
    state.bullet_vel = transport_vel
    return state, start_pos, midpoint_pos


def update_bullets(state: TensorState, config: ShipConfig) -> TensorState:
    """Compatibility wrapper that advances bullets without exposing segments."""
    state, _, _ = advance_bullets(state, config)
    return state


def resolve_collisions(
    state: TensorState,
    config: ShipConfig,
    combat_damage_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    bullet_start_pos: torch.Tensor | None = None,
    bullet_midpoint_pos: torch.Tensor | None = None,
) -> tuple[TensorState, torch.Tensor]:
    """Detect bullet-ship collisions, apply damage, and check game-over.

    Args:
        state: Current state (mutated in-place).
        config: Physics configuration.

    Returns:
        (state, dones) where dones is a (B,) bool tensor.
    """
    state = _apply_combat_damage(
        state,
        config,
        combat_damage_fn,
        bullet_start_pos,
        bullet_midpoint_pos,
    )
    dones = _check_game_over(state)
    return state, dones


def _combat_damage_tensors(
    ship_pos: torch.Tensor,
    ship_attitude: torch.Tensor,
    ship_alive: torch.Tensor,
    bullet_start_pos: torch.Tensor,
    bullet_midpoint_pos: torch.Tensor,
    bullet_pos: torch.Tensor,
    bullet_vel: torch.Tensor,
    bullet_active: torch.Tensor,
    bullet_remaining_damage: torch.Tensor,
    collision_radius: float,
    bullet_min_damage_frac: float,
    world_size: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure dense bullet-target kernel, suitable for ``torch.compile``."""

    batch_size, num_ships = ship_pos.shape
    num_bullets = bullet_pos.shape[2]
    device = ship_pos.device
    world_w, world_h = world_size
    num_flat_bullets = num_ships * num_bullets

    flat_bullet_active = bullet_active.view(batch_size, num_flat_bullets)
    flat_bullet_start = bullet_start_pos.view(batch_size, num_flat_bullets)
    flat_bullet_midpoint = bullet_midpoint_pos.view(batch_size, num_flat_bullets)
    flat_bullet_pos = bullet_pos.view(batch_size, num_flat_bullets)
    flat_bullet_vel = bullet_vel.view(batch_size, num_flat_bullets)
    flat_bullet_damage = bullet_remaining_damage.view(batch_size, num_flat_bullets)

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
    dist_sq = torch.minimum(first_dist_sq, second_dist_sq)

    owner_idx = torch.arange(num_flat_bullets, device=device) // num_bullets
    target_idx = torch.arange(num_ships, device=device)
    not_own_bullet = owner_idx.unsqueeze(1) != target_idx.unsqueeze(0)
    valid_hit = (
        (dist_sq < collision_radius**2)
        & ship_alive.unsqueeze(1)
        & flat_bullet_active.unsqueeze(2)
        & not_own_bullet.unsqueeze(0)
    )

    if bullet_min_damage_frac == 1.0:
        damage_per_hit = valid_hit.float() * flat_bullet_damage.unsqueeze(2)
    else:
        hit_angles = torch.angle(
            -flat_bullet_vel.unsqueeze(2) * torch.conj(ship_attitude.unsqueeze(1))
        )
        damage_scale = 1.0 - (1.0 - bullet_min_damage_frac) * torch.exp(
            -(hit_angles**2) * 4.0 / torch.pi
        )
        damage_per_hit = damage_scale * valid_hit.float() * flat_bullet_damage.unsqueeze(2)

    total_damage = damage_per_hit.sum(dim=1)
    per_shooter = damage_per_hit.view(batch_size, num_ships, num_bullets, num_ships).sum(dim=2)
    hit_any_ship = valid_hit.any(dim=2)
    next_bullet_active = (flat_bullet_active & ~hit_any_ship).view(
        batch_size, num_ships, num_bullets
    )
    return total_damage, per_shooter, next_bullet_active


def _apply_combat_damage(
    state: TensorState,
    config: ShipConfig,
    combat_damage_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    bullet_start_pos: torch.Tensor | None = None,
    bullet_midpoint_pos: torch.Tensor | None = None,
) -> TensorState:
    """Apply vectorized bullet damage and attribution to mutable state.

    Dense over all bullet slots (active and inactive): inactive slots are
    masked out of the hit test rather than compacted with nonzero/gather.
    This keeps every tensor shape static and avoids the host-device syncs
    that dynamic-shape indexing forces on the per-step hot path.

    Also fills state.damage_matrix (B, N_shooter, N_target) for this step and
    accumulates into state.cumulative_damage_matrix for episode-level attribution.
    """
    num_bullets = state.max_bullets

    # Reset per-step attribution; cumulative is carried forward across steps.
    state.damage_matrix.zero_()

    if num_bullets == 0:
        return state

    if bullet_start_pos is None:
        bullet_start_pos = state.bullet_pos
    if bullet_midpoint_pos is None:
        bullet_midpoint_pos = state.bullet_pos

    damage_fn = combat_damage_fn or _combat_damage_tensors
    total_damage, per_shooter, next_bullet_active = damage_fn(
        state.ship_pos,
        state.ship_attitude,
        state.ship_alive,
        bullet_start_pos,
        bullet_midpoint_pos,
        state.bullet_pos,
        state.bullet_vel,
        state.bullet_active,
        state.bullet_remaining_damage,
        config.collision_radius,
        config.bullet_min_damage_frac,
        config.world_size,
    )
    state.damage_matrix.copy_(per_shooter)
    state.cumulative_damage_matrix += per_shooter

    # Apply health reduction. Death is monotone (alive &= health > 0) so ships
    # flagged dead by other systems are never resurrected by the alive refresh.
    state.ship_health = state.ship_health - total_damage
    state.ship_alive = state.ship_alive & (state.ship_health > 0)
    state.ship_health = torch.clamp(state.ship_health, min=0.0)

    state.bullet_active = next_bullet_active

    return state


def _check_game_over(state: TensorState) -> torch.Tensor:
    """Return (B,) done mask — True when a team that exists is fully eliminated."""
    team0_alive = ((state.ship_team_id == 0) & state.ship_alive).sum(dim=1)  # (B,)
    team1_alive = ((state.ship_team_id == 1) & state.ship_alive).sum(dim=1)  # (B,)
    team0_exists = (state.ship_team_id == 0).any(dim=1)  # (B,)
    team1_exists = (state.ship_team_id == 1).any(dim=1)  # (B,)
    return (team0_exists & (team0_alive == 0)) | (team1_exists & (team1_alive == 0))
