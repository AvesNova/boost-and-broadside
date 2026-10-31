"""Ship flight (SPEC 3): the flight setting, the kick, and one kick-drift-kick tick.

A ship's flight variables are the canonical keys x y c s w alpha ca sa r E logn gx gy.
`stick` and `power` here are the stick level sigma and power mode pi as floats (not the
action indices); `actions_to_controls` converts. `F` holds each ship's own constants
(`ship_types.flight_of`): a constant that differs by type is a per-ship tensor.
"""

from typing import NamedTuple

import torch

from .constants import POWER_MODES, STICK_LEVELS, Flight
from .medium import medium
from .numerics import cs, rotate, smoothstep, softmin

FLIGHT_KEYS = ("x", "y", "c", "s", "w", "alpha", "ca", "sa", "r", "E", "logn", "gx", "gy")


def actions_to_controls(power_index: torch.Tensor, stick_index: torch.Tensor):
    """Action indices to (stick level sigma, power mode pi), SPEC 3.2."""
    sticks = torch.tensor(STICK_LEVELS, dtype=torch.float32, device=stick_index.device)
    powers = torch.tensor(POWER_MODES, dtype=torch.float32, device=power_index.device)
    return sticks[stick_index.long()], powers[power_index.long()]


def engine_fall(F: Flight, E: torch.Tensor) -> torch.Tensor:
    """phi(E): 0 below the knee, rising smoothly to 1 at a full store."""
    return smoothstep(((E / F.e_max - F.e_knee) / (1.0 - F.e_knee)).clamp(0.0, 1.0))


def lift_coefficient(F: Flight, alpha: torch.Tensor) -> torch.Tensor:
    x = (alpha / F.alpha_s).clamp(-2.0, 2.0)
    return F.cl_max * (2.0 * x - x * x.abs())


def lift_slope(F: Flight, alpha: torch.Tensor) -> torch.Tensor:
    x = (alpha / F.alpha_s).abs()
    return torch.where(x < 2.0, (F.cl_max / F.alpha_s) * (2.0 - 2.0 * x), 0.0)


class Setting(NamedTuple):
    """SPEC 3.4: what the controls ask of the airframe, held fixed over a half kick."""

    lateral: torch.Tensor  # l, lateral lift (m/s^2)
    exchange: torch.Tensor  # a, speed-for-store exchange (m/s^2)
    vertical: torch.Tensor  # z, vertical lift (m/s^2)
    thrust: torch.Tensor  # T (m/s^2)


def setting(F: Flight, u, E, stick, power) -> Setting:
    fall = engine_fall(F, E)
    commanded = torch.where(power == 0.0, F.a0 * (1.0 - fall), torch.where(power > 0.0, F.a_climb, -F.a_dive))
    lift_reach = F.cl_max * u * u
    budget = softmin(lift_reach, F.a_max, F.soft_p)  # L
    budget_safe = budget.clamp_min(1e-12)
    lateral_cmd = torch.sign(stick) * softmin(stick.abs() * lift_reach, F.a_max, F.soft_p)
    need = torch.sqrt(lateral_cmd * lateral_cmd + F.g * F.g)  # N
    flown = softmin(need, budget_safe, F.soft_p) / need.clamp_min(1e-12)  # q
    lateral = lateral_cmd * flown
    chi = (lateral.abs() / budget_safe).clamp(0.0, 1.0)
    exchange = commanded * torch.sqrt(1.0 - chi * chi) * flown - (1.0 - flown) * F.a_dive
    # Fade the exchange out at the ends of the store: charging stops at E_max, spending at 0.
    room = torch.where(exchange > 0.0, (F.e_max - E) / (F.band * F.e_max), E / (F.band * F.e_max))
    exchange = exchange * room.clamp(0.0, 1.0)
    engine = F.power - (F.power - F.p_min) * fall
    thrust = softmin(engine / u, F.f_static, 4.0) / F.mass
    return Setting(lateral, exchange, F.g * flown, thrust)


def rates(F: Flight, st: Setting, n, speed, lift, cos_alpha, inv_u2):
    """(du/dt, dE/dt) for a speed, lift coefficient, cos(alpha) and 1/u^2 (SPEC 3.4)."""
    induced = st.vertical * inv_u2
    drag = (F.cd0 + F.k_ind * (lift * lift + induced * induced)) * speed * speed
    return (st.thrust * cos_alpha - drag - st.exchange) / n, F.mass * speed * st.exchange / n


def expm2(a11, a12, a21, a22, dt: float):
    """The normative 2x2 matrix exponential of A dt: Horner series of exp(A dt/2), squared."""
    b11, b12, b21, b22 = (v * (dt / 2.0) for v in (a11, a12, a21, a22))
    h11, h12, h21, h22 = 1.0, 0.0, 0.0, 1.0  # innermost first: I + B/4, then I + (B/k) (...)
    for k in (4.0, 3.0, 2.0, 1.0):
        h11, h12, h21, h22 = (1.0 + (b11 * h11 + b12 * h21) / k, (b11 * h12 + b12 * h22) / k,
                              (b21 * h11 + b22 * h21) / k, 1.0 + (b21 * h12 + b22 * h22) / k)
    return (h11 * h11 + h12 * h21, h11 * h12 + h12 * h22, h21 * h11 + h22 * h21, h21 * h12 + h22 * h22)


def kick(F: Flight, ship: dict, n, gamma_x, gamma_y, stick, power, dt: float) -> dict:
    """Advance w, alpha, ca, sa, r, E by dt with the pose held (SPEC 3.5, normative).

    gamma_x, gamma_y: grad ln n in the nose frame.
    """
    u0, alpha0, ca0, sa0, E = ship["w"], ship["alpha"], ship["ca"], ship["sa"], ship["E"]
    u0_safe = u0.clamp_min(F.eps)
    first = setting(F, u0_safe, E, stick, power)
    lift0 = lift_coefficient(F, alpha0)
    du0, dE0 = rates(F, first, n, u0, lift0, ca0, 1.0 / (u0_safe * u0_safe))

    u = (u0 + 0.5 * dt * du0).clamp_min(F.eps)  # mid speed
    st = setting(F, u, E + 0.5 * dt * dE0, stick, power)
    inv_u2 = 1.0 / (u * u)
    path_rate = (lift0 * u * u + st.thrust * sa0) / (u * n) + (u / n) * (ca0 * gamma_y + sa0 * gamma_x)
    kappa = (lift_slope(F, alpha0) * u * u + st.thrust * ca0) / (u * n)
    omega = (F.omega_c * u / F.corner_speed).clamp_min(F.omega_floor) / n
    reach = (st.lateral.abs() * inv_u2 / F.cl_max).clamp(0.0, 1.0)
    d_alpha = torch.sign(st.lateral) * F.alpha_s * (1.0 - torch.sqrt(1.0 - reach)) - alpha0

    # The nose oscillator, linearized about the commanded angle of attack and solved exactly.
    m11, m12, m21, m22 = expm2(-kappa, 1.0, -omega * omega + 2.0 * F.zeta * omega * kappa, -2.0 * F.zeta * omega, dt)
    z_s = kappa * d_alpha
    y1 = -d_alpha
    y2 = (ship["r"] - path_rate) - z_s
    e = d_alpha + m11 * y1 + m12 * y2
    r = path_rate + z_s + m21 * y1 + m22 * y2

    ch, sh = cs(0.5 * e)
    ca_mid = ca0 * ch - sa0 * sh
    lift_mid = lift_coefficient(F, alpha0 + 0.5 * e)
    du1, _ = rates(F, st, n, u0, lift_mid, ca_mid, inv_u2)
    du2, dE2 = rates(F, st, n, (u0 + 0.5 * dt * du1).clamp_min(F.eps), lift_mid, ca_mid, inv_u2)

    c1, s1 = cs(e)
    ca, sa = rotate(ca0, sa0, c1, s1)
    return dict(ship, w=(u0 + dt * du2).clamp_min(F.eps), alpha=alpha0 + e, ca=ca, sa=sa, r=r,
                E=(E + dt * dE2).clamp_min(0.0).clamp_max(F.e_max))


def nose_frame(ship: dict):
    """grad ln n rotated into the nose frame: (gamma_x, gamma_y)."""
    c, s, gx, gy = ship["c"], ship["s"], ship["gx"], ship["gy"]
    return c * gx + s * gy, c * gy - s * gx


def flight_tick(F: Flight, h: float, ship: dict, stick, power, fields: dict) -> dict:
    """One tick of flight (SPEC 3.6): kick, drift along the exact nose arc, re-sample the medium, kick."""
    ship = kick(F, ship, torch.exp(ship["logn"]), *nose_frame(ship), stick, power, 0.5 * h)

    n = torch.exp(ship["logn"])
    gamma_x, gamma_y = nose_frame(ship)
    vx, vy = ship["w"] * ship["ca"] / n, -ship["w"] * ship["sa"] / n
    xi = 0.5 * h * (gamma_x * vx + gamma_y * vy)  # correct the speed to the index at mid-step
    stretch = 1.0 + xi * (1.0 + xi * (0.5 + xi / 6.0))
    vx, vy = vx / stretch, vy / stretch
    phi = ship["r"] * h
    p2 = phi * phi
    a1 = (1.0 - p2 * (1.0 / 6.0 - p2 / 120.0)) * h
    a2 = phi * (0.5 - p2 * (1.0 / 24.0 - p2 / 720.0)) * h
    bx, by = vx * a1 - vy * a2, vx * a2 + vy * a1
    c, s = ship["c"], ship["s"]
    moved_x, moved_y = c * bx - s * by, s * bx + c * by
    c, s = rotate(c, s, *cs(phi))
    norm = torch.rsqrt(c * c + s * s)
    anorm = torch.rsqrt(ship["ca"] * ship["ca"] + ship["sa"] * ship["sa"])
    x, y = ship["x"] + moved_x, ship["y"] + moved_y
    logn, gx, gy = medium(fields, x, y)
    ship = dict(ship, x=x, y=y, c=c * norm, s=s * norm, ca=ship["ca"] * anorm, sa=ship["sa"] * anorm,
                logn=logn, gx=gx, gy=gy)

    return kick(F, ship, torch.exp(logn), *nose_frame(ship), stick, power, 0.5 * h)
