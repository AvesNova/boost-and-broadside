"""v5 physics, optimized step: same model as flight5.py, fewer expensive operations.

Changes (no physics change beyond rounding / series truncation):
- reciprocals computed once (1/n, 1/u, 1/u^2); constant divisions folded into multiplies;
- thrust and the lift-vector split evaluated once per half kick at the predicted midpoint
  speed; the soft min as a * rsqrt(sqrt(1 + x^4));
- the 2x2 exponential of the nose oscillator by a degree-4 Taylor series on A h / 2,
  squared once (polynomial only; no cos/cosh/sin/sinh/exp);
- cos and sin of small angles (step rotation r h, half the angle-of-attack change) and the
  arc factors by series; n at the step midpoint by a cubic series of exp.
"""
import jax, jax.numpy as jnp
from flight5 import Ship, STICK, VERT, fields_eval, lift, lift_slope


def softmin4(a, b_inv):
    x = a * b_inv
    x2 = x * x
    return a * jax.lax.rsqrt(jnp.sqrt(1.0 + x2 * x2))


def small_cos_sin(x):
    x2 = x * x
    return 1.0 - x2 * (0.5 - x2 * (1.0 / 24.0 - x2 / 720.0)), x * (1.0 - x2 * (1.0 / 6.0 - x2 * (1.0 / 120.0 - x2 / 5040.0)))


def expm2(a11, a12, a21, a22, h):
    """exp(A h) for a batch of 2x2 matrices: Taylor degree 4 on A h / 2, then squared."""
    t = 0.5 * h
    b11, b12, b21, b22 = a11 * t, a12 * t, a21 * t, a22 * t
    # B^2
    c11, c12, c21, c22 = b11 * b11 + b12 * b21, b11 * b12 + b12 * b22, b21 * b11 + b22 * b21, b21 * b12 + b22 * b22
    # I + B + B^2/2 + B^3/6 + B^4/24 = I + B (I + B/2 (I + B/3 (I + B/4)))  via Horner on matrices
    def mm(x, y):
        return (x[0] * y[0] + x[1] * y[2], x[0] * y[1] + x[1] * y[3], x[2] * y[0] + x[3] * y[2], x[2] * y[1] + x[3] * y[3])
    B = (b11, b12, b21, b22)
    I = (1.0, 0.0, 0.0, 1.0)
    def add_scaled(x, k):  # I + k x
        return (1.0 + k * x[0], k * x[1], k * x[2], 1.0 + k * x[3])
    H = add_scaled(B, 0.25)
    H = add_scaled(mm(B, H), 1.0 / 3.0)
    H = add_scaled(mm(B, H), 0.5)
    H = add_scaled(mm(B, H), 1.0)
    return mm(H, H)


def kick(sh, u0, alpha0, ca0, sa0, r, E, n_inv, gbx, gby, stick, vmode, h):
    inv_band = 1.0 / (sh.band * sh.e_max)
    inv_amax = 1.0 / sh.a_max
    u0s = jnp.maximum(u0, sh.eps)

    def split(u, u_inv):
        L = softmin4(sh.cl_max * u * u, inv_amax)
        lat = stick * L
        vert = jnp.where(vmode == 0.0, sh.g, vmode * L)
        norm = jnp.sqrt(lat * lat + vert * vert)
        scale = softmin4(norm, 1.0 / jnp.maximum(L, 1e-12)) / jnp.maximum(norm, 1e-12)
        thrust = softmin4(sh.p_thrust * u_inv, 1.0 / sh.f_static) / sh.mass
        return lat * scale, vert * scale, thrust

    def rates(uu, ee, cl, caa, vert, thrust, u2_inv):
        cv = vert * u2_inv
        drag = (sh.cd0 + sh.k_ind * (cl * cl + cv * cv)) * uu * uu
        ax = sh.k_x * (sh.g - vert)
        ax = ax * jnp.where(ax > 0, jnp.clip(ee * inv_band, 0.0, 1.0), jnp.clip((sh.e_max - ee) * inv_band, 0.0, 1.0))
        charge = sh.p_store * jnp.clip((sh.e_max - ee) * inv_band, 0.0, 1.0)
        return (thrust * caa + ax - drag) * n_inv, charge - sh.mass * ax * uu * n_inv

    u0_inv = 1.0 / u0s
    _, vert0, thrust0 = split(u0s, u0_inv)
    cl0 = lift(sh, alpha0)
    du0, dE0 = rates(u0, E, cl0, ca0, vert0, thrust0, u0_inv * u0_inv)
    u = jnp.maximum(u0 + 0.5 * h * du0, sh.eps)
    u_inv = 1.0 / u
    lat, vert, t = split(u, u_inv)
    omega_f = u * n_inv * (ca0 * gby + sa0 * gbx)
    theta_dot0 = (cl0 * u * u + t * sa0) * u_inv * n_inv + omega_f
    kappa = (lift_slope(sh, alpha0) * u * u + t * ca0) * u_inv * n_inv
    Om = jnp.maximum(sh.w_c * u * (1.0 / sh.u_corner), sh.w_floor) * n_inv
    c = jnp.clip(jnp.abs(lat) * u_inv * u_inv * (1.0 / sh.cl_max), 0.0, 1.0)
    d_alpha = jnp.sign(lat) * sh.alpha_s * (1.0 - jnp.sqrt(1.0 - c)) - alpha0
    a11, a21, a22 = -kappa, -Om * Om + 2.0 * sh.zeta * Om * kappa, -2.0 * sh.zeta * Om
    e_s, z_s = d_alpha, kappa * d_alpha
    y1, y2 = -e_s, (r - theta_dot0) - z_s
    m11, m12, m21, m22 = expm2(a11, 1.0, a21, a22, h)
    e1 = e_s + m11 * y1 + m12 * y2
    r1 = theta_dot0 + z_s + m21 * y1 + m22 * y2
    ch, sh_ = small_cos_sin(0.5 * e1)
    cam = ca0 * ch - sa0 * sh_
    clm = lift(sh, alpha0 + 0.5 * e1)
    u2i = u_inv * u_inv
    du, dE = rates(u0, E, clm, cam, vert, t, u2i)
    du, dE = rates(jnp.maximum(u0 + 0.5 * h * du, sh.eps), E + 0.5 * h * dE, clm, cam, vert, t, u2i)
    c1, s1 = small_cos_sin(e1)
    return (jnp.maximum(u0 + h * du, sh.eps), alpha0 + e1, ca0 * c1 - sa0 * s1, sa0 * c1 + ca0 * s1,
            r1, jnp.clip(E + h * dE, 0.0, sh.e_max))


def step(sh, field, st, act, h):
    """State: x, y, c, s, u, alpha, cos alpha, sin alpha, r, E, n, gx, gy."""
    x, y, c, s, u, al, ca, sa, r, E, n, gx, gy = st
    stick, vmode = STICK[act[..., 1]].astype(x.dtype), VERT[act[..., 0]].astype(x.dtype)
    n_inv = 1.0 / n
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    u, al, ca, sa, r, E = kick(sh, u, al, ca, sa, r, E, n_inv, gbx, gby, stick, vmode, 0.5 * h)
    vbx, vby = u * ca * n_inv, -u * sa * n_inv
    xg = 0.5 * h * (gbx * vbx + gby * vby)
    k = 1.0 / (1.0 + xg * (1.0 + xg * (0.5 + xg / 6.0)))      # n / n_mid
    vbx, vby = vbx * k, vby * k
    phi = r * h
    cp, sp = small_cos_sin(phi)
    p2 = phi * phi
    a1 = (1.0 - p2 * (1.0 / 6.0 - p2 / 120.0)) * h
    a2 = phi * (0.5 - p2 * (1.0 / 24.0 - p2 / 720.0)) * h
    bx, by = vbx * a1 - vby * a2, vbx * a2 + vby * a1
    x, y = x + c * bx - s * by, y + s * bx + c * by
    c, s = c * cp - s * sp, s * cp + c * sp
    n, gx, gy = fields_eval(*field, x, y)
    n_inv = 1.0 / n
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    u, al, ca, sa, r, E = kick(sh, u, al, ca, sa, r, E, n_inv, gbx, gby, stick, vmode, 0.5 * h)
    return x, y, c, s, u, al, ca, sa, r, E, n, gx, gy


def init(x, y, head, speed, E, field):
    n, gx, gy = fields_eval(*field, x, y)
    z = 0.0 * x
    return x, y, jnp.cos(head), jnp.sin(head), speed * n, z, 1.0 + z, z, z, E, n, gx, gy
