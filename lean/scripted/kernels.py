"""The look-ahead's pair stage as one Triton kernel: for each (ship, scored enemy) pair, every
(my candidate a, his candidate b) at every look-ahead point, S_me and S_him summed over the
points, and his reply collapsed per row (lean.scripted.opponent `pair_tables` + `collapse`,
its plain twin, which runs on the CPU and wherever the switch is off; a test holds the kernel
to it). One program per pair holds the 21 x 21 tile in registers, so no 441-wide table ever
reaches memory. A `torch.library.triton_op`, so it compiles inside the agent's graph.
"""

import torch
import triton
import triton.language as tl

# On: the kernel runs on CUDA; off: the plain twin everywhere (read at trace time).
USE_KERNEL = True
# Per-ship constants the pair stage reads, in this order (columns of `constants`).
CONSTANTS = (
    "hit_radius",
    "nose_inner_cos",
    "inverse_nose_width",
    "inverse_max_energy",
    "gun_range",
    "bullet_drag",
    "bullet_launch_speed",
    "muzzle_speed",
    "inverse_reference_bullet_speed",
    "explosive_share",
    "inverse_collision_speed",
)
# The path fields per candidate and point, in this order (the last axis of `mine`).
FIELDS = (
    "x",
    "y",
    "path_cos",
    "path_sin",
    "nose_cos",
    "nose_sin",
    "speed",
    "world_speed",
    "energy",
    "log_index",
)
# Per-ship parameters, in this order; then the points' time weights and discounts (2 P).
PARAMETERS = (
    "threat_weight",
    "angles_weight",
    "energy_weight",
    "collision_weight",
    "lead_cap",
    "aim_blur",
    "target_size_minimum",
    "collision_margin",
    "closeness_range",
    "reply_temperature",
    "caution",
)
TINY = 1e-6


@triton.jit
def _gun(
    rel_x,
    rel_y,
    nose_x,
    nose_y,
    velocity_x,
    velocity_y,
    bullet_x,
    bullet_y,
    target_nose_x,
    target_nose_y,
    target_radius,
    target_inner,
    target_width,
    gun_range,
    drag,
    launch,
    reference,
    explosive,
    index,
    lead_cap,
    blur,
    size_min,
    time,
):
    """`opponent.gun_solution`: aim at the lead point, range band, kinetic worth, aspect."""
    distance = tl.sqrt(rel_x * rel_x + rel_y * rel_y + 1e-6)
    reach = tl.minimum(distance, lead_cap * gun_range)
    flight = (tl.exp(drag * reach) - 1.0) * index / (drag * launch)
    lead_x = rel_x + velocity_x * flight
    lead_y = rel_y + velocity_y * flight
    lead_length = tl.sqrt(lead_x * lead_x + lead_y * lead_y + 1e-6)
    off = 1.0 - (nose_x * lead_x + nose_y * lead_y) / lead_length
    tolerance = tl.maximum(target_radius / distance + blur * time, 0.5 * size_min)
    aligned = tl.exp(-off / (tolerance * tolerance))
    band = tl.minimum(tl.maximum((distance / gun_range - 0.8) / 0.4, 0.0), 1.0)
    in_range = 1.0 - band * band * (3.0 - 2.0 * band)
    ratio = launch * reference
    kinetic = explosive + (1.0 - explosive) * ratio * ratio * tl.exp(-2.0 * drag * distance)
    hit_cos = -(bullet_x * target_nose_x + bullet_y * target_nose_y)
    t = tl.minimum(tl.maximum((target_inner - hit_cos) * target_width, 0.0), 1.0)
    aspect = t * t * t * (t * (6.0 * t - 15.0) + 10.0)
    return aligned * in_range * kinetic * aspect


@triton.jit
def _pairs(
    mine,
    seen,
    constants,
    parameters,
    times,
    his_row,
    now,
    his_own,
    out,
    ENEMIES,
    C: tl.constexpr,
    P: tl.constexpr,
    F: tl.constexpr,
    Q: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)
    me = pair // ENEMIES
    him = tl.load(his_row + pair)
    a = tl.arange(0, BLOCK)
    b = tl.arange(0, BLOCK)
    a_ok = a < C
    b_ok = b < C

    my_radius = tl.load(constants + me * Q + 0)
    my_inner = tl.load(constants + me * Q + 1)
    my_width = tl.load(constants + me * Q + 2)
    my_inverse_energy = tl.load(constants + me * Q + 3)
    my_range = tl.load(constants + me * Q + 4)
    my_drag = tl.load(constants + me * Q + 5)
    my_launch = tl.load(constants + me * Q + 6)
    my_muzzle = tl.load(constants + me * Q + 7)
    my_reference = tl.load(constants + me * Q + 8)
    my_explosive = tl.load(constants + me * Q + 9)
    my_collision = tl.load(constants + me * Q + 10)
    his_radius = tl.load(constants + him * Q + 0)
    his_inner = tl.load(constants + him * Q + 1)
    his_width = tl.load(constants + him * Q + 2)
    his_range = tl.load(constants + him * Q + 4)
    his_drag = tl.load(constants + him * Q + 5)
    his_launch = tl.load(constants + him * Q + 6)
    his_muzzle = tl.load(constants + him * Q + 7)
    his_reference = tl.load(constants + him * Q + 8)
    his_explosive = tl.load(constants + him * Q + 9)

    threat_weight = tl.load(parameters + me * W + 0)
    angles_weight = tl.load(parameters + me * W + 1)
    energy_weight = tl.load(parameters + me * W + 2)
    collision_weight = tl.load(parameters + me * W + 3)
    lead_cap = tl.load(parameters + me * W + 4)
    blur = tl.load(parameters + me * W + 5)
    size_min = tl.load(parameters + me * W + 6)
    reach = tl.load(parameters + me * W + 7) * (my_radius + his_radius)
    closeness = tl.load(parameters + me * W + 8) * my_range
    temperature = tl.load(parameters + me * W + 9)
    caution = tl.load(parameters + me * W + 10)

    previous_x = tl.load(now + 2 * pair) + tl.zeros((BLOCK, BLOCK), tl.float32)
    previous_y = tl.load(now + 2 * pair + 1) + tl.zeros((BLOCK, BLOCK), tl.float32)
    s_me = tl.zeros((BLOCK, BLOCK), tl.float32)
    s_him = tl.zeros((BLOCK, BLOCK), tl.float32)
    started = 0.0
    for k in tl.static_range(P):
        time = tl.load(times + k)
        weight = tl.load(parameters + me * W + 11 + k)
        discount = tl.load(parameters + me * W + 11 + P + k)
        my_base = mine + ((me * C + a) * P + k) * F
        his_base = seen + ((him * C + b) * P + k) * F
        mx = tl.load(my_base + 0, mask=a_ok, other=0.0)
        my = tl.load(my_base + 1, mask=a_ok, other=0.0)
        m_path_x = tl.load(my_base + 2, mask=a_ok, other=1.0)
        m_path_y = tl.load(my_base + 3, mask=a_ok, other=0.0)
        m_nose_x = tl.load(my_base + 4, mask=a_ok, other=1.0)
        m_nose_y = tl.load(my_base + 5, mask=a_ok, other=0.0)
        m_speed = tl.load(my_base + 6, mask=a_ok, other=1.0)
        m_world = tl.load(my_base + 7, mask=a_ok, other=1.0)
        m_energy = tl.load(my_base + 8, mask=a_ok, other=0.0)
        m_index = tl.exp(tl.load(my_base + 9, mask=a_ok, other=0.0))
        hx = tl.load(his_base + 0, mask=b_ok, other=0.0)
        hy = tl.load(his_base + 1, mask=b_ok, other=0.0)
        h_path_x = tl.load(his_base + 2, mask=b_ok, other=1.0)
        h_path_y = tl.load(his_base + 3, mask=b_ok, other=0.0)
        h_nose_x = tl.load(his_base + 4, mask=b_ok, other=1.0)
        h_nose_y = tl.load(his_base + 5, mask=b_ok, other=0.0)
        h_speed = tl.load(his_base + 6, mask=b_ok, other=1.0)
        h_world = tl.load(his_base + 7, mask=b_ok, other=1.0)
        h_energy = tl.load(his_base + 8, mask=b_ok, other=0.0)
        h_index = tl.exp(tl.load(his_base + 9, mask=b_ok, other=0.0))
        # Bullet directions (map velocity plus muzzle speed along the nose), per candidate.
        mbx = m_world * m_path_x + my_muzzle * m_nose_x
        mby = m_world * m_path_y + my_muzzle * m_nose_y
        inverse = 1.0 / tl.sqrt(mbx * mbx + mby * mby + 1e-6)
        mbx, mby = mbx * inverse, mby * inverse
        hbx = h_world * h_path_x + his_muzzle * h_nose_x
        hby = h_world * h_path_y + his_muzzle * h_nose_y
        inverse = 1.0 / tl.sqrt(hbx * hbx + hby * hby + 1e-6)
        hbx, hby = hbx * inverse, hby * inverse
        rel_x = hx[None, :] - mx[:, None]
        rel_y = hy[None, :] - my[:, None]
        velocity_x = (h_world * h_path_x)[None, :] - (m_world * m_path_x)[:, None]
        velocity_y = (h_world * h_path_y)[None, :] - (m_world * m_path_y)[:, None]
        gun = _gun(
            rel_x,
            rel_y,
            m_nose_x[:, None],
            m_nose_y[:, None],
            velocity_x,
            velocity_y,
            mbx[:, None],
            mby[:, None],
            h_nose_x[None, :],
            h_nose_y[None, :],
            his_radius,
            his_inner,
            his_width,
            my_range,
            my_drag,
            my_launch,
            my_reference,
            my_explosive,
            m_index[:, None],
            lead_cap,
            blur,
            size_min,
            time,
        )
        threat = _gun(
            -rel_x,
            -rel_y,
            h_nose_x[None, :],
            h_nose_y[None, :],
            -velocity_x,
            -velocity_y,
            hbx[None, :],
            hby[None, :],
            m_nose_x[:, None],
            m_nose_y[:, None],
            my_radius,
            my_inner,
            my_width,
            his_range,
            his_drag,
            his_launch,
            his_reference,
            his_explosive,
            h_index[None, :],
            lead_cap,
            blur,
            size_min,
            time,
        )
        distance = tl.sqrt(rel_x * rel_x + rel_y * rel_y + 1e-6)
        my_off = (m_nose_x[:, None] * rel_x + m_nose_y[:, None] * rel_y) / distance
        his_off = -(h_nose_x[None, :] * rel_x + h_nose_y[None, :] * rel_y) / distance
        angles = 0.5 * (my_off - his_off)
        energy = (
            0.5 * ((m_speed * m_speed)[:, None] - (h_speed * h_speed)[None, :])
            + m_energy[:, None]
            - h_energy[None, :]
        ) * my_inverse_energy
        ratio = distance / closeness
        near = 1.0 / (1.0 + ratio * ratio)
        position = near * (angles_weight * angles + energy_weight * energy)
        move_x = rel_x - previous_x
        move_y = rel_y - previous_y
        move_squared = move_x * move_x + move_y * move_y + 1e-6
        along = -(previous_x * move_x + previous_y * move_y) / move_squared
        along = tl.minimum(tl.maximum(along, 0.0), 1.0)
        closest_x = previous_x + along * move_x
        closest_y = previous_y + along * move_y
        closest = closest_x * closest_x + closest_y * closest_y
        closing = tl.sqrt(move_squared) / (time - started)
        contact = tl.exp(-closest / (reach * reach)) * closing * my_collision
        collision = collision_weight * contact * discount
        s_me += weight * (gun - threat_weight * threat + position) - collision
        s_him += weight * (threat - threat_weight * gun - position) - collision
        previous_x = rel_x
        previous_y = rel_y
        started = time
    s_him += tl.load(his_own + pair * C + b, mask=b_ok, other=0.0)[None, :]
    # His reply collapsed per row (`opponent.collapse`).
    logits = tl.where(b_ok[None, :], s_him / temperature, -float("inf"))
    top = tl.max(logits, 1)
    log_w = logits - top[:, None]
    log_w = log_w - tl.log(tl.sum(tl.exp(log_w), 1))[:, None]
    w = tl.exp(log_w)
    mean = tl.sum(w * s_me, 1)
    centred = s_me - mean[:, None]
    variance = tl.sum(w * centred * centred, 1)
    small = tl.abs(caution) < 1e-3
    beta = tl.where(small, 1.0, caution)
    exponent = log_w + beta * centred
    peak = tl.max(exponent, 1)
    spread = (peak + tl.log(tl.sum(tl.exp(exponent - peak[:, None]), 1))) / beta
    spread = tl.where(small, 0.5 * caution * variance, spread)
    tl.store(out + pair * C + a, mean + spread, mask=a_ok)


@torch.library.triton_op("scripted::pair_scores", mutates_args=())
def pair_scores(
    mine: torch.Tensor,
    seen: torch.Tensor,
    constants: torch.Tensor,
    parameters: torch.Tensor,
    times: torch.Tensor,
    his_row: torch.Tensor,
    now: torch.Tensor,
    his_own: torch.Tensor,
    enemies: int,
) -> torch.Tensor:
    """The collapsed pair scores (pairs, C) float32. mine, seen: (ships, C, P, F) the own and
    the enemies' (as seen) paths, fields in FIELDS' order; constants (ships, Q) in CONSTANTS'
    order; parameters (ships, W): PARAMETERS' order, then the P time weights, then the P
    discounts; times (P,) seconds; his_row (pairs,) int32 the enemy's ship row (pairs = ships x
    enemies, pair p belongs to ship p // enemies); now (pairs, 2) him minus me now; his_own
    (pairs, C) his own-only score per candidate."""
    ships, candidates, points, fields = mine.shape
    pairs = his_row.shape[0]
    out = torch.empty(pairs, candidates, dtype=torch.float32, device=mine.device)
    torch.library.wrap_triton(_pairs)[(pairs,)](
        mine,
        seen,
        constants,
        parameters,
        times,
        his_row,
        now,
        his_own,
        out,
        enemies,
        C=candidates,
        P=points,
        F=fields,
        Q=constants.shape[1],
        W=parameters.shape[1],
        BLOCK=triton.next_power_of_2(candidates),
        num_warps=4,
    )
    return out
