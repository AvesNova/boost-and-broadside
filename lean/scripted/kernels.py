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
    "corner_speed",
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
    "boundary",
    "slow",
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
    "speed_weight",
    "boundary_weight",
    "field_weight",
    "preferred_speed",
)
TINY = 1e-6
# My candidates per pass of a program (his all at once): a tile of BLOCK_A x 32.
BLOCK_A = 4
WARPS = 1


@triton.jit
def _gun(
    rel_x,
    rel_y,
    distance,
    inverse_distance,
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
    inverse_range,
    drag,
    inverse_drag_launch,
    kinetic_scale,
    explosive,
    index,
    lead_reach,
    blur_time,
    half_size,
):
    """`opponent.gun_solution`: aim at the lead point, range band, kinetic worth, aspect (the
    same arithmetic, with reciprocals for divisions)."""
    reach = tl.minimum(distance, lead_reach)
    flight = (tl.exp(drag * reach) - 1.0) * index * inverse_drag_launch
    lead_x = rel_x + velocity_x * flight
    lead_y = rel_y + velocity_y * flight
    inverse_lead = tl.rsqrt(lead_x * lead_x + lead_y * lead_y + 1e-6)
    off = 1.0 - (nose_x * lead_x + nose_y * lead_y) * inverse_lead
    tolerance = tl.maximum(target_radius * inverse_distance + blur_time, half_size)
    inverse_tolerance = 1.0 / tolerance
    aligned = tl.exp(-off * inverse_tolerance * inverse_tolerance)
    band = tl.minimum(tl.maximum((distance * inverse_range - 0.8) * 2.5, 0.0), 1.0)
    in_range = 1.0 - band * band * (3.0 - 2.0 * band)
    kinetic = explosive + kinetic_scale * tl.exp(-2.0 * drag * distance)
    hit_cos = -(bullet_x * target_nose_x + bullet_y * target_nose_y)
    t = tl.minimum(tl.maximum((target_inner - hit_cos) * target_width, 0.0), 1.0)
    aspect = t * t * t * (t * (6.0 * t - 15.0) + 10.0)
    return aligned * in_range * kinetic * aspect


@triton.jit
def _bullet(world, path_x, path_y, nose_x, nose_y, muzzle):
    """The launch direction: map velocity plus the muzzle speed along the nose, normalised."""
    x = world * path_x + muzzle * nose_x
    y = world * path_y + muzzle * nose_y
    inverse = tl.rsqrt(x * x + y * y + 1e-6)
    return x * inverse, y * inverse


@triton.jit
def _pairs(
    mine,
    seen,
    constants,
    parameters,
    times,
    his_row,
    now,
    fields,
    out,
    cover,
    ENEMIES,
    SLOTS,
    FIELDS_COUNT,
    C: tl.constexpr,
    P: tl.constexpr,
    F: tl.constexpr,
    Q: tl.constexpr,
    W: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pair = tl.program_id(0)
    me = pair // ENEMIES
    him = tl.load(his_row + pair)
    b = tl.arange(0, BLOCK_B)
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
    his_corner = tl.load(constants + him * Q + 11)
    threat_weight = tl.load(parameters + me * W + 0)
    angles_weight = tl.load(parameters + me * W + 1)
    energy_weight = tl.load(parameters + me * W + 2)
    collision_weight = tl.load(parameters + me * W + 3)
    lead_cap = tl.load(parameters + me * W + 4)
    blur = tl.load(parameters + me * W + 5)
    half_size = 0.5 * tl.load(parameters + me * W + 6)
    reach = tl.load(parameters + me * W + 7) * (my_radius + his_radius)
    inverse_reach_squared = 1.0 / (reach * reach)
    inverse_closeness = 1.0 / (tl.load(parameters + me * W + 8) * my_range)
    temperature = tl.load(parameters + me * W + 9)
    caution = tl.load(parameters + me * W + 10)
    speed_weight = tl.load(parameters + me * W + 11)
    boundary_weight = tl.load(parameters + me * W + 12)
    field_weight = tl.load(parameters + me * W + 13)
    his_log_preferred = tl.log(tl.load(parameters + me * W + 14) * his_corner)
    my_lead_reach = lead_cap * my_range
    his_lead_reach = lead_cap * his_range
    my_inverse_range = 1.0 / my_range
    his_inverse_range = 1.0 / his_range
    my_inverse_drag_launch = 1.0 / (my_drag * my_launch)
    his_inverse_drag_launch = 1.0 / (his_drag * his_launch)
    my_ratio = my_launch * my_reference
    his_ratio = his_launch * his_reference
    my_kinetic = (1.0 - my_explosive) * my_ratio * my_ratio
    his_kinetic = (1.0 - his_explosive) * his_ratio * his_ratio
    now_x = tl.load(now + 2 * pair)
    now_y = tl.load(now + 2 * pair + 1)
    # His own-only terms per candidate, with my weights and my preferred speed for his type.
    his_own = tl.zeros((BLOCK_B,), tl.float32)
    for k in tl.static_range(P):
        weight = tl.load(parameters + me * W + 15 + k)
        his_base = seen + ((him * C + b) * P + k) * F
        h_speed = tl.load(his_base + 6, mask=b_ok, other=1.0)
        h_boundary = tl.load(his_base + 10, mask=b_ok, other=0.0)
        h_slow = tl.load(his_base + 11, mask=b_ok, other=0.0)
        his_own -= weight * (
            speed_weight * tl.abs(tl.log(tl.maximum(h_speed, 1e-6)) - his_log_preferred)
            + boundary_weight * h_boundary
            + field_weight * h_slow
        )
    small = tl.abs(caution) < 1e-3
    beta = tl.where(small, 1.0, caution)
    inverse_beta = 1.0 / beta
    inverse_temperature = 1.0 / temperature
    for chunk in range(0, tl.cdiv(C, BLOCK_A)):
        a = chunk * BLOCK_A + tl.arange(0, BLOCK_A)
        a_ok = a < C
        covered = tl.zeros((BLOCK_A,), tl.float32)
        previous_x = now_x + tl.zeros((BLOCK_A, BLOCK_B), tl.float32)
        previous_y = now_y + tl.zeros((BLOCK_A, BLOCK_B), tl.float32)
        s_me = tl.zeros((BLOCK_A, BLOCK_B), tl.float32)
        s_him = tl.zeros((BLOCK_A, BLOCK_B), tl.float32)
        started = 0.0
        for k in tl.static_range(P):
            time = tl.load(times + k)
            weight = tl.load(parameters + me * W + 15 + k)
            discount = tl.load(parameters + me * W + 15 + P + k)
            inverse_duration = 1.0 / (time - started)
            blur_time = blur * time
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
            # Cover: my point behind a field core from his, flying straight on (candidate 0).
            straight_x = tl.load(seen + ((him * C) * P + k) * F + 0)
            straight_y = tl.load(seen + ((him * C) * P + k) * F + 1)
            blocked = tl.zeros((BLOCK_A,), tl.int1)
            environment = me // SLOTS
            for field in range(0, FIELDS_COUNT):
                row = fields + environment * 5 * FIELDS_COUNT + field
                centre_x = tl.load(row)
                centre_y = tl.load(row + FIELDS_COUNT)
                radius = tl.load(row + 2 * FIELDS_COUNT)
                half_band = 0.5 * tl.load(row + 3 * FIELDS_COUNT)
                core = tl.maximum(radius - half_band, 0.0)
                core_squared = core * core
                outer = radius + half_band
                a_offset_x = mx - centre_x
                a_offset_y = my - centre_y
                b_offset_x = straight_x - centre_x
                b_offset_y = straight_y - centre_y
                a_squared = a_offset_x * a_offset_x + a_offset_y * a_offset_y
                b_squared = b_offset_x * b_offset_x + b_offset_y * b_offset_y
                core_a = a_squared < core_squared
                core_b = b_squared < core_squared
                both_in_field = (a_squared < outer * outer) & (b_squared < outer * outer)
                ends = (core_a ^ core_b) & ~both_in_field
                sight_x = mx - straight_x
                sight_y = my - straight_y
                sight_along = -(b_offset_x * sight_x + b_offset_y * sight_y)
                sight_across = b_offset_x * sight_y - b_offset_y * sight_x
                sight_length = sight_x * sight_x + sight_y * sight_y
                passes = (
                    ~(core_a | core_b)
                    & (sight_across * sight_across < core_squared * sight_length)
                    & (sight_along > 0.0)
                    & (sight_along < sight_length)
                )
                blocked = blocked | ends | passes
            covered += weight * blocked.to(tl.float32)
            mbx, mby = _bullet(m_world, m_path_x, m_path_y, m_nose_x, m_nose_y, my_muzzle)
            hbx, hby = _bullet(h_world, h_path_x, h_path_y, h_nose_x, h_nose_y, his_muzzle)
            rel_x = hx[None, :] - mx[:, None]
            rel_y = hy[None, :] - my[:, None]
            velocity_x = (h_world * h_path_x)[None, :] - (m_world * m_path_x)[:, None]
            velocity_y = (h_world * h_path_y)[None, :] - (m_world * m_path_y)[:, None]
            squared = rel_x * rel_x + rel_y * rel_y + 1e-6
            inverse_distance = tl.rsqrt(squared)
            distance = squared * inverse_distance
            gun = _gun(
                rel_x,
                rel_y,
                distance,
                inverse_distance,
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
                my_inverse_range,
                my_drag,
                my_inverse_drag_launch,
                my_kinetic,
                my_explosive,
                m_index[:, None],
                my_lead_reach,
                blur_time,
                half_size,
            )
            threat = _gun(
                -rel_x,
                -rel_y,
                distance,
                inverse_distance,
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
                his_inverse_range,
                his_drag,
                his_inverse_drag_launch,
                his_kinetic,
                his_explosive,
                h_index[None, :],
                his_lead_reach,
                blur_time,
                half_size,
            )
            my_off = (m_nose_x[:, None] * rel_x + m_nose_y[:, None] * rel_y) * inverse_distance
            his_off = -(h_nose_x[None, :] * rel_x + h_nose_y[None, :] * rel_y) * inverse_distance
            angles = 0.5 * (my_off - his_off)
            energy = (
                0.5 * ((m_speed * m_speed)[:, None] - (h_speed * h_speed)[None, :])
                + m_energy[:, None]
                - h_energy[None, :]
            ) * my_inverse_energy
            ratio = distance * inverse_closeness
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
            closing = tl.sqrt(move_squared) * inverse_duration
            contact = tl.exp(-closest * inverse_reach_squared) * closing * my_collision
            collision = collision_weight * contact * discount
            s_me += weight * (gun - threat_weight * threat + position) - collision
            s_him += weight * (threat - threat_weight * gun - position) - collision
            previous_x = rel_x
            previous_y = rel_y
            started = time
        s_him += his_own[None, :]
        # His reply collapsed per row (`opponent.collapse`).
        logits = tl.where(b_ok[None, :], s_him * inverse_temperature, -float("inf"))
        top = tl.max(logits, 1)
        log_w = logits - top[:, None]
        log_w = log_w - tl.log(tl.sum(tl.exp(log_w), 1))[:, None]
        w = tl.exp(log_w)
        mean = tl.sum(w * s_me, 1)
        centred = s_me - mean[:, None]
        variance = tl.sum(w * centred * centred, 1)
        exponent = log_w + beta * centred
        peak = tl.max(exponent, 1)
        spread = (peak + tl.log(tl.sum(tl.exp(exponent - peak[:, None]), 1))) * inverse_beta
        spread = tl.where(small, 0.5 * caution * variance, spread)
        tl.store(out + pair * C + a, mean + spread, mask=a_ok)
        tl.store(cover + pair * C + a, covered, mask=a_ok)


@torch.library.triton_op("scripted::pair_scores", mutates_args=())
def pair_scores(
    mine: torch.Tensor,
    seen: torch.Tensor,
    constants: torch.Tensor,
    parameters: torch.Tensor,
    times: torch.Tensor,
    his_row: torch.Tensor,
    now: torch.Tensor,
    fields: torch.Tensor,
    enemies: int,
    slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The collapsed pair scores and the cover, (pairs, C) float32 each. mine, seen: (ships, C,
    P, F) the own and the enemies' (as seen) paths, fields in FIELDS' order; constants (ships,
    Q) in CONSTANTS' order; parameters (ships, W): PARAMETERS' order, then the P time weights,
    then the P discounts; times (P,) seconds; his_row (pairs,) int32 the enemy's ship row
    (pairs = ships x enemies, pair p belongs to ship p // enemies; ships = environments x
    slots); now (pairs, 2) him minus me now; fields (E, 5, F) the environments' fields. The
    cover: per my candidate the time-weighted share of the points behind a field core from
    him flying straight on (`opponent._cover`'s blocked part)."""
    ships, candidates, points, width = mine.shape
    pairs = his_row.shape[0]
    out = torch.empty(pairs, candidates, dtype=torch.float32, device=mine.device)
    cover = torch.empty(pairs, candidates, dtype=torch.float32, device=mine.device)
    torch.library.wrap_triton(_pairs)[(pairs,)](
        mine,
        seen,
        constants,
        parameters,
        times,
        his_row,
        now,
        fields,
        out,
        cover,
        enemies,
        slots,
        fields.shape[-1],
        C=candidates,
        P=points,
        F=width,
        Q=constants.shape[1],
        W=parameters.shape[1],
        BLOCK_A=BLOCK_A,
        BLOCK_B=triton.next_power_of_2(candidates),
        num_warps=WARPS,
    )
    return out, cover
