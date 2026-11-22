"""A lean restatement of v2's flight tick for the real-scale rules (direct charge, vertical
exchange, separation, drag rise, mass 1, bleed exponent 2), per-ship constants precomputed.
Counted and compared one step against v2's flight_tick. Scratch prototype, torch on the CPU."""
import sys, dataclasses, math, types as _t
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/boost-and-broadside/src")
sys.path.insert(0, "/home/user/boost-and-broadside/docs/rewrite/sections/simplification")
import torch
import regions as R
from bnb.game import api
from bnb.contracts import Action
from bnb.game.physics import flight as F
from bnb.game.physics.medium import medium
from bnb.game.ship_types import per_ship

def softmin(a, b, q):
    low, high = torch.minimum(a, b), torch.maximum(a, b)
    return low * (1.0 + (low / high.clamp_min(1e-30)) ** q) ** (-1.0 / q)

def cs(angle):
    s = angle * angle
    return 1.0 - s * (0.5 - s * (1.0 / 24.0 - s / 720.0)), angle * (1.0 - s * (1.0 / 6.0 - s * (1.0 / 120.0 - s / 5040.0)))

def precompute(c, types, ship_type):
    """Per-ship constants (E, N), once per reset."""
    g = lambda name: per_ship(getattr(types, name), ship_type)
    P = _t.SimpleNamespace()
    P.cl, P.amax, P.cd0 = g("max_lift_coefficient"), g("max_acceleration"), g("zero_lift_drag")
    P.p_full, P.t0 = g("engine_power_full_store"), g("static_thrust")
    P.dp = g("engine_power") - P.p_full
    P.emax = g("max_energy"); P.inv_emax = 1.0 / P.emax
    P.stall = c.stall_angle * g("stall_angle_scale"); P.inv_stall = 1.0 / P.stall
    P.knee = c.engine_knee * g("engine_knee_scale"); P.inv_1mknee = 1.0 / (1.0 - P.knee)
    P.inv_fade = 1.0 / (c.store_fade_band * P.emax)
    P.climb, P.dive = c.climb_exchange * g("climb_exchange_scale"), c.dive_exchange * g("dive_exchange_scale")
    P.kind = c.induced_drag_factor * g("induced_drag_scale")
    P.inv_cl2 = 1.0 / (P.cl * P.cl); P.inv_cl = 1.0 / P.cl
    P.cl_over_stall = P.cl / P.stall
    P.freq = g("corner_nose_frequency") / torch.sqrt(P.amax / P.cl)
    P.fine, P.sustained = g("fine_stick"), g("sustained_stick")
    # stick lift fraction by action index (E, N, 7) and power sign table
    lv = torch.tensor([0.0, -1, 1, -2, 2, -3, 3])
    mag = torch.stack([torch.zeros_like(P.fine), P.fine, P.fine, P.sustained, P.sustained, torch.ones_like(P.fine), torch.ones_like(P.fine)], -1)
    P.stick_table = torch.sign(lv) * mag
    width = c.gravity / 32.0
    raw = torch.stack([torch.full_like(P.climb, c.gravity), c.gravity - P.dive, c.gravity + P.climb], -1)  # coast, boost, reverse
    P.commanded = torch.stack([torch.zeros_like(P.climb), -P.dive, P.climb], -1)
    P.demand = 0.5 * (raw + torch.sqrt(raw * raw + width * width))
    return P

def setting(c, P, T, speed, energy):
    t = ((energy * P.inv_emax - P.knee) * P.inv_1mknee).clamp(0.0, 1.0)
    fall = t * t * (3.0 - 2.0 * t)
    reach = P.cl * speed * speed
    budget = softmin(reach, P.amax, 16.0).clamp_min(1e-12)
    command = softmin(T.stick_abs * reach, P.amax, 16.0)
    sink_room = (energy * P.inv_fade).clamp(0.0, 1.0)
    budget2 = budget * budget
    held = softmin(T.demand * (1.0 - sink_room), budget, 16.0)
    lateral_size = softmin(command, torch.sqrt((budget2 - held * held).clamp_min(0.0)), 16.0)
    vertical = softmin(T.demand, torch.sqrt((budget2 - lateral_size * lateral_size).clamp_min(0.0)), 16.0)
    exchange = T.commanded - (T.demand - vertical)
    share = (lateral_size / budget).clamp(0.0, 1.0)
    headroom = ((P.emax - energy) * P.inv_fade).clamp(0.0, 1.0)
    exchange = exchange * torch.where(exchange > 0.0, headroom, sink_room)
    charge = P.dp * (1.0 - fall) * torch.sqrt(1.0 - share * share) * (vertical * T.inv_demand) * headroom
    thrust = softmin(P.p_full / speed, P.t0, 2.0)
    return T.stick_sign * lateral_size, exchange, vertical, thrust, charge

def rates(c, P, st, inv_index, speed, lift, angle_cos, inv_u2):
    lateral, exchange, vertical, thrust, charge = st
    induced = vertical * inv_u2
    s2 = lift * lift + induced * induced
    drag = (P.cd0 * (1.0 + (speed * (1.0 / c.critical_speed)) ** c.drag_rise_exponent)
            + P.kind * s2 * (1.0 + c.separation_drag * (s2 * P.inv_cl2) ** (0.5 * c.separation_exponent))) * speed * speed
    return (thrust * angle_cos - drag - exchange) * inv_index, (speed * exchange + charge) * inv_index

def lift_coefficient(P, angle):
    r = (angle * P.inv_stall).clamp(-2.0, 2.0)
    return P.cl * (2.0 * r - r * r.abs())

def expm2(a11, a12, a21, a22, dt):
    b = torch.stack([torch.stack([a11, a12], -1), torch.stack([a21, a22], -1)], -2) * (dt / 2.0)
    eye = torch.eye(2)
    h = eye + b / 4.0
    h = eye + (b @ h) / 3.0
    h = eye + (b @ h) / 2.0
    h = eye + b @ h
    h = h @ h
    return h[..., 0, 0], h[..., 0, 1], h[..., 1, 0], h[..., 1, 1]

def kick(c, P, T, s, inv_index, gx, gy, dt):
    speed0, angle0, cos0, sin0, energy, nose_rate = s.speed, s.angle, s.cos, s.sin, s.energy, s.nose_rate
    safe0 = speed0.clamp_min(c.speed_floor)
    first = setting(c, P, T, safe0, energy)
    lift0 = lift_coefficient(P, angle0)
    sr0, er0 = rates(c, P, first, inv_index, speed0, lift0, cos0, 1.0 / (safe0 * safe0))
    speed = (speed0 + 0.5 * dt * sr0).clamp_min(c.speed_floor)
    cur = setting(c, P, T, speed, energy + 0.5 * dt * er0)
    lateral, exchange, vertical, thrust, charge = cur
    inv_u2 = 1.0 / (speed * speed)
    over = inv_index / speed
    u2 = speed * speed
    path_rate = (lift0 * u2 + thrust * sin0) * over + (speed * inv_index) * (cos0 * gy + sin0 * gx)
    ratio = (angle0 * P.inv_stall).abs()
    slope = (P.cl_over_stall * (2.0 - 2.0 * ratio)).clamp_min(0.0)       # zero from twice the stall angle
    kappa = (slope * u2 + thrust * cos0) * over
    omega = (P.freq * speed).clamp_min(c.nose_frequency_floor) * inv_index
    reach = (lateral.abs() * inv_u2 * P.inv_cl).clamp(0.0, 1.0)
    step = torch.sign(lateral) * P.stall * (1.0 - torch.sqrt(1.0 - reach)) - angle0
    zw = 2.0 * c.nose_damping * omega
    m11, m12, m21, m22 = expm2(-kappa, torch.ones_like(kappa), zw * kappa - omega * omega, -zw, dt)
    steady = kappa * step
    y2 = (nose_rate - path_rate) - steady
    turn = step - m11 * step + m12 * y2
    new_rate = path_rate + steady - m21 * step + m22 * y2
    hc, hs = cs(0.5 * turn)
    mid_cos = cos0 * hc - sin0 * hs
    mid_lift = lift_coefficient(P, angle0 + 0.5 * turn)
    sr1, _ = rates(c, P, cur, inv_index, speed0, mid_lift, mid_cos, inv_u2)
    sr2, er2 = rates(c, P, cur, inv_index, (speed0 + 0.5 * dt * sr1).clamp_min(c.speed_floor), mid_lift, mid_cos, inv_u2)
    tc, ts = cs(turn)
    s.speed = (speed0 + dt * sr2).clamp_min(c.speed_floor)
    s.angle = angle0 + turn
    s.cos, s.sin = cos0 * tc - sin0 * ts, cos0 * ts + sin0 * tc
    s.nose_rate = new_rate
    s.energy = torch.minimum((energy + dt * er2).clamp_min(0.0), P.emax)
    return s

def nose_frame(s):
    return s.hc * s.gx + s.hs * s.gy, s.hc * s.gy - s.hs * s.gx

def flight_tick(c, P, h, s, stick_index, power_index, fields):
    T = _t.SimpleNamespace()
    stick = torch.gather(P.stick_table, -1, stick_index[..., None].long())[..., 0]
    T.stick_abs, T.stick_sign = stick.abs(), torch.sign(stick)
    pi = power_index[..., None].long()
    T.commanded = torch.gather(P.commanded, -1, pi)[..., 0]
    T.demand = torch.gather(P.demand, -1, pi)[..., 0]
    T.inv_demand = 1.0 / T.demand
    inv_index = torch.exp(-s.logn)
    s = kick(c, P, T, s, inv_index, *nose_frame(s), 0.5 * h)
    gx, gy = nose_frame(s)
    v = s.speed * inv_index
    vx, vy = v * s.cos, -v * s.sin
    xi = 0.5 * h * (gx * vx + gy * vy)
    stretch = 1.0 / (1.0 + xi * (1.0 + xi * (0.5 + xi / 6.0)))
    vx, vy = vx * stretch, vy * stretch
    arc = s.nose_rate * h
    a2 = arc * arc
    along = (1.0 - a2 * (1.0 / 6.0 - a2 / 120.0)) * h
    across = arc * (0.5 - a2 * (1.0 / 24.0 - a2 / 720.0)) * h
    bx, by = vx * along - vy * across, vx * across + vy * along
    s.x, s.y = s.x + (s.hc * bx - s.hs * by), s.y + (s.hs * bx + s.hc * by)
    ac, as_ = cs(arc)
    hc, hs = s.hc * ac - s.hs * as_, s.hc * as_ + s.hs * ac
    n = torch.rsqrt(hc * hc + hs * hs)
    s.hc, s.hs = hc * n, hs * n
    n = torch.rsqrt(s.cos * s.cos + s.sin * s.sin)
    s.cos, s.sin = s.cos * n, s.sin * n
    R.STACK.append("medium"); s.logn, s.gx, s.gy = medium(fields, s.x, s.y); R.STACK.pop()
    return kick(c, P, T, s, torch.exp(-s.logn), *nose_frame(s), 0.5 * h)

if __name__ == "__main__":
    torch.set_num_threads(4)
    E = api.PRESETS["real_scale_elimination"]
    types = api.ship_types(ship_set=E.ship_set)
    rules = api.rules_of(E, types=types)
    c, h = rules.flight, rules.tick_length
    B, N = 256, rules.num_ships
    g = torch.Generator().manual_seed(3)
    state = api.reset(rules, types, api.uniforms(rules, B, generator=g), fogged=torch.zeros(B, dtype=torch.bool))
    rnd = lambda k: torch.randint(0, k, (B, N), generator=g)
    worst = {}
    names = dict(x="x", y="y", hc="heading_cos", hs="heading_sin", speed="proper_speed", angle="angle_of_attack",
                 cos="angle_of_attack_cos", sin="angle_of_attack_sin", nose_rate="nose_rate", energy="energy",
                 logn="log_refractive_index", gx="log_index_gradient_x", gy="log_index_gradient_y")
    P = precompute(c, types, state.ships.ship_type)
    held = None
    for t in range(240):
        new = torch.stack([rnd(3), rnd(7), rnd(2)], -1)
        held = new if held is None else torch.where(torch.rand((B, N, 1), generator=g) < 0.1, new, held)
        a = Action(power=held[..., 0], stick=held[..., 1], shoot=held[..., 2])
        sh = state.ships
        # v2's flight alone, from this state
        airframe = F.airframe_of(types, sh.ship_type)
        stick, power = F.controls(airframe, a.power, a.stick)
        ref = F.flight_tick(c, airframe, h, sh, stick, power, state.fields)
        s = _t.SimpleNamespace(**{k: getattr(sh, v).clone() for k, v in names.items()})
        if t == 239:
            R.reset()
            with R.Counter():
                out = flight_tick(c, P, h, s, a.stick, a.power, state.fields)
            total = sum(v[0] for v in R.EXCLUSIVE.values())
            print("lean flight tick: %d compute ops (of which medium %d); v2 flight_tick+airframe_of+controls: 1575" % (total, R.EXCLUSIVE["medium"][0]))
            print(R.OPS["(outside any region)"].most_common(14))
        else:
            out = flight_tick(c, P, h, s, a.stick, a.power, state.fields)
        alive = sh.alive
        for k, v in names.items():
            d = (getattr(out, k) - getattr(ref, v)).abs()[alive]
            scale = getattr(ref, v).abs()[alive]
            worst[k] = max(worst.get(k, 0.0), d.max().item())
        state, _ = api.step(rules, types, state, a, api.spread(rules, B, generator=g), api.uniforms(rules, B, generator=g))
    print("one-step max abs difference, lean vs v2 flight_tick, 239 ticks x %d envs x %d ships (random sticky actions, fields on):" % (B, N))
    for k in names: print("  %-10s %.3e" % (k, worst[k]))
    print("ranges: speed %.1f..%.1f, energy %.0f..%.0f, |angle| max %.3f" % (state.ships.proper_speed.min(), state.ships.proper_speed.max(), state.ships.energy.min(), state.ships.energy.max(), state.ships.angle_of_attack.abs().max()))
