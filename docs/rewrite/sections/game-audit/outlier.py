import sys, dataclasses, types as _t
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/boost-and-broadside/src"); sys.path.insert(0, "/home/user/boost-and-broadside/docs/rewrite/sections/simplification")
import torch
torch.set_num_threads(4)
from lean_flight import *
E = api.PRESETS["real_scale_elimination"]
types = api.ship_types(ship_set=E.ship_set)
rules = api.rules_of(E, types=types); c, h = rules.flight, rules.tick_length
B, N = 256, rules.num_ships
g = torch.Generator().manual_seed(3)
state = api.reset(rules, types, api.uniforms(rules, B, generator=g), fogged=torch.zeros(B, dtype=torch.bool))
rnd = lambda k: torch.randint(0, k, (B, N), generator=g)
P = precompute(c, types, state.ships.ship_type)
full = dict(x="x", y="y", hc="heading_cos", hs="heading_sin", speed="proper_speed", angle="angle_of_attack", cos="angle_of_attack_cos", sin="angle_of_attack_sin", nose_rate="nose_rate", energy="energy", logn="log_refractive_index", gx="log_index_gradient_x", gy="log_index_gradient_y")
held = None; rows = []
for t in range(240):
    new = torch.stack([rnd(3), rnd(7), rnd(2)], -1)
    held = new if held is None else torch.where(torch.rand((B, N, 1), generator=g) < 0.1, new, held)
    a = Action(power=held[..., 0], stick=held[..., 1], shoot=held[..., 2])
    sh = state.ships
    airframe = F.airframe_of(types, sh.ship_type)
    stick, power = F.controls(airframe, a.power, a.stick)
    ref = F.flight_tick(c, airframe, h, sh, stick, power, state.fields)
    s = _t.SimpleNamespace(**{k: getattr(sh, v).clone() for k, v in full.items()})
    out = flight_tick(c, P, h, s, a.stick, a.power, state.fields)
    d = torch.where(sh.alive, (out.angle - ref.angle_of_attack).abs(), 0.0)
    big = d > 2e-6
    for e, n in big.nonzero().tolist():
        rows.append((d[e, n].item(), sh.proper_speed[e, n].item(), sh.energy[e, n].item(), sh.angle_of_attack[e, n].item(), sh.nose_rate[e,n].item(), a.stick[e, n].item(), a.power[e, n].item(), sh.ship_type[e,n].item(), sh.log_refractive_index[e,n].item(), (sh.energy[e,n]/497563.1875).item()))
    state, _ = api.step(rules, types, state, a, api.spread(rules, B, generator=g), api.uniforms(rules, B, generator=g))
print(len(rows), "samples above 2e-6 rad of", 240*B*N)
rows.sort(reverse=True)
print("diff speed energy angle nose_rate stick power type logn store_fraction")
for r in rows[:14]: print("  %.2e %.1f %.0f %.4f %.3f %d %d %d %.3f %.4f" % r)
import collections
print("stick histogram of outliers", collections.Counter(r[5] for r in rows), "power", collections.Counter(r[6] for r in rows))
print("store fraction range of outliers", min(r[9] for r in rows), max(r[9] for r in rows))
