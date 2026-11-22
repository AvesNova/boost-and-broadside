import sys, dataclasses, types as _t
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/boost-and-broadside/src"); sys.path.insert(0, "/home/user/boost-and-broadside/docs/rewrite/sections/simplification")
import torch
torch.set_num_threads(4)
import lean_flight as L
from lean_flight import *
E = api.PRESETS["real_scale_elimination"]
types = api.ship_types(ship_set=E.ship_set); types64 = api.ship_types(ship_set=E.ship_set, dtype=torch.float64)
rules = api.rules_of(E, types=types); c, h = rules.flight, rules.tick_length
B, N = 256, rules.num_ships
g = torch.Generator().manual_seed(3)
state = api.reset(rules, types, api.uniforms(rules, B, generator=g), fogged=torch.zeros(B, dtype=torch.bool))
rnd = lambda k: torch.randint(0, k, (B, N), generator=g)
names = dict(x="x", y="y", hc="heading_cos", speed="proper_speed", angle="angle_of_attack", nose_rate="nose_rate", energy="energy")
P = precompute(c, types, state.ships.ship_type)
def to64(tree):
    return type(tree)(**{f.name: (getattr(tree, f.name).double() if getattr(tree, f.name).is_floating_point() else getattr(tree, f.name)) for f in dataclasses.fields(tree)})
errs_lean = {k: [] for k in names}; errs_v2 = {k: [] for k in names}; speeds = []
held = None
for t in range(240):
    new = torch.stack([rnd(3), rnd(7), rnd(2)], -1)
    held = new if held is None else torch.where(torch.rand((B, N, 1), generator=g) < 0.1, new, held)
    a = Action(power=held[..., 0], stick=held[..., 1], shoot=held[..., 2])
    sh = state.ships
    airframe = F.airframe_of(types, sh.ship_type)
    stick, power = F.controls(airframe, a.power, a.stick)
    ref = F.flight_tick(c, airframe, h, sh, stick, power, state.fields)
    af64 = F.airframe_of(types64, sh.ship_type); st64, pw64 = F.controls(af64, a.power, a.stick)
    ref64 = F.flight_tick(c, af64, h, to64(sh), st64, pw64.double(), to64(state.fields))
    names_all = dict(L.__dict__.get("names_all", {}))
    full = dict(x="x", y="y", hc="heading_cos", hs="heading_sin", speed="proper_speed", angle="angle_of_attack", cos="angle_of_attack_cos", sin="angle_of_attack_sin", nose_rate="nose_rate", energy="energy", logn="log_refractive_index", gx="log_index_gradient_x", gy="log_index_gradient_y")
    s = _t.SimpleNamespace(**{k: getattr(sh, v).clone() for k, v in full.items()})
    out = flight_tick(c, P, h, s, a.stick, a.power, state.fields)
    alive = sh.alive
    for k, v in names.items():
        errs_lean[k].append((getattr(out, k).double() - getattr(ref64, v)).abs()[alive])
        errs_v2[k].append((getattr(ref, v).double() - getattr(ref64, v)).abs()[alive])
    speeds.append(sh.proper_speed[alive])
    state, _ = api.step(rules, types, state, a, api.spread(rules, B, generator=g), api.uniforms(rules, B, generator=g))
speeds = torch.cat(speeds)
print("error against v2 in float64, one flight tick; lean float32 | v2 float32  (median, 99.9th pct, max, speed at lean max)")
for k in names:
    el, ev = torch.cat(errs_lean[k]), torch.cat(errs_v2[k])
    i = el.argmax()
    print("  %-9s lean %.2e %.2e %.2e | v2 %.2e %.2e %.2e | speed at lean max %.1f, at v2 max %.1f" % (k, el.median(), el.quantile(0.999) if el.numel() < 16e6 else 0, el.max(), ev.median(), ev.quantile(0.999), ev.max(), speeds[i], speeds[ev.argmax()]))
print("samples", speeds.numel(), "speed min", speeds.min().item())
