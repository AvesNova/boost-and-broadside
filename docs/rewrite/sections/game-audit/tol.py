import sys, dataclasses
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/boost-and-broadside/src")
import torch
torch.set_num_threads(4)
from bnb.game import api
from bnb.contracts import Action
from bnb.game.physics import combat as _c
_c.contact_order = lambda tau, miss: (tau.float().view(torch.int32).long() << 31) | miss.float().view(torch.int32).long()
E = api.PRESETS["real_scale_elimination"]
types32 = api.ship_types(ship_set=E.ship_set)
types64 = api.ship_types(ship_set=E.ship_set, dtype=torch.float64)
rules = api.rules_of(E, types=types32)
# no-shrink variant
pool = rules.pools[0]
mode = dataclasses.replace(pool.mode, shrink_hold_seconds=1e9, shrink_end_lead_seconds=0.0)
rules_ns = dataclasses.replace(rules, pools=(dataclasses.replace(pool, mode=mode),))
from bnb.game.layout import environment, boundary_fraction
env = environment(rules_ns, 1)
for t in (0, 100, 5398, 5399, 5400):
    print("no-shrink boundary fraction at elapsed", t, boundary_fraction(env, torch.tensor([t])).item())
B = 64
g = torch.Generator().manual_seed(1)
s32 = api.reset(rules_ns, types32, api.uniforms(rules, B, generator=g), fogged=torch.zeros(B, dtype=torch.bool))
def cast(tree, dt):
    out = {}
    for f in dataclasses.fields(tree):
        v = getattr(tree, f.name)
        if dataclasses.is_dataclass(v): out[f.name] = cast(v, dt)
        else: out[f.name] = v.to(dt) if v.is_floating_point() else v.clone()
    return type(tree)(**out)
N = rules.num_ships
def acts(T, sticky=10):
    out = []
    cur = None
    for t in range(T):
        new = torch.stack([torch.randint(0,3,(B,N),generator=g), torch.randint(0,7,(B,N),generator=g), torch.randint(0,2,(B,N),generator=g)], -1)
        redraw = (torch.rand((B,N,1),generator=g) < 1.0/sticky) if cur is not None else torch.ones((B,N,1),dtype=torch.bool)
        cur = torch.where(redraw, new, cur) if cur is not None else new
        out.append(cur.clone())
    return out
T = 150
A = acts(T)
SP = [torch.zeros(B, N, 2) for _ in range(T)]  # no spread: deterministic bullets
U = [api.uniforms(rules, B, generator=g) for _ in range(T)]
def names(): return ["x","y","heading_cos","heading_sin","proper_speed","angle_of_attack","nose_rate","energy","shield"]
# (1) one-step error f32 vs f64 from identical (f32-representable) states, along a f32 trajectory
s = s32
worst = {n: 0.0 for n in names()}; rel = dict(worst)
multi64 = cast(s32, torch.float64)
div = []
flags_equal = True
for t in range(T):
    a = Action(power=A[t][...,0], stick=A[t][...,1], shoot=A[t][...,2])
    n32, e32 = api.step(rules_ns, types32, s, a, SP[t], U[t])
    n64, e64 = api.step(rules_ns, types64, cast(s, torch.float64), a, SP[t].double(), U[t].double())
    alive = n32.ships.alive
    for n in names():
        d = (getattr(n32.ships, n).double() - getattr(n64.ships, n)).abs()[alive]
        if d.numel(): worst[n] = max(worst[n], d.max().item())
    flags_equal &= bool((n32.ships.alive == n64.ships.alive).all()) and bool((n32.bullets.moves_left == n64.bullets.moves_left).all())
    # (2) free-running f64 trajectory vs f32 trajectory
    multi64, _ = api.step(rules_ns, types64, multi64, a, SP[t].double(), U[t].double())
    both = alive & multi64.ships.alive
    dpos = torch.sqrt((n32.ships.x.double()-multi64.ships.x)**2 + (n32.ships.y.double()-multi64.ships.y)**2)[both]
    dv = (n32.ships.proper_speed.double()-multi64.ships.proper_speed).abs()[both]
    de = (n32.ships.energy.double()-multi64.ships.energy).abs()[both]
    dsh = (n32.ships.shield.double()-multi64.ships.shield).abs()[both]
    mism = (n32.ships.alive != multi64.ships.alive).sum().item()
    if t+1 in (1, 5, 10, 30, 60, 90, 150):
        div.append((t+1, dpos.max().item(), dpos.median().item(), dv.max().item(), de.max().item(), dsh.max().item(), mism, int((~alive).sum())))
    s = n32
print("one-step max abs error f32 vs f64 over", T, "ticks x", B, "envs x", N, "ships:")
for n in names(): print(f"  {n:18s} {worst[n]:.3e}")
print("flags equal in every one-step comparison:", flags_equal)
print("free-running f32 vs f64 (tick, max dpos m, median dpos m, max dspeed, max denergy, max dshield, alive mismatches, dead):")
for r in div: print("  ", r)
print("typical magnitudes: |x| ~", s.ships.x.abs().max().item(), "speed ~", s.ships.proper_speed.max().item(), "energy ~", s.ships.energy.max().item())
