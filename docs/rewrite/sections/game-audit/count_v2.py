import sys, dataclasses, functools, importlib, inspect, pkgutil
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/boost-and-broadside/src")
sys.path.insert(0, "/home/user/boost-and-broadside/docs/rewrite/sections/simplification")
import torch
import regions as R
import bnb.game as G
from bnb.game import api
from bnb.contracts import Action
mods = []
for info in pkgutil.walk_packages(G.__path__, "bnb.game."):
    try: mods.append(importlib.import_module(info.name))
    except Exception as e: pass
wrapped = {}
def wrap(fn, label):
    if fn in wrapped: return wrapped[fn]
    @functools.wraps(fn)
    def w(*a, **k):
        R.STACK.append(label); R.CALLS[label] += 1
        try: return fn(*a, **k)
        finally: R.STACK.pop()
    wrapped[fn] = w
    return w
for m in [m for m in list(sys.modules.values()) if getattr(m, "__name__", "").startswith("bnb.")]:
    for attr, obj in list(vars(m).items()):
        if inspect.isfunction(obj) and obj.__module__.startswith("bnb.game") and not hasattr(obj, "__wrapped__"):
            setattr(m, attr, wrap(obj, f"{obj.__module__.replace('bnb.', '')}.{obj.__qualname__}"))

def run(name, game, B=3, perceive=False, active=None, top=40, **over):
    types = api.ship_types(ship_set=game.ship_set)
    rules = api.rules_of(game, types=types, **over)
    g = torch.Generator().manual_seed(0)
    state = api.reset(rules, types, api.uniforms(rules, B, generator=g), fogged=torch.zeros(B, dtype=torch.bool), active=active)
    N = rules.num_ships
    def act():
        return Action(power=torch.randint(0,3,(B,N),generator=g), stick=torch.randint(0,7,(B,N),generator=g), shoot=torch.randint(0,2,(B,N),generator=g))
    for _ in range(3):
        state, ev = api.step(rules, types, state, act(), api.spread(rules, B, generator=g), api.uniforms(rules, B, generator=g), active)
    a, s, u = act(), api.spread(rules, B, generator=g), api.uniforms(rules, B, generator=g)
    R.reset()
    with torch.no_grad(), R.Counter():
        moved, sweep = api.move(rules, types, state, a, s)
        n_move = sum(v[0] for v in R.EXCLUSIVE.values())
        state2, ev = api.resolve(rules, types, moved, sweep, a, u, active)
        n_res = sum(v[0] for v in R.EXCLUSIVE.values()) - n_move
        n_per = 0
        if perceive:
            out = api.perceive(rules, state2) if active is None else api.perceive(rules, state2, active)
            n_per = sum(v[0] for v in R.EXCLUSIVE.values()) - n_move - n_res
    tot = sum(v[0] for v in R.EXCLUSIVE.values()); views = sum(v[1] for v in R.EXCLUSIVE.values())
    print(f"\n== {name}: N={N} K={rules.bullet_slots} M={rules.num_fields}: move {n_move} + resolve {n_res} + perceive {n_per} = {tot} compute ops (+{views} views)")
    rows = sorted(((k, v[0]) for k, v in R.EXCLUSIVE.items()), key=lambda r: -r[1])
    for k, v in rows[:top]:
        print(f"  {v:>5}  calls {R.CALLS[k]:>4}  {k}")

E = api.PRESETS["real_scale_elimination"]
run("real_scale_elimination 5v5 random map", E, perceive=True)
run("real_scale_elimination 5v5, no fields", dataclasses.replace(E, num_fields=0), top=12)
run("real_scale_elimination 50v50", dataclasses.replace(E, ships_per_team=50), top=6)
run("real_scale_elimination 5v5 active(3,5)", E, active=torch.tensor([[3,5]]*3, dtype=torch.int32), top=8)
run("clover 2v2 fixed layout", api.PRESETS["clover"], top=12)

print("\n\n#### inclusive, 5v5 random map elimination")
def run_incl(game, B=3):
    types = api.ship_types(ship_set=game.ship_set)
    rules = api.rules_of(game, types=types)
    g = torch.Generator().manual_seed(0)
    state = api.reset(rules, types, api.uniforms(rules, B, generator=g), fogged=torch.zeros(B, dtype=torch.bool))
    N = rules.num_ships
    a = Action(power=torch.randint(0,3,(B,N),generator=g), stick=torch.randint(0,7,(B,N),generator=g), shoot=torch.randint(0,2,(B,N),generator=g))
    s, u = api.spread(rules, B, generator=g), api.uniforms(rules, B, generator=g)
    R.reset()
    with torch.no_grad(), R.Counter():
        st, ev = api.step(rules, types, state, a, s, u)
        api.perceive(rules, st)
    for k, v in sorted(R.INCLUSIVE.items(), key=lambda r: -r[1][0]):
        if v[0] >= 15: print(f"  {v[0]:>5}  calls {R.CALLS[k]:>4}  {k}")
    for k in ("game.physics.flight.kick","game.physics.combat.hits","game.physics.medium.medium","game.physics.numerics.softmin","game.physics.step.resolve","game.ship_types.per_ship","game.layout.environment","game.physics.spawning.spawn"):
        print(k, R.OPS[k].most_common(12))
run_incl(E)
