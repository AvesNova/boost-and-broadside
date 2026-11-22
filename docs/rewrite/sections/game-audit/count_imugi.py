import sys, dataclasses, functools, importlib, inspect, pkgutil, time
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/boost-and-broadside/src")
sys.path.insert(0, "/home/user/boost-and-broadside/docs/rewrite/sections/simplification")
import torch
torch.set_num_threads(4)
import regions as R
from bnb.game import api
from bnb.contracts import Action
from bnb.game.handling.characteristics import measure_characteristics
from bnb.controllers.imugi import Imugi
from bnb.controllers import imugi as IM
E = api.PRESETS["real_scale_elimination"]
types = api.ship_types(ship_set=E.ship_set)
rules = api.rules_of(E, types=types)
t = time.time(); ch = measure_characteristics(types, rules); print("characteristics", round(time.time()-t,1), "s")
for f in dataclasses.fields(ch): print("  ", f.name, getattr(ch, f.name).tolist() if hasattr(getattr(ch,f.name),'tolist') else getattr(ch,f.name))
for name in ("imugi", "imugi_combat", "imugi_doctrine", "imugi_frontline", "imugi_memory", "imugi_parameters", "scripted"):
    importlib.import_module(f"bnb.controllers.{name}")
import bnb.streams
prefixes = ("bnb.controllers.imugi", "bnb.controllers.scripted", "bnb.streams")
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
        if inspect.isfunction(obj) and obj.__module__.startswith(prefixes) and not hasattr(obj, "__wrapped__"):
            setattr(m, attr, wrap(obj, f"{obj.__module__.replace('bnb.', '')}.{obj.__qualname__}"))
        elif inspect.isclass(obj) and obj.__module__.startswith(prefixes) and obj.__module__ == m.__name__:
            for an, av in list(vars(obj).items()):
                if inspect.isfunction(av) and not an.startswith("__"):
                    setattr(obj, an, wrap(av, f"{obj.__module__.replace('bnb.', '')}.{obj.__qualname__}.{an}"))
B = 4
g = torch.Generator().manual_seed(0)
state = api.reset(rules, types, api.uniforms(rules, B, generator=g), fogged=torch.zeros(B, dtype=torch.bool))
state, obs, priv = api.perceive(rules, state)
print("obs", {f.name: tuple(getattr(obs, f.name).shape) for f in dataclasses.fields(obs)})
S_ = rules.ships_per_team
def run(label, **kw):
    agent = Imugi(ch, S_, **kw)
    def views(o):  # (B,2,...) -> (2B,...)
        return type(o)(**{f.name: getattr(o, f.name).flatten(0,1) for f in dataclasses.fields(o)})
    carry = agent.init_carry(2*B, torch.device("cpu"))
    st = state; ob = obs
    reset = torch.ones(2*B, dtype=torch.bool)
    for i in range(3):
        key = torch.randint(0, 2**62, (2*B,), generator=g)
        act, carry, _ = agent.step(carry, views(ob), reset, key)
        reset = torch.zeros(2*B, dtype=torch.bool)
    R.reset()
    with torch.no_grad(), R.Counter():
        key = torch.randint(0, 2**62, (2*B,), generator=g)
        act, carry, _ = agent.step(carry, views(ob), reset, key)
    tot = sum(v[0] for v in R.EXCLUSIVE.values())
    print(f"\n== Imugi {label}: {tot} compute ops per call")
    for k, v in sorted(R.INCLUSIVE.items(), key=lambda r: -r[1][0])[:30]:
        print(f"  incl {v[0]:>5} excl {R.EXCLUSIVE[k][0]:>5} calls {R.CALLS[k]:>4}  {k}")
    return act
a = run("default (stored doctrines, memory)")
print("action", a.power.shape, a.power.dtype)
run("no doctrines", doctrines=None)
run("no doctrines, no memory", doctrines=None, memory=False)
