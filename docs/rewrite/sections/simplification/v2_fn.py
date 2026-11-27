"""Per-function op counts inside v2's game and Imugi: every function and method defined in
bnb.game.* and bnb.controllers.imugi* is a region (exclusive counts)."""
# ruff: noqa: E402, F821  (a design-record script, kept as written)

import sys

sys.argv = [sys.argv[0], sys.argv[1], "ppo", "256", "128"]
src = open("v2_ops.py").read()
src = src[: src.index("with torch.no_grad():")]
exec(compile(src, "v2_ops_head", "exec"))
import functools
import importlib
import inspect
import pkgutil

import bnb.game as G

mods = []
for info in pkgutil.walk_packages(G.__path__, "bnb.game."):
    try:
        mods.append(importlib.import_module(info.name))
    except Exception:
        pass
for name in (
    "imugi",
    "imugi_combat",
    "imugi_doctrine",
    "imugi_frontline",
    "imugi_memory",
    "imugi_parameters",
    "scripted",
):
    mods.append(importlib.import_module(f"bnb.controllers.{name}"))
prefixes = ("bnb.game", "bnb.controllers.imugi", "bnb.controllers.scripted")
wrapped = {}


def wrap(fn, label):
    if fn in wrapped:
        return wrapped[fn]

    @functools.wraps(fn)
    def w(*a, **k):
        R.STACK.append(label)
        R.CALLS[label] += 1
        try:
            return fn(*a, **k)
        finally:
            R.STACK.pop()

    wrapped[fn] = w
    return w


all_mods = [m for m in list(sys.modules.values()) if getattr(m, "__name__", "").startswith("bnb.")]
for m in all_mods:
    for attr, obj in list(vars(m).items()):
        if (
            inspect.isfunction(obj)
            and obj.__module__.startswith(prefixes)
            and not hasattr(obj, "__wrapped__")
        ):
            setattr(m, attr, wrap(obj, f"{obj.__module__.replace('bnb.', '')}.{obj.__qualname__}"))
        elif (
            inspect.isclass(obj)
            and obj.__module__.startswith(prefixes)
            and obj.__module__ == m.__name__
        ):
            for an, av in list(vars(obj).items()):
                if inspect.isfunction(av) and not an.startswith("__"):
                    setattr(
                        obj,
                        an,
                        wrap(av, f"{obj.__module__.replace('bnb.', '')}.{obj.__qualname__}.{an}"),
                    )
with torch.no_grad():
    rollout._tick()
    R.reset()
    with R.Counter():
        rollout._tick()


def show(prefix, title, n=28):
    rows = [(k, v[0]) for k, v in R.EXCLUSIVE.items() if k.startswith(prefix)]
    total = sum(v for _, v in rows)
    print(f"\n== {title}: {total} ops exclusive in these functions ==")
    for k, v in sorted(rows, key=lambda r: -r[1])[:n]:
        print(f"  {v:>5}  calls {R.CALLS[k]:>4}  {k}")


show("game.", "game")
show("controllers.", "Imugi / scripted (2 calls per tick)")
