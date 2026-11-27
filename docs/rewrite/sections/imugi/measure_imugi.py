"""Imugi (v2) by behaviour: compute ops per call (regions.py rule: views excluded) and CPU
time per call, one thread, E environments, one team's views per call.

Each dispatched op is attributed to a behaviour from the Python call stack (function and
line inside bnb/controllers/imugi*.py and bnb/streams.py). Time per behaviour: each op's
own dispatch-to-return time, summed per behaviour, then scaled to the call's wall time
measured without the counter.

usage: OMP_NUM_THREADS=1 PYTHONPATH=.:src uv run --no-sync python measure_imugi.py [E] [warm_ticks]
"""

import collections
import dataclasses
import os
import pickle
import statistics
import sys
import time

import torch

torch.set_num_threads(1)
sys.path.insert(0, "docs/rewrite/sections/simplification")
import regions as R  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from bnb.controllers.imugi import Imugi  # noqa: E402
from bnb.game import api  # noqa: E402
from bnb.game.handling.characteristics import characteristics  # noqa: E402

E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
WARM = int(sys.argv[2]) if len(sys.argv) > 2 else 450
OUT = os.path.dirname(os.path.abspath(__file__))

preset = api.PRESETS["real_scale_elimination"]
types = api.ship_types(ship_set=preset.ship_set)
rules = api.rules_of(preset, types=types)
pool = rules.pools[0]
rules = dataclasses.replace(
    rules,
    pools=(
        dataclasses.replace(
            pool,
            mode=dataclasses.replace(
                pool.mode, shrink_hold_seconds=1e9, shrink_end_lead_seconds=0.0
            ),
        ),
    ),
)
t0 = time.time()
ch = characteristics(types, rules)
print("characteristics", round(time.time() - t0, 1), "s")
S = rules.ships_per_team
N = 2 * S

# --- warm states: Imugi against Imugi for WARM ticks, fog off, no shrink ---
g = torch.Generator().manual_seed(3)
state = api.reset(
    rules, types, api.uniforms(rules, E, generator=g), fogged=torch.zeros(E, dtype=torch.bool)
)
state, obs, _ = api.perceive(rules, state)
agent = Imugi(ch, S)
carry = agent.init_carry(2 * E, torch.device("cpu"))


def views(o):  # (E, 2, ...) -> (2E, ...): both teams' views
    return type(o)(**{f.name: getattr(o, f.name).flatten(0, 1) for f in dataclasses.fields(o)})


from bnb.contracts import Action  # noqa: E402

reset = torch.ones(2 * E, dtype=torch.bool)
t0 = time.time()
with torch.no_grad():
    for tick in range(WARM):
        key = torch.randint(0, 2**62, (2 * E,), generator=g)
        act, carry, _ = agent.step(carry, views(obs), reset, key)
        reset = torch.zeros(2 * E, dtype=torch.bool)
        a = Action(
            power=act.power.reshape(E, N),
            stick=act.stick.reshape(E, N),
            shoot=act.shoot.reshape(E, N),
        )
        state, ev = api.step(
            rules,
            types,
            state,
            a,
            api.spread(rules, E, generator=g),
            api.uniforms(rules, E, generator=g),
        )
        state, obs, _ = api.perceive(rules, state)
        if tick % 150 == 0:
            print("  warm tick", tick, round(time.time() - t0), "s", flush=True)
print("warm", WARM, "ticks in", round(time.time() - t0), "s")

# one team's views (team 0 of every environment) and that team's carry
obs0 = type(obs)(**{f.name: getattr(obs, f.name)[:, 0] for f in dataclasses.fields(obs)})
carry0 = type(carry)(
    **{
        f.name: getattr(carry, f.name).view(E, 2, *getattr(carry, f.name).shape[1:])[:, 0].clone()
        for f in dataclasses.fields(carry)
    }
)
reset0 = torch.zeros(E, dtype=torch.bool)
ships = obs0.ships
alive = ships[:, :S, 0]
from bnb.contracts.obs import SHIP  # noqa: E402

own_alive = (ships[:, :S, SHIP["alive"]] > 0.5).float().mean().item()
enemy_alive = (ships[:, S:, SHIP["alive"]] > 0.5).float().mean().item()
print(
    f"state at tick {WARM}: own alive {own_alive:.2f}, enemy alive {enemy_alive:.2f}, done {float(ev.done.float().mean()):.2f}"
)
torch.save({"obs": obs0, "carry": carry0}, os.path.join(OUT, f"imugi_state_E{E}.pt"))

# --- attribution ---
MODULE_FILES = ("controllers/imugi", "bnb/streams.py")


def frames():
    """Imugi frames, outer to inner: (file tail, function, line)."""
    out = []
    f = sys._getframe(2)
    while f is not None:
        name = f.f_code.co_filename
        if any(m in name for m in MODULE_FILES):
            out.append((os.path.basename(name), f.f_code.co_name, f.f_lineno))
        f = f.f_back
    return out[::-1]


OUTER = {  # a whole function -> behaviour, outermost match wins
    ("imugi_frontline.py", None): "Frontline layer",
    ("imugi.py", "mode_index"): "mode inference and per-view parameters",
    ("imugi.py", "resolved"): "mode inference and per-view parameters",
    ("imugi_memory.py", "remember"): "dead reckoning and last-seen memory",
    ("imugi_memory.py", "forgotten"): "dead reckoning and last-seen memory",
    ("imugi_combat.py", "doctrinal"): "type doctrines",
    ("imugi_combat.py", "advantage_weight"): "type doctrines",
    ("imugi_combat.py", "maneuvers"): "manoeuvres",
    ("imugi_combat.py", "commanded_stick"): "manoeuvres",
    ("imugi.py", "_agent_flags"): "personas: temperature and share",
}
INNER = {  # innermost match wins (after the keyed-draw check)
    "commit": "switch-rate commitments",
    "switch_probability": "switch-rate commitments",
    "ways_round": "switch-rate commitments",
    "in_own_time": "switch-rate commitments",
    "time_unit": "switch-rate commitments",
}
FUNCTION = {
    "choose_target": "target choice",
    "lead": "lead (intercept)",
    "stick_marginal": "stick: turn and level gates",
    "slow": "power: boost and reverse",
    "far": "power: boost and reverse",
    "power_marginal": "power: boost and reverse",
    "steer": "boundary steer",
}


def combat_line(line):
    if line <= 676:
        return "target choice"
    if line <= 686:
        return "lead (intercept)"
    if line <= 694:
        return "power: boost and reverse"
    if line <= 716:
        return "type doctrines"
    if line <= 733:
        return "stick: turn and level gates"
    if line <= 735:
        return "power: boost and reverse"
    if line <= 770:
        return "manoeuvres"
    return "gun gate"


def context_line(line):
    if 148 <= line <= 158:
        return "boundary steer"
    return "observation reading and per-type lookups"


def decide_line(line):  # imugi.py decide()
    if 129 <= line <= 134:
        return "personas: temperature and share"
    if 127 <= line <= 157:
        return "output assembly (mode select, dead ships, uniform rung)"
    return "observation reading and per-type lookups"


def classify(stack):
    for file, function, line in reversed(stack):
        if file == "streams.py" and function == "inverse_cdf":
            return "action sampling"
        if file == "streams.py" or function == "choice_uniforms":
            for f2, fn2, _ in reversed(stack):
                if (f2, fn2) == ("imugi_frontline.py", "_choice"):
                    return "keyed random draws [frontline goal, recovery]"
                if fn2 in (
                    "choose_target",
                    "stick_marginal",
                    "maneuvers",
                    "sample",
                    "_agent_flags",
                ):
                    return f"keyed random draws [{fn2}]"
            return "keyed random draws [?]"
    for file, function, line in stack:
        if (file, None) in OUTER:
            return OUTER[(file, None)]
        if (file, function) in OUTER:
            return OUTER[(file, function)]
    for file, function, line in reversed(stack):
        if function in INNER:
            return INNER[function]
    for file, function, line in reversed(stack):
        if function in FUNCTION:
            return FUNCTION[function]
        if file == "imugi_combat.py" and function == "combat":
            return combat_line(line)
        if file == "imugi_combat.py" and function == "context":
            return context_line(line)
        if file == "imugi.py" and function == "decide":
            return decide_line(line)
        if file == "imugi_parameters.py" and function in ("logistic", "floored", "ramp"):
            continue
    return "other: " + "/".join(f"{f}:{fn}:{ln}" for f, fn, ln in stack[-2:])


COUNT = collections.Counter()
VIEWS = collections.Counter()
SECONDS = collections.Counter()
LOGISTICS = collections.Counter()
TIMING = [False]


class Attribute(TorchDispatchMode):
    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        name = str(func)
        if name in R.METADATA:
            return func(*args, **(kwargs or {}))
        stack = frames()
        row = classify(stack)
        if getattr(func, "is_view", False):
            VIEWS[row] += 1
            return func(*args, **(kwargs or {}))
        COUNT[row] += 1
        if any(fn == "logistic" for _, fn, _ in stack) and name == "aten.sigmoid.default":
            LOGISTICS[row] += 1
        start = time.perf_counter()
        out = func(*args, **(kwargs or {}))
        SECONDS[row] += time.perf_counter() - start
        return out


def call(agent, key):
    return agent.step(carry0, obs0, reset0, key)


def wall(agent, repeats=40):
    keys = [torch.randint(0, 2**62, (E,), generator=g) for _ in range(repeats + 5)]
    with torch.no_grad():
        for k in keys[:5]:
            call(agent, k)
        times = []
        for k in keys[5:]:
            t = time.perf_counter()
            call(agent, k)
            times.append(time.perf_counter() - t)
    return statistics.median(times), min(times)


results = {}
for label, kw in (
    ("full (stored doctrines, memory)", {}),
    ("no doctrines", {"doctrines": None}),
    ("no doctrines, no memory", {"doctrines": None, "memory": False}),
):
    variant = Imugi(ch, S, **kw)
    median, best = wall(variant)
    COUNT.clear()
    VIEWS.clear()
    SECONDS.clear()
    LOGISTICS.clear()
    with torch.no_grad(), Attribute():
        call(variant, torch.randint(0, 2**62, (E,), generator=g))
    counted = dict(COUNT)
    seconds = dict(SECONDS)
    # op-timing pass repeated for a steadier share
    for _ in range(4):
        with torch.no_grad(), Attribute():
            call(variant, torch.randint(0, 2**62, (E,), generator=g))
    total_ops = sum(counted.values())
    total_seconds = sum(SECONDS.values())
    print(
        f"\n== Imugi {label}: E = {E} views, {S} own ships: {total_ops} compute ops "
        f"(+{sum(VIEWS.values())} views); wall median {median * 1e3:.2f} ms (min {best * 1e3:.2f})"
    )
    rows = {}
    for row in sorted(counted, key=lambda r: -counted[r]):
        share = SECONDS[row] / total_seconds
        rows[row] = {
            "ops": counted[row],
            "views": VIEWS[row],
            "share": share,
            "ms": share * median * 1e3,
            "logistics": LOGISTICS[row],
        }
        print(
            f"  {row:<58} {counted[row]:>5} ops  {VIEWS[row]:>4} views  "
            f"{LOGISTICS[row]:>3} logistics  time share {100 * share:5.1f}%  ~{share * median * 1e3:6.2f} ms"
        )
    results[label] = {
        "ops": total_ops,
        "median_ms": median * 1e3,
        "min_ms": best * 1e3,
        "rows": rows,
    }

# both teams in one call (per tick)
both = Imugi(ch, S)
obs_both = views(obs)
with torch.no_grad():
    keys = [torch.randint(0, 2**62, (2 * E,), generator=g) for _ in range(25)]
    rb = torch.zeros(2 * E, dtype=torch.bool)
    for k in keys[:5]:
        both.step(carry, obs_both, rb, k)
    ts = []
    for k in keys[5:]:
        t = time.perf_counter()
        both.step(carry, obs_both, rb, k)
        ts.append(time.perf_counter() - t)
results["both teams, one call (2E views)"] = {"median_ms": statistics.median(ts) * 1e3}
print(f"\nboth teams in one call ({2 * E} views): median {statistics.median(ts) * 1e3:.2f} ms")

with open(os.path.join(OUT, f"imugi_results_E{E}.pkl"), "wb") as f:
    pickle.dump(results, f)
