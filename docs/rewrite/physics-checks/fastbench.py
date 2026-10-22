"""Fast accuracy harness: small batch, cached reference, one process per run.

usage: fastbench.py REF CAND[,CAND...] [--fields N] [--rates 30,60] [--seconds 2] [--set k=v ...]
Candidates/refs are names in MODELS. Env flags like MIDSLIP/NOSE still apply.
"""
import argparse, dataclasses, hashlib, json, os, sys, time
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(__file__))
import torch
torch.set_num_threads(1)
from boost_and_broadside.config import EnvConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env import physics
from boost_and_broadside.profiles.rl import RL_PROFILE
import integrators as I, nose_alt

MODELS = {"c0": (I.c0, 1), "c1": (I.c1, 1), "c1k2": (I.c1, 2), "c3k2": (I.c3, 2),
          "alt": (nose_alt.step, 1), "g": (nose_alt.g_step, 1), "gk2": (nose_alt.g_step, 2)}
CACHE = os.path.join(os.path.dirname(__file__), ".refcache")
B, N, HOLD = 32, 10, 0.1

def setup(a):
    base = dataclasses.replace(RL_PROFILE.ship_config, world_size=(4096.0, 4096.0),
                               field_integrator="two_step", field_integration_substeps=1)
    over = {}
    for kv in a.set:
        k, v = kv.split("="); over[k] = float(v)
    base = dataclasses.replace(base, **over)
    tables = physics.action_tables(base, torch.device("cpu"))
    n_hold = int(round(a.seconds / HOLD))
    g = torch.Generator().manual_seed(1)
    sched = torch.stack([torch.randint(0, tables.thrust.shape[0], (n_hold, B, N), generator=g),
                         torch.randint(0, tables.turn_side.shape[0], (n_hold, B, N), generator=g),
                         torch.zeros(n_hold, B, N, dtype=torch.long)], -1)
    return base, tables, sched

def fly(args):
    a, name, hz, extra = args
    base, tables, sched = setup(a)
    fn, K = MODELS[name]
    cfg = dataclasses.replace(base, dt=1.0 / hz, **extra)
    env = TensorEnv(B, cfg, EnvConfig(num_ships=N, max_bullets=1, max_episode_steps=None, num_fields=a.fields), "cpu")
    env.reset(seed=123)
    s = env.state
    e0 = 0.5 * (s.ship_local_index * s.ship_vel.abs()) ** 2 + cfg.power_speed_constant * s.ship_power
    for h in range(sched.shape[0]):
        for _ in range(int(round(HOLD * hz))):
            s = fn(s, sched[h], cfg, tables, K)
    e1 = 0.5 * (s.ship_local_index * s.ship_vel.abs()) ** 2 + cfg.power_speed_constant * s.ship_power
    return name, hz, s.ship_pos, s.ship_attitude, ((e1 - e0) / e0).abs().mean().item()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref"); p.add_argument("cands")
    p.add_argument("--fields", type=int, default=0); p.add_argument("--rates", default="30,60,120")
    p.add_argument("--seconds", type=float, default=2.0); p.add_argument("--refhz", type=int, default=1920)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    t0 = time.time()
    flags = {k: os.environ.get(k) for k in ("MIDSLIP", "NOSE", "TAU", "RESERVE")}
    key = hashlib.md5(json.dumps([a.ref, a.fields, a.seconds, a.refhz, a.set, flags, B]).encode()).hexdigest()[:12]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, key + ".pt")
    jobs = [(a, c, int(hz), {}) for c in a.cands.split(",") for hz in a.rates.split(",")]
    have_ref = os.path.exists(path)
    if not have_ref:
        jobs.insert(0, (a, a.ref, a.refhz, {"field_integrator": "midpoint", "field_integration_substeps": 4}))
    with mp.get_context("fork").Pool(4) as pool:
        out = pool.map(fly, jobs)
    if not have_ref:
        _, _, rp, ra, _ = out.pop(0); torch.save((rp, ra), path)
    rp, ra = torch.load(path)
    W = 4096.0
    print(f"ref={a.ref}@{a.refhz}Hz fields={a.fields} seconds={a.seconds} set={a.set} flags={ {k: v for k, v in flags.items() if v} }")
    print(f"{'cand':>6} {'Hz':>4} {'pos mean m':>10} {'pos p99 m':>9} {'head deg':>8} {'|dE|/E':>9}")
    for name, hz, pos, att, de in out:
        d = pos - rp
        e = torch.complex((d.real + W / 2) % W - W / 2, (d.imag + W / 2) % W - W / 2).abs()
        hd = torch.angle(att * torch.conj(ra)).abs().rad2deg().mean()
        print(f"{name:>6} {hz:>4} {e.mean():10.3f} {torch.quantile(e.flatten(), .99):9.3f} {hd:8.3f} {de:9.2e}")
    print(f"[{time.time() - t0:.0f} s, reference {'cached' if have_ref else 'computed'}]")

if __name__ == "__main__":
    main()
