"""Accuracy, energy and cost of the integrator candidates (real env physics, CPU torch)."""
import dataclasses, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from boost_and_broadside.config import EnvConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env import physics
from boost_and_broadside.profiles.rl import RL_PROFILE
import integrators as I
import nose_alt

torch.manual_seed(0)
FIELDS = int(os.environ.get("FIELDS", "0"))
ENERGY = bool(os.environ.get("ENERGY"))
B, N, SECONDS, HOLD = 128, 10, 3.0, 0.1
base = RL_PROFILE.ship_config
base = dataclasses.replace(base, world_size=(4096.0, 4096.0), field_integrator="two_step", field_integration_substeps=1)
if os.environ.get("BIGPOWER"):
    base = dataclasses.replace(base, max_power=1e7)
if os.environ.get("NOCAP"):
    base = dataclasses.replace(base, max_lateral_accel=1e6)
if os.environ.get("NORATE"):
    base = dataclasses.replace(base, max_slip_rate=100.0)
if ENERGY:
    base = dataclasses.replace(base, zero_slip_drag_coeff=0.0, normal_slip_drag_coeff=0.0, stall_drag_coeff=0.0, passive_power_gain=0.0)
env_cfg = EnvConfig(num_ships=N, max_bullets=1, max_episode_steps=None, num_fields=FIELDS)
tables = physics.action_tables(base, torch.device("cpu"))
n_hold = int(round(SECONDS / HOLD))
g = torch.Generator().manual_seed(1)
sched = torch.stack([torch.randint(0, tables.thrust.shape[0], (n_hold, B, N), generator=g),
                     torch.randint(0, tables.turn_side.shape[0], (n_hold, B, N), generator=g),
                     torch.zeros(n_hold, B, N, dtype=torch.long)], -1)

def energy(s, cfg):
    return 0.5 * (s.ship_local_index * s.ship_vel.abs()) ** 2 + cfg.power_speed_constant * s.ship_power

def fly(fn, hz, K=1, **over):
    cfg = dataclasses.replace(base, dt=1.0 / hz, **over)
    env = TensorEnv(B, cfg, env_cfg, "cpu")
    env.reset(seed=123)
    s = env.state
    e0 = energy(s, cfg).clone()
    tph = int(round(HOLD * hz))
    in_field = 0.0
    for h in range(n_hold):
        for _ in range(tph):
            s = fn(s, sched[h], cfg, tables, K)
            in_field += float(((s.ship_local_index - 1).abs() > 1e-3).float().mean())
    return s.ship_pos.clone(), s.ship_attitude.clone(), (energy(s, cfg) - e0) / e0, in_field / (n_hold * tph)

REF = nose_alt.g_step if os.environ.get("GCMD") else (nose_alt.step if os.environ.get("ALT") else I.c1)
ref_pos, ref_att, _, frac = fly(REF, 3840, 1, field_integrator="midpoint", field_integration_substeps=4)
W = base.world_size[0]
def err(p):
    d = p - ref_pos
    return torch.complex((d.real + W/2) % W - W/2, (d.imag + W/2) % W - W/2).abs()
print(f"fields={FIELDS} energy_test={ENERGY} share of ship-ticks inside a field: {frac:.2f}")
print(f"{'candidate':>34} {'Hz':>4} {'pos mean m':>10} {'pos p99 m':>9} {'head deg':>8} {'|dE|/E mean':>11}")
CANDS = os.environ.get("CANDS", "c0,c1k1,c1k2,c3k1,c3k2").split(",")
ALL = {"c0": ("C0 today", I.c0, 1, {}), "c1k1": ("C1 symmetric, nose per half", I.c1, 1, {}),
       "c1k2": ("C1 + K=2 substeps", I.c1, 2, {}), "c1k4": ("C1 + K=4 substeps", I.c1, 4, {}),
       "c3k1": ("C3 exact flow K=1", I.c3, 1, {}), "c3k2": ("C3 exact flow K=2", I.c3, 2, {}),
       "c3k4": ("C3 exact flow K=4", I.c3, 4, {}),
       "altk1": ("alt nose, split K=1", nose_alt.step, 1, {}), "altk2": ("alt nose, split K=2", nose_alt.step, 2, {}),
       "gk1": ("g-command, split K=1", nose_alt.g_step, 1, {}), "gk2": ("g-command, split K=2", nose_alt.g_step, 2, {})}
cands = [ALL[c] for c in CANDS]
if FIELDS:
    cands.insert(1, ("C0 today, midpoint fields", I.c0, 1, {"field_integrator": "midpoint"}))
for name, fn, K, over in cands:
    for hz in (30, 60, 120):
        p, a, dE, _ = fly(fn, hz, K, **over)
        e = err(p)
        hd = torch.angle(a * torch.conj(ref_att)).abs().rad2deg().mean()
        print(f"{name:>34} {hz:>4} {e.mean():10.3f} {torch.quantile(e.flatten(), .99):9.3f} {hd:8.3f} {dE.abs().mean():11.2e}")
