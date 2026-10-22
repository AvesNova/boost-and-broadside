"""Flight integration error vs physics tick, using the real env physics (no fields).

Same random stick schedule (held 0.1 s per command, defined in seconds) flown at several
tick rates; positions and headings compared with a 960 Hz reference at 0.1 s marks.
"""
import copy, dataclasses, sys
import torch
from boost_and_broadside.config import EnvConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env import physics
from boost_and_broadside.profiles.rl import RL_PROFILE

torch.manual_seed(0)
base = RL_PROFILE.ship_config
B, N, SECONDS, HOLD = 256, 10, 3.0, 0.1
env_cfg = EnvConfig(num_ships=N, max_bullets=1, max_episode_steps=None, num_fields=0)
tables = physics.action_tables(base, torch.device("cpu"))
n_power, n_turn = tables.thrust.shape[0], tables.turn_side.shape[0]
n_hold = int(round(SECONDS / HOLD))
g = torch.Generator().manual_seed(1)
import os
MODE=os.environ.get('SCHED','both')
pw=torch.randint(0, n_power, (n_hold, B, N), generator=g)
tn=torch.randint(0, n_turn, (n_hold, B, N), generator=g)
if MODE=='turn': pw=torch.full_like(pw, int(os.environ.get('PW','1')))
if MODE=='thrust': tn=torch.zeros_like(tn)
sched = torch.stack([pw, tn,
                     torch.zeros(n_hold, B, N, dtype=torch.long)], -1)

def fly(hz):
    cfg = dataclasses.replace(base, dt=1.0 / hz)
    env = TensorEnv(B, cfg, env_cfg, "cpu")
    env.reset(seed=123)
    s = env.state
    ticks_per_hold = int(round(HOLD * hz))
    marks = []
    gen = torch.Generator().manual_seed(2)
    for h in range(n_hold):
        for _ in range(ticks_per_hold):
            s = KIN(s, sched[h], cfg, tables)
        marks.append((s.ship_pos.clone(), s.ship_attitude.clone(), s.ship_vel.clone()))
    return marks, cfg.world_size

import os, variants
KIN = variants.update_kinematics_sub if os.environ.get("SUB") else variants.update_kinematics
REFK = KIN; KIN = variants.update_kinematics; ref, ws = fly(960); KIN = REFK
print(f"{'rate':>6} {'pos err mean m (1s / 3s)':>26} {'pos err p99 m (3s)':>19} {'heading err mean deg (3s)':>26} {'speed err mean m/s':>19}")
for hz in (30, 60):
    out, _ = fly(hz)
    def perr(i):
        d = out[i][0] - ref[i][0]
        w, hgt = ws
        d = torch.complex((d.real + w/2) % w - w/2, (d.imag + hgt/2) % hgt - hgt/2)
        return d.abs()
    e1, e3 = perr(9), perr(n_hold - 1)
    ang = torch.angle(out[-1][1] * torch.conj(ref[-1][1])).abs().rad2deg()
    sp = (out[-1][2].abs() - ref[-1][2].abs()).abs()
    print(f"{hz:>4}Hz {e1.mean():12.3f} / {e3.mean():8.3f} {torch.quantile(e3.flatten(), .99):19.3f} {ang.mean():26.3f} {sp.mean():19.3f}")
