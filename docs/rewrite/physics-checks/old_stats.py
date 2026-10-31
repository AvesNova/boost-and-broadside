"""Headline stats of today's flight model (real torch env physics, RL profile ship, no fields)."""
import dataclasses, math, torch
from boost_and_broadside.config import EnvConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env import physics
from boost_and_broadside.profiles.rl import RL_PROFILE
cfg = dataclasses.replace(RL_PROFILE.ship_config, world_size=(65536.0, 65536.0))
tables = physics.action_tables(cfg, torch.device("cpu"))
HZ = round(1 / cfg.dt)

def fly(speed, power, act, seconds):
    env = TensorEnv(1, cfg, EnvConfig(num_ships=2, max_bullets=1, max_episode_steps=None, num_fields=0), "cpu")
    env.reset(seed=0)
    s = env.state
    s.ship_vel = torch.full_like(s.ship_vel, complex(speed, 0.0)); s.ship_attitude = torch.ones_like(s.ship_attitude)
    s.ship_slip = torch.zeros_like(s.ship_slip); s.ship_slip_rate = torch.zeros_like(s.ship_slip_rate)
    s.ship_power = torch.full_like(s.ship_power, float(power))
    a = torch.tensor([[act + [0], act + [0]]])
    sp, rate, pw = [], [], []
    for _ in range(int(seconds * HZ)):
        v0 = s.ship_vel[0, 0].clone()
        s = physics._update_kinematics(s, a, cfg, tables)
        v1 = s.ship_vel[0, 0]
        sp.append(abs(v1).item()); rate.append(math.atan2((v1 * v0.conj()).imag.item(), (v1 * v0.conj()).real.item()) * HZ); pw.append(s.ship_power[0, 0].item())
    return sp, rate, pw

def steady(x, sec=3):
    n = int(sec * HZ); return sum(x[-n:]) / n

print(f"dt {cfg.dt:.4f} s; max_lateral_accel {cfg.max_lateral_accel}; max_lift_coeff {cfg.max_lift_coeff}; min_speed {cfg.min_speed}; slip_fade_speed {cfg.slip_fade_speed}")
print(f"corner speed sqrt(a/C_L) {math.sqrt(cfg.max_lateral_accel / cfg.max_lift_coeff):.0f} m/s; instantaneous max rate {math.degrees(cfg.max_lateral_accel / math.sqrt(cfg.max_lateral_accel / cfg.max_lift_coeff)):.0f} deg/s; min instantaneous radius 1/C_Lmax {1 / cfg.max_lift_coeff:.0f} m")
sp, r, pw = fly(100.0, 50.0, [0, 0], 40.0); print(f"coast straight: {steady(sp):.1f} m/s, power {pw[-1]:.0f}")
sp, r, pw = fly(100.0, 100.0, [1, 0], 40.0); print(f"boost straight (sustained, power-limited): {steady(sp):.1f} m/s, power {pw[-1]:.1f}")
sp, r, pw = fly(160.0, 100.0, [1, 0], 3.0); print(f"boost straight while the store lasts (from 160): {max(sp):.0f} m/s peak")
for name, act in (("coast sharp", [0, 4]), ("coast normal", [0, 2]), ("boost sharp", [1, 4]), ("reverse sharp", [2, 4])):
    sp, r, pw = fly(100.0, 100.0, act, 40.0)
    u, w = steady(sp), steady(r)
    print(f"{name}: sustained {u:.1f} m/s, {math.degrees(w):.0f} deg/s, radius {u / max(abs(w), 1e-9):.0f} m, power {pw[-1]:.0f}")
# instantaneous rate vs speed (first 0.5 s from straight, sharp, coast): peak rate
for u0 in (40, 60, 100, 140):
    sp, r, pw = fly(float(u0), 100.0, [0, 4], 0.6)
    print(f"  from {u0} m/s, sharp: peak rate {math.degrees(max(map(abs, r))):.0f} deg/s, time to 90% of it {next(i for i, x in enumerate(r) if abs(x) >= 0.9 * max(map(abs, r))) / HZ:.2f} s")
