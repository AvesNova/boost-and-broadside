import dataclasses, os, sys, copy, torch
sys.path.insert(0, os.path.dirname(__file__))
from boost_and_broadside.config import EnvConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env import physics
from boost_and_broadside.profiles.rl import RL_PROFILE
import integrators as I
base = dataclasses.replace(RL_PROFILE.ship_config, world_size=(4096.0, 4096.0), dt=1/30)
env = TensorEnv(4, base, EnvConfig(num_ships=10, max_bullets=1, max_episode_steps=None, num_fields=0), "cpu")
env.reset(seed=123)
tables = physics.action_tables(base, torch.device("cpu"))
for pw in range(3):
  for tn in (0, 4):
    a = torch.zeros(4, 10, 3, dtype=torch.long); a[..., 0] = pw; a[..., 1] = tn
    s1 = copy.deepcopy(env.state); s1 = I.c3(s1, a, base, tables, 1)
    s2 = copy.deepcopy(env.state); fine = dataclasses.replace(base, dt=1/3840)
    for _ in range(128): s2 = I.c1(s2, a, fine, tables, 1)
    dv = (s1.ship_vel - s2.ship_vel).abs().max().item(); dp = (s1.ship_pos - s2.ship_pos).abs().max().item()
    print(f"power {pw} turn {tn}: max |dv| {dv:.4f} m/s  max |dpos| {dp:.4f} m  dpower {(s1.ship_power-s2.ship_power).abs().max().item():.4f}")
