"""Per-tick CPU cost of each kinematics candidate, and of a whole env tick for context."""
import copy, dataclasses, os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__))
from boost_and_broadside.config import EnvConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env import physics
from boost_and_broadside.profiles.rl import RL_PROFILE
import integrators as I
import nose_alt
torch.set_num_threads(int(os.environ.get("THREADS", "4")))
FIELDS = int(os.environ.get("FIELDS", "10"))
B, N = 1024, 10
cfg = dataclasses.replace(RL_PROFILE.ship_config, world_size=(4096.0, 4096.0))
env = TensorEnv(B, cfg, EnvConfig(num_ships=N, max_bullets=10, max_episode_steps=None, num_fields=FIELDS), "cpu")
env.reset(seed=1)
tables = physics.action_tables(cfg, torch.device("cpu"))
a = torch.stack([torch.randint(0, 3, (B, N)), torch.randint(0, tables.turn_side.shape[0], (B, N)), torch.randint(0, 2, (B, N))], -1)
def bench(fn, reps=30):
    s = copy.deepcopy(env.state)
    for _ in range(3): s = fn(s)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); s = fn(s); ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2] * 1e3
full = bench(lambda s: (setattr(env, "state", s), env.step(a))[0] or env.state)
print(f"fields={FIELDS}  whole env tick (ships, bullets, collisions): {full:.2f} ms")
for name, fn, K in (("C0 today", I.c0, 1), ("C1 K=1", I.c1, 1), ("C1 K=2", I.c1, 2), ("g-cmd K=1", nose_alt.g_step, 1), ("g-cmd K=2", nose_alt.g_step, 2)):
    print(f"  {name:>8}: {bench(lambda s: fn(s, a, cfg, tables, K)):.2f} ms per tick")
