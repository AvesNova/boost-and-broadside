"""Fixed-grid T-vs-S win rates for one v1 checkpoint, env + physics from the checkpoint.

Writes one JSON line per matchup (durable per batch). Games run in chunks of at
most --chunk parallel environments (memory bound at large ship counts). Ties are
split into capped (episode hit max_episode_steps) and other (mutual destruction).
Usage:
  fan_eval.py CKPT OUT.jsonl --trained 8 --scripted 8 10 12 14 16 --games 1024 --chunk 1024
"""

import argparse
import json
import math
import time
from dataclasses import replace

import torch

from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.config.defaults import MODEL_CONFIG, SHIP_CONFIG
from boost_and_broadside.evaluation.agents import resolve_agent_spec
from boost_and_broadside.evaluation.environment import (
    create_evaluation_env,
    resolve_evaluation_environment,
)
from boost_and_broadside.evaluation.match import MatchRunner
from boost_and_broadside.train.rl.checkpoint_schema import load_checkpoint_payload

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("out")
p.add_argument("--trained", type=int, default=4)
p.add_argument("--scripted", type=int, nargs="+", default=[4, 5, 6, 7, 8])
p.add_argument("--games", type=int, default=1024)
p.add_argument("--chunk", type=int, default=1024)
p.add_argument("--repeats", type=int, default=1)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--physics", choices=["checkpoint", "main"], default="checkpoint")
a = p.parse_args()

payload = load_checkpoint_payload(a.ckpt, map_location="cpu")
ship = ShipConfig(**payload["ship_config"]) if a.physics == "checkpoint" else SHIP_CONFIG
base_env = EnvConfig(**payload["env_config"])
torch.manual_seed(a.seed)
trained = resolve_agent_spec(
    a.ckpt, ship, MODEL_CONFIG, "cuda", num_ships=2, allow_config_drift=True
)
scripted = resolve_agent_spec("scripted", ship, MODEL_CONFIG, "cuda")
base_env, field_map = resolve_evaluation_environment(base_env, (trained, scripted))
print("env", base_env, "\nship min_damage_frac", ship.bullet_min_damage_frac, flush=True)
dev = torch.device("cuda")


def run_chunk(n0, n1, num_envs):
    num_ships = n0 + n1
    env = create_evaluation_env(
        num_envs, ship, replace(base_env, num_ships=num_ships), dev, field_map_config=field_map
    )
    results = torch.zeros(num_envs, dtype=torch.int32, device=dev)
    lengths = torch.zeros(num_envs, dtype=torch.int64, device=dev)
    capped = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    finished = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    runner = MatchRunner(
        env,
        [trained, scripted],
        team0_index=torch.zeros(num_envs, dtype=torch.long, device=dev),
        team1_index=torch.ones(num_envs, dtype=torch.long, device=dev),
        ship_config=ship,
        num_ships=num_ships,
    )
    runner.init_hidden()
    opts = {"team_sizes": (n0, n1)}
    env.reset(options=opts)
    while not finished.all():
        dones, truncated = runner.step()
        done_any = dones | truncated
        new = done_any & ~finished
        if new.any():
            lengths[new] = env.state.step_count[new].long()
            alive, team = env.state.ship_alive, env.state.ship_team_id
            t0a = (alive & (team == 0)).any(dim=1)
            t1a = (alive & (team == 1)).any(dim=1)
            w0 = new & t0a & ~t1a
            w1 = new & t1a & ~t0a
            results[w0] = 0
            results[w1] = 1
            results[new & ~w0 & ~w1] = 2
            capped |= new & truncated & ~dones
            finished |= new
        runner.reset_finished(done_any, options=opts)
    r = results.cpu()
    return (
        int((r == 0).sum()),
        int((r == 1).sum()),
        int((r == 2).sum()),
        int(capped.sum()),
        lengths.cpu().tolist(),
    )


with open(a.out, "a") as f:
    for rep in range(a.repeats):
        for s in a.scripted:
            t0 = time.time()
            trained.agent._num_ships = a.trained + s
            w = lost = t = cap = 0
            lens = []
            left = a.games
            while left > 0:
                k = min(a.chunk, left)
                cw, cl, ct, cc, cle = run_chunk(a.trained, s, k)
                w += cw
                lost += cl
                t += ct
                cap += cc
                lens += cle
                left -= k
            n = w + lost + t
            rec = dict(
                ckpt=a.ckpt,
                physics=a.physics,
                trained=a.trained,
                scripted=s,
                repeat=rep,
                wins=w,
                losses=lost,
                ties=t,
                capped=cap,
                games=n,
                win_rate=w / n,
                se=math.sqrt(w / n * (1 - w / n) / n),
                mean_steps=sum(lens) / n,
                max_steps=max(lens),
                max_episode_steps=base_env.max_episode_steps,
                seconds=time.time() - t0,
            )
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(
                f"{a.trained}v{s} rep{rep}: {w}/{n} = {w / n:.3f} ±{1.96 * rec['se']:.3f}  "
                f"ties {t} (capped {cap})  "
                f"steps {rec['mean_steps']:.0f}/{rec['max_steps']}  {rec['seconds']:.0f}s",
                flush=True,
            )
