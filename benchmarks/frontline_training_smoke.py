"""Small eager CPU PPO comparison; pass total ship count (8 or 10).

Run this same file with PYTHONPATH pointing at each revision's src directory.
Four updates, 512 environment decisions, seed 77, one thread; includes collection,
BC labels, auxiliary prediction, evaluation, optimizer work and checkpoint I/O.
This is an end-to-end smoke benchmark, not a GPU capacity estimate.
"""

import json
import sys
import tempfile
import time
from dataclasses import replace

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EloEvalConfig, ScaleConfig
from boost_and_broadside.config.resolve import resolve_profile
from boost_and_broadside.config.schedule import constant
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.ppo import PPOTrainer


def main():
    torch.set_num_threads(1)
    torch.manual_seed(77)
    r = resolve_profile(PROFILES["rl"])
    c = r.train_config
    c = replace(
        c,
        scales=(
            ScaleConfig(
                env_config=replace(c.scales[0].env_config, num_ships=int(sys.argv[1])), num_envs=8
            ),
        ),
        num_steps=16,
        num_minibatches=2,
        rollouts_per_update=1,
        microbatch_tokens=None,
        total_timesteps=512,
        checkpoint_dir=tempfile.mkdtemp(prefix="shield-train-"),
        schedule=replace(
            c.schedule,
            num_epochs=constant(1),
            league_fraction=constant(0),
            checkpoint_interval=constant(0),
        ),
        elo_eval=EloEvalConfig(2, 1, 4.0, 1000.0, 100),
        live_reference_probabilities=(),
        league_slots=1,
    )
    m = replace(r.model_config, d_model=32, n_heads=2, n_yemong_blocks=1)
    t = PPOTrainer(
        c,
        m,
        r.ship_config,
        device="cpu",
        compile_mode=None,
        use_wandb=False,
        scripted_agent=StochasticScriptedAgent(r.ship_config, StochasticAgentConfig()),
    )
    metrics = []
    t._enqueue_log = lambda data, step: metrics.append(
        {k: v for k, v in data.items() if k in ("perf/sps", "perf/ship_tps", "loss/total")}
    )
    start = time.perf_counter()
    t.train()
    elapsed = time.perf_counter() - start
    print(
        "BENCHMARK",
        json.dumps(
            dict(
                ships=int(sys.argv[1]),
                seconds=elapsed,
                steps=512,
                steps_per_second=512 / elapsed,
                metrics=metrics,
            )
        ),
    )


if __name__ == "__main__":
    main()
