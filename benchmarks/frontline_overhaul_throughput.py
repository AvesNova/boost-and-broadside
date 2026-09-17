"""Reproducible CPU Frontline timing; run against both source trees with PYTHONPATH.

Includes field physics, collisions, objectives and optionally the scripted agent,
team visibility and observation assembly. No rendering or frame-rate sleeping.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config.defaults import SHIP_CONFIG
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import frontline_ship_config
from boost_and_broadside.env.observation import observation_from_state
from boost_and_broadside.env.perception import team_visibility_from_state
from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG


@torch.inference_mode()
def measure(batch: int, decisions: int) -> list[dict]:
    rows = []
    for ships in (8, 10):
        config = replace(PLAY_ENV_CONFIG, num_ships=ships)
        ship = frontline_ship_config(SHIP_CONFIG)
        for mode in ("simulation", "scripted_observed"):
            env = TensorEnv(batch, ship, config, "cpu")
            env.reset(seed=123)
            agent = StochasticScriptedAgent(ship, StochasticAgentConfig())
            action = torch.zeros((batch, ships, 3), dtype=torch.long)
            action[:, :, 2] = 1
            samples = []
            for repeat in range(4):
                start = time.perf_counter()
                for _ in range(decisions):
                    if mode == "scripted_observed":
                        sight = team_visibility_from_state(env.state, ship, config)
                        action = agent.get_actions(env.state, sight.ship)
                    env.step(action)
                    if mode == "scripted_observed":
                        observation_from_state(env.state, ship)
                if repeat:
                    samples.append(1000 * (time.perf_counter() - start) / decisions)
            state_bytes = sum(v.numel() * v.element_size() for v in vars(env.state).values())
            rows.append(
                dict(
                    ships=ships,
                    mode=mode,
                    batch=batch,
                    samples_ms=samples,
                    median_ms=statistics.median(samples),
                    state_bytes=state_bytes,
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--decisions", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = dict(
        torch_version=torch.__version__,
        threads=1,
        device="cpu",
        results=measure(args.batch, args.decisions),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
