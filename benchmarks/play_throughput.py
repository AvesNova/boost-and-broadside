"""Benchmark the latency-sensitive one-environment Frontline play path.

The timing includes two scripted decisions, reward/observation construction,
physics, and offscreen rendering. Frame-rate sleeping and event polling are
excluded.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("HEADLESS", "1")

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config.defaults import SHIP_CONFIG
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import frontline_ship_config
from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG
from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-decisions", type=int, default=30)
    parser.add_argument("--timed-decisions", type=int, default=300)
    parser.add_argument("--window-size", type=int, default=900)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    ship_config = frontline_ship_config(SHIP_CONFIG)
    env = TensorEnv(1, ship_config, PLAY_ENV_CONFIG, "cpu")
    env.reset(seed=20260909)
    agent = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
    renderer = GameRenderer(
        ship_config,
        RenderConfig(window_size=args.window_size, fps=30, show_ui=True),
    )

    def decision() -> None:
        action = agent.get_actions(env.state)
        env.step(action)
        renderer.draw_frame(env.state)

    try:
        for _ in range(args.warmup_decisions):
            decision()
        started = time.perf_counter()
        for _ in range(args.timed_decisions):
            decision()
        elapsed = time.perf_counter() - started
    finally:
        renderer.close()

    simulated_seconds = (
        args.timed_decisions * PLAY_ENV_CONFIG.action_repeat * ship_config.dt
    )
    print(
        f"threads={args.threads} decisions={args.timed_decisions} "
        f"ms/decision={elapsed * 1e3 / args.timed_decisions:.3f} "
        f"realtime={simulated_seconds / elapsed:.3f}x"
    )


if __name__ == "__main__":
    main()
