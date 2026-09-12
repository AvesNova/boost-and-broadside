"""Benchmark the latency-sensitive one-environment Frontline play path.

The timing includes shared scripted analysis, perception, physics, and offscreen
rendering. Frame-rate sleeping and event polling are excluded.
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
from boost_and_broadside.env.perception import team_visibility_from_state
from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG
from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig, VisionMode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-decisions", type=int, default=30)
    parser.add_argument("--timed-decisions", type=int, default=300)
    parser.add_argument("--window-size", type=int, default=900)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--full-view", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    ship_config = frontline_ship_config(SHIP_CONFIG)
    env = TensorEnv(1, ship_config, PLAY_ENV_CONFIG, "cpu")
    env.reset(seed=20260909)
    agent = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
    renderer = GameRenderer(
        ship_config,
        RenderConfig(
            window_size=args.window_size,
            fps=30,
            show_ui=True,
            vision_mode=VisionMode.FULL if args.full_view else VisionMode.TEAM_0,
        ),
    )
    visibility = team_visibility_from_state(env.state, ship_config, PLAY_ENV_CONFIG)

    def decision() -> None:
        nonlocal visibility
        action = agent.get_actions(env.state, visibility.ship)
        env.step(action)
        visibility = team_visibility_from_state(env.state, ship_config, PLAY_ENV_CONFIG)
        renderer.draw_frame(env.state, visibility=visibility)

    try:
        for _ in range(args.warmup_decisions):
            decision()
        started = time.perf_counter()
        for _ in range(args.timed_decisions):
            decision()
        elapsed = time.perf_counter() - started
    finally:
        renderer.close()

    simulated_seconds = args.timed_decisions * PLAY_ENV_CONFIG.action_repeat * ship_config.dt
    print(
        f"threads={args.threads} decisions={args.timed_decisions} "
        f"view={'full' if args.full_view else 'team0'} "
        f"ms/decision={elapsed * 1e3 / args.timed_decisions:.3f} "
        f"realtime={simulated_seconds / elapsed:.3f}x"
    )


if __name__ == "__main__":
    main()
