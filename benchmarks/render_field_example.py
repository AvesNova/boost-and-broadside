"""Render a deterministic field-enabled Frontline map for visual review.

Example:
    uv run python benchmarks/render_field_example.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("HEADLESS", "1")

import pygame

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import FRONTLINE_WORLD_SIZE
from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG
from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--num-fields", type=int, default=PLAY_ENV_CONFIG.num_fields)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/internal/frontline-field-example-seed-20260909.png"),
    )
    args = parser.parse_args()

    ship_config = ShipConfig(world_size=FRONTLINE_WORLD_SIZE)
    env_config = replace(PLAY_ENV_CONFIG, num_fields=args.num_fields)
    env = TensorEnv(1, ship_config, env_config, "cpu")
    env.reset(seed=args.seed)
    renderer = GameRenderer(
        ship_config,
        RenderConfig(window_size=args.window_size, show_ui=False),
    )
    try:
        surface = renderer.draw_frame(env.state)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(surface, args.output)
    finally:
        renderer.close()
    print(f"wrote {args.output} (seed={args.seed}, fields={args.num_fields})")


if __name__ == "__main__":
    main()
