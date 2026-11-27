"""The game's own values for tests, read from the code, never copied into a test (lean/README.md,
"Tests read the code's values"): a test that needs the vision range, the gun range or the top
speed reads it here, so a design change moves the test's thresholds with it."""

import torch

from lean.game import core


def game_value(name: str, config: core.GameConfig | None = None, ship_type: int = 0) -> float:
    """One ship type's game constant `name` (core.ship_constants: e.g. "vision_range",
    "gun_range", "cruise_speed" (the top level speed), "stall_speed", "corner_speed",
    "max_shield", "hit_damage") in a run of `config` (default GameConfig())."""
    config = core.GameConfig() if config is None else config
    constants = core.ship_constants(config, torch.tensor([[ship_type]]))
    return float(torch.as_tensor(constants[name]).flatten()[0])
