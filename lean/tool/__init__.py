"""A small local web tool over the game's own code: ship tuning charts and the flight sandbox.
`python -m lean.tool` (needs the `tools` extra); see lean/tool/NOTES.md."""

import os

from lean.game import core

# The game's rules as the tool flies and charts them (one source for the sandbox, the flown
# checks and the charts): a fixed arena (no shrinking zone, no clock); BNB_POST_STALL=0 in the
# server's environment turns the post-stall regime off (GameConfig.post_stall), BNB_FIELDS=0
# removes the fields.
CONFIG_SETTINGS = {
    "zone_shrink_speed": 0.0,
    "post_stall": os.environ.get("BNB_POST_STALL", "1") != "0",
    **({"field_count": 0} if os.environ.get("BNB_FIELDS", "1") == "0" else {}),
}


def flight_config() -> core.GameConfig:
    """The GameConfig the tool flies and charts (CONFIG_SETTINGS)."""
    return core.GameConfig(**CONFIG_SETTINGS)
