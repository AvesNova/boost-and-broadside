"""The constants of SPEC.md, from spec/presets.json."""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT, "golden", "presets.json")) as fh:
    _DATA = json.load(fh)

FLIGHT: dict = _DATA["flight"]
PRESETS: dict = _DATA["presets"]


SIZE_KEYS = ("num_ships", "num_fields", "arena_radius")


def preset(spec) -> dict:
    """A preset by name, or a custom size {num_ships, num_fields, arena_radius} (SPEC 1.2)
    completed with the shared constants."""
    if isinstance(spec, str):
        return PRESETS[spec]
    return {**PRESETS["squad"], **{k: spec[k] for k in SIZE_KEYS}}


def map_uniform_count(spec) -> int:
    return 4 * preset(spec)["num_fields"] + 1


def reset_uniform_count(spec) -> int:
    """Tick uniforms per environment (game spec 2): the map, two team rotations, a key per
    ship."""
    return map_uniform_count(spec) + 2 + preset(spec)["num_ships"]
