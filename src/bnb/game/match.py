"""The game's public functions over the contracts: reset, the two halves of a tick, perceive.

The state is a dict of tensors, opaque outside `bnb.game` (the runtime only copies it).
Physics is the race arm (race SPEC, branch v3-general) until the physics section replaces it.
"""

import dataclasses

import torch

from bnb.contracts import Events, GameConfig, ticks_of

from .physics.constants import Preset, preset_of
from .physics.spawn import new_map
from .physics.step import ORIGIN_KEYS, move, reset as physics_reset, resolve as physics_resolve

__all__ = ["ORIGIN_KEYS", "move", "new_map", "preset", "reset", "resolve"]


def preset(cfg: GameConfig) -> Preset:
    """The physics constants for a config's sizes and time-native rules."""
    p = preset_of(dict(num_ships=cfg.num_ships, num_fields=cfg.num_fields,
                       arena_radius=cfg.arena_radius))
    return dataclasses.replace(p, episode_ticks=ticks_of(cfg.episode_s, p.h),
                               respawn_ticks=ticks_of(cfg.respawn_s, p.h))


def reset(p: Preset, uniforms: torch.Tensor) -> dict:
    """A fresh state from (B, 4M + 1) reset uniforms: new maps, every ship spawned, t = 0."""
    st = physics_reset(p, uniforms)
    B, N = st["x"].shape
    st["last_seen"] = torch.zeros((B, 2, N), dtype=torch.int32, device=uniforms.device)
    return st


def resolve(p: Preset, moved: dict, origin: dict, fresh: dict) -> tuple[dict, Events]:
    """Phases 4-8 of the tick (hits through episode end), with the events as a contract."""
    st, ev = physics_resolve(p, moved, origin, fresh)
    return st, Events(damage=ev["damage"], boundary_damage=ev["boundary_damage"], died=ev["died"],
                      fired=ev["fired"], spawned=st["spawned"], killer=ev["killer"],
                      done=ev["done"])
