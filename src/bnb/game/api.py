"""The game's public functions over the contracts.

    rules = rules_of(config); types = ship_types(device)
    state = reset(rules, types, uniforms(rules, B), fogged)
    moved, sweep = move(rules, types, state, action, spread(rules, B))
    state, events = resolve(rules, types, moved, sweep, uniforms(rules, B))
    state, obs, privileged = perceive(rules, state)

Every function is pure (tensors in, tensors out) and works eagerly or under
torch.compile, at any batch size, on any device. The state is opaque outside `bnb.game`.
The tick's randomness is an explicit input; `spread` and `uniforms` draw it. The ship card
`types` is always passed: it is data, which a sweep may change between calls.
"""

import torch

from .perception.observe import perceive
from .physics.step import move, reset, resolve, step
from .rules import Rules, rules_of
from .ship_types import ShipTypes, ship_types
from .state import State, Sweep, replace, sweep_like

__all__ = ["Rules", "ShipTypes", "State", "Sweep", "move", "perceive", "reset", "resolve",
           "rules_of", "ship_types", "spread", "start_clocks", "step", "sweep_like", "uniforms"]


def spread(rules: Rules, batch: int, device=None, generator=None) -> torch.Tensor:
    """(B, N, 2) standard normals: the bullet spread of one tick."""
    return torch.randn((batch, rules.num_ships, 2), device=device, generator=generator)


def uniforms(rules: Rules, batch: int, device=None, generator=None) -> torch.Tensor:
    """(B, U) uniforms in [0, 1): a tick's map, formation rotations and keys."""
    return torch.rand((batch, rules.tick_uniform_count), device=device, generator=generator)


def start_clocks(state: State, episode_tick: torch.Tensor) -> State:
    """The state with its episode clocks set to episode_tick (B,) (to stagger resets)."""
    return replace(state, match=replace(state.match, episode_tick=episode_tick.to(torch.int32)))
