"""One tick (SPEC 2) and a fresh reset (SPEC 9), as pure functions of the state dict."""

import torch

from .bullets import fire, move_bullets
from .combat import apply_damage, boundary_damage, hits
from .constants import Preset
from .flight import FLIGHT_KEYS, actions_to_controls, flight_tick
from .spawn import clear_envs, new_map, spawn
from .ship_types import flight_of, slot_types
from .state import blank_state, fields_of, select


def reset(p: Preset, uniforms: torch.Tensor) -> dict:
    """A fresh state: every environment draws a map and every ship spawns at t = 0."""
    B = uniforms.shape[0]
    done = torch.ones(B, dtype=torch.bool, device=uniforms.device)
    fresh = new_map(p, uniforms)
    st = dict(blank_state(p, B, uniforms.device), ship_type=slot_types(p, B, uniforms.device))
    st = clear_envs(p, st, done, fresh)
    return spawn(p, st, torch.ones((B, p.num_ships), dtype=torch.bool, device=uniforms.device), st["t"])


# The tick-start positions the hit test sweeps from (bullets after firing).
ORIGIN_KEYS = ("x", "y", "bx", "by")


def step(p: Preset, st: dict, actions: torch.Tensor, spread: torch.Tensor, fresh: dict):
    """Advance every environment by one tick. Returns (state, events).

    The tick's randomness comes in as inputs: spread (B, N, 2) normals for the bullets,
    and fresh = new_map(uniforms), the map each environment gets if it resets now.
    """
    return resolve(p, *move(p, st, actions, spread), fresh)


def move(p: Preset, st: dict, actions: torch.Tensor, spread: torch.Tensor):
    """Phases 1-3: fire, ships fly, bullets fly. Returns (moved state, origin), origin being
    the tick-start positions of ORIGIN_KEYS. Motion changes no flag, clock or ttl."""
    # 1. Fire, from the tick-start state.
    start = fire(p, st, actions[..., 2] == 1, spread)

    # 2. Ships fly; ships dead at tick start keep every flight variable.
    stick, power = actions_to_controls(actions[..., 0], actions[..., 1])
    F = flight_of(p.flights, start["ship_type"])
    flown = flight_tick(F, p.h, {k: start[k] for k in FLIGHT_KEYS}, stick, power, fields_of(start))
    moved = dict(start, **select(start["alive"], flown, start, FLIGHT_KEYS))

    # 3. Bullets fly (including the ones fired this tick).
    return move_bullets(p, moved), {k: start[k] for k in ORIGIN_KEYS}


def resolve(p: Preset, moved: dict, origin: dict, fresh: dict):
    """Phases 4-8 on the moved state: hits, damage, clocks, death and respawn, episode end."""
    alive0, active = moved["alive"], moved["ttl"] > 0  # as at tick start: motion changes neither

    # 4-5. Hits between the two motions, then damage, deaths and shields.
    hit = hits(p, origin, moved, active, alive0)
    boundary = boundary_damage(p, moved, alive0)
    st, died = apply_damage(p, moved, alive0, hit.taken + boundary)
    killer = torch.where(died, hit.first_shooter, -1).to(torch.int32)

    # 6. Clocks.
    st["ttl"] = torch.where(hit.bullet_hit, 0, torch.where(active, st["ttl"] - 1, st["ttl"]))
    st["gun_wait"] = (st["gun_wait"] - 1).clamp_min(0)

    # 7. Death and respawn (the spawns happen together with the resets below).
    was_dead = ~alive0
    respawn = torch.where(was_dead, st["respawn"] - 1, st["respawn"])
    due = was_dead & (respawn <= 0)
    st["alive"] = alive0 & ~died
    st["respawn"] = torch.where(died, p.respawn_ticks, respawn)
    st["w"] = torch.where(died, p.flight.eps, st["w"])
    st["r"] = torch.where(died, 0.0, st["r"])

    # 8. Episode clock; a finished environment gets a new map and all its ships spawn at t = 0.
    done = st["t"] + 1 >= p.episode_ticks
    st = clear_envs(p, st, done, fresh)  # sets t = 0 where done, the spawn clock of a reset
    st = spawn(p, st, due | done[:, None], st["t"])
    st["t"] = torch.where(done, 0, st["t"] + 1)

    events = dict(damage=hit.damage, boundary_damage=boundary, died=died, killer=killer, fired=moved["fired"],
                  done=done)
    return st, events
