"""One tick (game spec 2) and a fresh reset (game spec 9), as pure functions.

The tick is split where the runtime may cut stages: `move` (phases 1-3, the motions) and
`resolve` (phases 4-8, everything that depends on both motions). Randomness comes in as
inputs: spread (B, N, 2) standard normals and the (B, U) tick uniforms. `types` is the
ship card, tensors, so a sweep can vary it without recompiling.
"""

import torch

from bnb.contracts import Action, Events

from ..rules import Rules
from ..ship_types import ShipTypes
from ..state import State, Sweep, apply, blank, replace, where
from .bullets import fire, move_bullets
from .combat import apply_damage, boundary_damage, hits
from .flight import airframe_of, controls, flight_tick
from .spawning import TickUniforms, new_map, spawn


def reset(rules: Rules, types: ShipTypes, uniforms: torch.Tensor,
          fogged: torch.Tensor | None = None) -> State:
    """A fresh state from (B, U) tick uniforms: new maps, every ship spawned, tick 0.
    fogged (B,) bool: which environments hide what their teams cannot see (default all);
    it stays fixed through the episode resets that follow."""
    B, device = uniforms.shape[0], uniforms.device
    state = blank(rules, B, device)
    ship_type = torch.tensor(rules.slot_types, dtype=torch.int32, device=device)
    ships = replace(state.ships, ship_type=ship_type.expand(B, -1).contiguous())
    tick = TickUniforms(rules, uniforms)
    fields, axis_x, axis_y = new_map(rules, tick.map)
    ships = spawn(rules, types, ships, fields, axis_x, axis_y, torch.ones_like(ships.alive),
                  tick.rotation, tick.keys)
    match = replace(state.match, spawn_axis_x=axis_x, spawn_axis_y=axis_y)
    if fogged is not None:
        match = replace(match, fogged=fogged.to(torch.bool))
    return replace(state, ships=ships, fields=fields, match=match)


def move(rules: Rules, types: ShipTypes, state: State, action: Action,
         spread: torch.Tensor) -> tuple[State, Sweep]:
    """Phases 1-3: fire, ships fly, bullets fly. Returns the moved state and this tick's
    motions (the hit test's input). Motion changes no flag (but `fired`), clock or count."""
    ships0, bullets0 = fire(rules, types, state.ships, state.bullets, action.shoot == 1, spread)
    stick, power = controls(action.power, action.stick)
    airframe = airframe_of(types, ships0.ship_type)
    flown = flight_tick(rules.flight, airframe, rules.tick_length, ships0, stick, power,
                        state.fields)
    ships = where(ships0.alive, flown, ships0)  # ships dead at tick start keep everything
    bullets = move_bullets(rules, bullets0, state.fields)
    sweep = Sweep(ship_x=ships0.x, ship_y=ships0.y, ship_displacement_x=ships.x - ships0.x,
                  ship_displacement_y=ships.y - ships0.y, bullet_x=bullets0.x,
                  bullet_y=bullets0.y, bullet_displacement_x=bullets.x - bullets0.x,
                  bullet_displacement_y=bullets.y - bullets0.y)
    return replace(state, ships=ships, bullets=bullets), sweep


def resolve(rules: Rules, types: ShipTypes, state: State, sweep: Sweep,
            uniforms: torch.Tensor) -> tuple[State, Events]:
    """Phases 4-8 on the moved state: hits, damage, clocks, death and respawn, episode end."""
    ships, bullets, match = state.ships, state.bullets, state.match
    alive0, active = ships.alive, bullets.moves_left > 0  # as at tick start: motion keeps them

    # 4-5. Hits between the two motions, then damage, deaths and shields.
    hit = hits(rules, types, sweep, ships, bullets, alive0)
    boundary = boundary_damage(rules, ships, alive0)
    ships, died = apply_damage(rules, types, ships, alive0, hit.taken + boundary)
    killer = torch.where(died, hit.first_shooter, -1).to(torch.int32)

    # 6. Clocks.
    moves_left = torch.where(hit.bullet_hit, 0,
                             torch.where(active, bullets.moves_left - 1, bullets.moves_left))
    bullets = replace(bullets, moves_left=moves_left)

    # 7. Death and respawn (the spawns happen together with the resets below).
    was_dead = ~alive0
    respawn_wait = torch.where(was_dead, ships.respawn_wait - 1, ships.respawn_wait)
    due = was_dead & (respawn_wait <= 0)
    ships = replace(ships, alive=alive0 & ~died,
                    gun_cooldown=(ships.gun_cooldown - 1).clamp_min(0),
                    respawn_wait=torch.where(died, rules.respawn_ticks, respawn_wait),
                    proper_speed=torch.where(died, rules.flight.speed_floor, ships.proper_speed),
                    nose_rate=torch.where(died, 0.0, ships.nose_rate))

    # 8. Episode clock: a finished environment gets a new map, no bullets, cleared slots
    # and flags, and all its ships spawn. Its fog setting stays.
    done = match.episode_tick + 1 >= rules.episode_ticks
    tick = TickUniforms(rules, uniforms)
    fields, axis_x, axis_y = new_map(rules, tick.map)
    fields = where(done, fields, state.fields)
    axis_x = torch.where(done, axis_x, match.spawn_axis_x)
    axis_y = torch.where(done, axis_y, match.spawn_axis_y)
    bullets = where(done, apply(torch.zeros_like, bullets), bullets)
    keep = ~done[:, None]
    ships = replace(ships, next_slot=torch.where(keep, ships.next_slot, 0),
                    fired=ships.fired & keep)
    ships = spawn(rules, types, ships, fields, axis_x, axis_y, due | done[:, None],
                  tick.rotation, tick.keys)
    match = replace(match, episode_tick=torch.where(done, 0, match.episode_tick + 1),
                    spawn_axis_x=axis_x, spawn_axis_y=axis_y)

    events = Events(damage=hit.damage, boundary_damage=boundary, died=died,
                    fired=state.ships.fired, spawned=ships.spawned, killer=killer, done=done)
    return State(ships=ships, bullets=bullets, fields=fields, match=match), events


def step(rules: Rules, types: ShipTypes, state: State, action: Action, spread: torch.Tensor,
         uniforms: torch.Tensor) -> tuple[State, Events]:
    """One whole tick (phases 1-8; perception is separate)."""
    moved, sweep = move(rules, types, state, action, spread)
    return resolve(rules, types, moved, sweep, uniforms)
