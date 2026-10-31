"""One tick (game spec 2) and a fresh reset (game spec 9), as pure functions.

The tick is split where the runtime may cut stages: `move` (phases 1-3, the motions) and
`resolve` (phases 4-8, everything that depends on both motions). Randomness comes in as
inputs: spread (B, N, 2) standard normals and the (B, U) tick uniforms.

`types` is the ship card (tensors, so a sweep can vary it without recompiling); None
means the default card.
"""

import torch

from bnb.contracts import Action, Events

from ..rules import Rules
from ..ship_types import ShipTypes, ship_types
from ..state import State, Sweep, blank, replace, where
from ..state import apply as apply_fields
from .bullets import fire, move_bullets
from .combat import apply_damage, boundary_damage, hits
from .flight import airframe, controls, flight_tick
from .spawning import TickUniforms, new_map, spawn


def _card(types: ShipTypes | None, device) -> ShipTypes:
    return ship_types(device) if types is None else types


def reset(rules: Rules, uniforms: torch.Tensor, types: ShipTypes | None = None) -> State:
    """A fresh state from (B, U) tick uniforms: new maps, every ship spawned, t = 0."""
    B, device = uniforms.shape[0], uniforms.device
    types = _card(types, device)
    st = blank(rules, B, device)
    ship_type = torch.tensor(rules.slot_types, dtype=torch.int32, device=device)
    st = replace(st, ships=replace(st.ships, ship_type=ship_type.expand(B, -1).contiguous()))
    u = TickUniforms(rules, uniforms)
    fields, ax, ay = new_map(rules, u.map)
    ships = spawn(rules, types, st.ships, fields, ax, ay,
                  torch.ones_like(st.ships.alive), u.rotation, u.keys)
    return replace(st, ships=ships, fields=fields,
                   match=replace(st.match, axis_x=ax, axis_y=ay))


def move(rules: Rules, st: State, action: Action, spread: torch.Tensor,
         types: ShipTypes | None = None) -> tuple[State, Sweep]:
    """Phases 1-3: fire, ships fly, bullets fly. Returns the moved state and this tick's
    motions (the hit test's input). Motion changes no flag (but `fired`), clock or ttl."""
    types = _card(types, spread.device)
    ships0, bullets0 = fire(rules, types, st.ships, st.bullets, action.shoot == 1, spread)
    stick, power = controls(action.power, action.stick)
    A = airframe(types, ships0.ship_type)
    flown = flight_tick(rules.flight, A, rules.tick, ships0, stick, power, st.fields)
    ships = where(ships0.alive, flown, ships0)  # ships dead at tick start keep everything
    bullets = move_bullets(rules, bullets0, st.fields)
    sweep = Sweep(x=ships0.x, y=ships0.y, dx=ships.x - ships0.x, dy=ships.y - ships0.y,
                  bx=bullets0.x, by=bullets0.y, bdx=bullets.x - bullets0.x,
                  bdy=bullets.y - bullets0.y)
    return replace(st, ships=ships, bullets=bullets), sweep


def resolve(rules: Rules, st: State, sweep: Sweep, uniforms: torch.Tensor,
            types: ShipTypes | None = None) -> tuple[State, Events]:
    """Phases 4-8 on the moved state: hits, damage, clocks, death and respawn, episode end."""
    types = _card(types, uniforms.device)
    ships, bullets, match = st.ships, st.bullets, st.match
    alive0, active = ships.alive, bullets.ttl > 0  # as at tick start: motion changes neither

    # 4-5. Hits between the two motions, then damage, deaths and shields.
    hit = hits(rules, types, sweep, ships, bullets, alive0)
    boundary = boundary_damage(rules, ships, alive0)
    ships, died = apply_damage(rules, types, ships, alive0, hit.taken + boundary)
    killer = torch.where(died, hit.first_shooter, -1).to(torch.int32)

    # 6. Clocks.
    ttl = torch.where(hit.bullet_hit, 0, torch.where(active, bullets.ttl - 1, bullets.ttl))
    bullets = replace(bullets, ttl=ttl)

    # 7. Death and respawn (the spawns happen together with the resets below).
    was_dead = ~alive0
    respawn = torch.where(was_dead, ships.respawn - 1, ships.respawn)
    due = was_dead & (respawn <= 0)
    ships = replace(ships, alive=alive0 & ~died, gun_wait=(ships.gun_wait - 1).clamp_min(0),
                    respawn=torch.where(died, rules.respawn_ticks, respawn),
                    w=torch.where(died, rules.flight.eps, ships.w),
                    r=torch.where(died, 0.0, ships.r))

    # 8. Episode clock: a finished environment gets a new map, no bullets, cleared cursors
    # and flags, and all its ships spawn.
    done = match.t + 1 >= rules.episode_ticks
    u = TickUniforms(rules, uniforms)
    fields, ax, ay = new_map(rules, u.map)
    fields = where(done, fields, st.fields)
    ax, ay = torch.where(done, ax, match.axis_x), torch.where(done, ay, match.axis_y)
    bullets = where(done, apply_fields(torch.zeros_like, bullets), bullets)
    keep = ~done[:, None]
    ships = replace(ships, cursor=torch.where(keep, ships.cursor, 0), fired=ships.fired & keep)
    ships = spawn(rules, types, ships, fields, ax, ay, due | done[:, None], u.rotation, u.keys)
    match = replace(match, t=torch.where(done, 0, match.t + 1), axis_x=ax, axis_y=ay)

    events = Events(damage=hit.damage, boundary_damage=boundary, died=died, fired=st.ships.fired,
                    spawned=ships.spawned, killer=killer, done=done)
    return State(ships=ships, bullets=bullets, fields=fields, match=match), events


def step(rules: Rules, st: State, action: Action, spread: torch.Tensor,
         uniforms: torch.Tensor, types: ShipTypes | None = None) -> tuple[State, Events]:
    """One whole tick (phases 1-8; perception is separate)."""
    moved, sweep = move(rules, st, action, spread, types)
    return resolve(rules, moved, sweep, uniforms, types)


