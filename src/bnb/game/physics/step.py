"""One tick (game spec 2, modes spec 2) and a fresh reset (game spec 9), as pure functions.

The tick is split where the runtime may cut stages: `move` (phases 1-3, the motions) and
`resolve` (phases 4-8, everything that depends on both motions, with the mode phase after
damage). Randomness comes in as inputs: spread (B, N, 2) standard normals and the (B, U)
tick uniforms. `types` is the ship card, tensors, so a sweep can vary it without
recompiling. The mode (rules.mode) owns the win rule, the objective, spawn placement and
spawn protection.
"""

import torch

from bnb.contracts import Action, Events

from ..modes import phase as modes
from ..modes.spawns import placement
from ..rules import Rules
from ..ship_types import ShipTypes
from ..state import State, Sweep, apply, blank, replace, where
from .bullets import fire, move_bullets
from .combat import apply_damage, boundary_damage, hits
from .flight import airframe_of, controls, flight_tick
from .spawning import TickUniforms, new_map, spawn


def reset(rules: Rules, types: ShipTypes, uniforms: torch.Tensor,
          fogged: torch.Tensor | None = None) -> State:
    """A fresh state from (B, U) tick uniforms: new maps and modes, every ship spawned,
    tick 0. fogged (B,) bool: which environments hide what their teams cannot see (default
    all); it stays fixed through the episode resets that follow."""
    B, device = uniforms.shape[0], uniforms.device
    state = blank(rules, B, device)
    ship_type = torch.tensor(rules.slot_types, dtype=torch.int32, device=device)
    ships = replace(state.ships, ship_type=ship_type.expand(B, -1).contiguous())
    tick = TickUniforms(rules, uniforms)
    fields, axis_x, axis_y = new_map(rules, tick.map)
    mode, fields = modes.new_match(rules, tick.mode, tick.map, fields, state.mode)
    everyone = torch.ones_like(state.mode.front, dtype=torch.bool)
    place = placement(rules, tick.mode, mode, ships.x, ships.y, ships.alive, everyone, axis_x,
                      axis_y, tick.rotation, tick.keys)
    ships = spawn(rules, types, ships, fields, torch.ones_like(ships.alive), place)
    match = replace(state.match, spawn_axis_x=axis_x, spawn_axis_y=axis_y)
    if fogged is not None:
        match = replace(match, fogged=fogged.to(torch.bool))
    return State(ships=ships, bullets=state.bullets, fields=fields, match=match, mode=mode)


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
    """Phases 4-8 on the moved state: hits, damage, the mode phase, clocks, death and
    respawn, episode end."""
    ships, bullets, match = state.ships, state.bullets, state.match
    alive0, active = ships.alive, bullets.moves_left > 0  # as at tick start: motion keeps them

    # 4-5. Hits between the two motions (none deal damage to a protected ship; the bullet
    # is still spent), then damage, deaths and shields.
    hit = hits(rules, types, sweep, ships, bullets, alive0)
    damage, first_shooter = hit.damage, hit.first_shooter
    guarded = modes.protected(rules, state.mode, ships)
    if guarded is not None:
        damage = torch.where(guarded[:, None, :], 0.0, damage)
        first_shooter = torch.where(guarded, -1, first_shooter)
    boundary = boundary_damage(rules, ships, alive0)
    applied = apply_damage(rules, types, ships, alive0, damage.sum(1) + boundary)
    ships, died = applied.ships, applied.died
    killer = torch.where(died, first_shooter, -1).to(torch.int32)

    # The mode phase: the mode state, the objective events, and whether the match is won.
    alive = alive0 & ~died
    phase = modes.mode_phase(rules, state.mode, ships, alive, died)
    done = (match.episode_tick + 1 >= rules.episode_ticks) | phase.won
    result = torch.where(phase.won, phase.result, modes.clock_result(rules, phase.mode))
    result = torch.where(done, result, 0)

    # 6. Clocks.
    moves_left = torch.where(hit.bullet_hit, 0,
                             torch.where(active, bullets.moves_left - 1, bullets.moves_left))
    bullets = replace(bullets, moves_left=moves_left)

    # 7. Death and respawn (the spawns happen together with the resets below).
    was_dead = ~alive0
    respawn_wait = torch.where(was_dead, ships.respawn_wait - 1, ships.respawn_wait)
    due = was_dead & (respawn_wait <= 0)
    ships = replace(ships, alive=alive,
                    gun_cooldown=(ships.gun_cooldown - 1).clamp_min(0),
                    respawn_wait=torch.where(died, rules.respawn_ticks, respawn_wait),
                    proper_speed=torch.where(died, rules.flight.speed_floor, ships.proper_speed),
                    nose_rate=torch.where(died, 0.0, ships.nose_rate))

    # 8. Episode end: a finished environment gets a new map and mode state, no bullets,
    # cleared slots and flags, and all its ships spawn. Its fog setting stays.
    tick = TickUniforms(rules, uniforms)
    fields, axis_x, axis_y = new_map(rules, tick.map)
    mode, fields = modes.new_match(rules, tick.mode, tick.map, fields,
                                   apply(torch.zeros_like, phase.mode))
    fields = where(done, fields, state.fields)
    mode = where(done, mode, phase.mode)
    axis_x = torch.where(done, axis_x, match.spawn_axis_x)
    axis_y = torch.where(done, axis_y, match.spawn_axis_y)
    bullets = where(done, apply(torch.zeros_like, bullets), bullets)
    keep = ~done[:, None]
    ships = replace(ships, next_slot=torch.where(keep, ships.next_slot, 0),
                    fired=ships.fired & keep)
    place = placement(rules, tick.mode, mode, ships.x, ships.y, alive, done, axis_x, axis_y,
                      tick.rotation, tick.keys)
    ships = spawn(rules, types, ships, fields, due | done[:, None], place)
    match = replace(match, episode_tick=torch.where(done, 0, match.episode_tick + 1),
                    spawn_axis_x=axis_x, spawn_axis_y=axis_y)

    events = Events(damage=damage, boundary_damage=boundary, died=died,
                    fired=state.ships.fired, spawned=ships.spawned, killer=killer, done=done,
                    shield_lost=applied.shield_lost, shield_restored=applied.shield_restored,
                    objective_progress=phase.objective_progress, objective=phase.objective,
                    result=result)
    return State(ships=ships, bullets=bullets, fields=fields, match=match, mode=mode), events


def step(rules: Rules, types: ShipTypes, state: State, action: Action, spread: torch.Tensor,
         uniforms: torch.Tensor) -> tuple[State, Events]:
    """One whole tick (phases 1-8; perception is separate)."""
    moved, sweep = move(rules, types, state, action, spread)
    return resolve(rules, types, moved, sweep, uniforms)
