"""One tick (game spec 2, modes spec 2) and a fresh reset (game spec 9), as pure functions.

The tick is split where the runtime may cut stages: `move` (phases 1-3, the motions) and
`resolve` (phases 4-8, everything that depends on both motions, with the mode phase after
damage). Randomness comes in as inputs: spread (B, N, 2) standard normals and the (B, U)
tick uniforms. `types` is the ship card, tensors, so a sweep can vary it without
recompiling. Each environment's mode owns its win rule, objective, spawn placement and
spawn protection; those steps run on the mode's slices of the batch (`layout.per_mode`), the
rest once over the batch with the per-environment rules (`layout.environment`).

Benched ships (unequal teams at one static size): `active` (B, 2) int32, each team's active
ship count per environment, a runtime input of `reset`, `resolve` and `step` (None: every
ship active, the game unchanged). A team's benched ships are its last S - active slots
(`benched_of`). They are kept as dead ships waiting to respawn, alive false with
respawn_wait held at the environment's full respawn length (`_bench`), so they never act,
fly, fire, take or deal damage, are seen by the enemy, score or respawn; team deathmatch's
winning score counts the enemy's active ships only (`modes.phase.mode_phase`). The counts
hold through the episode resets: a caller passes the same tensor every tick.
"""

import torch

from bnb.contracts import Action, Events

from ..layout import Environment, environment, per_mode
from ..modes import phase as modes
from ..modes.rules import FRONTLINE, ModeRules
from ..modes.spawns import placement
from ..rules import Rules
from ..ship_types import ShipTypes, per_ship, ticks_between_shots
from ..state import Ships, State, Sweep, apply, benched_of, blank, replace, where
from .bullets import fire, move_bullets
from .combat import apply_damage, boundary_damage, hits, quiet
from .flight import airframe_of, controls, flight_tick
from .spawning import TickUniforms, new_map, spawn


def _new_matches(rules: Rules, arena: Environment, tick: TickUniforms, previous_mode) -> tuple:
    """Each environment's new map and mode state (the mode's map on its own slices)."""
    fields, axis_x, axis_y = new_map(rules, arena.arena_radius, arena.map_scale, tick.map)

    def match(mode_rules: ModeRules, arena, uniforms, map_uniforms, fields, previous_mode):
        return modes.new_match(
            mode_rules,
            arena.arena_radius,
            arena.map_scale,
            uniforms[:, : mode_rules.uniform_count(rules.num_ships)],
            map_uniforms,
            fields,
            previous_mode,
            rules.field_disc_fraction,
        )

    mode, fields = per_mode(
        rules, tick.map.shape[0], match, arena, tick.mode, tick.map, fields, previous_mode
    )
    return mode, fields, axis_x, axis_y


def _placement(
    rules: Rules,
    arena: Environment,
    tick: TickUniforms,
    mode,
    ships,
    alive,
    starting,
    axis_x,
    axis_y,
    benched=None,
):
    """Every ship's spawn placement by its environment's mode (selected later by the mask).
    benched (B, N) or None: ships that do not spawn, which spawning ships keep no distance
    from."""

    def place(
        mode_rules: ModeRules,
        arena,
        uniforms,
        mode,
        x,
        y,
        alive,
        starting,
        axis_x,
        axis_y,
        rotation,
        keys,
        benched,
    ):
        return placement(
            rules,
            mode_rules,
            arena.arena_radius,
            arena.map_scale,
            uniforms[:, : mode_rules.uniform_count(rules.num_ships)],
            mode,
            x,
            y,
            alive,
            starting,
            axis_x,
            axis_y,
            rotation,
            keys,
            benched,
        )

    return per_mode(
        rules,
        alive.shape[0],
        place,
        arena,
        tick.mode,
        mode,
        ships.x,
        ships.y,
        alive,
        starting,
        axis_x,
        axis_y,
        tick.rotation,
        tick.keys,
        benched,
    )


def _bench(rules: Rules, ships: Ships, benched: torch.Tensor, respawn_ticks: torch.Tensor) -> Ships:
    """The ships with the benched ones (B, N) held as dead ships waiting to respawn: alive
    and spawned false, respawn_wait the environment's full respawn length (B,) ticks, and
    the speed floor and still nose a death leaves. Their other values stay as the last spawn
    placed them (a fresh ship's, at its spawn point), as a dead ship keeps its last ones."""
    return replace(
        ships,
        alive=ships.alive & ~benched,
        spawned=ships.spawned & ~benched,
        respawn_wait=torch.where(benched, respawn_ticks[:, None], ships.respawn_wait),
        proper_speed=torch.where(benched, rules.flight.speed_floor, ships.proper_speed),
        nose_rate=torch.where(benched, 0.0, ships.nose_rate),
    )


def _vision_ranges(rules: Rules, types: ShipTypes, ship_type: torch.Tensor) -> torch.Tensor:
    """(B, N) each ship's vision range (m), its type's: kept in the match for perception."""
    return rules.vision_range * per_ship(types.vision_range_scale, ship_type)


def reset(
    rules: Rules,
    types: ShipTypes,
    uniforms: torch.Tensor,
    fogged: torch.Tensor | None = None,
    *,
    ship_type: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
) -> State:
    """A fresh state from (B, U) tick uniforms: new maps and modes, every ship spawned,
    tick 0. fogged (B,) bool: which environments hide what their teams cannot see (default
    all); it stays fixed through the episode resets that follow. ship_type (B, N) int32:
    each ship's type, per environment (default the rules' lineup, `Rules.slot_types`); a
    ship keeps its type through the episode resets that follow. active (B, 2) int32: each
    team's active ships, the rest benched (see the module docstring; default every ship).
    Raises ValueError for a card whose fire rate is not a whole number of ticks at the
    rules' tick."""
    ticks_between_shots(
        types.fire_rate, rules.tick_length
    )  # refuses a fire rate the tick cannot keep
    B, device = uniforms.shape[0], uniforms.device
    state = blank(rules, B, device)
    arena = environment(rules, B, device)
    if ship_type is None:
        ship_type = torch.tensor(rules.slot_types, dtype=torch.int32, device=device)
        ship_type = ship_type.expand(B, -1).contiguous()
    else:
        if ship_type.shape != (B, rules.num_ships):
            raise ValueError(f"ship_type {tuple(ship_type.shape)}: need ({B}, {rules.num_ships})")
        ship_type = ship_type.to(device=device, dtype=torch.int32, copy=True)
    ships = replace(state.ships, ship_type=ship_type)
    tick = TickUniforms(rules, uniforms)
    mode, fields, axis_x, axis_y = _new_matches(rules, arena, tick, state.mode)
    everyone = torch.ones_like(state.mode.front, dtype=torch.bool)
    benched = None if active is None else benched_of(rules.ships_per_team, active)
    place = _placement(
        rules, arena, tick, mode, ships, ships.alive, everyone, axis_x, axis_y, benched
    )
    ships = spawn(rules, types, ships, fields, torch.ones_like(ships.alive), place)
    if benched is not None:
        ships = _bench(rules, ships, benched, arena.respawn_ticks)
    match = replace(
        state.match,
        spawn_axis_x=axis_x,
        spawn_axis_y=axis_y,
        vision_range=_vision_ranges(rules, types, ships.ship_type),
    )
    if fogged is not None:
        match = replace(match, fogged=fogged.to(torch.bool))
    return State(ships=ships, bullets=state.bullets, fields=fields, match=match, mode=mode)


def move(
    rules: Rules, types: ShipTypes, state: State, action: Action, spread: torch.Tensor
) -> tuple[State, Sweep]:
    """Phases 1-3: fire, ships fly, bullets fly. Returns the moved state and this tick's
    motions (the hit test's input). Motion changes no flag (but `fired`), clock or count."""
    ships0, bullets0 = fire(rules, types, state.ships, state.bullets, action.shoot == 1, spread)
    stick, power = controls(action.power, action.stick)
    airframe = airframe_of(types, ships0.ship_type)
    flown = flight_tick(
        rules.flight, airframe, rules.tick_length, ships0, stick, power, state.fields
    )
    ships = where(ships0.alive, flown, ships0)  # ships dead at tick start keep everything
    bullets = move_bullets(rules, types, bullets0, state.fields, ships0.ship_type)
    sweep = Sweep(
        ship_x=ships0.x,
        ship_y=ships0.y,
        ship_displacement_x=ships.x - ships0.x,
        ship_displacement_y=ships.y - ships0.y,
        bullet_x=bullets0.x,
        bullet_y=bullets0.y,
        bullet_displacement_x=bullets.x - bullets0.x,
        bullet_displacement_y=bullets.y - bullets0.y,
    )
    return replace(state, ships=ships, bullets=bullets), sweep


def resolve(
    rules: Rules,
    types: ShipTypes,
    state: State,
    sweep: Sweep,
    action: Action,
    uniforms: torch.Tensor,
    active: torch.Tensor | None = None,
) -> tuple[State, Events]:
    """Phases 4-8 on the moved state: hits, damage, the mode phase, clocks, death and
    respawn, episode end. action: the tick's action, as `move` was given (the shield's
    recharge runs only on a quiet one, `combat.quiet`). active (B, 2) int32: each team's
    active ships, as `reset` was given (default every ship)."""
    ships, bullets, match = state.ships, state.bullets, state.match
    B = ships.x.shape[0]
    arena = environment(rules, B, ships.x.device)
    benched = None if active is None else benched_of(rules.ships_per_team, active)
    alive0, flying = ships.alive, bullets.moves_left > 0  # as at tick start: motion keeps them

    # 4-5. Hits between the two motions (none deal damage to a protected ship; the bullet
    # is still spent), then damage, deaths and shields (the recharge on quiet actions).
    hit = hits(rules, types, sweep, ships, bullets, alive0)
    damage, first_shooter = hit.damage, hit.first_shooter
    enemy_damage_by_aspect, first_aspect = hit.enemy_damage_by_aspect, hit.first_aspect
    if any(mode.name == FRONTLINE for mode in rules.modes):  # the only mode that protects
        guarded = per_mode(
            rules,
            B,
            lambda mode_rules, mode, ships: modes.protected(rules, mode_rules, mode, ships),
            state.mode,
            ships,
        )
        damage = torch.where(guarded[:, None, :], 0.0, damage)
        first_shooter = torch.where(guarded, -1, first_shooter)
        enemy_damage_by_aspect = torch.where(guarded[..., None], 0.0, enemy_damage_by_aspect)
        first_aspect = torch.where(guarded, -1, first_aspect)
    boundary = boundary_damage(rules, arena.arena_radius, ships, alive0)
    applied = apply_damage(rules, types, ships, alive0, damage.sum(1) + boundary, quiet(action))
    ships, died = applied.ships, applied.died
    killer = torch.where(died, first_shooter, -1).to(torch.int32)
    kill_aspect = torch.where(died, first_aspect, -1).to(torch.int32)

    # The mode phase: the mode state, the objective events, and whether the match is won.
    alive = alive0 & ~died

    def mode_phase(mode_rules: ModeRules, mode, ships, alive, died, active):
        phase = modes.mode_phase(rules, mode_rules, mode, ships, alive, died, active)
        return phase, modes.clock_result(mode_rules, phase.mode, ships, alive)

    phase, clock_result = per_mode(rules, B, mode_phase, state.mode, ships, alive, died, active)
    done = (match.episode_tick + 1 >= arena.episode_ticks) | phase.won
    result = torch.where(phase.won, phase.result, clock_result)
    result = torch.where(done, result, 0)

    # 6. Clocks.
    moves_left = torch.where(
        hit.bullet_hit, 0, torch.where(flying, bullets.moves_left - 1, bullets.moves_left)
    )
    bullets = replace(bullets, moves_left=moves_left)

    # 7. Death and respawn (the spawns happen together with the resets below).
    was_dead = ~alive0
    respawn_wait = torch.where(was_dead, ships.respawn_wait - 1, ships.respawn_wait)
    due = was_dead & (respawn_wait <= 0)
    ships = replace(
        ships,
        alive=alive,
        gun_cooldown=(ships.gun_cooldown - 1).clamp_min(0),
        respawn_wait=torch.where(died, arena.respawn_ticks[:, None], respawn_wait),
        proper_speed=torch.where(died, rules.flight.speed_floor, ships.proper_speed),
        nose_rate=torch.where(died, 0.0, ships.nose_rate),
    )

    # 8. Episode end: a finished environment gets a new map and mode state, no bullets,
    # cleared slots and flags, and all its ships spawn. Its fog setting stays.
    tick = TickUniforms(rules, uniforms)
    mode, fields, axis_x, axis_y = _new_matches(rules, arena, tick, phase.mode)
    fields = where(done, fields, state.fields)
    mode = where(done, mode, phase.mode)
    axis_x = torch.where(done, axis_x, match.spawn_axis_x)
    axis_y = torch.where(done, axis_y, match.spawn_axis_y)
    bullets = where(done, apply(torch.zeros_like, bullets), bullets)
    keep = ~done[:, None]
    ships = replace(
        ships, next_slot=torch.where(keep, ships.next_slot, 0), fired=ships.fired & keep
    )
    place = _placement(rules, arena, tick, mode, ships, alive, done, axis_x, axis_y, benched)
    ships = spawn(rules, types, ships, fields, due | done[:, None], place)
    if benched is not None:  # a benched ship spawns with an episode's reset, and is benched again
        ships = _bench(rules, ships, benched, arena.respawn_ticks)
    match = replace(
        match,
        episode_tick=torch.where(done, 0, match.episode_tick + 1),
        spawn_axis_x=axis_x,
        spawn_axis_y=axis_y,
        vision_range=_vision_ranges(rules, types, ships.ship_type),
    )

    events = Events(
        damage=damage,
        boundary_damage=boundary,
        died=died,
        fired=state.ships.fired,
        spawned=ships.spawned,
        killer=killer,
        done=done,
        shield_lost=applied.shield_lost,
        shield_restored=applied.shield_restored,
        objective_progress=phase.objective_progress,
        objective=phase.objective,
        result=result,
        enemy_damage_by_aspect=enemy_damage_by_aspect,
        kill_aspect=kill_aspect,
    )
    return State(ships=ships, bullets=bullets, fields=fields, match=match, mode=mode), events


def step(
    rules: Rules,
    types: ShipTypes,
    state: State,
    action: Action,
    spread: torch.Tensor,
    uniforms: torch.Tensor,
    active: torch.Tensor | None = None,
) -> tuple[State, Events]:
    """One whole tick (phases 1-8; perception is separate); active as for `resolve`."""
    moved, sweep = move(rules, types, state, action, spread)
    return resolve(rules, types, moved, sweep, action, uniforms, active)
