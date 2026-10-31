"""The race correctness interface over bnb.game, eager on the CPU in float32.

The race speaks short flat keys (x, c, w, bx, fx, t, ...); bnb.game uses typed state with
full names. Everything about that translation lives here:
- states travel as numpy dicts under the race keys (STATE_KEYS);
- constant overrides use the race's presets.json keys (FLIGHT_SHARED, CARD, RULES);
- the observation goes back to the race layout: rows in sim order, `ally` from
  team_relative, times in ticks, no team_relative / time_since_seen / spawned / fired /
  medium columns, fields without the band width.
Bullets are not observed (contract decision 2026-10-31).
"""

import dataclasses

import numpy as np
import torch

from bnb.contracts import Action, GameConfig
from bnb.contracts.obs import FIELD, SHIP
from bnb.game import api
from bnb.game.physics import flight as flight_module
from bnb.game.physics.bullets import bullet_motion
from bnb.game.physics.medium import medium
from bnb.game.rules import FlightConstants, Rules
from bnb.game.state import Bullets, Fields, Match, Ships, State, blank

from race_presets import preset as race_preset

# Race key -> (state part, field).
STATE_KEYS = {
    "x": ("ships", "x"), "y": ("ships", "y"), "c": ("ships", "heading_cos"),
    "s": ("ships", "heading_sin"), "w": ("ships", "proper_speed"),
    "alpha": ("ships", "angle_of_attack"), "ca": ("ships", "angle_of_attack_cos"),
    "sa": ("ships", "angle_of_attack_sin"), "r": ("ships", "nose_rate"), "E": ("ships", "energy"),
    "logn": ("ships", "log_refractive_index"), "gx": ("ships", "log_index_gradient_x"),
    "gy": ("ships", "log_index_gradient_y"), "shield": ("ships", "shield"),
    "shield_wait": ("ships", "shield_recharge_wait"), "alive": ("ships", "alive"),
    "fired": ("ships", "fired"), "spawned": ("ships", "spawned"),
    "respawn": ("ships", "respawn_wait"), "gun_wait": ("ships", "gun_cooldown"),
    "cursor": ("ships", "next_slot"), "ship_type": ("ships", "ship_type"),
    "bx": ("bullets", "x"), "by": ("bullets", "y"), "bdx": ("bullets", "direction_x"),
    "bdy": ("bullets", "direction_y"), "bw": ("bullets", "proper_speed"),
    "blogn": ("bullets", "log_refractive_index"), "bgx": ("bullets", "log_index_gradient_x"),
    "bgy": ("bullets", "log_index_gradient_y"), "ttl": ("bullets", "moves_left"),
    "fx": ("fields", "x"), "fy": ("fields", "y"), "fr": ("fields", "radius"),
    "fw": ("fields", "band_width"), "flevel": ("fields", "level"),
    "t": ("match", "episode_tick"), "axis_x": ("match", "spawn_axis_x"),
    "axis_y": ("match", "spawn_axis_y"),
}
FLIGHT_KEYS = ("x", "y", "c", "s", "w", "alpha", "ca", "sa", "r", "E", "logn", "gx", "gy")
BULLET_KEYS = ("bx", "by", "bdx", "bdy", "bw", "blogn", "bgx", "bgy")
# presets.json keys -> bnb.game names.
FLIGHT_SHARED = dict(mass="mass", g="gravity", alpha_s="stall_angle", k_ind="induced_drag_factor",
                     e_knee="engine_knee", band="store_fade_band", a0="coast_exchange",
                     a_climb="climb_exchange", a_dive="dive_exchange",
                     soft_p="lift_softmin_exponent", zeta="nose_damping",
                     omega_floor="nose_frequency_floor", eps="speed_floor")
CARD = dict(cl_max="max_lift_coefficient", a_max="max_acceleration", cd0="zero_lift_drag",
            power="engine_power", p_min="engine_power_full_store", f_static="static_thrust",
            e_max="max_energy", omega_c="corner_nose_frequency", shield_max="max_shield",
            hit_radius="hit_radius", damage="hit_damage", muzzle_speed="muzzle_speed",
            bullet_cost="bullet_energy_cost", fire_interval="ticks_between_shots",
            spawn_energy="spawn_energy")
RULES = dict(h="tick_length", episode_ticks="episode_ticks", slots="bullet_slots",
             bullet_moves="bullet_moves", bullet_drag="bullet_drag", spread="bullet_spread",
             head_on_fraction="head_on_fraction", shield_delay="shield_recharge_delay",
             shield_regen="shield_recharge_rate", boundary_base="boundary_damage_base",
             boundary_per_m="boundary_damage_per_metre", respawn_ticks="respawn_ticks",
             spawn_speed="spawn_speed", spawn_offset="spawn_offset", spawn_radius="spawn_radius",
             field_width="field_band_width", vision="vision_range")
RACE_SHIP = dict(x="x", y="y", c="heading_cos", s="heading_sin", vx="velocity_x",
                 vy="velocity_y", w="proper_speed", alpha="angle_of_attack", r="nose_rate",
                 E="energy", shield="shield", alive="alive", visible="visible", ally=None,
                 gun_wait="gun_cooldown", respawn="respawn_wait", type="ship_type")
TYPES = api.ship_types()


def rules(spec) -> Rules:
    p = race_preset(spec)
    return api.rules_of(GameConfig(p["num_ships"] // 2, p["num_fields"], p["arena_radius"]))


def to_torch(arrays: dict) -> dict:
    return {k: torch.as_tensor(np.asarray(v)) for k, v in arrays.items()}


def to_numpy(tensors: dict) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in tensors.items()}


def flat(state: State) -> dict:
    """The state under the race keys (no ticks_since_seen or fogged: the race has none)."""
    return {k: getattr(getattr(state, part), name) for k, (part, name) in STATE_KEYS.items()}


def unflat(p: Rules, arrays: dict) -> State:
    """A race state dict as a State; the parts the race lacks start blank (fogged)."""
    values = to_torch(arrays)
    base = blank(p, values["x"].shape[0])
    parts = {"ships": {}, "bullets": {}, "fields": {}, "match": dataclasses.asdict(base.match)}
    for key, (part, name) in STATE_KEYS.items():
        parts[part][name] = values[key]
    return State(ships=Ships(**parts["ships"]), bullets=Bullets(**parts["bullets"]),
                 fields=Fields(**parts["fields"]), match=Match(**parts["match"]))


def fields_of(arrays: dict) -> Fields:
    f = to_torch(arrays)
    return Fields(x=f["fx"], y=f["fy"], radius=f["fr"], band_width=f["fw"], level=f["flevel"])


def ships_of(arrays: dict) -> Ships:
    """Ships with the given race flight keys (and ship_type); every other field zero."""
    values = to_torch(arrays)
    zeros = torch.zeros_like(values["x"])
    given = {STATE_KEYS[k][1]: v for k, v in values.items() if k in STATE_KEYS}
    return Ships(**{f.name: given.get(f.name, zeros.int() if f.name == "ship_type" else zeros)
                    for f in dataclasses.fields(Ships)})


def race_observation(p: Rules, state: State) -> dict:
    _, obs, truth = api.perceive(p, state)
    B, T, N, _ = obs.ships.shape
    inverse = torch.argsort(truth.view_order[0], dim=-1)  # (T, N): row of each sim ship
    views = torch.arange(T)[:, None]
    ships = obs.ships[:, views, inverse]
    columns = []
    for race, name in RACE_SHIP.items():
        if race == "ally":
            columns.append((ships[..., SHIP["team_relative"]] == 0).float())
        elif race in ("gun_wait", "respawn"):
            columns.append(torch.round(ships[..., SHIP[name]] / p.tick_length))
        else:
            columns.append(ships[..., SHIP[name]])
    fields = obs.fields[:, 0][..., [FIELD[k] for k in ("x", "y", "radius", "level")]]
    return dict(ships=torch.stack(columns, -1), fields=fields, globals=obs.globals[:, 0, :2])


class RaceArm:
    name = "bnb"

    def fields(self, fields, x, y):
        out = medium(fields_of(fields), torch.as_tensor(x), torch.as_tensor(y))
        return tuple(v.numpy() for v in out)

    def flight_tick(self, ship, power, stick, fields, flight=None):
        """flight: presets.json flight overrides, applied to every ship type."""
        flight = dict(flight or {})
        constants = FlightConstants(**{FLIGHT_SHARED[k]: flight.pop(k) for k in list(flight)
                                       if k in FLIGHT_SHARED})
        types = api.ship_types(**{CARD[k]: v for k, v in flight.items()})
        sigma, pi = flight_module.controls(torch.as_tensor(power), torch.as_tensor(stick))
        ships = ships_of({k: ship[k] for k in ship if k in FLIGHT_KEYS + ("ship_type",)})
        airframe = flight_module.airframe_of(types, ships.ship_type)
        out = flight_module.flight_tick(constants, airframe, Rules.tick_length, ships, sigma, pi,
                                        fields_of(fields))
        return {k: getattr(out, STATE_KEYS[k][1]).numpy() for k in FLIGHT_KEYS}

    def bullet_tick(self, bullets, fields):
        values = to_torch(bullets)
        b = Bullets(**{STATE_KEYS[k][1]: values[k] for k in BULLET_KEYS},
                    moves_left=torch.ones_like(values["bx"], dtype=torch.int32))
        out = bullet_motion(rules("squad"), b, fields_of(fields))
        return {k: getattr(out, STATE_KEYS[k][1]).numpy() for k in BULLET_KEYS}

    def reset(self, preset, uniforms):
        p = rules(preset)
        return to_numpy(flat(api.reset(p, TYPES, torch.as_tensor(uniforms))))

    def step(self, preset, state, actions, spread, uniforms):
        p = rules(preset)
        st, ev = api.step(p, TYPES, unflat(p, state), Action.unpack(torch.as_tensor(actions)),
                          torch.as_tensor(spread), torch.as_tensor(uniforms))
        events = {k: getattr(ev, k) for k in ("damage", "boundary_damage", "died", "killer", "fired", "done")}
        return to_numpy(flat(st)), to_numpy(events), to_numpy(race_observation(p, st))

    def observe(self, preset, state):
        p = rules(preset)
        return to_numpy(race_observation(p, unflat(p, state)))

    def chaser(self, preset, state):
        from race_chaser import chaser

        p = rules(preset)
        return chaser(p, unflat(p, state)).numpy()
