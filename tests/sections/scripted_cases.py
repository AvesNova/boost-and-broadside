"""Constructed observations for the scripted agent's behavioural tests (scripted spec A7)."""

import math

import torch

from bnb.contracts import GameConfig, Obs
from bnb.contracts.obs import GLOBAL, GLOBAL_COLUMNS, SHIP, SHIP_COLUMNS, ZONE, ZONE_COLUMNS
from bnb.game import api
from bnb.game.handling import characteristics

RULES = api.rules_of(GameConfig(2, 1, 2600.0))
TYPES = api.ship_types()
CHARACTERISTICS = characteristics(TYPES, RULES)
ARENA_RADIUS = 2600.0


def value(name: str, ship_type: int) -> float:
    return float(getattr(CHARACTERISTICS, name)[ship_type])


def ship(x: float = 0.0, y: float = 0.0, heading: float = 0.0, speed: float | None = None,
         ship_type: int = 0, team_relative: int = 0, alive: bool = True, visible: bool = True,
         energy: float | None = None, shield: float | None = None, gun_cooldown: float = 0.0,
         velocity: tuple[float, float] | None = None) -> torch.Tensor:
    """One observation row: flying along its nose at `speed` (cruise by default), with a
    full store and shield unless given."""
    speed = value("cruise_speed", ship_type) if speed is None else speed
    row = torch.zeros(len(SHIP_COLUMNS))
    velocity = velocity or (speed * math.cos(heading), speed * math.sin(heading))
    shown = dict(alive=float(alive), visible=float(visible), ship_type=float(ship_type),
                 team_relative=float(team_relative))
    hidden = dict(x=x, y=y, heading_cos=math.cos(heading), heading_sin=math.sin(heading),
                  velocity_x=velocity[0], velocity_y=velocity[1], proper_speed=speed,
                  energy=value("max_energy", ship_type) if energy is None else energy,
                  shield=value("max_shield", ship_type) if shield is None else shield,
                  gun_cooldown=gun_cooldown)
    for name, number in shown.items():
        row[SHIP[name]] = number
    if visible:
        for name, number in hidden.items():
            row[SHIP[name]] = number
    return row


def absent(ship_type: int = 0, team_relative: int = 1) -> torch.Tensor:
    return ship(ship_type=ship_type, team_relative=team_relative, alive=False, visible=False)


def zone(x: float, y: float, role: int, progress: float = 0.0, direction: float = 0.0,
         radius: float = 330.0) -> torch.Tensor:
    row = torch.zeros(len(ZONE_COLUMNS))
    for name, number in dict(x=x, y=y, radius=radius, role=role, progress=progress,
                             direction=direction, present=1.0).items():
        row[ZONE[name]] = number
    return row


def observation(own: list[torch.Tensor], enemies: list[torch.Tensor],
                zones: list[torch.Tensor] | None = None,
                arena_radius: float = ARENA_RADIUS) -> Obs:
    """One view (R = 1); Frontline when zones are given, team deathmatch otherwise."""
    assert len(own) == len(enemies)
    rows = torch.stack(own + enemies)
    globals_ = torch.zeros(len(GLOBAL_COLUMNS))
    globals_[GLOBAL["arena_radius"]] = arena_radius
    globals_[GLOBAL["fogged"]] = 1.0
    globals_[GLOBAL["mode_frontline" if zones else "mode_team_deathmatch"]] = 1.0
    zone_rows = torch.stack(zones) if zones else torch.zeros((5, len(ZONE_COLUMNS)))
    return Obs(ships=rows[None], fields=None, globals=globals_[None], zones=zone_rows[None])
