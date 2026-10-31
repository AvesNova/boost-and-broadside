"""The fixed numerics of the game (game spec 1, 3.2): plain Python numbers.

Compiled graphs specialize on these, so they cost nothing at run time. The ship-card
parameters that differ by ship type are tensors instead (`ship_types.ShipTypes`), so one
compiled graph serves any values of them.
"""

import dataclasses
import math

from bnb.contracts import GameConfig, ticks_of

STICK_LEVELS = (0.0, -0.4, 0.4, -0.75, 0.75, -1.0, 1.0)  # sigma by stick index; + turns left
POWER_MODES = (0.0, -1.0, 1.0)  # pi by power index: coast, boost, reverse
FIELD_LEVELS = (-2, -1, 1, 2)
LN_SQRT2 = 0.5 * math.log(2.0)
MAP_SCALE_RADIUS = 2600.0  # s_a = R_a / 2600


@dataclasses.dataclass(frozen=True)
class FlightConstants:
    """Game spec 3.2, the constants every ship type shares."""

    mass: float = 1.0  # kg
    g: float = 27.0  # m/s^2, lift that holds level flight
    alpha_s: float = math.radians(15.0)  # stall angle of attack
    k_ind: float = 11.4  # induced-drag factor
    e_knee: float = 0.5  # store fraction where engine power starts to fall
    band: float = 0.02  # store-end fade band, fraction of E_max
    a0: float = 3.0  # coast exchange (m/s^2)
    a_climb: float = 27.0  # reverse exchange
    a_dive: float = 27.0  # boost exchange
    soft_p: float = 16.0  # soft-min exponent of the lift budget
    zeta: float = 0.7  # nose damping ratio
    omega_floor: float = 1.0  # nose frequency floor (rad/s)
    eps: float = 1e-6  # speed floor (m/s)


@dataclasses.dataclass(frozen=True)
class Rules:
    """Sizes and every shared constant of one game (game spec 1). Static: a different value
    is a different compiled graph."""

    ships_per_team: int
    num_fields: int
    arena_radius: float  # R_a (m)
    episode_ticks: int = 1800
    respawn_ticks: int = 240
    team_types: tuple[int, ...] = ()  # ship type per slot, the same for both teams
    num_teams: int = 2
    slots: int = 6  # K, bullet slots per ship
    tick: float = 1.0 / 30.0  # h (s)
    bullet_moves: int = 12
    bullet_drag: float = 8e-4  # k_b (1/m)
    spread: float = 12.0  # sigma_b, per axis (m/s, proper)
    head_on_fraction: float = 0.3
    shield_delay: float = 6.0  # s
    shield_regen: float = 12.0  # per s
    boundary_base: float = 5.0  # per s
    boundary_per_m: float = 0.05  # per s per m of depth
    spawn_speed: float = 30.0  # proper (m/s)
    spawn_offset: float = 0.8  # spawn centres at +-0.8 R_a along the spawn axis
    spawn_radius: float = 120.0  # rho_s before map scaling (m)
    field_width: float = 40.0  # b (m)
    vision: float = 1024.0  # V (m)
    flight: FlightConstants = FlightConstants()

    def __post_init__(self) -> None:
        if not self.team_types:  # within a team, slots alternate starting with a fighter
            object.__setattr__(self, "team_types", tuple(i % 2 for i in range(self.ships_per_team)))
        if len(self.team_types) != self.ships_per_team:
            raise ValueError("team_types needs one ship type per slot")

    @property
    def num_ships(self) -> int:
        return self.num_teams * self.ships_per_team

    @property
    def slot_types(self) -> tuple[int, ...]:
        """Ship type of every sim index (team-major)."""
        return self.team_types * self.num_teams

    @property
    def map_scale(self) -> float:
        return self.arena_radius / MAP_SCALE_RADIUS

    @property
    def map_uniforms(self) -> int:
        """Uniforms per environment for a map: 4 per field plus the spawn axis."""
        return 4 * self.num_fields + 1

    @property
    def num_uniforms(self) -> int:
        """Uniforms per environment per tick (game spec 2): the map, a formation rotation
        per team, a formation key per ship."""
        return self.map_uniforms + self.num_teams + self.num_ships


def rules_of(cfg: GameConfig, **overrides) -> Rules:
    """The rules for a config's sizes and time-native rules; `overrides` replace defaults."""
    tick = overrides.pop("tick", Rules.tick)
    return Rules(ships_per_team=cfg.ships_per_team, num_fields=cfg.num_fields,
                 arena_radius=cfg.arena_radius, tick=tick,
                 episode_ticks=ticks_of(cfg.episode_s, tick),
                 respawn_ticks=ticks_of(cfg.respawn_s, tick), **overrides)
