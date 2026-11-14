"""The cheese agents: six crude one-trick opponents, to check that a trained policy cannot be
beaten by a single plain strategy and to give the league a few such players.

Each is a stateless function of the observation, playing by the same rules as every
controller: it reads only the ship rows, the zone rows and the globals, uses an enemy row only
where it is visible and alive, and draws nothing (the decision key is unused: the same
observation gives the same action). Each is batched tensor code with no host reads, so it
captures in a CUDA graph and compiles like the scripted agent. The flight reuses the scripted
chaser's bearing-to-stick rule (`bnb.controllers.scripted.stick_toward`).

Every agent keeps inside the shrinking boundary (`keep_inside`), and when it has no enemy to
play against patrols the map centre (`patrol`). Outside Frontline the two zone agents orbit the
map centre in place of a zone.
"""

import math
from typing import NamedTuple

import torch

from bnb.contracts import Action, Obs, Outputs
from bnb.contracts.obs import GLOBAL, SHIP, ZONE
from bnb.game.handling.characteristics import Characteristics
from bnb.game.ship_types import per_ship

from .scripted import FIRE_BEARING, SHOOT_IN_GUN_RANGES, stick_toward
from .scripted_combat import TINY, signed_angle

PREFIX = "scripted:"  # a cheese agent's player name is PREFIX + its name
COAST, BOOST, REVERSE = 0, 1, 2  # the power actions
ENEMY_DEFENSE, OWN_DEFENSE, OWN_SPAWN = 3, 1, 0  # zone roles for the viewer (ZONE_COLUMNS)

EDGE_MARGIN = 0.25  # the boundary's inner band, in boundary radii, where a ship turns inward
ORBIT_OUTER, ORBIT_INNER = 0.8, 0.4  # a zone orbit's radii, in zone radii: head in outside the
# outer, fly tangent inside the inner, a blend between
PATROL_RADIUS = 0.4  # the fallback patrol's radius, in boundary radii
FORMATION_RADIUS = 0.5  # deathball: the formation's radius, in gun ranges
STANDOFF = 0.9  # kiter: the distance it holds, in gun ranges
STANDOFF_BAND = 0.1  # kiter: the band around it where it neither closes nor backs off
NEAR_ZONE = 1.5  # turtle: enemies within this many zone radii of its zone are engaged


class Sight(NamedTuple):
    """What a decision reads: the own ships (R, S) and the visible living enemies (R, E)."""

    x: torch.Tensor
    y: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    alive: torch.Tensor  # (R, S) bool
    reach: torch.Tensor  # (R, S) each ship's gun range, m
    enemy_x: torch.Tensor
    enemy_y: torch.Tensor
    seen: torch.Tensor  # (R, E) bool: visible and alive
    boundary: torch.Tensor  # (R,) the boundary's radius now, m
    zones: torch.Tensor  # (R, Z, columns)
    index: torch.Tensor  # (S,) the ship's slot


def look(obs: Obs, characteristics: Characteristics, ships_per_team: int) -> Sight:
    """The own ships and the visible living enemies of views (R, ...)."""
    S = ships_per_team
    own, enemy = obs.ships[:, :S], obs.ships[:, S:]
    globals_ = obs.globals
    return Sight(
        own[..., SHIP["x"]],
        own[..., SHIP["y"]],
        own[..., SHIP["heading_cos"]],
        own[..., SHIP["heading_sin"]],
        own[..., SHIP["alive"]] > 0.5,
        per_ship(characteristics.gun_range, own[..., SHIP["ship_type"]].round().long()),
        enemy[..., SHIP["x"]],
        enemy[..., SHIP["y"]],
        (enemy[..., SHIP["visible"]] > 0.5) & (enemy[..., SHIP["alive"]] > 0.5),
        globals_[:, GLOBAL["boundary_fraction"]] * globals_[:, GLOBAL["arena_radius"]],
        obs.zones,
        torch.arange(S, device=own.device),
    )


def unit(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x, y) scaled to length 1, and the length (at least TINY)."""
    length = torch.sqrt(x * x + y * y).clamp_min(TINY)
    return x / length, y / length, length


def closest(
    sight: Sight, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The offset (R, K) from each point (x, y) (R, K) to its nearest visible living enemy
    (ties to the lowest row), and whether there is one (R, K). The offset is 0 without."""
    to_x = sight.enemy_x[:, None, :] - x[..., None]
    to_y = sight.enemy_y[:, None, :] - y[..., None]
    squared = torch.where(sight.seen[:, None, :], to_x * to_x + to_y * to_y, math.inf)
    pick = squared.argmin(-1, keepdim=True)
    found = sight.seen.any(-1)[:, None].expand_as(x)
    to_x, to_y = to_x.gather(-1, pick)[..., 0], to_y.gather(-1, pick)[..., 0]
    return torch.where(found, to_x, 0.0), torch.where(found, to_y, 0.0), found


def keep_inside(sight: Sight, direction_x: torch.Tensor, direction_y: torch.Tensor, extra=0.0):
    """A wanted direction (R, S) turned toward the map centre as the ship nears the boundary:
    all the way at the boundary, none inside its inner band of EDGE_MARGIN radii; `extra` (R,
    1, or 0) adds to that weight (a ship that wants the centre anyway)."""
    ahead_x, ahead_y, _ = unit(direction_x, direction_y)
    centre_x, centre_y, distance = unit(-sight.x, -sight.y)
    radius = sight.boundary[:, None]
    margin = EDGE_MARGIN * radius + 1.0
    weight = ((distance - (radius - margin)) / margin + extra).clamp(0.0, 1.0)
    return (1.0 - weight) * ahead_x + weight * centre_x, (
        1.0 - weight
    ) * ahead_y + weight * centre_y


def orbit(offset_x: torch.Tensor, offset_y: torch.Tensor, radius: torch.Tensor):
    """The direction that circles a point, given the offset to it and an orbit radius:
    toward it outside ORBIT_OUTER radii, tangent (counter-clockwise) inside ORBIT_INNER, a
    blend between."""
    to_x, to_y, distance = unit(offset_x, offset_y)
    span = (ORBIT_OUTER - ORBIT_INNER) * radius
    angle = 0.5 * math.pi * ((ORBIT_OUTER * radius - distance) / span).clamp(0.0, 1.0)
    return (
        torch.cos(angle) * to_x - torch.sin(angle) * to_y,
        torch.cos(angle) * to_y + torch.sin(angle) * to_x,
    )


def patrol(sight: Sight):
    """The fallback direction without an enemy: orbit the map centre at PATROL_RADIUS of the
    boundary."""
    return orbit(-sight.x, -sight.y, PATROL_RADIUS * sight.boundary[:, None])


class Play(NamedTuple):
    """One agent's decision per own ship (R, S): power, stick, shoot (int)."""

    power: torch.Tensor
    stick: torch.Tensor
    shoot: torch.Tensor


def fly(
    sight: Sight,
    direction_x: torch.Tensor,
    direction_y: torch.Tensor,
    power: torch.Tensor,
    shoot: torch.Tensor | None = None,
    extra=0.0,
) -> Play:
    """Turn toward a wanted direction (kept inside the boundary) at `power`, shooting where
    `shoot` (bool; never when None). Dead ships take the no-op."""
    wanted_x, wanted_y = keep_inside(sight, direction_x, direction_y, extra)
    stick = stick_toward(signed_angle(sight.cos, sight.sin, wanted_x, wanted_y))
    shoot = torch.zeros_like(sight.alive) if shoot is None else shoot
    return Play(
        torch.where(sight.alive, power, 0),
        torch.where(sight.alive, stick, 0),
        torch.where(sight.alive & shoot, 1, 0),
    )


def aimed(sight: Sight, to_x: torch.Tensor, to_y: torch.Tensor, found: torch.Tensor):
    """Whether each ship's nose is within FIRE_BEARING of the offset and the target within the
    gun's reach (as the chaser's shot)."""
    bearing = signed_angle(sight.cos, sight.sin, to_x, to_y).abs()
    distance = torch.sqrt(to_x * to_x + to_y * to_y)
    return found & (bearing < FIRE_BEARING) & (distance < SHOOT_IN_GUN_RANGES * sight.reach)


def zone_of(sight: Sight, role: int):
    """The zone with this role: (x, y, radius) (R, 1), the map centre with a patrol's radius
    where the view has none (a mode without zones)."""
    zones = sight.zones
    mask = (zones[..., ZONE["present"]] > 0.5) & (zones[..., ZONE["role"]] == role)
    found = mask.any(-1)

    def pick(name: str, fallback: torch.Tensor):
        value = torch.where(mask, zones[..., ZONE[name]], 0.0).sum(-1)
        return torch.where(found, value, fallback)[:, None]

    return (
        pick("x", torch.zeros_like(sight.boundary)),
        pick("y", torch.zeros_like(sight.boundary)),
        pick("radius", PATROL_RADIUS * sight.boundary),
    )


def zone_rusher(sight: Sight) -> Play:
    """Ignore combat: full boost to the enemy's defense zone (the zone the team has yet to
    take; the roles rotate at each capture, so it is always the next), and once inside it
    orbit, coasting, until it is captured. Never fires."""
    x, y, radius = zone_of(sight, ENEMY_DEFENSE)
    offset_x, offset_y = x - sight.x, y - sight.y
    inside = offset_x * offset_x + offset_y * offset_y < radius * radius
    direction_x, direction_y = orbit(offset_x, offset_y, radius)
    return fly(sight, direction_x, direction_y, torch.where(inside, COAST, BOOST))


def turtle(sight: Sight) -> Play:
    """Orbit the zones the team holds, coasting: its own defense zone and its own spawn zone,
    the ships spread over them by slot (even slots the defense, odd the spawn). Engage the
    nearest visible enemy within NEAR_ZONE zone radii of its zone: turn to it and fire when
    aligned; leave the zone for it only while it is there."""
    even = (sight.index % 2 == 0)[None, :]
    defense, spawn = zone_of(sight, OWN_DEFENSE), zone_of(sight, OWN_SPAWN)
    x, y, radius = (torch.where(even, a, b) for a, b in zip(defense, spawn, strict=True))
    to_x, to_y, found = closest(sight, x, y)  # the nearest enemy to the zone
    near = found & (to_x * to_x + to_y * to_y < (NEAR_ZONE * radius) ** 2)
    foe_x, foe_y = x + to_x - sight.x, y + to_y - sight.y  # the ship's offset to it
    orbit_x, orbit_y = orbit(x - sight.x, y - sight.y, radius)
    direction_x = torch.where(near, foe_x, orbit_x)
    direction_y = torch.where(near, foe_y, orbit_y)
    return fly(sight, direction_x, direction_y, 0, near & aimed(sight, foe_x, foe_y, near))


def evader(sight: Sight, shrink: torch.Tensor) -> Play:
    """Never engage: turn directly away from the nearest visible enemy, boosting while it is
    within two gun ranges, drifting to the map centre as the boundary shrinks (`shrink`
    (R,): 1 less the boundary's fraction of the arena). Never fires."""
    to_x, to_y, found = closest(sight, sight.x, sight.y)
    patrol_x, patrol_y = patrol(sight)
    away_x, away_y = torch.where(found, -to_x, patrol_x), torch.where(found, -to_y, patrol_y)
    close = found & (to_x * to_x + to_y * to_y < (2.0 * sight.reach) ** 2)
    return fly(sight, away_x, away_y, torch.where(close, BOOST, COAST), extra=shrink[:, None])


def deathball(sight: Sight) -> Play:
    """Hold a tight formation about the team's centre and focus fire: every ship turns toward
    the visible enemy nearest the formation's centre and fires at it when aligned, but a ship
    more than FORMATION_RADIUS gun ranges from the centre first returns to it. With no enemy
    visible the formation heads for the map centre."""
    count = sight.alive.sum(-1, keepdim=True).clamp_min(1)
    centre_x = (torch.where(sight.alive, sight.x, 0.0)).sum(-1, keepdim=True) / count
    centre_y = (torch.where(sight.alive, sight.y, 0.0)).sum(-1, keepdim=True) / count
    to_x, to_y, found = closest(sight, centre_x, centre_y)  # (R, 1)
    goal_x = torch.where(found, centre_x + to_x, 0.0) - sight.x
    goal_y = torch.where(found, centre_y + to_y, 0.0) - sight.y
    home_x, home_y = centre_x - sight.x, centre_y - sight.y
    straggling = home_x * home_x + home_y * home_y > (FORMATION_RADIUS * sight.reach) ** 2
    direction_x = torch.where(straggling, home_x, goal_x)
    direction_y = torch.where(straggling, home_y, goal_y)
    found = found.expand_as(sight.x)
    power = torch.where(found, BOOST, COAST)
    return fly(sight, direction_x, direction_y, power, aimed(sight, goal_x, goal_y, found))


def jouster(sight: Sight) -> Play:
    """Head-on at full boost to the nearest visible enemy, firing whenever aligned, with no
    defensive manoeuvre."""
    to_x, to_y, found = closest(sight, sight.x, sight.y)
    patrol_x, patrol_y = patrol(sight)
    return fly(
        sight,
        torch.where(found, to_x, patrol_x),
        torch.where(found, to_y, patrol_y),
        torch.where(found, BOOST, COAST),
        aimed(sight, to_x, to_y, found),
    )


def kiter(sight: Sight) -> Play:
    """Hold at STANDOFF gun ranges from the nearest visible enemy: beyond the band boost
    toward it, inside it turn away and boost (back off), in it aim, coast and fire."""
    to_x, to_y, found = closest(sight, sight.x, sight.y)
    distance = torch.sqrt(to_x * to_x + to_y * to_y)
    low, high = (STANDOFF - STANDOFF_BAND) * sight.reach, (STANDOFF + STANDOFF_BAND) * sight.reach
    backing = found & (distance < low)
    patrol_x, patrol_y = patrol(sight)
    direction_x = torch.where(found, torch.where(backing, -to_x, to_x), patrol_x)
    direction_y = torch.where(found, torch.where(backing, -to_y, to_y), patrol_y)
    power = torch.where(found & ((distance < low) | (distance > high)), BOOST, COAST)
    return fly(sight, direction_x, direction_y, power, aimed(sight, to_x, to_y, found))


NAMES = ("zone_rusher", "turtle", "evader", "deathball", "jouster", "kiter")
POLICIES = {
    "zone_rusher": lambda sight, shrink: zone_rusher(sight),
    "turtle": lambda sight, shrink: turtle(sight),
    "evader": evader,
    "deathball": lambda sight, shrink: deathball(sight),
    "jouster": lambda sight, shrink: jouster(sight),
    "kiter": lambda sight, shrink: kiter(sight),
}


def player_name(name: str) -> str:
    """The player name of a cheese agent: `scripted:<name>`."""
    return PREFIX + name


def agent_of(player: str) -> str | None:
    """The cheese agent a player name means (`scripted:<name>`), or None for another name."""
    name = player[len(PREFIX) :] if player.startswith(PREFIX) else None
    return name if name in NAMES else None


class Cheese:
    """The cheese agents as one controller. kinds: one agent's name (every view plays it), or
    a tensor (R,) int64 of each view's agent, an index into `NAMES` (-1: none, the no-op),
    written in place between segments (the league seat and the match seat set their rows'
    kinds); every agent's decision is then computed and each view takes its own. With a
    name only that agent is computed."""

    reads = ("ships", "zones", "globals")

    def __init__(
        self,
        characteristics: Characteristics,
        ships_per_team: int,
        kinds: str | torch.Tensor,
    ) -> None:
        if isinstance(kinds, str) and kinds not in NAMES:
            raise ValueError(f"cheese agent {kinds!r} is not one of {NAMES}")
        self.characteristics, self.ships_per_team, self.kinds = (
            characteristics,
            ships_per_team,
            kinds,
        )

    def init_carry(self, views: int, device: torch.device) -> None:
        """No carry: the agents are stateless."""
        return None

    def step(self, carry: None, obs: Obs, reset: torch.Tensor, decision_key: torch.Tensor):
        """One decision for views (R, ...): each view's agent's action per own ship. Draws
        nothing, so the decision key is unused."""
        sight = look(obs, self.characteristics, self.ships_per_team)
        globals_ = obs.globals
        shrink = 1.0 - globals_[:, GLOBAL["boundary_fraction"]]
        names = (self.kinds,) if isinstance(self.kinds, str) else NAMES
        packed = torch.zeros(sight.x.shape + (3,), dtype=torch.int32, device=sight.x.device)
        for name in names:
            play = POLICIES[name](sight, shrink)
            action = torch.stack([play.power, play.stick, play.shoot], -1).to(torch.int32)
            if isinstance(self.kinds, str):
                packed = action
            else:
                mine = (self.kinds == NAMES.index(name))[:, None, None]
                packed = torch.where(mine, action, packed)
        return Action.unpack(packed), carry, Outputs()
