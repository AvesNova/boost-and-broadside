"""Static game configuration: counts and sizes the compiled graphs specialize on."""

import dataclasses

TEAMS = 2  # a game has exactly two teams (author decision): T in shapes
MAX_SHIPS = 255  # stored ship indices are int32 with -1 for none; the cap is a design limit
MAX_FIELDS = 64  # sight-line core membership is an int64 bitmask
TEAM_DEATHMATCH, FRONTLINE, SANDBOX = "team_deathmatch", "frontline", "sandbox"
ELIMINATION = "elimination"
MODES = (TEAM_DEATHMATCH, FRONTLINE, ELIMINATION, SANDBOX)
# The modes a fixed layout may place: those whose spawns draw tick uniforms of their own
# (the jitter reads them) and own no zones (Frontline's mirror the fields about its ring).
LAYOUT_MODES = (TEAM_DEATHMATCH, ELIMINATION)
STANDARD_SHIP_SET = "standard"


@dataclasses.dataclass(frozen=True)
class FixedField:
    """A field of a fixed layout, and with it its image under the map's 180-degree rotation
    (the game's maps are point-symmetric): centre x, y (m, from the map centre, before the
    layout's rotation), radius (m), and the levels it draws from, uniformly at every new
    match, its image drawing the same."""

    x: float
    y: float
    radius: float
    levels: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class FixedSpawn:
    """Where a team 0 slot spawns in a fixed layout, team 1's same slot at its image under
    the 180-degree rotation: x, y (m, from the map centre, before the layout's rotation);
    heading (rad, from +x, counterclockwise) or None, toward the map centre; position_jitter
    (m), a uniform point in a disc of this radius around it; heading_jitter (rad), a uniform
    turn within plus or minus this."""

    x: float
    y: float
    heading: float | None = None
    position_jitter: float = 0.0
    heading_jitter: float = 0.0


@dataclasses.dataclass(frozen=True)
class MapLayout:
    """A fixed map in place of the random one (a game without a layout draws its map).

    fields: half the map's fields, each with its point image (`FixedField`), so 2 x
        len(fields) is the game's field count; empty: the fields are drawn as without a
        layout.
    spawns: team 0's spawn point per slot (`FixedSpawn`), team 1's their images; empty:
        the mode places the ships as without a layout.
    rotated: the whole layout turns about the map centre by a uniformly random angle at
        every new match (the spawn axis's draw), so a fixed map is met at every bearing as
        random maps are; False keeps it as written.
    """

    fields: tuple[FixedField, ...] = ()
    spawns: tuple[FixedSpawn, ...] = ()
    rotated: bool = True


def layout_of(data: dict) -> MapLayout:
    """A layout from its JSON data (`dataclasses.asdict`'s, lists for tuples)."""
    return MapLayout(
        fields=tuple(
            FixedField(
                float(field["x"]),
                float(field["y"]),
                float(field["radius"]),
                tuple(int(level) for level in field["levels"]),
            )
            for field in data.get("fields", ())
        ),
        spawns=tuple(
            FixedSpawn(
                float(spawn["x"]),
                float(spawn["y"]),
                None if spawn.get("heading") is None else float(spawn["heading"]),
                float(spawn.get("position_jitter", 0.0)),
                float(spawn.get("heading_jitter", 0.0)),
            )
            for spawn in data.get("spawns", ())
        ),
        rotated=bool(data.get("rotated", True)),
    )


@dataclasses.dataclass(frozen=True)
class GameConfig:
    """Sizes and rules of one match. Everything else comes from the game's constants.

    Teams are equal in size and team-major: ship `n = team * ships_per_team + slot`.
    A batch of several games (mixed spec 1) shares the sizes (ships, fields); mode,
    arena_radius, spawn_groups, episode_seconds and respawn_seconds are then the defaults its pools
    override (`Pool`).

    mode: the game's rules (modes spec 1): "team_deathmatch", "frontline", "elimination"
        (team deathmatch without respawns: a team with no ship alive loses; at the clock
        more ships alive wins, then more total shield) or "sandbox" (no score and no win,
        formation spawns: the physics game the race tests run).
    episode_seconds, respawn_seconds: time-native rules, whole multiples of the tick or the
        rules fail to build (ticks_of); None takes the mode's (the game's `modes.rules`).
        Elimination has no respawn: a dead ship waits out the episode, so it takes no
        respawn_seconds.
    spawn_groups: team deathmatch only, ships per team in each spawn group (summing to
        ships_per_team); empty means uniform spawns.
    kills_per_ship: team deathmatch's winning score is kills_per_ship * ships_per_team.
    ship_set: the ship types and the rules their cards were solved under
        (`bnb.game.ships`): "standard", or another set's directory name ("real-scale").
        The whole batch shares it (a pool cannot change it).
    layout: a fixed map (`MapLayout`: fields, spawn points) in team deathmatch or
        elimination; None draws the map at random at every new match.
    """

    ships_per_team: int
    num_fields: int
    arena_radius: float  # m
    episode_seconds: float | None = None
    respawn_seconds: float | None = None
    mode: str = TEAM_DEATHMATCH
    spawn_groups: tuple[int, ...] = ()
    kills_per_ship: int = 5
    ship_set: str = STANDARD_SHIP_SET
    layout: MapLayout | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.ships_per_team or self.num_ships > MAX_SHIPS:
            raise ValueError(f"need 1 <= ships per team and at most {MAX_SHIPS} ships")
        if not 0 <= self.num_fields <= MAX_FIELDS:
            raise ValueError(f"need 0 to {MAX_FIELDS} fields, got {self.num_fields}")
        if not self.arena_radius > 0.0:
            raise ValueError("arena radius must be positive")
        if self.mode not in MODES:
            raise ValueError(f"mode {self.mode!r} is not one of {MODES}")
        if self.spawn_groups and (
            self.mode != TEAM_DEATHMATCH
            or min(self.spawn_groups) < 1
            or sum(self.spawn_groups) != self.ships_per_team
        ):
            raise ValueError(
                f"spawn groups {self.spawn_groups} must be positive team "
                f"deathmatch groups summing to {self.ships_per_team} ships"
            )
        if self.kills_per_ship < 1:
            raise ValueError("kills_per_ship must be at least 1")
        if self.mode == ELIMINATION and self.respawn_seconds is not None:
            raise ValueError("elimination has no respawn: leave respawn_seconds out")
        if self.layout is not None:
            self._check_layout(self.layout)

    def _check_layout(self, layout: MapLayout) -> None:
        if self.mode not in LAYOUT_MODES:
            raise ValueError(f"a fixed layout needs one of the modes {LAYOUT_MODES}")
        if layout.fields and 2 * len(layout.fields) != self.num_fields:
            raise ValueError(
                f"a layout of {len(layout.fields)} fields and their images is "
                f"{2 * len(layout.fields)} fields, not the game's {self.num_fields}"
            )
        if layout.spawns and len(layout.spawns) != self.ships_per_team:
            raise ValueError(
                f"a layout of {len(layout.spawns)} spawn points needs "
                f"{len(layout.spawns)} ships per team, not {self.ships_per_team}"
            )
        if layout.spawns and self.spawn_groups:
            raise ValueError("a layout's spawn points leave no spawn groups to draw")
        for field in layout.fields:
            if field.radius <= 0.0 or not field.levels or 0 in field.levels:
                raise ValueError(f"field {field}: a positive radius and non-zero levels")

    @property
    def num_ships(self) -> int:
        """N: TEAMS times ships_per_team."""
        return TEAMS * self.ships_per_team


def ticks_of(seconds: float, tick: float) -> int:
    """A time-native rule in whole ticks. Fails unless `seconds` is an exact multiple of `tick`
    (to 1e-9 relative), so every rule event lands on a decision step."""
    count = round(seconds / tick)
    if abs(count * tick - seconds) > 1e-9 * max(abs(seconds), tick):
        raise ValueError(f"{seconds} s is not a whole number of {tick} s ticks")
    return count
