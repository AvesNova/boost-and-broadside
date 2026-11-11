"""Static game configuration: counts and sizes the compiled graphs specialize on."""

import dataclasses

TEAMS = 2  # a game has exactly two teams (author decision): T in shapes
MAX_SHIPS = 255  # stored ship indices are int32 with -1 for none; the cap is a design limit
MAX_FIELDS = 64  # sight-line core membership is an int64 bitmask
TEAM_DEATHMATCH, FRONTLINE, SANDBOX = "team_deathmatch", "frontline", "sandbox"
MODES = (TEAM_DEATHMATCH, FRONTLINE, SANDBOX)
STANDARD_SHIP_SET = "standard"


@dataclasses.dataclass(frozen=True)
class GameConfig:
    """Sizes and rules of one match. Everything else comes from the game's constants.

    Teams are equal in size and team-major: ship `n = team * ships_per_team + slot`.
    A batch of several games (mixed spec 1) shares the sizes (ships, fields); mode,
    arena_radius, spawn_groups, episode_seconds and respawn_seconds are then the defaults its pools
    override (`Pool`).

    mode: the game's rules (modes spec 1): "team_deathmatch", "frontline" or "sandbox" (no
        score and no win, formation spawns: the physics game the race tests run).
    episode_seconds, respawn_seconds: time-native rules, whole multiples of the tick or the
        rules fail to build (ticks_of); None takes the mode's (the game's `modes.rules`).
    spawn_groups: team deathmatch only, ships per team in each spawn group (summing to
        ships_per_team); empty means uniform spawns.
    kills_per_ship: team deathmatch's winning score is kills_per_ship * ships_per_team.
    ship_set: the ship types and the rules their cards were solved under
        (`bnb.game.ships`): "standard", or another set's directory name ("real-scale").
        The whole batch shares it (a pool cannot change it).
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
