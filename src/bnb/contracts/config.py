"""Static game configuration: counts and sizes the compiled graphs specialize on."""

import dataclasses

MAX_SHIPS = 255  # ship indices fit in uint8; 255 is reserved for "none"
MAX_FIELDS = 64  # sight-line core membership is an int64 bitmask
MODES = ("team_deathmatch", "frontline", "sandbox")


@dataclasses.dataclass(frozen=True)
class GameConfig:
    """Sizes and rules of one match. Everything else comes from the game's constants.

    Teams are equal in size and team-major: ship `n = team * ships_per_team + slot`.

    mode: the game's rules (modes spec 1): "team_deathmatch", "frontline" or "sandbox" (no
        score and no win, formation spawns: the physics game the race tests run).
    episode_s, respawn_s: time-native rules, whole multiples of the tick or the rules fail
        to build (ticks_of); None takes the mode's (the game's `modes.rules`).
    spawn_groups: team deathmatch only, ships per team in each spawn group (summing to
        ships_per_team); empty means uniform spawns.
    kills_per_ship: team deathmatch's winning score is kills_per_ship * ships_per_team.
    """

    ships_per_team: int
    num_fields: int
    arena_radius: float  # m
    episode_s: float | None = None
    respawn_s: float | None = None
    num_teams: int = 2
    mode: str = "team_deathmatch"
    spawn_groups: tuple[int, ...] = ()
    kills_per_ship: int = 5

    def __post_init__(self) -> None:
        if self.num_teams != 2:
            raise ValueError("only two teams are supported for now")
        if not 1 <= self.ships_per_team or self.num_ships > MAX_SHIPS:
            raise ValueError(f"need 1 <= ships per team and at most {MAX_SHIPS} ships")
        if not 1 <= self.num_fields <= MAX_FIELDS:
            raise ValueError(f"need 1 to {MAX_FIELDS} fields, got {self.num_fields}")
        if not self.arena_radius > 0.0:
            raise ValueError("arena radius must be positive")
        if self.mode not in MODES:
            raise ValueError(f"mode {self.mode!r} is not one of {MODES}")
        if self.spawn_groups and (self.mode != "team_deathmatch"
                                  or min(self.spawn_groups) < 1
                                  or sum(self.spawn_groups) != self.ships_per_team):
            raise ValueError(f"spawn groups {self.spawn_groups} must be positive team "
                             f"deathmatch groups summing to {self.ships_per_team} ships")
        if self.kills_per_ship < 1:
            raise ValueError("kills_per_ship must be at least 1")

    @property
    def num_ships(self) -> int:
        return self.num_teams * self.ships_per_team


def ticks_of(seconds: float, tick: float) -> int:
    """A time-native rule in whole ticks. Fails unless `seconds` is an exact multiple of `tick`
    (to 1e-9 relative), so every rule event lands on a decision step."""
    count = round(seconds / tick)
    if abs(count * tick - seconds) > 1e-9 * max(abs(seconds), tick):
        raise ValueError(f"{seconds} s is not a whole number of {tick} s ticks")
    return count
