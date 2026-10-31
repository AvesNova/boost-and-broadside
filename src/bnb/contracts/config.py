"""Static game configuration: counts and sizes the compiled graphs specialize on."""

import dataclasses

MAX_SHIPS = 255  # ship indices fit in uint8; 255 is reserved for "none"
MAX_FIELDS = 64  # sight-line core membership is an int64 bitmask


@dataclasses.dataclass(frozen=True)
class GameConfig:
    """Sizes of one match. Everything else comes from the physics constants for now.

    Teams are equal in size and team-major: ship `n = team * ships_per_team + slot`.
    """

    ships_per_team: int
    num_fields: int
    arena_radius: float  # m
    num_teams: int = 2

    def __post_init__(self) -> None:
        if self.num_teams != 2:
            raise ValueError("only two teams are supported for now")
        if not 1 <= self.ships_per_team or self.num_ships > MAX_SHIPS:
            raise ValueError(f"need 1 <= ships per team and at most {MAX_SHIPS} ships")
        if not 1 <= self.num_fields <= MAX_FIELDS:
            raise ValueError(f"need 1 to {MAX_FIELDS} fields, got {self.num_fields}")
        if not self.arena_radius > 0.0:
            raise ValueError("arena radius must be positive")

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
