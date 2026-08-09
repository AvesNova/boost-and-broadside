"""Typed fleet-size parsing shared by evaluation commands."""

import re
from typing import NamedTuple

_MATCHUP_PATTERN = re.compile(r"(?P<team0>[1-9]\d*)(?:v(?P<team1>[1-9]\d*))?", re.ASCII)


class Matchup(NamedTuple):
    """Ship counts for team 0 and team 1.

    ``NamedTuple`` keeps the legacy tuple unpacking/equality contract while making
    side ownership explicit to new callers.
    """

    team0: int
    team1: int

    @property
    def num_ships(self) -> int:
        """Total ships in the environment."""
        return self.team0 + self.team1

    def __str__(self) -> str:
        return f"{self.team0}v{self.team1}"


DEFAULT_MATCHUP = Matchup(4, 4)


class MatchupParseError(ValueError):
    """A fleet-size string is malformed or has a non-positive side."""


def parse_matchup(spec: str) -> Matchup:
    """Parse ``4``, ``4v4``, or an asymmetric value such as ``3v5``.

    A bare count applies to both teams. Whitespace, signs, zero, extra separators,
    and partial integers are rejected instead of being interpreted or skipped.
    """
    match = _MATCHUP_PATTERN.fullmatch(spec)
    if match is None:
        raise MatchupParseError(
            f"invalid matchup {spec!r}; expected a positive count or TEAM0vTEAM1"
        )
    team0 = int(match.group("team0"))
    team1_text = match.group("team1")
    return Matchup(team0, team0 if team1_text is None else int(team1_text))


def parse_matchups(specs: list[str] | tuple[str, ...] | None) -> list[Matchup]:
    """Parse a list, using the project-wide 4v4 default when omitted."""
    return [DEFAULT_MATCHUP] if not specs else [parse_matchup(spec) for spec in specs]

