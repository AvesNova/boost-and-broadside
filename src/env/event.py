"""
Game event system for tracking significant occurrences during gameplay.

Defines event types and the GameEvent dataclass for logging damage, deaths, and outcomes.
"""

from dataclasses import dataclass
from enum import StrEnum, auto


class EventType(StrEnum):
    """Types of events that can occur during gameplay."""

    DAMAGE = auto()
    DEATH = auto()
    WIN = auto()
    TIE = auto()


@dataclass
class GameEvent:
    """
    Represents a single game event.

    Attributes:
        event_type: The type of event that occurred.
        team_id: ID of the team involved (if applicable).
        ship_id: ID of the ship involved (if applicable).
        amount: Numerical value associated with the event (e.g., damage amount).
    """

    event_type: EventType
    team_id: int | None = None
    ship_id: int | None = None
    amount: float | None = None
