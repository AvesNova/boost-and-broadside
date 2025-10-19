from dataclasses import dataclass
from enum import StrEnum, auto


class EventType(StrEnum):
    DAMAGE = auto()
    DEATH = auto()
    WIN = auto()
    LOSS = auto()


@dataclass
class Event:
    event_type: EventType
    source_ship: int | None = None
    target_ship: int | None = None
    source_team: int | None = None
    target_team: int | None = None
    amount: float | None = None  # damage, healing, etc.
