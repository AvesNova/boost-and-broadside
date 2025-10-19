from dataclasses import dataclass
from enum import StrEnum, auto


class EventType(StrEnum):
    DAMAGE = auto()
    DEATH = auto()
    WIN = auto()
    TIE = auto()


@dataclass
class GameEvent:
    event_type: EventType
    team_id: int | None = None
    ship_id: int | None = None
    amount: float | None = None  # damage, healing, etc.
