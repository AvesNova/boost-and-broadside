from enum import IntEnum, StrEnum, auto


class Actions(IntEnum):
    """Action indices for ship control."""

    forward = 0
    backward = auto()
    left = auto()
    right = auto()
    sharp_turn = auto()
    shoot = auto()
