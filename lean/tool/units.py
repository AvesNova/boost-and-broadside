"""The tool's units, in one place: the server (chart axes, labels, notes) and the page (sliders,
readouts, the sandbox's HUD; it gets `descriptors()` from /api/presets) both convert through this.

Three systems: "game" (metric at the game's time scale), "real_metric" (real time: the game's
speeds, rates and Ps divided by the time factor k, seconds times k) and "real_imperial" (real
time, mph, ft, ft/min for Ps as the energy-manoeuvrability charts show it). Turn rates are deg/s
and loads g in all three. The solve and the saved presets stay in game units; only the display
converts. A value shows as game_value * scale * k**-time_power.
"""

import math

SYSTEMS = ("game", "real_metric", "real_imperial")
LEGACY = {"real": "real_metric"}  # the old two-way switch
NAMES = {
    "game": "Game (metric, game time)",
    "real_metric": "Real, metric",
    "real_imperial": "Real, imperial",
}
FEET, MILES_PER_HOUR, FEET_PER_MINUTE = 3.280839895, 2.236936292, 196.8503937
HORSEPOWER = 745.7  # W
DEGREES = 180.0 / math.pi

# kind: (time power, metric label, imperial label, metric scale, imperial scale); the game system
# has time power 0 (its values are the game's own)
KINDS = {
    "length": (0, "m", "ft", 1.0, FEET),
    "speed": (1, "m/s", "mph", 1.0, MILES_PER_HOUR),
    "acceleration": (2, "m/s2", "ft/s2", 1.0, FEET),
    "angular_rate": (1, "deg/s", "deg/s", DEGREES, DEGREES),
    "frequency": (1, "1/s", "1/s", 1.0, 1.0),
    "time": (-1, "s", "s", 1.0, 1.0),
    "inverse_length": (0, "1/m", "1/ft", 1.0, 1.0 / FEET),
    "climb": (1, "m/s", "ft/min", 1.0, FEET_PER_MINUTE),
    "power": (3, "kW", "hp", 1e-3, 1.0 / HORSEPOWER),
}
# radius guides on the doghouse, round numbers in the system's own length
RADIUS_GUIDES = {
    "game": (150, 200, 300, 500, 1000, 2000),
    "real_metric": (150, 200, 300, 500, 1000, 2000),
    "real_imperial": (500, 700, 1000, 1500, 3000, 6000),
}


def system_of(name: str | None) -> str:
    """A system's name (the old "real" is real_metric; None is game; unknown names raise)."""
    name = LEGACY.get(name, name) or "game"
    if name not in SYSTEMS:
        raise ValueError(f"unknown units {name!r}; one of {SYSTEMS}")
    return name


def descriptors() -> dict:
    """{system: {kind: {scale, time_power, label}}}: what the page needs to convert."""
    out = {}
    for system in SYSTEMS:
        out[system] = {}
        for kind, (power, metric, imperial, metric_scale, imperial_scale) in KINDS.items():
            out[system][kind] = dict(
                scale=imperial_scale if system == "real_imperial" else metric_scale,
                time_power=0 if system == "game" else power,
                label=imperial if system == "real_imperial" else metric,
            )
    return out


_DESCRIPTORS = descriptors()


def factor(kind: str, system: str, time_scale: float) -> float:
    """The number a game value of this kind is multiplied by to display it."""
    d = _DESCRIPTORS[system_of(system)][kind]
    return d["scale"] * time_scale ** -d["time_power"]


def label(kind: str, system: str) -> str:
    return _DESCRIPTORS[system_of(system)][kind]["label"]


def to_display(kind: str, value, system: str, time_scale: float):
    return value * factor(kind, system, time_scale)


def from_display(kind: str, value, system: str, time_scale: float):
    return value / factor(kind, system, time_scale)


def describe(system: str, time_scale: float) -> str:
    """One line for a chart's note."""
    system = system_of(system)
    if system == "game":
        return f"Game units at {time_scale:g}x real time (m, m/s)."
    unit = "mph, ft, ft/min" if system == "real_imperial" else "m, m/s"
    return (
        f"Real units ({unit}): the game at {time_scale:g}x shown at 1x"
        f" (speeds, rates, Ps / {time_scale:g})."
    )
