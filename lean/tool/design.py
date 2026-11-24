"""What the web tool computes, through the game's own entry points (lean.game.tuning): a ship's
row from its inputs, the charts (drawn by the committed charts' matplotlib code,
docs/rewrite/sections/turns/doghouse.py) and the key numbers, and the flown checks
(lean.tool.flown). No physics lives here.

The inputs are generated from the tuning route itself (`inputs`): the handles `tuning.handles_of`
returns (the performance targets and the numbers v2's closed forms take), and every card field
`tuning.handles_row` takes from its base row rather than from the handles (the flight terms,
the stick levels, the nose; and anything a later card adds). A field the tuning starts to take
appears in the page with no change here; `INFO` only gives known inputs a label, a unit, a group
and a range. Switches: each game term `tuning.POINT_MASS_TERMS` can turn off (its off value),
and matching the low-speed acceleration (off: the preset's static thrust stays, as before).
"""

import io
import math
import threading

import numpy as np

from lean.game import card, tuning

from . import flown

SOLVE_SPEED_STEP = 0.5  # m/s: the handles solve on a coarser speed grid than the default, for speed
LOCK = threading.RLock()  # the card is process-global (tuning.flying): one computation at a time

# Handles the page does not offer: the square polar only (handles_row raises otherwise), and the
# zoom acceleration, which follows the dive's (one trade).
FIXED_HANDLES = ("half_lift_drag", "zoom_acceleration")
# The switch of the low-speed acceleration: matched (on) or the preset's static thrust (off).
MATCH = "low_speed_acceleration"

GROUPS = (
    "Performance targets",
    "Airframe",
    "Game terms",
    "Flaps",
    "Stick and nose",
    "Guns",
    "Shield, head-on and vision",
    "Other",
)
# name: (label, unit, group, low, high, relative): the range as multiples of the preset's value
# (relative) or in the input's units.
INFO = {
    "cruise_speed": ("cruise (top) speed", "m/s", "Performance targets", 0.6, 1.4, True),
    "sustained_turn_rate": (
        "settled full-stick turn rate",
        "rad/s",
        "Performance targets",
        0.5,
        1.6,
        True,
    ),
    "low_speed_acceleration": (
        "low-speed acceleration (matched when on)",
        "m/s2",
        "Performance targets",
        0.3,
        2.5,
        True,
    ),
    "dive_acceleration": ("dive and zoom trade", "m/s2", "Performance targets", 0.3, 1.6, True),
    "stall_speed": ("stall speed (1 g)", "m/s", "Airframe", 0.6, 1.5, True),
    "g_limit": ("g limit", "g", "Airframe", 0.4, 1.6, True),
    "zero_lift_drag": ("zero-lift drag", "1/m", "Airframe", 0.4, 2.5, True),
    "stall_angle": ("stall angle", "rad", "Airframe", 0.5, 1.6, True),
    "critical_speed": ("drag rise: critical speed", "m/s", "Game terms", 250.0, 1000.0, False),
    "drag_rise_exponent": ("drag rise: exponent", "", "Game terms", 2.0, 40.0, False),
    "separation_drag": ("separation drag (at maximum lift)", "", "Game terms", 0.0, 1.0, False),
    "separation_exponent": ("separation drag: exponent", "", "Game terms", 2.0, 16.0, False),
    "lift_softmin_exponent": (
        "lift softness: exponent (low = soft)",
        "",
        "Game terms",
        2.0,
        64.0,
        False,
    ),
    "thrust_softmin_exponent": (
        "thrust softness: exponent (low = soft)",
        "",
        "Game terms",
        2.0,
        64.0,
        False,
    ),
    "nose_damping": ("nose damping", "", "Stick and nose", 0.2, 1.5, False),
    "corner_nose_frequency": (
        "nose frequency at corner speed",
        "rad/s",
        "Stick and nose",
        0.3,
        2.0,
        True,
    ),
    "fine_stick": ("fine stick", "of full", "Stick and nose", 0.02, 1.0, False),
    "sustained_stick": ("sustained stick", "of full", "Stick and nose", 0.05, 1.0, False),
    "flap_lift": ("extra lift", "of clean", "Flaps", 0.0, 0.5, False),
    "flap_full_speed_ratio": ("full below", "x stall speed", "Flaps", 1.0, 2.0, False),
    "flap_clear_speed_ratio": ("clear above", "x stall speed", "Flaps", 1.05, 3.0, False),
    "flap_drag_scale": ("drag of the extra lift", "", "Flaps", 0.0, 3.0, True),
    "hit_damage": ("hit damage", "", "Guns", 0.3, 3.0, True),
    "explosive_share": ("explosive share", "", "Guns", 0.0, 1.0, False),
    "fire_rate": ("fire rate", "1/s", "Guns", 0.3, 2.0, True),
    "muzzle_speed": ("muzzle speed", "m/s", "Guns", 0.5, 2.0, True),
    "bullet_spread": ("bullet spread", "m/s", "Guns", 0.0, 3.0, True),
    "bullet_drag": ("bullet drag", "1/m", "Guns", 0.0, 4.0, True),
    "gun_range": ("gun range", "m", "Guns", 0.3, 2.5, True),
    "hit_radius": ("hit radius", "m", "Guns", 0.3, 3.0, True),
    "core_radius": ("core radius", "m", "Guns", 0.3, 3.0, True),
    "max_shield": ("shield", "", "Shield, head-on and vision", 0.3, 3.0, True),
    "shield_recharge_rate": (
        "shield recharge rate",
        "1/s",
        "Shield, head-on and vision",
        0.2,
        4.0,
        True,
    ),
    "shield_recharge_delay": (
        "shield recharge wait",
        "s",
        "Shield, head-on and vision",
        0.2,
        4.0,
        True,
    ),
    "head_on_fraction": ("head-on damage share", "", "Shield, head-on and vision", 0.0, 1.0, False),
    "head_on_angle": ("head-on angle", "rad", "Shield, head-on and vision", 0.1, 1.57, False),
    "head_on_sharpness": ("head-on sharpness", "", "Shield, head-on and vision", 0.3, 3.0, True),
    "vision_range": ("vision range", "m", "Shield, head-on and vision", 0.3, 3.0, True),
}

PRESET_ROWS = {name: tuning.row_of(name) for name in card.TYPE_NAMES}  # as imported
_HANDLES: dict[str, dict] = {}
_ROWS: dict[tuple, dict] = {}
_CHECKS: dict[
    tuple, tuple[dict, dict]
] = {}  # (factor, row) -> (lowest speeds, flown points), game units
_PRESET_STALL: dict[str, float] = {}  # preset -> its stall speed at the game's time factor


def _handles(preset: str) -> dict[str, float]:
    if preset not in _HANDLES:
        _HANDLES[preset] = tuning.handles_of(PRESET_ROWS[preset])
    return _HANDLES[preset]


def input_names(preset: str) -> list[str]:
    """Every input of the tuning route for the preset: its handles, then the card fields
    handles_row takes from the base row (not set by the handles' closed forms, not solved)."""
    row, handles = PRESET_ROWS[preset], _handles(preset)
    computed = (
        set(tuning.direct_fields(handles, handles["cruise_speed"]))
        | set(tuning.SOLVED)
        | {"exchange_acceleration"}
    )
    return [n for n in handles if n not in FIXED_HANDLES] + [
        n for n in row if n not in computed and n not in handles
    ]


def _defaults(preset: str) -> tuple[dict[str, float], frozenset]:
    """The preset's input values and the switches it has off."""
    row, handles = PRESET_ROWS[preset], _handles(preset)
    values = {n: handles[n] if n in handles else row[n] for n in input_names(preset)}
    off = {n for n, value in tuning.POINT_MASS_TERMS.items() if n in values and values[n] == value}
    return values, frozenset(off | {MATCH})


def inputs(preset: str) -> list[dict]:
    """The page's inputs, in group order: name, label, unit, group, value, low, high, and
    `switch` (on: bool) where the input can be turned off."""
    values, off = _defaults(preset)
    out = []
    for name in input_names(preset):
        value = values[name]
        label, unit, group, low, high, relative = INFO.get(
            name, (name.replace("_", " "), "", None, 0.5, 2.0, True)
        )
        if group is None:
            group = (
                "Flaps"
                if name.startswith("flap_")
                else "Game terms"
                if name in card.FLIGHT_TERMS
                else "Other"
            )
        if not math.isfinite(
            value
        ):  # a term off in the preset (critical speed inf): the slider's top
            value = high if not relative else 1.0
        if relative:
            low, high = (low * value, high * value) if value else (0.0, 1.0)
        entry = dict(
            name=name,
            label=label,
            unit=unit,
            group=group,
            value=value,
            low=min(low, value),
            high=max(high, value),
        )
        if name in tuning.POINT_MASS_TERMS or name == MATCH:
            entry["switch"] = dict(on=name not in off, label="match" if name == MATCH else "on")
        out.append(entry)
    return sorted(out, key=lambda e: GROUPS.index(e["group"]))


def presets() -> dict:
    """Per card type: its inputs."""
    return _plain({name: dict(inputs=inputs(name)) for name in card.TYPE_NAMES})


def build_row(preset: str, values: dict[str, float] | None = None, off=None) -> dict[str, float]:
    """The preset's row with the inputs `values` changed and the switches `off` off (None: the
    preset's), through tuning.handles_row; the preset's own row when nothing changes."""
    defaults, default_off = _defaults(preset)
    changed = {
        n: float(v)
        for n, v in (values or {}).items()
        if n in defaults and abs(float(v) - defaults[n]) > 1e-9 * abs(defaults[n])
    }
    off = default_off if off is None else frozenset(off)
    base = PRESET_ROWS[preset]
    if not changed and off == default_off:
        return base
    key = (preset, tuple(sorted(changed.items())), tuple(sorted(off)))
    if key not in _ROWS:
        own = _handles(preset)
        merged = {**defaults, **changed}
        handles = {**own, **{n: merged[n] for n in own if n in merged}}
        handles["zoom_acceleration"] = -handles["dive_acceleration"]
        if MATCH in off:
            handles[MATCH] = None
        new_base = {**base, **{n: merged[n] for n in base if n in merged and n not in own}}
        new_base.update({n: v for n, v in tuning.POINT_MASS_TERMS.items() if n in off})
        if len(_ROWS) > 64:
            _ROWS.clear()
        _ROWS[key] = tuning.handles_row(handles, new_base, speed_step=SOLVE_SPEED_STEP, fast=True)
    return _ROWS[key]


def _chart():
    """The committed charts' module (docs/rewrite/sections/turns/doghouse.py), loaded once."""
    global _CHART
    if _CHART is None:
        import importlib.util
        import sys
        from pathlib import Path

        folder = Path(__file__).resolve().parents[2] / "docs" / "rewrite" / "sections" / "turns"
        sys.path.insert(0, str(folder))
        spec = importlib.util.spec_from_file_location("doghouse_chart", folder / "doghouse.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CHART = module
    return _CHART


_CHART = None


def _stall_speed() -> float:
    """The card's type 0's 1 g stall speed at the card's time scale."""
    return math.sqrt(card.GRAVITY / card.CARD["max_lift_coefficient"][0])


def _key(row: dict, factor: float) -> tuple:
    return (factor, tuple(sorted(row.items())))


def _preset_checks(preset: str) -> tuple[dict, dict]:
    """The preset's flown checks at the game's time factor, cached (call outside tuning.flying:
    the card is process-global and flying does not nest across time scales)."""
    row = PRESET_ROWS[preset]
    key = _key(row, tuning.TIME_FACTOR)
    if key not in _CHECKS:
        with tuning.flying(row, tuning.TIME_FACTOR):
            lowest = flown.achievable_speeds()
            _CHECKS[key] = (lowest, _chart().flown_points(lowest))
            _PRESET_STALL[preset] = _stall_speed()
    return _CHECKS[key]


def _lowest(preset: str, row: dict, factor: float) -> tuple[dict[str, float], str, dict | None]:
    """Inside flying(row, factor), after _preset_checks(preset): the lowest flown speeds per line
    and their source, and the flown points if checked. Flown for this design if its checks ran
    ("flown"), else the preset's flown speeds scaled by the ratio of the stall speeds
    ("estimate"): the slider path never flies."""
    checked = _CHECKS.get(_key(row, factor))
    if checked is not None:
        return checked[0], "flown", checked[1]
    preset_lowest = _CHECKS[_key(PRESET_ROWS[preset], tuning.TIME_FACTOR)][0]
    ratio = _stall_speed() / _PRESET_STALL[preset]
    return {n: v * ratio for n, v in preset_lowest.items()}, "estimate", None


def warm(preset: str = "p51d") -> None:
    """The preset's flown checks at the game's time factor (cached): run once at start."""
    with LOCK:
        _preset_checks(preset)


def fly_checks(preset: str, values, off, factor: float) -> dict:
    """Fly the design's checks (lean.tool.flown: the lowest speed of each line and the key
    points), cache them for its charts, and return them with the quasi-steady points (game
    units at `factor`)."""
    with LOCK:
        _preset_checks(preset)
        row = build_row(preset, values, off)
        chart = _chart()
        with tuning.flying(row, factor):
            key = _key(row, factor)
            if key not in _CHECKS:
                lowest = flown.achievable_speeds()
                _CHECKS[key] = (lowest, chart.flown_points(lowest))
            lowest, points = _CHECKS[key]
            quasi = chart.key_points(lowest)
    return _plain(dict(factor=factor, lowest=lowest, flown=points, quasi=quasi))


def _plain(value):
    """numpy to JSON-ready values, non-finite numbers as null."""
    if isinstance(value, np.ndarray):
        return [_plain(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    return value


_COMPUTED: dict[
    tuple, object
] = {}  # the key numbers and sustained curves per (what, design, factor, source)


def _cached(what: str, preset: str, row: dict, factor: float, compute):
    """compute(lowest, source, points) inside flying(row, factor), cached per design, time scale
    and source of the lowest speeds (a flown check replaces an estimate)."""
    key = (what, _key(row, factor), _key(row, factor) in _CHECKS)
    if key not in _COMPUTED:
        if len(_COMPUTED) > 256:
            _COMPUTED.clear()
        with tuning.flying(row, factor):
            _COMPUTED[key] = compute(*_lowest(preset, row, factor))
    return _COMPUTED[key]


def _numbers(lowest, source, points) -> dict:
    """The key numbers of the card's type 0 (inside flying)."""
    key = _chart().key_points(lowest)
    performance = tuning.performance({name: values[0] for name, values in card.CARD.items()})
    key.update(
        {
            name: performance[name]
            for name in ("top_speed", "best_climb", "best_climb_speed", "stall_speed")
        }
    )
    return dict(key=key, lowest=lowest, lowest_source=source, flown=points)


def envelope(preset: str, values, off, factor: float) -> dict:
    """JSON-ready: the key numbers for the tuned design and for the preset as it stands (the
    reference), in game units at `factor` times real time."""
    with LOCK:
        _preset_checks(preset)
        row = build_row(preset, values, off)
        design = _cached("numbers", preset, row, factor, _numbers)
        reference = _cached("numbers", preset, PRESET_ROWS[preset], factor, _numbers)
    return _plain(dict(factor=factor, design=design, reference=reference))


def _png(figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, facecolor=_chart().SURFACE)
    _chart().plt.close(figure)
    return buffer.getvalue()


def doghouse_png(preset: str, values, off, factor: float, real: bool) -> bytes:
    """The doghouse of the tuned design as a PNG, drawn by the committed charts' code. real:
    divide speeds, rates and Ps by the time factor (1x real time; at 1x the units are real)."""
    chart = _chart()
    with LOCK:
        _preset_checks(preset)
        row = build_row(preset, values, off)
        title = f"{preset}, " + ("the preset" if row == PRESET_ROWS[preset] else "this design")
        with tuning.flying(row, factor):
            lowest, source, points = _lowest(preset, row, factor)
            figure = chart.doghouse_figure(
                title, lowest, real and factor != 1.0, factor, points, source
            )
            return _png(figure)


def sustained_png(preset: str, values, off, factor: float, real: bool) -> bytes:
    """The sustained turn (rate and radius against speed) of the tuned design and of the preset
    (dotted), as a PNG in the doghouse's style."""
    chart = _chart()
    with LOCK:
        _preset_checks(preset)
        row = build_row(preset, values, off)
        curves = [
            (
                label,
                _cached(
                    "sustained",
                    preset,
                    this,
                    factor,
                    lambda lowest, *_: chart.sustained_curve(lowest, factor),
                ),
                reference,
            )
            for label, this, reference in (
                ("this design", row, False),
                (f"{preset} (preset)", PRESET_ROWS[preset], True),
            )
        ]
        units = "Real units" if real and factor != 1.0 else "Game units"
        note = (
            f"Coast, spawn store. {units} at {factor:g}x time."
            " Each line from the lowest speed the game flies full stick."
        )
        return _png(chart.sustained_figure(curves, real and factor != 1.0, factor, note))
