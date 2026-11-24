"""What the web tool computes, through the game's own entry points (lean.game.tuning): a ship's
row from its handles, and the turn envelope with the key numbers. No physics lives here."""

import threading

import numpy as np

from lean.game import card, tuning

# The handles the Ships tab moves (game units at the game's time factor; a preset's other
# handles are kept; the low-speed acceleration is left free, the static thrust staying the
# preset's): name, label, unit, lowest and highest as multiples of the preset's value.
SLIDERS = (
    ("cruise_speed", "cruise (top) speed", "m/s", 0.6, 1.4),
    ("stall_speed", "stall speed (1 g)", "m/s", 0.6, 1.5),
    ("g_limit", "g limit", "g", 0.4, 1.6),
    ("sustained_turn_rate", "settled full-stick turn rate", "rad/s", 0.5, 1.6),
    ("zero_lift_drag", "zero-lift drag", "1/m", 0.4, 2.5),
    ("stall_angle", "stall angle", "rad", 0.5, 1.6),
    ("dive_acceleration", "dive/zoom trade", "m/s2", 0.3, 1.6),
)
SPEEDS = np.linspace(20.0, 470.0, 226)
STICKS = np.linspace(0.0, 1.0, 101)
SOLVE_SPEED_STEP = 0.5  # m/s: the handles solve on a coarser speed grid than the default, for speed
LOCK = threading.RLock()  # the card is process-global (tuning.flying): one computation at a time
_REFERENCE: dict[tuple[str, float], dict] = {}


def presets() -> dict:
    """Per card type: its handles and the sliders' ranges."""
    out = {}
    for name in card.TYPE_NAMES:
        handles = tuning.handles_of(tuning.row_of(name))
        out[name] = dict(
            handles=handles,
            sliders=[
                dict(name=n, label=label, unit=unit, value=handles[n], low=low * handles[n],
                     high=high * handles[n])
                for n, label, unit, low, high in SLIDERS
            ],
        )
    return out


def build_row(preset: str, handles: dict[str, float] | None) -> dict[str, float]:
    """The preset's row with `handles` (a subset of its handles) changed, through the tuning's
    handles route; the preset's own row when nothing changes."""
    base = tuning.row_of(preset)
    if not handles:
        return base
    own = tuning.handles_of(base)
    if all(abs(handles[n] - own[n]) <= 1e-9 * abs(own[n]) for n in handles):
        return base
    new = {**own, **handles, "low_speed_acceleration": None}
    new["zoom_acceleration"] = -new["dive_acceleration"]
    return tuning.handles_row(new, base, speed_step=SOLVE_SPEED_STEP, fast=True)


def _numbers(factor: float) -> dict:
    """The charts' arrays and the key numbers of the card's type 0 (inside tuning.flying)."""
    own = tuning.row_of(card.TYPE_NAMES[0])
    grid = tuning.envelope(own, SPEEDS, STICKS, minimum_speed=50.0 * factor)
    top = float(np.nanmax(grid["rate"][-1]))
    rate_axis = np.linspace(0.0, 1.15 * top, 120)
    stick = lambda s: grid["rate"][int(round(s * (len(STICKS) - 1)))]  # noqa: E731
    dive = tuning.envelope(own, SPEEDS, np.array([1.0]), power=1)["rate"][-1]
    climb = tuning.envelope(own, SPEEDS, np.array([1.0]), power=2)["rate"][-1]
    performance = tuning.performance(own)
    key = {
        name: grid[name]
        for name in (
            "corner_speed", "corner_rate", "best_sustained_speed", "best_sustained_rate",
            "tightest_sustained_speed", "tightest_sustained_radius",
        )
    }
    key.update({name: performance[name] for name in ("top_speed", "best_climb", "best_climb_speed", "stall_speed")})
    return dict(
        speeds=SPEEDS, rate_axis=rate_axis, ps=tuning.ps_over_rate(grid, rate_axis),
        full=grid["rate"][-1], sustained_stick=stick(own["sustained_stick"]),
        fine_stick=stick(own["fine_stick"]), dive=dive, climb=climb,
        held_rate=grid["sustained_rate"], held_radius=grid["sustained_radius"],
        key=key,
    )


def _plain(value):
    """numpy to JSON-ready lists, non-finite numbers as null."""
    if isinstance(value, np.ndarray):
        return [_plain(x) for x in value] if value.ndim > 1 else [_plain(float(x)) for x in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def envelope(preset: str, handles: dict[str, float] | None, factor: float) -> dict:
    """JSON-ready: the charts' arrays and key numbers for the tuned design and for the preset
    as it stands (the reference), in game units at `factor` times real time."""
    with LOCK:
        if (preset, factor) not in _REFERENCE:
            with tuning.flying(tuning.row_of(preset), factor):
                _REFERENCE[preset, factor] = _numbers(factor)
        with tuning.flying(build_row(preset, handles), factor):
            design = _numbers(factor)
    return _plain(dict(factor=factor, design=design, reference=_REFERENCE[preset, factor]))


_DOGHOUSE = None


def _doghouse_module():
    """The PNG charts' own drawing code (docs/rewrite/sections/turns/doghouse.py), loaded once,
    so the tool draws exactly what the committed charts draw (one drawing, matplotlib)."""
    global _DOGHOUSE
    if _DOGHOUSE is None:
        import importlib.util
        import sys
        from pathlib import Path

        folder = Path(__file__).resolve().parents[2] / "docs" / "rewrite" / "sections" / "turns"
        sys.path.insert(0, str(folder))
        spec = importlib.util.spec_from_file_location("doghouse_chart", folder / "doghouse.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _DOGHOUSE = module
    return _DOGHOUSE


def doghouse_png(preset: str, handles: dict[str, float] | None, factor: float, real: bool) -> bytes:
    """The doghouse of the tuned design as a PNG, drawn by the same matplotlib code as the
    committed charts. real: halve speeds, rates and Ps for the 1x-real view (only at the game's
    2x; at 1x the game units are already real)."""
    import io

    chart = _doghouse_module()
    plt = chart.plt
    with LOCK:
        with tuning.flying(build_row(preset, handles), factor):
            figure, ax = plt.subplots(figsize=(7.5, 7.4), facecolor=chart.SURFACE)
            chart.draw_panel(figure, ax, 0, "tuned design", None, real and factor != 1.0)
            figure.text(0.01, 0.004, chart.foot(0, real and factor != 1.0), color=chart.INK_2, fontsize=7)
            figure.tight_layout(rect=(0, 0.25, 1, 1))
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=110, facecolor=chart.SURFACE)
            plt.close(figure)
    return buffer.getvalue()
