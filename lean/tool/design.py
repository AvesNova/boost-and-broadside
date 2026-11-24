"""What the web tool computes, through the game's own entry points (lean.game.tuning): a ship's
row from its design inputs (tuning.design_row), what it implies physically and its warnings, the
charts (drawn by the committed charts' matplotlib code, docs/rewrite/sections/turns/doghouse.py)
and the key numbers, and the flown checks (lean.tool.flown). No physics lives here.

The sliders are generated from the chosen profile's design-input table (tuning.PROFILES:
`inputs`: name, label, unit, group, how it sets the card, the slider's range), each starting at
the preset's reference value under that profile (tuning.reference) with its provenance tag; no
code here knows any profile's inputs. Only moved inputs are sent; the server solves the
row in the fast path (Broyden), warm from the preset's last solve (its row and Jacobian).
"""

import io
import math
import os
import threading
import time

import numpy as np

from lean.game import card, tuning

from . import flown, units

# The solve fails fast (tuning.design_row): after PATIENCE iterations without the error falling
# by 1%, or after TIME_LIMIT seconds (BNB_SOLVE_TIME_LIMIT; the warm try and the cold retry
# share it).
PATIENCE = 6
TIME_LIMIT = float(os.environ.get("BNB_SOLVE_TIME_LIMIT", "4"))
UNMET = 1e-3  # an input whose solved value is off its target by more than this (relative) is unmet

SHUTDOWN = threading.Event()  # set when the server stops: every running solve ends at once

LOCK = threading.RLock()  # the card is process-global (tuning.flying): one computation at a time

PRESET_ROWS = {name: tuning.row_of(name) for name in card.TYPE_NAMES}  # as imported
_SOLVED: dict[tuple, tuning.Solved] = {}
_CHECKS: dict[
    tuple, tuple[dict, dict]
] = {}  # (factor, row) -> (lowest speeds, flown points), game units
_PRESET_STALL: dict[str, float] = {}  # preset -> its stall speed at the game's time factor


def inputs(preset: str, profile: str | None = None) -> list[dict]:
    """The page's sliders, in the profile's table order (default the preset's own profile):
    name, label, unit, kind, group, method (how it sets the card), source (provenance), value
    and reference (the preset's), low, high."""
    reference = tuning.reference(preset, profile)
    out = []
    for spec in reference.inputs:
        value = reference.design[spec.name]
        scale = value if spec.relative else 1.0
        low, high = spec.low * scale, spec.high * scale
        out.append(
            dict(
                name=spec.name,
                label=spec.label,
                unit=spec.unit,
                kind=spec.kind,
                group=spec.group,
                method=spec.method,
                sets=spec.sets,
                source=spec.source,
                value=value,
                reference=value,
                low=min(low, value),
                high=max(high, value),
            )
        )
    return out


def presets() -> dict:
    """Per card type: its own profile and, per profile (tuning.PROFILES), its inputs."""
    return _plain(
        {
            name: dict(
                profile=tuning.REAL_AIRCRAFT[name].profile,
                profiles={profile: inputs(name, profile) for profile in tuning.PROFILES},
            )
            for name in card.TYPE_NAMES
        }
    )


def profiles() -> dict:
    """The profiles' names and labels, in tuning.PROFILES order."""
    return {name: profile.label for name, profile in tuning.PROFILES.items()}


def _profile(preset: str, profile: str | None) -> str:
    """The profile's name (default the preset's own); ValueError for an unknown one."""
    profile = profile or tuning.REAL_AIRCRAFT[preset].profile
    if profile not in tuning.PROFILES:
        raise ValueError(f"not a design-input profile: {profile}")
    return profile


def solve(
    preset: str,
    values: dict | None = None,
    fixed: dict | None = None,
    profile: str | None = None,
    progress=None,
    cancel=None,
) -> tuning.Solved:
    """The preset's design under the profile (default its own) with the inputs `values` moved,
    solved (tuning.design_row, fast path,
    warm: `_solved`); the preset's own row, unsolved, when nothing moves; `fixed` (a saved
    card row) as it is. Unknown names raise ValueError; a solve that does not converge raises
    RuntimeError (tuning.SolveFailed when it stopped early: stalled, out of time, impossible).
    `progress` (tuning.SolveStep each iteration) and `cancel` (a threading.Event: set, the solve
    raises tuning.SolveCancelled within one evaluation and changes nothing) are for the page."""
    if fixed is not None:
        if set(fixed) != set(card.CARD_FIELDS):
            raise ValueError("a fixed ship's row must have the card's fields")
        return tuning.Solved({name: float(fixed[name]) for name in card.CARD_FIELDS}, 0.0, 0, None)
    profile = _profile(preset, profile)
    reference = tuning.reference(preset, profile).design
    unknown = set(values or {}) - set(reference)
    if unknown:
        raise ValueError(f"not design inputs: {sorted(unknown)}")
    changed = {
        n: float(v)
        for n, v in (values or {}).items()
        if abs(float(v) - reference[n]) > 1e-9 * abs(reference[n])
    }
    key = (preset, profile, tuple(sorted(changed.items())))
    if key not in _SOLVED:
        if len(_SOLVED) > 64:
            _SOLVED.clear()
        _SOLVED[key] = (
            _solved(preset, profile, changed, progress, cancel)
            if changed
            else tuning.Solved(
                tuning.reference(preset, profile).start_row or PRESET_ROWS[preset], 0.0, 0, None
            )
        )
    return _SOLVED[key]


_WARM: dict[tuple, tuning.Solved] = {}  # (preset, profile) -> its last solve (row, Jacobian)


def _solved(preset: str, profile: str, changed: dict, progress=None, cancel=None) -> tuning.Solved:
    """The fast solve, warm from the preset's last solve (its row and Jacobian: a slider drag
    moves a little each time), else, or when that does not converge, from the preset's row."""
    reference = tuning.reference(preset, profile)
    last = _WARM.get((preset, profile))
    solved = None
    watch = dict(
        progress=progress,
        cancel=_Cancel(cancel),
        patience=PATIENCE,
        deadline=time.monotonic() + TIME_LIMIT,
    )
    if last is not None:
        try:
            solved = tuning.design_row(
                reference, changed, start=last.row, fast=True, jacobian=last.jacobian, **watch
            )
        except tuning.SolveFailed as failed:
            if failed.reason != "stalled":  # out of time, or no design at all: no retry
                raise
        except RuntimeError, ValueError, IndexError, ArithmeticError:  # app.UNSOLVABLE
            solved = None
    if solved is None:
        solved = tuning.design_row(
            reference, changed, start=PRESET_ROWS[preset], fast=True, **watch
        )
    if solved.jacobian is not None:
        _WARM[(preset, profile)] = solved
    return solved


class _Cancel:
    """The solve's cancel token: the request's own, or the server stopping."""

    def __init__(self, token):
        self.token = token

    def is_set(self) -> bool:
        return SHUTDOWN.is_set() or (self.token is not None and self.token.is_set())


def failure_report(
    error: tuning.SolveFailed,
    preset: str,
    profile: str | None,
    system: str,
    factor: float,
    moved: list[str] | None = None,
) -> dict:
    """What the page shows for a solve that stopped early: the reason, a sentence, and the
    inputs not met (relative error above UNMET, largest first) as {name, label, unit, target,
    achieved, error}: the nearest values found, in the units `system` at time scale `factor`
    (error a fraction). `nearest` marks them as a compromise that does not meet every input."""
    table = {spec.name: spec for spec in tuning.reference(preset, _profile(preset, profile)).inputs}

    def shown(spec, value):
        return units.to_display(spec.kind, value, system, factor) if spec.kind else value

    unmet = []
    for name, target, r in zip(error.names, error.targets, error.residuals):
        spec = table[name]
        if abs(r) > UNMET:
            unit = units.label(spec.kind, system) if spec.kind else spec.unit
            unmet.append(
                dict(
                    name=name,
                    label=spec.label,
                    unit=unit,
                    target=shown(spec, target),
                    achieved=shown(spec, target * math.exp(r)),
                    error=math.expm1(r),
                )
            )
    unmet.sort(key=lambda u: -abs(u["error"]))
    if error.reason == "impossible":
        text = error.message
        if moved and not any(name in text for name in moved):
            text += f"; inputs moved: {', '.join(moved)}"
    else:
        worst = max((abs(u["error"]) for u in unmet), default=0.0)
        names = (
            ", ".join(u["label"] for u in unmet if abs(u["error"]) >= 0.25 * worst)
            or "the inputs together"
        )
        why = (
            f"no solution in {TIME_LIMIT:g} s"
            if error.reason == "time"
            else "the error stopped falling"
        )
        text = f"Cannot meet {names}: {why}, nearest found is {worst:.1%} off"
    return dict(
        reason=error.reason, message=text, unmet=unmet, nearest=error.reason != "impossible"
    )


def build_row(
    preset: str, values: dict | None = None, fixed: dict | None = None, profile: str | None = None
) -> dict:
    return solve(preset, values, fixed, profile).row


def saved_record(name: str, state: dict, saved_at: str) -> dict:
    """What a saved preset stores: its state (preset, profile, the inputs moved, time scale,
    units), the profile and its design-input table's version, and the solved card row, so it
    flies as saved even after the table changes. A fixed ship saved again keeps its row and
    stays fixed."""
    fixed = state.get("fixed")
    preset = state.get("preset", "p51d")
    profile = _profile(preset, state.get("profile"))
    version = None if fixed else tuning.inputs_version(tuning.PROFILES[profile].inputs)
    return dict(
        name=name,
        saved_at=saved_at,
        base_preset=preset,
        state=state,
        profile=profile,
        inputs_version=version,
        row=build_row(preset, state.get("values"), fixed, profile),
    )


def loaded_record(record: dict) -> dict | None:
    """A saved preset as the page loads it: editable sliders when its profile's table version is
    today's (a record without a profile was saved under the documents table); otherwise a fixed
    ship from its stored row, with the inputs of its profile (or, if gone, the default's)
    measured from that row (tuning.design_of). None for a record without a row (an old format:
    not listed)."""
    if not isinstance(record.get("row"), dict) or set(record["row"]) != set(card.CARD_FIELDS):
        return None
    profile = record.get("profile") or tuning.DEFAULT_PROFILE
    known = tuning.PROFILES.get(profile)
    editable = known is not None and record.get("inputs_version") == tuning.inputs_version(
        known.inputs
    )
    out = dict(record, profile=profile if known else tuning.DEFAULT_PROFILE, editable=editable)
    if not editable:
        table = (known or tuning.PROFILES[tuning.DEFAULT_PROFILE]).inputs
        with LOCK:
            out["measured"] = _plain(tuning.design_of(record["row"], table))
    return out


def ship(
    preset: str,
    values: dict[str, float] | None = None,
    fixed: dict | None = None,
    profile: str | None = None,
) -> dict:
    """The design's implied physical quantities (tuning.implied, real units), its warnings
    (tuning.warnings), its solved card row with the solve's residual, iterations and condition
    number, and what it flies (tuning.design_of, game units)."""
    with LOCK:
        solved = solve(preset, values, fixed, profile)
        reference = tuning.reference(preset, _profile(preset, profile))
        numbers = tuning.performance(solved.row)
        return _plain(
            dict(
                implied=tuning.implied(solved.row, reference.airframe),
                warnings=tuning.warnings(solved.row, reference, numbers),
                row=solved.row,
                solve=dict(
                    residual=solved.residual,
                    iterations=solved.iterations,
                    condition=solved.condition,
                ),
                flies=tuning.design_of(solved.row, reference.inputs),
            )
        )


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


def fly_checks(
    preset: str, values, factor: float, fixed: dict | None = None, profile: str | None = None
) -> dict:
    """Fly the design's checks (lean.tool.flown: the lowest speed of each line and the key
    points), cache them for its charts, and return them with the quasi-steady points (game
    units at `factor`)."""
    with LOCK:
        _preset_checks(preset)
        row = build_row(preset, values, fixed, profile)
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


def envelope(
    preset: str, values, factor: float, fixed: dict | None = None, profile: str | None = None
) -> dict:
    """JSON-ready: the key numbers for the tuned design and for the preset as it stands (the
    reference), in game units at `factor` times real time."""
    with LOCK:
        _preset_checks(preset)
        row = build_row(preset, values, fixed, profile)
        design = _cached("numbers", preset, row, factor, _numbers)
        reference = _cached("numbers", preset, PRESET_ROWS[preset], factor, _numbers)
    return _plain(dict(factor=factor, design=design, reference=reference))


def _png(figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, facecolor=_chart().SURFACE)
    _chart().plt.close(figure)
    return buffer.getvalue()


def doghouse_png(
    preset, values, factor: float, system: str = "game", fixed=None, profile=None
) -> bytes:
    """The doghouse of the tuned design as a PNG, drawn by the committed charts' code, in the
    units system (lean.tool.units: game, real_metric, real_imperial)."""
    chart = _chart()
    with LOCK:
        _preset_checks(preset)
        row = build_row(preset, values, fixed, profile)
        title = f"{preset}, " + ("the preset" if row == PRESET_ROWS[preset] else "this design")
        with tuning.flying(row, factor):
            lowest, source, points = _lowest(preset, row, factor)
            figure = chart.doghouse_figure(
                title, lowest, False, factor, points, source, system=units.system_of(system)
            )
            return _png(figure)


def sustained_png(
    preset, values, factor: float, system: str = "game", fixed=None, profile=None
) -> bytes:
    """The sustained turn (rate and radius against speed) of the tuned design and of the preset
    (dotted), as a PNG in the doghouse's style."""
    chart = _chart()
    with LOCK:
        _preset_checks(preset)
        row = build_row(preset, values, fixed, profile)
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
        system = units.system_of(system)
        note = (
            f"Coast, spawn store. {units.describe(system, factor)}"
            " Each line from the lowest speed the game flies full stick."
        )
        return _png(chart.sustained_figure(curves, False, factor, note, system=system))
