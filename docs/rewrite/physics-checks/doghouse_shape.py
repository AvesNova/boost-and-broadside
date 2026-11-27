"""Why the doghouse's Ps contours slide into the max-lift line, and where the gap comes from
(docs/rewrite/doghouse-shape-investigation.md). Analysis only: every number from the game's own
flight functions (core.steady_rates through per-type constants in float64) on card rows solved
by lean.game.tuning.design_row; no default changes.

    UV_PROJECT_ENVIRONMENT=.../v2/.venv OMP_NUM_THREADS=4 PYTHONPATH=. \
        uv run --no-sync python docs/rewrite/physics-checks/doghouse_shape.py [part ...]

Parts: polar, steepness, sweep, thrust, push, prop, figures (default all). Results go to
docs/rewrite/physics-checks/doghouse_shape.json, PNGs to docs/rewrite/doghouse-shape-*.png.
Real units throughout the outputs (the game at 2x shown at 1x: speeds, rates and Ps halved).
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lean.game import card, core, tuning

torch.set_num_threads(4)
HERE = Path(__file__).parent
OUT_JSON = HERE / "doghouse_shape.json"
OUT_PNG = HERE.parent
F = tuning.TIME_FACTOR
G = card.GRAVITY  # game gravity
DEG = 180.0 / math.pi
# Chart normalisation for angles: each axis as a fraction of a doghouse's range, plotted square
# (real units: speed 0-240 m/s, turn rate 0-40 deg/s; the committed chart is about square).
SPEED_RANGE, RATE_RANGE = 240.0, 40.0
PS_LEVELS = (10.0, 0.0, -10.0, -20.0, -40.0, -60.0)  # real m/s


# --------------------------------------------------------------------------- flight helpers


def constants(row):
    with torch.no_grad(), tuning._with_row(row) as index:
        c = core.per_type_constants(core.GameConfig(), torch.tensor([index]))
    return SimpleNamespace(
        **{
            k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in vars(c).items()
        }
    )


def steady(c, speed, stick):
    """Real units: turn rate (deg/s), Ps (m/s), at game speed(s) and stick(s), coasting, mid store."""
    speed = torch.as_tensor(speed, dtype=torch.float64)
    stick = torch.as_tensor(stick, dtype=torch.float64)
    if stick.dim() == 0:
        stick = torch.full_like(speed, float(stick))
    with torch.no_grad():
        rate, speed_rate, store_rate = core.steady_rates(c, speed, stick, 0, 0.5 * c.max_energy)
    ps = (speed * speed_rate + store_rate) / G
    return (rate * DEG / F).numpy(), (ps / F).numpy()


def corner_game(row):
    return math.sqrt(row["max_acceleration"] / row["max_lift_coefficient"])


def stall_game(row):
    return math.sqrt(G / row["max_lift_coefficient"])


def contour_angles(row, levels=PS_LEVELS, depths=(0.002, 0.01, 0.05)):
    """Per Ps level (real m/s) that meets the max-lift line below corner: the speed it meets it
    (real m/s) and the angle between the contour and the line (deg, in the square-normalised
    chart) measured with the contour's chord to `depth` (fraction of the rate axis) inside the
    line. Also the same for the plain point-mass row? no: the caller passes rows."""
    c = constants(row)
    v = np.linspace(stall_game(row) * 1.02, corner_game(row) * 0.97, 1200)
    rate_b, ps_b = steady(c, v, 1.0)
    x = v / F / SPEED_RANGE
    y = rate_b / RATE_RANGE
    out = {}
    for level in levels:
        # Ps along the line falls with speed below corner (checked: monotone in every row here)
        if not (ps_b.max() >= level >= ps_b.min()):
            out[str(level)] = None
            continue
        k = int(np.nonzero(ps_b < level)[0][0])
        t = (level - ps_b[k - 1]) / (ps_b[k] - ps_b[k - 1])
        v0 = v[k - 1] + t * (v[k] - v[k - 1])
        dpsdx = (ps_b[k] - ps_b[k - 1]) / (x[k] - x[k - 1])
        dydx = (y[k] - y[k - 1]) / (x[k] - x[k - 1])
        r0, p0 = steady(c, np.array([v0]), 1.0)
        entry = {"speed": v0 / F}
        for depth in depths:
            # the stick whose rate is `depth` of the axis inside the line, by bisection
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                r, _ = steady(c, np.array([v0]), mid)
                if (r0[0] - r[0]) / RATE_RANGE > depth:
                    lo = mid
                else:
                    hi = mid
            r, p = steady(c, np.array([v0]), 0.5 * (lo + hi))
            py = (p0[0] - p[0]) / ((r0[0] - r[0]) / RATE_RANGE)  # dPs/dy, inward chord
            contour_slope = dydx - dpsdx / py
            angle = abs(math.degrees(math.atan(contour_slope) - math.atan(dydx)))
            entry[f"angle_{depth}"] = angle
            entry[f"dps_dy_{depth}"] = py  # m/s of Ps per full rate axis
        entry["dps_dx_along"] = dpsdx
        out[str(level)] = entry
    return out


# --------------------------------------------------------------------------- tables and solves


def replaced(inputs, **specs):
    """A design-input table with the named inputs replaced (None: dropped)."""
    out = []
    names = {s.name for s in inputs}
    for spec in inputs:
        if spec.name in specs:
            if specs[spec.name] is not None:
                out.append(specs[spec.name])
        else:
            out.append(spec)
    extra = [s for n, s in specs.items() if n not in names and s is not None]
    return tuple(out) + tuple(extra)


SEPARATION_CLOSED = tuning.scaled("separation_drag", "separation drag", "Flight terms")
SOFTNESS_CLOSED = tuning.scaled("thrust_softmin_exponent", "thrust softness", "Flight terms")
GAP_SOLVES_SOFTNESS = tuning.flown(
    "gap", "gap", "Performance", "thrust_softmin_exponent", first_guess=2.0
)
CLIMB_SPEED_SOLVES_STATIC = tuning.flown(
    "best_climb_speed", "best-climb speed", "Performance", "static_thrust"
)
GAP_SOLVES_STATIC = tuning.flown("gap", "gap", "Performance", "static_thrust")
SUSTAINED_SPEED_SOLVES_STATIC = tuning.flown(
    "sustained_speed", "best sustained speed", "Performance", "static_thrust"
)
STATIC_CLOSED = tuning.DESIGN["static_thrust"]


def _induced_of_oswald(oswald):
    a = tuning.P51D
    return 1.0 / (math.pi * oswald * a.aspect_ratio) / a.per_metre / card.INDUCED_DRAG_FACTOR


OSWALD_CLOSED = tuning.DesignInput(
    "oswald",
    "Oswald e (low lift)",
    "Airframe",
    "induced_drag_scale",
    lambda row, numbers: tuning.implied(row, tuning.P51D)["oswald"],
    lambda value, design: _induced_of_oswald(value),
)
GAP_SOLVES_SEPARATION = tuning.DESIGN["gap"]

TABLES = {
    # the documents table: gap solves the separation drag (the default)
    "documents": tuning.DESIGN_INPUTS,
    # the separation drag closed, the gap a readout
    "separation_closed": replaced(
        tuning.DESIGN_INPUTS, gap=None, separation_drag=SEPARATION_CLOSED
    ),
    # separation closed, the thrust softness closed, best-climb speed and gap readouts
    "thrust_closed": replaced(
        tuning.DESIGN_INPUTS,
        gap=None,
        separation_drag=SEPARATION_CLOSED,
        best_climb_speed=None,
        thrust_softmin_exponent=SOFTNESS_CLOSED,
    ),
    # separation closed; the gap solves the thrust softness, best-climb speed solves the static
    # thrust (static thrust no longer closed)
    "gap_by_thrust": replaced(
        tuning.DESIGN_INPUTS,
        gap=GAP_SOLVES_SOFTNESS,
        separation_drag=SEPARATION_CLOSED,
        best_climb_speed=CLIMB_SPEED_SOLVES_STATIC,
        static_thrust=None,
    ),
    # separation closed; the gap solves the thrust softness, best-climb speed released
    "gap_by_softness": replaced(
        tuning.DESIGN_INPUTS,
        gap=GAP_SOLVES_SOFTNESS,
        separation_drag=SEPARATION_CLOSED,
        best_climb_speed=None,
    ),
    # e closed (sustained rate a readout); the gap solves the separation drag
    "oswald_closed": replaced(tuning.DESIGN_INPUTS, sustained_rate=None, oswald=OSWALD_CLOSED),
    # e closed, separation closed: sustained rate and gap readouts
    "oswald_separation_closed": replaced(
        tuning.DESIGN_INPUTS,
        sustained_rate=None,
        oswald=OSWALD_CLOSED,
        gap=None,
        separation_drag=SEPARATION_CLOSED,
    ),
}
for _table in TABLES.values():
    tuning.check_inputs(_table)

P51D = tuning.REAL_AIRCRAFT["p51d"]
_REFERENCES = {}


def reference(table):
    if table not in _REFERENCES:
        values = dict(P51D.values, separation_drag=0.0, oswald=0.8)
        aircraft = dataclasses.replace(P51D, values=values)
        ref = tuning.reference_of(aircraft, TABLES[table])
        _REFERENCES[table] = ref
    return _REFERENCES[table]


def solve(table, start=None, **design):
    ref = reference(table)
    if start is None and table == "documents":
        start = tuning.row_of("p51d")
    solved = tuning.design_row(ref, design, start=start)
    return solved.row


def readout(row):
    p = tuning.in_real_units(tuning.performance(row))
    imp = tuning.implied(row, P51D.airframe)
    mph = 1.0 / tuning.MPH
    return dict(
        top_speed_mph=p["top_speed"] * mph,
        best_climb_fpm=p["best_climb"] / tuning.FEET_PER_MINUTE,
        best_climb_speed_mph=p["best_climb_speed"] * mph,
        corner_speed=p["corner_speed"],
        corner_rate=p["corner_rate"] * DEG,
        sustained_speed=p["sustained_speed"],
        sustained_rate=p["sustained_rate"] * DEG,
        min_radius_speed=p["min_radius_speed"],
        min_radius=p["min_radius"],
        sustained_radius=p["sustained_radius"],
        gap=p["gap"],
        oswald=imp["oswald"],
        cd0=imp["zero_lift_drag_coefficient"],
        propeller_efficiency=imp["propeller_efficiency"],
        figure_of_merit=imp["figure_of_merit"],
        static_thrust_over_weight=imp["static_thrust_over_weight"],
        separation_drag=row["separation_drag"],
        separation_exponent=row["separation_exponent"],
        thrust_softness=row["thrust_softmin_exponent"],
        thrust_loss_at_sustained=tuning._thrust_loss(row, p["sustained_speed"] * F),
        sustained_stick=float(constants(row).sustained_stick[0]),
    )


def effective_oswald_at_max_lift(row):
    """e of the whole lift-dependent drag at C_Lmax: k (1 + s) read as 1 / (pi e AR)."""
    imp = tuning.implied(row, P51D.airframe)
    return imp["oswald"] / (1.0 + row["separation_drag"])


def safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except (RuntimeError, ValueError) as error:
        return None, str(error)[:200]


# --------------------------------------------------------------------------- parts


def part_polar(results):
    """The drag polar of the current row and the point mass, and Ps along/inside the line."""
    row = tuning.row_of("p51d")
    a = P51D.airframe
    k = 1.0 / (math.pi * tuning.implied(row, a)["oswald"] * a.aspect_ratio)
    cd0 = tuning.implied(row, a)["zero_lift_drag_coefficient"]
    cl_max = 1.6
    s, m = row["separation_drag"], row["separation_exponent"]
    cl = np.linspace(0.0, cl_max, 161)
    parabolic = cd0 + k * cl**2
    walled = cd0 + k * cl**2 * (1.0 + s * (cl / cl_max) ** m)
    slope_ratio = 1.0 + s * (m + 2.0) / 2.0
    share_last_tenth = 1.0 - 0.9**m  # of the separation drag, gained in the last 10% of C_L
    results["polar"] = dict(
        cd0=cd0,
        k=k,
        oswald=tuning.implied(row, a)["oswald"],
        separation=s,
        exponent=m,
        cd_at_clmax_parabolic=float(parabolic[-1]),
        cd_at_clmax=float(walled[-1]),
        slope_ratio_at_clmax=slope_ratio,
        separation_share_last_tenth=share_last_tenth,
        cd_at_0p9=float(cd0 + k * (0.9 * cl_max) ** 2 * (1 + s * 0.9**m)),
        effective_oswald_at_clmax=effective_oswald_at_max_lift(row),
    )
    # thrust tilt: 1 - cos(stall angle) of the thrust lost at the line
    stall = card.STALL_ANGLE * row["stall_angle_scale"]
    results["polar"]["thrust_tilt_loss_at_line"] = 1.0 - math.cos(stall)
    results["polar_curves"] = dict(
        cl=cl.tolist(), parabolic=parabolic.tolist(), walled=walled.tolist()
    )


def part_steepness(results):
    """Ps against the inward distance from the line at a few speeds, current vs point mass."""
    rows = {
        "current": tuning.row_of("p51d"),
        "point_mass": tuning.point_mass(tuning.row_of("p51d")),
    }
    out = {}
    for name, row in rows.items():
        c = constants(row)
        per = {}
        for v_real in (60.0, 70.0, 80.0, 90.0, 100.0, 115.0):
            v = np.array([v_real * F])
            sticks = 1.0 - np.concatenate([[0.0], np.geomspace(1e-4, 0.6, 120)])
            rates, ps = zip(*(steady(c, v, s_) for s_ in sticks))
            rates, ps = np.array(rates)[:, 0], np.array(ps)[:, 0]
            line_rate, line_ps = rates[0], ps[0]
            depth = (line_rate - rates) / RATE_RANGE
            per[str(v_real)] = dict(
                line_rate=float(line_rate),
                line_ps=float(line_ps),
                drop_1pct=float(np.interp(0.01, depth, line_ps - ps)),
                drop_5pct=float(np.interp(0.05, depth, line_ps - ps)),
                drop_20pct=float(np.interp(0.20, depth, line_ps - ps)),
                depth=depth.tolist(),
                ps=ps.tolist(),
            )
        out[name] = per
    results["steepness"] = out


def part_sweep(results):
    """Separation exponent and coefficient, the other documents targets held."""
    rows = []
    start = tuning.row_of("p51d")
    # (a) the gap held at 0.85 (documents table), the exponent lowered
    for m in (24.0, 16.0, 8.0, 4.0, 2.0):
        row, err = safe(solve, "documents", separation_exponent=m)
        entry = {"table": "documents (gap 0.85 solves s)", "exponent": m}
        if row is None:
            entry["error"] = err
        else:
            entry.update(readout(row))
            entry["effective_oswald_at_clmax"] = effective_oswald_at_max_lift(row)
            entry["angles"] = contour_angles(row)
        rows.append(entry)
        print(
            entry.get("exponent"),
            entry.get("separation_drag"),
            entry.get("oswald"),
            entry.get("error", ""),
            flush=True,
        )
    # (b) the coefficient set, the gap released
    for m in (2.0, 4.0, 8.0, 16.0):
        for s in (0.0, 0.25, 0.5, 1.0, 2.0):
            if s == 0.0 and m != 16.0:
                continue
            row, err = safe(
                solve, "separation_closed", start=start, separation_exponent=m, separation_drag=s
            )
            entry = {
                "table": "separation closed (gap a readout)",
                "exponent": m,
                "set_separation": s,
            }
            if row is None:
                entry["error"] = err
            else:
                entry.update(readout(row))
                entry["effective_oswald_at_clmax"] = effective_oswald_at_max_lift(row)
                entry["angles"] = contour_angles(row)
            rows.append(entry)
            print(m, s, entry.get("gap"), entry.get("oswald"), entry.get("error", ""), flush=True)
    results["sweep"] = rows
    results["current"] = dict(
        readout(start),
        angles=contour_angles(start),
        effective_oswald_at_clmax=effective_oswald_at_max_lift(start),
    )
    pm = tuning.point_mass(start)
    results["point_mass_of_current"] = dict(angles=contour_angles(pm))


def part_thrust(results):
    """No separation drag: the thrust curve's shape (softness, static thrust) against the gap and
    the best sustained speed, with top speed, best climb and best sustained rate held."""
    rows = []
    start = {**tuning.row_of("p51d"), "separation_drag": 0.0}
    for static in (0.45, 0.61, 0.8, 1.0):
        for q in (1.0, 1.5, 2.0, 2.83, 4.0):
            design = dict(separation_drag=0.0, thrust_softmin_exponent=q, static_thrust=static * G)
            row, err = safe(solve, "thrust_closed", start=start, **design)
            entry = {"static_over_weight": static, "softness": q}
            if row is None:
                entry["error"] = err
            else:
                entry.update(readout(row))
            rows.append(entry)
            print(
                static,
                q,
                entry.get("gap"),
                entry.get("sustained_speed"),
                entry.get("oswald"),
                entry.get("best_climb_speed_mph"),
                entry.get("error", ""),
                flush=True,
            )
    results["thrust"] = rows


def part_oswald(results):
    """e held (0.8 assumed, 0.9 NACA P-51B-like): the best sustained rate a readout; the gap
    solving the separation drag at each exponent, and the separation drag set (gap readout)."""
    rows = []
    start = tuning.row_of("p51d")
    for e in (0.8, 0.9):
        for m in (2.0, 4.0, 8.0, 16.0):
            row, err = safe(solve, "oswald_closed", start=start, oswald=e, separation_exponent=m)
            entry = {"table": "e held, gap 0.85 solves s", "oswald_set": e, "exponent": m}
            if row is None:
                entry["error"] = err
            else:
                entry.update(readout(row))
                entry["effective_oswald_at_clmax"] = effective_oswald_at_max_lift(row)
                entry["angles"] = contour_angles(row)
            rows.append(entry)
            print(
                e,
                m,
                entry.get("separation_drag"),
                entry.get("sustained_rate"),
                entry.get("sustained_speed"),
                entry.get("error", ""),
                flush=True,
            )
        for m, sep in ((2.0, 0.0), (2.0, 0.5), (2.0, 1.0), (4.0, 0.5), (4.0, 1.0)):
            row, err = safe(
                solve,
                "oswald_separation_closed",
                start=start,
                oswald=e,
                separation_exponent=m,
                separation_drag=sep,
            )
            entry = {
                "table": "e held, s set",
                "oswald_set": e,
                "exponent": m,
                "set_separation": sep,
            }
            if row is None:
                entry["error"] = err
            else:
                entry.update(readout(row))
                entry["effective_oswald_at_clmax"] = effective_oswald_at_max_lift(row)
                entry["angles"] = contour_angles(row)
            rows.append(entry)
            print(
                e,
                m,
                sep,
                entry.get("gap"),
                entry.get("sustained_rate"),
                entry.get("sustained_speed"),
                entry.get("error", ""),
                flush=True,
            )
    results["oswald"] = rows


def part_levers(results):
    """e held at 0.85, exponent 4, the gap 0.85 solving the separation drag: which documented
    target moves the best sustained speed (power through the climb, top speed, stall speed), and
    the thrust's shape (softness and static thrust; best-climb speed then a readout)."""
    rows = []
    start = tuning.row_of("p51d")
    base = dict(oswald=0.85, separation_exponent=4.0)
    ref = reference("oswald_closed")
    cases = [("base", {})]
    for f in (1.25, 1.5, 2.0):
        cases.append((f"best climb x{f}", dict(best_climb=ref.design["best_climb"] * f)))
    for f in (0.9, 1.1):
        cases.append((f"top speed x{f}", dict(top_speed=ref.design["top_speed"] * f)))
    for f in (0.9, 1.1):
        cases.append((f"stall speed x{f}", dict(stall_speed=ref.design["stall_speed"] * f)))
    for name, change in cases:
        row, err = safe(solve, "oswald_closed", start=start, **base, **change)
        entry = {"case": name}
        if row is None:
            entry["error"] = err
        else:
            entry.update(readout(row))
            entry["effective_oswald_at_clmax"] = effective_oswald_at_max_lift(row)
            entry["angles"] = contour_angles(row, levels=(0.0, -20.0))
        rows.append(entry)
        print(
            name,
            entry.get("sustained_speed"),
            entry.get("sustained_rate"),
            entry.get("gap"),
            entry.get("error", ""),
            flush=True,
        )
    table = replaced(
        TABLES["oswald_closed"], best_climb_speed=None, thrust_softmin_exponent=SOFTNESS_CLOSED
    )
    values = dict(P51D.values, separation_drag=0.0, oswald=0.8)
    ref2 = tuning.reference_of(dataclasses.replace(P51D, values=values), table)
    for static in (0.45, 0.61, 1.0):
        for q in (1.5, 2.83, 8.0):
            design = dict(base, thrust_softmin_exponent=q, static_thrust=static * G)
            row, err = safe(lambda: tuning.design_row(ref2, design, start=start).row)
            entry = {"case": f"static {static} W, softness {q}"}
            if row is None:
                entry["error"] = err
            else:
                entry.update(readout(row))
            rows.append(entry)
            print(
                entry["case"],
                entry.get("sustained_speed"),
                entry.get("sustained_rate"),
                entry.get("gap"),
                entry.get("best_climb_speed_mph"),
                entry.get("error", ""),
                flush=True,
            )
    results["levers"] = rows


def part_push(results):
    """The author's push: performance points, best sustained speed raised, the gap widened."""
    ref = tuning.reference("p51d", "performance_points")
    rows = []
    for speed_real in (81.19, 90.0, 100.0, 113.5):
        for gap in (0.85, 0.75):
            row, err = safe(
                lambda: (
                    tuning.design_row(
                        ref, dict(sustained_speed=speed_real * F, gap=gap), start=ref.start_row
                    ).row
                )
            )
            entry = {"sustained_speed_target": speed_real, "gap_target": gap}
            if row is None:
                entry["error"] = err
            else:
                entry.update(readout(row))
                entry["effective_oswald_at_clmax"] = effective_oswald_at_max_lift(row)
            rows.append(entry)
            print(
                speed_real,
                gap,
                entry.get("oswald"),
                entry.get("separation_drag"),
                entry.get("error", ""),
                flush=True,
            )
    results["push"] = rows


def part_prop(results):
    """The 1 g excess-thrust picture: with a parabolic polar the sustained turn's
    omega^2 = (T - D_1g) / k', so the best sustained speed is the speed of most 1 g excess thrust
    (where it lies inside the line), and the sustained turn leaves the line where the drag at
    C_Lmax equals the thrust. Thrust curves: the game's softmin, and actuator-disc momentum
    theory with a constant profile efficiency (FM = that efficiency at rest)."""
    row = tuning.point_mass({**tuning.row_of("p51d")})
    a = P51D.airframe
    v = np.linspace(30.0, 175.0, 600)  # real m/s
    imp = tuning.implied(tuning.row_of("p51d"), a)
    cd0, e = imp["zero_lift_drag_coefficient"], imp["oswald"]
    k = 1.0 / (math.pi * e * a.aspect_ratio)
    w = a.mass * tuning.REAL_GRAVITY
    power = a.power * imp["propeller_efficiency"]
    t0 = tuning.to_real(row["static_thrust"], -2) * a.mass
    soft = row["thrust_softmin_exponent"]

    def softmin(x, y, q):
        lo, hi = np.minimum(x, y), np.maximum(x, y)
        return lo * (1 + (lo / hi) ** q) ** (-1 / q)

    def actuator(v, profile, shaft):
        """T from P profile = T (v + w_i), T = 2 rho A w_i (v + w_i)."""
        area = a.propeller_disc
        out = np.empty_like(v)
        for i, vi in enumerate(v):
            lo, hi = 0.0, 1e6
            for _ in range(200):
                t = 0.5 * (lo + hi)
                wi = 0.5 * (-vi + math.sqrt(vi * vi + 2 * t / (tuning.AIR_DENSITY * area)))
                if t * (vi + wi) > profile * shaft:
                    hi = t
                else:
                    lo = t
            out[i] = 0.5 * (lo + hi)
        return out

    curves = {
        "constant power P eta / v": power / v,
        "game softmin (q 2.83, T0 0.61 W)": softmin(power / v, t0, soft),
        "actuator disc, profile eff. 0.86": actuator(v, 0.86, a.power),
        "constant thrust (jet-like)": np.full_like(v, softmin(power / 81.19, t0, soft)),
    }
    drag_1g = 0.5 * tuning.AIR_DENSITY * v**2 * a.wing_area * cd0 + k * w**2 / (
        0.5 * tuning.AIR_DENSITY * v**2 * a.wing_area
    )
    out = {"v": v.tolist(), "drag_1g": drag_1g.tolist(), "cd0": cd0, "oswald": e, "curves": {}}
    cl_max = 1.6
    q = 0.5 * tuning.AIR_DENSITY * v**2 * a.wing_area
    drag_max = q * (cd0 + k * cl_max**2)
    stall = math.sqrt(w / (0.5 * tuning.AIR_DENSITY * a.wing_area * cl_max))
    for name, thrust in curves.items():
        excess = thrust - drag_1g
        # sustained load factor from omega^2 = (T - D1g) g^2 / (k W^2 / q...) : do it directly
        # n^2 - 1 = (T - D_1g) q / (k W^2) (parabolic polar, level turn)
        n2 = 1.0 + excess * q / (k * w**2)
        n_lift = q * cl_max / w
        n = np.sqrt(np.clip(np.minimum(n2, n_lift**2), 1.0, None))
        rate = (
            np.where(
                (v > stall) & (excess > 0), np.sqrt(n**2 - 1) * tuning.REAL_GRAVITY / v, np.nan
            )
            * DEG
        )
        radius = v / np.radians(rate)
        best = int(np.nanargmax(rate))
        tight = int(np.nanargmin(radius))
        cross = np.nonzero((thrust - drag_max) < 0)[0]
        out["curves"][name] = dict(
            thrust=thrust.tolist(),
            rate=rate.tolist(),
            best_speed=float(v[best]),
            best_rate=float(rate[best]),
            tight_speed=float(v[tight]),
            gap=float(v[tight] / v[best]),
            excess_peak_speed=float(v[int(np.argmax(np.where(v > stall, excess, -1e9)))]),
            line_leave_speed=float(v[cross[0]]) if len(cross) else None,
        )
        print(name, out["curves"][name]["best_speed"], out["curves"][name]["gap"], flush=True)
    out["stall"] = stall
    out["md_speed"] = float(
        (k * w**2 / ((0.5 * tuning.AIR_DENSITY * a.wing_area) ** 2 * cd0)) ** 0.25
    )
    results["prop"] = out


def RECOMMENDED_ROWS():
    """The candidate models drawn in doghouse-shape-recommended.png."""
    start = tuning.row_of("p51d")
    ref = reference("oswald_closed")
    return [
        ("now: e 0.97 (solved), exponent 16", start),
        (
            "e 0.85, exponent 4, gap 0.85",
            solve("oswald_closed", start=start, oswald=0.85, separation_exponent=4.0),
        ),
        (
            "e 0.85, exponent 2, gap 0.85",
            solve("oswald_closed", start=start, oswald=0.85, separation_exponent=2.0),
        ),
        (
            "e 0.85, exponent 4, best climb x1.5",
            solve(
                "oswald_closed",
                start=start,
                oswald=0.85,
                separation_exponent=4.0,
                best_climb=ref.design["best_climb"] * 1.5,
            ),
        ),
    ]


PARTS = dict(
    polar=part_polar,
    steepness=part_steepness,
    sweep=part_sweep,
    thrust=part_thrust,
    push=part_push,
    prop=part_prop,
    oswald=part_oswald,
    levers=part_levers,
)


def main(argv):
    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    for name in argv or list(PARTS):
        if name == "figures":
            continue
        PARTS[name](results)
        OUT_JSON.write_text(json.dumps(results, indent=1))
    if not argv or "figures" in argv:
        import doghouse_shape_figures  # noqa: F401  (sibling module)

        doghouse_shape_figures.main(results)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main(sys.argv[1:])
