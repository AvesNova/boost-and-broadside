"""The reference chart (ship-card spec 6): every handle, parameter and shown quantity of a
card, printed per type, and its charts (the Ps curve, the E-M doghouse and the similarity
ratios against the P-51D), written with matplotlib (the optional `tools` extra).
"""

import dataclasses
import json
import pathlib
from typing import NamedTuple

import torch

from ..rules import Rules
from ..ship_types import DEFAULTS, SCALED, ShipTypes, ship_types
from . import shown as shown_module
from .handles import DEGREES, NAMES, UNITS, Handles
from .manoeuvres import batched, cruise_speed
from .solver import measure

TYPE_NAMES = ("fighter", "heavy")
CHARTS = ("ps_curve.png", "doghouse.png", "similarity.png")


def load_card(path: str | None) -> ShipTypes:
    """The card in a JSON file of ship-card fields (per-type lists; missing fields keep the
    default), or the default card; float64."""
    values = json.loads(pathlib.Path(path).read_text()) if path else {}
    return ship_types(dtype=torch.float64, **values)


class Report(NamedTuple):
    """A card's measured handles, shown quantities and similarity ratios, for printing and
    charting."""

    handles: Handles  # (num_types,) fields
    view: shown_module.Shown  # (1, num_types, ...)
    ratios: dict  # ours, (num_types,) per ratio
    reference: dict  # the P-51D's, per ratio


def report(rules: Rules, types: ShipTypes) -> Report:
    """Measure one card (num_types,) and gather what the chart prints and plots."""
    types = batched(types)
    handles = measure(types, rules)
    view = shown_module.shown(rules, types)
    ratios = shown_module.similarity_ratios(shown_module.similarity_inputs(rules, handles, view))
    return Report(
        Handles(**{n: getattr(handles, n)[0] for n in NAMES}),
        view,
        {k: v[0] for k, v in ratios.items()},
        shown_module.similarity_ratios(shown_module.P51D),
    )


def _row(label: str, values, digits: int = 4) -> str:
    return f"  {label:46s}" + "".join(f"{float(v):>14.{digits}g}" for v in values)


def lines(rules: Rules, types: ShipTypes, card: Report) -> list[str]:
    """The printed chart."""
    names = TYPE_NAMES[: len(card.handles.ceiling)]
    header = f"  {'':46s}" + "".join(f"{name:>14s}" for name in names)
    out = [f"Ship card at {1.0 / rules.tick_length:g} Hz", "", "Handles", header]
    for name in NAMES:
        unit, factor = UNITS[name]
        out.append(_row(f"{name} ({unit})" if unit else name, getattr(card.handles, name) * factor))
    out += ["", "Parameters", header]
    for name in DEFAULTS:
        out.append(_row(name, getattr(types, name)))
    for name, (owner, shared) in SCALED.items():
        reference = getattr(rules.flight if owner == "flight" else rules, shared)
        out.append(_row(f"  = {shared}", reference * getattr(types, name)))
    view = card.view
    out += [
        "",
        "Shown",
        header,
        _row("corner speed (m/s)", view.corner_speed[0]),
        _row("instantaneous peak turn (deg/s)", view.peak_turn_rate[0] * DEGREES),
        _row("  at speed (m/s)", view.peak_turn_speed[0]),
        _row("minimum radius (m)", view.minimum_radius[0]),
    ]
    for i, stick in enumerate(shown_module.STICKS):
        out += [
            _row(f"stick {stick:g}: gun offset (deg)", view.datum.angle[0, :, i] * DEGREES),
            _row(f"stick {stick:g}: coast datum speed (m/s)", view.datum.speed[0, :, i]),
            _row(
                f"stick {stick:g}: coast datum turn (deg/s)",
                view.datum.turn_rate[0, :, i] * DEGREES,
            ),
        ]
    out.append(_row("top speed (m/s)", view.top_speed[0]))
    for condition, seconds in view.endurance.items():
        out.append(_row(f"firing endurance, {condition.replace('_', ' ')} (s)", seconds[0]))
    out.append(_row("damage per second", view.damage_per_second[0]))
    out += [
        "",
        "Similarity against the P-51D (ship-card.md)",
        f"  {'':46s}"
        + "".join(f"{name:>14s}" for name in names)
        + f"{'P-51D':>14s}"
        + "".join(f"{name + '/P-51D':>16s}" for name in names),
    ]
    for ratio, ours in card.ratios.items():
        reference = card.reference[ratio]
        out.append(
            _row(ratio, [*ours, reference])
            + "".join(f"{float(v / reference):>16.3g}" for v in ours)
        )
    out += ["", "P-51D inputs: " + ", ".join(f"{k} {v:g}" for k, v in shown_module.P51D.items())]
    return out


def _curves(rules: Rules, types: ShipTypes, card: Report):
    """Speed grids and the instantaneous rates on them, per stick level (0 is straight)."""
    stall = card.handles.stall_speed
    corner = card.view.corner_speed[0]
    top = torch.maximum(card.view.top_speed[0], 1.6 * corner)
    fraction = torch.linspace(0.0, 1.0, shown_module.SPEED_GRID, dtype=stall.dtype)
    speeds = ((1.01 * stall)[:, None] + (top - 1.01 * stall)[:, None] * fraction)[None]
    return {
        stick: shown_module.instant(rules, types, speeds, stick)
        for stick in (0.0, *shown_module.STICKS)
    }


def write_charts(rules: Rules, types: ShipTypes, card: Report, directory: str) -> list[str]:
    """Write CHARTS into directory; returns their paths. Raises ImportError without
    matplotlib (the optional tools extra)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("the charts need matplotlib: uv sync --extra tools") from error
    types = batched(types)
    out = pathlib.Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    curves = _curves(rules, types, card)
    names = TYPE_NAMES[: len(card.handles.ceiling)]
    constants = rules.flight
    sustained = shown_module.settled(rules, types, shown_module.SUSTAINED_CURVE)
    cruise = cruise_speed(rules, types, types.max_energy[..., None])[0, :, 0]
    paths = []

    figure, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4.5), squeeze=False)
    for t, (name, axis) in enumerate(zip(names, axes[0])):
        for stick, curve in curves.items():
            axis.plot(
                curve.speed[0, t],
                curve.specific_power[0, t],
                label="straight" if stick == 0.0 else f"stick {stick:g}",
            )
        axis.axhline(0.0, color="grey", linewidth=0.8)
        axis.axvline(
            float(card.view.corner_speed[0, t]), color="grey", linestyle=":", label="corner speed"
        )
        axis.axvline(float(cruise[t]), color="grey", linestyle="--", label="cruise")
        axis.set(
            title=f"{name}: specific excess power, coasting",
            xlabel="speed (m/s)",
            ylabel="Ps (W/kg)",
        )
        axis.legend(fontsize=8)
    paths.append(out / CHARTS[0])
    figure.tight_layout()
    figure.savefig(paths[-1], dpi=120)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4.5), squeeze=False)
    for t, (name, axis) in enumerate(zip(names, axes[0])):
        speed = curves[1.0].speed[0, t]
        lift_limit = types.max_lift_coefficient[0, t] * speed**2
        stall_boundary = (lift_limit**2 - constants.gravity**2).clamp_min(0.0).sqrt() / speed
        acceleration = types.max_acceleration[0, t]
        g_line = torch.sqrt(acceleration**2 - constants.gravity**2) / speed
        axis.plot(speed, stall_boundary * DEGREES, "k--", linewidth=1, label="stall boundary")
        axis.plot(speed, g_line * DEGREES, "k-.", linewidth=1, label="g limit")
        for stick in shown_module.STICKS:
            axis.plot(speed, curves[stick].turn_rate[0, t] * DEGREES, label=f"stick {stick:g}")
        axis.plot(
            sustained.speed[0, t],
            sustained.turn_rate[0, t] * DEGREES,
            "o-",
            color="black",
            markersize=3,
            label="sustained (coasting)",
        )
        axis.axvline(
            float(card.view.corner_speed[0, t]), color="grey", linestyle=":", label="corner speed"
        )
        axis.set(
            title=f"{name}: E-M turn performance",
            xlabel="speed (m/s)",
            ylabel="turn rate (deg/s)",
            ylim=(0.0, 1.15 * float(g_line.max() * DEGREES)),
        )
        axis.legend(fontsize=8)
    paths.append(out / CHARTS[1])
    figure.tight_layout()
    figure.savefig(paths[-1], dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    labels = list(card.ratios)
    height = 0.8 / len(names)
    for t, name in enumerate(names):
        values = [float(card.ratios[k][t] / card.reference[k]) for k in labels]
        axis.barh([i + t * height for i in range(len(labels))], values, height=height, label=name)
    axis.axvline(1.0, color="black", linewidth=0.8)
    axis.set_xscale("log")
    axis.set_yticks([i + 0.4 - height / 2 for i in range(len(labels))], labels, fontsize=8)
    axis.set(title="similarity ratios: ours / P-51D", xlabel="ratio (log scale)")
    axis.legend()
    paths.append(out / CHARTS[2])
    figure.tight_layout()
    figure.savefig(paths[-1], dpi=120)
    plt.close(figure)
    return [str(p) for p in paths]


def save_card(types: ShipTypes, path: str) -> None:
    """A card as the JSON load_card reads."""
    values = {f.name: getattr(types, f.name).tolist() for f in dataclasses.fields(types)}
    pathlib.Path(path).write_text(json.dumps(values, indent=1))
