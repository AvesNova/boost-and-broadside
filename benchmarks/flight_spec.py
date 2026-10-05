"""Solve a handling spec, then check and correct it against the flight harness.

The linear solve lives in ``boost_and_broadside.config.handling``; this script
adds the simulated forward model. It prints the closed-form handling of the
live configuration, the measured envelope next to it, and, with ``--correct``,
the targets the closed forms would need for the measured envelope to land on
the spec (the correction stage of the redesign plan, section 5.2).

Example:
    uv run --no-sync python -m benchmarks.flight_spec --correct
"""

from __future__ import annotations

import argparse

import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.config.defaults import SHIP_CONFIG, STARTING_HANDLING
from boost_and_broadside.config.handling import (
    PARAMETERS,
    correct,
    handling,
    quantities,
)
from boost_and_broadside.env.frontline import frontline_ship_config

# Quantity -> (power mode, turn, terminal row key) in ``measure_terminal``.
MEASURED_TERMINALS = {
    "cruise_speed": ("coast", "GO_STRAIGHT", "proper_speed"),
    "top_speed": ("boost", "GO_STRAIGHT", "proper_speed"),
    "unlimited_top_speed": ("boost_unlimited", "GO_STRAIGHT", "proper_speed"),
    "sharp_sustained_speed": ("boost", "SHARP_LEFT", "proper_speed"),
    "sharp_sustained_rate": ("boost", "SHARP_LEFT", "turn_rate_deg_s"),
    "coast_sharp_speed": ("coast", "SHARP_LEFT", "proper_speed"),
    "coast_normal_speed": ("coast", "TURN_LEFT", "proper_speed"),
    "sharp_radius": ("coast", "SHARP_LEFT", "radius_world_px"),
    "normal_radius": ("coast", "TURN_LEFT", "radius_world_px"),
}


def measure(config: ShipConfig, device: torch.device) -> dict[str, float]:
    """The harness's value of every quantity it can measure, at n = 1."""
    from benchmarks.flight_envelope import measure_terminal

    rows = measure_terminal(config, device, indices=(1.0,), seconds=90.0)
    by_key = {(r["power_mode"], r["turn"]): r for r in rows}
    values = {
        name: abs(by_key[(mode, turn)][key])
        for name, (mode, turn, key) in MEASURED_TERMINALS.items()
    }
    return values


def measure_onset(config: ShipConfig, device: torch.device) -> float:
    """Onset to 90% at 60 px/s. Whole ticks, so the correction stage leaves it out."""
    from benchmarks.flight_envelope import measure_nose

    return measure_nose(config, device, speeds=(60.0,))[0]["onset_90_seconds"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--correct", action="store_true", help="run the correction stage")
    args = parser.parse_args()
    device = torch.device(args.device)
    live = frontline_ship_config(SHIP_CONFIG)
    closed = handling(live)
    catalogue = {q.name: q for q in quantities(live.max_power)}
    targets = STARTING_HANDLING.targets()

    with torch.inference_mode():
        measured = measure(live, device)
        measured["onset_time"] = measure_onset(live, device)
    print("== Live handling: spec, closed form, measured (n = 1)")
    print(f"  {'quantity':<24}{'spec':>9}{'closed':>9}{'measured':>10}{'error':>8}  unit")
    for name, value in closed.items():
        spec = targets.get(name)
        got = measured.get(name)
        error = "" if got is None else f"{got / value - 1:+.1%}"
        print(
            f"  {name:<24}{'' if spec is None else f'{spec:.4g}':>9}{value:>9.4g}"
            f"{'' if got is None else f'{got:.4g}':>10}{error:>8}  {catalogue[name].unit}"
        )

    if args.correct:
        pinned = {name: v for name, v in targets.items() if name in MEASURED_TERMINALS}
        with torch.inference_mode():
            result = correct(targets, live, lambda c: measure(c, device))
        print(f"\n== Correction: {result.iterations} iteration(s)")
        for name in pinned:
            print(
                f"  {name:<24} target {targets[name]:>8.4g}  closed form asked "
                f"{result.corrected_targets[name]:>8.4g}  error {result.errors.get(name, 0):+.2%}"
            )
        print(f"  {'parameter':<24}{'live':>12}{'corrected':>12}")
        corrected = result.solution.config
        for name in PARAMETERS:
            print(f"  {name:<24}{getattr(live, name):>12.5g}{getattr(corrected, name):>12.5g}")


if __name__ == "__main__":
    main()
