"""Are the design knobs independent? (docs/explanation/game/ship-design.md#independence)

    uv run --no-sync python -m bench.knob_independence --out DIR

At each base type in `--bases`, every knob of `bnb.game.ship_design.KNOBS` is moved by
+-STEP of its envelope deviation (the fire rate to the neighbouring rate), plus two
set-wide knobs, the engine knee (its card scale) and a global charge multiplier (every
type's coast exchange scale with its engine power, as a climb multiplier would); each
moved card is the instant estimate, and its outcome-relevant quantities are measured: the
scripted agent's characteristics (cruise, top and corner speeds, the turn rates at the
three sticks, the climb power, the gun range, the bullet speed, the store) and closed
forms of the fight (the time to recover 40% of the store, the times to kill shields of 80,
120, 160 and 200 (the breaking hits, smooth in the damage, then one), the interval between
shots, the hits a 24-damage gun needs on its shield, the bullet's time to 400 m, the
dispersion, the seconds of continuous fire a store pays for). The Jacobian (log quantities
per normalised knob) gives, per base, the singular values, the condition number and the
pairs of knobs whose columns align (|cos| > `--align`). `independence.json` and the tables
(Markdown) in `--out`. CPU only.
"""

import argparse
import dataclasses
import json
import math
import pathlib

import torch

from bnb.game import api
from bnb.game.handling.characteristics import measure_characteristics
from bnb.game.ship_design import (
    CONTINUOUS,
    FIRE_RATES,
    NAMES,
    Envelope,
    design_cards,
    handles_of,
    knobs_of,
    set_handles,
)
from bnb.game.ship_types import ShipTypes

STEP = 0.25  # envelope deviations
QUANTITIES = (
    "cruise_speed",
    "top_speed",
    "corner_speed",
    "turn_rate_1",
    "turn_rate_2",
    "turn_rate_3",
    "climb_power",
    "gun_range",
    "bullet_speed",
    "store",
    "recovery_time",
    "kill_time_80",
    "kill_time_120",
    "kill_time_160",
    "kill_time_200",
    "fire_interval",
    "hits_taken",
    "flight_time_400",
    "dispersion",
    "fire_seconds",
)
GLOBALS = ("engine_knee", "global_charge")


def quantities(card: ShipTypes, rules) -> torch.Tensor:
    """(types, len(QUANTITIES)) the logs of the quantities of each type of a card."""
    measured = measure_characteristics(card, rules)
    power = measured.climb_power.double()
    drag = measured.bullet_drag.double()
    speed = measured.bullet_speed.double()
    damage, rate = card.hit_damage.double(), card.fire_rate.double()
    store = card.max_energy.double()
    columns = [
        measured.cruise_speed.double(),
        measured.top_speed.double(),
        measured.corner_speed.double(),
        *measured.turn_rate.double().unbind(-1),
        power,
        measured.gun_range.double(),
        speed,
        store,
        0.4 * store / power,
        # a shield's kill: the breaking hits, then one (smooth in the damage)
        *((shield / damage + 1.0) / rate for shield in (80.0, 120.0, 160.0, 200.0)),
        1.0 / rate,
        card.max_shield.double() / 24.0 + 1.0,
        torch.expm1(drag * 400.0) / (drag * speed),
        card.bullet_spread_scale.double() * rules.bullet_spread / card.muzzle_speed.double(),
        store / (card.bullet_energy_cost.double() * rate),
    ]
    return torch.stack(columns, -1).log()


def moved(base_card: ShipTypes, index: int, name: str, factor: float) -> ShipTypes:
    """A one-type card with a set-wide knob moved: the knee scale, or the charge (the coast
    exchange scale and the engine power's charging part, so the cruise stays)."""
    fields = {
        f.name: getattr(base_card, f.name)[index : index + 1].clone()
        for f in dataclasses.fields(base_card)
    }
    if name == "engine_knee":
        fields["engine_knee_scale"] *= factor
    else:
        full = fields["engine_power_full_store"]
        fields["engine_power"] = full + (fields["engine_power"] - full) * factor
        fields["coast_exchange_scale"] *= factor
    return ShipTypes(**fields)


def main() -> None:
    """The analysis the module describes."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--bases", default="p51d,spitfire-ix,fw190-a8,p47d,a6m5,ki84,albatross,zerstorer"
    )
    parser.add_argument("--align", type=float, default=0.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    game = api.PRESETS["real_scale_deathmatch"]
    rules = api.rules_of(game)
    handles = set_handles()
    envelope = Envelope.of(handles)
    bases = args.bases.split(",")
    rows, labels = [], []
    for key in bases:
        base = handles[key]
        knobs = knobs_of(base)
        rows.append((key, base))
        labels.append((key, "base", 0.0))
        for index, name in enumerate(CONTINUOUS):
            for sign in (1.0, -1.0):
                factor = math.exp(sign * STEP * float(envelope.deviation[index]))
                rows.append((key, handles_of(base, {name: knobs[name] * factor}, rules)))
                labels.append((key, name, sign))
        rates = list(FIRE_RATES)
        position = rates.index(knobs["fire_rate"]) if knobs["fire_rate"] in rates else None
        for sign in (1.0, -1.0):
            step = None if position is None else position + int(sign)
            if step is not None and 0 <= step < len(rates):
                rows.append((key, handles_of(base, {"fire_rate": rates[step]}, rules)))
                labels.append((key, "fire_rate", sign))
    card = design_cards(rules, [key for key, _ in rows], [h for _, h in rows])
    measured = quantities(card, rules)
    # the set-wide knobs, on each base's own card (the instant one at its base handles)
    base_rows = [i for i, label in enumerate(labels) if label[1] == "base"]
    extra = []
    for i in base_rows:
        for name in GLOBALS:
            for sign in (1.0, -1.0):
                factor = math.exp(sign * STEP * math.log(2.0))  # a factor 2 is one unit
                extra.append((moved(card, i, name, factor), (labels[i][0], name, sign)))
    extra_values = [quantities(c, rules)[0] for c, _ in extra]
    report = {"step": STEP, "quantities": QUANTITIES, "bases": {}}
    lines = ["# Knob independence\n"]
    for key in bases:
        base_index = labels.index((key, "base", 0.0))
        columns, names = [], []
        for name in NAMES:
            plus = [i for i, label in enumerate(labels) if label == (key, name, 1.0)]
            minus = [i for i, label in enumerate(labels) if label == (key, name, -1.0)]
            if name == "fire_rate":  # per rate step (a discrete knob: one step is a unit)
                ends = plus + minus or []
                if not ends:
                    continue
                if plus and minus:
                    column = (measured[plus[0]] - measured[minus[0]]) / 2.0
                elif plus:
                    column = measured[plus[0]] - measured[base_index]
                else:
                    column = measured[base_index] - measured[minus[0]]
            else:
                column = (measured[plus[0]] - measured[minus[0]]) / (2.0 * STEP)
            columns.append(column)
            names.append(name)
        knob_count = len(names)
        for name in GLOBALS:
            plus = next(
                v for v, (_, label) in zip(extra_values, extra) if label == (key, name, 1.0)
            )
            minus = next(
                v for v, (_, label) in zip(extra_values, extra) if label == (key, name, -1.0)
            )
            columns.append((plus - minus) / (2.0 * STEP))
            names.append(name)
        jacobian = torch.stack(columns, -1)  # (quantities, knobs)
        searched = jacobian[:, :knob_count]
        singular = torch.linalg.svdvals(searched)
        unit = jacobian / jacobian.norm(dim=0, keepdim=True).clamp_min(1e-12)
        cosines = unit.T @ unit
        singular_unit = torch.linalg.svdvals(unit[:, :knob_count])
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if abs(float(cosines[i, j])) > args.align:
                    pairs.append((names[i], names[j], float(cosines[i, j])))
        pairs.sort(key=lambda pair: -abs(pair[2]))
        report["bases"][key] = dict(
            knobs=names,
            jacobian=jacobian.tolist(),
            singular_values=singular.tolist(),
            condition=float(singular[0] / singular[-1]),
            condition_unit_columns=float(singular_unit[0] / singular_unit[-1]),
            aligned=pairs,
        )
        lines.append(
            f"\n## {key}\n\nsingular values (searched knobs, per envelope deviation): "
            + ", ".join(f"{value:.3g}" for value in singular.tolist())
            + f"; condition {float(singular[0] / singular[-1]):.3g} (unit columns "
            f"{float(singular_unit[0] / singular_unit[-1]):.3g})\n\n"
            "| Knob | Knob | cos |\n|---|---|---|\n"
            + "".join(f"| {a} | {b} | {c:+.2f} |\n" for a, b, c in pairs)
        )
    (out / "independence.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "independence.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
