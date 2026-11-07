"""Markdown tables of the state-delta statistics and code comparison
(bench/state_deltas_measure.py, bench/state_deltas_codes.py), printed to stdout for
docs/rewrite/sections/state-deltas.md."""

import json
import pathlib

SCRATCH = pathlib.Path("/tmp/bnb-scratch-deltas")
PLAYERS = ("scripted", "calm-fern")
PRESETS = ("frontline", "deathmatch", "deathmatch_wide")
ORDER = (
    "position",
    "velocity",
    "heading",
    "nose_rate",
    "angle_of_attack",
    "shield_recharge_wait",
    "shield",
    "energy",
    "gun_cooldown",
    "log_refractive_index",
)
UNITS = dict(
    position="m",
    velocity="m/s",
    heading="rad",
    nose_rate="rad/s",
    angle_of_attack="rad",
    shield_recharge_wait="s",
    shield="fraction of max",
    energy="fraction of max",
    gun_cooldown="ticks",
    log_refractive_index="ln n",
)
SPLIT_BASE = dict(
    position="arc",
    velocity="rotated",
    heading="dead_reckoning",
    nose_rate="persistence",
    angle_of_attack="persistence",
    shield_recharge_wait="countdown",
    shield="recharge",
    energy="persistence",
    gun_cooldown="countdown",
    log_refractive_index="dead_reckoning",
)


def g(value) -> str:
    if value is None or value != value:
        return "-"
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3g}"


def percent(value) -> str:
    return f"{100 * value:.1f}%" if value >= 0.001 or value == 0 else f"{100 * value:.2g}%"


def load():
    data = {}
    for player in PLAYERS:
        for preset in PRESETS:
            path = SCRATCH / "stats" / player / f"{preset}.json"
            if path.exists():
                data[player, preset] = json.loads(path.read_text())
    return data


def counts_table(data) -> list[str]:
    lines = [
        "| player | preset | environments | ticks | episode ends | games touched | "
        "living ship-ticks (1 / 8 / 32-tick pairs) |",
        "|---|---|---|---|---|---|---|",
    ]
    for (player, preset), entry in data.items():
        m = entry["meta"]
        lines.append(
            f"| {player} | {preset} | {m['environments']} | {m['segments']} x "
            f"{m['ticks_per_segment']} | {m['episode_ends']} | {m['games_touched']} | "
            f"{m['ship_ticks_1']:,} / {m['ship_ticks_8']:,} / {m['ship_ticks_32']:,} |"
        )
    return lines


def quantity_table(data, quantity) -> list[str]:
    vector = quantity in ("position", "velocity")
    head = (
        "| player | preset | baseline | k | zero | < f32 | mean | p50 | p90 | p99 | p99.9 | max |"
    )
    rule = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    if vector:
        head += " x p50 / p99.9 | y p50 / p99.9 | x mean / y mean |"
        rule += "---|---|---|"
    lines = [head, rule]
    for (player, preset), entry in data.items():
        for base, horizons in entry["stats"][quantity].items():
            for horizon, components in sorted(horizons.items(), key=lambda kv: int(kv[0])):
                main = components.get("norm") or components.get("angle") or components["value"]
                row = (
                    f"| {player} | {preset} | {base} | {horizon} | {percent(main['zero'])} | "
                    f"{percent(main['below_float32'])} | {g(main['mean'])} | {g(main['p50'])} | "
                    f"{g(main['p90'])} | {g(main['p99'])} | {g(main['p99.9'])} | "
                    f"{g(main['max'])} |"
                )
                if vector:
                    x, y = components["x"], components["y"]
                    row += (
                        f" {g(x['p50'])} / {g(x['p99.9'])} | {g(y['p50'])} / {g(y['p99.9'])} |"
                        f" {g(x['mean'])} / {g(y['mean'])} |"
                    )
                lines.append(row)
    return lines


def body_table(data, quantity) -> list[str]:
    """The body-frame components of a 2-D change: p50 / p99.9 of |component| and the signed
    mean, along and across the nose and the velocity at t."""
    parts = ("nose_along", "nose_across", "track_along", "track_across")
    lines = [
        "| player | preset | baseline | k | " + " | ".join(parts) + " | track skipped |",
        "|---|---|---|---|" + "---|" * (len(parts) + 1),
    ]
    for (player, preset), entry in data.items():
        for base, horizons in entry["stats"][quantity].items():
            for horizon, components in sorted(horizons.items(), key=lambda kv: int(kv[0])):
                cells = [
                    f"{g(components[c]['p50'])} / {g(components[c]['p99.9'])} "
                    f"(mean {g(components[c]['mean'])})"
                    for c in parts
                ]
                skipped = percent(components["track_along"]["skipped"])
                lines.append(
                    f"| {player} | {preset} | {base} | {horizon} | " + " | ".join(cells)
                    + f" | {skipped} |"
                )  # fmt: skip
    return lines


BODY_SITUATIONS = ("boost", "coast", "reverse", "stick_centred", "stick_full")


def body_split_table(data, quantity, base) -> list[str]:
    """Signed mean and p50 / p99.9 of each body component by power and stick, 1 tick."""
    lines = [
        "| player | preset | component | " + " | ".join(BODY_SITUATIONS) + " |",
        "|---|---|---|" + "---|" * len(BODY_SITUATIONS),
    ]
    for (player, preset), entry in data.items():
        for component in ("nose_along", "nose_across", "track_along", "track_across"):
            split = entry["splits"][quantity][f"{base}:{component}"]
            cells = []
            for situation in BODY_SITUATIONS:
                inside = split[situation]["in"]
                cells.append(
                    "-"
                    if inside["n"] == 0
                    else f"{percent(inside['share'])}: mean {g(inside['mean'])}, "
                    f"{g(inside['p50'])} / {g(inside['p99.9'])}"
                )
            lines.append(f"| {player} | {preset} | {component} | " + " | ".join(cells) + " |")
    return lines


def budget_table(sweep: dict) -> list[str]:
    lines = [
        "| bins | width at p50 (x median) | width at p99.9 | nearest-centre error | H |",
        "|---|---|---|---|---|",
    ]
    for bins, row in sweep["sweep"].items():
        lines.append(
            f"| {bins} | {g(row['width_p50'])} ({g(row['width_p50_fraction'])}) | "
            f"{g(row['width_p999'])} | {g(row['nearest'])} | {row['H']:.2f} |"
        )
    return lines


def split_table(data, quantity) -> list[str]:
    base = SPLIT_BASE[quantity]
    keys = list(data)
    head = "| situation | " + " | ".join(f"{p} {s}" for p, s in keys) + " |"
    lines = [head, "|---|" + "---|" * len(keys)]
    situations = next(iter(data.values()))["splits"][quantity][base]
    for situation in situations:
        cells = []
        for key in keys:
            entry = data[key]["splits"][quantity][base][situation]
            inside, outside = entry["in"], entry["out"]
            if inside["n"] == 0:
                cells.append("-")
                continue
            cells.append(
                f"{percent(inside['share'])}: {g(inside['p50'])} / {g(inside['p99.9'])} "
                f"(out {g(outside.get('p50'))} / {g(outside.get('p99.9'))})"
            )
        lines.append(f"| {situation} | " + " | ".join(cells) + " |")
    return lines


def code_table(entry) -> list[str]:
    lines = [
        "| candidate | bins | coverage | width at p50 (x median) | width at p99.9 | "
        "nearest-centre error | H (nats) | H_self (nats) | clipped |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, c in entry["candidates"].items():
        lines.append(
            f"| {name} | {c['bins']} | {100 * c['coverage']:.2f}% | {g(c['width_p50'])} "
            f"({g(c['width_p50_fraction'])}) | {g(c['width_p999'])} | {g(c['nearest'])} | "
            f"{c['H']:.2f} | {c['H_self']:.2f} | {c['tail']} |"
        )
    return lines


def main() -> None:
    data = load()
    out = ["## Samples", "", *counts_table(data), ""]
    for quantity in ORDER:
        out += [f"## {quantity} ({UNITS[quantity]})", "", *quantity_table(data, quantity), ""]
        if quantity in ("position", "velocity"):
            out += ["Body frames: p50 / p99.9 (signed mean)", "", *body_table(data, quantity), ""]
            for base in ("persistence", SPLIT_BASE[quantity]):
                out += [f"Body components by power and stick ({base}, 1 tick)", "",
                        *body_split_table(data, quantity, base), ""]  # fmt: skip
        out += [f"Splits ({SPLIT_BASE[quantity]}, 1 tick): share: p50 / p99.9 inside "
                "(outside)", "", *split_table(data, quantity), ""]  # fmt: skip
    codes = json.loads((SCRATCH / "codes.json").read_text())
    out += ["# Codes", ""]
    for name, entry in codes.items():
        if "candidates" not in entry:
            out += [f"## {name}", "", "```", json.dumps(entry), "```", ""]
            continue
        summary = {k: entry[k] for k in entry if k not in ("candidates", "budget")}
        out += [f"## {name}", "", " ".join(f"{k} {g(v)}" for k, v in summary.items()), ""]
        out += [*code_table(entry), ""]
        budget = entry.get("budget", {})
        sweeps = {"": budget} if "sweep" in budget else budget
        for component, sweep in sweeps.items():
            out += [f"Symlog bins sweep {component} (median {g(sweep['median'])}, "
                    f"p99.9 {g(sweep['p999'])})", "", *budget_table(sweep), ""]  # fmt: skip
    print("\n".join(out))


if __name__ == "__main__":
    main()
