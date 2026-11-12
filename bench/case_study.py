"""A case study of one trade-off: variants of a gun, a shield or a flight value on fixed
airframes, rated against each other with the league's tournament on the no-recompile arena
(docs/explanation/game/case-studies.md).

    uv run --no-sync python -m bench.case_study play --spec SPEC.json --out DIR
    uv run --no-sync python -m bench.case_study report --spec SPEC.json --out DIR

SPEC.json:

- "preset" (real_scale_deathmatch by default);
- "airframes": base type keys of the real-scale set; every variant flies each of them;
- "variants": each {"name", "knobs": absolute knob values (`bnb.game.ship_design.KNOBS`),
  "scale": knob multipliers, "handles": handle multipliers (any handle, e.g.
  "shots_per_store": 0.25 for four times the shot cost, "shield_recharge_delay": 2), "card":
  card-field multipliers (e.g. "engine_knee_scale": 1.4, the knee at 0.7)}; the first variant
  is the baseline (usually no change);
- "cross": who plays whom: "airframe" (each variant only the variants on its own airframe,
  and the baselines all play each other, which ties the airframes' scales together) or "all";
- "target_stderr" (Elo, 25 by default) and "budget" (games, 60,000 by default).

`play` rates every (airframe, variant) slot with the league's A-optimal tournament
(`bnb.league.tournament.fit_field`; at most 36 slots, the first airframe's baseline the
anchor at 0), each batch appended to DIR/batches.jsonl as it ends (resumed if rerun), every
ship of a team the slot's type, the scripted agent on both sides (the preset's real-scale set,
T = 1). Bounds: the target, the budget and `--hours`.

`report` writes DIR/report.json and DIR/report.md: each slot's Elo against its airframe's
baseline (± its standard error), a linear model of that Elo in the variants' log changes
(Elo per doubling of each varied quantity, by least squares over the variants of each
airframe and pooled), the majority cycles among the slots and every pair's score.
"""

import argparse
import datetime
import json
import math
import pathlib
import time

import numpy as np
import torch

from bench.lock import bench_lock
from bench.ship_design_search import ARENA_BLOCKS, BLOCK_GAMES, SLOTS, DesignArena, padded
from bnb.game import api
from bnb.game.ship_design import design_cards, handles_of, knobs_of, set_handles
from bnb.league import cyclic
from bnb.league.matches import ArenaConfig
from bnb.league.tournament import TournamentConfig, fit_field


def slots_of(spec: dict) -> list[tuple[str, dict]]:
    """Every (airframe, variant) slot, airframe by airframe."""
    return [(key, variant) for key in spec["airframes"] for variant in spec["variants"]]


def name_of(key: str, variant: dict) -> str:
    """A slot's name: its airframe and its variant."""
    return f"{key}: {variant['name']}"


def variant_handles(base: dict, variant: dict, rules) -> dict:
    """A variant's handles on an airframe: its knobs (absolute, then multiplied), then its
    handle multipliers."""
    knobs = knobs_of(base)
    changed = dict(variant.get("knobs", {}))
    for name, factor in variant.get("scale", {}).items():
        changed[name] = changed.get(name, knobs[name]) * factor
    handles = handles_of(base, changed, rules)
    for name, factor in variant.get("handles", {}).items():
        handles[name] = handles[name] * factor
    return handles


def card_of(spec: dict, rules):
    """The arena's card for a spec's slots (padded to SLOTS)."""
    found = set_handles()
    slots = slots_of(spec)
    if len(slots) > SLOTS:
        raise SystemExit(f"{len(slots)} slots: at most {SLOTS}")
    handles = [variant_handles(found[key], variant, rules) for key, variant in slots]
    card = design_cards(rules, [key for key, _ in slots], handles)
    for index, (_, variant) in enumerate(slots):  # card-field multipliers (the live knee)
        for name, factor in variant.get("card", {}).items():
            getattr(card, name)[index] *= factor
    return padded(card)


def allowed_of(spec: dict) -> torch.Tensor:
    """Who may play whom: the spec's "cross"."""
    slots = slots_of(spec)
    count = len(slots)
    allowed = ~torch.eye(count, dtype=torch.bool)
    if spec.get("cross", "airframe") == "airframe":
        for i, (a, va) in enumerate(slots):
            for j, (b, vb) in enumerate(slots):
                baselines = va is spec["variants"][0] and vb is spec["variants"][0]
                allowed[i, j] = i != j and (a == b or baselines)
    return allowed


def play(args) -> None:
    """The `play` command (the module docstring)."""
    spec = json.loads(pathlib.Path(args.spec).read_text())
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preset = spec.get("preset", "real_scale_deathmatch")
    rules = api.rules_of(api.PRESETS[preset])
    names = [name_of(key, variant) for key, variant in slots_of(spec)]
    card = card_of(spec, rules)
    config = TournamentConfig(
        ArenaConfig(blocks=ARENA_BLOCKS, block_environments=BLOCK_GAMES),
        target_stderr=spec.get("target_stderr", 25.0),
        budget=spec.get("budget", 60_000),
        allocation="a_optimal",
    )
    deadline = time.time() + 3600.0 * args.hours
    with bench_lock():
        arena = DesignArena(preset, torch.device("cuda"))
        arena.arena.set_card(card)

        def batch(matchups):
            if time.time() > deadline:
                raise SystemExit(f"past the {args.hours} h bound: rerun to resume")
            return torch.as_tensor(arena.play(None, matchups))

        def log(line: str) -> None:
            print(f"{datetime.datetime.now():%H:%M:%S} {line}", flush=True)

        fit_field(
            batch,
            names,
            0,
            allowed_of(spec),
            config,
            log,
            spec.get("name", "case"),
            out / "batches.jsonl",
        )


def counts_of(record: pathlib.Path, names: list[str]) -> torch.Tensor:
    """(K, K, 3) each ordered pair's wins, ties and losses from a record."""
    count = len(names)
    counts = torch.zeros((count, count, 3), dtype=torch.float64)
    for line in record.read_text().splitlines():
        row = json.loads(line)
        for (first, second), (won, tied, lost) in zip(row["matchups"], row["results"], strict=True):
            a, b = names.index(first), names.index(second)
            counts[a, b] += torch.tensor([won, tied, lost], dtype=torch.float64)
            counts[b, a] += torch.tensor([lost, tied, won], dtype=torch.float64)
    return counts


def log_changes(spec: dict, rules) -> tuple[list[str], np.ndarray]:
    """The quantities the variants vary and each slot's log change of them from its
    airframe's baseline (a knob's, or a handle's)."""
    found = set_handles()
    quantities: list[str] = []
    for variant in spec["variants"]:
        for name in list(variant.get("knobs", {})) + list(variant.get("scale", {})):
            if name not in quantities:
                quantities.append(name)
        for kind in ("handles", "card"):
            for name in variant.get(kind, {}):
                label = f"{kind[:-1] if kind == 'handles' else kind} {name}"
                if label not in quantities:
                    quantities.append(label)
    rows = []
    for key, variant in slots_of(spec):
        base = found[key]
        mine = variant_handles(base, variant, rules)
        base_knobs = knobs_of(variant_handles(base, spec["variants"][0], rules))
        knobs = knobs_of(mine)
        row = []
        for quantity in quantities:
            if quantity.startswith(("handle ", "card ")):
                kind, name = quantity.split(" ", 1)  # the multiplier on what the knobs give
                kind = "handles" if kind == "handle" else "card"
                row.append(math.log(variant.get(kind, {}).get(name, 1.0)))
            else:
                row.append(math.log(knobs[quantity] / base_knobs[quantity]))
        rows.append(row)
    return quantities, np.array(rows)


def report(args) -> None:
    """The `report` command (the module docstring)."""
    spec = json.loads(pathlib.Path(args.spec).read_text())
    out = pathlib.Path(args.out)
    preset = spec.get("preset", "real_scale_deathmatch")
    rules = api.rules_of(api.PRESETS[preset])
    slots = slots_of(spec)
    names = [name_of(key, variant) for key, variant in slots]
    last = json.loads((out / "batches.jsonl").read_text().splitlines()[-1])
    by_name = {row["player"]: row for row in last["ratings"]}
    elo = np.array([by_name[name]["rating"] for name in names])  # Elo, the anchor at 0
    stderr = np.array([by_name[name]["stderr"] for name in names])
    baseline = {
        key: elo[names.index(name_of(key, spec["variants"][0]))] for key in spec["airframes"]
    }
    relative = np.array([elo[i] - baseline[key] for i, (key, _) in enumerate(slots)])
    quantities, changes = log_changes(spec, rules)
    models = {}
    for scope in [*spec["airframes"], "pooled"]:
        rows = [i for i, (key, _) in enumerate(slots) if scope in ("pooled", key)]
        x, y = changes[rows], relative[rows]
        weights = 1.0 / np.maximum(stderr[rows], 5.0) ** 2
        used = [q for q in range(len(quantities)) if np.abs(x[:, q]).max() > 1e-9]
        if not used:
            continue
        a = x[:, used] * np.sqrt(weights)[:, None]
        b = y * np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(a, b, rcond=None)
        fitted = x[:, used] @ coefficients
        models[scope] = dict(
            elo_per_doubling={
                quantities[q]: float(c * math.log(2.0)) for q, c in zip(used, coefficients)
            },
            rms_residual=float(np.sqrt(np.mean((y - fitted) ** 2))),
        )
    counts = counts_of(out / "batches.jsonl", names)
    cycles = cyclic.majority_cycles(counts)
    games = counts.sum(-1)
    score = (counts[..., 0] + 0.5 * counts[..., 1]) / games.clamp_min(1.0)
    result = dict(
        games=int(last["games"]),
        slots=[
            dict(
                slot=name,
                elo=float(elo[i]),
                stderr=float(stderr[i]),
                against_baseline=float(relative[i]),
            )
            for i, name in enumerate(names)
        ],
        models=models,
        cycles=[dict(cycle=[names[a], names[b], names[c]], weakest=w) for a, b, c, w in cycles],
        pairs=[
            dict(first=names[i], second=names[j], games=int(games[i, j]), score=float(score[i, j]))
            for i in range(len(names))
            for j in range(i + 1, len(names))
            if games[i, j] > 0
        ],
    )
    (out / "report.json").write_text(json.dumps(result, indent=1) + "\n")
    lines = [f"# {spec.get('name', 'case study')}: {result['games']} games\n\n"]
    lines.append("| Slot | Elo against its airframe's baseline |\n|---|---|\n")
    lines += [
        f"| {s['slot']} | {s['against_baseline']:+.0f} ± {s['stderr']:.0f} |\n"
        for s in result["slots"]
    ]
    for scope, model in models.items():
        lines.append(
            f"\n**{scope}** (RMS residual {model['rms_residual']:.0f} Elo): "
            + ", ".join(
                f"{q} {v:+.0f} Elo a doubling" for q, v in model["elo_per_doubling"].items()
            )
            + "\n"
        )
    lines.append(f"\nMajority cycles: {len(cycles)}\n")
    lines += [
        f"- {' > '.join(c['cycle'])} > (weakest {c['weakest']:.2f})\n"
        for c in result["cycles"][:20]
    ]
    # The docs checks read every committed page's front matter (`covers`)
    (out / "report.md").write_text("---\ncovers: [bench/case_study.py]\n---\n\n" + "".join(lines))
    print("".join(lines))


def main() -> None:
    """The commands the module describes."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("play")
    one.add_argument("--spec", required=True)
    one.add_argument("--out", required=True)
    one.add_argument("--hours", type=float, default=1.0)
    two = sub.add_parser("report")
    two.add_argument("--spec", required=True)
    two.add_argument("--out", required=True)
    args = parser.parse_args()
    {"play": play, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
