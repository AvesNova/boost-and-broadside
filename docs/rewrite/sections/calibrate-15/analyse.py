"""Queue item 15's analysis of calibrate.py's collections (team1, team4, team16 in DATA):
quantiles, the data-rule knees of the next-state columns (symlog_preview.data_knee), the bin
occupancy of every categorical support (the current one and a proposed one), the hit-class
shares and the scaled features' ranges.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/calibrate-15/analyse.py DATA

Occupancy: each target to its nearest centre in the support's bin space (lean.learn.returns'
usage rule; beyond the outermost centres into the end bins), or for an angle the circle's bin
it falls in. Reported: bins holding 99% of the mass (largest first), the largest bin's share,
bins under 0.1% ("empty"), the perplexity (effective bins), the share beyond the outermost
centres (clamped).
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from lean.contract import SHIP_SCALARS
from lean.learn.config import LearnConfig
from lean.model.geometry import GeometryColumn, circular_centres
from lean.model.policy import ModelConfig
from lean.model.value import buffered_centres, symlog

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "symlog_preview", HERE.parent / "scoreboard" / "symlog_preview.py"
)
symlog_preview = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(symlog_preview)

SHIP_SCALAR_NAMES = (
    "velocity x (symlog /200)",
    "velocity y (symlog /200)",
    "speed (symlog /200)",
    "nose rate (symlog /1)",
    "angle of attack (/0.25)",
    "angle cos",
    "angle sin",
    "store fraction",
    "shield fraction",
    "shield (/100)",
    "shield wait (/3, clip)",
    "gun cooldown (ticks)",
    "inside zone edge (/1000, clip)",
    "outward x",
    "outward y",
    "heading cos",
    "heading sin",
    "bullet share",
)
GLOBAL_NAMES = (
    "zone x",
    "zone y",
    "zone radius",
    "next zone x",
    "next zone y",
    "next zone radius",
    "stage time left",
    "own alive",
    "enemy alive",
)
FIELD_NAMES = ("field radius (/1000)", "field band (/40)", "field level (/2)")


def occupancy(values: np.ndarray, column: GeometryColumn) -> dict:
    x = torch.from_numpy(values).float()
    if column.circular:
        centres = circular_centres(column.knee, column.bins)
        width = float(centres[1] - centres[0])
        s = symlog(torch.atan2(torch.sin(x), torch.cos(x)), column.knee)
        index = torch.floor((s - (centres[0] - width / 2)) / width).long() % column.bins
        clamped = 0.0
    else:
        centres = buffered_centres(column.low, column.high, column.knee, column.bins)
        width = float(centres[1] - centres[0])
        s = symlog(x, column.knee)
        index = ((s - centres[0]) / width).round().clamp(0, column.bins - 1).long()
        clamped = float(((s < centres[0]) | (s > centres[-1])).float().mean())
    counts = torch.bincount(index, minlength=column.bins).double()
    share = (counts / counts.sum()).numpy()
    ordered = np.cumsum(np.sort(share)[::-1])
    positive = share[share > 0]
    return {
        "bins": column.bins,
        "hold_99": int(np.sum(ordered < 0.99)) + 1,
        "largest": float(share.max()),
        "empty": int(np.sum(share < 0.001)),
        "perplexity": float(math.exp(-np.sum(positive * np.log(positive)))),
        "clamped": clamped,
    }


def occupancy_line(o: dict) -> str:
    return (
        f"99% in {o['hold_99']}/{o['bins']}, largest {o['largest']:.1%}, empty {o['empty']}, "
        f"perplexity {o['perplexity']:.1f}, clamped {o['clamped']:.2%}"
    )


def q(stats: dict, key: str, *shares: str) -> str:
    entry = stats["quantiles"].get(key, {})
    return " / ".join(f"{entry.get(share, float('nan')):.4g}" for share in shares)


def main() -> None:
    data = Path(sys.argv[1])
    proposal = (
        json.loads((HERE / "proposal.json").read_text())
        if (HERE / "proposal.json").exists()
        else {}
    )
    sizes = [n for n in (4, 1, 16) if (data / f"team{n}.json").exists()]
    stats = {n: json.loads((data / f"team{n}.json").read_text()) for n in sizes}
    arrays = {n: np.load(data / f"team{n}.npz") for n in sizes}
    for n in sizes:
        s = stats[n]
        lengths = " / ".join(f"{s['episode_ticks'][k]:.0f}" for k in ("0.5", "0.99", "1.0"))
        print(
            f"{n}v{n}: {s['envs']} environments x {s['ticks']} ticks, {s['episodes']} episodes, "
            f"{s['seconds']:.0f} s; episode ticks median / p99 / max {lengths}"
        )
    keys = ("0.0", "0.001", "0.01", "0.5", "0.99", "0.999", "1.0")
    learn = LearnConfig()
    print("\n## Value returns (min / 0.1% / 1% / median / 99% / 99.9% / max)")
    for level in learn.level_names:
        full = ("damage", "kill_death", "objective_progress", "objective", "outcome").index(level)
        current = GeometryColumn(
            level, 63, learn.value_knee[full], learn.value_low[full], learn.value_high[full]
        )
        new = proposal.get(f"value_{level}")
        for n in sizes:
            print(
                f"- {level} {n}v{n} ({stats[n]['counts'][f'value_{level}']} rows): "
                f"{q(stats[n], f'value_{level}', *keys)}"
            )
            values = arrays[n][f"value_{level}"]
            print(
                f"    current {current.low}..{current.high} knee {current.knee}: "
                f"{occupancy_line(occupancy(values, current))}"
            )
            if new:
                column = GeometryColumn(level, 63, new[2], new[0], new[1])
                print(
                    f"    proposed {column.low}..{column.high} knee {column.knee}: "
                    f"{occupancy_line(occupancy(values, column))}"
                )

    model = ModelConfig()
    print("\n## Next-state targets")
    for spec in model.next_state_columns:
        key = f"next_{spec.name}"
        for n in sizes:
            print(f"- {spec.name} {n}v{n}: {q(stats[n], key, *keys)}")
        values = arrays[4][key]
        knee = symlog_preview.data_knee(values.astype(np.float64))
        print(
            f"    data-rule knee (4v4): {knee['knee']} (every side {knee['knee_every_side']}), "
            f"exact zeros {float(np.mean(values == 0)):.3f}"
        )
        current = GeometryColumn(spec.name, 63, spec.knee, spec.low, spec.high)
        print(
            f"    current {spec.low}..{spec.high} knee {spec.knee}: "
            f"{occupancy_line(occupancy(values, current))}"
        )
        new = proposal.get(key)
        if new:
            column = GeometryColumn(spec.name, 63, new[2], new[0], new[1])
            print(
                f"    proposed {column.low}..{column.high} knee {column.knee}: "
                f"{occupancy_line(occupancy(values, column))}"
            )

    print("\n## Geometry targets")
    columns = list(model.pair_columns) + [model.field_column, model.edge_column]
    for column in columns:
        key = column.name if column.name in ("field_edge", "map_edge") else f"pair_{column.name}"
        for n in sizes:
            print(
                f"- {column.name} {n}v{n} ({stats[n]['counts'][key]}): "
                f"{q(stats[n], key, *keys)}; "
                f"current: {occupancy_line(occupancy(arrays[n][key], column))}"
            )
        new = proposal.get(key)
        if new:
            proposed = GeometryColumn(
                column.name,
                new.get("bins", column.bins),
                new["knee"],
                new.get("low", 0.0),
                new.get("high", 0.0),
                column.circular,
                column.both_orders,
            )
            for n in sizes:
                print(
                    f"    proposed {new} {n}v{n}: "
                    f"{occupancy_line(occupancy(arrays[n][key], proposed))}"
                )

    print("\n## Hit outcomes (missed / enemy / team-mate)")
    for n in sizes:
        counts = np.array(stats[n]["hit_counts"], dtype=np.float64)
        shares = counts / counts.sum()
        print(
            f"- {n}v{n}: {int(counts.sum())} bullets: "
            + " / ".join(f"{share:.4f}" for share in shares)
        )

    print("\n## Scaled features (attended rows): 1% / 50% / 99% / max |x|; flags")
    names = [("ship_features", i, SHIP_SCALAR_NAMES[i]) for i in range(SHIP_SCALARS)]
    names += [("globals", i, GLOBAL_NAMES[i]) for i in range(len(GLOBAL_NAMES))]
    names += [("field_features", i, FIELD_NAMES[i]) for i in range(len(FIELD_NAMES))]
    for prefix, i, name in names:
        for n in sizes:
            values = np.abs(arrays[n][f"{prefix}_{i}"])
            p1, p50, p99, top = np.quantile(values, [0.01, 0.5, 0.99, 1.0])
            flag = ""
            if p50 > 10 or (0 < p99 < 0.01):
                flag = "  <-- off"
            print(f"- {name} {n}v{n}: {p1:.3g} / {p50:.3g} / {p99:.3g} / {top:.3g}{flag}")
    for n in sizes:
        values = arrays[n]["since_seen"]
        print(
            f"- since_seen {n}v{n}: "
            + " / ".join(f"{v:.3g}" for v in np.quantile(values, [0.01, 0.5, 0.99, 1.0]))
        )


if __name__ == "__main__":
    main()
