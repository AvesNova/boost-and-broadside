"""The persistence delta code's starting marginal (`bnb.models.yemong.delta.MARGINAL`) and a
check of its knees and bounds, from segments recorded by `bench/state_deltas_collect.py`
on this tree (scripted self-play).

Per component of `delta.COMPONENTS`, over the one-tick pairs alive at t and t + 1 with no
reset between: the share exactly 0, 16 quantiles of the rest, the median |change|, the
p99.9 and max |change| against the code's bound, the share clamped to the padded range, and
the first cross-entropy of the marginal bias against the targets' empirical marginal
entropy. Prints the MARGINAL table to paste and writes the numbers as JSON.

Run (CPU only, capped):
    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 nice -n 15 \
        env CUDA_VISIBLE_DEVICES="" uv run --no-sync python bench/persistence_marginal.py \
        --segments /tmp/bnb-scratch-arm2/segments
"""

import argparse
import json
import pathlib

import numpy as np
import torch

from bnb.contracts.obs import SHIP
from bnb.game import api
from bnb.models.yemong import codes, delta
from bnb.models.yemong.inputs import truth
from bnb.models.yemong.state_code import state_code_ranges

QUANTILES = 16


def pairs(path: pathlib.Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(means at t, means at t + 1, ship type) of the living one-tick pairs of a segment."""
    data = np.load(path)
    rows = torch.from_numpy(data["rows"]).double()  # (L, B, N, C)
    done = torch.from_numpy(data["done"])
    offset = torch.zeros(rows.shape[1], 2, dtype=rows.dtype)
    means = truth(rows, offset[None])
    alive = rows[..., SHIP["alive"]] > 0.5
    keep = alive[:-1] & alive[1:] & ~done[:-1, :, None]
    ship_type = rows[..., SHIP["ship_type"]].long()
    return means[:-1][keep], means[1:][keep], ship_type[:-1][keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", default="/tmp/bnb-scratch-arm2/segments")
    parser.add_argument("--out", default="/tmp/bnb-scratch-arm2/marginal.json")
    args = parser.parse_args()
    torch.set_num_threads(8)
    rules, types = api.rules_of(api.PRESETS["frontline"]), api.ship_types()
    code = delta.DeltaCode(state_code_ranges(rules, types)).double()
    changes, slows = [], []
    for path in sorted(pathlib.Path(args.segments).glob("*/*/segment_*.npz")):
        now, after, kind = pairs(path)
        change, slow = code.changes(now, after, kind)
        changes.append(change)
        slows.append(slow)
        print(path, change.shape[0], flush=True)
    change, slow = torch.cat(changes), torch.cat(slows)
    report = dict(pairs=change.shape[0], slow_share=slow.double().mean().item(), components={})
    table = {}
    for i, component in enumerate(delta.COMPONENTS):
        values = change[:, i]
        zero = values == 0
        rest = values[~zero]
        levels = (torch.arange(QUANTILES, dtype=torch.float64) + 0.5) / QUANTILES
        quantiles = torch.quantile(rest[torch.randperm(rest.numel())[:4_000_000]], levels)
        magnitude = values.abs()
        coordinate = codes.symlog_coordinate(values, component.knee, component.spacing)
        _, above = codes.symlog_sides(component.bins, component.below)
        clipped = (coordinate < -component.low) | (coordinate > above + codes.PADDING_BINS)
        table[component.name] = (
            round(zero.double().mean().item(), 4),
            [float(f"{q:.4g}") for q in quantiles],
        )
        report["components"][component.name] = dict(
            zero_share=zero.double().mean().item(),
            median=magnitude.median().item(),
            median_above_noise=magnitude[magnitude > 1e-6].median().item(),
            p999=torch.quantile(
                magnitude[torch.randperm(magnitude.numel())[:4_000_000]], 0.999
            ).item(),
            max=magnitude.max().item(),
            bound=component.bound,
            knee=component.knee,
            clipped=clipped.double().mean().item(),
        )
    # the first cross-entropy of the marginal bias as measured, against the empirical one
    delta.MARGINAL.update({name: (share, tuple(q)) for name, (share, q) in table.items()})
    logits = code.marginal_logits().double()
    sample = change[torch.randperm(change.shape[0])[:500_000]]
    target, _ = code.encode(sample)
    first = codes.grouped_cross_entropy(logits.expand_as(target), target, code.runs).mean(0)
    empirical = torch.log(target.mean(0).clamp_min(1e-300)).expand_as(target)
    best = codes.grouped_cross_entropy(empirical, target, code.runs).mean(0)
    for i, component in enumerate(delta.COMPONENTS):
        report["components"][component.name].update(
            first_cross_entropy=first[i].item(), marginal_entropy=best[i].item()
        )
    report["first_cross_entropy"], report["marginal_entropy"] = (
        first.sum().item(),
        best.sum().item(),
    )
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("MARGINAL = {")
    for name, (share, quantiles) in table.items():
        print(f'    "{name}": ({share}, ({", ".join(repr(q) for q in quantiles)})),')
    print("}")


if __name__ == "__main__":
    main()
