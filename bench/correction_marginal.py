"""The error head's knees and starting marginal (`bnb.models.yemong.correction.KNEES` and
`MARGINAL`), measured on a short CPU run of the arm at initialisation: the belief of a hidden
ship is then the physics head's open-loop forecast (the head starts at its marginal, the
error head at zero error), and e = truth(t + 1) - forecast(t + 1) is what the error head must
learn first.

The run's config is a production run's config.json (the arm-2 BC run's by default) with
`yemong.delta_split` on (and `--baseline` as the delta baseline), scaled to the CPU. Per
update it collects, prepares the labels and reads e on the correction mask (hidden enemies
alive at t and t + 1); nothing is trained. Per component: the median |e| (position and
velocity pooled over the three axes: one knee per quantity), the share exactly 0 and 16
quantiles of the rest, the max against the code's bound, and the medians by hidden age.
Prints the KNEES and MARGINAL tables to paste and writes the numbers as JSON.

Run (CPU only, capped):
    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 nice -n 15 \
        env CUDA_VISIBLE_DEVICES="" uv run --no-sync python bench/correction_marginal.py \
        --out /tmp/bnb-scratch-arms34/correction_marginal.json
"""

import argparse
import dataclasses
import json
import pathlib

import torch

from bnb.learn import labels
from bnb.models.yemong import correction
from bnb.runs.serialise import from_json
from bnb.runtime.lanes import LaneConfig
from bnb.runtime.train import TrainConfig, Trainer

QUANTILES = 16
AGES = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf"))  # s


def main() -> None:
    """Collect, measure and print."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/vizia/avesnova/runs-v2/2026-11-08-bc-arm2-persistence/config.json",
    )
    parser.add_argument("--baseline", default="persistence")
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", default="/tmp/bnb-scratch-arms34/correction_marginal.json")
    args = parser.parse_args()
    torch.set_num_threads(8)
    config, _ = from_json(TrainConfig, json.loads(pathlib.Path(args.config).read_text()))
    yemong = dataclasses.replace(config.yemong, delta_split=True, delta_baseline=args.baseline)
    config = dataclasses.replace(
        config,
        environments=args.environments,
        length=args.length,
        shards=1,
        lookahead=args.length,
        lanes=LaneConfig(environments=0),
        precision="float32",
        host_store=False,
        compiled=False,
        seed=args.seed,
        yemong=yemong,
        learner=dataclasses.replace(config.learner, micro_batch_rows=4096),
    )
    torch.manual_seed(args.seed)
    trainer = Trainer(config, torch.device("cpu"), diagnostics=False)
    model, learner = trainer.model, trainer.ppo
    errors, ages = [], []
    for update in range(args.updates):
        store, bootstrap = trainer.collect()
        learner.prepare(store, bootstrap)
        views = torch.arange(store.views)
        batches = list(views.split(max(1, 4096 // config.game.num_ships)))
        with torch.no_grad():
            for trajectory, shot_labels in learner.staged(store, batches):
                label = learner.labels(trajectory, shot_labels)
                unrolled = learner.unroll(trajectory)
                _, _, error = labels.correction_targets(
                    model, trajectory, unrolled.next_state_logits
                )
                mask = label["correction_mask"]
                own = trajectory.log_probability.shape[-1]
                age = trajectory.replay.age.float() * model.tick_length
                age = torch.cat([age.new_zeros(age.shape[:-1] + (own,)), age], -1)
                errors.append(error[mask].double())
                ages.append(age[mask].double())
        print(update, sum(e.shape[0] for e in errors), flush=True)
    error, age = torch.cat(errors), torch.cat(ages)
    code = model.correction_code
    report = dict(rows=error.shape[0], components={}, knees={}, ages={})
    table = {}
    levels = (torch.arange(QUANTILES, dtype=torch.float64) + 0.5) / QUANTILES
    for i, component in enumerate(code.components):
        values = error[:, i]
        zero = values == 0
        rest = values[~zero]
        if rest.numel():
            quantiles = torch.quantile(rest[torch.randperm(rest.numel())[:4_000_000]], levels)
        else:  # every error exactly 0
            quantiles = torch.zeros_like(levels)
        table[component.name] = (
            round(zero.double().mean().item(), 4),
            tuple(float(f"{q:.4g}") for q in quantiles),
        )
        magnitude = values.abs()
        report["components"][component.name] = dict(
            zero_share=zero.double().mean().item(),
            median=magnitude.median().item(),
            p99=torch.quantile(magnitude[:4_000_000], 0.99).item(),
            max=magnitude.max().item(),
            bound=component.bound,
            beyond_bound=(magnitude > component.bound).double().mean().item(),
        )
    for name, columns in [
        ("position", slice(0, 3)),
        ("velocity", slice(3, 6)),
        *((n, slice(6 + k, 7 + k)) for k, n in enumerate(correction.SCALAR_QUANTITIES)),
    ]:
        magnitude = error[:, columns].abs().flatten()
        nonzero = magnitude[magnitude > 1e-6]
        report["knees"][name] = dict(
            median=magnitude.median().item(),
            median_above_noise=nonzero.median().item() if nonzero.numel() else None,
            zero_share=(magnitude <= 1e-6).double().mean().item(),
        )
        by_age = {}
        for low, high in zip(AGES[:-1], AGES[1:], strict=True):
            members = (age >= low) & (age < high)
            if members.any():
                part = error[members][:, columns].abs()
                by_age[f"{low}-{high}"] = dict(rows=int(members.sum()), median=part.median().item())
        report["ages"][name] = by_age
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("MARGINAL = {")
    for name, (share, quantiles) in table.items():
        print(f'    "{name}": ({share}, ({", ".join(repr(q) for q in quantiles)})),')
    print("}")


if __name__ == "__main__":
    main()
