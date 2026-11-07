"""Measure the acceleration inputs' knees and padding shares on the recorded segments
(bench/state_deltas_collect.py; docs/rewrite/sections/acceleration-inputs-brief.md).

Per player and preset, from the privileged truth rows: every one-tick pair (t - 1, t) with
the ship alive at both, no reset between them and no spawn at t (no respawns), the
acceleration along and across the velocity at t (the heading below 1 m/s) and the nose
rate's, per second. Prints and writes the median |value| per component (per player and
preset, and pooled at equal weight over a subsample of each), the share below 1 m/s, and
the share of samples past the last real bin's centre (the padding) and the code's mean
padding mass for the chosen knees and ranges (`bnb.models.yemong.inputs.ACCELERATIONS`).

Run (CPU only, capped):
    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 nice -n 15 \
        env CUDA_VISIBLE_DEVICES="" uv run --no-sync python bench/acceleration_inputs_measure.py
"""

import argparse
import json
import pathlib

import numpy as np
import torch

from bnb.models.yemong import codes
from bnb.models.yemong.inputs import ACCELERATIONS, SLOW_SPEED

SEGMENTS = pathlib.Path("/tmp/bnb-scratch-deltas/segments")
OUTPUT = pathlib.Path("/tmp/bnb-scratch-acceleration/measure.json")
SUBSAMPLE = 400_000  # per player and preset, for the pooled median


def segment_samples(path: pathlib.Path, columns: list[str], dt: float) -> tuple[np.ndarray, int]:
    """((M, 3) along, across, angular of the valid pairs; the slow-frame count)."""
    data = np.load(path)
    rows = data["rows"].astype(np.float64)
    C = {name: i for i, name in enumerate(columns)}
    col = lambda name: rows[..., C[name]]  # noqa: E731
    alive = col("alive") > 0.5
    spawned = col("spawned") > 0.5
    done = data["done"]
    pair = alive[1:] & alive[:-1] & ~spawned[1:] & ~done[:-1, :, None]
    vx, vy, omega = col("velocity_x"), col("velocity_y"), col("nose_rate")
    dvx, dvy = (vx[1:] - vx[:-1]) / dt, (vy[1:] - vy[:-1]) / dt
    speed = np.hypot(vx[1:], vy[1:])
    slow = speed < SLOW_SPEED
    safe = np.where(slow, 1.0, speed)
    ux = np.where(slow, col("heading_cos")[1:], vx[1:] / safe)
    uy = np.where(slow, col("heading_sin")[1:], vy[1:] / safe)
    along = dvx * ux + dvy * uy
    across = dvy * ux - dvx * uy
    angular = (omega[1:] - omega[:-1]) / dt
    samples = np.stack([along[pair], across[pair], angular[pair]], -1)
    return samples, int(slow[pair].sum())


def padding(values: np.ndarray) -> tuple[list[float], list[float]]:
    """Per component, the share of samples past the last real centre and the code's mean
    padding mass."""
    beyond, mass = [], []
    for i, component in enumerate(ACCELERATIONS):
        x = torch.from_numpy(values[:, i])
        code, _ = codes.symlog_encode(x, component.knee, component.bound, component.bins)
        pad = codes.PADDING_BINS
        beyond.append(float((x.abs() > component.bound).double().mean()))
        mass.append(float((code[:, :pad].sum(-1) + code[:, -pad:].sum(-1)).mean()))
    return beyond, mass


def main() -> None:
    """Measure every player and preset, then the pooled subsample; print and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--segments", type=pathlib.Path, default=SEGMENTS)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    args = parser.parse_args()
    generator = np.random.default_rng(0)
    report, pooled = {}, []
    names = [component.name for component in ACCELERATIONS]
    for meta_path in sorted(args.segments.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text())
        key = f"{meta['player']}/{meta['preset']}"
        parts, slow = [], 0
        for segment in sorted(meta_path.parent.glob("segment_*.npz")):
            samples, slow_count = segment_samples(segment, meta["columns"], meta["tick_length"])
            parts.append(samples)
            slow += slow_count
            print(f"{key} {segment.name}: {len(samples):,} pairs", flush=True)
        values = np.concatenate(parts)
        beyond, mass = padding(values)
        report[key] = dict(
            pairs=len(values),
            slow_frame=slow / len(values),
            median=dict(zip(names, np.median(np.abs(values), 0).tolist(), strict=True)),
            p999=dict(zip(names, np.quantile(np.abs(values), 0.999, 0).tolist(), strict=True)),
            max=dict(zip(names, np.abs(values).max(0).tolist(), strict=True)),
            beyond_bound=dict(zip(names, beyond, strict=True)),
            padding_mass=dict(zip(names, mass, strict=True)),
        )
        print(json.dumps({key: report[key]}, indent=1), flush=True)
        pick = generator.choice(len(values), min(SUBSAMPLE, len(values)), replace=False)
        pooled.append(values[pick])
    values = np.concatenate(pooled)
    beyond, mass = padding(values)
    report["pooled"] = dict(
        pairs=len(values),
        median=dict(zip(names, np.median(np.abs(values), 0).tolist(), strict=True)),
        p999=dict(zip(names, np.quantile(np.abs(values), 0.999, 0).tolist(), strict=True)),
        max=dict(zip(names, np.abs(values).max(0).tolist(), strict=True)),
        beyond_bound=dict(zip(names, beyond, strict=True)),
        padding_mass=dict(zip(names, mass, strict=True)),
    )
    print(json.dumps({"pooled": report["pooled"]}, indent=1), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
