"""Export a W&B run's config, summary, history, and files for in-repo archival.

Usage:
    uv run --no-sync scripts/export_wandb_run.py \
        --run vizia128/boost-and-broadside/chpl40cj \
        --out checkpoints/resilient-resonance-682/wandb_export
"""

import argparse
import json
from pathlib import Path

import wandb


def _json_default(obj):
    # wandb summary can hold numpy scalars / nested media refs.
    return str(obj)


def export_run(run_path: str, out_dir: Path, samples: int | None = 2000) -> None:
    api = wandb.Api()
    run = api.run(run_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Metadata: config (hyperparameters) + final summary.
    (out_dir / "config.json").write_text(
        json.dumps(dict(run.config), indent=2, default=_json_default)
    )
    (out_dir / "summary.json").write_text(
        json.dumps(dict(run.summary), indent=2, default=_json_default)
    )
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "id": run.id,
                "name": run.name,
                "path": "/".join(run.path),
                "state": run.state,
                "created_at": str(run.created_at),
                "url": run.url,
                "tags": list(run.tags),
            },
            indent=2,
            default=_json_default,
        )
    )

    # Scalar history -> newline-delimited JSON. A full unsampled scan_history()
    # over this run's ~300 metric keys is impractically slow/large (hours, GB),
    # and wandb.ai keeps the full-fidelity copy; sampled history captures the
    # training curves at ~`samples` points per key in seconds. Pass samples=None
    # for the full unsampled dump.
    history = (
        run.scan_history()
        if samples is None
        else run.history(samples=samples, pandas=False)
    )
    with (out_dir / "history.jsonl").open("w") as fh:
        for row in history:
            fh.write(json.dumps(row, default=_json_default) + "\n")

    # Files logged to the run (media, etc.).
    files_dir = out_dir / "files"
    for f in run.files():
        f.download(root=str(files_dir), exist_ok=True)

    print(f"Exported {run.name} ({run.id}) -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="entity/project/run_id")
    p.add_argument("--out", required=True, type=Path, help="output directory")
    p.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="points per key for sampled history; use 0 for full unsampled scan",
    )
    args = p.parse_args()
    export_run(args.run, args.out, samples=args.samples or None)


if __name__ == "__main__":
    main()
