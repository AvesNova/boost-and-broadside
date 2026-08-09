"""Ingest a W&B run's config, summary, history, and files as a stored artifact.

This is the only place in the project that talks to W&B. Publication never does:
it renders from what this script stored, so a canonical figure can be rebuilt on
a machine with no credentials and no network. Ingestion is deliberately a script
rather than a ``bnb`` command — it is a one-off import from another system, not
part of the measurement surface.

Usage:
    uv run scripts/export_wandb_run.py \
        --run vizia128/boost-and-broadside/chpl40cj \
        --run-name resilient-resonance-682
"""

import argparse
import json
from pathlib import Path

import wandb

from boost_and_broadside.artifacts import ArtifactRecipe, ArtifactStore, Invocation

_SCHEMA_VERSION = 1


def _json_default(obj):
    # wandb summary can hold numpy scalars / nested media refs.
    return str(obj)


def _dump(payload) -> str:
    return json.dumps(payload, indent=2, default=_json_default)


def export_run(
    run_path: str,
    *,
    run_name: str | None = None,
    samples: int | None = 2000,
    checkpoint_root: str | Path = "checkpoints",
    standalone_root: str | Path = "artifacts",
) -> Path:
    """Store one W&B run as a ``wandb-export`` artifact and return its directory."""

    api = wandb.Api()
    run = api.run(run_path)

    store = ArtifactStore(
        checkpoint_root=checkpoint_root,
        standalone_root=standalone_root,
        invocation=Invocation(
            argv=("export_wandb_run.py", "--run", run_path),
            command="wandb-export",
            execution={"samples": samples},
        ),
    )
    owner = (
        store.run_owner(run_name)
        if run_name is not None and (Path(checkpoint_root) / run_name).is_dir()
        else store.standalone_owner()
    )
    artifact = store.create(
        ArtifactRecipe(
            artifact_type="wandb-export",
            result_schema_version=_SCHEMA_VERSION,
            subjects={"wandb_run": "/".join(run.path), "run": run_name},
            parameters={"samples": samples},
        ),
        owner,
    )

    artifact.write_json(dict(run.config), "config.json")
    artifact.write_json(dict(run.summary), "summary.json")
    artifact.write_json(
        {
            "id": run.id,
            "name": run.name,
            "path": "/".join(run.path),
            "state": run.state,
            "created_at": str(run.created_at),
            "url": run.url,
            "tags": list(run.tags),
        },
        "run_meta.json",
    )

    # Scalar history -> newline-delimited JSON. A full unsampled scan_history()
    # over this run's ~300 metric keys is impractically slow/large (hours, GB),
    # and wandb.ai keeps the full-fidelity copy; sampled history captures the
    # training curves at ~`samples` points per key in seconds. Pass samples=None
    # for the full unsampled dump.
    history = run.scan_history() if samples is None else run.history(samples=samples, pandas=False)
    (artifact.path / "history.jsonl").write_text(
        "".join(json.dumps(row, default=_json_default) + "\n" for row in history)
    )
    artifact.attach("history.jsonl")

    files_dir = artifact.path / "files"
    for stored in run.files():
        stored.download(root=str(files_dir), exist_ok=True)
    for path in sorted(files_dir.rglob("*")) if files_dir.exists() else []:
        if path.is_file():
            artifact.attach(str(path.relative_to(artifact.path)))

    artifact.complete()
    print(f"Exported {run.name} ({run.id}) -> {artifact.path}")
    return artifact.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="entity/project/run_id")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Checkpoint directory that owns this export; omit for a standalone artifact.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="points per key for sampled history; use 0 for full unsampled scan",
    )
    args = parser.parse_args()
    export_run(args.run, run_name=args.run_name, samples=args.samples or None)


if __name__ == "__main__":
    main()
