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

A run downloaded before the artifact store existed is already on disk as a plain
directory. ``--from-directory`` promotes exactly those bytes into an artifact
without contacting W&B, so an export that predates the store is citable on the
same terms as one taken today:

    uv run scripts/export_wandb_run.py \
        --from-directory checkpoints/resilient-resonance-682/wandb_export \
        --run-name resilient-resonance-682
"""

import argparse
import json
import shutil
from pathlib import Path

from boost_and_broadside.artifacts import ArtifactRecipe, ArtifactStore, Invocation

_SCHEMA_VERSION = 1
# The payload a W&B export is made of. `files/` is copied wholesale beside them.
_DOCUMENTS = ("config.json", "summary.json", "run_meta.json")
_HISTORY = "history.jsonl"


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

    import wandb

    api = wandb.Api()
    run = api.run(run_path)

    store = ArtifactStore(
        checkpoint_root=checkpoint_root,
        standalone_root=standalone_root,
        invocation=Invocation(
            argv=("scripts/export_wandb_run.py", "--run", run_path),
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


def ingest_export_directory(
    directory: str | Path,
    *,
    run_name: str | None = None,
    checkpoint_root: str | Path = "checkpoints",
    standalone_root: str | Path = "artifacts",
) -> Path:
    """Promote an already-downloaded export directory into a stored artifact.

    Nothing is fetched and nothing is recomputed: the stored bytes are copied
    into a managed artifact directory and hashed there, so what publication
    renders is provably the same export that was downloaded. The number of
    history points W&B was asked for is not recoverable from the files, so it is
    recorded as unknown rather than assumed to be this script's default.
    """

    source = Path(directory)
    for name in (*_DOCUMENTS, _HISTORY):
        if not (source / name).is_file():
            raise FileNotFoundError(f"{source} is not a W&B export: no {name}")
    meta = json.loads((source / "run_meta.json").read_text())

    store = ArtifactStore(
        checkpoint_root=checkpoint_root,
        standalone_root=standalone_root,
        invocation=Invocation(
            argv=("scripts/export_wandb_run.py", "--from-directory", str(source)),
            command="wandb-export",
            execution={"network": False},
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
            subjects={"wandb_run": meta.get("path"), "run": run_name},
            parameters={"samples": None},
            sources={"export_directory": source.as_posix()},
        ),
        owner,
    )

    for name in _DOCUMENTS:
        artifact.write_json(json.loads((source / name).read_text()), name)
    shutil.copyfile(source / _HISTORY, artifact.path / _HISTORY)
    artifact.attach(_HISTORY)

    files = source / "files"
    if files.is_dir():
        shutil.copytree(files, artifact.path / "files")
        for path in sorted((artifact.path / "files").rglob("*")):
            if path.is_file():
                artifact.attach(str(path.relative_to(artifact.path)))

    artifact.complete()
    print(f"Ingested {source} -> {artifact.path}")
    return artifact.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subject = parser.add_mutually_exclusive_group(required=True)
    subject.add_argument("--run", help="entity/project/run_id")
    subject.add_argument(
        "--from-directory",
        help="An export already on disk; ingested offline, with no W&B call.",
    )
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
    if args.from_directory is not None:
        ingest_export_directory(args.from_directory, run_name=args.run_name)
        return
    export_run(args.run, run_name=args.run_name, samples=args.samples or None)


if __name__ == "__main__":
    main()
