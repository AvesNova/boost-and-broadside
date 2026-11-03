"""Explicit uploads (runs spec 6): a chosen checkpoint, or the league's snapshots, as a W&B
artifact named for the run and the step. Nothing uploads automatically; W&B is the only
target for now. W&B's own settings (the entity, `WANDB_MODE=offline`) come from its
environment variables."""

import pathlib

from .records import Run

PROJECT = "boost-and-broadside-v2"  # the diagnostics spec's project (author, 2026-11-02)


def artifact_name(run: Run, step: int | None, league: bool) -> str:
    """`<run>-league`, or `<run>-step_<step>`."""
    return f"{run.record['name']}-league" if league else f"{run.record['name']}-step_{step}"


def upload(run: Run, step: int | None = None, league: bool = False) -> str:
    """Upload the run's checkpoint at `step` (the newest by default), or with `league` its
    league directory, as a W&B artifact; returns what was uploaded."""
    import wandb

    if league:
        directory, kind, step = run.directory / "league", "league", None
        if not any(directory.iterdir()):
            raise FileNotFoundError(f"run {run.record['name']} has no league snapshots")
    else:
        directory, kind = run.checkpoint_at(step), "checkpoint"
        step = int(directory.name[len("step_") :])
    name = artifact_name(run, step, league)
    session = wandb.init(
        project=PROJECT,
        job_type="upload",
        name=f"upload {name}",
        config=dict(run=run.record["name"], step=step),
    )
    try:
        artifact = wandb.Artifact(name, type=kind, metadata=dict(run=run.record["name"], step=step))
        artifact.add_dir(str(pathlib.Path(directory)))
        session.log_artifact(artifact)
    finally:
        session.finish()
    return f"uploaded {directory} as the W&B artifact {name} ({kind})"
