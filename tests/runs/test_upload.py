"""Uploads (runs spec 6, acceptance 11): a chosen checkpoint as a W&B artifact, in W&B's
offline mode only (no test touches the network)."""

import json
import sys

import pytest
import runs_cases as cases
import torch

from bnb import cli
from bnb.runs import upload
from bnb.runs.records import Run

pytest.importorskip("wandb")


def test_a_checkpoint_uploads_as_an_artifact_named_for_the_run_and_step(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    monkeypatch.setenv("WANDB_SILENT", "true")
    (tmp_path / "wandb").mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bnb",
            "train",
            *cases.TRAIN_OPTIONS,
            "--updates",
            "1",
            "--name",
            "a",
            "--runs-directory",
            str(tmp_path),
        ],
    )
    with torch.random.fork_rng(devices=[]):
        cli.main()
    run = Run.open(tmp_path, "a")
    step = run.checkpoint_steps()[-1]
    message = upload.upload(run, step)
    assert message.endswith(f"as the W&B artifact a-step_{step} (checkpoint)")
    (records,) = (tmp_path / "wandb").rglob("offline-run-*/run-*.wandb")  # one offline run
    assert f"a-step_{step}".encode() in records.read_bytes()  # its artifact, logged locally
    with pytest.raises(FileNotFoundError, match="keeps no checkpoint at step 7"):
        upload.upload(run, 7)
    with pytest.raises(FileNotFoundError, match="has no league snapshots"):
        upload.upload(run, league=True)
    assert json.loads((run.directory / "run.json").read_text())["status"] == "finished"


def test_a_run_streams_its_records_to_wandb(monkeypatch, tmp_path):
    """`bnb train --wandb` with the real W&B library (offline): the run gets a W&B id, and
    the update's series reach the W&B run's records."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    monkeypatch.setenv("WANDB_SILENT", "true")
    (tmp_path / "wandb").mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bnb",
            "train",
            *cases.TRAIN_OPTIONS,
            "--updates",
            "1",
            "--name",
            "a",
            "--wandb",
            "--runs-directory",
            str(tmp_path),
        ],
    )
    with torch.random.fork_rng(devices=[]):
        cli.main()
    run = Run.open(tmp_path, "a")
    assert json.loads((run.directory / "run.json").read_text())["wandb_id"]
    (records,) = run.directory.rglob("offline-run-*/run-*.wandb")  # in the run's directory
    assert b"policy/kl" in records.read_bytes()
