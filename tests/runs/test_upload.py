"""Uploads (runs spec 6, acceptance 11): a chosen checkpoint as a W&B artifact; the W&B
stream and its telemetry shut down once. In W&B's offline mode only (no test touches the
network)."""

import json
import os
import subprocess
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


TELEMETRY = """
import sys
from wandb.analytics.opentelemetry import opentelemetry_proxy
from bnb.runs.sinks import shutdown_telemetry
proxy = opentelemetry_proxy.get_open_telemetry_proxy()
proxy._providers()  # what an online run's telemetry builds on its first record
if sys.argv[1] == "shut":
    shutdown_telemetry()
"""


@pytest.mark.parametrize("shut", [False, True])
def test_the_telemetry_is_shut_down_once(tmp_path, shut):
    """At the end of an online run W&B's telemetry shut OpenTelemetry's meter provider down
    at exit, and the provider's own exit handler shut it down again ("shutdown can only be
    called once"). `shutdown_telemetry`, which the W&B sink calls after finishing the run,
    shuts it down before exit, once. The base URL is a closed local port: nothing leaves
    the machine."""
    environment = dict(
        os.environ,
        WANDB_MODE="online",
        WANDB_BASE_URL="http://127.0.0.1:9",
        WANDB_DIR=str(tmp_path),
    )
    done = subprocess.run(
        [sys.executable, "-c", TELEMETRY, "shut" if shut else "leave"],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert ("shutdown can only be called once" in done.stderr) is not shut, done.stderr[-2000:]
