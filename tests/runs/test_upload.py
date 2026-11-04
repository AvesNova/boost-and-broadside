"""Uploads (runs spec 6, acceptance 11): a chosen checkpoint as a W&B artifact; the W&B
stream, its views saved at the start, and its telemetry shut down once. In W&B's offline
mode only (no test touches the network)."""

import json
import os
import subprocess
import sys
import warnings

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


def test_a_run_streams_its_records_to_wandb(monkeypatch, tmp_path, capsys):
    """`bnb train --wandb` with the real W&B library (offline): the run gets a W&B id, and
    the update's series reach the W&B run's records. An offline run saves no views."""
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
    assert "offline W&B run: no dashboards saved" in capsys.readouterr().out


def _wandb_run(monkeypatch, tmp_path, *options: str) -> Run:
    """`bnb train --wandb` (offline), one update."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    monkeypatch.setenv("WANDB_SILENT", "true")
    (tmp_path / "wandb").mkdir()
    arguments = [*cases.TRAIN_OPTIONS, "--updates", "1", "--name", "a", "--wandb", *options]
    monkeypatch.setattr(
        sys, "argv", ["bnb", "train", *arguments, "--runs-directory", str(tmp_path)]
    )
    with torch.random.fork_rng(devices=[]):
        cli.main()
    return Run.open(tmp_path, "a")


@pytest.mark.parametrize("outcome", ["saved", "failed", "no_tools"])
def test_a_wandb_run_saves_its_views_once_its_first_record_is_logged(
    monkeypatch, tmp_path, capsys, outcome
):
    """Online, the run saves its headline and curated views after its first record and prints
    both links; a failure, or `wandb-workspaces` missing, warns once and the run goes on.
    (The W&B run is offline, seen as online here; the save itself is replaced, so nothing
    reaches the server.)"""
    from wandb.sdk import wandb_run

    from bnb.runs import sinks
    from bnb.runtime import dashboards

    saves = []

    def save(name, entity, project, objective):
        saves.append((name, project, objective))
        if outcome == "failed":
            raise RuntimeError("the server said no")
        if outcome == "no_tools":
            raise ModuleNotFoundError("No module named 'wandb_workspaces'", name="wandb_workspaces")
        return {view: f"https://app.example/w/e/p?nw={view}" for view in ("headline", "curated")}

    monkeypatch.setattr(wandb_run.Run, "offline", property(lambda self: False))
    monkeypatch.setattr(dashboards, "save", save)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = _wandb_run(monkeypatch, tmp_path)
    assert run.record["status"] == "finished"
    assert saves == [("a", sinks.PROJECT, "ppo")]  # once, after the first record
    out = capsys.readouterr().out
    shown = [w for w in caught if issubclass(w.category, sinks.DashboardWarning)]
    if outcome == "saved":
        assert "dashboard headline: https://app.example/w/e/p?nw=headline" in out
        assert "dashboard curated: https://app.example/w/e/p?nw=curated" in out
        assert not shown
    else:
        (warning,) = shown
        assert "dashboard headline" not in out
        expected = "the tools extra" if outcome == "no_tools" else "the server said no"
        assert expected in str(warning.message)


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


def test_bnb_runs_dashboard_prints_the_working_links(monkeypatch, tmp_path, capsys):
    """`bnb runs dashboard RUN` saves the views for the run's objective and prints the
    links `bnb.runtime.dashboards.save` returns (under the app's path prefix,
    `test_a_views_link_keeps_the_apps_path_prefix`)."""
    from bnb.runtime import dashboards

    run = _wandb_run(monkeypatch, tmp_path)
    saves = []

    def save(name, entity, project, objective):
        saves.append((name, entity, objective))
        return {"headline": "https://app.example/w/e/p?nw=h", "curated": "https://c"}

    monkeypatch.setattr(dashboards, "save", save)
    monkeypatch.setenv("WANDB_ENTITY", "someone")
    capsys.readouterr()
    monkeypatch.setattr(
        sys, "argv", ["bnb", "runs", "dashboard", "a", "--runs-directory", str(tmp_path)]
    )
    cli.main()
    assert saves == [(run.record["name"], "someone", "ppo")]
    assert capsys.readouterr().out.splitlines() == [
        "headline: https://app.example/w/e/p?nw=h",
        "curated: https://c",
    ]
