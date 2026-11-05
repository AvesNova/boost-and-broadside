"""The media of a training run (`bnb figures video`, `bnb figures watch`): a policy trained
on a mixed batch plays every mode of the mix, the video's tally, the watcher's polling of a
run's checkpoints, and its companion W&B run (in W&B's offline mode only: no test touches
the network). The video itself needs an OpenGL context: its test is a gpu test."""

import dataclasses
import json
import pathlib

import numpy as np
import pytest
import torch
from play_cases import tiny_policy

from bnb.figures import video, watch
from bnb.game import api
from bnb.game.api import PRESETS
from bnb.play.loop import MatchEvent
from bnb.play.sides import mismatch
from bnb.runs.checkpoint import read_policy
from bnb.runs.records import Run, write_json
from bnb.runs.sinks import STEP


def test_a_policy_of_one_mode_plays_another_only_with_any_mode(tmp_path):
    policy = tiny_policy(tmp_path / "policy", PRESETS["frontline"])
    record = read_policy(policy).raw
    deathmatch = PRESETS["deathmatch"]
    rules, types_ = api.rules_of(deathmatch), api.ship_types()
    assert "mode" in mismatch(record, deathmatch, rules, types_)
    assert mismatch(record, deathmatch, rules, types_, any_mode=True) is None
    fewer = dataclasses.replace(deathmatch, num_fields=6)
    assert "num_fields" in mismatch(record, fewer, api.rules_of(fewer), types_, any_mode=True)


def test_the_tally_credits_kills_to_the_killer_s_team_and_counts_every_loss():
    tally = video.Tally(np.array([0, 0, 1, 1]))
    tally.count(
        [
            MatchEvent(1, "kill", ship=2, other=0),  # team 0 kills
            MatchEvent(2, "kill", ship=0, other=-1),  # the boundary: a loss, no kill
            MatchEvent(3, "capture", team=1),
            MatchEvent(4, "end", team=0),
            MatchEvent(9, "end", team=-1),
        ]
    )
    summary = tally.summary()
    assert (summary["policy_kills"], summary["scripted_kills"]) == (1, 0)
    assert (summary["policy_losses"], summary["scripted_losses"]) == (1, 1)
    assert (summary["policy_captures"], summary["scripted_captures"]) == (0, 1)
    assert (summary["policy_wins"], summary["policy_defeats"], summary["draws"]) == (1, 0, 1)


def _run(tmp_path, name: str = "a") -> Run:
    """A live run (this process) with no checkpoint yet."""
    return Run.create(tmp_path, name, {}, ["bnb", "train"], checkpoint_every=1)


def _checkpoint(run: Run, step: int) -> pathlib.Path:
    path = run.checkpoint_path(step)
    path.mkdir()
    (path / "checkpoint.json").write_text("{}")
    return path


def _record(run: Run, checkpoint: pathlib.Path, config=None) -> dict:
    """What a make_media writes, without making anything."""
    directory = watch.media_directory(run, checkpoint)
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(
        checkpoint=str(checkpoint),
        environment_steps=int(checkpoint.name[len("step_") :]),
        figures={},
        videos={},
        seconds=0.0,
        logged=False,
    )
    write_json(directory / watch.RECORD, record)
    return record


def test_pending_checkpoints_are_the_complete_ones_without_media(tmp_path):
    run = _run(tmp_path)
    first, second = _checkpoint(run, 10), _checkpoint(run, 20)
    (run.directory / "checkpoints" / "step_30.tmp").mkdir()  # being written
    assert watch.pending(run) == [first, second]
    _record(run, first)
    assert watch.pending(run) == [second]
    assert watch.unlogged(run) == [watch.media_directory(run, first)]


def test_the_watcher_makes_each_checkpoint_once_and_stops_after_the_run(tmp_path):
    run = _run(tmp_path)
    _checkpoint(run, 10)
    made, polls = [], []

    def make(run, checkpoint, config):
        made.append(checkpoint.name)
        if checkpoint.name == "step_20":
            raise FileNotFoundError("the retention deleted it")
        return _record(run, checkpoint)

    def sleep(seconds):
        polls.append(seconds)
        if len(polls) == 1:
            _checkpoint(run, 20)  # deleted while it is made: skipped
            _checkpoint(run, 30)
        elif len(polls) == 2:
            run.update(status="finished")

    config = watch.WatchConfig(interval=5.0)
    said = []
    watch.watch(tmp_path, "a", config, make=make, sleep=sleep, say=said.append)
    # The last pass finds the run stopped and step_20 still without media (here it is
    # still listed): one more try, nothing made, and the watcher stops.
    assert made == ["step_10", "step_20", "step_30", "step_20"]
    assert made[:3] == ["step_10", "step_20", "step_30"]
    assert made.count("step_10") == made.count("step_30") == 1
    assert all(seconds == 5.0 for seconds in polls)
    assert any("skipped step_20" in line for line in said)


def test_a_part_that_fails_is_recorded_and_a_deleted_checkpoint_is_not(tmp_path):
    run = _run(tmp_path)
    checkpoint = _checkpoint(run, 10)

    def broken():
        raise ValueError("no such head")

    assert "ValueError: no such head" in watch.attempt(broken, checkpoint)["error"]
    checkpoint.rename(checkpoint.with_name("step_10.deleting"))
    with pytest.raises(FileNotFoundError, match="is gone"):
        watch.attempt(broken, checkpoint)


def test_the_numbers_log_as_their_numeric_leaves():
    numbers = dict(own=dict(rows=3, error=0.5, by_age=[1.0, 2.0]), note="text", seen=True)
    assert watch.scalars(numbers, "numbers/frontline/next_state") == {
        "numbers/frontline/next_state/own/rows": 3,
        "numbers/frontline/next_state/own/error": 0.5,
        "numbers/frontline/next_state/seen": True,
    }


def test_the_companion_run_logs_the_media_at_the_checkpoint_s_steps(monkeypatch, tmp_path):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    run = _run(tmp_path, "trained")
    run.update(wandb_id="abc123")
    checkpoint = _checkpoint(run, 4096)
    record = _record(run, checkpoint)
    directory = watch.media_directory(run, checkpoint)
    from bnb.play.png import write_png

    (directory / "frontline").mkdir()
    write_png(directory / "frontline" / "shots_error.png", np.zeros((4, 4, 3), np.uint8))
    (directory / "videos").mkdir()
    (directory / "videos" / "frontline_seed0.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    record["figures"]["frontline"] = dict(
        sets=dict(shots=dict(pngs=["frontline/shots_error.png"], numbers=dict(slope=0.9))),
        capture_seconds=1.0,
        draw_seconds=2.0,
    )
    record["videos"]["frontline"] = dict(
        path="videos/frontline_seed0.mp4", game_seconds=100.0, policy_kills=3
    )
    record["videos"]["deathmatch"] = dict(error="RuntimeError: no EGL")
    write_json(directory / watch.RECORD, record)
    log = watch.MediaLog(run, watch.WatchConfig())
    try:
        assert log.session.id == "abc123media" == watch.companion_id(run)
        assert log.session.name == "trained-media" and log.session.group == "trained"
        assert "trained" in log.session.notes and "abc123" in log.session.notes
        entries = log.entries(record, directory)
        assert entries[STEP] == 4096
        assert entries["numbers/frontline/shots/slope"] == 0.9
        assert entries["game/frontline/policy_kills"] == 3
        assert {"video/frontline", "figures/frontline/shots_error"} <= entries.keys()
        assert "video/deathmatch" not in entries  # a failed part logs nothing
        log.log(directory)
    finally:
        log.close()
    assert json.loads((directory / watch.RECORD).read_text())["logged"] is True
    assert watch.unlogged(run) == []
    (records,) = watch.figures_root(run).rglob("offline-run-*/run-*.wandb")
    assert b"figures/frontline/shots_error" in records.read_bytes()


@pytest.mark.gpu  # an OpenGL context (EGL); the agent runs on the CPU
@pytest.mark.slow
def test_a_video_of_a_frontline_policy_plays_deathmatch(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # the agent on the CPU
    policy = tiny_policy(tmp_path / "policy", PRESETS["frontline"], "yemong")
    config = video.VideoConfig(seconds=1.0, frame_rate=10, width=320, height=180)
    path = tmp_path / "videos" / video.video_name("deathmatch", config)
    numbers = video.render(policy, "deathmatch", config, path)
    assert path.read_bytes()[4:8] == b"ftyp"  # an MP4
    assert numbers["frames"] == 10 and numbers["bytes"] == path.stat().st_size
    assert numbers["game_seconds"] == pytest.approx(4.0, abs=0.1)  # 1 s at 4x
