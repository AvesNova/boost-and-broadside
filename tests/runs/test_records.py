"""The run record (runs spec 5, 8): names, retention, staleness, offline results."""

import datetime
import json
import os
import random
import socket

import runs_cases as cases

from bnb.runs import provenance, records
from bnb.runs.checkpoint import save_policy
from bnb.runs.records import RECENT, Run, retained, run_name


def test_a_run_name_is_the_date_the_label_and_two_words_drawn_again_on_a_clash(tmp_path):
    date = datetime.date(2026, 11, 2)
    first = run_name("mixed", tmp_path, date, random.Random(0))
    _, _, _, label, adjective, noun = first.split("-")
    assert first.startswith("2026-11-02-mixed-") and label == "mixed"
    assert adjective in records.ADJECTIVES and noun in records.NOUNS
    (tmp_path / first).mkdir()
    assert run_name("mixed", tmp_path, date, random.Random(0)) != first


def test_the_retention_keeps_the_newest_eight_a_geometric_history_and_the_final():
    steps = list(range(1, 101))  # checkpoints every step, newest 100
    keep = retained(steps, 1, final=3)
    assert set(steps[-RECENT:]) <= keep
    ages = sorted(100 - step for step in keep if step != 3)
    assert ages == [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 99]  # 99: the oldest
    assert 3 in keep


def test_the_retention_keeps_a_logarithmic_history_as_a_run_grows():
    """Applied after every new checkpoint, it keeps the newest eight, the first, and a few
    in between. A deleted checkpoint cannot come back, so the history has gaps where the
    nearest to an age was deleted earlier (at 300: ages 8, 19, 75 and 299)."""
    kept: list[int] = []
    for step in range(1, 301):
        kept = sorted(retained(kept + [step], 1))
    ages = sorted(300 - step for step in kept)
    assert ages[:RECENT] == list(range(RECENT))
    older = ages[RECENT:]
    assert older[0] == RECENT and older[-1] == 299  # the first checkpoint, nearest every age
    assert len(older) <= 9  # one per doubling at most
    assert all(later <= 4 * earlier for earlier, later in zip(older, older[1:]))


def test_a_running_record_whose_process_is_gone_shows_as_stale(tmp_path):
    run = Run.create(tmp_path, "a", cases.tiny_train_config(), ["bnb", "train"], 64)
    assert run.shown_status() == "running"  # this process
    run.update(pid=2**22 + 12345)  # above the default pid_max: no such process
    assert run.shown_status() == "running (stale)"
    run.update(host="another-host")
    assert run.shown_status() == "running"  # another host's process cannot be checked


def test_run_json_is_rewritten_atomically_and_reopens(tmp_path):
    run = Run.create(tmp_path, "a", cases.tiny_train_config(), ["bnb", "train"], 64)
    run.start_segment(0)
    run.end_segment(128, "stopped")
    again = Run.open(tmp_path, "a")
    assert again.record == run.record and again.record["status"] == "stopped"
    assert not any(path.name.endswith(".tmp") for path in run.directory.iterdir())
    segment = again.record["segments"][0]
    assert (segment["start_step"], segment["end_step"]) == (0, 128)
    assert json.loads((run.directory / "config.json").read_text())["num_envs"] == 16


def test_an_offline_result_records_its_provenance_and_its_policies_by_hash(tmp_path):
    run = Run.create(tmp_path, "a", cases.tiny_train_config(), ["bnb", "train"], 64)
    save_policy(run.directory / "league" / "snapshot_0", cases.tiny_model("relational"))
    directory = provenance.write_result(run, "ratings", dict(elo=1000.0),
                                        dict(snapshot=run.directory / "league" / "snapshot_0"))
    result = json.loads((directory / "result.json").read_text())
    assert directory.parent == run.directory / "artifacts" / "ratings"
    assert result["payload"] == dict(elo=1000.0)
    assert result["inputs"]["snapshot"]["sha256"] == provenance.file_hash(
        run.directory / "league" / "snapshot_0" / "checkpoint.json")
    origin = result["provenance"]
    assert origin["host"] == socket.gethostname() and origin["commit"] == provenance.commit()[0]
    assert {"python", "torch", "cuda", "command", "time", "dirty", "gpu"} <= origin.keys()
    second = provenance.write_result(run, "ratings", {}, {})
    assert second != directory
    assert os.path.isfile(second / "result.json")
