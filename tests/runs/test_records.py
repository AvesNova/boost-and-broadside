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


def test_the_retention_keeps_the_newest_eight_one_per_doubling_band_and_the_final():
    keep = retained(range(101), final=3)  # ordinals 0 to 100
    assert set(range(100 - RECENT + 1, 101)) <= keep and 3 in keep
    ages = sorted(100 - ordinal for ordinal in keep if ordinal != 3)
    assert ages == [0, 1, 2, 3, 4, 5, 6, 7, 12, 20, 36, 100]  # i = 88, 80, 64, 0


def test_the_retention_applied_after_every_write_is_stable_with_one_checkpoint_per_band():
    """Each band of ages [2^j, 2^(j+1)) from 8 on holds exactly one checkpoint at every
    newest ordinal, so what was deleted is never needed again."""
    kept: set[int] = set()
    for newest in range(2000):
        kept = retained(kept | {newest}, final=None)
        ages = sorted(newest - ordinal for ordinal in kept)
        assert ages[:RECENT] == list(range(min(RECENT, newest + 1)))
        for band in range(3, newest.bit_length()):
            assert sum(1 << band <= age < 2 << band for age in ages) == 1, (newest, band)
        assert len(kept) <= RECENT + newest.bit_length()
    final = 1234
    kept = set()
    for newest in range(2000):
        kept = retained(kept | {newest}, final=final if newest >= final else None)
    assert final in kept


def test_the_run_keeps_its_checkpoints_by_ordinal_and_deletes_by_renaming(tmp_path):
    run = Run.create(tmp_path, "a", cases.tiny_train_config(), ["bnb", "train"], 64)
    for ordinal in range(40):
        run.checkpoint_path(10 * ordinal).mkdir()
        run.added_checkpoint(10 * ordinal)
    ordinals = {int(step) // 10 for step in run.record["checkpoint_ordinals"]}
    assert ordinals == retained(range(40))
    assert [step // 10 for step in run.checkpoint_steps()] == sorted(ordinals)
    (tmp_path / "a" / "checkpoints" / "step_5.deleting").mkdir()  # a crash's leftovers
    (tmp_path / "a" / "checkpoints" / "step_7.tmp").mkdir()
    assert 5 not in run.checkpoint_steps() and 7 not in run.checkpoint_steps()
    run.checkpoint_path(400).mkdir()
    run.added_checkpoint(400, final=True)
    names = {path.name for path in (tmp_path / "a" / "checkpoints").iterdir()}
    assert not any(name.endswith((".deleting", ".tmp")) for name in names)
    assert run.record["checkpoint_ordinals"]["400"] == 40


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
    directory = provenance.write_result(
        run, "ratings", dict(elo=1000.0), dict(snapshot=run.directory / "league" / "snapshot_0")
    )
    result = json.loads((directory / "result.json").read_text())
    assert directory.parent == run.directory / "artifacts" / "ratings"
    assert result["payload"] == dict(elo=1000.0)
    assert result["inputs"]["snapshot"]["sha256"] == provenance.file_hash(
        run.directory / "league" / "snapshot_0" / "checkpoint.json"
    )
    origin = result["provenance"]
    assert origin["host"] == socket.gethostname() and origin["commit"] == provenance.commit()[0]
    assert {"python", "torch", "cuda", "command", "time", "dirty", "gpu"} <= origin.keys()
    second = provenance.write_result(run, "ratings", {}, {})
    assert second != directory
    assert os.path.isfile(second / "result.json")
