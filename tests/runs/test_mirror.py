"""`bnb runs mirror` (`bnb.runs.mirror`): a run's local metrics into W&B, idempotently, a
chain's stages grouped by the chain, following the run until its exit file. With a stand-in
W&B module (no network): it keeps each W&B run's logged points and summary across inits,
as a resumed W&B run does."""

import json
import shutil
import sys
import types

import pytest

from bnb.runs import mirror
from bnb.runs.sinks import METRICS, STEP


class Session:
    def __init__(self, store: dict, **settings) -> None:
        self.settings, self.store = settings, store
        self.points = store.setdefault(settings["id"], [])
        self.summary = {STEP: self.points[-1][STEP]} if self.points else {}
        self.entity, self.offline, self.finished = "someone", True, False

    def define_metric(self, *arguments, **settings) -> None:
        pass

    def log(self, values: dict) -> None:
        self.points.append(values)

    def finish(self) -> None:
        self.finished = True


@pytest.fixture
def wandb(monkeypatch):
    """A stand-in `wandb`: `init` returns a Session; `.sessions` lists them."""
    module = types.ModuleType("wandb")
    module.store, module.sessions = {}, []

    def init(**settings):
        module.sessions.append(Session(module.store, **settings))
        return module.sessions[-1]

    module.init = init
    module.Table = lambda columns, data: ("table", tuple(columns))
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def record(step: int) -> dict:
    return {STEP: step, "loss/total": 1.0 / step, "note": "text", "table/x": None}


def write_run(directory, name, status="running", chain=None, steps=(), partial=False):
    run = directory / name
    run.mkdir(parents=True, exist_ok=True)
    entry = dict(name=name, status=status, wandb_id=None)
    if chain:
        entry["chain"] = dict(name=chain, stage=name.rsplit("-", 1)[1], start_step=0)
    (run / "run.json").write_text(json.dumps(entry))
    (run / "config.json").write_text(json.dumps(dict(objective="bc", seed=0)))
    with open(run / METRICS, "a") as handle:
        for step in steps:
            handle.write(json.dumps(record(step)) + "\n")
        if partial:
            handle.write('{"counters/environment_steps": 9')  # a line being written


def write_chain(directory, name):
    (directory / name).mkdir(parents=True)
    stages = dict(bc=f"{name}-bc", rl=f"{name}-rl")
    (directory / name / "chain.json").write_text(json.dumps(dict(name=name, stages=stages)))


def test_a_chains_stages_mirror_once_each_grouped_by_the_chain(wandb, tmp_path):
    write_chain(tmp_path, "c")
    write_run(tmp_path, "c-bc", chain="c", steps=(1, 2, 3), partial=True)
    session = mirror.Mirror(tmp_path, "c", report=lambda line: None)
    assert session.sync() == 3
    (first,) = wandb.sessions
    assert first.settings["name"] == "c-bc" and first.settings["group"] == "c"
    assert first.settings["id"] == mirror.mirror_id("c-bc") and first.settings["resume"] == "allow"
    assert first.settings["config"] == dict(objective="bc", seed=0) and first.finished
    assert first.points[0] == {STEP: 1, "loss/total": 1.0}  # numbers only, as the live sink
    assert session.sync() == 0 and len(wandb.sessions) == 1  # nothing new: no W&B run opened

    # the partial line completes and more follow; a lost state file is recovered from W&B
    with open(tmp_path / "c-bc" / METRICS, "a") as handle:
        handle.write("}\n" + json.dumps(record(10)) + "\n")
    (tmp_path / "c-bc" / mirror.MIRROR_STATE).unlink()
    assert session.sync() == 2
    assert [point[STEP] for point in wandb.store[mirror.mirror_id("c-bc")]] == [1, 2, 3, 9, 10]
    state = json.loads((tmp_path / "c-bc" / mirror.MIRROR_STATE).read_text())
    assert state == dict(wandb_id=mirror.mirror_id("c-bc"), step=10)

    write_run(tmp_path, "c-rl", chain="c", steps=(1, 2))
    assert session.sync() == 2 and wandb.sessions[-1].settings["group"] == "c"


def test_a_run_that_streamed_live_continues_its_wandb_run(wandb, tmp_path):
    write_run(tmp_path, "a", steps=(1, 2))
    entry = json.loads((tmp_path / "a" / "run.json").read_text())
    (tmp_path / "a" / "run.json").write_text(json.dumps(entry | dict(wandb_id="abc123")))
    wandb.store["abc123"] = [{STEP: 1}]  # what the live sink logged
    assert mirror.Mirror(tmp_path, "a", report=lambda line: None).sync() == 1
    assert wandb.sessions[0].settings["group"] is None
    assert [point[STEP] for point in wandb.store["abc123"]] == [1, 2]


def test_a_chain_is_over_at_its_exit_file_or_with_no_stage_left_to_run(tmp_path):
    write_chain(tmp_path, "c")
    session = mirror.Mirror(tmp_path, "c")
    assert not session.over()  # no stage yet
    write_run(tmp_path, "c-bc", chain="c")
    assert not session.over()
    write_run(tmp_path, "c-bc", status="finished", chain="c")
    assert not session.over()  # the handover: the RL stage is still to come
    write_run(tmp_path, "c-rl", status="failed", chain="c")
    assert session.over()
    write_run(tmp_path, "c-rl", status="running", chain="c")
    assert not session.over()
    (tmp_path / "c.exit").write_text("0\n")
    assert session.over()


def test_follow_fetches_and_stops_after_the_exit_file(wandb, tmp_path, monkeypatch):
    """A fetch that copies from a stand-in remote: the follow syncs, sleeps, sees the exit
    file and syncs the records written before it."""
    remote, cache = tmp_path / "remote", tmp_path / "cache"
    write_chain(remote, "c")
    write_run(remote, "c-bc", chain="c", steps=(1,))

    def fetch(paths):
        for path in paths:
            if (remote / path).is_file():
                (cache / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(remote / path, cache / path)

    def sleep(seconds):
        write_run(remote, "c-bc", status="finished", chain="c", steps=(2,))
        write_run(remote, "c-rl", status="finished", chain="c", steps=(1, 2))
        (remote / "c.exit").write_text("0\n")

    session = mirror.Mirror(cache, "c", fetch, report=lambda line: None)
    monkeypatch.setattr(mirror.time, "sleep", sleep)
    session.follow(every=1)
    assert [point[STEP] for point in wandb.store[mirror.mirror_id("c-bc")]] == [1, 2]
    assert [point[STEP] for point in wandb.store[mirror.mirror_id("c-rl")]] == [1, 2]


def test_an_unknown_run_is_refused(wandb, tmp_path):
    with pytest.raises(FileNotFoundError, match="no run or chain 'nope'"):
        mirror.mirror(tmp_path, "nope")
