"""The GPU job queue (`bnb.runs.queue`, `bnb queue`), with stand-in commands
(`queue_stand_ins.py`): jobs run one at a time in order, each under the lock wrapper; a
dependency waits; a ready job preempts a background one, which resumes after it; a runner
that died leaves its job interrupted and re-queued; a removal and the runner's own stop are
graceful; the ETA comes from a job's progress lines."""

import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

import pytest

from bnb import cli
from bnb.runs import queue as job_queue
from bnb.runs.queue import Queue, Runner

STAND_INS = pathlib.Path(__file__).with_name("queue_stand_ins.py")
BNB = [sys.executable, str(STAND_INS), "bnb"]
LOCK = STAND_INS  # run as `python queue_stand_ins.py lock COMMAND...` through `lock_wrapper`


@pytest.fixture
def place(tmp_path, monkeypatch):
    """A queue file, a runs directory and the stand-ins' records, all temporary."""
    monkeypatch.setenv("QUEUE_TEST_EVENTS", str(tmp_path / "events.txt"))
    monkeypatch.setenv("QUEUE_TEST_LOCKS", str(tmp_path / "locks.txt"))
    monkeypatch.setenv("QUEUE_TEST_SECONDS", "0.05")
    (tmp_path / "runs").mkdir()
    return tmp_path


def lock_wrapper(place: pathlib.Path) -> pathlib.Path:
    """A wrapper script the runner runs as `python WRAPPER COMMAND...`: the stand-in lock."""
    wrapper = place / "lock.py"
    wrapper.write_text(
        "import runpy, sys\n"
        f"sys.argv = [{str(STAND_INS)!r}, 'lock', *sys.argv[1:]]\n"
        f"runpy.run_path({str(STAND_INS)!r}, run_name='__main__')\n"
    )
    return wrapper


def add(place, kind, *arguments, **options) -> dict:
    return Queue(place / "queue.json").add(
        kind,
        list(arguments),
        program=BNB,
        cwd=place,
        runs_directory=place / "runs",
        **options,
    )


def runner(place, lock=True) -> Runner:
    return Runner(Queue(place / "queue.json"), lock_wrapper(place) if lock else None, poll=0.02)


def events(place) -> list[list[str]]:
    path = place / "events.txt"
    return [line.split() for line in path.read_text().splitlines()] if path.exists() else []


def jobs(place) -> dict[int, dict]:
    return {job["id"]: job for job in Queue(place / "queue.json").read()["jobs"]}


def wait_for(condition, seconds: float = 30.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def test_jobs_run_one_at_a_time_in_order_each_under_the_lock(place):
    for name in ("a", "b", "c"):
        add(place, "train", "--name", name, "--updates", "2")
    runner(place).run(until_idle=True)
    assert events(place) == [[name, "0", "2", "finished"] for name in ("a", "b", "c")]
    assert {job["state"] for job in jobs(place).values()} == {"finished"}
    locks = [line.split() for line in (place / "locks.txt").read_text().splitlines()]
    assert [line[0] for line in locks] == ["start", "end"] * 3  # one lock per job, never shared
    assert all(float(a[1]) <= float(b[1]) for a, b in zip(locks, locks[1:]))
    job = jobs(place)[1]
    log = (place / "logs" / "queue" / "1.log").read_text()
    assert log.startswith("[queue] job 1 attempt 1") and "update    2" in log
    assert job["started"] and job["ended"] and job["returncode"] == 0
    assert job["attempts"][0]["state"] == "finished"


def test_a_job_waits_for_its_dependency_and_a_failed_one_blocks_it(place):
    add(place, "train", "--name", "champion", "--updates", "5", background=True)
    add(place, "ratings", "champion", after=1)  # not ready: it does not preempt job 1
    add(place, "train", "--name", "fail-a", "--updates", "1", after=1)
    add(place, "ratings", "fail-a", after=3)
    add(place, "train", "--name", "b", "--updates", "1", after=2)
    runner(place).run(until_idle=True)
    assert events(place) == [
        ["champion", "0", "5", "finished"],
        ["ratings", "champion", "--runs-directory", str(place / "runs")],
        ["b", "0", "1", "finished"],
    ]
    states = {number: job["state"] for number, job in jobs(place).items()}
    assert states == {1: "finished", 2: "finished", 3: "failed", 4: "queued", 5: "finished"}
    assert jobs(place)[3]["returncode"] == 3
    listing = job_queue.describe(Queue(place / "queue.json"))
    assert listing[4].split()[:3] == ["4", "after", "3"]


def test_a_ready_job_preempts_the_background_one_which_resumes_after_it(place):
    champion = ("--name", "champion", "--updates", "40", "--device", "cpu", "--wandb")
    add(place, "train", *champion, background=True)
    log = place / "logs" / "queue" / "1.log"

    def branch() -> None:
        wait_for(lambda: log.exists() and "update    3" in log.read_text())
        add(place, "train", "--name", "branch", "--updates", "2")
        add(place, "ratings", "branch", after=2)

    adding = threading.Thread(target=branch)
    adding.start()
    runner(place).run(until_idle=True)
    adding.join()
    (first, second, rated, resumed) = events(place)
    assert first[0] == "champion" and first[3] == "stopped" and 3 <= int(first[2]) < 40
    assert second == ["branch", "0", "2", "finished"] and rated[0] == "ratings"
    assert resumed == ["champion", first[2], "40", "finished"]
    champion = jobs(place)[1]
    assert champion["state"] == "finished" and champion["kind"] == "resume"
    assert champion["background"] and champion["arguments"] == [
        "champion",
        "--runs-directory",
        str(place / "runs"),
        "--device",
        "cpu",
        "--wandb",
    ]  # what a checkpoint does not keep
    assert [attempt["state"] for attempt in champion["attempts"]] == ["stopped", "finished"]
    assert champion["attempts"][0]["stop"] == "preempted"
    assert champion["note"] == "preempted by job 2: queued again as a resume"


def test_a_job_running_when_the_runner_died_is_interrupted_and_resumed(place):
    job = add(place, "train", "--name", "a", "--updates", "3")
    # the dead runner's job: one update and a checkpoint, then the process gone
    run = place / "runs" / "a"
    (run / "checkpoints" / "step_1").mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    record = dict(name="a", status="running", pid=dead.pid, budget=3, done=1, segments=[])
    (run / "run.json").write_text(json.dumps(record))
    with Queue(place / "queue.json").edit() as queue:
        queue["jobs"][0].update(state="running", pid=dead.pid, started=job["added"])
        queue["jobs"][0]["attempts"].append(dict(started=job["added"], state="running"))
    runner(place).run(until_idle=True)
    assert events(place) == [["a", "1", "3", "finished"]]
    recovered = jobs(place)[1]
    assert [attempt["state"] for attempt in recovered["attempts"]] == ["interrupted", "finished"]
    assert recovered["kind"] == "resume" and recovered["state"] == "finished"


def test_a_job_whose_process_outlived_the_runner_is_waited_for(place):
    add(place, "train", "--name", "a", "--updates", "5")
    orphan = subprocess.Popen(
        [*BNB, "train", "--name", "a", "--updates", "5", "--runs-directory", str(place / "runs")]
    )
    with Queue(place / "queue.json").edit() as queue:
        queue["jobs"][0].update(state="running", pid=orphan.pid)
        queue["jobs"][0]["attempts"].append(dict(started=None, state="running"))
    add(place, "train", "--name", "b", "--updates", "1")
    threading.Thread(target=orphan.wait).start()  # reaped, as init would
    runner(place).run(until_idle=True)
    assert events(place) == [["a", "0", "5", "finished"], ["b", "0", "1", "finished"]]
    assert jobs(place)[1]["state"] == "finished"


def test_removing_a_running_job_stops_it_gracefully(place):
    add(place, "train", "--name", "a", "--updates", "40")
    add(place, "train", "--name", "b", "--updates", "1")
    queue = Queue(place / "queue.json")
    log = place / "logs" / "queue" / "1.log"

    def remove() -> None:
        wait_for(lambda: log.exists() and "update    2" in log.read_text())
        assert "after its update" in queue.remove(1)

    removing = threading.Thread(target=remove)
    removing.start()
    runner(place).run(until_idle=True)
    removing.join()
    assert [line[0] for line in events(place)] == ["a", "b"] and events(place)[0][3] == "stopped"
    assert jobs(place)[1]["state"] == "removed"
    assert queue.remove(2) == "removed job 2" and 2 not in jobs(place)
    with pytest.raises(job_queue.QueueError, match="no job 7"):
        queue.remove(7)


def test_a_stopped_runner_stops_its_job_queues_it_again_and_exits(place):
    add(place, "train", "--name", "a", "--updates", "60")
    environment = dict(os.environ, PYTHONPATH=str(STAND_INS.parent))
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from bnb.cli import main; main()",
            "queue",
            "run",
            "--queue-file",
            str(place / "queue.json"),
            "--lock",
            str(lock_wrapper(place)),
            "--poll",
            "0.02",
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log = place / "logs" / "queue" / "1.log"
    wait_for(lambda: log.exists() and "update    2" in log.read_text(), 60)
    process.send_signal(signal.SIGTERM)  # as `systemctl --user stop` with KillMode=mixed
    output = process.communicate(timeout=60)[0]
    assert process.returncode == 0, output
    ((name, first, last, status),) = events(place)
    assert (name, first, status) == ("a", "0", "stopped")
    job = jobs(place)[1]
    assert job["state"] == "queued" and job["kind"] == "resume"
    assert job["attempts"][0]["stop"] == "runner stopped"
    runner(place).run(until_idle=True)  # the next start resumes it
    assert events(place)[-1] == ["a", last, "60", "finished"]


def test_only_one_runner_runs_a_queue(place):
    queue = Queue(place / "queue.json")
    import fcntl

    with open(place / "queue.json.runner", "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(job_queue.QueueError, match="another runner"):
            Runner(queue, None, poll=0.02).run(until_idle=True)


def test_the_eta_reads_the_jobs_progress_lines(place):
    log = place / "1.log"
    lines = ["[queue] job 1 attempt 1", "run a in /x/a"]
    lines += [f"update {n:4d}  rollout   1500 ms  learn    500 ms  entropy" for n in (1, 2, 3)]
    log.write_text("\n".join(lines) + "\n")
    mtime = time.time()
    os.utime(log, (mtime, mtime))
    job = dict(kind="train", arguments=["--updates", "33"], run="a", runs_directory=None)
    want = time.strftime("%H:%M", time.localtime(mtime + 30 * 2.0))  # 30 updates of 2 s
    assert job_queue.progress(job, log) == want
    bc = [
        f"update {n:4d}  steps {n * 1_000_000:>11,}  rollout    900 ms  learn    100 ms  bc kl"
        for n in (1, 2, 3)
    ]
    log.write_text("\n".join(["[queue] job 1 attempt 2", *bc]) + "\n")
    os.utime(log, (mtime, mtime))
    job = dict(kind="resume", arguments=["a", "--bc-budget-steps", "13000000"], run=None)
    job["runs_directory"] = None
    want = time.strftime("%H:%M", time.localtime(mtime + 10 * 1.0))  # 10 updates of 1 s
    assert job_queue.progress(job, log) == want
    rating = "frontline: batch 3, 3072 games, the largest standard error 40.1 (the batch "
    log.write_text(rating + "took 20 s); done about 21:40 (9 batches), by 22:10 at the latest\n")
    assert job_queue.progress(dict(kind="ratings"), log) == "21:40 (frontline)"


def test_bnb_queue_add_list_and_log(place, monkeypatch, capsys):
    queue_file = str(place / "queue.json")

    def bnb(*arguments: str) -> str:
        monkeypatch.setattr(sys, "argv", ["bnb", "queue", *arguments, "--queue-file", queue_file])
        cli.main()
        return capsys.readouterr().out

    out = bnb("add", "train", "--preset", "mixed", "--name", "a", "--updates", "2")
    assert out.startswith("job 1: train --preset mixed --name a --updates 2 --runs-directory")
    bnb("add", "--after", "1", "--background", "resume", "a", "--updates", "4")
    first, second = jobs(place)[1], jobs(place)[2]
    assert first["runs_directory"] == str(cli.runs_directory_of(None).resolve())
    assert (second["after"], second["background"], second["run"]) == (1, True, "a")
    assert job_queue.command(second)[-7:-2] == ["train", "--resume", "a", "--updates", "4"]
    with pytest.raises(SystemExit):  # a typo fails when it is queued, not on the GPU
        bnb("add", "train", "--updatess", "2")
    with pytest.raises(SystemExit, match="queue a resume as"):
        bnb("add", "train", "--resume", "a")
    listing = bnb("list").splitlines()
    assert listing[1].split()[:3] == ["1", "queued", "train"]
    assert listing[2].split()[:4] == ["2", "after", "1", "resume"]
    assert bnb("log", "1").strip().endswith("1.log:")
