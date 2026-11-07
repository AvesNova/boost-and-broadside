"""The GPU job queue (`bnb queue`): a durable job list and one runner, so the GPU runs one
job at a time, in order.

A job is a `bnb train` (a new run), a resume of a run, or a `bnb ratings`, with an optional
dependency (`after`: it waits until that job has finished) and a background flag: a
background job (the champion's continuation) runs only while no other job is ready, and a
job that becomes ready stops it gracefully (SIGTERM, `bnb.runs.session.StopRequest`), runs,
and the background job goes back into the queue as a resume of its run.

The queue file is plain JSON (`Queue`), one entry per job, rewritten atomically under a
file lock on every state change, so a crash or a reboot loses nothing. The runner
(`Runner`) runs each job under the bench lock (a wrapper, by default
`~/.local/share/bnb/locked.py`) with its output in `logs/queue/<job>.log` beside the queue
file. A job that was running when the runner died is marked interrupted at the runner's next
start and, for training, re-queued as a resume; one whose process is still alive is waited
for first. `eta` reads a job's progress from its own log lines.
"""

import contextlib
import datetime
import fcntl
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

from .records import Run, write_json

DATA = pathlib.Path.home() / ".local" / "share" / "bnb"
QUEUE_FILE = DATA / "queue.json"
LOCK_WRAPPER = DATA / "locked.py"
KINDS = ("train", "resume", "ratings")
# queued: waiting (or ready); running; the rest are ended (a resume of a stopped run is a new
# attempt of the same job, queued again)
STATES = ("queued", "running", "finished", "failed", "stopped", "interrupted", "removed")
ENDED = ("finished", "failed", "stopped", "interrupted", "removed")
# what the runner does with a job whose stop it asked for
PREEMPTED, RUNNER_STOPPED, REMOVED = "preempted", "runner stopped", "removed"
# `bnb train`'s progress lines (`bnb.cli`): PPO's and BC's, and its first line for a new run
PPO_LINE = re.compile(r"^update\s+(\d+)\s+rollout\s+(\d+) ms\s+learn\s+(\d+) ms")
BC_LINE = re.compile(r"^update\s+(\d+)\s+steps\s+([\d,]+)\s+rollout\s+(\d+) ms\s+learn\s+(\d+) ms")
RUN_LINE = re.compile(r"^run (\S+) in (.+)$")
# `bnb ratings`' batch lines (`bnb.league.tournament`): the preset and its expected end
RATINGS_LINE = re.compile(r"^(\S+): batch \d+, .*; done about (\d\d:\d\d)")
# a new run's options its resume needs again, as a checkpoint does not keep them, and whether
# each takes a value
CARRIED = {"--device": True, "--wandb": False}
# the progress lines an ETA averages over, after the segment's first (it compiles)
ETA_LINES = 5


def now() -> str:
    """The local time, ISO 8601 to the second."""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def clock(stamp: str | None) -> str:
    """An ISO time as the local HH:MM, or "-"."""
    if not stamp:
        return "-"
    return datetime.datetime.fromisoformat(stamp).astimezone().strftime("%H:%M")


def option(arguments: list[str], name: str) -> str | None:
    """The value of `--name VALUE` or `--name=VALUE` in `arguments` (the last one given)."""
    found = None
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            found = arguments[index + 1]
        elif argument.startswith(name + "="):
            found = argument[len(name) + 1 :]
    return found


class QueueError(ValueError):
    """A queue command that does not fit the queue (an unknown job, a bad dependency)."""


class Queue:
    """The queue file at `path` (`QUEUE_FILE` by default): {"next_id", "jobs"}, each job a
    dict (`add`). Every change is a read, modify and atomic write under the file lock
    (`edit`), so `bnb queue add` and the runner never lose each other's changes."""

    def __init__(self, path: pathlib.Path = QUEUE_FILE) -> None:
        self.path = pathlib.Path(path)

    @property
    def logs(self) -> pathlib.Path:
        """The jobs' logs: `logs/queue/` beside the queue file."""
        return self.path.parent / "logs" / "queue"

    def log_path(self, job: dict) -> pathlib.Path:
        """Where job `job` logs."""
        return self.logs / f"{job['id']}.log"

    def read(self) -> dict:
        """The queue as it is now (empty when there is no file)."""
        if not self.path.exists():
            return dict(next_id=1, jobs=[])
        return json.loads(self.path.read_text())

    @contextlib.contextmanager
    def edit(self):
        """The queue under the file lock, written back (atomically, flushed) after the block."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(self.path.name + ".lock"), "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                queue = self.read()
                yield queue
                write_json(self.path, queue)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def add(
        self,
        kind: str,
        arguments: list[str],
        *,
        program: list[str],
        cwd: pathlib.Path,
        runs_directory: pathlib.Path | None = None,
        after: int | None = None,
        background: bool = False,
    ) -> dict:
        """Append a queued job and return it. kind: "train" (`arguments` are `bnb train`'s),
        "resume" (a run's name, then `bnb train --resume`'s other options, such as a new
        budget) or "ratings" (`bnb ratings`'); program: the command that runs `bnb`; cwd: the
        job's working directory (its commit); runs_directory: the runs directory it uses
        (added to the arguments, so the runner's own directory does not matter); after: a
        job it waits for; background: preemptible (the champion's continuation)."""
        if kind not in KINDS:
            raise QueueError(f"kind {kind!r} is not one of {KINDS}")
        arguments = list(arguments)
        if kind == "train" and option(arguments, "--resume") is not None:
            raise QueueError("queue a resume as `bnb queue add resume RUN ...`")
        if kind == "resume" and (not arguments or arguments[0].startswith("-")):
            raise QueueError("a resume names its run first: `bnb queue add resume RUN ...`")
        if runs_directory is not None and option(arguments, "--runs-directory") is None:
            arguments += ["--runs-directory", str(runs_directory)]
        with self.edit() as queue:
            if after is not None and not any(job["id"] == after for job in queue["jobs"]):
                raise QueueError(f"no job {after} to wait for")
            job = dict(
                id=queue["next_id"],
                kind=kind,
                arguments=arguments,
                after=after,
                background=background,
                state="queued",
                added=now(),
                started=None,
                ended=None,
                returncode=None,
                pid=None,
                cwd=str(cwd),
                program=list(program),
                run=run_name_given(kind, arguments),
                runs_directory=option(arguments, "--runs-directory"),
                stop=None,
                note=None,
                attempts=[],
            )
            queue["next_id"] += 1
            queue["jobs"].append(job)
        return job

    def remove(self, number: int) -> str:
        """Remove job `number`: a queued or ended one at once, a running one once the runner
        has stopped it gracefully (marked `stop`). Returns what happened."""
        with self.edit() as queue:
            job = find(queue, number)
            if job["state"] == "running":
                job["stop"] = REMOVED
                return f"job {number} is running: the runner stops it after its update"
            queue["jobs"].remove(job)
            return f"removed job {number}"


def find(queue: dict, number: int) -> dict:
    """Job `number` of `queue`."""
    for job in queue["jobs"]:
        if job["id"] == number:
            return job
    raise QueueError(f"no job {number} in the queue")


def run_name_given(kind: str, arguments: list[str]) -> str | None:
    """The run a job's arguments name: a resume's first argument, a train's `--name`."""
    if kind == "resume":
        return arguments[0]
    if kind == "train":
        return option(arguments, "--name")
    return None


def command(job: dict) -> list[str]:
    """The `bnb` command line a job runs (without the lock's wrapper)."""
    if job["kind"] == "resume":
        name, *rest = job["arguments"]
        return [*job["program"], "train", "--resume", name, *rest]
    return [*job["program"], job["kind"], *job["arguments"]]


def ready(queue: dict, job: dict) -> bool:
    """Whether a queued job may start: its dependency, if any, has finished."""
    if job["state"] != "queued":
        return False
    if job["after"] is None:
        return True
    dependency = next((other for other in queue["jobs"] if other["id"] == job["after"]), None)
    return dependency is not None and dependency["state"] == "finished"


def next_job(queue: dict) -> dict | None:
    """The job the runner starts next: the first ready job in order that is not in the
    background, else the first ready background job."""
    waiting = [job for job in queue["jobs"] if ready(queue, job)]
    foreground = [job for job in waiting if not job["background"]]
    return (foreground or waiting or [None])[0]


def run_of(job: dict, log: pathlib.Path) -> Run | None:
    """The run a train or resume job trains, once it exists: the name its arguments give or
    its log's first line (`run NAME in DIRECTORY`)."""
    if job["kind"] == "ratings":
        return None
    name, directory = job.get("run"), job.get("runs_directory")
    if (name is None or directory is None) and log.exists():
        with open(log) as lines:
            for line in lines:
                found = RUN_LINE.match(line.strip())
                if found:
                    path = pathlib.Path(found.group(2))
                    name, directory = found.group(1), str(path.parent)
                    break
    if name is None or directory is None:
        return None
    try:
        return Run.open(pathlib.Path(directory), name)
    except FileNotFoundError, json.JSONDecodeError:
        return None


def resume_arguments(job: dict, run: Run) -> list[str]:
    """A resume's arguments for a job that trained `run`: a resume job's own; for a new
    run's, its name, the runs directory and the options a checkpoint does not keep
    (`CARRIED`: the device and the W&B stream). The budget and every setting come from the
    checkpoint."""
    if job["kind"] == "resume":
        return list(job["arguments"])
    arguments = [run.record["name"], "--runs-directory", str(run.directory.parent)]
    for name, takes_value in CARRIED.items():
        value = option(job["arguments"], name)
        if takes_value and value is not None:
            arguments += [name, value]
        elif not takes_value and name in job["arguments"]:
            arguments.append(name)
    return arguments


def as_resume(job: dict, run: Run, note: str) -> None:
    """Queue `job` again as a resume of `run` (to its own budget, in the checkpoint), with
    the same id, place, dependency and background flag."""
    job.update(
        kind="resume",
        arguments=resume_arguments(job, run),
        run=run.record["name"],
        runs_directory=str(run.directory.parent),
        state="queued",
        pid=None,
        stop=None,
        note=note,
    )


def settle(job: dict, returncode: int | None, log: pathlib.Path) -> None:
    """A job's state once its process has ended (`returncode` None: unknown, as when the
    runner died), by why it stopped (`job["stop"]`, the runner's request) and, for
    training, by its run's status:

    - finished: exit 0 (and, for training, the run finished);
    - stopped: a training run that stopped on request. One the runner stopped (preempted,
      or the runner itself stopping) is queued again as a resume of its run; a stop the
      runner did not ask for (a STOP file, a signal by hand) stays stopped;
    - interrupted: the runner died while it ran, or it failed while being stopped. Training
      with a checkpoint is queued again as a resume, a train that never began its run as
      itself; ratings the runner stopped are queued again as they were (a tournament goes
      on from its record);
    - removed: a removal stopped it; failed: anything else."""
    run = run_of(job, log)
    status = None if run is None else run.record["status"]
    asked = job["stop"]
    if asked == REMOVED:
        ended = "removed"
    elif job["kind"] != "ratings" and status == "finished" and returncode in (0, None):
        ended = "finished"
    elif returncode is None:
        ended = "interrupted"
    elif job["kind"] == "ratings":
        ended = "finished" if returncode == 0 else "interrupted" if asked else "failed"
    elif returncode == 0 and status == "stopped":
        ended = "stopped"
    else:
        ended = "interrupted" if asked else "failed"
    job.update(returncode=returncode, ended=now(), pid=None, state=ended, stop=None)
    if job["attempts"]:
        job["attempts"][-1].update(
            ended=job["ended"], state=ended, returncode=returncode, stop=asked
        )
    reason = job["note"] if asked == PREEMPTED and job["note"] else asked
    again = (ended == "stopped" and asked in (PREEMPTED, RUNNER_STOPPED)) or (
        ended == "interrupted" and job["kind"] != "ratings"
    )
    if again and run is not None and run.checkpoint_steps():
        as_resume(job, run, f"{reason or 'interrupted'}: queued again as a resume")
    elif again and run is None and job["kind"] == "train":
        job.update(
            state="queued", note=f"{reason or 'interrupted'} before its run began: queued again"
        )
    elif ended == "interrupted" and job["kind"] == "ratings" and asked:
        job.update(state="queued", note=f"{asked}: queued again (it goes on from its record)")


def alive(pid: int) -> bool:
    """Whether process `pid` exists and is not a zombie."""
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError, ProcessLookupError, IndexError:
        return False


def children(pid: int) -> list[int]:
    """The direct children of process `pid` (from /proc)."""
    found = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError, ProcessLookupError, IndexError:
            continue
        if int(fields[1]) == pid:
            found.append(int(entry.name))
    return found


def progress(job: dict, log: pathlib.Path) -> str | None:
    """The job's expected end as a clock time (HH:MM), from its log's progress lines: for
    training, the mean update time of the segment's last few lines (rollout plus learning,
    after its first) times the updates or steps left of its budget, from the log's last
    write; for ratings, the tournament's own estimate for the preset it is on. None when it
    cannot tell yet."""
    if not log.exists():
        return None
    lines = log.read_text(errors="replace").splitlines()
    if job["kind"] == "ratings":
        for line in reversed(lines):
            found = RATINGS_LINE.match(line)
            if found:
                return f"{found.group(2)} ({found.group(1)})"
        return None
    segment = 0  # the attempt's lines start after the runner's last header
    for index, line in enumerate(lines):
        if line.startswith("[queue] "):
            segment = index + 1
    ppo = [found for line in lines[segment:] if (found := PPO_LINE.match(line))]
    bc = [found for line in lines[segment:] if (found := BC_LINE.match(line))]
    rows = bc or ppo
    if len(rows) < 2:
        return None
    timed = rows[1:][-ETA_LINES:]
    seconds = sum(
        int(row.group(row.re.groups - 1)) + int(row.group(row.re.groups)) for row in timed
    )
    seconds /= 1000 * len(timed)
    budget = budget_of(job, log, bc=bool(bc))
    if budget is None:
        return None
    if bc:
        steps = [int(row.group(2).replace(",", "")) for row in rows]
        per_update = steps[-1] - steps[-2]
        left = max(budget - steps[-1], 0) / per_update if per_update > 0 else None
    else:
        left = max(budget - int(rows[-1].group(1)), 0)
    if left is None:
        return None
    written = datetime.datetime.fromtimestamp(log.stat().st_mtime).astimezone()
    return (written + datetime.timedelta(seconds=left * seconds)).strftime("%H:%M")


def budget_of(job: dict, log: pathlib.Path, bc: bool) -> int | None:
    """A training job's budget: PPO's total updates, BC's environment steps. The job's own
    option, else its run's (the start's config.json, then each segment's override); None
    before the run exists."""
    name = "--bc-budget-steps" if bc else "--updates"
    given = option(job["arguments"], name)
    if given is not None:
        return int(given)
    run = run_of(job, log)
    if run is not None:
        path = "stop.budget_steps" if bc else "updates"
        budget = None
        config = run.directory / "config.json"
        if config.exists():
            value = json.loads(config.read_text())
            for part in path.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            budget = value
        for segment in run.record.get("segments", []):
            budget = segment.get("overrides", {}).get(path, budget)
        if budget is not None:
            return int(budget)
    return None


class Runner:
    """The queue's one runner: takes the next ready job (`next_job`), runs it under the lock
    wrapper (`lock`, a Python script that runs its arguments under the bench lock; None:
    no wrapper) with `python`, logs it, and records every state change in the queue. It
    preempts a background job for a ready one, stops a running job a removal marked, and on
    SIGTERM or SIGINT stops the running job gracefully, queues it again and exits. poll:
    seconds between looks at the queue and the job."""

    def __init__(
        self,
        queue: Queue,
        lock: pathlib.Path | None = LOCK_WRAPPER,
        python: str = sys.executable,
        poll: float = 2.0,
    ) -> None:
        self.queue, self.lock, self.python, self.poll = queue, lock, python, poll
        self.process: subprocess.Popen | None = None
        self.job: int | None = None  # the running job's id
        self.pid: int | None = None  # its process (the wrapper's, with a lock)
        self.asked: str | None = None  # the stop the runner asked of it
        self.signalled: list[int] = []  # the processes the stop went to
        self.stopping = False

    # ---- the loop

    def run(self, until_idle: bool = False) -> None:
        """Run jobs until SIGTERM or SIGINT (or, with `until_idle`, until nothing is running
        or ready). Only one runner per queue file: a second exits with the reason."""
        self.queue.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.queue.path.with_name(self.queue.path.name + ".runner"), "a+") as held:
            try:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise QueueError(f"another runner holds {self.queue.path}") from None
            previous = {
                number: signal.signal(number, self._stop)
                for number in (signal.SIGTERM, signal.SIGINT)
            }
            try:
                self._recover()
                self._loop(until_idle)
            finally:
                for number, handler in previous.items():
                    signal.signal(number, handler)

    def _stop(self, number: int, frame) -> None:
        self.stopping = True
        print(f"[queue] {signal.Signals(number).name}: stopping", flush=True)

    def _loop(self, until_idle: bool) -> None:
        while True:
            if self.job is not None:
                self._watch()
            if self.job is None:
                if self.stopping:
                    return
                with self.queue.edit() as queue:
                    job = next_job(queue)
                    if job is not None:
                        self._start(job)
                if job is None and until_idle:
                    return
            time.sleep(self.poll)

    # ---- one job

    def _start(self, job: dict) -> None:
        """Start `job` (inside the queue's edit) and mark it running."""
        log = self.queue.log_path(job)
        log.parent.mkdir(parents=True, exist_ok=True)
        line = command(job)
        if self.lock is not None:
            line = [self.python, str(self.lock), *line]
        attempt = len(job["attempts"]) + 1
        with open(log, "a") as out:
            out.write(f"[queue] job {job['id']} attempt {attempt} at {now()}: {' '.join(line)}\n")
            out.flush()
            self.process = subprocess.Popen(
                line,
                cwd=job["cwd"],
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # a terminal's Ctrl-C reaches the runner only
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
            )
        self.job, self.pid, self.asked, self.signalled = job["id"], self.process.pid, None, []
        job.update(state="running", started=now(), ended=None, pid=self.pid, returncode=None)
        job["attempts"].append(dict(started=job["started"], ended=None, state="running"))
        print(f"[queue] job {job['id']} started ({job['kind']}), log {log}", flush=True)

    def _watch(self) -> None:
        """Look at the running job: settle it once ended, else stop it when a ready job
        preempts it, a removal asks, or the runner is stopping."""
        returncode = self._ended()
        with self.queue.edit() as queue:
            job = find(queue, self.job)
            if returncode is not False:
                if self.asked and not job["stop"]:
                    job["stop"] = self.asked
                settle(job, returncode, self.queue.log_path(job))
                print(f"[queue] job {job['id']} {job['state']} ({returncode})", flush=True)
                self.job = self.process = self.pid = None
                return
            if self.asked:
                return
            ready_job = next_job(queue)
            if job["stop"] == REMOVED:
                self._ask(job, REMOVED)
            elif self.stopping:
                self._ask(job, RUNNER_STOPPED)
            elif job["background"] and ready_job is not None and not ready_job["background"]:
                self._ask(job, PREEMPTED)
                job["note"] = f"preempted by job {ready_job['id']}"

    def _ended(self) -> int | None | bool:
        """The running job's exit code once it and the processes signalled have ended (None
        for an adopted process, whose code is unknown), else False."""
        if self.process is not None:
            returncode = self.process.poll()
            if returncode is None:
                return False
        elif alive(self.pid):
            return False
        else:
            returncode = None
        if any(alive(pid) for pid in self.signalled):
            return False  # the wrapper ended before the job it ran
        return returncode

    def _ask(self, job: dict, why: str) -> None:
        """Stop the running job gracefully: SIGTERM to the job's own process (the lock
        wrapper's children, or the process itself without one), once."""
        targets = children(self.pid) if self.lock is not None else [self.pid]
        targets = targets or [self.pid]
        for pid in targets:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
        self.asked, self.signalled = why, targets
        job["stop"] = why
        print(f"[queue] job {job['id']}: {why}, asked to stop", flush=True)

    def _recover(self) -> None:
        """At the start: a job marked running belongs to a runner that died. Its process, if
        still alive, is adopted (waited for, then settled); else it is settled as
        interrupted."""
        with self.queue.edit() as queue:
            for job in queue["jobs"]:
                if job["state"] != "running":
                    continue
                if job["pid"] is not None and alive(job["pid"]) and self.job is None:
                    self.job, self.pid, self.process = job["id"], job["pid"], None
                    print(f"[queue] job {job['id']}: adopted process {job['pid']}", flush=True)
                    continue
                settle(job, None, self.queue.log_path(job))
                print(f"[queue] job {job['id']}: interrupted, now {job['state']}", flush=True)


def describe(queue: Queue) -> list[str]:
    """`bnb queue list`'s lines: per job its id, state, kind, run or arguments, start and end
    (HH:MM), ETA while running, the dependency and the note."""
    data = queue.read()
    lines = [f"{'job':>4}  {'state':12} {'kind':8} {'start':5}  {'end':5}  {'eta':13} what"]
    for job in data["jobs"]:
        state = job["state"] + (" (bg)" if job["background"] else "")
        if job["state"] == "queued" and job["after"] is not None and not ready(data, job):
            state = f"after {job['after']}"
        what = job["run"] or " ".join(job["arguments"])
        eta = progress(job, queue.log_path(job)) if job["state"] == "running" else None
        line = (
            f"{job['id']:>4}  {state:12} {job['kind']:8} {clock(job['started']):5}  "
            f"{clock(job['ended']):5}  {eta or '-':13} {what}"
        )
        if job["note"]:
            line += f"  [{job['note']}]"
        lines.append(line)
    return lines
