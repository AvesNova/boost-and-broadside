"""Stand-ins for the queue's tests (`test_queue.py`): a `bnb` that trains and rates nothing,
and a bench lock wrapper that takes no lock.

    python queue_stand_ins.py bnb train --name RUN --runs-directory D [--updates N] ...
    python queue_stand_ins.py bnb train --resume RUN --runs-directory D ...
    python queue_stand_ins.py bnb ratings ...
    python queue_stand_ins.py lock COMMAND...

The `bnb` keeps a run directory as `bnb train` does where the queue reads it (run.json's
name and status, checkpoints/step_<update>/), prints `bnb train`'s first line and PPO's
progress lines, an update every QUEUE_TEST_SECONDS, and on SIGTERM stops after the update
in progress, as `bnb.runs.session.StopRequest` does. Its budget is --updates (3 by
default) for a new run, kept in the run for a resume. A run named fail-... exits 3 after its
first update. Each process appends `<run> <first update> <last update> <status>` (or
`ratings <arguments>`) to QUEUE_TEST_EVENTS. The lock appends `start <time> <pid>` and
`end <time>` to QUEUE_TEST_LOCKS around the command it runs, and like the real wrapper does
not pass signals on.
"""

import json
import os
import pathlib
import signal
import subprocess
import sys
import time


def option(arguments, name, default=None):
    return arguments[arguments.index(name) + 1] if name in arguments else default


def event(line: str) -> None:
    with open(os.environ["QUEUE_TEST_EVENTS"], "a") as events:
        events.write(line + "\n")


def train(arguments) -> int:
    name = option(arguments, "--resume") or option(arguments, "--name")
    directory = pathlib.Path(option(arguments, "--runs-directory")) / name
    record = directory / "run.json"
    if "--resume" in arguments:
        run = json.loads(record.read_text())
    else:
        (directory / "checkpoints").mkdir(parents=True)
        run = dict(name=name, segments=[], budget=int(option(arguments, "--updates", 3)), done=0)
        print(f"run {name} in {directory}", flush=True)
    run["budget"] = int(option(arguments, "--updates", run["budget"]))
    run.update(status="running", pid=os.getpid())
    record.write_text(json.dumps(run))
    stopping = []
    signal.signal(signal.SIGTERM, lambda number, frame: stopping.append(number))
    first = run["done"]
    while run["done"] < run["budget"] and not stopping:
        time.sleep(float(os.environ.get("QUEUE_TEST_SECONDS", "0.05")))
        run["done"] += 1
        print(f"update {run['done']:4d}  rollout     30 ms  learn     20 ms  kl +0.0", flush=True)
        if name.startswith("fail-"):
            return 3
    (directory / "checkpoints" / f"step_{run['done']}").mkdir(exist_ok=True)
    run["status"] = "finished" if run["done"] >= run["budget"] else "stopped"
    record.write_text(json.dumps(run))
    event(f"{name} {first} {run['done']} {run['status']}")
    return 0


def main() -> int:
    role, *arguments = sys.argv[1:]
    if role == "lock":
        with open(os.environ["QUEUE_TEST_LOCKS"], "a") as locks:
            locks.write(f"start {time.time()} {os.getpid()}\n")
        code = subprocess.run(arguments).returncode
        with open(os.environ["QUEUE_TEST_LOCKS"], "a") as locks:
            locks.write(f"end {time.time()}\n")
        return code
    kind, *arguments = arguments
    if kind == "train":
        return train(arguments)
    time.sleep(float(os.environ.get("QUEUE_TEST_SECONDS", "0.05")))
    event("ratings " + " ".join(arguments))
    return 0


if __name__ == "__main__":
    sys.exit(main())
