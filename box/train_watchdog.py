#!/usr/bin/env python3
"""Watch a `bnb train --objective chain` run on a box from the laptop; exit on an alert or a
milestone, so a supervising session waking on the exit can look and re-arm it.

The box side is `box/dry_run.sh`, which writes, beside the chain's directory in the runs
directory: `<chain>.log` (the chain's stdout and stderr), `<chain>.pid` (the trainer's
process id, the python process itself) and `<chain>.exit` (its exit code, once it ends).

Every poll (one ssh call, BatchMode and a connect timeout) reads the log's new bytes, the
process's /proc/<pid>/cmdline (never a name match: `pgrep -f` matches its own ssh command),
the exit file, the GPU utilisation and the volume's free disk. It prints one line per event
and exits after a poll with an ALERT (exit code 1) or a MILESTONE (exit code 0):

- ALERT dead: the process is gone without the chain's finish (or with a nonzero exit code);
- ALERT stall: no new update line for --stall-minutes in a stage that has logged one, or no
  first update within --compile-minutes of a stage's start (the first compile);
- ALERT nonfinite: nan or inf in an update line, or nonzero `nonfinite beliefs`;
- ALERT error: Traceback, an ...Error, out of memory, Killed, a failed or stopped run;
- ALERT trainer <rule>: the first time per stage a trainer ALERT rule (gauge_runaway,
  rating_drop, recompile, ...) appears on an update line; its WARNING rules print a NOTE
  once per stage and do not exit (`--quiet-alerts` silences named rules);
- ALERT sps: the median of the last 3 steady updates below --sps-fraction of the expected;
- ALERT gpu idle: utilisation 0 on --gpu-zero-polls polls in a row in a steady stage;
- ALERT disk: the volume's free space under --disk-min-gb;
- ALERT ssh: --ssh-failures failed polls in a row (a single failure only prints a NOTE);
- MILESTONE: BC steady (3 updates after the first), BC done, RL steady, RL done, chain
  finished (exit code 0).

State (log offset, stages, milestones and alerts already reported) lives in --state, so a
re-armed watchdog reports nothing twice. A condition alert (stall, gpu, disk, ssh, sps)
repeats only after --repeat-minutes while it holds.

Test mode: `--replay LOG` feeds a local log through the same logic on a simulated clock
(each update line's rollout + learn time), re-arming after every exit; see
tests/box/test_train_watchdog.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time

KST = datetime.timezone(datetime.timedelta(hours=9))

# the trainer's rules (`bnb.runs.triggers.default_rules`): ALERT severity; the rest warn
TRAINER_ALERTS = (
    "skipped_step",
    "recompile",
    "memory",
    "behaviour_mismatch",
    "nonfinite_input",
    "never_seen_ship",
    "throughput",
    "rating_drop",
    "gauge_runaway",
    "combat_collapse",
)

UPDATE = re.compile(r"^update\s+(\d+)\s")
STEPS = re.compile(r"\ssteps\s+([\d,]+)\s")
TIMES = re.compile(r"rollout\s+(\d+)\s+ms\s+learn\s+(\d+)\s+ms")
NONFINITE = re.compile(r"(?<![A-Za-z_])[+-]?(nan|inf)(?![A-Za-z_])", re.IGNORECASE)
BELIEFS = re.compile(r"nonfinite beliefs\s+(\d+)")
ALERTS = re.compile(r"\salerts:\s+(.*)$")
RUN_START = re.compile(r"^run (\S+) in (\S+)")
RUN_END = re.compile(r"^run (\S+) (finished|stopped|failed|\w+) at step (\d+)")
ERROR = re.compile(
    r"Traceback|\w*Error\b|out of memory|OutOfMemory|\bOOM\b|\bKilled\b|"
    r"Segmentation fault|core dumped|CUDA error|illegal memory access"
)


@dataclasses.dataclass
class Settings:
    """The thresholds (see the module docstring)."""

    stall_minutes: float = 10.0
    compile_minutes: float = 80.0
    bc_sps: float = 21_900.0
    rl_sps: float = 23_000.0
    sps_fraction: float = 0.6
    steps_per_update: int = 1_048_576
    steady_after: int = 3  # updates after the first one before the stage is steady
    gpu_zero_polls: int = 3
    disk_min_gb: float = 20.0
    ssh_failures: int = 4
    repeat_minutes: float = 30.0
    quiet_alerts: tuple[str, ...] = ()


def new_state() -> dict:
    return dict(
        offset=0,
        stages=[],  # dict(name, kind, start, updates, last_update, last_time, sps, finished, seen)
        start=None,  # first poll's time: the chain's own setup counts as a compile phase
        milestones=[],
        ssh_failures=0,
        gpu_zero=0,
        fired={},  # condition alert -> time last reported
        chain_line=False,
        polls=0,
    )


def kind_of(name: str, line: str | None = None) -> str | None:
    if line is not None:
        if " bc kl " in line:
            return "bc"
        if " entropy " in line and " kl " in line:
            return "rl"
    if name.endswith("-bc"):
        return "bc"
    if name.endswith("-rl"):
        return "rl"
    return None


def clock(now: float) -> str:
    return datetime.datetime.fromtimestamp(now, KST).strftime("%m-%d %H:%M KST")


class Watch:
    """The logic: `poll(snapshot)` returns the events of one poll as (level, text)."""

    def __init__(self, settings: Settings, state: dict | None = None):
        self.settings = settings
        self.state = state or new_state()

    # ---- helpers
    def stage(self) -> dict | None:
        return self.state["stages"][-1] if self.state["stages"] else None

    def condition(self, events, key: str, now: float, text: str) -> None:
        """A condition alert, repeated only after repeat_minutes while it holds."""
        last = self.state["fired"].get(key)
        if last is None or now - last >= self.settings.repeat_minutes * 60:
            self.state["fired"][key] = now
            events.append(("ALERT", text))

    def clear(self, key: str) -> None:
        self.state["fired"].pop(key, None)

    def milestone(self, events, key: str, text: str) -> None:
        if key not in self.state["milestones"]:
            self.state["milestones"].append(key)
            events.append(("MILESTONE", text))

    # ---- one poll
    def poll(self, snapshot: dict) -> list[tuple[str, str]]:
        """snapshot: ok (the ssh call worked), now, log (new complete lines' text), size and
        the offset it was read at, pid, alive, exit (None until the exit file exists), gpu
        (utilisation %, or None), disk_gb (free, or None), error (when not ok)."""
        events: list[tuple[str, str]] = []
        state, settings = self.state, self.settings
        now = snapshot["now"]
        state["polls"] += 1
        if state["start"] is None:
            state["start"] = now
        if not snapshot["ok"]:
            state["ssh_failures"] += 1
            if state["ssh_failures"] >= settings.ssh_failures:
                self.condition(
                    events,
                    "ssh",
                    now,
                    f"ssh: {state['ssh_failures']} failed polls in a row ({snapshot.get('error')})",
                )
            else:
                events.append(
                    ("NOTE", f"ssh poll failed ({state['ssh_failures']}): {snapshot.get('error')}")
                )
            return events
        state["ssh_failures"] = 0
        self.clear("ssh")
        if snapshot.get("reset"):
            events.append(("NOTE", "the log shrank: read again from its start"))
        for line in snapshot["log"].splitlines():
            self.line(line.rstrip("\r"), now, events)
        self.health(snapshot, now, events)
        return events

    def line(self, line: str, now: float, events) -> None:
        state, settings = self.state, self.settings
        if line.startswith("chain "):
            state["chain_line"] = True
        start = RUN_START.match(line)
        if start:
            state["stages"].append(
                dict(
                    name=start.group(1),
                    kind=kind_of(start.group(1)),
                    start=now,
                    updates=0,
                    last_update=None,
                    last_time=None,
                    sps=[],
                    finished=None,
                    seen=[],
                )
            )
            return
        stage = self.stage()
        end = RUN_END.match(line)
        if end and stage is not None and end.group(1) == stage["name"]:
            stage["finished"] = end.group(2)
            if end.group(2) == "finished":
                kind = (stage["kind"] or "run").upper()
                self.milestone(
                    events,
                    f"done:{stage['name']}",
                    f"{kind} done: {stage['name']} finished at step {int(end.group(3)):,} "
                    f"({stage['updates']} updates logged)",
                )
            else:
                events.append(("ALERT", f"error: {line.strip()}"))
            return
        update = UPDATE.match(line)
        if update is None:
            if ERROR.search(line):
                events.append(("ALERT", f"error: {line.strip()[:300]}"))
            return
        if stage is None:  # an update before any run line: a log without its head
            stage = dict(
                name="?",
                kind=None,
                start=now,
                updates=0,
                last_update=None,
                last_time=None,
                sps=[],
                finished=None,
                seen=[],
            )
            state["stages"].append(stage)
        if stage["kind"] is None:
            stage["kind"] = kind_of(stage["name"], line)
        number = int(update.group(1))
        stage["updates"] += 1
        stage["last_update"] = number
        stage["last_time"] = now
        self.clear(f"stall:{stage['name']}")
        self.clear(f"compile:{stage['name']}")
        label = f"{(stage['kind'] or '?').upper()} update {number}"
        if NONFINITE.search(line):
            events.append(("ALERT", f"nonfinite: {label}: {line.strip()[:400]}"))
        beliefs = BELIEFS.search(line)
        if beliefs and int(beliefs.group(1)) > 0:
            events.append(("ALERT", f"nonfinite: {label}: nonfinite beliefs {beliefs.group(1)}"))
        alerts = ALERTS.search(line)
        if alerts:
            for name in alerts.group(1).split():
                rule = name.split("/")[0]
                if rule in settings.quiet_alerts or name in settings.quiet_alerts:
                    continue
                if name in stage["seen"]:
                    continue
                stage["seen"].append(name)
                if rule in TRAINER_ALERTS:
                    events.append(("ALERT", f"trainer {name}: first in {stage['name']} at {label}"))
                else:
                    events.append(
                        ("NOTE", f"trainer warning {name}: first in {stage['name']} at {label}")
                    )
        times = TIMES.search(line)
        if times and stage["updates"] > 1:  # a stage's first logged update carries the compile
            steps = STEPS.search(line)
            per_update = settings.steps_per_update
            if steps and stage.get("last_steps") is not None:
                per_update = int(steps.group(1).replace(",", "")) - stage["last_steps"]
            seconds = (int(times.group(1)) + int(times.group(2))) / 1e3
            if seconds > 0 and per_update > 0:
                stage["sps"] = (stage["sps"] + [per_update / seconds])[-3:]
        steps = STEPS.search(line)
        if steps:
            stage["last_steps"] = int(steps.group(1).replace(",", ""))
        expected = settings.bc_sps if stage["kind"] == "bc" else settings.rl_sps
        if stage["updates"] > settings.steady_after and len(stage["sps"]) == 3:
            median = statistics.median(stage["sps"])
            if median < settings.sps_fraction * expected:
                self.condition(
                    events,
                    f"sps:{stage['name']}",
                    now,
                    f"sps: {label}: median of the last 3 updates {median:,.0f} steps/s, "
                    f"below {settings.sps_fraction:.0%} of the expected {expected:,.0f}",
                )
            else:
                self.clear(f"sps:{stage['name']}")
        if stage["updates"] == settings.steady_after + 1 and stage["sps"]:
            kind = (stage["kind"] or "run").upper()
            self.milestone(
                events,
                f"steady:{stage['name']}",
                f"{kind} steady: {stage['name']} update {number}, "
                f"{statistics.median(stage['sps']):,.0f} steps/s (expected {expected:,.0f})",
            )

    def health(self, snapshot: dict, now: float, events) -> None:
        state, settings = self.state, self.settings
        stage = self.stage()
        steady = stage is not None and stage["updates"] > 0 and stage["finished"] is None
        # the process
        if not snapshot["alive"]:
            finished_rl = (
                stage is not None and stage["finished"] == "finished" and stage["kind"] == "rl"
            )
            if snapshot["exit"] == "0" and finished_rl:
                self.milestone(events, "chain", "chain finished (exit code 0)")
            else:
                exit_code = snapshot["exit"] if snapshot["exit"] is not None else "unknown"
                self.condition(
                    events,
                    "dead",
                    now,
                    f"dead: trainer pid {snapshot.get('pid')} is gone "
                    f"(exit code {exit_code}; last stage "
                    f"{stage['name'] if stage else 'none'} "
                    f"{'finished' if stage and stage['finished'] else 'unfinished'})",
                )
            return
        # progress
        if steady:
            if now - stage["last_time"] > settings.stall_minutes * 60:
                self.condition(
                    events,
                    f"stall:{stage['name']}",
                    now,
                    f"stall: no update line for {(now - stage['last_time']) / 60:.0f} min "
                    f"(last {stage['name']} update {stage['last_update']})",
                )
        else:
            since = stage["start"] if stage is not None and stage["finished"] is None else None
            if since is None:  # the launch, or the hand-over between the stages
                since = state["start"] if stage is None else self.finish_time(stage, now)
            name = stage["name"] if stage is not None and stage["finished"] is None else "next"
            if now - since > settings.compile_minutes * 60:
                self.condition(
                    events,
                    f"compile:{name}",
                    now,
                    f"stall: no first update {(now - since) / 60:.0f} min after the start of "
                    f"{name} (compile limit {settings.compile_minutes:.0f} min)",
                )
        # the GPU, while a stage runs its updates
        gpu = snapshot.get("gpu")
        if steady and gpu is not None:
            state["gpu_zero"] = state["gpu_zero"] + 1 if gpu == 0 else 0
            if state["gpu_zero"] >= settings.gpu_zero_polls:
                self.condition(
                    events, "gpu", now, f"gpu idle: utilisation 0 on {state['gpu_zero']} polls"
                )
            elif gpu > 0:
                self.clear("gpu")
        else:
            state["gpu_zero"] = 0
        disk = snapshot.get("disk_gb")
        if disk is not None:
            if disk < settings.disk_min_gb:
                self.condition(events, "disk", now, f"disk: {disk:.1f} GB free on the volume")
            else:
                self.clear("disk")

    def finish_time(self, stage: dict, now: float) -> float:
        """When a finished stage ended (the watchdog's clock): its last update line's time."""
        stage.setdefault("finish_time", stage["last_time"] or now)
        return stage["finish_time"]

    def status(self, snapshot: dict) -> str:
        stage = self.stage()
        part = "no stage yet"
        if stage is not None:
            sps = f"{statistics.median(stage['sps']):,.0f}/s" if stage["sps"] else "-"
            part = (
                f"{stage['name']} update {stage['last_update']} "
                f"({stage['updates']} logged) sps {sps}"
                f"{' ' + stage['finished'] if stage['finished'] else ''}"
            )
        return (
            f"status: {part}; gpu {snapshot.get('gpu')}%, disk {snapshot.get('disk_gb')} GB, "
            f"alive {snapshot.get('alive')}"
        )


# ---- the box side


REMOTE = r"""
R={runs}; C={chain}; O={offset}
P=$(cat $R/$C.pid 2>/dev/null)
echo "@@PID ${{P:-none}}"
if [ -n "$P" ] && [ -r /proc/$P/cmdline ]; then
  echo "@@CMDLINE $(tr '\0' ' ' < /proc/$P/cmdline)"
else
  echo "@@CMDLINE none"
fi
echo "@@EXIT $(cat $R/$C.exit 2>/dev/null || echo none)"
G=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
echo "@@GPU $(echo "$G" | head -1)"
echo "@@DISK $(df -B1 --output=avail $R 2>/dev/null | tail -1)"
S=$(stat -c %s $R/$C.log 2>/dev/null || echo 0)
echo "@@SIZE $S"
if [ "$S" -lt "$O" ]; then O=0; fi
echo "@@FROM $O"
echo "@@LOG"
tail -c +$((O + 1)) $R/$C.log 2>/dev/null | head -c {cap}
"""


def parse_remote(raw: bytes, chain: str) -> dict:
    """The REMOTE script's output: the fields and the log's bytes."""
    marker = raw.find(b"@@LOG\n")
    if marker < 0:
        raise ValueError("no @@LOG marker in the box's reply")
    head, log = raw[:marker].decode(errors="replace"), raw[marker + len(b"@@LOG\n") :]
    fields = {}
    for line in head.splitlines():
        if line.startswith("@@"):
            key, _, value = line[2:].partition(" ")
            fields[key] = value.strip()
    pid = fields.get("PID", "none")
    cmdline = fields.get("CMDLINE", "none")
    alive = (
        pid != "none"
        and cmdline != "none"
        and "bnb" in cmdline
        and "train" in cmdline
        and chain in cmdline
    )
    exit_code = fields.get("EXIT", "none")
    gpu = fields.get("GPU", "")
    disk = fields.get("DISK", "")
    return dict(
        pid=pid,
        alive=alive,
        exit=None if exit_code == "none" else exit_code,
        gpu=int(gpu) if gpu.isdigit() else None,
        disk_gb=int(disk) / 1e9 if disk.isdigit() else None,
        size=int(fields.get("SIZE", "0") or 0),
        start=int(fields.get("FROM", "0") or 0),
        raw_log=log,
    )


def complete_lines(raw_log: bytes) -> tuple[str, int]:
    """The text of the complete lines and their byte count (a partial last line waits)."""
    end = raw_log.rfind(b"\n") + 1
    return raw_log[:end].decode(errors="replace"), end


def remote_snapshot(arguments, offset: int) -> dict:
    script = REMOTE.format(
        runs=arguments.runs_directory, chain=arguments.chain, offset=offset, cap=4_000_000
    )
    command = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={arguments.connect_timeout}",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3",
        arguments.host,
        "bash -s",
    ]  # fmt: skip
    now = time.time()
    try:
        done = subprocess.run(
            command, input=script.encode(), capture_output=True, timeout=arguments.ssh_timeout
        )
        if done.returncode != 0:
            raise RuntimeError(f"ssh exit {done.returncode}: {done.stderr.decode()[-200:].strip()}")
        fields = parse_remote(done.stdout, arguments.chain)
    except (subprocess.TimeoutExpired, RuntimeError, ValueError, OSError) as error:
        return dict(ok=False, now=now, error=str(error)[:300])
    text, used = complete_lines(fields.pop("raw_log"))
    return dict(
        ok=True,
        now=now,
        log=text,
        reset=fields["start"] < offset,
        next_offset=fields["start"] + used,
        **fields,
    )


# ---- replay (test mode)


def replay_snapshots(log_path: pathlib.Path, settings: Settings, interval: float, options):
    """Snapshots of a local log revealed on a simulated clock: each line appears at the
    cumulative rollout + learn time of the update lines up to it (plus --replay-compile
    seconds before a stage's first update); the process ends with the log (`--replay-end`)."""
    lines = log_path.read_text(errors="replace").splitlines(keepends=True)
    t0 = 1_800_000_000.0
    times, now = [], t0
    for line in lines:
        if RUN_START.match(line):
            now += options.replay_compile
        times.append(now)
        timing = TIMES.search(line) if UPDATE.match(line) else None
        if timing:
            # the line is printed once its update ends: it appears after its own time
            now += (int(timing.group(1)) + int(timing.group(2))) / 1e3
            times[-1] = now
    end = max(times[-1] if times else t0, t0) + interval
    shown, now = 0, t0
    overrides = json.loads(options.replay_overrides) if options.replay_overrides else {}
    exit_code = {"exit0": "0", "exit1": "1", "alive": None}[options.replay_end]
    poll = 0
    while True:
        now += interval
        poll += 1
        start = shown
        while shown < len(lines) and times[shown] <= now:
            shown += 1
        over = now > end
        snapshot = dict(
            ok=True,
            now=now,
            log="".join(lines[start:shown]),
            pid="12345",
            alive=not over or options.replay_end == "alive",
            exit=exit_code if over else None,
            gpu=95,
            disk_gb=150.0,
        )
        snapshot.update(overrides.get(str(poll), {}))
        yield snapshot
        if over or poll > options.replay_max_polls:
            return


# ---- the command line


def load_state(path: pathlib.Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError, json.JSONDecodeError:
        return None


def save_state(path: pathlib.Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state))
    os.replace(temporary, path)


def emit(level: str, text: str, now: float) -> None:
    print(f"{clock(now)} {level} {text}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="box5090")
    parser.add_argument("--chain", help="the chain's name (its log, pid and exit files)")
    parser.add_argument("--runs-directory", default="/workspace/runs")
    parser.add_argument(
        "--state", help="state file (default ~/.local/state/bnb/watchdog-CHAIN.json)"
    )
    parser.add_argument("--interval", type=float, default=120.0, help="seconds between polls")
    parser.add_argument("--connect-timeout", type=int, default=15)
    parser.add_argument("--ssh-timeout", type=float, default=90.0)
    parser.add_argument(
        "--status-every", type=int, default=15, help="a status line every N polls (0: never)"
    )
    parser.add_argument("--stall-minutes", type=float, default=Settings.stall_minutes)
    parser.add_argument("--compile-minutes", type=float, default=Settings.compile_minutes)
    parser.add_argument("--bc-sps", type=float, default=Settings.bc_sps)
    parser.add_argument("--rl-sps", type=float, default=Settings.rl_sps)
    parser.add_argument("--sps-fraction", type=float, default=Settings.sps_fraction)
    parser.add_argument("--steps-per-update", type=int, default=Settings.steps_per_update)
    parser.add_argument("--gpu-zero-polls", type=int, default=Settings.gpu_zero_polls)
    parser.add_argument("--disk-min-gb", type=float, default=Settings.disk_min_gb)
    parser.add_argument("--ssh-failures", type=int, default=Settings.ssh_failures)
    parser.add_argument("--repeat-minutes", type=float, default=Settings.repeat_minutes)
    parser.add_argument("--quiet-alerts", nargs="*", default=[], help="trainer rules to ignore")
    parser.add_argument("--replay", help="test mode: a local log on a simulated clock")
    parser.add_argument(
        "--replay-compile",
        type=float,
        default=1800.0,
        help="seconds before each stage's first update",
    )
    parser.add_argument("--replay-end", choices=("exit0", "exit1", "alive"), default="exit0")
    parser.add_argument("--replay-overrides", help='JSON {"poll": {field: value}} for chosen polls')
    parser.add_argument("--replay-max-polls", type=int, default=10_000)
    options = parser.parse_args(argv)
    settings = Settings(
        stall_minutes=options.stall_minutes,
        compile_minutes=options.compile_minutes,
        bc_sps=options.bc_sps,
        rl_sps=options.rl_sps,
        sps_fraction=options.sps_fraction,
        steps_per_update=options.steps_per_update,
        gpu_zero_polls=options.gpu_zero_polls,
        disk_min_gb=options.disk_min_gb,
        ssh_failures=options.ssh_failures,
        repeat_minutes=options.repeat_minutes,
        quiet_alerts=tuple(options.quiet_alerts),
    )
    if options.replay:  # every event, re-arming after each exit (a fresh state each run)
        watch = Watch(settings)
        exits = 0
        for snapshot in replay_snapshots(
            pathlib.Path(options.replay), settings, options.interval, options
        ):
            events = watch.poll(snapshot)
            for level, text in events:
                emit(level, text, snapshot["now"])
            if any(level in ("ALERT", "MILESTONE") for level, _ in events):
                exits += 1
                watch = Watch(settings, json.loads(json.dumps(watch.state)))  # re-armed
        print(f"replay: {exits} exits", flush=True)
        return 0
    if not options.chain:
        parser.error("--chain is required (or --replay)")
    state_path = pathlib.Path(
        options.state or pathlib.Path.home() / ".local/state/bnb" / f"watchdog-{options.chain}.json"
    )
    watch = Watch(settings, load_state(state_path))
    print(
        f"{clock(time.time())} watching {options.chain} on {options.host} every "
        f"{options.interval:.0f} s (state {state_path}, {watch.state['polls']} earlier polls, "
        f"milestones {watch.state['milestones']})",
        flush=True,
    )
    while True:
        snapshot = remote_snapshot(options, watch.state["offset"])
        events = watch.poll(snapshot)
        if snapshot["ok"]:
            watch.state["offset"] = snapshot["next_offset"]
        save_state(state_path, watch.state)
        for level, text in events:
            emit(level, text, snapshot["now"])
        if options.status_every and watch.state["polls"] % options.status_every == 1:
            emit("STATUS", watch.status(snapshot), snapshot["now"])
        levels = {level for level, _ in events}
        if "ALERT" in levels:
            if snapshot["ok"] and snapshot.get("log"):
                tail = snapshot["log"].strip().splitlines()[-8:]
                print(
                    "  last log lines:\n" + "\n".join("    " + line[:300] for line in tail),
                    flush=True,
                )
            return 1
        if "MILESTONE" in levels:
            return 0
        time.sleep(options.interval)


if __name__ == "__main__":
    sys.exit(main())
