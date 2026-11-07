---
covers: [src/bnb/runs/queue.py, src/bnb/runs/session.py, src/bnb/cli.py]
reviewed: 2026-11-07 21:30
---

# Queueing GPU jobs

GPU work goes through one queue, `bnb queue` (`bnb.runs.queue`), so the GPU runs one job
at a time, in order. A job is a new training run, a resume of a run, or a tournament. One
runner takes the next ready job, runs it under the bench lock and records each change in
the queue file. While nothing else is ready it continues a background job (the current
champion); a job that becomes ready stops the champion gracefully, runs, and the champion
resumes after it.

## Start the runner

The runner is a long-lived process; run it as a systemd user unit, from any worktree: it
only schedules, and each job runs in the directory it was queued from, with that
environment's `bnb`:

<!-- not-runnable: starts a long-lived unit that takes the GPU -->
```bash
systemd-run --user --unit=bnb-queue -p KillMode=mixed -p TimeoutStopSec=1h \
    --working-directory="$PWD" "$PWD/.venv/bin/bnb" queue run
journalctl --user -u bnb-queue -f              # the runner's own lines
systemctl --user stop bnb-queue                # stop it gracefully
```

- `KillMode=mixed` sends the stop's SIGTERM to the runner alone, which stops the running
  job gracefully (after its update, with a checkpoint), queues it again as a resume, and
  exits. Without it systemd signals every process of the unit at once, including the
  bench lock's wrapper, which then releases the lock while the job is still finishing.
- The stop timeout bounds that wait; a long update and its checkpoint must fit in it.
- A unit starts with systemd's environment, not the shell's: give it what the jobs need
  with `--setenv=NAME=value` (W&B reads its key from `~/.netrc`).
- `--lock PATH` names the bench lock's wrapper (default `~/.local/share/bnb/locked.py`, a
  script that runs its arguments under the lock); `--lock none` runs without one.
  `--queue-file PATH` uses another queue (default `~/.local/share/bnb/queue.json`).
- Only one runner runs a queue; a second exits with the reason.

## Queue jobs

<!-- not-runnable: queues GPU jobs in the machine's queue -->
```bash
# the champion's continuation, preemptible
uv run --no-sync bnb queue add --background resume 2026-11-07-mixed-calm-fern --updates 4000
# a branch to its budget, then its ratings
uv run --no-sync bnb queue add train --preset mixed --name 2026-11-07-arm-d \
    --set learner.learning_rate=0.0001 --updates 600
uv run --no-sync bnb queue add --after 2 ratings 2026-11-07-arm-d --with \
    runs/2026-11-07-mixed-calm-fern/policies/step_400000000/exponential_average
# a BC run continued past its budget
uv run --no-sync bnb queue add resume 2026-11-07-mixed-amber-acorn --bc-budget-steps 200000000
```

- `bnb queue add KIND ARGUMENTS...`: `train` takes `bnb train`'s arguments (a new run;
  give it `--name` so a later job can rate or resume it), `resume` a run's name and then
  `bnb train --resume`'s options (a larger budget, for example), `ratings` `bnb ratings`'
  arguments. The arguments are checked when the job is queued, so a typo fails at once,
  not when the job reaches the GPU.
- `--after JOB` waits until that job has finished; a job whose dependency failed stays
  queued. `--background` marks the champion's continuation.
- The job runs in the directory it was queued from, with the `bnb` of that environment,
  and with the runs directory resolved there (added as `--runs-directory`).

## The order and preemption

The runner starts the first ready job in the queue's order that is not in the background,
else the first ready background job. When a job becomes ready while a background job runs,
the runner sends the background job's training process SIGTERM: it finishes its update,
writes a checkpoint and exits (`bnb.runs.session.StopRequest`). The ready job runs, then
the background job, queued again in its place as a resume of its run (with its device and
W&B options), continues to its own budget. A job that is waiting for its dependency does
not preempt.

## Watch the queue

<!-- not-runnable: reads the machine's queue -->
```bash
uv run --no-sync bnb queue list
uv run --no-sync bnb queue log 3 --lines 20
uv run --no-sync bnb queue remove 4
```

- `list` prints each job's number, state (`queued`, `after N` while it waits for job N,
  `running`, `finished`, `failed`, `stopped`, `interrupted`, `removed`; `(bg)` for a
  background job), kind, start and end times, the run or the arguments, and a note (why it
  was queued again). A running job's ETA is a clock time from its own progress lines: for
  training, the mean update time of its last few lines times the updates or steps left of
  its budget; for ratings, the tournament's own estimate for the preset it is on.
- `log` prints a job's log, `~/.local/share/bnb/logs/queue/<job>.log` (beside the queue
  file), where every attempt of the job appends.
- `remove` takes a queued job out at once; a running one is stopped gracefully first and
  marked `removed` (its run stays stopped, ready to resume by hand).

## What survives a crash

The queue file is plain JSON, one entry per job with its state, times, attempts and
note, rewritten atomically on every change, so a crash or a reboot loses nothing. When the
runner starts again, a job still marked running belongs to the runner that died: if its
process is still alive, the runner waits for it; otherwise the job is marked interrupted
and, for training with a checkpoint, queued again as a resume. A tournament that was
interrupted is left interrupted; queued again, it goes on from its own record of batches.

A run stopped by hand (a `STOP` file or a signal sent to it,
[Resuming](resuming-forking-sharing.md#start-and-stop-a-run)) is marked `stopped` and not
queued again.
