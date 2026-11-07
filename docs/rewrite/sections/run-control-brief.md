# Brief: graceful stop, a GPU job queue, BC policy snapshots, averages after the warm-up

**Decided by the author (2026-11-07):** training (BC and PPO) must stop gracefully on
request; GPU work goes through a queue tool instead of hand-written bash scripts; while
the GPU would be idle, the queue continues the current champion; a ready branch preempts
it, trains to its budget and is rated, then the champion resumes. And the uniform average
starts after the warm-up.

## 1. Graceful stop (BC and PPO)

- `bnb train` finishes the current update, writes a training checkpoint, records the stop
  in the run's record (`run.json`: stopped at step S, not finished) and exits 0 when either
  happens: the file `<run directory>/STOP` appears (checked after each update), or the
  process gets SIGTERM or SIGINT (a second signal aborts at once). `systemctl --user stop`
  sends SIGTERM, so a unit stop is graceful. Delete the STOP file when it has been honoured.
- `bnb train --resume NAME` (with an optional larger budget) continues exactly from that
  checkpoint (within rounding, as resumes are today).
- Tests (CPU): a STOP file after update k gives a checkpoint at k and a clean exit; SIGTERM
  the same; the resumed run equals an uninterrupted one (BC and PPO, tiny configs).

## 2. BC policy snapshots

BC saves its policies (live, exponential, uniform) at fixed environment-step points
(`--set` a cadence, default every 10M steps, plus the end) under `policies/step_<S>/...`
like PPO does, so a run continued past its budget keeps its earlier averages (today only
`policy-*` exists and is overwritten). `bnb ratings --with` accepts these paths.

## 3. Averages after the warm-up

`LeagueConfig.uniform_from` defaults to the run's learning-rate warm-up
(`learner.warmup_steps`) instead of its own 10M constant: one source, so a run with a
shorter warm-up (a fine-tune) starts its uniform average when its warm-up ends. An explicit
`--set` still overrides. Old build records are unaffected (it is a training setting, not a
build one). Test: a run with warm-up W starts counting at the first update past W.

## 4. The queue (`bnb queue`)

A durable job list and one runner, so the GPU runs one job at a time, in order:
- `bnb queue add <kind> ...`: kinds `train` (a new run: the `bnb train` arguments),
  `resume` (a run name, an optional new budget), `ratings` (the `bnb ratings` arguments);
  an optional `--after <job>` dependency; `--background` marks the champion continuation
  (preemptible).
- `bnb queue list` (state, start and end times, ETA from the job's own progress lines),
  `bnb queue remove <job>`, `bnb queue log <job>`.
- `bnb queue run`: the runner (started as a systemd user unit, for example
  `systemd-run --user --unit=bnb-queue bnb queue run`). It takes the next ready job, runs
  it under the bench lock (`~/.local/share/bnb/locked.py`) with its log in
  `~/.local/share/bnb/logs/queue/<job>.log`, and records each state change in the queue
  file (flushed, so a crash or reboot loses nothing; a job that was running when the runner
  died is marked interrupted and, for training, re-queued as a resume).
- Preemption: when a non-background job is ready and a background job is running, the
  runner stops the background job gracefully (part 1), runs the ready job, then re-queues
  the background job as a resume.
- The queue file lives at `~/.local/share/bnb/queue.json` (configurable). Plain JSON, one
  entry per job. No secrets in it.
- Tests (CPU, with stand-in commands): order, dependencies, preemption and re-queue,
  crash recovery, the lock is taken per job.

## Docs

`docs/guides/` (a page on running and queueing jobs, with the commands), the learner page
(snapshots, averages), the runs spec if it describes the record.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-run-control`, branch `feat/run-control` (from
  v2). Scratch `/tmp/bnb-scratch-run-control`.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`, at most 8 threads; BC runs own the
  GPU. Never take the bench lock (tests use a stand-in lock path). Never touch the real
  `~/.local/share/bnb/queue.json` or running units; tests use temporary paths.
- **Memory:** anything long under `systemd-run --user --scope -p MemoryMax=8G
  -p MemorySwapMax=0`.
- **Git:** never use `git stash`.
- **Python:** `uv sync --all-extras`, then `uv run --no-sync ...`.
- **Progress notes:** `/tmp/bnb-scratch-run-control/progress.md`, `## HH:MM` from
  `date +%H:%M`, 1-3 bullets, ETAs as clock times. Start with a plan note.
- **Tests:** what you add, the CPU resume tests, the quick tier
  (`pytest -m "not slow and not gpu"`), ruff, docs.

## Done

- Commit and push to `feat/run-control` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message: what each part does, the exact commands to start the runner and queue the
  next jobs, tests, anything open, the time.
