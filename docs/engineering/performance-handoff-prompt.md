# Real-time experiment handoff

Copy everything below this line into a fresh Codex task.

---

Work in `/home/vizia/avesnova/boost-and-broadside` on branch
`perf/realtime-experiment-handoff` until the hard deadline
**2026-09-20 22:00 Asia/Seoul**. Deliver correctness-checked, measured
improvements to the 50v50 Frontline match with two policies and a clear decision
report. Continue independently while useful authorized work remains, but finish
early if the deliverables are complete.

Check the clock on startup; do not launch experiments after the deadline. This
branch is already prepared and committed. Check its status before editing; if
you use worktrees, base them on this branch and include any later user changes.

## Rules and starting context

Never alter power profiles, CPU governors, GPU power limits, sleep/keep-awake
settings, or any other power-saving settings. The machine is plugged in and
intensive experiments are authorized under its current settings. Do not treat
this authorization as permission to change those settings.

Start by reading:

- `docs/engineering/performance-audit/README.md` for the result index and exact
  commands;
- `docs/engineering/realtime-and-training-audit.md` for interpretation;
- `docs/engineering/performance-audit/results/` and
  `docs/engineering/performance-audit/logs/` when a result needs
  verification.

The archived audit used an RTX 4070 Laptop (8 GiB), PyTorch 2.13.0+cu130, and
current power-saver settings. Its key evidence: at 50v50, a complete two-policy
CUDA frame was about 200 ms; render time was about 92--109 ms; CPU projectile
visibility dominated CPU observation assembly; a compiled real
visibility kernel was roughly 20x faster in isolation on CUDA but did not make
complete frames faster. Treat those as hypotheses and baselines, not proof that
any proposed change works. The current compatible runtime observation schema is
`frontline_shields_v9`; do not silently load incompatible checkpoints.
CUDA was available outside the sandbox even when sandbox checks returned false.
Use a permitted GPU execution path; do not disable security controls globally.

Preserve user changes and keep the reference behavior runnable. Make local
checkpoint commits for your own reviewed changes on this experiment branch;
do not publish or merge to another branch. Do not use destructive git
commands. Do not run overlapping CPU-heavy or GPU-heavy measurements.

## Team and coordination

Act as the high-capability orchestrator (Astra/high where selectable). Delegate
bounded work: Terra/medium for numerical or renderer code; Luna/medium for
mechanical edits, short tests, result extraction, and documentation; Sol/high
only for a well-described blocker after two focused attempts. There are four
slots total, including you. Give every worker explicit file ownership and a
concrete completion criterion. Review their diff and evidence yourself.

Use one project-wide benchmark lock. Workers may inspect code and run cheap
tests concurrently, but compilation, benchmarks, GPU work, and CPU-heavy builds
must be serialized. Save concise artifacts, not raw terminal dumps.

At every task boundary update `artifacts/deadline-experiment/WORKLOG.md` and
`state.json` atomically with owner, status, revision/diff, next exact command,
result paths, active processes, and blockers. Reconcile state before retrying on
resume. Account quota and reset timing are unknown: use bounded work chunks and
save progress before long runs. If actual telemetry is available, reserve about
20% of remaining allowance for integration/reporting. Otherwise report usage as
unknown; elapsed time is not quota consumption. Workers share usage constraints.
Do not purchase credits, change accounts, or enable paid API fallback.

Before depending on unattended continuation, spend at most 30 minutes establishing
a supported local restart mechanism, if available. No scheduler is preinstalled
for this task. Prefer a supervisor that saves the exact session ID, uses a singleton
lock, resumes only this task after the reported quota reset, and enforces the
deadline outside the model. Respect weekly limits too. Where supported, the CLI
can resume an explicit session with `codex exec resume <SESSION_ID>`. Test recovery
with dummy commands; avoid duplicate agents or busy model polling. If no supported
mechanism can be established, report that limitation explicitly and continue with
durable checkpoints; do not claim that a prompt guarantees automatic recovery.

## Objective and bounded work

Improve the complete path, not a benchmark fragment. The priority order is:

1. Establish a small reproducible baseline and deterministic parity fixtures.
   Reuse the archive rather than repeating the whole audit.
2. Refactor one substantial simulation/observation region into explicit tensor
   state inputs and outputs. Compare eager and `torch.compile` around that
   region; remove whole categories of needless dispatch, allocations, and
   bookkeeping unnecessary for interactive play. Preserve simulation rules, policy observation,
   action buffering, recurrent belief semantics, reward behavior where required,
   reset, and termination.
3. Add only exact, defensible visibility/geometry pruning. Prioritize inactive
   projectiles and provably irrelevant pairs. Retain a reference path and prove
   event/observation parity before timing. Do not count an isolated LOS win as a
   game win.
4. In a separate implementation track, prototype a GPU render path with a
   Pygame window/input integration and ModernGL. Build a compact immutable render
   snapshot, batch ships and projectiles, and provide a GPU fog pass that retains
   team visibility and toroidal camera behavior. First make the information
   correct and measurable. A stylized representative scene is welcome only after
   the performance path works.
5. Integrate only candidates that pass parity and show a repeatable end-to-end
   gain. If a track stalls, preserve its boundary and findings, then move to the
   next highest-value item.

Time-box an initial simulation approach to four focused hours before reassessing.
If whole-step compilation stalls, retain useful smaller boundaries and optimize
lean inference or exact pruning. If GPU renderer setup stalls, preserve a compact
snapshot interface and improve current projection/allocation handling. A complete
compiled tick and final artwork are stretch goals, not prerequisites for a useful
measured result. Develop simulation and renderer code independently, but serialize
resource-heavy validation.

Do not undertake broad engine or language migrations, policy retraining, agent
cadence changes, long art production, packaging work, or an unbounded simulator
rewrite in this deadline experiment.

## Correctness and measurement protocol

Use fixed seeds and recorded action traces. Before a long compile or benchmark,
run cheap differential cases that cover toroidal seams, projectile ring reuse,
collision ties, shields, destruction/respawn, reset, and termination. Require
exact discrete events and documented float tolerances. Keep a current reference
implementation available.

Screen with a short alternating-order A/B comparison. Promote a candidate only
after parity and at least three alternating-order pairs. Record command, git
revision and dirty state, software/device/power state, model/config, RNG trace,
compiler settings, warmup, sample count, startup cost, render resolution, and
timeouts. Include two distinct policies in the 50v50 end-to-end measurement;
label random-policy results plainly.
Set per-process compiler/benchmark timeouts before launching, preserve partial
logs, and avoid repeating completed tests without a new change or hypothesis.

Report p50/p95/p99 and deadline misses for complete frames, along with phase
attribution. Do not add separately measured phase medians. Render benchmarks
must state whether they include presentation/flip and fog. For training, screen
environment-only throughput at B=1, 32, 128, and one VRAM-fitting larger batch,
including reset work. Run a bounded rollout/PPO A/B only for an integrated winner
with enough remaining time; otherwise label full-training benefit unproven.

## Time gates and final deliverables

At 20:00 KST stop starting new implementation experiments. By 21:30 KST finish
integration, final checks, raw artifacts, and the report. At 22:00 KST terminate
only experiment processes owned by this task, persist final state, and stop.

Write `docs/engineering/deadline-experiment-results.md`. Include a table with:
candidate, scope, parity result, complete-frame latency, rendering result,
environment throughput, full-training result if measured, startup cost,
limitations, and keep/reject/defer decision. Link every raw artifact and exact
command. Distinguish 30 Hz simulation from 60 FPS interpolated presentation.
State honestly if no configuration reaches 30 Hz.
Keep new raw results under `docs/engineering/performance-experiments/` for the final
handoff; use the ignored `artifacts/deadline-experiment/` directory for scratch work.
Include observed model usage when available; do not invent costs or probabilities.

At handoff, leave successful changes reviewable, failed experiments isolated or
reverted without touching user work, and `WORKLOG.md` sufficient for another
agent to resume safely.
