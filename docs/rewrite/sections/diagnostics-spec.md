# Diagnostics spec (section 9)

Status: spec, 2026-11-01. Decisions: `docs/rewrite/diagnostics.md` (the audit, complete
2026-10-22: the generic checklist, groups 1–20, the ungrouped items, the trigger rules
and the dashboards). That audit is the normative list of series. This spec says how v2
produces, reads, checks and shows them. **(v2)** marks a translation of an audit item to
the v2 stack. **(proposal)** marks a choice the spec makes.

Order: after sections 6, 7 and 8, so every source exists. Groups whose source does not
exist yet (for example the play shell's replay viewer) are built to the data format and
listed.

## 1. Collection

- **`bnb/runtime/metrics.py`:** the collector.
  - Every producer (game analytics in the rollout, the learner, the BC learner, the
    league, the model's heads) returns its series as **device tensors** in a flat
    mapping, `{"group/quantity/slice": tensor}`.
  - The collector accumulates them on the device over an update.
  - It reads them back **once per update** (and once per diagnostic cadence for the
    cadence groups), as one stacked transfer.
- **The method rules** (audit, "Rules and method") are requirements:
  - no host synchronisation per step or per minibatch; the per-epoch KL read for early
    stopping is the one exception;
  - raw values, never pre-smoothed (the justified EMAs are the return scale's state and
    the live rating);
  - names are quantity first (`{group}/{quantity}/{slice}`), full words.
- **Cadences:**
  - per update (U): everything cheap;
  - every 10 updates (C): histograms' replacements, `repr/`, the gradient decomposition,
    the actor/critic split;
  - offline (O): reports built after training.

  The gradient decomposition runs as its own compiled function at the cadence only, so
  the normal update keeps its graph **(v2: the audit's JAX "separately jitted
  function")**.
- **Producers own their series.** Each section's module that computes a quantity also
  emits its metrics; the collector knows nothing about them. A new series needs no change
  to the collector.

## 2. Sinks

- **W&B**, project `boost-and-broadside-v2` (author, 2026-11-02), sampled history as
  before. Tests and offline runs
  use W&B's offline mode or a local JSONL sink; no test touches the network.
- **A local JSONL file per run** with the same records, so any analysis can run without
  W&B.
- **The fired-events table** (rule, update, value, threshold, the panel to open) goes to
  both.

## 3. Trigger rules

The final rule list in the audit ("Final trigger rules"), as written:
- **The alerts:** skipped step, recompile, memory, C4 mismatch, non-finite input,
  never-seen ship, throughput, rating drop; and since 2026-11-13 the combat collapse per
  fighting mode (team deathmatch, elimination; the audit's "Final trigger rules").
- **The warnings:** entropy collapse, critic useless, support clipping, envelope
  violation, representation collapse, dormant units, update ratio, label validity,
  sampling collapse, reward hacking, head worse than trivial, sigma miscalibrated, side
  imbalance.

How they run:
- **On the host,** on the numbers the per-update readback already holds: no device work
  and no extra synchronisation.
- **Rule kinds:** invariant, anomaly (z-score against the run's own EMA, or a drop from
  the running peak) and trend (a slope over a window).
- **Against noise:**
  - warm-up suppression;
  - hysteresis (k consecutive updates);
  - a cooldown per rule.

  The thresholds are placeholders, tuned after the first runs.
- **Rule state** (EMAs, peaks, cooldowns) is saved in the checkpoint, so a resume does
  not re-arm everything.
- **Outputs:** `alerts/{rule}` (0 or 1 per update), `alerts/active` (the count), and the
  fired-events table.
- **Only the skipped-step rule acts:** a checkpoint plus the batch dump (section 6's
  non-finite handling). Everything else reports. Dashboard only; no `wandb.alert`.

## 4. Dashboards

- **The three views** of the audit ("Dashboard layout"):
  - the 12-panel headline;
  - the curated view, about 141 panels in open and collapsed sections, with a `bc/`
    section for BC runs;
  - W&B's automatic workspace.
- **Defined in code** with `wandb-workspaces` and versioned in the repo
  (`bnb/runtime/dashboards.py`, or `tools/`). A command builds or updates a run's
  workspace. If that package's API proves unstable, the fallback is one saved workspace
  made by hand, with its export committed (author, 2026-11-02: in code if it can be
  done).
- **The x-axis** is `counters/env_steps` by default.

## 5. v2 translations of audit items (v2)

- **No host syncs (A6):** a test runs the update under
  `torch.cuda.set_sync_debug_mode("error")`, allowing only the per-epoch KL read. The
  rollout already has the CUDA-graph capture test.
- **Determinism (A8):** a test checks that the same seed gives the same first updates.
  - Bitwise on the CPU, where it costs nothing.
  - On the GPU within a float tolerance, with integers and categoricals exact (author,
    2026-11-03: the run's integrity within rounding, not its bits; contracts, "Resume:
    integrity, not bits").
- **Resume (A9):** stopped and resumed equals uninterrupted (section 8 covers the
  league's part).
- **The action metrics** use the joint 42-way action. Factor entropies are the joint's
  marginals (power 3, stick 7, shoot 2).
- **Per pool and mode:** episode, reward and rating series are sliced by the pool's game
  (Frontline, team deathmatch squad, battle map) and fog. The batch-composition series
  (`batch/token_share/{mode}`, `{opponent_kind}`) count learner tokens.
- **The belief and next-state series** are in the model's frame (the per-team position
  offset applied). Errors are differences, so the offset cancels; the check is that no
  series reports an absolute position.
- **Absent zones:** zone-dependent series cover present zones only.
- **Shot series:** every age, as section 6 labels them.
- **The replay of one environment every N updates (H1):** the state trajectory in the
  format the play shell (section 10) reads. The viewer itself is section 10's.

## 6. Acceptance

1. **Every series against a hand-computed case:** a small constructed batch, rollout or
   update with a known answer, for every group. Templated series need one case per
   template.
2. **The method rules:**
   - the no-sync test (section 5);
   - a test that every logged name is quantity-first and in full words, checked against
     a list of the allowed abbreviations;
   - a test that no logged series is an EMA except the justified ones.
3. **Trigger rules:** each fires on a synthetic series that should trip it and stays
   quiet on one that should not, with warm-up, hysteresis and cooldown. Rule state
   survives a checkpoint round trip.
4. **The skipped-step action:** a constructed non-finite step writes the checkpoint and
   the dump, and training continues.
5. **Dashboards:** the code builds the headline and curated views for a test run (W&B
   offline, or the saved-workspace fallback).
6. **Cost:**
   - the per-update series add at most 2% to update time;
   - the cadence groups, amortised over 10 updates, at most 5%.

   Report both at the 5v5 training cells with Yemong. **These are soft caps (author,
   2026-11-02):** measured and reported, never a reason to drop a useful diagnostic. A
   series that cannot be made cheaper may go over, if the notes say what it costs and
   why it stays. Faster is still better.
7. **The audit's coverage:** a table in the notes lists every series of the audit,
   whether it is built and by which producer, or why not (its source does not exist).
