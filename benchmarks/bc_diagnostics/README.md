# BC turn-head diagnostics

Throwaway-grade probes written for the investigation in
[`docs/internal/bc-turn-head-investigation.md`](../../docs/internal/bc-turn-head-investigation.md).
They build a `bc`-profile trainer at a checkpoint, collect rollouts with the Elo
evaluator stubbed out, and measure the behaviour-cloning KL from several angles.
They are diagnostics, not a supported CLI: paths and the checkpoint are set in
`harness.py`.

## Session 1 — what the residual is

```
uv run --no-sync python benchmarks/bc_diagnostics/exp1_reconcile.py 128
uv run --no-sync python benchmarks/bc_diagnostics/exp2_drift.py 12     # writes dump_rollout11.pt
uv run --no-sync python benchmarks/bc_diagnostics/exp3_turn.py         # reads that dump
uv run --no-sync python benchmarks/bc_diagnostics/exp4_grad.py
uv run --no-sync python benchmarks/bc_diagnostics/exp5_ns.py
uv run --no-sync python benchmarks/bc_diagnostics/exp6_angle.py        # reads that dump
```

## Session 2 — where it comes from

Collection takes `<tag> <n_rollouts> <n_burn_in> <seed>`. Three independent tags
are used so that no probe or head is ever scored on a state it was fitted on:
`train` fits the probes, `heldout` reports them, `heldout2` reports anything that
had to spend `heldout` fitting a turn head.

```
uv run --no-sync python benchmarks/bc_diagnostics/exp7_collect.py train    3 3 0
uv run --no-sync python benchmarks/bc_diagnostics/exp7_collect.py heldout  3 3 1
uv run --no-sync python benchmarks/bc_diagnostics/exp7_collect.py heldout2 3 3 2
uv run --no-sync python benchmarks/bc_diagnostics/exp7_probe.py frontline   # layerwise bearing probes
uv run --no-sync python benchmarks/bc_diagnostics/exp7_probe.py personal
uv run --no-sync python benchmarks/bc_diagnostics/exp8_oracle.py            # oracle bearing + error->KL
uv run --no-sync python benchmarks/bc_diagnostics/exp9_collect_obs.py train   3 3 0
uv run --no-sync python benchmarks/bc_diagnostics/exp9_collect_obs.py heldout 3 3 1
uv run --no-sync python benchmarks/bc_diagnostics/exp9_repr.py              # absolute vs ego-relative
uv run --no-sync python benchmarks/bc_diagnostics/exp10_deepsets.py         # sum-pooling probe
uv run --no-sync python benchmarks/bc_diagnostics/exp11_intervention.py     # predicted bearing -> KL
uv run --no-sync python benchmarks/bc_diagnostics/exp12_calib.py            # how accurate must it be
uv run --no-sync python benchmarks/bc_diagnostics/exp13_side.py             # left/right asymmetry
```

`exp7_collect.py` supersedes `exp9_collect_obs.py` — it dumps the raw observation
channels alongside the activations, so the two are guaranteed to describe the
same rollouts. `exp9_repr.py` still reads the older `obs_*.pt` layout.

Each `exp*_rows.json` is the raw output backing a table in the write-up and is
committed. The `.pt` dumps are not: `probe_*.pt` is 213 MB each and `obs_*.pt`
25 MB. `BC_DIAG_DIR` redirects where all of them are written and read.

## Two things that will silently ruin a measurement

* Any offline BC evaluator must reduce over `bc_valid & actor_mask & alive`.
  Dropping `actor_mask` under `paradigm=ego_pass` doubles the denominator with
  opponent-team tokens and inflates the reported KL by about 1.9×. The cheap
  check: the valid denominator must equal `num_steps × num_envs × num_ships / 2`.
* Temporal sublayers are reached through `forward_sequence`, not `forward`, so a
  `register_forward_hook` never fires on them, and in sequence mode they emit
  `(B·N, T, D)` where the spatial sublayers emit `(T·B, N+M, D)`. Reshaping one
  as the other misaligns states against labels, and a probe reports that as "this
  layer carries no information" rather than failing. `exp7_collect.py` handles
  both.
