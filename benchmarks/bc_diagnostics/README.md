# BC turn-head diagnostics

Throwaway-grade probes written for the investigation in
[`docs/internal/bc-turn-head-investigation.md`](../../docs/internal/bc-turn-head-investigation.md).
They build a `bc`-profile trainer at a checkpoint, collect rollouts with the Elo
evaluator stubbed out, and measure the behaviour-cloning KL from several angles.
They are diagnostics, not a supported CLI: paths and the checkpoint are set in
`harness.py`.

```
uv run --no-sync python benchmarks/bc_diagnostics/exp1_reconcile.py 128
uv run --no-sync python benchmarks/bc_diagnostics/exp2_drift.py 12     # writes dump_rollout11.pt
uv run --no-sync python benchmarks/bc_diagnostics/exp3_turn.py         # reads that dump
uv run --no-sync python benchmarks/bc_diagnostics/exp4_grad.py
uv run --no-sync python benchmarks/bc_diagnostics/exp5_ns.py
uv run --no-sync python benchmarks/bc_diagnostics/exp6_angle.py        # reads that dump
```

`BC_DIAG_DIR` redirects where dumps are written and read (default: this
directory). The per-token dump is ~18 MB and is not committed.

Any offline BC evaluator must reduce over `bc_valid & actor_mask & alive`.
Dropping `actor_mask` under `paradigm=ego_pass` doubles the denominator with
opponent-team tokens and inflates the reported KL by about 1.9×.
