# Gradients 11 progress (queue item 11, branch `gradients-11`)

Author decisions of 2026-11-28 ~15:30: the entropy bonus 0.003 in BC too; port v2's gradient
decomposition to lean; run the diagnostic (BC ~6 updates from fresh weights, then ~3 RL
updates). GPU jobs are `bnb-11-*` units under the GPU lock, one at a time
(`gradients-11/run.sh`); logs in `data/gradients11/logs/` (not committed).

## 2026-11-28

- ~15:35 code in (89172b04): entropy in BC (`update`'s BC weights), `lean.learn.gradients`
  (off by default, `LearnConfig.gradient_diagnostics_every`), `minibatch_loss(parts=True)`,
  `training.loss_inputs` (the loop's argument building, shared with the diagnostic), the trainer's
  gradient lines. Found and fixed: `geometry.draw_keys` compared `cuda` with `cuda:0` as two
  devices, so on the GPU the pair keys came from torch's default CUDA generator, not the update's
  (`same_device`). New tests (test_gradients.py) pass, 8.
- 15:37 diagnostic started (`bnb-11-diag`, load 0.52, performance profile): `lean.train --stage
  all --bc-updates 6 --updates 3 --gradient-diagnostics-every 1` (RL continues from the BC
  stage's EMA hand-off, as a real run). ETA 16:20 (worst 16:50): compile ~3 min, ~110 s per
  update plus the eager pass (not yet measured).
- 15:40 quick tier lean/learn + model + eval: 199 passed (during the diagnostic's compile).
