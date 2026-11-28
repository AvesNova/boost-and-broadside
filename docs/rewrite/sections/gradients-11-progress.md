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
- 16:01 diagnostic done (success). Per update with the eager pass on: ~135-140 s (BC
  7,459-7,835 steps/s after the first; without the pass ~108 s at fixes-10's numbers, so the
  pass costs about 30 s; not a quiet-machine timing). Peak 1.74 GiB. BC bc_kl 2.39 -> 0.81 over
  6 updates; RL teacher weight 0.5 / 0.475 / 0.451. The KL warning fired on BC updates 1-5 and
  RL 0-2 (BC update 4: epochs 0.07 / 0.28 / 0.43 / 0.57). Log copied to
  `gradients-11/diag-run1.log`; tables (`gradients-11/tables.py`) in `gradients-11/tables.md`.
- 16:01-16:06 compile check (cold, load 6.6 at start: the diagnostic's own tail): 6 graphs, 0
  graph breaks, no recompile, minibatch_loss 1,645 traced operations (AOT forward 2,086,
  backward 2,223: equal to the baseline), total 11,509; cold 168.6 s (budget 175 s): within the
  baseline.

### Diagnostic results (numbers only; per-update tables in gradients-11/tables.md)

Trunk = `encoder` + `blocks`, 1,829,952 parameters (of 2,269,524). Shares are of the raw
gradient before the clip, on the first minibatch (384 views, two halves of 24 micro-batches).

BC, means over updates 0-5 (trunk share / trunk norm / cosine with bc / coherent fraction):

| term | trunk share | trunk norm | cos with bc | coherent | model share |
|---|---|---|---|---|---|
| bc | 0.929 | 1.25e-01 | 1 | 0.927 | 0.984 |
| entropy | -0.002 | 2.36e-03 | -0.453 | 0.995 | -0.005 |
| value | 0.008 | 1.31e-02 | -0.152 | 0.990 | 0.010 |
| next_state | 0.010 | 1.31e-02 | -0.119 | 0.983 | 0.004 |
| hit | 0.000 | 1.93e-04 | 0.097 | 0.689 | 0.000 |
| pair | 0.004 | 6.98e-03 | -0.314 | 0.931 | 0.000 |
| field | 0.054 | 5.28e-03 | -0.273 | 0.886 | 0.002 |
| edge | -0.003 | 1.29e-02 | -0.135 | 0.964 | 0.005 |
| surrogate | 0 | 0 | n/a | n/a | 0 |

RL, means over updates 0-2 (cosine with the surrogate):

| term | trunk share | trunk norm | cos with surrogate | coherent | model share |
|---|---|---|---|---|---|
| surrogate | -0.017 | 4.06e-03 | 1 | 0.028 | -0.001 |
| teacher | 0.817 | 1.57e-01 | -0.103 | 0.990 | 0.857 |
| entropy | -0.006 | 5.50e-04 | 0.119 | 0.997 | -0.004 |
| value | 0.050 | 4.16e-02 | 0.076 | 0.995 | 0.055 |
| next_state | 0.084 | 3.80e-02 | 0.079 | 0.996 | 0.036 |
| hit | 0.001 | 2.37e-04 | 0.021 | 0.942 | 0.000 |
| pair | 0.028 | 1.61e-02 | 0.005 | 0.995 | 0.007 |
| field | 0.014 | 1.67e-02 | 0.024 | 0.998 | 0.019 |
| edge | 0.029 | 2.18e-02 | 0.023 | 0.984 | 0.031 |

Value by level (trunk share, RL updates 0 / 1 / 2): damage 0.034 / 0.026 / -0.019, kill_death
0.042 / 0.033 / -0.023, outcome 0.047 / 0.037 / -0.028; trunk agreement within value
0.997 / 0.996 / 0.989. BC: each level 0.004-0.029 or slightly negative, agreement 0.61-0.995.
Next state by column (coherent share of the next-state term, RL means): shield ~0.64, energy
~0.17, along / across ~0.07-0.08, speed ~0.02, heading ~0.01 (BC similar, shield 0.45-0.77 on
updates 0-4, energy 0.63 on update 5). Pair by target (coherent share, RL means): line of sight
rate ~0.45, gun error ~0.22, range ~0.17, range rate ~0.09, energy difference ~0.07, train
angle ~0.001. Actor share in the trunk: BC 0.65-0.94, RL 0.82 / 0.72 / 0.79.

Observations against the brief's flags:
- Over ~50% of the trunk step: the policy-side teacher term in every update (BC's bc 0.43-1.24,
  its own stage's policy term; RL's teacher 0.64 / 0.63 / 1.18). No auxiliary term over 50% in
  any update; the largest single one is field 0.345 in BC update 2.
- The surrogate in RL: trunk share -0.017 mean, norm 4e-03 (teacher 1.6e-01, about 40x), coherent
  fraction -0.05 / 0.00 / 0.13 (the halves' surrogate gradients nearly orthogonal).
- Negative cosine with the policy term: in BC, pair negative in 6 of 6 updates (-0.03 to
  -0.67), field 5 of 6, entropy 4 of 6 (-0.70 to -1.00 on updates 1-3, 5), value and
  next_state 3 of 6; on BC updates 2 and 4 nearly every auxiliary part is at -0.3 to -0.75 with
  bc. In RL the teacher's cosine with the surrogate is 0.11 / 0.22 / -0.64; no auxiliary term
  negative in more than one of the three updates.
- Not measured: anything beyond 6 + 3 updates from fresh weights; no EMA of the cosines (Du et
  al.'s caveat: single-minibatch cosines are noisy).
