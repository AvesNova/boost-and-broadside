# Config, checkpoint, and evaluation refactor plan

Working plan for the config-system rewrite, the resume fix, and the
`good-leaf-719` headline evaluation. Internal: it records reasoning,
measurements, and sequencing, not reader-facing documentation. Reader-facing
material lives in [training.md](../training.md).

Written so a fresh session can pick this up without re-deriving anything.
Current as of `fb34a0e`, branch `feature/gradient-diagnostics`.

**Read [Findings](#findings) before starting any step.** Several steps exist
because of specific measured defects, and the evidence is not obvious from the
code alone.

---

## State

| | |
|---|---|
| Branch | `feature/gradient-diagnostics` (pushed) |
| HEAD | `fb34a0e` Raise the RL learning rate peak to 4.5e-4 |
| Run 719 | `good-leaf-719`, 992M/1000M steps, update 997/1004, live Elo ~1730 |
| Run 716 | `lunar-cosmos-716`, complete, 500M steps |
| Run 682 | `resilient-resonance-682`, current headline; 7 artifact kinds, 24M |
| GPU | single RTX 4070 Laptop, 8188 MiB; a full-width run holds ~7200 MiB |

## Sequencing rule

The **evaluation campaign runs before the config refactor.** The refactor
deletes profiles and rewrites the resolver, which the evaluation modes depend
on; landing it mid-campaign risks breaking the tooling exactly when it is
needed. Bank the artifacts first.

---

## Steps

### 1. Resume fix + provenance hygiene — CPU — **DONE** (`c090f88`, `ea78e93`)

- [x] Split a pure `_apply_schedule_state(step)` out of `_refresh_training_schedule`
      (resolve schedule at step, compute `bc_factor` from the restored eval
      window, set the three coefficients, set optimizer LR explicitly, apply
      group scales to component weights + `refresh_component_weights()`)
- [x] Call it at the end of `load_checkpoint`
- [x] Regression test: drive BC to zero, save, load into a fresh trainer,
      assert coefficient-for-coefficient equality. **Must fail before the fix.**
- [x] Test that the first post-resume update is in-family (`bc_loss == 0`, KL normal)
- [x] `run.json` records git SHA + dirty flag
- [x] A hard crash records `status: failed` (today it still says `running`)

All three resume tests were confirmed failing before the fix and passing after;
they live in `tests/train/test_checkpoint.py::TestResumeRestoresScheduleState`.
Full suite 1575 passed, `bnb smoke` 17/17.

Blocks: step 9.

### 2. 719 completes

- [ ] `status: complete`, final checkpoint written, GPU free

### 3. Evaluation campaign — GPU, ~1 day

In this order:

- [ ] `elo-calibrate` on 719 (required for any headline claim)
- [ ] `elo-calibrate` on 716 (for the live-vs-calibrated bias question)
- [ ] `elo-scale` on 719
- [ ] `crossover` on 719
- [ ] `semi-random` ladder on 719
- [ ] `ar-report` on 719
- [ ] `noise-calibration` on 719
- [ ] `wandb-export` on 719

All land in `checkpoints/good-leaf-719/artifacts/` (already the default —
`ArtifactStore` routes run-scoped artifacts there). **682's artifacts must not
be touched.**

Done when: 7 artifact kinds present for 719, plus 716 calibration.
Blocks: steps 4, 7.

### 4. Comparison + decision — needs user

- [ ] Write up 719 vs 682: calibrated Elo, scale, crossover, and the config
      deltas that might explain the gap. State descriptively — the runs differ
      in fields, reward schema, LR and budget, so nothing here is causal.
- [ ] Live-vs-calibrated bias verdict across 716 and 719: consistent or not,
      and whether it is usable for ablation ranking / early stopping
- [ ] **USER DECISION:** replace 682, or present both as a comparison

Blocks: step 7, and step 9's choice of objective.

### 5. Config refactor — CPU, the large block

Each sub-step independently shippable; run suite + smoke after each.

- [ ] **5a** Drop non-field support; merge `rl`/`rl-fields`; BC becomes an
      overlay; hoist the 19 duplicated values into one base.
      Deletes: `profiles/rl.py`, `profiles/bc.py`, `make_bc_schedule_spec`,
      `tests/config/test_bc_profile.py`, `test_rl_fields_resolved_diff_…`,
      the `REWARDS`/`FIELD_REWARDS` split, `field_map is None` branches
- [ ] **5b** Collapse Spec + Config into one schema with explicit intent/derived
      field pairs (`gamma_per_tick` stored, `gamma` derived). Derivation becomes
      `derive(config) -> config`. Deletes most of `schema.py` and much of
      `resolve.py`
- [ ] **5c** Flatten schedules to `[step, value, interp]` keypoint tables
      (verified bit-identical on the real LR schedule — see Findings)
- [ ] **5d** `checkpoints/<run>/config.json` holding intent + source + overrides +
      code provenance; written once, never rewritten; resume reads it
- [ ] **5e** Positional `key=value` overrides; `--continue RUN`;
      `--from RUN [--at STEP]`. Unknown path / derived field / type mismatch → error
- [ ] **5f** Delete the fingerprint superstructure: drift guard,
      `--allow-config-drift` on the training path, the fingerprint pin test,
      the S01 snapshot tests, `_INTENDED_DIVERGENCE`

**Keep:** `canonical_json` (needed to save and diff at all), the
feature/physics guard in `policy_io` (different job — protects evaluation
correctness), a local `hash(dict)` inside `vram.py` for the cache key.

Done when: changing a config value is one edit.
Blocks: step 6.

### 6. Migrate 682 and 719 configs — CPU

- [ ] One-time script emitting new-format `config.json` for both runs
- [ ] **Preserve absence, do not backfill.** 682 genuinely had no
      `field_damage_taken_weight` and no `total_timesteps`
- [ ] Regression test loading 682, 716, 719

Precedent exists: `scripts/migrate_682.py`, `tests/migration/`.

### 7. Docs — after step 4

- [ ] Hand-written, including the analysis prose. Scope depends on the step 4
      decision.
- [ ] Repoint `docs/publications.toml` at the chosen run's **figures** artifact.
      Charts now render per-run (`bnb figures --run RUN`, see `figure_set.py`),
      so this edit is only the choice of which run illustrates the docs. 719's
      set is already rendered at
      `checkpoints/good-leaf-719/artifacts/figures/20260821T040737Z-4da049c9`.

### 8. Docs check — optional, recommended

- [ ] `<!--cfg:run=… path=…-->` annotations on config-derived numbers
- [ ] One test resolving each against the named run's stored config, read as
      **data** (dict lookup, not dataclass rebuild — 682 predates the current
      schema). A missing path is a loud failure, and that is the feature.

~40 lines plus annotations. Guards against docs drifting toward `main` on
ordinary config edits, which is frequent, rather than against headline changes,
which are rare.

### 9. Ablation + sweep infrastructure

- [ ] Seed the global numpy RNG (minibatch order) — paired comparisons are
      currently impossible
- [ ] Short proxy config, validated against a result already known
- [ ] Objective from step 4's verdict (post-hoc calibration, or live Elo if it
      proves reliably biased)
- [ ] Local job runner (one GPU = depth-1 queue)
- [ ] W&B Sweeps with flattened `wandb.config`; positional `key=value` matches
      `${args_no_hyphens}` natively. Launch queue only if moving to cloud.

---

## Findings

Evidence behind the steps, so a fresh session does not re-derive it.

### The resume bug (step 1)

`load_checkpoint` restores weights, optimizer, scalers, avg model, eval windows
and step counters, but **not** the schedule-derived state — `_schedule_state`,
`_behavior_cloning_coef`, `_scripted_win_rate`. Those keep the values `__init__`
computed from `_resolve_schedule(schedule, 0)`, i.e. **step zero**. And `train()`
calls `_update_epochs` *before* `_refresh_training_schedule`, so the first update
after any resume uses step-0 coefficients.

Two consequences. BC's effective weight is `schedule × max(0, 1 −
win_rate/0.45)`; the decay factor is derived from the eval window and is not
restored, so a run that had correctly decayed BC to **0.0** resumes at **2.0**.
And `_scripted_win_rate = 0.0` makes `_effective_target_kl()` read the loose
0.1 instead of the tightened 0.02, so the KL gate lets the bad update run
further.

Measured on run 717, update 148 being the first post-resume update:

| | 147 | **148** | 149 |
|---|---:|---:|---:|
| `bc_loss` | 0 | **2.231** | 0 |
| `policy/kl` | 0.0085 | **1.462** | 0.008 |
| `gradient_norm` | 0.79 | **9.56** | 0.96 |
| `loss/total` | 0.074 | **5.022** | 0.227 |
| scripted win rate | 0.87 | 0.86 | **0.19** |
| live Elo | 1316 | 1390 | **908** |

~400 Elo lost, ~20 updates (2h) to recover, never fully back to trend. Arithmetic
confirms the mechanism: `bc_winrate_target = 0.45`, and update 149's logged
`bc_factor` of 0.5778 is exactly `1 − 0.19/0.45`.

Only misfires on resume — on a fresh run BC *should* start at full strength,
which is why nothing catches it normally. **Still required after the config
rewrite**: this is derived runtime state, not a config value.

### Profiles vs defaults is inverted (step 5a)

19 values are byte-identical across all three profiles and duplicated in each:
`clip_coef`, `max_grad_norm`, `return_ema_alpha`, `return_min_span`,
`advantage_min_rms`, `return_quantile_samples`, `histogram_interval`,
`log_interval`, `league_size`, `league_slots`, `elo_temperature`,
`elo_milestone_gap`, `bc_winrate_target`, `num_steps`, `num_minibatches`,
`logical_batch_tokens`, `action_repeat`, `num_ships`, `checkpoint_dir`.

Meanwhile `defaults.py` holds `FIELD_REWARDS` and `make_bc_schedule_spec`, each
used by exactly one profile.

`rl` vs `rl-fields` differ in 8 leaves, of which `num_envs` (3904 → 2592) is
*derived* from the token budget and `total_timesteps` is a recent edit. Dropping
non-field support merges them completely. There is no such thing as a model that
"doesn't support fields" — fields are extra non-recurrent tokens in `N+M`, and
`num_fields=0` is a degenerate configuration, not an architecture variant.

`bc` vs `rl` differ in 7 values: `next_state_coef`, `total_timesteps`, and four
schedule entries (`policy_gradient_coef` 1→0, `behavior_cloning_coef` 2→1,
`league_fraction` 0.5→0, `target_kl` 0.1→None) plus a differently-shaped
`learning_rate`. As an overlay, `tests/config/test_bc_profile.py` (181 lines)
becomes unnecessary — it exists solely to police "BC differs from RL only in
these named ways", which an overlay guarantees by construction.

### Two schemas, three edits per hyperparameter (step 5b)

`ProfileSpec` + 6 sub-specs on one side, `TrainConfig` + 5 configs on the other,
with `resolve.py` (556 lines) translating. Adding one hyperparameter means
editing the Spec, the resolver line that copies it, and `TrainConfig`. Only
about five fields are genuinely derived (the discount pair and two component
tables, `num_envs`, `rollouts_per_update`, `microbatch_tokens`); the rest are
passthroughs.

### Flat schedules are behaviour-preserving (step 5c)

The real LR schedule as `[step, value, interp_to_next]`:

```
[[0, 1e-7, "linear"], [5_000_000, 4.5e-4, "hold"],
 [100_000_000, 4.5e-4, "exponential"], [500_000_000, 1.5e-4, "hold"]]
```

Verified against the compiled schedule at 10 probes including boundaries and
past-the-end clamping: `max |flat − current| = 0.000e+00`. `join` was only ever
concatenation — every segment starts at its own activation step — so the nesting
layer disappears. `constant`/`linear`/`stepped`/`exponential`/`cosine`/`join`
collapse to one structure with an interpolation tag.

Note schedules are **pure functions of `global_step`** — there is no schedule
state to save. Save the definition, evaluate at the restored step.

### JSON vs YAML (step 5d)

JSON as the artifact of record; render to YAML/markdown only for display, never
round-trip. YAML 1.1 parses `1e-7` as a **string** (needs `1.0e-7`), and `no`
as `False`; the configs are full of scientific-notation learning rates. JSON has
one canonical byte form, which matters because these get diffed.

### The docs already contradict the figures (steps 7, 8)

Every published chart comes from 682. What 682 actually trained with, against
what `docs/training.md` currently claims:

| Component | docs say | 682 actually |
|---|---:|---:|
| `ally_win` | 1.5 | **4.0** |
| `shoot_quality` | off | **0.1** |
| `combat_damage_taken` | 0.5 | **did not exist** |
| `field_damage_taken` | off / 0.5 | **did not exist** |
| `field_death` | off / 1.0 | **did not exist** |

682 had no fields at all, and predates `resolved_config` entirely (its
checkpoint has `obstacle_cache`, no `total_timesteps`). The prose has been
tracking `main` while the figures stayed pinned to the run. Editing the
`shoot_quality` row to "off" on 2026-08-18 moved the docs *further* from the run
they illustrate.

### VRAM (step 3)

A full-width run holds ~7200 MiB of 8188, leaving ~560 MiB free — not enough for
a second CUDA process (context alone is ~300 MiB, and calibration defaults to
16384 parallel games). Evaluation requires the GPU essentially alone.

---

## Decisions made

- 719 runs to completion rather than stopping early
- All 7 artifact kinds for 719
- 682's artifacts preserved untouched
- No one-command docs repoint pipeline — headline changes are rare and the
  analysis prose has to change anyway
- Retroactive config migration for 682 and 719 is acceptable
- Keep Python as the config generator; add a data variation layer; do not adopt
  Hydra (conflicts with frozen dataclasses and the declarative CLI registry, and
  the pieces actually needed are ~150 lines)

## Open questions

- Does `--continue RUN key=value` fork a new W&B run or extend the existing one?
  Leaning fork, so no run has two configs in its history.
- Is the step 8 docs check worth building? Recommended, but user undecided.
