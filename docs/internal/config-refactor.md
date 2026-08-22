# Step 8 — the config refactor

High-level plan for the large block in
[refactor-plan.md](refactor-plan.md#8-config-refactor--cpu-the-large-block).
Internal: sequencing and reasoning, not reader documentation.

The findings that motivate each sub-step are in the parent plan and are not
repeated here. Read [Profiles vs defaults is
inverted](refactor-plan.md#profiles-vs-defaults-is-inverted-step-8a), [Two
schemas](refactor-plan.md#two-schemas-three-edits-per-hyperparameter-step-8b),
and [Continuation changes a run's config
mid-flight](refactor-plan.md#continuation-changes-a-runs-config-mid-flight-steps-8d-8e)
first.

---

## Why

Step 10 is a hyperparameter sweep. A sweep is an instrument for changing config
values quickly and recording exactly which values produced which result, and
the current system is bad at both halves:

- **Changing one value costs three edits** — the Spec, the resolver line, and
  `TrainConfig` — across `schema.py`, `resolve.py` (556 lines) and
  `training.py`. Some also need a fourth in `profiles/`.
- **Three profiles duplicate 19 identical values**, so a change that should be
  global is three edits or a silent inconsistency.
- **What a run trained with is not recoverable as data.** It survives only as
  the `resolved_config` blob inside a `.pt`, and the docs have already drifted
  from it (see [the docs
  contradiction](refactor-plan.md#the-docs-already-contradict-the-figures-steps-6-9)).
- **The fingerprint superstructure refuses the thing step 8e makes
  supported.** A deliberate mid-run config change is precisely what the drift
  guard exists to reject.

**Done when:** changing a hyperparameter is one edit, and
`checkpoints/<run>/config.json` answers "what was this trained with, at this
step" without loading a checkpoint.

## Shape of the end state

```
bnb train                         # the one profile
bnb train lr_peak=3e-4 num_fields=8
bnb train --continue good-leaf-719 entropy_coef=0.002
bnb train --from good-leaf-719 --at 250000000
```

One profile defines intent. `derive(config) -> config` fills the ~5 genuinely
mechanical fields. Overrides are positional `key=value` against leaf paths.
Every launch appends a segment to the run's `config.json`.

## Invariants

These hold after every sub-step, and each has a test that fails loudly rather
than a convention:

1. **719, 716 and 682 stay loadable and stay rateable.** The evaluation modes
   must keep producing the same numbers; 682 in particular is field-free and
   predates `resolved_config` entirely.
2. **`num_fields` is a sequence length, not an architecture switch.** The
   network is identical at 0, 1 and 8 fields: `num_fields` only sets the token
   count `N + M`, and no weight shape depends on it — `encoder_split` splits on
   the ship/field *boundary*, which is well defined when the field slice is
   empty. So there is no "field-free model" to support, which is the whole
   argument for 8a. What must keep working is `num_fields=0` as a
   *configuration*: the `num_fields == 0` branches in `env/`, `observation.py`,
   `physics.py` and `renderer.py` are how 682 is still evaluated, and they stay.
   Only the config-layer branching goes.
3. **Schedules stay bit-identical.** 8c is verified numerically, not by
   inspection — the existing probe showed `max |flat − current| = 0.000e+00`
   and that is the acceptance bar.
4. **The physics guard in `policy_io` survives 8f.** Deleting the *config*
   drift guard must not delete the *feature/physics* one; they are different
   checks that happen to share the `--allow-config-drift` flag name.
5. Full suite (1594 tests) plus `--smoke` green before each commit.

## Sequence

Each sub-step is independently shippable and separately committed. The order is
forced: 8a shrinks the surface 8b has to collapse, 8b gives 8d a single schema
to serialise, and 8d gives 8e something to append to.

### 8a — one profile

Merge `rl` and `rl-fields`; BC becomes an overlay rather than a peer. Hoist the
19 shared values into one base.

Deletes `profiles/rl.py`, `profiles/bc.py`, `make_bc_schedule_spec`, the
`REWARDS`/`FIELD_REWARDS` split, and `tests/config/test_bc_profile.py` (181
lines policing a relationship an overlay guarantees by construction).

Touches the smoke matrix: `train-rl`, `train-rl-fields` and `train-bc` become
one training case plus a BC-overlay case.

*Risk:* `bc` differs from `rl` in 7 values, and the overlay has to reproduce
all 7. Check by resolving both before and after and diffing the leaves.

### 8b — one schema

Collapse `ProfileSpec` + 6 sub-specs and `TrainConfig` + 5 configs into one
schema with explicit intent/derived pairs (`gamma_per_tick` stored, `gamma`
derived). `resolve.py` becomes `derive(config) -> config`.

This is the largest single change and the one most likely to want splitting
again mid-flight — a reasonable seam is to land the merged schema with
`resolve.py` still in place, then delete the resolver.

*Risk:* the resolved-config document changes shape, which is what
`evaluation/subjects.py` and `policy_io.py` read out of old checkpoints.
Those readers need a compatibility path before 8b lands, not after.

### 8c — flat schedules

`[step, value, interp]` keypoint tables replacing the compiled-closure specs.
Behaviour-preserving; verify at boundaries and past-the-end clamping.

### 8d — `checkpoints/<run>/config.json`

An **append-only list of segments**, each keyed by the `global_step` it takes
effect at and carrying intent, source, overrides and code provenance. The API
is `config_at(step)` and `latest(run)`; a bare `load_config(run)` returning
"the" config is the shape to avoid.

JSON is the record; YAML only for display (YAML 1.1 parses `1e-7` as a string).

### 8e — overrides and continuation

Positional `key=value`; `--continue RUN` extending the same run and re-attaching
to the same W&B run via `resume_wandb_run_id`; `--from RUN [--at STEP]` forking.

`--continue` appends a segment and logs the changed keys as an event at the
switch step, so a chart shows where the settings moved.

*Open:* whether a continuation that changes `num_ships` or `num_fields` should
be refused. Those change the task, so the Elo series would not be one series.
Leaning refuse, with `--from` as the supported route.

### 8f — delete the fingerprint superstructure

The drift guard, `--allow-config-drift` on the training path, the fingerprint
pin test, the S01 snapshot tests, `_INTENDED_DIVERGENCE`.

**Keep:** `canonical_json`, the feature/physics guard in `policy_io`, a local
`hash(dict)` inside `vram.py`, and the artifact recipe digest — derived at write
time, never checked against a pin, so it carries no maintenance cost. The VRAM
cache key (`profile_fingerprint` through `launch.py` and `vram_probe.py`) is a
cache key, not a guard, and stays; it may want renaming so the distinction is
visible.

The one place to re-examine is `open_resumable`, the only point where a digest
actually gates an action.

## Then

Step 9 migrates 682 and 719 to the new format (preserving absence, not
backfilling). Step 10 is unblocked once 9 lands.
