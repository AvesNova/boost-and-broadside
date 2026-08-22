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

### 8a — one profile — **DONE**

`rl` and `rl-fields` merged; BC is now `replace()` on `RL_PROFILE`. Deleted
`profiles/rl_fields.py`, `make_bc_schedule_spec`, the `REWARDS`/`FIELD_REWARDS`
split, and `tests/config/test_bc_profile.py` (181 lines policing a relationship
the overlay guarantees by construction).

Verified by resolving every profile before and after and diffing the leaves: the
merged `rl` is **byte-identical to the old `rl-fields`**, and BC moved only by
gaining the fielded environment. BC's divergence from RL is exactly the 7 leaves
its docstring names, now asserted as data in one test.

**Part of 8f came forward.** The S01 snapshot tests, `_INTENDED_DIVERGENCE` and
the fingerprint pin test are keyed by the three profile names this step merges,
so keeping them would have meant a divergence list describing the merge itself,
and re-pinning six fingerprints for a test 8f deletes. They went with 8a.

Two tests were **inverted** rather than deleted, because the property they
protected still matters in the opposite direction:

- *no profile imports another* → *the base does not import an overlay*. The
  overlay direction is the mechanism; what must not happen is a cycle.
- *profiles are independent values* → *an overlay shares the base's sub-spec
  objects but not its identity*. Sharing is the guarantee; the copy is the
  safety.

It also caught an eighth mode. `capture` reads a run, and it reads the
checkpoint's `env_config` -- which does carry `num_fields` -- while never reading
the `field_map` beside it, so it belonged in the step 7 sweep and was invisible
there only because the fixture was field-free. Merging the profiles made the
fixture fielded and the `capture` smoke case failed immediately. Fixed by
factoring the field-map read out of `load_run_config` into `recorded_field_map`,
which both now use, and covered by a test verified failing against the old code.

Also: the whole VRAM shard ladder moved, because the one profile is 12 entity
tokens wide rather than 8. The valid widths are now
`(7776, 2592, 864, 288, 96, 32)` and the default is 2592, not 3904. The measured
8 GB row in `memory-optimization.md` was taken at 31,232 resident tokens against
the current 31,104 — within 0.4%, so it is expected to carry, but it is flagged
there as un-reprobed rather than quietly restated.

### 8b — one schema

Split at the seam the plan named. Both halves are done.

**8b-1 — flatten the profile.** `EnvironmentSpec`, `RolloutSpec`,
`DiscountSpec`, `ObjectiveSpec`, `OptimizerSpec` and `LeagueSpec` are gone;
`ProfileSpec` holds their fields directly. No sub-spec was ever passed anywhere
on its own, so the grouping bought nothing and cost a hop at every read. The
resolved output is **identical for both profiles**, verified leaf by leaf
against the previous commit.

Where this shows up: the smoke fixture went from six nested `replace()` calls to
one, and BC's overlay from three to a flat list. `_validate_profile` lost its
sub-object unpacking. Adding a hyperparameter is now two files rather than four.

**8b-2 — delete the pass-through. DONE.** 25 of the 33 `TrainConfig` fields were
being copied from the profile one line at a time. They are now copied by name:
`resolve_profile` takes every field whose name appears on both schemas, and
spells out only the eight that are genuinely derived.

Adding a plain hyperparameter is a two-line change — one field on each dataclass
— with nothing to edit in the resolver.

**The naming convention became load-bearing**, which is the price. A transformed
value must be named differently on the two sides (`gamma_per_tick`/`gamma`,
`schedule_spec`/`schedule`), or it would be copied straight through and the
trainer would read stated intent as a derived result — a per-tick discount
silently becoming a per-decision one. The eight derived names are pinned in
`test_only_untransformed_intent_shares_a_name_with_the_resolved_config`, so
arriving in that set by accident fails loudly.

*The compatibility risk this step was supposed to carry did not exist.*
`TrainConfig` is never reconstructed from stored data — `_rebuild_config` is only
applied to `ModelConfig`, `ShipConfig`, `EnvConfig` and `FieldMapConfig`. The only
stored fields anything reads back are `train_config.field_map`,
`train_config.paradigm`, and three document-level keys, all via `.get()` and all
unchanged. Verified directly: 682, 716 and 719 all load, and
`tests/evaluation/test_landmark_runs.py` now checks that every commit.

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
