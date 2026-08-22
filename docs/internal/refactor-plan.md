# Config, checkpoint, evaluation, and sweep plan

Working plan for the 719 headline changeover, the config-system rewrite, and the
hyperparameter search. Internal: it records reasoning, measurements, and
sequencing, not reader-facing documentation. Reader-facing material lives in
[training.md](../training.md).

Written so a fresh session can pick this up without re-deriving anything.
Current as of `6d073d5`, branch `feature/gradient-diagnostics`.

**Read [Findings](#findings) before starting any step.** Several steps exist
because of specific measured defects, and the evidence is not obvious from the
code alone.

---

## State

| | |
|---|---|
| Branch | `feature/gradient-diagnostics` (pushed) |
| HEAD | `6d073d5` Drop the descriptions under plot titles |
| Run 719 | `good-leaf-719`, complete, 999.3M steps, **calibrated 1748.4 ± 9.8**. Landmark, in-repo via git-lfs. All 7 artifact kinds + figures. |
| Run 716 | `lunar-cosmos-716`, complete, 499.7M steps, calibrated 1634.4. Calibration only. |
| Run 682 | `resilient-resonance-682`, outgoing headline. Artifacts and lfs payload stay in-repo. |
| GPU | single RTX 4070 Laptop, 8188 MiB. 719 measured **2705 env steps/s** with `reward_full` diagnostics on. |

## Decisions made

- **719 replaces 682 outright.** Not a side-by-side: 682 stops being the
  reference run. A short comparison section is retained somewhere in the docs,
  and that is the only place 682 appears as a result.
- README's lead figure becomes `elo_scale_scripted_1000.png`, replacing
  `crossover_phase.png`.
- Order of work: **C (loose ends) → D (fields coverage) → B (config refactor)**,
  i.e. steps 4, 5, 6 → 7 → 8 below.
- Charts render per-run (`bnb figures --run RUN`); `docs/publications.toml` only
  chooses which run illustrates the docs.
- Sweeps use **successive halving with one config per arm**, not per-block
  hyperparameters. See step 10 and the Findings entry on why.
- Keep Python as the config generator; add a data variation layer; do not adopt
  Hydra.
- No one-command docs repoint pipeline — headline changes are rare and the
  analysis prose has to change anyway.

---

## Steps

### 1. Resume fix + provenance hygiene — **DONE** (`c090f88`, `ea78e93`)

- [x] `_apply_schedule_state(step)` split out of `_refresh_training_schedule`
- [x] Called at the end of `load_checkpoint`
- [x] Three regression tests, all confirmed failing before the fix
- [x] `run.json` records git SHA + dirty flag
- [x] A hard crash records `status: failed`

`tests/train/test_checkpoint.py::TestResumeRestoresScheduleState`.

### 2. 719 completes — **DONE**

- [x] `status: complete`, update 1004/1004, final checkpoint written

### 3. Evaluation campaign — **DONE**

- [x] `elo-calibrate` on 719 and on 716
- [x] `elo-scale`, `crossover`, `semi-random`, `ar-report`,
      `noise-calibration`, `wandb-export` on 719
- [x] 719 promoted to a landmark run, git-lfs, 145 MB (`624af8d`, `479f14a`)

Cost five latent bug fixes; see [the fields gap](#the-fields-gap-step-7).
682's artifacts were verified untouched.

---

### 4. Close out the loose ends — CPU

- [x] Full test suite — **1594 passed** at `b91b152`.
- [x] **Figures artifacts accumulated when they should replace.** 719 had two,
      identical recipe digests, *both committed*, the older carrying the
      subtitles since removed. `ArtifactStore.create_stable` gives a derived
      artifact a fixed path, replacing any prior one; measurements still
      accumulate. Figures now live at `checkpoints/<run>/artifacts/figures`.
- [x] Stale figures artifact removed from git.

The deciding argument was not tidiness. `docs/publications.toml` has to *quote*
the figures path, so a per-render identity would break the docs pointer on every
re-render — the opposite of what a stable reference is for. Step 6 depends on
this.

**Not doing:** continuing the unconverged `elo-scale` sizes (16v16 ±15.1,
32v32 ±32.5, 64v64 ±50.5). Crossover carries the same result at those widths
with converged win rates. Revisit only if a rating-vs-fleet-size figure with
error bars is wanted.

### 5. Prior-runs log — **DONE**

- [x] `docs/training-runs.md`: one short entry per completed run — what it did,
      its headline settings, links to its artifacts. Deliberately brief and not
      a comparison; the point is to show progress and that more than one run
      succeeded.
- [x] Ratings stated per run, with one closing note on why 682's and 719's are
      not the same scale. 716 and 719 share physics, so those two *are* compared.
- [ ] Link it from `evaluation.md` and the README nav during step 6.

### 6. Docs changeover — after step 5

- [ ] Repoint every `docs/publications.toml` entry at 719's figures artifact.
- [ ] README lead figure → `elo_scale_scripted_1000.png`, with new caption and
      refreshed selected-results table (719's crossover numbers, not 682's).
- [ ] Rewrite the reward-weight and config prose against **719's stored
      config**, not `main`. See [the docs contradiction](#the-docs-already-contradict-the-figures-steps-6-9).
- [ ] `bnb publish`, then `bnb publish --check` clean.

### 7. Fields coverage for the evaluation modes — CPU, ~0.5 d

The gap that cost step 3 a day. No evaluation mode is exercised end to end
against a run with fields, so all six passed smoke while being unable to
measure the only kind of run currently trained.

- [x] `build_synthetic_run(..., profile="rl-fields")` — the existing smoke
      fixture, parameterised by profile, so the fields run is built through the
      same production serializers as the field-free one
- [x] All six modes driven against it in `tests/modes/test_fields_evaluation.py`
- [x] Assertions are on the environment actually built, via a `TensorEnv` spy —
      patched at the simulator rather than at any one helper, because the modes
      reach it by three different routes
- [x] **Verified against the pre-fix code in a scratch worktree: all seven fail**,
      three with `assert []` (no environment at all), two with
      `num_fields=[0, 0]` (the silent ones), and one with `{2} == {6}` — the
      hidden state sized for two ships instead of two ships and four fields.

Not done: a fields smoke case. The matrix already runs `train-rl-fields`, and
these tests cover the evaluation side more precisely than a smoke launch would.

Blocks: nothing, but do it before step 8 rewrites the resolver these modes read.

### 8. Config refactor — CPU, the large block

Each sub-step independently shippable; run suite + smoke after each.
Full plan, invariants and risks: [config-refactor.md](config-refactor.md).

- [ ] **8a** Drop non-field support; merge `rl`/`rl-fields`; BC becomes an
      overlay; hoist the 19 duplicated values into one base.
      Deletes: `profiles/rl.py`, `profiles/bc.py`, `make_bc_schedule_spec`,
      `tests/config/test_bc_profile.py`, the `REWARDS`/`FIELD_REWARDS` split,
      the config-layer `field_map is None` branches. The `num_fields == 0`
      branches in `env/` stay — that is how 682 is evaluated.
- [ ] **8b** Collapse Spec + Config into one schema with explicit
      intent/derived field pairs (`gamma_per_tick` stored, `gamma` derived).
      Derivation becomes `derive(config) -> config`
- [ ] **8c** Flatten schedules to `[step, value, interp]` keypoint tables
      (verified bit-identical on the real LR schedule)
- [ ] **8d** `checkpoints/<run>/config.json` holding intent + source +
      overrides + code provenance. **Append-only list of segments**, each keyed
      by the `global_step` it takes effect at — not a single document. See
      [continuation](#continuation-changes-a-runs-config-mid-flight-steps-8d-8e).
      Readers ask for the config *at a step*, or the latest; `config_at(step)`
      is the API, and evaluation modes rating a final checkpoint want the last
      segment.
- [ ] **8e** Positional `key=value` overrides; `--continue RUN`;
      `--from RUN [--at STEP]`. `--continue` extends the **same** run and logs
      to the **same** W&B run; it appends a config segment and re-attaches via
      `resume_wandb_run_id`. Emit the changed keys as a logged event at the
      switch step so a chart shows where the settings moved.
- [ ] **8f** Delete the fingerprint superstructure: drift guard,
      `--allow-config-drift` on the training path, the fingerprint pin test,
      the S01 snapshot tests, `_INTENDED_DIVERGENCE`

**Keep:** `canonical_json`, the feature/physics guard in `policy_io`, a local
`hash(dict)` inside `vram.py`. **Also keep** the artifact recipe digest — it is
derived at write time and never checked against a pin, so it carries no
maintenance cost. The one place to re-examine is `open_resumable`, the only
point where a digest actually gates an action.

Done when: changing a config value is one edit.
Blocks: step 9.

### 9. Migrate 682 and 719 configs — CPU

- [ ] One-time script emitting new-format `config.json` for both runs
- [ ] **Preserve absence, do not backfill.** 682 genuinely had no
      `field_damage_taken_weight` and no `total_timesteps`
- [ ] Regression test loading 682, 716, 719

Precedent: `scripts/migrate_682.py`, `tests/migration/`.

### 10. Hyperparameter search

**Successive halving, one config per arm.** Per-block hyperparameters were
considered and rejected: they are greedy, they force every later arm to resume
from a shared checkpoint, and they save nothing that pruning does not already
save. See [the block-boundary analysis](#why-blocks-do-not-work-and-what-does-step-10).

- [ ] **Seed the global numpy RNG.** Minibatch order is currently unseeded, so
      paired comparisons are impossible. Hard prerequisite.
- [ ] Diagnostics off for sweep arms
- [ ] Gate schedule: all arms → 50M, top half → 100M, top half → 250M
- [ ] Objective: post-hoc calibration on survivors; smoothed live Elo (mean of
      the last 50 updates) for the gates
- [ ] **Record each arm's rank at 50M and 100M and check afterwards whether it
      predicted the 250M finish.** The whole scheme assumes early rank predicts
      late rank; that is untested, and this makes the first sweep test it for
      the price of bookkeeping.
- [ ] Local job runner (one GPU = depth-1 queue)
- [ ] W&B Sweeps with flattened `wandb.config`; positional `key=value` matches
      `${args_no_hyphens}` natively. Launch queue only if moving to cloud.

### 11. Docs check — optional, recommended

- [ ] `<!--cfg:run=… path=…-->` annotations on config-derived numbers
- [ ] One test resolving each against the named run's stored config, read as
      **data**. A missing path is a loud failure, and that is the feature.

---

## Findings

### 719's results

Calibrated with scripted anchored at 1000, all three endpoints in one
tournament so they are directly comparable:

| | calibrated | ± |
|---|---:|---:|
| `ckpt_999309312` (final) | **1748.4** | 9.8 |
| `best_training` | 1732.2 | 9.7 |
| `best_avg` | 1708.5 | 9.5 |

The final checkpoint is the strongest, so it is the headline. `best_training`
was selected on *live* Elo, which peaked ~1750 mid-run — it captured a lucky
window, not a better policy. 716 repeats the ordering (1634.4 / 1628.6 /
1471.2), and its `best_avg` trails by 163, which is worth a look during the
config refactor.

Rating by fleet size, and crossover:

| team size | elo-scale | ± | crossover ratio |
|---|---:|---:|---:|
| 1v1 | 1426.3 | 9.4 | 2.00× |
| 2v2 | 1564.7 | 8.5 | 2.00× |
| **4v4** (trained width) | **1765.1** | 8.7 | 1.75× |
| 8v8 | 2000.1 | 10.3 | 1.63× |
| 16v16 | 2160.2 | 15.1 *unconv.* | 1.50× |
| 32v32 | 2204.0 | 32.5 *unconv.* | 1.41× |
| 64v64 | 1761.4 | 50.5 *unconv.* | 1.25× |

Two independent cross-checks. 4v4 reads 1765.1 here against 1748.4 from the
calibration tournament — different field, different reference, agreeing inside
their error bars. And 64v64 turns down in *both* measurements (crossover's win
rate at equal numbers falls 99.2% → 95.1%), so the degradation there is real
rather than the noise the ±50 alone would suggest. It is still a policy trained
at 4v4 winning 95% at sixteen times its training width.

### 682 and 719 are not on one Elo scale

Elo is only defined within a pool playing one game, and these two played
different games. The stationary rungs, which are the same agents in both fits,
land in completely different places:

| rung | 682 | 719 | diff |
|---|---:|---:|---:|
| random | **−335.3** | **+137.5** | +472.8 |
| `semi_scripted_0p2` | −199.6 | 228.6 | +428.2 |
| `semi_scripted_0p5` | 328.6 | 538.1 | +209.4 |
| `semi_scripted_0p8` | 735.0 | 821.7 | +86.7 |
| `semi_scripted_0p95` | 934.7 | 950.1 | +15.4 |
| scripted | 1000.0 | 1000.0 | — |

The random→scripted span is **1335 Elo under 682's physics and 862 under
719's**. Anchoring scripted at 1000 pins one point; it does not fix the spacing.

The cause is that the two runs trained under different physics, not merely
different field counts:

| | 682 | 719 |
|---|---:|---:|
| `bullet_min_damage_frac` | 0.1 | **1.0** |
| `bullet_energy_cost` | 3.0 | **2** |
| `max_bullets` | 20 | **10** |
| `action_repeat` | 1 | **2** |
| `spawn_resource_spread` | 0.0 | **0.25** |
| `num_fields` | 0 | **4** |
| bullet encoder | **none** | reads bullets |

Under 719's rules every bullet hits for full damage and shooting is cheaper, so
random flailing is far more dangerous and the ladder compresses. That is a
property of the game, not of either policy.

Consequently:

```
final, raw:             682  1772.2   719  1748.4   <- 682 higher by 23.8
final, in ladder spans: 682   0.578   719   0.868   <- 719 ahead by 50%
```

Both come from the same data and point opposite ways. Normalised to the
random→scripted distance, 719 sits 0.87 spans above scripted against 682's 0.58.

This also killed the head-to-head idea: any arena is one run's home ground, and
682 additionally has no bullet encoder, so it would be blind to bullets that now
hit for full damage. It would have measured whose physics the arena used.

### Cross-run comparison within one physics is valid to ~10 Elo

719 and 716 share physics, unlike 682. They were fitted in separate tournaments,
so their scales are only comparable if the shared stationary field agrees. It
does — and this is the check that fails for 682: every rung matches
between the two fits within **9.9 Elo** (`0p5`: 538.1 vs 529.1; `0p8`: 821.7 vs
819.4). So **719 finishes +114 over 716** is a real statement — descriptively.
The runs differ in learning rate, shoot-quality weighting and budget.

### Live Elo is usable for ranking, not for stopping

Drift is `calibrated − live`. Using the run's *final* live Elo, which is what a
sweep would rank on:

| estimator | 719 | 716 | spread |
|---|---:|---:|---:|
| last update only | +53.6 | −45.4 | 99.0 |
| mean of last 25 | +15.0 | −18.7 | 33.7 |
| mean of last 50 | +16.4 | −14.6 | 31.0 |
| mean of last 100 | +19.7 | −2.4 | 22.1 |

Smoothed, per-run error is ~15–20 and a pairwise difference carries ~30. Good
enough to rank arms differing by 50+ Elo; not good enough for early stopping,
where the single-update value is off by ±50. Restricted to the >1000 regime the
per-snapshot picture is mean |drift| 13.7, max 31.1 — 719 within ±10 throughout,
716 reaching −31, so there is **no consistent direction** to correct for. From
n=2 runs of one profile.

### Why blocks do not work, and what does (step 10)

719's Elo gain by segment:

| segment | gain |
|---|---:|
| 0–50M | **+1088** |
| 50–150M | +341 |
| 150–250M | +187 |
| 250–350M | +87 |
| 350–450M | **+22** |
| 450M–1B | ~+43 |

Behaviour-cloning decays to zero at **22M steps** and is off entirely by 37M, so
the whole BC/ladder/win-rate-gate regime lives inside the first segment.

A 300–450M block would produce **+22 Elo of movement against a ±30 pairwise
measurement error** — every arm ties, and nothing can be ranked. Past ~300M is
not searchable; pick it with two or three confirmation runs.

Staging also saves less than it appears. Total step-budget is `N₁×50M + N₂×200M`
staged versus `N×250M` single-block: 20-then-6 arms costs 2200M, eight arms
costs 2000M. The saving was never in the staging, it was in the pruning — so
take the pruning and drop the greedy handoff.

Successive halving at 2705 steps/s (50M ≈ 5.1 h, 250M ≈ 25.7 h): 8 arms → 4 → 2
costs ≈ 3.9 days against ≈ 8.5 for brute force, with no config changing mid-run.
0–50M is an unusually good gate: 5% of a full run, +1088 Elo of spread, and the
only place BC and the ladder are live.

### The fields gap (step 7)

Five latent bugs, one cause — fields were added to training and the evaluation
stack never followed, while 682 (field-free, and *imported from disk* rather
than exported live) was the last thing anything was measured on.

| mode | failure |
|---|---|
| `elo-calibrate`, `elo-scale`, `semi-random` | no field-map intent read from the run |
| `crossover` | same; hard failure |
| `ar-report`, `noise-calibration` | same, **silent** — would have measured 719 in an empty arena |
| `noise-calibration` | field tokens dropped from its own report layout |
| `export_wandb_run.py` | nested summary proxies; this path had never completed |

Fixed in `89c7c72`, `6102a82`, `6f04814`.

### The resume bug (step 1, fixed)

`load_checkpoint` restored weights, optimizer, scalers, eval windows and
counters but not the schedule-derived state, and `train()` calls `_update_epochs`
*before* `_refresh_training_schedule`. A run that had correctly decayed BC to
zero resumed at full strength. Measured on run 717, update 148 being the first
after a resume:

| | 147 | **148** | 149 |
|---|---:|---:|---:|
| `bc_loss` | 0 | **2.231** | 0 |
| `policy/kl` | 0.0085 | **1.462** | 0.008 |
| `gradient_norm` | 0.79 | **9.56** | 0.96 |
| scripted win rate | 0.87 | 0.86 | **0.19** |
| live Elo | 1316 | 1390 | **908** |

~400 Elo, ~20 updates to recover. Only ever misfired on resume, which is why
nothing caught it. **Still required after the config rewrite**: derived runtime
state, not a config value.

### Profiles vs defaults is inverted (step 8a)

19 values are byte-identical across all three profiles and duplicated in each:
`clip_coef`, `max_grad_norm`, `return_ema_alpha`, `return_min_span`,
`advantage_min_rms`, `return_quantile_samples`, `histogram_interval`,
`log_interval`, `league_size`, `league_slots`, `elo_temperature`,
`elo_milestone_gap`, `bc_winrate_target`, `num_steps`, `num_minibatches`,
`logical_batch_tokens`, `action_repeat`, `num_ships`, `checkpoint_dir`.
Meanwhile `defaults.py` holds `FIELD_REWARDS` and `make_bc_schedule_spec`, each
used by exactly one profile.

`rl` vs `rl-fields` differ in 8 leaves, of which `num_envs` is *derived*.
Dropping non-field support merges them: there is no such thing as a model that
"doesn't support fields" — `num_fields=0` is a degenerate configuration, not an
architecture variant. `bc` vs `rl` differ in 7 values, so BC is an overlay, and
`tests/config/test_bc_profile.py` (181 lines) exists only to police a
relationship an overlay guarantees by construction.

### Two schemas, three edits per hyperparameter (step 8b)

`ProfileSpec` + 6 sub-specs on one side, `TrainConfig` + 5 configs on the other,
`resolve.py` (556 lines) translating. Adding one hyperparameter means editing
the Spec, the resolver line, and `TrainConfig`. Only about five fields are
genuinely derived.

### Flat schedules are behaviour-preserving (step 8c)

```
[[0, 1e-7, "linear"], [5_000_000, 4.5e-4, "hold"],
 [100_000_000, 4.5e-4, "exponential"], [500_000_000, 1.5e-4, "hold"]]
```

Verified against the compiled schedule at 10 probes including boundaries and
past-the-end clamping: `max |flat − current| = 0.000e+00`. Schedules are pure
functions of `global_step` — save the definition, evaluate at the restored step.

### JSON vs YAML (step 8d)

JSON as the artifact of record; render to YAML only for display. YAML 1.1 parses
`1e-7` as a **string** and `no` as `False`, and the configs are full of
scientific-notation learning rates.

### The docs already contradict the figures (steps 6, 9)

Every published chart comes from 682. What 682 actually trained with, against
what `docs/training.md` claims:

| Component | docs say | 682 actually |
|---|---:|---:|
| `ally_win` | 1.5 | **4.0** |
| `shoot_quality` | off | **0.1** |
| `combat_damage_taken` | 0.5 | **did not exist** |
| `field_damage_taken` | off / 0.5 | **did not exist** |
| `field_death` | off / 1.0 | **did not exist** |

682 had no fields at all and predates `resolved_config` entirely. The prose has
been tracking `main` while the figures stayed pinned to the run. The step 6
rewrite must be against 719's stored config, and this is the argument for step 11.

### VRAM

A full-width run holds ~7200 MiB of 8188, leaving ~560 MiB — not enough for a
second CUDA process. Evaluation needs the GPU essentially alone. Note
`nvidia-smi` currently fails with a driver/library mismatch (NVML 595.84) after
a host driver upgrade; new CUDA processes still initialise, but VRAM monitoring
needs a reboot.

### Continuation changes a run's config mid-flight (steps 8d, 8e)

`--continue RUN key=value` extends the same run and logs to the same W&B run.
Decided; the alternative was forking a new run so that no run ever held two
configs.

The cost is that "the run's config" stops being a value and becomes a function
of step, and three things follow.

`config.json` cannot be write-once. It is a list of segments, each recording the
step it took effect at, the keys that changed, and the code provenance of the
process that made the change — a continuation after a week of edits runs
different code, and the segment is the only place that is visible.

Anything that reads a run's config has to say *which* config it means. Rating a
final checkpoint wants the last segment. The step 11 docs check wants the
segment in force when the cited number was produced, which for a headline claim
is again the last one. A bare `load_config(run)` returning "the" config is the
shape to avoid.

And the drift guard becomes actively wrong rather than merely costly: a
deliberate mid-run config change is exactly what it is built to refuse. This
strengthens step 8f — with `--continue`, config drift within a run is a
supported operation, not a corruption to detect.

## Open questions

- Does ranking at 250M predict ranking at 1B? Step 10's gate check answers the
  50M→250M half; this half stays open.
- Should a continuation that changes the *environment* (ship count, field
  count) be refused rather than appended? Those are not hyperparameters — they
  change the task, so the Elo history before and after would not be one series.
  Leaning refuse, with `--from RUN` as the supported route.
