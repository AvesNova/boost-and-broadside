# Mode refactor — live execution ledger

This is the single live status and handoff document for
`docs/internal/mode-refactor-plan.md`. The plan owns architecture and scope; this ledger owns
execution order, agent assignments, prompts, progress, evidence, and deviations.

Delete this file with the plan after the refactor lands and its durable contracts have moved into
code, tests, and reader-facing documentation.

## Operating contract

- Branch: `mode-system-refactor`, created from `main` at `afdf406`.
- Top-level execution is **100% sequential**: the user launches one primary section owner at a
  time, and exactly one section may be `in_progress`.
- A primary owner may spawn subagents or concurrent writers within its own section. It owns their
  coordination, integration, verification, and final handoff.
- Concurrent writers use non-overlapping file ownership or separate worktrees. They never edit the
  same files concurrently, and no delegated agent may begin work assigned to a later section.
- A section starts only when every earlier row is `completed` and the worktree is clean.
- Each implementation agent owns one bounded section, its tests, its commits, and its handoff.
- Review agents do not change product code. They may update this ledger with findings.
- A blocking review finding inserts a new sequential remediation row immediately after the review;
  all later work remains pending until remediation and re-review complete.
- Existing/user changes are preserved. Agents inspect before editing and never reset unrelated work.
- Every implementation commit must be green for its stated verification. Milestone gates run the
  broader suites named in the overall plan.
- Decisions or scope changes go in the overall plan once; execution consequences go here. Do not
  duplicate evolving facts across other temporary documents.

## Status vocabulary

- `pending`: prerequisites are incomplete; agent must not start.
- `in_progress`: the one active section.
- `blocked`: cannot proceed without an explicit decision or failed prerequisite.
- `completed`: committed, verified, and handed off.

## Current state

- Active section: none; `S07R` is completed.
- Next section: `S08` — mode consolidation.
- Blocking issue: none.
- Landmark migration: scheduled for `S15`, after all target schemas stabilize.

## Sequential queue

| ID | Plan phase | Status | Agent type | Model / effort | Mission |
|---|---:|---|---|---|---|
| S00 | governance | completed | planning lead | Sol / extra high | Finalize plan, branch, ledger, and prompts |
| S01 | 0 | completed | characterization engineer | Terra / high | Capture behavior, config, CLI, and publication baselines |
| S02 | 1 | completed | configuration architect | Sol / extra high | Move profiles under `src`; add independent specs/resolution/fingerprints without changing RL behavior |
| S03 | 1 gate | completed | configuration reviewer | Sol / extra high | Review resolved-config equivalence, dependency direction, and schema/fingerprint design |
| S03R | 1 gate remediation | completed | configuration remediator + reviewer | Sol / extra high | Fix S03 blockers and obtain an independent configuration re-review |
| S04 | 2 | completed | evaluation refactorer | Sol / high | Extract typed sizes, run catalog, match/environment, and tournament engines |
| S05 | 3 | completed | CLI engineer | Sol / high | Replace `main.py --mode` with the strict installed `bnb` subcommand CLI |
| S06 | 4 | completed | smoke/test engineer | Sol / high | Build synthetic checkpoint fixtures and fully isolated sequential subprocess smoke coverage |
| S07 | 2–4 gate | completed | integration reviewer | Sol / extra high | Review shared engines, CLI contracts, smoke isolation, and behavior preservation |
| S07R | 2–4 gate remediation | completed | integration remediator + reviewer | Sol / extra high | Close S07 blockers and obtain an independent shared/CLI/smoke re-review |
| S08 | 5 | pending | mode consolidation engineer | Terra / high | Consolidate training, retire modes/flags, and fix field-capable evaluation |
| S09 | 6 | pending | artifact/publication architect | Sol / extra high | Implement artifacts, provenance, raw samples, publication manifest, and offline render checks |
| S10 | 6 gate | pending | artifact reviewer | Sol / extra high | Review schemas, identity, atomicity, resume, Git-ignore safety, and offline publication |
| S11 | 7 | pending | training-profile engineer | Sol / high | Correct BC independently and validate its allowed differences from RL |
| S12 | 8 | pending | live-Elo engineer | Sol / extra high | Implement/document approximate live Elo separately from calibrated Elo |
| S13 | 9 | pending | VRAM engineer | Sol / extra high | Implement resolution precedence, probing, cache fingerprints, and provenance |
| S14 | 7–9 gate | pending | training-systems reviewer | Sol / extra high | Review BC, live Elo, and VRAM behavior together before checkpoint schema freeze |
| S15 | 10 | pending | checkpoint migration engineer | Sol / extra high | Migrate the complete 682 checkpoint set once into the frozen current schema |
| S16 | 10 | pending | landmark/publication integrator | Sol / high | Backfill 682 artifacts and raw samples; select and regenerate canonical publications |
| S17 | 10 gate | pending | migration/reproducibility reviewer | Sol / extra high | Independently verify 682 equivalence, completeness, provenance, and publication reproducibility |
| S18 | 11 | pending | documentation/cleanup engineer | Terra / high | Complete repo-wide docs and remove obsolete paths, names, and temporary compatibility residue |
| S19 | final gate | pending | final branch reviewer | Sol / extra high | Review the complete branch against the plan and run final acceptance checks |

Model names for agent configuration are `gpt-5.6-sol` and `gpt-5.6-terra`; “extra high” maps to
`xhigh`. An agent may lower effort for purely mechanical commands inside its section, but the
section's architectural decisions and final review use the assigned level.

## Common implementation-agent prompt

Give the assigned agent this prompt together with its mission card below:

```text
You own exactly section <SECTION_ID> of the Boost and Broadside mode refactor on branch
mode-system-refactor. The user launches primary sections sequentially; do not begin a later section.
Within your section, you may spawn subagents or concurrent writers when useful. Give them bounded,
non-overlapping ownership, use separate worktrees for overlapping write surfaces, and remain
responsible for integrating and verifying all delegated work.

Before acting, read these files completely:
- docs/internal/mode-refactor-plan.md
- docs/internal/mode-refactor-status.md

Confirm every earlier ledger section is completed and the worktree contains no unexplained
changes. Mark only your section in_progress in the ledger, then implement only its stated scope.
Preserve user changes and do not begin any later section.

Use characterization tests and resolved diffs to distinguish mechanical refactoring from intended
behavior changes. Add tests at the earliest useful seam. Run the section's narrow checks throughout,
then its required completion checks. Review your own final diff.

When complete:
1. create one or more focused commits on the current branch;
2. update only your ledger row to completed;
3. fill in its handoff record with commits, tests, behavior/config diffs, decisions, and remaining
   risks;
4. verify the worktree is clean;
5. stop without beginning the next section.

If blocked, mark the section blocked, record the exact evidence and required decision, and stop.
Do not guess through a behavior-changing ambiguity.
```

## Common review-agent prompt

```text
You own exactly review section <SECTION_ID> on branch mode-system-refactor. The user launches
primary sections sequentially; do not begin a later section. You may delegate bounded independent
review or test work within this section, then consolidate and verify the evidence yourself.

Read docs/internal/mode-refactor-plan.md and docs/internal/mode-refactor-status.md completely.
Confirm all earlier sections are completed. Mark only your review section in_progress.

Review the named committed range and run the review section's verification. Prioritize behavioral
regressions, architecture-contract violations, invalid config or artifact provenance, missing tests,
and unsafe failure behavior. Do not edit product code. Record findings with file/symbol references.

If there are no blocking findings, mark the review completed, record the evidence, commit the ledger
update, verify a clean worktree, and stop. If there are blocking findings, complete the review record,
insert a pending remediation section immediately after this one, mark later sections pending, commit
the ledger update, and stop. Never begin remediation yourself.
```

## Mission cards

### S00 — Planning, branch, and governance

Agent: planning lead, Sol extra high.

Steps:

1. Create `mode-system-refactor` directly from `main` without losing the planning documents.
2. Finalize the overall architecture, CLI, configuration, artifact, testing, migration, and
   publication decisions.
3. Create this sequential ledger and ready-to-use agent prompts.
4. Verify no contradictory migration/parallel/compatibility language remains.
5. Record the starting commit and worktree state.

Done when: both planning documents agree, the branch is correct, the queue is fully ordered, and the
planning baseline is committed.

### S01 — Characterization baseline

Agent: characterization engineer, Terra high.

Steps:

1. Inventory every current mode, flag, default, subject-resolution behavior, hardcoded environment,
   output path, and published asset/source.
2. Snapshot complete resolved RL and RL-fields configs; snapshot current BC separately as stale
   evidence, not desired behavior.
3. Add characterization tests around resume/run selection, size parsing, dispatch, and output paths
   that later mechanical refactors must preserve until their explicit breaking phase.
4. Add a machine-readable publication inventory covering all current docs outputs.
5. Run pytest and ruff; record any pre-existing failures without weakening tests.

Done when: later agents can mechanically prove what changed and why.

### S02 — Configuration foundation

Agent: configuration architect, Sol extra high.

Steps:

1. Move top-level `runs/` into `src/boost_and_broadside/profiles/` and update packaging/imports.
2. Add independent profile-intent schemas, project defaults, resolver, validation, stable canonical
   serialization, profile fingerprint, and resolved-config fingerprint.
3. Ensure no profile imports or derives from another profile.
4. Centralize environment/token sizing and time-normalized discount derivations.
5. Keep resolved RL/RL-fields field-for-field equal to S01 snapshots; do not correct BC yet.
6. Add source tracking for resolved launch values and `--print-config` support at the service layer.

Done when: config tests, snapshot equivalence, full pytest, and ruff pass.

### S03 — Configuration gate review

Agent: configuration reviewer, Sol extra high.

Review:

- S01-to-S02 committed range;
- independent-profile guarantee and dependency graph;
- semantic versus machine-specific boundaries;
- canonical serialization/fingerprint stability;
- RL/RL-fields exact equivalence and honest handling of stale BC;
- checkpoint and artifact readiness of the resolved schema.

Done when: no blocking correctness or architecture findings remain.

### S03R — Configuration gate remediation and re-review

Agent: configuration remediator plus an independent review owner, Sol extra high.

Steps:

1. Reproduce every blocking finding in the S03 handoff before editing product code.
2. Make the resolved configuration deeply immutable, or otherwise make it impossible to emit a
   configuration whose retained fingerprint describes different values; add a regression test that
   exercises the nested component-discount mappings and the stored/printed document.
3. Resolve launch width and rollout shard count together after explicit `num_envs` precedence so a
   machine-sizing override cannot silently change the profile's nominal logical token budget; test
   a materially smaller valid width, not only a near-default value.
4. Keep `grad_checkpoint` out of the semantic profile fingerprint while retaining it in complete
   resolved launch configuration/provenance; do not begin the broader S13 VRAM probe/cache work.
5. Replace the remaining live `runs/` path references made stale by S02 and add a focused guard for
   deleted profile-path references.
6. Re-run snapshot equivalence, focused config tests, full pytest, ruff, range whitespace checks,
   wheel/package verification, and an independent re-review of the S01-through-S03R range.

Done when: every S03 finding is closed with regression coverage, default RL/RL-fields/stale-BC
behavior remains exactly equal to S01 evidence, fingerprints match the documented semantic/machine
boundary, and the independent re-review reports no blocking finding.

### S04 — Shared evaluation primitives

Agent: evaluation refactorer, Sol high.

Steps:

1. Add typed `Matchup` parsing with 4v4 defaults and asymmetric support.
2. Add exact run resolution and distinct checkpoint-selection policies; remove library `sys.exit`.
3. Extract match runner, field-capable environment factory, next-state harness, and tournament engine
   under `evaluation/`.
4. Move user-facing modes onto the shared engines without changing their command/output semantics.
5. Make `team_pma_k` required and update all callers/tests.
6. Enforce with a dependency test that no user-facing mode imports another user-facing mode.

Done when: characterization tests, focused mode tests, full pytest, and ruff pass.

### S05 — `bnb` CLI

Agent: CLI engineer, Sol high.

Steps:

1. Register `bnb` in `pyproject.toml` and add the subcommand parser/registry.
2. Implement the final hyphenated command list and per-command option ownership.
3. Require exact subjects; remove `none`, `latest`, implicit 682, pathless resume, and default RL.
4. Make resume/pretraining exclusive and implement no-side-effect help plus `--print-config`.
5. Delete root `main.py` and the old `--mode` entry point with no compatibility aliases.
6. Add generated parser/help/error contract tests and update command documentation.

Done when: installed `uv run bnb ...` tests, CLI contracts, full pytest, and ruff pass.

### S06 — Smoke system

Agent: smoke/test engineer, Sol high.

Steps:

1. Extract a production checkpoint-payload builder suitable for a current-schema synthetic run.
2. Build the minimal checkpoint/roster/Elo fixture without depending on 682.
3. Define a deterministic smoke case for every mode/profile.
4. Run cases one at a time in fresh subprocesses with fixed timeouts and temporary roots.
5. Assert no writes escape those roots and the checkout remains unchanged.
6. Add focused single-case selection for diagnosis while keeping `bnb smoke` as the full matrix.

Done when: smoke passes from an empty real checkpoint directory plus full pytest and ruff.

### S07 — Shared/CLI/smoke gate review

Agent: integration reviewer, Sol extra high.

Review S04–S06 for behavior preservation, strict CLI failure behavior, packaging, subprocess
isolation, fixture fidelity, test gaps, and forbidden cross-mode dependencies.

Done when: no blocking findings remain and the full smoke/test/lint milestone is green.

### S07R — Shared/CLI/smoke gate remediation and re-review

Agent: integration remediator plus an independent review owner, Sol extra high.

Steps:

1. Make Elo-scale construct one player/metadata record for the final checkpoint whether or not that
   step is already a roster ladder milestone; cover a valid run whose final step is absent from the
   roster.
2. Resolve roster-declared tournament checkpoints strictly within the selected exact run and
   validate their recorded identity/step instead of accepting an existing foreign path.
3. Put complete resolved-config drift enforcement on full resume, keep documented BC-to-RL
   pretraining possible, and exercise both real loader methods rather than only the helper.
4. Validate resume and pretraining subjects before trainer/device allocation; reject magic/path-like
   run subjects and malformed matchups at parsing or the typed mode boundary instead of skipping.
5. Translate expected print-config, checkpoint, and runtime input failures into concise CLI errors
   without tracebacks, while preserving unexpected internal failures; add corrupt-input and invalid
   print-config coverage and make the printed launch honest about its validated execution settings.
6. Make smoke isolation detect writes outside each case root, ignored writes to real output roots,
   and checkout changes on every success/failure/timeout path; redirect mutable home/cache state and
   terminate complete subprocess trees on timeout, including capture's ffmpeg child.
7. Build the synthetic ladder checkpoint with the production policy-only payload, test every fixture
   checkpoint family, and exercise the registered CLI handler/adapter wiring end to end (or with an
   equivalent subprocess integration seam) rather than proving only command-name equality.
8. Make Ruff reproducible without the ignored local `wandb/` directory, remove the four committed
   EOF whitespace errors, and run focused tests, full pytest, full smoke, clean-archive Ruff, range
   whitespace, wheel/installed-help checks, and an independent re-review.

Done when: every S07 finding is closed by regression coverage, the checks are reproducible from a
clean archive/checkout, and the independent reviewer reports no remaining blocker. S08 remains
pending until this row is completed.

### S08 — Mode consolidation

Agent: mode consolidation engineer, Terra high.

Steps:

1. Consolidate training under `train --profile` and delete `bc-warmstart`.
2. Add arbitrary-agent Bradley–Terry calibration, then delete `elo-stats`.
3. Move AR orchestration into the mode with 4v4 canonical default.
4. Make `collect-stats` field-capable through the shared environment factory.
5. Remove dead defaults, arguments, no-op flags, and direct mode-level assumptions.
6. Update smoke cases and CLI tests as each mode changes.

Done when: no retired modes/flags/imports remain and full smoke/test/lint checks pass.

### S09 — Artifact and publication infrastructure

Agent: artifact/publication architect, Sol extra high.

Steps:

1. Implement artifact IDs, schemas, recipe/execution manifests, hashes, atomic writes, and safe
   resume verification.
2. Record normalized commands, resolved settings and sources, configs/fingerprints, inputs, seeds,
   code/dependency/device provenance, and allowlisted environment data.
3. Add optional compressed raw samples for AR/noise and ignore them even under landmark whitelists.
4. Move compute modes to managed Tier A artifacts and prevent them from writing docs.
5. Add `docs/publications.toml`, offline `bnb publish`, generated provenance, stale-output handling,
   and `publish --check`.
6. Test with synthetic fixtures; leave 682 backfill for S16.

Done when: fixture publication is reproducible offline and artifact safety tests, full pytest, smoke,
ruff, and publish checks pass.

### S10 — Artifact gate review

Agent: artifact reviewer, Sol extra high.

Review S09 for identity completeness, schema evolution, atomicity, partial-write recovery, resume
correctness, raw-sample retention/ignore behavior, secret exposure, offline enforcement, deterministic
rendering, and manifest ownership of every current published output.

Done when: no blocking findings remain.

### S11 — BC correction

Agent: training-profile engineer, Sol high.

Steps:

1. Encode BC independently with current RL-aligned non-objective values.
2. Preserve the explicit BC objective schedule, budget, league-disabled behavior, and
   `next_state_coef=1.0` unless separately approved.
3. Add a named allowed-difference invariant between resolved BC and RL.
4. Review the complete old/current/desired BC diff field by field.
5. Run smoke and a bounded validation appropriate to BC initialization.

Done when: every difference is named and tested, with no inheritance or accidental drift.

### S12 — Live Elo

Agent: live-Elo engineer, Sol extra high.

Steps:

1. Derive live rungs as random 0, scripted 1000, and semi-random `1000*p`.
2. Remove fitted live-ladder fields from the training configuration.
3. Rename schema/logging/chart interfaces so `live_elo` and `calibrated_elo` are distinct.
4. Keep semi-random measurement as validation rather than a training prerequisite.
5. Run a bounded validation and inspect roster/milestone differences.

Done when: tests and documentation enforce the semantic distinction and validation is recorded.

### S13 — VRAM resolution

Agent: VRAM engineer, Sol extra high.

Steps:

1. Implement the documented three-tier resolution with honest equivalence labels.
2. Apply precedence: profile/derived → cache or preset → explicit launch overrides → validation.
3. Probe in fresh subprocesses and write the cache atomically.
4. Fingerprint GPU identity/memory/MIG, software stack, model/profile, dtype, compile, and probe.
5. Record resolved values and their sources in checkpoints/artifacts and `--print-config`.
6. Mark unmeasured tiers provisional and test cache invalidation/conflicts.

Done when: focused probe/cache tests, training smoke, full pytest, and ruff pass.

### S14 — Training-systems gate review

Agent: training-systems reviewer, Sol extra high.

Review S11–S13 together for optimizer/task changes, config provenance, live-Elo league behavior,
override conflicts, memory-equivalence claims, cache invalidation, checkpoint payload completeness,
and readiness to freeze migration target schemas.

Done when: target checkpoint/config/artifact schemas are explicitly frozen for S15.

### S15 — Complete one-time 682 migration

Agent: checkpoint migration engineer, Sol extra high.

Steps:

1. Inventory and hash every final, best, recent-average, and ladder policy required for landmark
   reproduction.
2. Implement the one-off offline transformation without adding runtime migration code.
3. Migrate tensor names/shapes and known historical config/schema provenance into the frozen format;
   record unknown history honestly.
4. Produce a tracked per-file migration report with old/new hashes and transformations.
5. Compare fixed-observation logits/distributions, values, recurrent state, and next-state outputs.
6. Run seeded zero-field end-to-end equivalence scenarios.
7. Verify every migrated file loads through the ordinary strict loader.

Done when: the complete set is migrated and all equivalence/completeness checks pass.

### S16 — Landmark artifacts and publication

Agent: landmark/publication integrator, Sol high.

Steps:

1. Compute/backfill the 682 Tier A artifacts required by the publication inventory.
2. Retain compressed AR/noise raw samples locally and verify they are ignored.
3. Record exact migrated checkpoint hashes, commands, settings, seeds, and environment provenance.
4. Select exact artifacts in `docs/publications.toml`.
5. Regenerate all canonical figures, reports, and curated replays offline.
6. Run `bnb publish --check` and explain every intentional output diff.

Done when: the canonical publication set is complete, reproducible, and fully traced to 682.

### S17 — Migration/reproducibility gate review

Agent: migration/reproducibility reviewer, Sol extra high.

Independently verify the migrated file inventory, transformations, hashes, fixed-observation and
scenario equivalence, absence of runtime migrations, artifact provenance, ignored raw samples,
publication manifest coverage, and offline no-diff regeneration.

Done when: no blocking migration or reproducibility findings remain.

### S18 — Documentation and cleanup

Agent: documentation/cleanup engineer, Terra high.

Steps:

1. Sweep all reader and engineering docs for the final `bnb` commands and 4v4 defaults.
2. Link changing result provenance to the generated publication index rather than duplicating it.
3. Remove old mode names, flags, `runs/` imports, sentinel semantics, direct docs writes, and stale
   scripts/files.
4. Run repository-wide searches, docs checks, full pytest, smoke, ruff, and publish checks.

Done when: code and docs expose only the final system and the worktree is clean.

### S19 — Final branch acceptance

Agent: final branch reviewer, Sol extra high.

Review the complete branch against `main` and every locked decision. Run the full test suite, smoke
matrix, ruff, strict CLI checks, config snapshot/invariant checks, artifact integrity checks,
migration equivalence checks, and offline publication check. Inspect for unrelated changes, stale
temporary paths, missing provenance, unsafe defaults, and undocumented behavioral differences.

Done when: no blocking findings remain and the branch is ready for human review/merge. Record final
evidence here; do not merge.

## Handoff records

Each section appends its record below when it completes. Do not replace earlier evidence.

### S00 handoff

- Status: completed
- Agent/model/effort: planning lead / `gpt-5.6-sol` / extra high
- Commit(s): `6649f72` — planning baseline; followed by a status-only closure commit
- Tests/checks: branch/base verification; plan/ledger consistency searches; trailing-whitespace check;
  clean-worktree check after closure
- Behavior/config changes: documentation/governance only
- Decisions/deviations: primary section owners are user-launched sequentially; section owners may
  orchestrate bounded subagents and concurrent writers, with safe file/worktree ownership
- Remaining risks: none; S01 is the next authorized section

### S01 handoff

- Status: completed
- Agent/model/effort: characterization engineer / `gpt-5.6-terra` / high
- Commit(s): `1d47502` — mode/config/CLI/publication characterization baseline
- Tests/checks and results: `uv run pytest tests/test_mode_refactor_baseline.py -q` (10 passed);
  `uv run pytest -q` (601 passed); `uv run ruff check .` (passed); `git diff --check` (passed)
- Behavior/config changes: none. RL and RL-fields are recorded as exact legacy baselines; the
  separate `bc-stale` snapshot deliberately records the pre-correction BC configuration for S11.
- Files/artifacts produced: `docs/internal/mode-characterization.json`; three complete profile
  snapshots under `tests/fixtures/mode_refactor/`; `tests/test_mode_refactor_baseline.py`.
- Decisions/deviations from plan: schedule callables are captured by module/qualified name plus
  their closure construction values, which preserves the complete current schedule definition
  before a resolver exists. The inventory records the sole untraceable asset producer honestly:
  `docs/policy_architecture.png` is not recorded in this repository.
- Review findings addressed: Ruff import/line-length findings in the new test were fixed before
  the full suite.
- Remaining risks or required follow-up: the legacy snapshots intentionally retain old names,
  defaults, sentinels, and mtime selection until the later breaking sections replace them. S02
  must keep RL/RL-fields field-for-field equal to these snapshots; S11 owns the BC behavior change.

### S02 handoff

- Status: completed
- Agent/model/effort: configuration architect / `gpt-5.6-sol` / extra high
- Commit(s): `b6f9c53` — independent profiles and resolved-configuration foundation; followed by
  this status-only closure commit
- Tests/checks and results: `uv run pytest tests/config/test_resolution.py
  tests/test_mode_refactor_baseline.py tests/test_main.py -q` (38 passed); `uv run pytest -q`
  (620 passed); `uv run ruff check .` (passed); `git diff --check` (passed); wheel build to `/tmp`
  included every new config/profile module; installed-package import from outside the checkout
  returned exactly `bc`, `rl`, and `rl-fields`; repository search found no remaining `runs` import
  or moved-path reference
- Behavior/config changes: no training behavior change. Resolved RL, RL-fields, and stale BC match
  their S01 snapshots exactly. The resolved RL-to-RL-fields diff is limited to the named existing
  field intent: field map, field reward weights, field count/aligned env count, reference ladder,
  and random Elo. BC remains intentionally stale for S11.
- Files/artifacts produced: packaged independent specs under `boost_and_broadside.profiles`; config
  schema/defaults/resolver/declarative schedules/canonical fingerprints/service modules; source and
  dependency guards in `tests/config/test_resolution.py`; top-level `runs/` removed; moved-path
  imports and documentation links updated
- Decisions/deviations from plan: schedule intent is declarative and compiled into the unchanged
  runtime closures, so durable fingerprints never depend on callable identity/module names. The
  profile fingerprint excludes legacy rollout/microbatch machine presets; the resolved fingerprint
  includes their final values. Sources are recorded separately, so an explicit CLI override equal
  to the resolved default keeps the same value fingerprint while recording `cli`. The legacy
  `bc_warmstart` pretrain stage remains an unregistered independent transitional spec until S08.
- Review findings addressed: exact one-ULP discount drift was eliminated with decimal exponentiation;
  every durable resolved leaf has a closed-vocabulary source; profile modules are AST-guarded from
  importing one another or runtime engines; invalid post-override launches fail validation; schemas
  and current fingerprints have explicit versioned golden tests
- Remaining risks or required follow-up: legacy callers still consume the resolved wrapper's
  field-compatible `TrainConfig` projection; S05/S13 must thread the wrapper and source/fingerprint
  document into launch/checkpoint provenance as planned. Scripted-agent defaults still live outside
  the training fingerprint and must be recorded by the artifact/live-Elo provenance work. S11 owns
  the BC correction. `scripts/bench_mem.py` retains pre-existing stale `replace()` fields unrelated
  to the import move.

### S03 handoff

- Status: completed; blocking findings require S03R before S04
- Agent/model/effort: configuration reviewer / `gpt-5.6-sol` / extra high
- Commit(s) reviewed: `1d47502`, `b9581c4`, `b6f9c53`, and `d34c39b` in committed range
  `76b4a00..d34c39b`; followed by this review-ledger commit
- Tests/checks and results: `uv run pytest tests/config/test_resolution.py
  tests/test_mode_refactor_baseline.py tests/test_main.py -q` (38 passed); `uv run pytest -q`
  (620 passed); `uv run ruff check .` (passed); `git diff --check 76b4a00..d34c39b` (passed);
  wheel build contained every config/profile module; installed import outside the checkout returned
  exactly `rl`, `rl-fields`, and `bc`; RL fingerprints matched across distinct `PYTHONHASHSEED`
  values; direct mutation, launch-sizing, dependency, and stale-path probes recorded below
- Behavior/config changes: review made no product-code change. Default resolved RL, RL-fields, and
  stale BC remain exact S01 snapshot matches, and the named RL-to-RL-fields differences remain
  limited to existing field intent. Registered profiles are independent values and have no
  cross-profile or runtime-engine imports.
- Files/artifacts produced: ledger update only; wheel and probe output were temporary under `/tmp`
- Decisions/deviations from plan: S03 does not approve the configuration foundation. S03R is
  inserted immediately after this gate; S04 and every later section remain pending.
- Review findings addressed: none; review agents do not edit product code
- Blocking findings:
  1. `ResolvedTrainConfig` is documented as immutable at
     `src/boost_and_broadside/config/schema.py:148`, but `TrainConfig.component_gammas` and
     `component_lambdas` are mutable dictionaries at
     `src/boost_and_broadside/config/training.py:255`. Mutating a resolved mapping changed the
     document emitted by `resolved_profile_document` (`config/service.py:14`) while its retained
     `resolved_config_fingerprint` stayed unchanged. The direct probe changed `ally_win` to `0.5`
     and reported `config_changed=True` with `fingerprint_unchanged=True`. This makes checkpoint or
     artifact provenance capable of asserting a hash for different configuration values, and no
     test covers deep immutability or document/hash consistency.
  2. `resolve_profile` derives `rollouts_per_update` from the profile launch preset before applying
     an explicit `num_envs` override (`config/resolve.py:279` versus `config/resolve.py:316`). A
     valid override from 3904 to 1952 environments retained three shards and passed validation,
     reducing effective update tokens from 11,993,088 to 5,996,544 against the declared 12,000,000
     logical budget. This turns a documented launch-sizing/VRAM control into a silent optimization
     change and violates the fixed-logical-budget shard contract; the existing override test at
     `tests/config/test_resolution.py:145` changes width only slightly and does not assert budget or
     shard behavior.
  3. The profile fingerprint payload removes only `launch_defaults` at
     `src/boost_and_broadside/config/resolve.py:74`, so it includes
     `ModelConfig.grad_checkpoint` (`config/core.py:248`). A direct probe changing only
     `grad_checkpoint=False` to `True` changed `profile_fingerprint` even though the resolved
     `TrainConfig` was identical. The plan categorizes gradient checkpointing as a machine/VRAM
     setting with the same mathematical objective, while `profile_fingerprint` must represent
     semantic experiment intent and exclude hardware choices. Existing tests cover rollout and
     microbatch presets but not this machine field.
  4. S02 deleted top-level `runs/` and its handoff says no moved-path reference remains, but live
     references still exist at `main.py:3`, `docs/getting-started.md:107`,
     `docs/getting-started.md:180`, `STYLE_GUIDE.md:106`, `STYLE_GUIDE.md:124`, and
     `src/boost_and_broadside/train/rl/sigreg.py:8`. The two reader-facing links target a deleted
     directory, and the style guidance now names a nonexistent authoritative location. S02 updated
     other references in the same files, so these are incomplete migration regressions rather than
     intentionally deferred CLI cleanup.
- Remaining risks or required follow-up: S03R must close all four findings and receive independent
  re-review before S04 starts. S05/S13 still own the planned CLI and broader VRAM/cache integration;
  S11 still owns the intentional BC correction.

### S03R handoff

- Status: completed
- Agent/model/effort: configuration remediator plus independent review owner / `gpt-5.6-sol` /
  extra high
- Commit(s): `0255c64` — close the four S03 configuration blockers; `98e81f5` — enforce exact
  fixed aligned logical batches after independent review; followed by this status-only closure
  commit
- Tests/checks and results: reproduced all four original probes before product edits; focused
  config/characterization/dispatch tests (45 passed); checkpoint compatibility tests (50 passed,
  2 CUDA skips); final `uv run pytest -q` (621 passed, 6 CUDA skips); `uv run ruff check .`
  (passed); `git diff --check` for both the S03R follow-up and `76b4a00..98e81f5` ranges (passed);
  wheel contained every config/profile module and an install under `/tmp` imported exactly `bc`,
  `rl`, and `rl-fields`; fingerprints matched across distinct `PYTHONHASHSEED` values; independent
  re-review focused tests (36 passed) and probes reported no blocking finding
- Behavior/config changes: default resolved RL, RL-fields, and stale BC remain exactly equal to
  their S01 snapshots, and all resolved-config fingerprints are unchanged. A valid half-width RL
  launch now resolves from 3,904 environments/3 shards to 1,952 environments/6 shards, preserving
  the characterized 11,993,088-token aligned logical batch. Widths that cannot preserve that batch
  with an integer shard count now fail loudly. Profile fingerprints intentionally changed for all
  profiles because machine-only `grad_checkpoint` is excluded; the complete resolved fingerprint
  still includes it.
- Files/artifacts produced: immutable resolved component-discount mappings with checkpoint/W&B
  serialization support; exact shard-count derivation and validation; regression coverage for
  mutation/document consistency, preserving and divergent widths, gradient-checkpoint fingerprint
  boundaries, and deleted `runs/` profile-path references; stale references corrected in the legacy
  entry-point docstring, getting-started guide, style guide, and SIGReg documentation; build and
  installation artifacts were temporary under `/tmp`
- Decisions/deviations from plan: environment alignment makes the characterized default RL batch
  11,993,088 entity tokens versus the 12,000,000 nominal intent. Launch overrides preserve that
  aligned batch exactly; a width without an integer preserving shard count is a contradictory input
  and is rejected. The broader VRAM preset/probe/cache system remains reserved for S13.
- Review findings addressed: all four S03 blockers are closed. The first independent pass found
  nearest-integer shard rounding still admitted +32.7% and +96.6% divergent batches; `98e81f5`
  replaced rounding with exact divisibility, added 3,872/7,776/23,040 rejection coverage, and the
  independent re-review reported no remaining blocker.
- Remaining risks or required follow-up: S05/S13 still must thread the complete resolved document
  and source map into launch/checkpoint provenance and implement the planned CLI/VRAM cache layers;
  S11 still owns the intentional BC correction. S04 is the next authorized section.

### S04 handoff

- Status: completed
- Agent/model/effort: evaluation refactorer / `gpt-5.6-sol` / high
- Commit(s): `3cac020` — mark S04 active; `d41f340` — shared evaluation primitives,
  mode integration, required TeamPMA inputs, and focused tests; followed by this status-only closure
  commit
- Tests/checks and results: characterization/evaluation/mode gate (107 passed); expanded focused
  policy/roster/mode gate (244 passed, 2 CUDA skips); final `.venv/bin/pytest -q` (652 passed,
  6 CUDA skips); `.venv/bin/ruff check .` (passed);
  `git diff --check` (passed); setuptools package discovery included
  `boost_and_broadside.evaluation`; direct imports of every evaluation module passed; repository
  guards found no library `sys.exit`, cross-user-facing-mode import, omitted `team_pma_k` policy
  constructor, or stale `modes.agent_factory`/`modes.match` import
- Behavior/config changes: resolved RL, RL-fields, and stale-BC snapshot characterization remains
  exact; no training configuration or successful mode output contract changed. Matchup parsing now
  produces a typed tuple-compatible `Matchup`, preserves asymmetric inputs, rejects non-positive or
  malformed inputs, and exposes the locked 4v4 default. Final/resumable checkpoint policies use
  numeric step metadata rather than filename lexicography or mtime. Library failures raise typed or
  ordinary exceptions rather than terminating the process. `team_pma_k` is now required by
  `YemongPolicy`, `build_policy`, and roster policy loading, with every caller explicit.
- Files/artifacts produced: packaged `boost_and_broadside.evaluation` modules for agents,
  field-aware environment construction, match execution, next-state imagination, typed run catalog,
  typed sizes, and tournaments; focused tests under `tests/evaluation/`; no durable runtime artifact
- Decisions/deviations from plan: pre-S05 `latest`/`none` success behavior remains behind explicitly
  named legacy catalog adapters so S04 stays mechanically comparable while strict exact-run APIs are
  ready for the CLI break. Field-bearing environments require an explicit `FieldMapConfig` or an
  existing cache; S04 does not invent a map distribution when provenance is unavailable. Plot and
  history helpers remain non-user-facing mode-adjacent renderers; the dependency guard forbids only
  imports between modules that expose user-facing `run_*` entry points.
- Review findings addressed: self-review caught and removed the roster loader's remaining empty
  `team_pma_k` default; missing checkpoint roots now raise typed catalog errors; the shared
  tournament APIs were made public rather than leaving cross-mode private imports.
- Remaining risks or required follow-up: S05 must remove the legacy run/sentinel adapters and
  translate catalog exceptions through the strict `bnb` CLI. S08 still owns threading field-map
  intent into `collect-stats` and the planned mode consolidation; until then field configs without an
  explicit map source continue to fail loudly rather than silently selecting a different map
  distribution.

### S05 handoff

- Status: completed
- Agent/model/effort: CLI engineer / `gpt-5.6-sol` / high
- Commit(s): `21bbf84` — mark S05 active; `63baa0d` — installed `bnb` parser, runtime
  adapters, exact-subject break, checkpoint provenance, and contract tests; `0bd735a` — update
  command documentation; followed by this status-only closure commit
- Tests/checks and results: focused CLI/config/catalog/checkpoint gate (120 passed, 2 CUDA skips);
  final `.venv/bin/pytest -q` (694 passed, 6 CUDA skips); `.venv/bin/ruff check .` (passed);
  `git diff --check` (passed); installed `uv run --no-sync bnb` and `bnb --help` returned help with
  no dispatch; installed missing-resume, sentinel-run, and resume/pretrain-conflict probes exited 2
  at parsing; `uv build --wheel --no-build-isolation` succeeded and included `cli.py`,
  `cli_commands.py`, and `entry_points.txt`
- Behavior/config changes: the CLI break is intentional and complete: root `main.py`, `--mode`,
  underscore spellings, pathless resume, implicit/default RL, hardcoded landmark runs, and magic
  `latest`/`none` selection are gone. `train --profile bc|rl|rl-fields` resolves the existing
  independent profiles; resume selects the greatest numeric step only within an exact named run and
  is exclusive with `--pretrain-from`. Collect-stats, feature-stats, and noise-calibration now use
  the locked 4v4 default, and capture scratch defaults to `out/`. Resolved RL, RL-fields, and stale
  BC remain exact S01 snapshot matches. Production training checkpoints now carry the complete
  resolved config/fingerprints/source map plus actual device, seed, compile, W&B, and drift settings;
  mismatched resolved configs fail resume unless drift is explicitly allowed, in which case they warn.
- Files/artifacts produced: packaged `boost_and_broadside.cli` registry and lazy
  `cli_commands` adapters; generated parser/help/error/ownership tests in `tests/test_cli.py`;
  reader-facing `bnb` examples and entry-point guidance; temporary wheel under `/tmp` only
- Decisions/deviations from plan: parser construction imports only the pure profile registry and
  defers Torch/mode engines until dispatch, so help and `--print-config` do not construct a trainer
  or environment. `smoke` and `publish` are present in the final command registry but fail explicitly
  as unavailable until S06 and S09 implement their assigned systems. Artifact-backed Elo reanalysis,
  VRAM policy/cache options, arbitrary-agent calibration, and removal of the transitional
  `elo_stats`/legacy warmstart modules remain with S09, S13, and S08 respectively. Existing AR
  two-scenario orchestration remains unchanged for S08's canonical 4v4 consolidation.
- Review findings addressed: self-review added full resolved-config checkpoint provenance and loud
  resume-drift validation, replaced a fixed default RNG seed with the recorded process-generated
  PyTorch seed when `--seed` is omitted, enforced the 4v4 default at the remaining CLI-owned analysis
  seams, removed every transitional mtime/sentinel catalog adapter, and verified invalid devices are
  translated into concise CLI errors
- Remaining risks or required follow-up: S06 must replace the registered smoke placeholder with the
  isolated subprocess matrix. S08 still owns arbitrary-agent Elo calibration, deletion of
  `elo_stats` and legacy warmstart code, AR orchestration consolidation, field-intent threading, and
  dead mode-argument cleanup. S09 must replace current compute-to-doc behavior and the publish
  placeholder with artifacts/offline publication; S13 owns `--vram` probing/cache resolution.

### S06 handoff

- Status: completed
- Agent/model/effort: smoke/test engineer / `gpt-5.6-sol` / high
- Commit(s): `2fc844e` — mark S06 active; `ff26dab` — production payload builders,
  synthetic current-schema run, isolated smoke registry/runner, CLI selection, and tests; followed
  by this status-only closure commit
- Tests/checks and results: focused fixture/checkpoint gate (36 passed, 2 CUDA skips); final
  `.venv/bin/pytest -q` (715 passed, 6 CUDA skips); clean-checkout `.venv/bin/bnb smoke` (all 14
  sequential subprocess cases passed); `.venv/bin/ruff check .` (passed); range `git diff --check`
  (passed); wheel build included `smoke.py`, `cli.py`, `train/rl/checkpoint.py`, and the `bnb` entry
  point; `bnb smoke --help` exposed only the focused diagnostic `--case` modifier
- Behavior/config changes: `bnb smoke` now runs the full registry-owned matrix; focused diagnosis
  uses `bnb smoke --case NAME`. Normal mode behavior and all resolved RL/RL-fields/stale-BC
  configurations remain unchanged. Production checkpoint serialization was mechanically factored
  through pure policy and full-resume payload builders plus the existing atomic temp-and-replace
  path; payload keys and ordinary save/load behavior remain covered and unchanged.
- Files/artifacts produced: packaged `boost_and_broadside.smoke`; a deterministic synthetic run
  builder with a resumable random-policy `step_*.pt`, same-step ladder policy, minimal
  `roster.json`, and `elo_history.jsonl`; generated coverage in `tests/test_smoke.py`; no durable
  runtime artifact (all smoke roots are temporary and cleaned)
- Decisions/deviations from plan: the 14 cases cover every current runtime handler, with separate
  cases for `train` profiles `rl`, `rl-fields`, and `bc`. `smoke` is the orchestrator rather than a
  recursive case, and `publish` remains the explicit S09 placeholder rather than being treated as a
  successful runtime mode. Each child starts from empty checkpoint/artifact/scratch roots, runs on
  CPU with fixed seeds and cache paths, disables publication/report rendering, and is checked for
  root escapes and checkout changes. The synthetic ladder snapshot is required by Elo-scale's
  production tournament contract and uses the same current payload as the final step.
- Review findings addressed: self-review upgraded the initial policy-only fixture into a fully
  resumable `step_*.pt` and proved ordinary trainer reload; moved heavy mode imports into child
  execution so the parent runner cannot create matplotlib/pygame cache droppings; prohibited PNG
  and Markdown smoke outputs; resolved repository discovery through Git so an installed `bnb`
  executable works from the checkout; verified the capture case through the real ffmpeg path
- Remaining risks or required follow-up: S07 is the next authorized integration review. The capture
  smoke case exercises the existing external ffmpeg runtime dependency and will fail loudly where
  ffmpeg is unavailable. S09 must add publication implementation and its separate offline checks.

### S07 handoff

- Status: completed; blocking findings require S07R before S08
- Agent/model/effort: integration reviewer with independent S04, S05, and S06 audit tracks /
  `gpt-5.6-sol` / extra high
- Commit(s) reviewed: `3cac020`, `d41f340`, `0735f70`, `21bbf84`, `63baa0d`,
  `0bd735a`, `02271a4`, `2fc844e`, `ff26dab`, and `f8d9f4c` in committed range
  `c778ef5..f8d9f4c`; `c621f63` marked S07 active, followed by this review-ledger commit
- Tests/checks and results: `.venv/bin/pytest -q` (715 passed, 6 skipped); isolated
  `.venv/bin/bnb smoke` (all 14 sequential cases passed); local `.venv/bin/ruff check .`
  (passed), but Ruff against a clean `git archive f8d9f4c` failed I001 at
  `train/rl/logging.py:324` because the working checkout's ignored `wandb/` directory changes import
  classification; `git diff --check c778ef5..f8d9f4c` failed on four extra EOF blank lines; wheel
  build with a temporary UV cache succeeded and an unpacked wheel printed installed `bnb --help`
  from outside the checkout; exact-run, provenance, loader-ordering, CLI-error, fixture, and smoke
  escape/timeout probes recorded below
- Behavior/config changes: review made no product-code or configuration change. Full tests and the
  happy-path smoke matrix remain green, but the negative-path probes demonstrate unsafe resume,
  evaluation, CLI, and isolation behavior that prevents approval.
- Files/artifacts produced: ledger updates only; wheel, clean archive, installed-package unpack,
  synthetic runs, and all direct probes were temporary under `/tmp`
- Decisions/deviations from plan: S07 does not approve the shared/CLI/smoke gate. S07R is inserted
  immediately after this section and S08 plus every later primary section remain pending.
- Review findings addressed: none; review agents did not edit product code
- Blocking findings:
  1. `build_players` appends the numerically final checkpoint as `ckpt_<step>` when that step is not
     already a roster milestone (`evaluation/tournament.py:263`), and Elo-scale appends the same
     checkpoint again as `final` (`modes/elo_scale.py:165`) while metadata contains only `final`
     (`modes/elo_scale.py:131`). A synthetic valid run with no final-step ladder entry failed at
     `run_elo_scale_mode` (`modes/elo_scale.py:339`) with `loaded tournament field does not match
     stored metadata`. S06's fixture masks the defect by making its ladder and final step identical.
  2. Exact-run ladder selection accepts `Path(entry["path"])` whenever that foreign path exists
     (`evaluation/run_catalog.py:123`); Elo-scale repeats the rule in `_checkpoint_path`
     (`modes/elo_scale.py:124`). A probe for exact run `exact/` selected `other/foreign.pt`, so a
     copied/stale roster can silently evaluate another run and invalidate subject provenance. No
     test asserts containment, step identity, or recorded file identity.
  3. Complete resolved-config drift enforcement is attached to `load_pretrained_weights`
     (`train/rl/checkpoint.py:501`) rather than `load_checkpoint` (`checkpoint.py:539`). A
     BC-fingerprinted full checkpoint resumed into an RL-fingerprinted trainer with drift disabled,
     while the documented BC-to-RL pretraining handoff was rejected for the same fingerprint
     difference. Tests at `tests/train/test_checkpoint.py:278` call only the helper and never either
     loader method.
  4. `_train` constructs `PPOTrainer` before resolving `--pretrain-from`
     (`cli_commands.py:128`). A missing-path probe reported
     `TRAINER_CONSTRUCTED_BEFORE_PRETRAIN_VALIDATION=True`; the real constructor can allocate field
     maps, environments, policy/optimizer state, and rollout buffers before rejecting invalid input.
     The existing stub test uses an already present file and does not cover ordering.
  5. Strict CLI validation and failure translation are incomplete. The print-config path is outside
     the error-translation block (`cli.py:404`): an invalid aligned `--num-envs 3872` exited 1 with a
     full traceback, and corrupt explicit checkpoints can leak loader exceptions. `--resume latest`
     and `--resume none` also parse and exit 0 when combined with `--print-config`, despite D13's
     parser-error contract, while invalid device/execution settings are neither validated nor shown
     by the purported complete printed launch. Tests cover one runtime `ValueError` and successful
     print-config only (`tests/test_cli.py:184`).
  6. The typed matchup contract is undone by `run_collect_stats_mode`, which catches
     `MatchupParseError`, prints `Skipping`, and returns success (`modes/collect.py:43`). Installed
     `bnb collect-stats --team0 scripted --team1 random --sizes 0v4 --device cpu` exited 0 without
     running a game. This violates the plan's reject-rather-than-skip rule and has no CLI/mode
     regression test.
  7. Smoke isolation observes only descendants of the case root (`smoke.py:310`) and ordinary Git
     status/diff (`smoke.py:632`), so a sibling escape and an ignored real `checkpoints/escaped.pt`
     both went undetected. Timeout/isolation exceptions continue before the checkout comparison
     (`smoke.py:690`); a temp-repository probe changed a tracked file, raised `TimeoutExpired`, and
     left `TIMEOUT_CHANGE_SURVIVED=True`. The child is not placed in a process group, so a timeout
     also need not terminate capture's ffmpeg descendant. Existing tests create only an unexpected
     child inside the root and a PNG inside artifacts.
  8. The synthetic ladder writes the full resume payload (`smoke.py:245` and `smoke.py:272`) instead
     of production's policy-only ladder provenance: the direct probe reported identical ladder/step
     key sets and `LADDER_HAS_OPTIMIZER=True`. Smoke tests inspect only the resumable checkpoint.
     In addition, cases branch directly to mode functions (`smoke.py:387`) rather than invoking the
     registered CLI handlers; the registry test proves only equal command-name sets, so adapter
     wiring can regress while every smoke case remains green.
  9. The committed milestone is not cleanly reproducible. Ruff passes only while the ignored local
     top-level `wandb/` makes `wandb` look first-party; a clean archive fails the pre-existing import
     block at `src/boost_and_broadside/train/rl/logging.py:324`. Range whitespace verification exits
     2 for extra EOF blank lines in `evaluation/__init__.py`, `evaluation/sizes.py`,
     `tests/evaluation/test_sizes.py`, and `tests/evaluation/test_tournament.py`, contrary to the S04
     handoff's recorded green check.
- Remaining risks or required follow-up: S07R must close all nine findings and receive independent
  re-review before S08 begins. S08 still owns the planned mode consolidation/retirements and field
  threading; S09 and later sections retain their existing artifact, training, migration, and
  publication scopes.

### S07R handoff

- Status: completed
- Agent/model/effort: integration remediator plus independent read-only reviewer /
  `gpt-5.6-sol` / extra high
- Commit(s): `e76aa38` — mark S07R active; `4243c55` — close the nine shared/CLI/smoke
  blockers; `a13a94b` — harden independently identified device and checkpoint error boundaries;
  `04d97cb` — validate checkpoint state mappings; followed by this status-only closure commit
- Tests/checks and results: all nine negative probes reproduced before remediation; focused
  checkpoint/CLI tests passed (108 passed, 2 skipped); final full `.venv/bin/pytest -q` passed
  (742 passed, 6 skipped); `.venv/bin/bnb smoke` passed all 14 isolated cases and reported the
  checkout unchanged; local and clean-archive Ruff passed; `git diff --check c778ef5..04d97cb`
  passed; a wheel built successfully, contained `execution.py` and the `bnb` entry point, and its
  unpacked package printed `bnb --help` outside the checkout. The independent reviewer separately
  passed a 184-test focused suite (2 skipped), the full suite (742 passed, 6 skipped), all 14 smoke
  cases, clean-archive Ruff/wheel/help, range whitespace, and direct invalid-subject probes, then
  reported no remaining blocker.
- Behavior/config changes: intended strict behavior changes only: Elo-scale uses one verified final
  policy; roster ladders cannot escape or misidentify the exact run; full resume enforces complete
  resolved-config provenance while BC-to-RL pretraining remains allowed; invalid subjects,
  matchups, devices, corrupt checkpoints, and incompatible checkpoint state fail concisely before
  expensive allocation. Printed resolved config now includes validated execution settings. No
  profile source or resolved training value changed, and the S01 behavior/config characterizations
  remain exact.
- Files/artifacts produced: shared evaluation/CLI/checkpoint/smoke code and regression tests;
  `src/boost_and_broadside/execution.py` centralizes validated execution settings. Clean archive,
  wheel, unpacked-package, synthetic-run, and probe artifacts were temporary under `/tmp`; the
  build backend's inspected checkout-local `build/` output was removed.
- Decisions/deviations from plan: roster entries are portable basenames resolved only beneath the
  selected run; a true final checkpoint replaces a same-step ladder entry and keeps the `final`
  identity; smoke fixtures use a policy-only ladder at a distinct step, traverse the registered CLI
  adapter exactly once, redirect mutable home/cache state, compare sibling/ignored output and Git
  state on every exit path, and terminate the subprocess process group on timeout. Independent
  review added availability/index checks for supported device backends plus narrow normalization of
  malformed state-dict inputs without widening the global exception catch.
- Review findings addressed: all nine S07 blockers, plus the independent follow-up findings for
  unavailable non-CUDA backends, incompatible loadable state dictionaries, and non-mapping state
  payloads. Final independent disposition: no blocking findings remain in
  `c8e3c04..04d97cb`; S07R is ready to close.
- Remaining risks or required follow-up: accelerator-positive execution paths were reviewed
  statically on a CPU-only host; CUDA/MPS/XPU hardware was unavailable. Real ffmpeg capture passed;
  forced process-tree timeout behavior is covered by focused tests. S08 remains the next authorized
  section and was not begun.

### Future handoff template

```text
### <SECTION_ID> handoff

- Status: completed | blocked
- Agent/model/effort:
- Commit(s):
- Tests/checks and results:
- Behavior/config changes:
- Files/artifacts produced:
- Decisions/deviations from plan:
- Review findings addressed:
- Remaining risks or required follow-up:
```
