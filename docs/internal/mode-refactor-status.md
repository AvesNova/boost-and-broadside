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

- Active section: none; no implementation agent has been started.
- Next section: `S01` — characterization baseline.
- Blocking issue: none.
- Landmark migration: scheduled for `S15`, after all target schemas stabilize.

## Sequential queue

| ID | Plan phase | Status | Agent type | Model / effort | Mission |
|---|---:|---|---|---|---|
| S00 | governance | completed | planning lead | Sol / extra high | Finalize plan, branch, ledger, and prompts |
| S01 | 0 | pending | characterization engineer | Terra / high | Capture behavior, config, CLI, and publication baselines |
| S02 | 1 | pending | configuration architect | Sol / extra high | Move profiles under `src`; add independent specs/resolution/fingerprints without changing RL behavior |
| S03 | 1 gate | pending | configuration reviewer | Sol / extra high | Review resolved-config equivalence, dependency direction, and schema/fingerprint design |
| S04 | 2 | pending | evaluation refactorer | Sol / high | Extract typed sizes, run catalog, match/environment, and tournament engines |
| S05 | 3 | pending | CLI engineer | Sol / high | Replace `main.py --mode` with the strict installed `bnb` subcommand CLI |
| S06 | 4 | pending | smoke/test engineer | Sol / high | Build synthetic checkpoint fixtures and fully isolated sequential subprocess smoke coverage |
| S07 | 2–4 gate | pending | integration reviewer | Sol / extra high | Review shared engines, CLI contracts, smoke isolation, and behavior preservation |
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
