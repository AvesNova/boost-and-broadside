# Mode system refactor — plan

Working document for the CLI/mode/configuration restructure. Delete when the work lands.

Goal: replace the repository-local `main.py --mode ...` dispatcher with a small installed
`bnb` CLI; make modes one coherent system with explicit inputs, independent profiles,
hermetic smoke coverage, reproducible artifacts, honest output locations, and no
hand-transcribed project constants.

The refactor deliberately makes a clean CLI break. There are no hidden aliases, magic
sentinels, or compatibility shims for the old command line.

---

## 1. Locked decisions

| # | Decision | Consequence |
|---|---|---|
| D1 | **Eight total ships — 4v4 — is the default** for every applicable mode except `play` (human 1v1 duel) | `EnvConfig.num_ships=8`; textual matchup defaults are `4v4`, never `8v8` |
| D2 | **`bc_warmstart` is dropped entirely** | BC and RL are separate `train` invocations joined explicitly with `--pretrain-from` |
| D3 | **Training modes collapse to `train --profile {rl,rl-fields,bc}`** | Profiles are registry entries, not modes; fields remain a profile choice, never a flag |
| D4 | **Evaluation modes remain distinct and share engines** | `elo-calibrate`, `elo-scale`, and `semi-random` remain peers; reusable match/tournament machinery moves outside `modes/` |
| D5 | **`elo-stats` retires after arbitrary agents are supported by `elo-calibrate`** | Preserve arbitrary-agent Bradley–Terry round robins before deleting the sequential K-factor implementation |
| D6 | **Smoke is a first-class subprocess test system** | `bnb smoke` runs every registered smoke case in an isolated process and temporary artifact root; it never writes to `checkpoints/` or `docs/` |
| D7 | **Live training Elo uses a deliberately approximate fixed ladder** | random = 0, scripted = 1000, semi-random rung = `1000·p`; this is for live curriculum/rating stability, not a claim that the rungs are post-hoc calibrated |
| D8 | **Published Elo is post-hoc calibrated Elo** | Published results use Bradley–Terry calibration from match counts; live Elo and calibrated Elo are named and documented distinctly |
| D9 | **`_ROLLOUT_TOKENS` is an allowed VRAM knob** | At fixed logical token budget it changes shard count and sampling composition; the resolved shard count is recorded and reported |
| D10 | **VRAM probing writes a reusable local cache** | `.vram.json` is gitignored and fingerprinted; `--vram auto` reads it and `--vram probe` writes it |
| D11 | **Durable outputs are artifacts; canonical docs are manifest-selected views** | Compute modes write versioned artifacts, `bnb publish` is the only docs writer, and `docs/publications.toml` selects exact sources |
| D12 | **The CLI is an installed `bnb` executable with subcommands** | Add a project script entry point; remove root `main.py`, `--mode`, underscore command names, old flag aliases, and the default-to-RL behavior |
| D13 | **No magic run sentinels or implicit landmark run** | Remove `"none"`, `"latest"`, implicit 682, and pathless `--resume`; required subjects fail at argument parsing |
| D14 | **All Python code lives under `src/`** | Replace top-level `runs/` with `src/boost_and_broadside/profiles/`; installed and repository execution use the same imports |
| D15 | **Profiles do not inherit from one another** | RL, RL-fields, and BC independently declare intent; a resolver composes project defaults, derived values, machine settings, and explicit launch overrides |
| D16 | **Explicit user launch overrides have the highest normal precedence** | profile intent → derived defaults → VRAM cache/preset → explicit CLI launch overrides → validation; contradictions fail loudly |
| D17 | **Raw analysis samples are retained when practical but never git-tracked** | An artifact contains publishable aggregates plus an optional ignored `samples/` payload for future reanalysis |
| D18 | **Canonical publication is clean and offline** | Publishing never simulates or contacts W&B; canonical sources must come from clean committed code/artifacts and are verified with `publish --check` |
| D19 | **The complete 682 checkpoint set is migrated once, offline, after the target schemas stabilize** | Every final/best/recent/ladder policy needed for reproduction is updated; there is no generic loader migration or support for arbitrary legacy checkpoints |
| D20 | **Top-level section owners are launched sequentially** | The user starts one primary implementation/review section at a time; its owner may use subagents or concurrent writers within that section, but the next primary section waits for a committed, tested, reviewed-as-required handoff |

### Live Elo contract (D7/D8)

Training needs cheap, stationary references before the live policy can reliably beat scripted.
It therefore uses an intentionally approximate anchored scale:

- uniform random is fixed at live Elo 0;
- the stochastic scripted controller is fixed at live Elo 1000;
- a semi-random controller that uses the full scripted action with probability `p` is assigned
  live Elo `1000·p`;
- above scripted, normal live Elo updates continue on the same numerical scale.

This approximation is rough at the weak end and sufficiently accurate near scripted:

| p | `rl` fitted, regauged | `rl-fields` fitted, regauged | linear | Δ rl | Δ fields |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 93.8 | 122.7 | 200 | −106.2 | −77.3 |
| 0.3 | 196.3 | 219.3 | 300 | −103.7 | −80.7 |
| 0.4 | 351.1 | 348.9 | 400 | −48.9 | −51.1 |
| 0.5 | 465.3 | 481.8 | 500 | −34.7 | −18.2 |
| 0.6 | 604.9 | 604.6 | 600 | +4.9 | +4.6 |
| 0.7 | 698.8 | 716.0 | 700 | −1.2 | +16.0 |
| 0.8 | 804.2 | 797.5 | 800 | +4.2 | −2.5 |
| 0.9 | 898.3 | 930.7 | 900 | −1.7 | +30.7 |
| 0.95 | 957.8 | 987.2 | 950 | +7.8 | +37.2 |

The live value is a training instrument: opponent sampling, continuous progress reporting,
and league milestone placement. It must be logged as `live_elo` and described as approximate.
Post-hoc `calibrated_elo` fits actual match outcomes and is the source for published results.
The two should never be silently substituted in artifacts, chart labels, or prose.

Changing the live ladder is still behavioral: `elo_milestone_gap=200` decides when frozen
checkpoints enter the league. Validate the linear ladder with a short run and roster diff even
though the approximation itself is accepted.

---

## 2. CLI contract

### Entry point

`pyproject.toml` registers:

```toml
[project.scripts]
bnb = "boost_and_broadside.cli:main"
```

Canonical usage is:

```bash
uv run bnb train --profile rl
uv run bnb capture --run resilient-resonance-682 --sizes 4v4 --seeds 0
uv run bnb collect-stats --team0 scripted --team1 random --sizes 4v4
uv run bnb publish --check
uv run bnb smoke
```

`uv run` still selects the locked project environment; `bnb` invokes the installed package
entry point instead of executing a repository file. With the virtual environment already
active, `bnb ...` works directly.

Calling `bnb` with no subcommand prints help and exits without side effects. Calling a command
without a required run/profile/input is an argparse error. No command defaults to training.

### Final command list

`train`, `play`, `watch`, `capture`, `collect-stats`, `crossover`, `elo-calibrate`,
`elo-scale`, `semi-random`, `ar-report`, `noise-calibration`, `feature-stats`, `publish`,
`smoke`.

The old `bc`, `rl`, and `rl_fields` modes become `train --profile bc|rl|rl-fields`.
`bc-warmstart` and `elo-stats` disappear.

### Parsing and naming rules

- Command and option names use hyphens, never underscores.
- Each subcommand declares only applicable options; irrelevant options fail in argparse.
- Required subjects are actually required. There is no `"none"`, `"latest"`, or implicit 682.
- `--resume PATH_OR_RUN` always has a value and is mutually exclusive with `--pretrain-from`.
- Do not force different units behind a falsely universal flag. Reuse an option definition only
  when its meaning and unit are identical.
- Every budget option names or documents its unit: games, games per pair, decision steps,
  physics ticks, sample windows, or optimizer/environment steps.
- Semantic profile choices are not exposed through a generic `--set key=value` escape hatch.
- `--print-config` resolves, validates, fingerprints, and prints a training launch without
  constructing the trainer or allocating the environment.

### Modifier vocabulary

Modifier definitions may be shared, but each command opts into them explicitly:

| Concern | Options | Contract |
|---|---|---|
| execution | `--device`, `--seed`, `--allow-config-drift` | Stored in every durable artifact; drift is loud |
| subject | `--profile`, `--run`, `--team0`, `--team1`, `--agents` | Exact subjects only; command-specific required/exclusive rules |
| fleet | `--sizes` | Typed matchups such as `4v4`, `3v4`, or lists/ranges where the command supports them |
| training | `--resume`, `--pretrain-from`, `--compile`, `--no-wandb`, `--print-config` | Only on `train`; resume and pretraining are exclusive |
| launch sizing | `--vram`, `--num-envs`, `--microbatch-tokens` | Explicit values outrank cache/presets, then the result is validated |
| Elo | `--target-stderr`, `--max-batches`, command-specific game budget | Only on relevant Elo commands |
| capture | `--scenarios`, `--seeds`, `--fps`, `--max-steps`, `--hold-ms`, `--gif`, `--out` | `--max-steps` explicitly means physics ticks |
| artifact input | `--from-artifact` | Only on a mode with a real cheap reanalysis path, initially Elo calibration |
| publication | `--target`, `--check` | `publish` only |

There is no global `--smoke`: smoke owns its tiny fixed cases through `bnb smoke`. Compute modes
do not have `--no-plots`, because they do not publish. Durable artifacts receive managed paths;
`--out` remains only for scratch-producing commands such as capture.

---

## 3. Configuration and profile design

### Layers

The current configuration mixes project constants, profile intent, mechanical derivations,
and per-machine memory choices. Split those concerns:

```text
src/boost_and_broadside/
    config/
        schema.py          resolved configuration dataclasses
        defaults.py        SHIP, MODEL, REWARDS, ELO evaluation defaults
        resolve.py         mechanical derivation, override application, validation
        fingerprint.py     canonical serialization and stable hashes
    profiles/
        __init__.py        PROFILES registry
        rl.py              independent RL ProfileSpec
        rl_fields.py       independent field RL ProfileSpec
        bc.py              independent BC ProfileSpec
```

`runs/` is deleted. No profile imports another profile and there is no `BASE TrainConfig`,
`replace(RL, ...)`, class inheritance, or implicit delta chain.

### Intent versus resolved configuration

A `ProfileSpec` describes semantic intent. It may be composed from immutable project-level
components, but it is not a launch-ready `TrainConfig` and does not borrow another profile:

```python
BC_PROFILE = ProfileSpec(
    name="bc",
    objective=BehaviorCloningSpec(...),
    environment=EnvironmentSpec(num_ships=8, num_fields=0, ...),
    rollout=RolloutSpec(...),
    optimizer=OptimizerSpec(...),
    budget=2_000_000_000,
)
```

The resolver performs only named mechanical work:

- environment/entity tokens → aligned `num_envs`;
- action repeat → time-normalized fallback and per-component gamma/lambda;
- field intent → field environment, rewards, and field-map requirements;
- live reference probabilities → the fixed live Elo ladder;
- VRAM policy/cache → microbatch and rollout-shard settings;
- explicit CLI launch overrides → final launch values;
- validation → a complete immutable `ResolvedTrainConfig`.

The checkpoint and artifact store the complete resolved config, not merely the profile name.

### Override precedence

```text
independent profile intent
→ mechanical/project defaults
→ VRAM preset or matching cache entry
→ explicit CLI launch overrides
→ validation
→ resolved configuration
```

User overrides have the highest normal priority, but only for deliberately exposed launch
settings. Contradictory inputs fail instead of silently winning by order. Record both every
resolved value and its source (`profile`, `derived`, `vram-cache`, `vram-preset`, or `cli`).

Smoke is separate: its registry-owned fixed cases cannot be enlarged through launch overrides.

### Fingerprints

- `profile_fingerprint`: semantic experiment intent; excludes hardware choices.
- `resolved_config_fingerprint`: the complete launched configuration, including resolved VRAM
  and explicit overrides.

Use canonical serialization. Do not hash object `repr`s, unordered mappings, filesystem mtimes,
or other unstable representations.

### BC correction

BC is stale because it has not followed intentional project changes. Rebuild it independently
to match the current RL values wherever the BC objective does not require a difference:

- eight total ships / 4v4, matching action repeat and spawn configuration;
- the same logical batch, token sizing, minibatches, microbatching, and bounded quantiles;
- the same component gamma/lambda tables and other current optimizer/scaler values;
- an explicit BC schedule with policy gradient and league opposition disabled;
- its own training budget;
- keep `next_state_coef=1.0` as the current explicit BC choice unless a separate experiment
  justifies changing it.

The BC file repeats or composes named project defaults intentionally; it never imports RL.
Add an invariant test that compares resolved BC and RL configurations and allows only a named
set of objective-required differences. Future changes then fail review visibly instead of
propagating through inheritance or drifting silently.

---

## 4. Shared evaluation architecture

User-facing modes must never import another user-facing mode. Move shared machinery to a
non-mode package:

```text
src/boost_and_broadside/evaluation/
    match.py             MatchRunner and shared next-state harness
    tournament.py        Tournament, Player, progress, adaptive match allocation
    run_catalog.py       typed run discovery and checkpoint selection
    sizes.py             typed Matchup/FleetSize parsing
    artifacts.py         artifact creation, manifests, atomic persistence
```

Bradley–Terry fitting remains in its existing focused math module unless a later package move
improves dependency direction.

### Run and checkpoint selection

Do not collapse different selection policies into one `find_run_dir`:

- resolve an exact named run;
- select the latest resumable `step_*.pt` *within that exact run*;
- select the numerically final training step;
- select a named best policy;
- select all tournament-eligible ladder policies;
- accept an explicit checkpoint path where the command contract allows it.

Return typed values and raise typed exceptions; library code never calls `sys.exit`. CLI handlers
translate domain errors into concise user-facing failures. Selection uses recorded/numeric step
metadata, not filesystem modification time.

### Size parsing

One parser returns a typed `Matchup(team0, team1)` and preserves asymmetric inputs. A bare `4`
means `4v4`; `EnvConfig.num_ships` is then `team0 + team1`. Reject zero, negative, malformed, or
ambiguous values rather than skipping them.

### Field-capable environment construction

Provide one evaluation environment factory that constructs the required `FieldMapCache` whenever
`num_fields > 0`. `collect-stats` currently instantiates `TensorEnv` directly and cannot evaluate
field profiles; fixing this belongs in the shared factory rather than one mode.

Make `YemongPolicy.team_pma_k` required and pass it explicitly in roster/tests. Its empty default
can silently construct a different architecture and is not valid under arbitrary reward mixes.

---

## 5. Artifact and publication system

### Output taxonomy

| Class | Meaning | Location | Tracked? |
|---|---|---|---|
| run artifact | Durable evidence about an exact run/checkpoint | `checkpoints/<run>/artifacts/<type>/<id>/` | Ignored by default; selected landmark aggregates/manifests may be promoted |
| standalone/profile artifact | Durable evidence without one owning run | `artifacts/<type>/<id>/` | No by default |
| published view | Canonical reader-facing output selected by the publication manifest | `docs/` | Yes |
| scratch | Disposable clips, exploratory renders, temporary exports | `out/` | Never |
| local cache | Recomputable machine state such as `.vram.json` | documented cache path | Never; not an artifact |

The same mode can produce artifacts in different ownership locations depending on its subjects.
For example, `feature-stats` depends on both acting agents, the environment/profile, matchup,
budget, and seeds; it is not automatically a property of the profile alone.

### Artifact layout and identity

Example:

```text
checkpoints/resilient-resonance-682/artifacts/noise-calibration/
    20260809T142500Z-a81bc39e/
        artifact.json        recipe, execution provenance, schemas, hashes
        result.json          publishable aggregates and report inputs
        result.npz           bounded numeric arrays when JSON is inappropriate
        samples/             optional compressed raw samples; always gitignored
```

The ID combines a readable creation time and a short stable recipe hash. Never overwrite an
unrelated prior artifact merely because it has the same type.

`artifact.json` records at least:

- artifact type and schema version;
- original argv and a normalized canonical `uv run bnb ...` command;
- resolved arguments, defaults, configuration, and both config fingerprints;
- the source of every resolved launch value;
- exact agent descriptions and scripted-agent configuration;
- checkpoint paths, global steps, and SHA-256 hashes;
- source artifact IDs and hashes;
- matchups, budgets, seeds, field-map inputs, and RNG configuration;
- Git commit and clean/dirty status;
- `uv.lock` hash;
- Python, Torch, CUDA, GPU, compile mode, and relevant deterministic settings;
- producer/renderer version and creation time.

Capture only an allowlisted environment description; never dump environment variables or secrets.
Writes are atomic. Resumable compute updates a progress artifact atomically and verifies its recipe
before continuing.

### Raw samples (D17)

AR and noise analysis retain compressed raw samples when practical so later algorithms can
reanalyze them without replaying the environment. Raw samples are not required for the current
renderer: `result.json`/`result.npz` contains the sufficient aggregates used by publication.

Add explicit ignore rules after the landmark-run whitelist so `samples/` is never accidentally
tracked, even below a promoted checkpoint directory. The manifest records raw-sample hashes and
whether the local sample payload is present. A clean checkout can still reproduce every published
view from the tracked/promoted aggregate; reanalysis requiring raw samples is available only where
the ignored payload has been retained or separately archived.

### Compute, reanalysis, and rendering

- Normal mode execution performs simulation/measurement and writes a Tier A artifact.
- `--from-artifact` exists only where a genuine cheap reanalysis path exists. Initially this is
  Elo calibration refitting from stored win/tie matrices.
- Do not invent a universal `--refit` or `recompute` contract. Add a mode-specific path when stored
  samples make a real analysis useful.
- Compute modes never write canonical plots or Markdown.
- Capture writes scratch media under `out/` unless a publication recipe explicitly promotes it.

### Publication manifest

`docs/publications.toml` is the only hand-maintained mapping from canonical outputs to exact
artifact sources:

```toml
[publications.elo_curve]
artifact = "checkpoints/resilient-resonance-682/artifacts/elo-calibration/<id>"
renderer = "elo-curve-v2"
output = "docs/results/elo_curve.png"
```

`bnb publish`:

1. validates the manifest and every source hash/schema;
2. requires source artifacts produced from a clean commit;
3. performs no simulation and no network access;
4. renders only declared outputs;
5. removes or reports stale outputs formerly owned by the manifest;
6. writes a generated provenance index under `docs/results/` so prose can link to one stable
   source instead of copying run names/settings into many documents;
7. writes atomically.

`bnb publish --target NAME` renders one entry. `bnb publish --check` renders into a temporary
directory and fails on missing, stale, or changed outputs. Renderers strip nondeterministic image
metadata; checks may compare decoded pixels where byte identity is not stable.

W&B export/import is a separate ingestion operation that writes a Tier A artifact. Publication
only consumes the stored export and never authenticates to W&B.

Publication inventory includes all eight current top-level result figures, the policy architecture
figure, AR-report outputs, noise-calibration outputs, and curated replay GIFs. The manifest is the
inventory; do not rely on a hand-maintained figure count in code or prose.

---

## 6. Smoke and testing contracts

### Synthetic run fixture

Build a temporary run through the production checkpoint serialization path with:

- a randomly initialized current-schema `step_*.pt`;
- complete model, ship, environment, profile, and resolved-launch configuration;
- `roster.json` and `elo_history.jsonl` with the smallest valid contents;
- any minimum ladder/best policies required by a tournament smoke case;
- deterministic seeds.

Do not depend on the landmark run. Extract a pure checkpoint-payload builder if constructing a
full trainer is otherwise required merely to serialize a fixture.

### Subprocess runner

`bnb smoke` walks the registry and starts every case in a fresh subprocess with:

- its own temporary checkpoint/artifact/scratch roots;
- a fixed timeout;
- tiny registry-owned budgets and 4v4 or smaller mode-specific sizes;
- SDL dummy configuration where needed;
- plots/publication disabled by construction;
- captured stdout/stderr and a concise aggregate report.

The runner verifies that each process exits successfully, writes only within its temporary roots,
leaves the checkout unchanged, and cleans up. It must pass from a clean checkout with an empty
real `checkpoints/` directory.

### Test layers

Add tests as soon as the corresponding seam exists:

1. characterization/snapshot tests before moving current behavior;
2. pure unit tests for size parsing, run selection, config resolution, fingerprints, and schemas;
3. generated CLI contract tests for every registered command and irrelevant-option rejection;
4. subprocess smoke tests for every mode/profile case;
5. artifact atomicity, resume-recipe, provenance, and ignore-policy tests;
6. publication tests from fixed tiny Tier A fixtures, including an offline-network guard;
7. behavioral validation runs for live Elo and BC changes.

Every commit runs its relevant narrow tests and ruff. Milestones run the full pytest suite, the
complete smoke matrix, ruff, and applicable publication checks. A phase is a milestone, not a
one-commit constraint.

Work follows `docs/internal/mode-refactor-status.md`. The user launches primary section owners one
at a time. Within its bounded section, an owner may delegate exploration, implementation, testing,
or review to subagents and may use concurrent writers when file ownership is non-overlapping or
separate worktrees make integration safe. The primary owner remains responsible for reconciling all
work, running the section's checks, recording one coherent handoff, and stopping before the next
section. Fresh primary contexts reduce accumulated noise; the plan, ledger, commits, tests, and
resolved-config diffs carry continuity.

---

## 7. Implementation sequence

### Phase 0 — Contracts and characterization

- Land this plan's CLI/config/artifact contracts before structural implementation.
- Inventory current mode defaults, flags, hardcoded environments, outputs, and published sources.
- Snapshot the fully resolved current RL, RL-fields, and BC configs.
- Add characterization tests around run/checkpoint selection and mode dispatch seams.
- Add an executable artifact/publication inventory test so no existing published output is lost.

**Done when:** current behavior is recorded well enough to distinguish mechanical movement from an
intentional change.

### Phase 1 — Package and configuration foundation

- Move all `runs/` code into `src/boost_and_broadside/profiles/`; delete top-level `runs/`.
- Add config schemas, independent `ProfileSpec`s, resolver, validation, canonical serialization,
  and both fingerprints.
- Move project constants out of training-profile modules into `config/defaults.py`.
- Add one named token/env sizing derivation rather than duplicating integer arithmetic.
- Initially prove resolved RL/RL-fields equality to their characterization snapshots. Keep BC's
  intentional rewrite for Phase 7 so movement and behavior are separate.

### Phase 2 — Shared evaluation primitives

- Add typed size parsing and run/checkpoint selection.
- Extract match, next-state, environment-factory, and tournament engines under `evaluation/`.
- Move all user-facing modes onto those engines without changing their output semantics yet.
- Make `team_pma_k` required and fix all callers.

**Done when:** no user-facing mode imports any other user-facing mode.

### Phase 3 — `bnb` CLI and registry

- Add the project script and subcommand parser generated from `ModeSpec`s.
- Move per-command adapters out of the entry point; registry defaults are factories/resolvers, not
  duplicated static `EnvConfig`s.
- Add `--print-config` and strict subject/exclusivity rules.
- Delete root `main.py`, `--mode`, magic sentinels, implicit runs, aliases, and pathless resume.
- Update CLI documentation in the same breaking change.

### Phase 4 — Smoke system

- Build the synthetic run fixture and subprocess runner.
- Give every registered command/profile a deterministic smoke case.
- Enforce temporary roots and checkout-clean assertions.
- Run the full smoke matrix in CI/local milestone checks.

No further consolidation or behavioral change lands until this phase is green.

### Phase 5 — Mode consolidation

- Replace `bc`, `rl`, and `rl_fields` dispatch with `train --profile ...`.
- Delete `bc-warmstart`; document the explicit two-command pretrain handoff.
- Add arbitrary-agent support to `elo-calibrate`, including no-run artifact ownership, then delete
  `elo-stats`.
- Move the AR report's orchestration into its mode and make its canonical default 4v4.
- Fix field-capable `collect-stats` through the shared environment factory.
- Delete dead arguments and no-op flags, including `feature-stats.output_dir` and `--fast-cache`.
- Replace dead checkpoint defaults with required explicit subjects.

### Phase 6 — Artifact store and publication infrastructure

- Implement artifact schemas, IDs, manifests, atomic persistence, resume verification, and optional
  ignored raw samples.
- Move crossover, Elo scale, semi-random, AR, noise, and feature statistics to managed artifacts.
- Add Tier A raw/aggregate outputs for AR and noise analysis.
- Add `docs/publications.toml`, `bnb publish`, generated provenance, and `publish --check`.
- Convert existing render scripts into renderer functions or delete them after parity is proven.
- Keep W&B ingestion separate and make its output conform to the artifact contract.
- Add `artifacts/`, `out/`, `.vram.json`, and all nested `samples/` paths to `.gitignore` with tests
  covering the landmark whitelist interaction.

Use synthetic/current-schema fixtures for this phase. Landmark backfill and the final no-diff
publication gate wait until the 682 migration in Phase 10.

**Done when:** fixture-selected canonical outputs regenerate offline, artifact and manifest
contracts are covered, and no compute mode can write under `docs/`.

### Phase 7 — BC correction

- Independently encode the intended current BC profile described in §3.
- Add the named BC-vs-RL allowed-difference invariant test.
- Apply stale-value corrections as an explicit behavioral change after the old/new resolved diff is
  reviewed field by field.
- Smoke the final BC profile and run a bounded validation before treating it as the pretraining path.

### Phase 8 — Live Elo ladder

- Derive live reference rungs from configured probabilities using `1000·p`.
- Remove fitted ladder/random fields from the resolved training schema.
- Rename logs/configuration/documentation so `live_elo` and `calibrated_elo` cannot be confused.
- Keep `semi-random` as a validation tool for the accepted approximation.
- Run a bounded validation and inspect roster/milestone differences before adoption.

### Phase 9 — VRAM resolution

VRAM categories describe guarantees honestly:

| Tier | Knobs | Guarantee |
|---|---|---|
| 1 — same mathematical objective | gradient checkpointing, microbatch tokens | Equivalent objective/update within floating-point tolerance; not bit-identical |
| 2 — same nominal logical batch | rollout tokens/shard count at fixed total tokens | Same nominal tokens and optimizer-step count; different env-stream count, temporal correlation, and minibatch composition |
| 3 — experiment change | total token budget, minibatches, fleet size | Changes the optimization or task |

- `--vram auto|probe|reprobe|off|8|16|24|32`.
- Probe in fresh subprocesses; never retry OOM configurations in a fragmented parent allocator.
- Cache keys/fingerprints include GPU identity/UUID and total memory, MIG where relevant,
  profile/model fingerprints, dtype, compile mode, PyTorch/CUDA versions, and probe version.
- Only measured rows are called measured; extrapolations are labelled provisional.
- Explicit CLI launch knobs outrank the cache/preset and the complete resolution/source map is
  printed by `--print-config` and stored in checkpoints/artifacts.
- Cache writes are atomic.

### Phase 10 — One-time 682 migration and landmark backfill

This is an explicit, one-off repository migration, not runtime compatibility infrastructure.

- Freeze and record the target checkpoint, config, observation, artifact, and provenance schemas.
- Preserve an inventory and SHA-256 hashes of every original 682 policy file.
- Migrate every final, best, recent-average, and ladder checkpoint required to recompute the
  landmark results.
- Apply the measured tensor/key/config transformation, add known historical `ship_config` and
  schema provenance, and represent genuinely unknown historical launch values as unknown rather
  than inventing them.
- Record per-file original hash, migrated hash, transformation version, tensor mapping, and
  validation result in a tracked migration report within the landmark run.
- Validate exact tensor mappings and zero-field functional equivalence on fixed observations:
  policy logits/distributions, values, recurrent state, and next-state outputs. Use bitwise checks
  where the execution path permits them and documented tolerances where kernels do not.
- Run seeded zero-field scenario comparisons as an end-to-end confirmation.
- Backfill the landmark Tier A artifacts and raw local samples needed by AR/noise reanalysis.
- Select the exact landmark artifacts in `docs/publications.toml` and run the full offline
  `bnb publish --check` gate.

**Done when:** every required 682 policy loads through the ordinary current loader with no migration
path, the equivalence report passes, and every canonical published output regenerates from the
manifest with no unexplained diff.

### Phase 11 — Documentation and cleanup

- Sweep README, getting started, evaluation, training, replays, architecture, engineering notes,
  and internal training plans.
- Replace repeated run-specific provenance with stable links to the generated publication index.
- Confirm no old command names, `--mode`, `runs/` imports, 8v8-default claims, sentinel values,
  implicit 682, or compute-to-doc paths remain.
- Keep the plan and live ledger through final branch review. Remove them only during human-approved
  merge preparation, after their durable contracts have landed in code/tests/documentation.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Breaking CLI strands stale docs/scripts | Intentional clean break; repository-wide search and CLI contract tests in the same change |
| Config resolver accidentally changes RL/RL-fields | Snapshot equality before behavioral work; canonical resolved-config diffs |
| BC correction hides an unintended change | Independent profile plus named allowed-difference test and separate validation phase |
| Approximate live Elo changes league membership | Separate terminology, bounded validation, roster and milestone diff |
| Run selection loads the wrong policy | Exact subjects, typed selection policies, numeric/recorded steps, no implicit latest |
| Synthetic fixture drifts from checkpoint schema | Production payload builder plus schema tests |
| Artifact is overwritten or resumed with different inputs | Recipe-derived ID, atomic writes, recipe verification before resume |
| Raw samples consume large disk or enter Git/LFS | Compressed/bounded chunks, explicit ignored `samples/`, ignore tests after landmark whitelist |
| Publication silently uses the wrong run | One reviewed manifest with exact artifact/source hashes |
| Publication needs network or undocumented local state | Offline guard, clean-source requirement, stored W&B ingestion artifact, `publish --check` |
| Migrating only part of 682 makes landmark recomputation incomplete | Inventory and migrate every final/best/recent/ladder policy before validation |
| The one-time migration fabricates unavailable history | Record known historical values and explicit unknowns; never substitute current launch settings silently |
| Refactoring before migration leaves landmark gates temporarily blocked | Use synthetic fixtures through Phase 9; reserve landmark backfill and final publication acceptance for Phase 10 |
| PNG byte output varies despite equivalent rendering | Strip metadata; compare decoded pixels where needed; record renderer/dependency versions |
| Provenance leaks secrets | Allowlist execution metadata; never snapshot environment variables or credentials |
| VRAM cache survives an incompatible software/model change | Comprehensive fingerprint and explicit `reprobe` |
| Explicit overrides create an invalid launch | Apply last, validate the complete configuration, and fail loudly on contradictions |

---

## 9. Out of scope

- `label_scale` may be mis-calibrated. `feature-stats` suggests substantial changes, but calibration
  is a separate measured investigation.
- Cleanup of existing smoke droppings is handled separately; Phase 4 prevents recurrence.
