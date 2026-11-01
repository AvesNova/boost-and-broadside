# Cleanup pass spec

Status: spec, 2026-11-01. Sources: the consistency audit
(`../audits/2026-11-01-consistency.md`, at `49e5e2a9`) and the mutation sweep of the
merged sections (177 mutants, 43 survivors; the real gaps are restated in section 3 as
behaviour, so no tool is needed to act on them). Code style: `docs/contributing/code-style.md` (drafted with this
spec, for the author's review). **(proposal)** marks a choice the spec makes, implemented
as written and listed for the author.

## 0. Scope and order

- **When:** after sections 6 (learner) and 1b (ship card) merge. Re-check every audit item
  against the merged tree first: some were sent to section 6, and both sections changed
  files the audit cites. An item already fixed is reported as such, not redone.
- **The rule of the pass:** the game, the models and the learner compute the same numbers
  afterwards. Every reference fixture passes unchanged, the goldens are never touched,
  and the bench cells stay within noise (section 6).
- **Running alongside:** sections 4 and 7 (scripted agent and BC) and 10 (the play
  shell) may run at the same time. So:
  - leave `bnb/controllers/scripted.py` (section 4 replaces it) and `bnb/play/`
    alone;
  - first, format the whole tree with `ruff format` (section 1.0), in one commit;
  - then do the renames of section 1.1, in one commit, and push both at once, so the
    other branches can merge them early;
  - merge `origin/v2` into the cleanup branch at least once a day.
- **Out of scope:**
  - the native goldens that replace the race adapter (`tests/reference/race_arm.py`);
    they need fixtures generated from v1 and come separately;
  - `REFERENCE_ARM` stays one variable for every suite: the fixture validation sets it
    per run;
  - any behaviour change.

## 1. Names

### 1.0 The formatter (author, 2026-11-01)

- Run `uv run --no-sync ruff format` over `src`, `bench` and `tests` (except
  `tests/reference`, which ruff excludes), in a commit of its own that changes nothing
  else.
- Add that commit's hash to `.git-blame-ignore-revs` at the repository root, so
  `git blame` skips it.
- A test runs `ruff format --check` and fails on unformatted code.
- When this spec was written, it reformatted 66 files (about 4,400 changed lines). An
  open branch merging it should format its own files with `ruff format` first; most
  conflicts then resolve to the formatted side.

### 1.1 Contract renames (first commit)

- **`logp` → `log_probability`** in `Outputs` and `Trajectory`. Bump the schema version
  of each contract that changes.
- **Unit suffixes in full words:** `Pool.episode_s` → `episode_seconds`,
  `Pool.respawn_s` → `respawn_seconds`, and any other `_s`, `_m` or `_ms` field. A
  helper that converts is named for what it returns (`ticks_of` stays).
- **`done` and `reset`:** `reset` means the environment reset *before* this observation,
  and `done` means it ends *after* this step, everywhere (`Controller.step`, `Trajectory`,
  `Events`, Yemong). Section 6 was asked to do this; finish whatever is left.

### 1.2 Shape letters

- One meaning per letter, project-wide. Keep the conventions' letters (`B`, `T` teams,
  `S`, `N`, `M` fields, `K` bullet slots) and add **(proposal)**:
  - `Z` zone slots;
  - `L` reward levels;
  - `C` spawn candidates.
- In `learn/rewards.py`, the targets axis is ships, so it is `N`.
- Add the new letters to the conventions table in `contracts.md`.
- Letters appear only in shape comments and docstrings. In code, the dimension is a
  named variable (`levels`, `candidates`).

### 1.3 Abbreviations and ambiguous names

- **Full words** (scaffold rule 10) in every file the audit lists:
  - `flight.py`: `gx`, `gy`;
  - `rollout.py`: `st`, `ev`, `buf`, `ctrl`, `t_idx`, `dev`, `e`, `self.T/S/N/B`;
  - `train.py`, `lineup.py`, `realtime.py`, `cli.py` and `bench/`.

  Then sweep the rest of `src/` and `bench/` for the same.
- **`environment`, not `env`,** in names. `num_envs` becomes `environments`, or a name
  that says what is counted. The CLI's option names follow.
- **"tick":**
  - the duration is `tick_length` everywhere (`Yemong.tick`, `ShipStateCode.tick`);
  - the integer counter is `episode_tick`;
  - the float at `observe.py:84` is named for what it is (the elapsed fraction of the
    episode);
  - `TickUniforms` keeps its name (the random draws of one tick).
- **"direction":** drop `Capture.direction` (section 2). The observed zone column stays
  `direction`, documented as the meter's movement.
- **`_inside`:** `_in_zones` (`phase.py`) and `_clip_to_disc` (`spawns.py`), or names
  that say the same.
- **`compile` as a parameter:** `compiled: bool`, so it no longer shadows the builtin.
- **Stale `team_rel`** in `mlp.py` and its other uses: `team_relative`.

## 2. Dead code, fixture-shaped code, and the opaque state

- **Delete:**
  - `state_code.MEAN`;
  - `Hits.taken` and its test;
  - the E501 exemption for `perception/sight.py`. Fix the comment on the `physics/*`
    exemption.
- **Fixture-shaped code:** make the code serve the game, and adapt the test adapter,
  never the goldens:
  - `frontline.capture_step`'s `alive` parameter and its `Capture.direction` output go.
    `tests/reference/rewards/rewards_arm.py` computes what it needs itself;
  - `Encoder(zone_inputs=None)` goes, and the Yemong reference adapter passes absent
    zones (`present` = 0);
  - `inputs.py` stops handling `obs.zones is None`. Zones are required (Z = 5 since
    section 3c).
- **No production caller:** `ReturnScale`, `outcome_targets`, `HexGrid`,
  `poisson_terms`, `trajectory_code`, `outcome_code`, `grouped_cross_entropy`,
  `velocity_sharp`.
  - After section 6 merges, keep each one only if production code calls it or a pending
    spec names it (`scripted-bc-spec.md`, `league-spec.md`, `diagnostics-spec.md`).
  - Delete the rest with their tests.
  - List the verdict for each in the notes.
- **The game state is opaque to the runtime** (audit fix 3). Today `runtime/rollout.py`
  reads and writes `state.match.ticks_since_seen` and imports `game.layout.environment`.
  - `game.api` gains what the runtime needs:
    - the part of the state that `perceive` changes, as a pytree the runtime copies
      without naming its fields **(proposal)**;
    - `episode_ticks(rules, batch, device)`, or whatever the runtime asks
      `environment` for.
  - Afterwards nothing under `runtime/` names a field of `State`. A test walks
    `runtime/`'s source for attribute access on `match`, `ships`, `bullets` and `mode`
    and fails on any.
  - The stated rule in `state.py` and `api.py` then holds as written.
  - Measure the rollout before and after (section 6). A whole-state copy is acceptable
    only if it costs nothing measurable.
- **Process-global settings and compiles in library code** (audit fix 4). Verify against
  the merged tree; section 6 was told. Nothing outside the entry points (`cli.py`,
  `bench/`, `__main__`) may:
  - call `torch.manual_seed`;
  - set TF32, thread counts or dynamo limits;
  - call `torch.compile`.

  The runtime may compile stages: scaffold rule 2 forbids compiling in *section* code,
  not in the runtime. A test enforces both, by walking the source.

## 3. Tests the mutation sweep asks for

Each item is a mutation that left the whole suite passing. Write a test that fails under
it, and confirm that it does: apply the change by hand in your worktree, watch the test
fail, then revert. Name the test for the behaviour, not the mutation.

**Game, physics:**
1. A dead ship with shoot pressed, its gun cooled down and energy to spare, fires
   nothing: no bullet, no energy spent, `fired` false (`bullets.py`, `fire`).
2. A ship whose energy exactly equals the shot's cost fires (the `>=` boundary).

**Game, perception:**
3. With fog off, `ticks_since_seen` stays 0 for every ship, a dead one included, over
   several ticks.
4. The observation's reset flag (the global column) is 1 on the first observation of an
   episode and 0 on the next.
5. `Privileged.ships`, in a fogged environment with a hidden living enemy and a dead
   one, holds the rows the contract states (`visible` 1, `ticks_since_seen` 0, true
   positions). Fix the docstring first (section 4), then pin it.
6. The observed respawn wait is in seconds: a wait of `w` ticks shows `w · tick_length`.

**Game, modes:**

7. Team 1's view shows the Frontline front negated, in a scenario where the front has
   actually moved. Make the seat-symmetry test reach a moved front too: today the front
   never leaves 0 within its horizon, so the test cannot see this.
8. A dead ship inside a zone does not count toward its capture.
9. On the tick a capture completes, every zone's observed movement is 0.
10. Spawn-zone protection uses each ship's own team, at `S ≥ 2`:
    - every ship of either team inside its own team's spawn zone is protected;
    - no ship is protected inside the enemy's spawn zone.

    Today the tests check ship 0 only, so `repeat` and `repeat_interleave` cannot be told
    apart.
11. On a respawn that is not a reset, enemies of every group count toward the spawn
    choice. On a reset, only same-group rivals count.
12. A team whose ships are all dead respawns at the uniform fallback points, not anchored
    on stale positions.

    If you judge 11 or 12 to be an equivalent mutation, write the argument in the notes
    instead of a test.

**Rewards** (hand-computed cases in `tests/sections/`; the reference fixtures do not
reach these paths):

13. The objective-progress level carries `objective_progress`, not `objective`.
14. Boundary damage followed by a shield restore:
    - the recharge deficit includes the boundary row;
    - the boundary ledger falls by its share;
    - the ship rows are attributed only their share.
15. A ship hit and then restored, both counted in the same tick: the ledger total used
    for kills and for recharge includes this tick's combat damage.

**Runtime** (re-check against section 6's rollout; drop an item whose code is gone):

16. The stored reward is the learner's seat team: run with seats drawn at random, and
    check every stored reward against the game's reward for that seat.
17. Seats are redrawn at episode ends: over many episodes, both seats occur, and no seat
    changes mid-episode.
18. `ticks_since_seen` carries across rollout steps: a ship hidden for several ticks
    shows a rising count in the stored observations.
19. Each stored `done` belongs to its own environment: environments with different
    episode lengths in one batch.

**Contracts:**

20. `environments_per_pool` rejects a share that gives a fractional count (for example
    0.25 of 10).
21. `environment_multiple` accounts for the fog share (share ½ with fog share ½ needs a
    multiple of 4).
22. `ticks_of` rejects a time just off a whole tick (for example 1.01 s at 1/30 s).

**Yemong** (section 6 changed `inputs.py` and `model.py`; re-check each against the
merged code):

23. A ship that spawned this tick enters with the null pending action, not the stale
    enemy-action forecast.
24. The local-presence input counts only valid belief rows: a ship whose belief is invalid
    adds no presence.
25. The time-since-seen input is in seconds: an age of `a` ticks shows `a · tick_length`.
26. The predicted enemy position is taken at the image nearest the belief, modulo the
    position period. Test it with a belief and a prediction on opposite sides of a period
    boundary.
27. `unroll` from a nonzero carry whose first step is a reset equals `unroll` from a zero
    carry. Today's unroll-equals-steps tests never start a segment with a reset.
28. A reset in the middle of a segment: `unroll` equals the step loop around that reset.
    If `starts` and `_later(starts)` cannot be told apart by any input, write the
    argument instead.

The sweep ran 177 of its 179 mutants (two Yemong mutants, `Y08` and `Y15`, were not run).
Its log and report are not available to you; this section is the complete list.

## 4. Docs, docstrings and contracts.md

- **Retire `STYLE_GUIDE.md`:**
  - delete it;
  - `docs/contributing/code-style.md` (the author-reviewed draft) replaces it;
  - `docs/contributing/writing-docs.md` already points its "Code rules" at it;
  - fix every reference to the old guide.
- **Docstrings against the code** (audit section 3):
  - `cli.py:1` (`bnb train`, and the claim that the CLI is the one place that seeds);
  - `Privileged.ships`;
  - `obs.py:4` (only ship rows carry `team_relative`);
  - `config.py:5` and `test_contracts.py:19` (ship indices are int32 −1, not uint8 255);
  - `model.py:264` (the memory shape is `(..., M + Z, Dm)`);
  - `api.py:13` (the functions that take no ship card).
- **Defaults and counts in prose:**
  - Docstrings that restate a config default (`combat.py:90`, `spawns.py:89,159,181`,
    `spawning.py:84`, `layout.py:25`, `scripted.py:24` once section 4 has merged)
    instead name the field.
  - Counts written into prose (`action.py:34`, `observe.py:39,67`, `rewards.py:171`,
    `heads.py:13`, `mlp.py:50`, `inputs.py:60,67,99`, `model.py:112,130,359`,
    `belief.py:18-20`, `policy.py:1,10`) are removed, or derived from the constant they
    count.
- **Missing docstrings:** every public module, class and function has one
  (`uv run --no-sync python -m tests.docs.docstrings` reported 67 before sections 6 and
  1b). Then make it a check in the suite, so the count stays at zero.
- **`contracts.md`:**
  - "Current schema":
    - drop "required argument of every game function";
    - complete the Game API list (`ship_types`, `start_clocks`, `sweep_like`, and the
      functions added by sections 6 and 1b);
    - remove the contradicted "mode features come with the modes section";
    - `values (R, S, K)` gets its own letter;
    - drop `GameConfig.num_teams`, which duplicates `rules.TEAMS`. Delete the field in
      the code too, or say why it stays.
  - "Skeleton implementation notes": update or delete each stale note (`last_seen`, the
    seat-layout xfail, `presets.json`, the ledger, the uint8 indices). The ship-index
    convention row in "Conventions" says int32 −1.
- **Contract dtypes (rule 8):** the contract test checks the dtype and shape of every
  field of the `Events`, `Obs` and `Privileged` the game produces, at two batch sizes and
  in every mode.

## 5. Layering and duplication

- **`test_imports.py`** resolves relative imports (`from ..x import y`) to their absolute
  module before checking the dependency rule. Fix whatever it then finds.
- **`PRESETS`** moves from `game/modes/rules.py` to `runtime/`. The game keeps only rule
  defaults.
- **`test_game_section.py`** stops importing from `race_arm`. Copy the helper it needs
  into the test, or replace it with a native equivalent.
- **One definition each** (audit section 6):
  1. a ship's team (`arange(N) // S`): one function in `game`, used everywhere;
  2. the enemy flip;
  3. flight-path velocity;
  4. disc sampling;
  5. the magic numbers:
     - 0.85, the spawn and map disc fraction;
     - the zone count 5: one `ZONE_SLOTS` in `contracts`;
     - 2600;
     - the 40 m band width.

     Each becomes one named value where it belongs. If two uses share a number by
     coincidence, not by intent, give them separate names, and say which in the notes;
  6. the mode-name strings;
  7. two-hot encoding;
  8. the action sizes `[3, 7, 2]` and the joint-action formula: tests and `bench/` use
     the contract's constants and `to_joint`;
  9. the batch multiple 8 in `bench/sizes.py`;
  10. `INFINITY`.

## 6. Test hygiene

- **Markers:**
  - every test that takes more than 10 s on a quiet machine carries `slow` (measure with
    `--durations=0`, list the times in the notes);
  - every test that can touch CUDA carries `gpu`, including `test_yemong_model.py`'s
    "CUDA when visible" path. `-m "not gpu"` must never initialise CUDA. Check this with
    `CUDA_VISIBLE_DEVICES=` and a guard;
  - the `not slow and not gpu` tier should take under 3 minutes on a quiet machine
    **(proposal)**. Report its time.
- **No shared random state:**
  - remove the global reseeding (`yemong_cases.py:27`, `test_learning.py:18`);
  - every test that draws numbers uses a local `torch.Generator`
    (`test_perception.py:66,93` and any others);
  - a fixture in `tests/conftest.py`, used by every test, fails a test that changes
    torch's global CPU RNG state, so a new unseeded draw or a global reseed is caught
    **(proposal)**.
- **`test_game_section.py:318`** reads dynamo's process-wide counters after a reset. It
  runs in its own subprocess, or measures recompiles some other way that does not depend
  on what ran before it.
- **Gaps:**
  - `test_no_controller_takes_privileged` covers `YemongController` too;
  - `test_yemong_model.py:33` uses `GLOBAL["fogged"]`;
  - the redundant assert at `test_perception.py:74` goes;
  - the learner equivalence suite no longer skips (section 6 implements its arm).
    Confirm it runs.

## 7. Acceptance

1. **The full suite passes**, with nothing skipped except what needs hardware you lack.
   Ruff is clean.
2. **Equivalence:**
   - every file under `tests/reference/` except the two adapters named in section 2 is
     unchanged;
   - the goldens are byte-identical;
   - the reference suites pass.
3. **The sweep's tests** (section 3) exist, and each fails under its mutation. The notes
   list each test against its item, or the equivalence argument.
4. **The source-walking tests:**
   - no `State` field access in `runtime/`;
   - no global settings or compiles outside the entry points and the runtime;
   - relative imports resolved in the layering test;
   - the docstring check at zero.
5. **Speed:** the bench cells (`bench/`) before and after, on the same machine, with the
   benchmark lock held:
   - every cell within noise (2%);
   - the rollout change of section 2 measured on its own.

   Numbers taken while other agents load the CPU are provisional; say so.
6. **Docs:** `contracts.md` and the user docs match the code. The docs checks pass with
   no `PossiblyStalePage` warning left unresolved.
7. **Size:** lines added and removed, by package.
