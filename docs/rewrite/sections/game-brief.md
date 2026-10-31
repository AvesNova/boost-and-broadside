# Brief: section 1, game

You are implementing the **game** section of Boost and Broadside v2: the `bnb.game` package
(physics, perception, spawns, maps). The goals are **speed and maintainability together**:
fast batched GPU training, the same code at batch size 1 on the CPU for live play, and code
the next person can read and change.

## Where you work

- **Your worktree:** `/home/vizia/avesnova/bnb-game`, branch `section/game`. Work only
  here.
- **Scratch:** use `/tmp/bnb-scratch-game` for temporary files.
- **Do not read or modify** any other directory under `/home/vizia/avesnova`. In
  particular, stay out of the other worktrees, the v1 code, and the `physics-race*`
  repositories.
- **Python:** set up with `uv sync` in your worktree, then always run through
  `uv run --no-sync ...` (never plain `python`).

## Read first

1. `docs/rewrite/sections/game.md`: the decisions behind this section, with the
   author's reasons.
2. `docs/rewrite/sections/game-spec.md`: **the spec you implement**. Parts marked (v2)
   are new.
3. `docs/rewrite/contracts.md`: the types between sections. `bnb.game` must keep
   producing these.
4. `docs/rewrite/scaffold.md`, section 3 (design rules), and `README.md`.
5. `src/bnb/`, especially `game/`, and `tests/`.

## What to do

1. **Implement the v2 spec** in `bnb.game`:
   - friendly fire, never hitting the shooter;
   - bullet drag on proper speed;
   - the shuffled, rotated spawn formation, driven by the per-tick uniform layout of
     spec section 2;
   - point-symmetric maps;
   - the v2 combat constants;
   - the shield rule.

   `game/physics` is the race's PyTorch implementation of the previous spec. Change it
   as you see fit.
2. **Fix the race review's findings:**
   - **State as typed dataclasses:** `Ships`, `Bullets`, `Fields`, `Match`, or your own
     split if better. Register them as pytrees, e.g. with `bnb.contracts.tree.contract`.
     State stays opaque outside `bnb.game`.
   - **Remove the `Preset.flight` trap:** a "shared" flight property that silently
     returns the fighter's values.
   - **No process-global settings in library code:** no TF32 switches, thread counts,
     recompile limits, global seeding, or guard skipping.
3. **Constants:**
   - Ship-card parameters are per-type tensors: one compiled graph serves any values, and
     a sweep can vary them.
   - Fixed numerics (tick length, soft-min exponent, band widths) stay Python numbers.
   - Eager mode must keep working everywhere (used for compile-free tuning).
4. **Perception:**
   - Port `game/perception` (sight and observe) to the new state.
   - The sight rules and the observation contract are fixed.
   - Making the sight test faster at general sizes is in scope.
   - There are no bullets in the observation, and no bullet sight test.
5. **Tests:**
   - **Everything must pass:** `uv run --no-sync pytest`.
   - **Acceptance tests are fixed:** you may edit only the adapter
     `tests/reference/race_arm.py` (and `race_chaser.py`) to fit your API. Never change
     the other files under `tests/reference/` (test files, helpers, conftest, goldens).
   - **The spawn-leak xfail:** remove the strict xfail marker on
     `tests/sections/test_perception.py::test_spawn_layout_does_not_reveal_the_seat` once
     it passes.
   - **Add your own tests** under `tests/sections/`:
     - sight lines against a brute-force closest-point check;
     - eager vs `torch.compile` agreement for the tick and perception (CPU is fine);
     - CUDA-graph replays drawing fresh randomness;
     - anything else you judge important.
6. **Speed:**
   - Measure with `uv run --no-sync python -m bench.run`: about five minutes, sealed
     random sizes, appends to `bench/results.jsonl`. The baseline is the entry for commit
     `4cc7c13c`.
   - Optimize for **general** speed, not one size. No algorithm branches gated on preset
     or size unless the same benchmark run measures the code without them.
   - Any ad-hoc GPU timing must hold the benchmark lock (`bench.lock.bench_lock()` in
     your script). Never wrap a lock-taking command in plain `flock`. Keep benchmarking
     proportionate: a few runs, not a sweep.
7. **Commit and push often:**
   - commit to `section/game`, then `git push`;
   - author Julian Kizanis <julianzia@gmail.com>;
   - end every commit message with
     `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Boundaries

- **You may change:**
  - `src/bnb/game/**`;
  - `tests/sections/**`;
  - the reference adapter (`tests/reference/race_arm.py`, `race_chaser.py`);
  - `src/bnb/runtime/*`, but **only** the mechanical changes the new game API forces.
    List each in your report.
- **You may not change:**
  - `src/bnb/contracts/**`;
  - the acceptance tests and goldens;
  - the docs, apart from your notes file below;
  - `bench/`.
- **If a contract or the spec looks wrong:** do not work around it silently. Take the
  most literal reading, keep going, and write it up as a request in your notes.
- **Precision:** float32 throughout. The tolerances already allow for float32 rounding,
  so do not add float64 hacks without a measured reason.
- **Smoothness and repeatability are traded by judgement:** smooth and differentiable
  formulations where they cost (almost) nothing; a 0.01% cost for repeatability is worth
  it, a large one is not.

## Hand back

Write `docs/rewrite/sections/game-notes.md` with:
- **Design:** the layout of `bnb.game`, the state types, the compile and stage
  assumptions.
- **Decisions and deviations:** each with its reason.
- **Benchmark:** before and after on the same table as the baseline.
- **Runtime:** the changes you had to make there.
- **Requests and open questions:** contract or spec requests, and anything unresolved.
- **Size:** line counts per module.

Your final message should summarize the same, in a page or less.
