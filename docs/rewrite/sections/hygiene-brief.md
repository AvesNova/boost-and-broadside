# Brief: model settings in TrainConfig, and test hygiene

Follow-ups from the cleanup pass, the league and the play shell. Read
`docs/rewrite/contracts.md` (conventions, then the newest entries, especially "Resume:
integrity, not bits"), `docs/contributing/code-style.md` and
`docs/contributing/writing-docs.md` first.

## What to do, one commit each

1. **Model settings and the ship card in `TrainConfig`** (cleanup spec 8.3; the cleanup
   notes' "Left for a fresh agent" section says what blocked it):
   - `TrainConfig` holds the model's settings (`YemongConfig`, `RelationalConfig`) and the
     ship card. They replace `relational_hidden`.
   - `Rollout` and `Trainer` take the ship card instead of calling `api.ship_types(device)`
     themselves.
   - The CLI changes a setting with `--set yemong.width=256` (`TrainConfig.model` stays the
     model's name).
   - `run.json` and the checkpoint record the settings. A resume refuses a changed model
     setting; only `OVERRIDABLE` fields may change.
   - Tests: a width change reaches the built model; it round-trips through a checkpoint; a
     resume with a different width is refused with its reason.
2. **GPU resume compares within rounding** (author, 2026-11-03). The GPU cases of
   `tests/runs/test_resume.py`, and any other GPU test that asserts bit equality between
   processes, compare within a float tolerance:
   - CPU cases stay bitwise;
   - choose the tolerance so a state left out of the checkpoint still fails. Check this
     by hand: skip restoring one generator, then the optimizer state, watch both fail,
     and revert;
   - the integer and categorical fields (counters, actions, results) stay exact.
3. **The CUDA guard's false errors.** On a CUDA machine, some non-`gpu` tests initialise
   CUDA when run alone. Known cases: `test_bc.py::test_reinforcement_learning_starts_from_the_clone`,
   `test_mixed.py::test_a_compiled_mixed_tick_equals_each_game_compiled_alone`, and
   `test_realtime_steps_one_match[True]`. Find why (an optimizer step, or the compiler
   probing devices), and fix it so a CPU test touches no CUDA. Mark a test `gpu` only if it
   really needs the GPU. Run each listed test alone to confirm, plus every non-`gpu` test
   with `CUDA_VISIBLE_DEVICES=""`.
4. **The game state's integer dtypes.** `mode.score` is int32 after a reset and int64 after
   a tick. Counters are int32 by contract: find every field whose dtype changes across a
   tick, fix it, and add a test that a tick keeps every state field's dtype.
5. **`bnb.runs.records.retained`'s docstring** says league snapshots follow its rule; they
   don't (`bnb.league.pool` has its own). Fix the docstring.
6. **The quick tier is 4 minutes again** (`-m "not slow"`; the target is about 2.5).
   Measure with `--durations=40`, mark tests over 10 s `slow`, and keep a fast variant
   where the slow one guards something the quick tier needs.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-hygiene`, branch `fix/hygiene` (from v2). Work
  only here. Scratch in `/tmp/bnb-scratch-hygiene`.
- **Do not read** any other directory under `/home/vizia/avesnova`.
- **Python:** `uv sync`, then always `uv run --no-sync ...`.
- **GPU:** every GPU run holds the benchmark lock (the wrapper
  `/tmp/bnb-scratch-fast-yemong/locked.py <command...>`). The play fix round shares it.
- **Boundaries:**
  - stay out of `src/bnb/play/` (the play fix round is there);
  - the goldens and `tests/reference/` are fixed;
  - nothing the game, models or learner compute may change, beyond item 4's dtypes.

## Done

- **The full suite** passes under the lock, once at the end. Report the quick tier's time,
  on the GPU and with `CUDA_VISIBLE_DEVICES=""`.
- **Commits** to `fix/hygiene`, pushed (`git push -u origin fix/hygiene`). Author Julian
  Kizanis <julianzia@gmail.com>; end every message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Notes:** `docs/rewrite/sections/hygiene-notes.md`, with each item and a Progress list.
- **Final message:** half a page.
