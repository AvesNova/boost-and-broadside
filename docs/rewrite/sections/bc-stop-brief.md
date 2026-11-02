# Brief: BC's score stop as a confidence bound

You are fixing **when behaviour cloning stops** in Boost and Broadside v2. The decision is
made (author, 2026-11-02; the last bullet of `docs/rewrite/sections/bc.md`); you implement
it, test it, and show it works on the toy run.

## The problem

`bnb.learn.bc.StopRule` stops BC the first time the live score against the teacher, over
the last 1000 games, reaches `StopConfig.score_target` (0.45). It is checked every update,
so it fires on the first lucky window: the toy clone's fresh 512-game scores after
stopping were 0.478, 0.447 and 0.405. Also, at the default label floor (0.01) the clone
copies a floored distribution but is scored against a teacher that plays without the
floor; that run never reached 0.45 in 100M steps.

## The decision

1. **The score stop is a lower confidence bound:** it fires when the window's score minus
   two standard errors reaches the target. The standard error is the window's own (per-game
   scores 1, 0.5, 0: keep the sum of squares beside the sum, trimmed the same way), not a
   worst case. Keep the per-update window and its exact trimming.
2. **The live score is against the teacher as the labels describe it:** the scripted
   opponent in BC's scoring pool plays by sampling the labels' distribution (the
   `TeacherConfig`: its floor, and its temperature, which defaults to 1). With floor 0 and
   temperature 1 that is the scripted agent unchanged. Say in your notes how you did it
   (for example a controller wrapper in `bnb.runtime.cloning`), and keep the scripted
   agent's own play unchanged everywhere else.
3. **The toy test** (`tests/sections/test_bc_toy.py`) keeps its bar (0.45 minus two
   standard errors of its 512 games) and scores the clone against the same teacher as
   the stop rule. Try it at the **default floor**. Remove its `xfail` mark once it passes.
   If the run hits its budget (100M) before the score stop, do not raise the budget or
   change the target: stop, record the curve (live score, its lower bound and the KL
   against steps) in the notes, and say so in your final message.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-bc-stop`, branch `fix/bc-stop` (from v2). Work
  only here; scratch in `/tmp/bnb-scratch-bc-stop`.
- **Do not read or modify** any other directory under `/home/vizia/avesnova`.
- **Python:** `uv sync`, then always `uv run --no-sync ...`.
- **GPU:** you are the only agent on the GPU, but the coordinator may measure too. Every
  GPU run (tests included) holds the benchmark lock: `bench.lock.bench_lock()` in a
  script, or a small wrapper that takes it around a command. Never plain `flock`.
- **Keep your context small:** read only what a step needs; long output to files, read
  their tails. Targeted tests while you work; the full suite once, at the end.

## Read first

`docs/rewrite/sections/bc.md` (decisions), the BC part of
`docs/rewrite/sections/scripted-bc-notes.md` (design, the toy result, the fix round),
`src/bnb/learn/bc.py`, `src/bnb/runtime/cloning.py`, `tests/sections/test_bc.py`,
`tests/sections/test_bc_toy.py`, and `docs/contributing/code-style.md`.

## Tests

- Unit tests of the stop rule: the bound's arithmetic on a known window (including a
  trimmed oldest update), a window at exactly the target does not stop, one clearly above
  does, and the KL plateau and budget rules are unchanged.
- A test that the scoring opponent's action distribution equals the labels' (floor and
  temperature applied), and that floor 0, temperature 1 leaves the scripted agent's play
  unchanged.
- Break each new behaviour by hand, watch its test fail, revert.
- Everything passes: `uv run --no-sync pytest` (under the lock). Never change
  `tests/reference/` or the goldens.

## Docs

Update the BC pages (`docs/explanation/behaviour-cloning.md`, `docs/guides/behaviour-cloning.md`)
where they describe the stop rule. The docs tests run in the suite; follow
`docs/contributing/writing-docs.md`.

## Commits

Commit to `fix/bc-stop` and push (`git push -u origin fix/bc-stop`). Author Julian Kizanis
<julianzia@gmail.com>; end every commit message with
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Hand back

Add a section "Stop rule (2026-11-02)" to `docs/rewrite/sections/scripted-bc-notes.md`:
what changed, the toy run's result (where it stopped, by which rule, the fresh score),
and the suite's result. Your final message summarises the same in half a page.
