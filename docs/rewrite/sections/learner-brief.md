# Brief: section 6, the learner

You are implementing **the learner** for Boost and Broadside v2: PPO with v1's math
(per-level returns, the categorical critic, the return scale, the auxiliary losses), the
labels, shot re-simulation, streaming the rollout through pinned CPU RAM, the toy
learner, and the precision matrix. The goals are **speed and maintainability together**:
the fastest straightforward design, and code the next person can read and change.

## Where you work

- **Your worktree:** `/home/vizia/avesnova/bnb-learner`, branch `section/learner`. Work
  only here.
- **Scratch:** use `/tmp/bnb-scratch-learner` for temporary files.
- **Do not read or modify** any other directory under `/home/vizia/avesnova`. In
  particular, stay out of the other worktrees, the v1 code, and the `physics-race*`
  repositories.
- **Python:** set up with `uv sync` in your worktree, then always run through
  `uv run --no-sync ...` (never plain `python`).

## Read first

1. `docs/rewrite/sections/learner.md`: the decisions behind this section.
2. `docs/rewrite/sections/learner-spec.md`: **the spec you implement**.
3. `tests/reference/learner/README.md`: the fixture interface `learner_arm.py` adapts.
4. `docs/rewrite/sections/yemong-spec.md` and `yemong-notes.md`: the model you train, its
   heads, codes, replay and `unroll`.
5. `docs/rewrite/sections/rewards-spec.md`, `modes-notes.md` and `mixed-notes.md`: the
   rewards, the modes and the mixed batch the learner consumes.
6. `docs/rewrite/contracts.md` (the current-schema section) and `src/bnb/contracts/`.
7. `docs/rewrite/scaffold.md`: section 3 (design rules) and section 4 (layout and
   dependency rule).
8. `src/bnb/`, especially `learn/`, `runtime/`, `models/`, `controllers/`, the game's
   bullet code (`game/physics/bullets.py`, `combat.py`), and `tests/`.

## What to do

1. **Implement the spec.** It replaces the skeleton's `bnb/learn/ppo.py`, and the toy
   learner replaces the skeleton's toy learning test in `tests/loop/test_learning.py`.
2. **Tests:**
   - **Everything must pass:** `uv run --no-sync pytest`, nothing skipped except what
     needs hardware you lack.
   - **Fixed:** the files under `tests/reference/` other than the adapters
     (`learner_arm.py`, `race_arm.py`, `race_chaser.py`, `rewards_arm.py`,
     `yemong_arm.py`), which you may change only where the new code forces it. Never
     change the other files, the goldens included.
   - **`learner_arm.py`:** make `make_arm()` return an implementation whose methods are
     thin translations onto `bnb.learn`. The fixtures must test the learner's own code,
     not a copy written for them.
   - **Add your own** under `tests/sections/` for every acceptance item. Give a test that
     needs the GPU the `gpu` marker.
3. **Speed:**
   - Measure what the spec asks for (the **(measure)** items, the precision matrix, the
     throughput cells).
   - Any GPU timing must hold the benchmark lock (`bench.lock.bench_lock()` in your
     script). Never wrap a lock-taking command in plain `flock`, and never run GPU tests
     while your own benchmark holds the GPU.
   - Keep it proportionate: the precision matrix's learning comparisons are short
     toy-learner runs, not long trainings.
4. **Commit and push often:**
   - commit to `section/learner`, then `git push -u origin section/learner`;
   - author Julian Kizanis <julianzia@gmail.com>;
   - end every commit message with
     `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Boundaries

- **You may change:**
  - `src/bnb/learn/**` and `src/bnb/runtime/**`;
  - `src/bnb/models/relational.py` (new), and the stub models where the toy learner
    replaces them;
  - `src/bnb/cli.py`;
  - `src/bnb/contracts/**`, only for the spec's additions (what `Trajectory` stores: the
    privileged rows, every row's action, the shot capture; the learner's configuration);
  - `src/bnb/game/**`, only to **expose** each ship's launch state for the shot capture
    and to make the launch, flight and swept-hit functions callable by the
    re-simulation. Game behaviour must not change: the game's tests and fixtures must
    pass untouched;
  - `src/bnb/models/yemong/**`, only for replay compaction and what the losses need from
    `unroll`;
  - `src/bnb/controllers/**`, only where the change forces it;
  - `tests/sections/**`, `tests/loop/test_learning.py`, the adapters above, and your
    notes file.

  List every change outside `bnb.learn`, `bnb.runtime` and `bnb.models.relational` in
  your notes.
- **You may not change:** `bench/`, the fixed tests and goldens, or the docs (apart from
  your notes file).
- **Layering:** `bnb.learn` may import `bnb.game` for the re-simulation, as the spec says.
  Reuse the game's functions; never reimplement them.
- **If the spec or a contract looks wrong:** take the most literal reading, keep going,
  and write it up as a request in your notes.
- **Names:** full words.
- **No legacy:** no shims or dead code. The skeleton PPO goes.
- **Precision:** float32 by default. The matrix's other configurations are settings, and
  recurrent state, norm statistics, scan accumulations and loss reductions stay float32
  in all of them.

## Hand back

Write `docs/rewrite/sections/learner-notes.md` with:
- **Design.**
- **Decisions and deviations,** each with its reason.
- **Measurements:** the precision matrix, streaming cost and bytes, the toy learner's
  budget and win rate, the throughput cells.
- **Changes outside `bnb.learn`, `bnb.runtime` and `bnb.models.relational`.**
- **Requests and open questions.**
- **Size.**

Your final message should summarize the same, in a page or less.
