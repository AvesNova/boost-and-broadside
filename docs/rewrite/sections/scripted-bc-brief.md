# Brief: sections 4 and 7, the scripted agent and BC

You are implementing **the scripted agent** and **behaviour cloning** for Boost and
Broadside v2, in two stages. First the scripted agent, a memoryless controller that
reads only the observation and gives a 42-way action distribution, with its acceptance
passing. Then BC over the learner's machinery, with the scripted agent as the teacher.
The goals are **speed and maintainability together**: the fastest straightforward
design, and code the next person can read and change.

## Where you work

- **Your worktree:** `/home/vizia/avesnova/bnb-scripted`, branch `section/scripted`. Work
  only here.
- **Scratch:** use `/tmp/bnb-scratch-scripted` for temporary files.
- **Do not read or modify** any other directory under `/home/vizia/avesnova`. In
  particular, stay out of the other worktrees, the v1 code, and the `physics-race*`
  repositories.
- **Python:** set up with `uv sync` in your worktree, then always run through
  `uv run --no-sync ...` (never plain `python`).

## Read first

1. `docs/rewrite/sections/scripted.md` and `bc.md`: the decisions behind these sections.
2. `docs/rewrite/sections/scripted-bc-spec.md`: **the spec you implement**.
3. `docs/rewrite/sections/learner-spec.md` and `learner-notes.md`: the learner you build
   BC on.
4. `docs/rewrite/sections/game-spec.md` (flight, bullets, arena),
   `modes-spec.md`, `mixed-notes.md`, and `ship-card-spec.md` (its measurement module,
   which the scripted agent's characteristics share).
5. `docs/rewrite/contracts.md` (the current-schema section) and `src/bnb/contracts/`.
6. `docs/rewrite/scaffold.md`: section 3 (design rules) and section 4 (layout and
   dependency rule).
7. `src/bnb/`, especially `controllers/`, `learn/`, `runtime/`, `models/relational.py`,
   `game/`, and `tests/`.

## What to do

1. **Implement the spec, part A then part B.** Commit part A with its acceptance
   passing before starting part B.
2. **Tests:**
   - **Everything must pass:** `uv run --no-sync pytest`, nothing skipped except what
     needs hardware you lack.
   - **Fixed:** the files under `tests/reference/` other than the adapters, which you
     may change only where the new code forces it. Never change the other files, the
     goldens included.
   - **Add your own** under `tests/sections/` for every acceptance item. Give a test that
     needs the GPU the `gpu` marker.
3. **Speed and budgets:**
   - Measure what the spec asks for.
   - **The parameter search is capped at about an hour of GPU time.** The scripted
     agent's job is a sensible distribution, not peak strength: if it does not reach
     0.70 against the chaser after the capped search, report the score and move on.
   - Any GPU timing must hold the benchmark lock (`bench.lock.bench_lock()` in your
     script). Never wrap a lock-taking command in plain `flock`, and never run GPU tests
     while your own benchmark holds the GPU.
4. **Commit and push often:**
   - commit to `section/scripted`, then `git push -u origin section/scripted`;
   - author Julian Kizanis <julianzia@gmail.com>;
   - end every commit message with
     `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Boundaries

- **You may change:**
  - `src/bnb/controllers/**`;
  - `src/bnb/game/handling.py` (create it, or extend it if section 1b has merged), for
    the characteristics only; game behaviour must not change;
  - `src/bnb/learn/**` and `src/bnb/runtime/**`, for BC and its rollouts;
  - `src/bnb/cli.py`;
  - `src/bnb/contracts/**`, only for the spec's additions;
  - `tests/sections/**`, the adapters above, and your notes file.

  List every change outside `bnb.controllers` and `bnb.learn` in your notes.
- **You may not change:** `bench/`, the fixed tests and goldens, or the docs (apart from
  your notes file).
- **The scripted agent reads only the observation.** Never `Privileged`, never game
  state.
- **If the spec or a contract looks wrong:** take the most literal reading, keep going,
  and write it up as a request in your notes. Items marked (proposal): implement them as
  written and list them.
- **Names:** full words.
- **No legacy:** no shims or dead code.
- **Precision:** float32.

## Hand back

Write `docs/rewrite/sections/scripted-bc-notes.md` with:
- **Design.**
- **Decisions and deviations,** each with its reason.
- **Measurements:**
  - the parameter search and the chosen values;
  - the scores against random and the chaser per mode;
  - the agent's cost per tick;
  - the teacher's natural entropy;
  - the toy BC result and budget;
  - BC throughput.
- **Changes outside `bnb.controllers` and `bnb.learn`.**
- **Requests and open questions.**
- **Size.**

Your final message should summarize the same, in a page or less.
