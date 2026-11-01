# Brief: section 3c, mixed batches

You are implementing **mixed batches** for Boost and Broadside v2: several games (team
deathmatch on the squad and battle maps, Frontline, fogged and clear) in one batch, one
set of tensors and one compiled graph. The goals are **speed and maintainability
together**: the fastest straightforward design, and code the next person can read and
change.

## Where you work

- **Your worktree:** `/home/vizia/avesnova/bnb-mixed`, branch `section/mixed`. Work only
  here.
- **Scratch:** use `/tmp/bnb-scratch-mixed` for temporary files.
- **Do not read or modify** any other directory under `/home/vizia/avesnova`. In
  particular, stay out of the other worktrees, the v1 code, and the `physics-race*`
  repositories.
- **Python:** set up with `uv sync` in your worktree, then always run through
  `uv run --no-sync ...` (never plain `python`).

## Read first

1. `docs/rewrite/sections/learner.md`: the decisions behind this section.
2. `docs/rewrite/sections/mixed-spec.md`: **the spec you implement**.
3. `docs/rewrite/sections/modes-spec.md`, `modes-notes.md` and `yemong-notes.md`: the
   modes and the model you extend.
4. `docs/rewrite/contracts.md` (the current-schema section) and `src/bnb/contracts/`.
5. `docs/rewrite/scaffold.md`: section 3 (design rules) and section 4 (layout and
   dependency rule).
6. `src/bnb/`, especially `game/`, `runtime/`, `models/yemong/`, and `tests/`.

## What to do

1. **Implement the spec.**
2. **Tests:**
   - **Everything must pass:** `uv run --no-sync pytest`, nothing skipped except what
     needs hardware you lack.
   - **Fixed:** the files under `tests/reference/` other than the adapters (`race_arm.py`,
     `race_chaser.py`, `rewards_arm.py`, `yemong_arm.py`), which you may change only
     where the new configuration forces it. Never change the other files, the goldens
     included.
   - **Add your own** under `tests/sections/` for every acceptance item. Give a test that
     needs the GPU the `gpu` marker.
3. **Speed:**
   - Measure what the spec asks for.
   - Any GPU timing must hold the benchmark lock (`bench.lock.bench_lock()` in your
     script). Never wrap a lock-taking command in plain `flock`, and never run GPU tests
     while your own benchmark holds the GPU.
   - Keep it proportionate.
4. **Commit and push often:**
   - commit to `section/mixed`, then `git push -u origin section/mixed`;
   - author Julian Kizanis <julianzia@gmail.com>;
   - end every commit message with
     `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Boundaries

- **You may change:**
  - `src/bnb/game/**` and `src/bnb/runtime/**`;
  - `src/bnb/cli.py`;
  - `src/bnb/contracts/**`, only for the spec's additions (`Pool` fields, the zone
    `present` column, what moving the mode to pools forces in `GameConfig`);
  - `src/bnb/models/yemong/**`, only for absent zones and their rotation;
  - `src/bnb/learn/**` and `src/bnb/controllers/**`, only where the change forces it;
  - `tests/sections/**`, the adapters above, and your notes file.

  List every change outside `bnb.game` and `bnb.runtime` in your notes.
- **You may not change:** `bench/`, the fixed tests and goldens, or the docs (apart from
  your notes file).
- **If the spec or a contract looks wrong:** take the most literal reading, keep going,
  and write it up as a request in your notes. Items marked (proposal): implement them as
  written and list them.
- **Names:** full words.
- **No legacy:** no shims or dead code.
- **Precision:** float32.

## Hand back

Write `docs/rewrite/sections/mixed-notes.md` with:
- **Design.**
- **Decisions and deviations,** each with its reason.
- **Measurements.**
- **Changes outside `bnb.game` and `bnb.runtime`.**
- **Requests and open questions.**
- **Size.**

Your final message should summarize the same, in a page or less.
