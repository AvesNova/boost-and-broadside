# Brief: sections 3 and 3b, modes and rewards

You are implementing two coupled sections of Boost and Broadside v2:
- **Modes:** game rules in `bnb.game` (team deathmatch, Frontline, a sandbox).
- **Rewards:** events to per-level rewards, learning side, in `bnb.learn.rewards`.

The goals are **speed and maintainability together**: fast batched GPU training, the same
code at batch size 1 on the CPU, and code the next person can read and change.

## Where you work

- **Your worktree:** `/home/vizia/avesnova/bnb-modes`, branch `section/modes`. Work only
  here.
- **Scratch:** use `/tmp/bnb-scratch-modes` for temporary files.
- **Do not read or modify** any other directory under `/home/vizia/avesnova`. In
  particular, stay out of the other worktrees, the v1 code, and the `physics-race*`
  repositories. The goldens are your only view of v1, by design.
- **Python:** set up with `uv sync` in your worktree, then always run through
  `uv run --no-sync ...` (never plain `python`).

## Read first

1. `docs/rewrite/sections/modes.md` and `rewards.md`: the decisions, with the author's
   reasons.
2. `docs/rewrite/sections/modes-spec.md` and `rewards-spec.md`: **the specs you
   implement**.
3. `tests/reference/rewards/README.md`: the equivalence interface.
4. `docs/rewrite/sections/game-spec.md` (the game you extend), `game-notes.md`, and
   `yemong-notes.md` (the model whose inputs you extend).
5. `docs/rewrite/contracts.md` (the current-schema section) and `src/bnb/contracts/`.
6. `docs/rewrite/scaffold.md`: section 3 (design rules) and section 4 (layout and
   dependency rule).
7. `src/bnb/`, especially `game/`, `runtime/`, `models/yemong/`, and `tests/`.

## What to do

1. **Implement both specs:**
   - the modes (team deathmatch uniform and grouped, Frontline, sandbox);
   - their spawns;
   - the mode phase;
   - the new events, state and observation;
   - Yemong's zone token and global columns;
   - the rewards and their ledger;
   - the rollout wiring.
2. **Write the equivalence adapter** `tests/reference/rewards/rewards_arm.py`: a thin
   translation onto your code. If a method needs more than renaming and reshaping, your
   module's interface is probably wrong.
3. **Tests:**
   - **Everything must pass:** `uv run --no-sync pytest`, with the rewards equivalence
     tests running, not skipped.
   - **Fixed:** the files under `tests/reference/` other than `rewards_arm.py` and the
     race adapter (`race_arm.py`, `race_chaser.py`), which you may change only to select
     sandbox mode. Never change the other files, the goldens included.
   - **Add your own** under `tests/sections/` for every acceptance item of both specs.
     Give a test that needs the GPU the `gpu` marker.
4. **Speed:**
   - Report the cost of the mode phase and of the reward step at the bench sizes.
   - Any GPU timing must hold the benchmark lock (`bench.lock.bench_lock()` in your
     script). Never wrap a lock-taking command in plain `flock`.
   - Keep it proportionate: a few runs, not a sweep. Optimize the general path; no
     size-gated branches.
5. **Commit and push often:**
   - commit to `section/modes`, then `git push -u origin section/modes`;
   - author Julian Kizanis <julianzia@gmail.com>;
   - end every commit message with
     `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Boundaries

- **You may change:**
  - `src/bnb/game/**` and `src/bnb/learn/rewards.py` (new);
  - `src/bnb/contracts/**`, but only for the spec's additions: `GameConfig` fields,
    `Events` fields, `Obs.zones`, the global columns, and `Trajectory.reward`'s level axis;
  - `src/bnb/models/yemong/**`, only for the zone token, the global columns and the
    reward-level names;
  - `src/bnb/runtime/**`, to wire the rewards and the mode configuration;
  - `src/bnb/learn/ppo.py`, only to sum the reward levels;
  - `src/bnb/controllers/**`, only where the new observation forces it;
  - `tests/sections/**`, the two adapters named above, and your notes file.

  List every change outside `bnb.game` and `bnb.learn.rewards` in your notes.
- **You may not change:** `bench/`, the fixed tests and goldens, or the docs (apart from
  your notes file).
- **If a spec, contract or fixture looks wrong:** do not work around it silently. Take
  the most literal reading, keep going, and write it up as a request in your notes. Items
  marked (proposal) in the specs: implement them as written and list them.
- **Names:** full words (scaffold rule 10).
- **No legacy:** no shims or dead code.
- **Precision:** float32. Never store a rounded copy of a value a compiled stage also uses
  unrounded.
- **Smoothness and repeatability:** traded by judgement.

## Hand back

Write `docs/rewrite/sections/modes-notes.md` with:
- **Design:** the mode structure, the state, the mode phase, the spawn draws, the rewards
  and ledger.
- **Decisions and deviations:** each with its reason.
- **Measurements:** the speed report.
- **Changes outside `bnb.game`.**
- **Requests and open questions.**
- **Size:** line counts per module.

Your final message should summarize the same, in a page or less.
