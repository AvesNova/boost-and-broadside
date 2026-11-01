# Brief: the cleanup pass

You are doing **the cleanup pass** for Boost and Broadside v2. It covers names, dead and
fixture-shaped code, the runtime's reach into the game state, docstrings and the
contracts document, duplication, test hygiene, and the tests a mutation sweep found
missing. **Nothing the game, models or learner compute may change.** The goal is a tree
the next person can read and trust: one name per idea, one definition per value, and tests
that fail when the behaviour breaks.

## Where you work

- **Your worktree:** `/home/vizia/avesnova/bnb-cleanup`, branch `section/cleanup`. Work
  only here.
- **Scratch:** use `/tmp/bnb-scratch-cleanup` for temporary files.
- **Do not read or modify** any other directory under `/home/vizia/avesnova`. In
  particular, stay out of the other worktrees, the v1 code, and the `physics-race*`
  repositories.
- **Python:** set up with `uv sync` in your worktree, then always run through
  `uv run --no-sync ...` (never plain `python`).

## Read first

1. `docs/rewrite/sections/cleanup-spec.md`: **the spec you implement**.
2. `docs/rewrite/audits/2026-11-01-consistency.md`: the audit behind it. It was taken at
   an older commit, so check every line it cites against the current tree.
3. `docs/contributing/code-style.md` and `docs/contributing/writing-docs.md`: the rules
   your changes follow.
4. `docs/rewrite/scaffold.md`: section 3 (design rules) and section 4 (layout and the
   dependency rule).
5. `docs/rewrite/contracts.md`: the conventions and the current-schema section.
6. The section notes, for what each section decided: `docs/rewrite/sections/*-notes.md`.

## What to do

1. **Implement the spec,** in this order:
   1. the contract renames (spec 1.1), in one commit, pushed at once;
   2. names;
   3. dead code and the opaque state;
   4. the sweep's tests;
   5. docs;
   6. layering and duplication;
   7. test hygiene.

   Commit each group separately, so each can be reviewed alone.
2. **Tests:**
   - **Everything must pass:** `uv run --no-sync pytest`, nothing skipped except what
     needs hardware you lack.
   - **Fixed:** everything under `tests/reference/`, except the two adapters the spec
     names (`rewards/rewards_arm.py` and the Yemong reference adapter). Never change the
     goldens.
   - **Add your own** under `tests/sections/`, `tests/contract/` or `tests/loop/`, as
     fits. Give a test that needs the GPU the `gpu` marker, and a test over 10 s the
     `slow` marker.
   - **For each sweep test** (spec section 3), apply its mutation by hand, watch the
     test fail, revert, and note it.
3. **Speed:**
   - Run the bench cells before you start and again at the end, on the same machine.
   - Any GPU timing must hold the benchmark lock (`bench.lock.bench_lock()` in your
     script). Never wrap a lock-taking command in plain `flock`, and never run GPU tests
     while your own benchmark holds the GPU.
   - Other agents may be loading the machine. Record the load average with each number,
     and call numbers taken under load provisional.
4. **Docs.** The docs in `docs/` and `README.md` are checked by the suite.
   - `uv run --no-sync pytest` fails if any of these breaks: a backticked name, a
     repository path, a `bnb` command option, a runnable example, or
     `mkdocs build --strict`. Your renames will break some of these; fix them all.
   - Tables of values from the code are generated; never copy them into prose.
   - When you change code that a page lists under `covers:`, pytest warns
     `PossiblyStalePage`. Reread that page and fix it, or set its `reviewed:` date if it
     is still correct (`uv run --no-sync python -m tests.docs.staleness` lists them).
   - Follow `docs/contributing/writing-docs.md`: say what exists and why; no evaluative
     adjectives; no design history; every number says where it was measured.
5. **Stay mergeable:**
   - merge `origin/v2` into your branch at least once a day, and whenever it moves;
   - other branches are open while you work (the scripted agent and BC, the play shell).
     Keep your diff out of their files: `bnb/controllers/scripted.py` and `bnb/play/`.
6. **Commit and push often:**
   - commit to `section/cleanup`, then `git push -u origin section/cleanup`;
   - author Julian Kizanis <julianzia@gmail.com>;
   - no `Co-Authored-By` or other attribution lines in the message.

## Boundaries

- **You may change:**
  - `src/**`, `bench/`, and `tests/` outside `tests/reference/` (apart from the two
    adapters);
  - `docs/rewrite/contracts.md`, the user docs (`docs/` outside `docs/rewrite/`), and
    `README.md`;
  - `pyproject.toml`, for markers and the ruff exemptions;
  - `STYLE_GUIDE.md`, to delete it;
  - your notes file.
- **You may not change:**
  - the goldens, or `tests/reference/` beyond the two adapters;
  - `docs/rewrite/` other than `contracts.md` and your notes;
  - `bnb/controllers/scripted.py`, `bnb/play/`.
- **No behaviour change.** If you find a bug, do not fix it here. Write a test marked
  `xfail(strict=True)` with the reason, and list it in the notes; it is fixed
  separately.
- **If the spec or a contract looks wrong:** take the most literal reading, keep going,
  and write it up as a request in your notes. Items marked (proposal): implement them as
  written and list them.
- **Names:** full words.
- **No legacy:** no shims, aliases for old names, or dead code. A rename renames every
  use.

## Hand back

Write `docs/rewrite/sections/cleanup-notes.md` with:
- **Each audit item and each sweep item,** with what was done, or why not ("already fixed
  by section 6", "kept: named by the league spec", "equivalent mutation: …").
- **Decisions and deviations,** each with its reason.
- **Bugs found** (the `xfail` tests).
- **Measurements:** the bench cells before and after, the rollout change, the test-tier
  times, the slow-test durations.
- **Requests and open questions.**
- **Size.**

Your final message should summarize the same, in a page or less.
