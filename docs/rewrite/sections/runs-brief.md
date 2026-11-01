# Brief: section 11, runs and checkpoints

You are implementing **runs and checkpoints** for Boost and Broadside v2:
- the run directory and its record;
- policy files and training checkpoints that carry their own settings;
- model versions with a compatibility set;
- exact resume, fork and warm start;
- the memory probe;
- provenance for offline results;
- explicit uploads.

The goals are **correctness and maintainability**: a resumed run must equal an
uninterrupted one, a refusal must say exactly why, and the code must be small enough to
read in one sitting.

## Where you work

- **Your worktree:** `/home/vizia/avesnova/bnb-runs`, branch `section/runs`. Work only
  here.
- **Scratch:** use `/tmp/bnb-scratch-runs` for temporary files.
- **Do not read or modify** any other directory under `/home/vizia/avesnova`. In
  particular, stay out of the other worktrees, the v1 code, and the `physics-race*`
  repositories.
- **Python:** set up with `uv sync` in your worktree, then always run through
  `uv run --no-sync ...` (never plain `python`).

## Keep your context small

- Read only the files a step needs.
- Send long output (test runs) to a file and read its tail.
- Run targeted tests while you work, and the full suite at milestones.

## Read first

1. `docs/rewrite/sections/runs.md`: the decisions behind this section.
2. `docs/rewrite/sections/runs-spec.md`: **the spec you implement**.
3. `docs/rewrite/sections/learner-spec.md` and `learner-notes.md`: the trainer, the
   learner and the store you checkpoint.
4. `docs/rewrite/contracts.md` (the current-schema section) and `src/bnb/contracts/`.
5. `docs/rewrite/scaffold.md` (section 3, design rules; section 4, the layer order) and
   `docs/contributing/code-style.md`.
6. `src/bnb/runtime/` (`train.py`, `rollout.py`, `store.py`), `src/bnb/learn/ppo.py`,
   `src/bnb/models/yemong/model.py`, `src/bnb/models/relational.py`, `src/bnb/cli.py`.

## What to do

1. **Implement the spec,** in this order:
   1. `save_policy` and `load_policy` first, pushed early: the BC section calls them;
   2. serialisation;
   3. the registry, versions and build records;
   4. training checkpoints and the exact resume;
   5. the run directory and the CLI;
   6. the probe, provenance and uploads.
2. **Tests:**
   - **Everything must pass:** `uv run --no-sync pytest`, nothing skipped except what
     needs hardware you lack.
   - **Fixed:** everything under `tests/reference/` and the goldens; never change them.
     Splitting Yemong's constructor (spec 4) must leave the Yemong fixtures passing.
   - **Add your own** under `tests/runs/` for every acceptance item. Give a test that
     needs the GPU the `gpu` marker, and one over 10 s the `slow` marker.
   - **Break each new behaviour by hand,** watch its test fail, and revert.
3. **Speed and size:**
   - Measure what spec acceptance 12 asks for, with the benchmark lock held
     (`bench.lock.bench_lock()` in your script). Never wrap a lock-taking command in
     plain `flock`, and never run GPU tests while your own benchmark holds the GPU.
   - If other agents are loading the machine, mark the numbers provisional; the
     coordinator re-measures on a quiet machine.
4. **Docs.** The docs in `docs/` and `README.md` are checked by the suite.
   - `uv run --no-sync pytest` fails if any of these breaks: a backticked name, a
     repository path, a `bnb` command option, a runnable example, or
     `mkdocs build --strict`. Fix those as part of your change.
   - Tables of values from the code are generated; never copy them into prose.
   - When you change code that a page lists under `covers:`, pytest warns
     `PossiblyStalePage`. Reread that page and fix it, or set its `reviewed:` date if it
     is still correct (`uv run --no-sync python -m tests.docs.staleness` lists them).
   - When your section is done, replace `docs/explanation/in-progress/runs.md` with
     prose (an explanation, plus the guide the spec names) and set its status in
     `docs/roadmap.yml`.
   - Follow `docs/contributing/writing-docs.md`: say what exists and why; no evaluative
     adjectives; no design history; every number says where it was measured.
5. **Commit and push often:**
   - commit to `section/runs`, then `git push -u origin section/runs`;
   - author Julian Kizanis <julianzia@gmail.com>;
   - end every commit message with
     `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Boundaries

- **You may change:**
  - `src/bnb/runs/**` (new);
  - `src/bnb/cli.py`;
  - `src/bnb/runtime/**` and `src/bnb/learn/**`, only where checkpointing needs a hook
    (state to save, a resume path);
  - `src/bnb/models/**`, for the registry, `VERSION`, `build_record` and the
    constructor split;
  - `src/bnb/contracts/**`, only for the serialisation helper if it belongs there;
  - `pyproject.toml` (safetensors), `.gitignore` (`runs/`);
  - `tests/**` outside `tests/reference/`;
  - the user docs (`docs/` outside `docs/rewrite/`, `README.md`), and your notes file.

  List every change outside `bnb.runs` in your notes.
- **You may not change:** `bench/`, the fixed tests and goldens, `bnb/play/`,
  `bnb/controllers/scripted.py`, or `docs/rewrite/` (apart from your notes file).
- **Other open branches:** the scripted agent and BC may run beside you. Keep out of
  their files, and merge `origin/v2` whenever it moves.
- **If the spec or a contract looks wrong:** take the most literal reading, keep going,
  and write it up as a request in your notes. Items marked (proposal): implement them as
  written and list them.
- **Names:** full words.
- **No legacy:** no shims or dead code.

## Hand back

Write `docs/rewrite/sections/runs-notes.md` with:
- **Design.**
- **Decisions and deviations,** each with its reason.
- **Measurements:** checkpoint sizes and times, throughput with and without
  checkpoints, and the exact-resume result (bitwise or within what tolerance, and why).
- **Changes outside `bnb.runs`.**
- **Requests and open questions.**
- **Size.**
- **A Progress section,** kept current as you go (what is done, what is in flight, what
  is next), so a fresh agent could take over.

Your final message should summarize the same, in a page or less.
