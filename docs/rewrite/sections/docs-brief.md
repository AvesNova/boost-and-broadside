# Brief: the documentation site

You are building **the documentation** for Boost and Broadside v2: a MkDocs Material site
whose reference is generated from the code, plus checks that make stale docs fail the
test suite. The goal is documentation that is **easy to traverse, correct, and stays
correct** as the code changes.

## Where you work

- **Your worktree:** `/home/vizia/avesnova/bnb-docs`, branch `docs/site`. Work only here.
- **Scratch:** use `/tmp/bnb-scratch-docs` for temporary files.
- **Do not read or modify** any other directory under `/home/vizia/avesnova`. In
  particular, stay out of the other worktrees, the v1 code, and the `physics-race*`
  repositories.
- **Python:** set up with `uv sync` in your worktree, then always run through
  `uv run --no-sync ...` (never plain `python`).
- **No GPU.** Other agents are using it. Run the suite with `-m "not gpu"` and say so in
  your report.
- **Nothing is published.** Build locally only; do not deploy anywhere.

## Read first

1. `README.md`, `STYLE_GUIDE.md`, `pyproject.toml`.
2. `docs/rewrite/scaffold.md` (layout, dependency rule, section plan) and
   `docs/rewrite/contracts.md` (the current-schema section).
3. The merged sections' specs and notes under `docs/rewrite/sections/`: `game-*`,
   `modes-*`, `rewards-spec.md`, `yemong-*`, `mixed-*`. Also the decision docs for the
   unbuilt sections (`learner.md`, `ship-card.md`, `scripted.md`, `bc.md`, `league.md`),
   for the roadmap only.
4. `src/bnb/` and `tests/`.

## What to build

### 1. The site

- **MkDocs Material with mkdocstrings** (Python handler), in a `docs` optional extra in
  `pyproject.toml`; `mkdocs.yml` at the repo root.
- **User docs live in `docs/`.** `docs/rewrite/` holds the internal design records: keep
  them out of the site (`exclude_docs`) and do not edit them.
- **Four kinds of page:**
  - **Getting started:** installation (uv, Python and torch versions, the CUDA
    requirement), running the tests, a first training run, a benchmark.
  - **Guides:** task-oriented. At least: running and configuring training (presets,
    pools, the mixed batch), benchmarking (the lock, cells), working with the reference
    fixtures (`REFERENCE_ARM`, what goldens are, never editing them), adding a mode.
  - **Reference:** generated. One page per package (`contracts`, `game`, `controllers`,
    `models`, `learn`, `runtime`, `cli`, `bench`) from docstrings and signatures.
  - **Explanation:**
    - the architecture: the layers and the dependency rule, the stateful/pure split,
      compiled stages and CUDA graphs;
    - the game: physics, perception, fog, modes, rewards;
    - the model (Yemong);
    - the training loop;
    - the roadmap: sections and their status.
- **Only merged, stable parts get prose:** setup, architecture, contracts, game, modes,
  rewards, Yemong, mixed batches. The learner, ship card, scripted agent, BC, league,
  diagnostics and play shell get a short placeholder each, saying it is in progress and
  what it will cover. The generated reference covers whatever exists.
- **The README** becomes a short entry point: what the project is, the quickstart, and
  links into the site. Keep its "previous code is at tag `v1-torch-final`" line.

### 2. Generated content

- **Tables from the code at build time,** never copied by hand: the observation's ship,
  field, zone and global columns; `REWARD_LEVELS`; the action layout (`NUM_POWER`,
  `NUM_STICK`, `NUM_SHOOT`, `STICK_LEVELS`, `POWER_MODES`); the ship card defaults; the
  game presets; the training mix.
- **How:** an MkDocs hook or plugin (for example `mkdocs-macros`, or a small hook in
  `docs/hooks/`) that imports `bnb` and renders the table where a page asks for it.
- **The roadmap's status** comes from one small data file (for example
  `docs/roadmap.yml`: section, title, status, spec link), not from prose.

### 3. Staleness checks, inside `pytest`

Put them under `tests/docs/`, so every agent's normal `uv run --no-sync pytest` runs
them.

1. **The site builds in strict mode** (`mkdocs build --strict` into a temporary
   directory). Broken cross-references, missing pages and bad mkdocstrings targets fail.
   Mark it `slow` if it takes more than a few seconds.
2. **Every code reference in the prose resolves.** Dotted names (`bnb.x.y`, as
   mkdocstrings links or in backticks) must import, and repository paths
   (`src/bnb/...`, `tests/...`, `bench/...`) must exist. A failure names the page and
   line.
3. **Examples run.** Fenced command and Python blocks marked as runnable execute, for
   example with a marker comment or a fence attribute. Keep runnable examples CPU-only
   and fast; mark the rest as not runnable.
4. **Coverage map, warning only.** Each prose page declares what it covers in front
   matter (`covers: [src/bnb/game/physics/]`).
   - A check compares git history and emits a **warning** (not a failure) listing pages
     whose covered code changed in commits after the page's last change.
   - Also expose it as a command, for example `uv run --no-sync python -m
     tests.docs.staleness`, that prints the report.
   - It must not fail on a shallow clone or outside git: skip cleanly.
5. **Docstrings of the public API.** A report, not a gate: list the public modules,
   classes and functions with a missing or one-word docstring. Do not edit `src/`: other
   agents are changing it. Put the list in your notes.

### 4. The style guide for docs

Write `docs/contributing/writing-docs.md`:
- what each kind of page is for;
- how to link code, embed a generated table, mark a runnable example, declare
  `covers:`;
- the voice: say what exists, why it exists and how it works; neutral and precise; no
  self-praise, no evaluative adjectives, no design-debate history (that belongs in
  `docs/rewrite/`); every number says where it was measured; short sentences, full words.

## Tests

- **Everything must pass:** `uv run --no-sync pytest -m "not gpu"`, plus ruff.
- **Do not change** anything under `src/`, `bench/`, `tests/` outside `tests/docs/`, or
  `docs/rewrite/`.

## Commit and push often

- Commit to `docs/site`, then `git push -u origin docs/site`.
- Author Julian Kizanis <julianzia@gmail.com>.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Boundaries

- **You may change:** `docs/` (except `docs/rewrite/`), `mkdocs.yml`, `README.md`,
  `pyproject.toml` (the `docs` extra and its tool config only), `uv.lock` as the extra
  forces, and `tests/docs/**`.
- **Facts come from the code and the merged specs and notes.** Where they disagree, the
  code wins; list the disagreement in your notes. Do not invent behaviour, numbers or
  plans.
- **Names:** full words. **No legacy:** nothing stale, no placeholder text beyond the
  in-progress pages.

## Hand back

Write `docs/rewrite/sections/docs-notes.md` (the one file you may add under
`docs/rewrite/`) with:
- **Design:** the site layout, the generation mechanism, each check.
- **Decisions and deviations,** each with its reason.
- **Disagreements between docs and code** that you found.
- **The docstring report.**
- **How a section agent keeps the docs current:** a short paragraph we can paste into
  future briefs.
- **Size.**

Your final message should summarize the same, in a page or less.
