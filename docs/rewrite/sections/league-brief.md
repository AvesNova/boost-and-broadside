# Brief: the league (section 8)

You are implementing section 8 of Boost and Broadside v2: the opponent pool, its
sampling, the live rating gauge during training, snapshots, and the published ratings
after training (`bnb ratings`, `bnb head-to-head`).

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-league`, branch `section/league` (from v2).
  Work only here. Scratch in `/tmp/bnb-scratch-league`.
- **Do not read** any other directory under `/home/vizia/avesnova`: no v1, no
  `physics-race*`, no other worktree.
- **Python:** `uv sync`, then always `uv run --no-sync ...`.
- **GPU:** every GPU run holds the benchmark lock (`bench.lock.bench_lock()`; never plain
  `flock`). The wrapper `/tmp/bnb-scratch-fast-yemong/locked.py` runs a command under it:
  `uv run --no-sync python /tmp/bnb-scratch-fast-yemong/locked.py <command...>`.
- **Keep your context small:** read only what a step needs, send long output to files and
  read their tails.

## Read first

1. `docs/rewrite/sections/league-spec.md`: **the spec you implement.**
2. `docs/rewrite/sections/league.md`: the decisions behind it.
3. `tests/reference/league/README.md` and `league_arm.py`: the fixtures and the adapter
   you fill in (`make_arm()`). Six equivalence tests skip today until it exists.
4. `docs/rewrite/contracts.md`: the conventions, then the entries for section 11 (runs and
   checkpoints), the scripted agent and BC, the kernels, and the store.
5. `docs/rewrite/scaffold.md`: section 3 (design rules) and section 4 (layout and the
   dependency rule; `bnb.league` sits above `bnb.runs`).
6. `docs/contributing/code-style.md` and `docs/contributing/writing-docs.md`.

## What to do

Implement the spec in its own order, one commit per part, each pushed:
1. the ratings (fits, live gauge, `rate_against`, `allocate`) against the fixtures;
2. the pool, retention and snapshots;
3. the slots, the stationary ladder, the drain and in-graph sampling, in the captured
   rollout;
4. resume: the pool, ratings, records and sampling state in the training checkpoint;
5. the metrics;
6. `bnb ratings` and `bnb head-to-head`.

## Tests

- **Fixed:** `tests/reference/` and the goldens, except `tests/reference/league/league_arm.py`,
  which you fill in.
- **Spec section 8** is the acceptance list: write a test for each item.
- **The tiers:** a test that needs the GPU carries the `gpu` marker and skips cleanly
  without CUDA; a test over 10 s carries `slow`. The quick tier (`-m "not slow"`) stays
  around 2.5 minutes. The suite's guards fail a non-`gpu` test that initialises CUDA and a
  test that changes torch's global CPU generator (use the `global_generator` fixture).
- **Exact resume** (`tests/runs/test_resume.py`) must keep passing, and a resumed league
  run must equal a straight one (spec 8.6).
- **Break by hand:** for the sampling, the drain and the gauge, introduce a bug, watch a
  test fail, revert, and note it.
- **Everything passes:** the full suite under the lock, once at the end, nothing skipped
  except what needs hardware you lack.

## Measure

- Spec 8.7: the rollout with the league against self-play only, at the 5v5 training cell
  with Yemong (bf16, kernels on), and the update's time. Record the load average with
  every number; numbers under load are provisional.

## Boundaries

- **You may change:** `src/bnb/league/**`; `src/bnb/runtime/**` and `src/bnb/runs/**` as
  the league needs (slots in the rollout, checkpoint contents); `src/bnb/cli.py`; tests
  outside `tests/reference/` plus `league_arm.py`; the user docs; your notes.
- **Do not change** the game, the models, the learner's losses, or `bnb/play/`.
- **No behaviour change outside the league:** a run without opponents trains exactly as
  before.
- **If the spec looks wrong or underspecified:** take the most literal reading, keep
  going, and write it up as a request in your notes. Items marked (proposal) are
  implemented as written and listed.
- **Names:** full words. **No legacy:** no shims.

## Commits

Commit to `section/league` and push (`git push -u origin section/league`). Author Julian
Kizanis <julianzia@gmail.com>; end every commit message with
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Hand back

Write `docs/rewrite/sections/league-notes.md` with:
- each spec item, with what was done;
- the acceptance tests and the hand breaks;
- the measurements;
- deviations, requests and open questions;
- a Progress section kept current, so a fresh agent can take over.

This is a long task. Stop at a clean commit if you have been working long. Your final
message summarizes it in a page or less.
