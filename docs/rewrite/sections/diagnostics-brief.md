# Brief: diagnostics (section 9)

You are building v2's diagnostics: the series training emits, how they are collected
without slowing training, where they go (W&B and a local JSONL), the trigger rules that
flag problems, and the dashboards. These are what the author will read to judge the first
BC and RL runs, so they must be right: every series checked against a case with a known
answer.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-diagnostics`, branch `section/diagnostics`
  (from v2). Work only here. Scratch in `/tmp/bnb-scratch-diagnostics`.
- **Do not read** any other directory under `/home/vizia/avesnova`: no v1, no
  `physics-race*`, no other worktree.
- **Python:** `uv sync`, then always `uv run --no-sync ...`. Add a library when a task
  needs it (for example `wandb-workspaces`); tooling-only ones go in an optional extra.
- **GPU:** every GPU run holds the benchmark lock (`bench.lock.bench_lock()`; never plain
  `flock`). The wrapper `/tmp/bnb-scratch-fast-yemong/locked.py <command...>` runs a
  command under it.
- **No network in tests:** W&B offline mode or the JSONL sink.
- **Keep your context small:** the audit is long. Read it by group as you build each,
  send long output to files, and read their tails.

## Read first

1. `docs/rewrite/sections/diagnostics-spec.md`: **the spec you implement.**
2. `docs/rewrite/diagnostics.md`: the audit, the normative list of series, rules and
   dashboards. Read "Rules and method", "Final trigger rules" and "Dashboard layout" in
   full; read each group when you build it.
3. `docs/rewrite/contracts.md`: the conventions, then the newest entries (the learner, BC,
   the kernels and bf16, the store, the league, "Resume: integrity, not bits").
4. `docs/rewrite/scaffold.md` sections 3 and 4, `docs/contributing/code-style.md`, and
   `docs/contributing/writing-docs.md`.
5. The code that produces series today: `src/bnb/learn/` (the learner and BC report),
   `src/bnb/runtime/train.py` and `rollout.py`, `src/bnb/league/` (its metrics), and
   `src/bnb/runs/` (the W&B upload).

## What to do

In this order, one commit or more each, pushed:
1. the collector and the sinks (W&B, JSONL), with the existing series moved onto them.
   Live logging (author, 2026-11-03): `bnb train --wandb` streams each update's record to
   W&B during the run (project `boost-and-broadside-v2`, the run's name), as well as to
   the local JSONL. Without the flag nothing leaves the machine, and `bnb runs upload`
   stays for runs trained offline;
2. the method-rule tests (no host sync, the naming check, the no-EMA check);
3. the series, group by group, each with its hand-computed case. Start with what the
   first BC and RL runs need most: the learner's losses and ratios, entropy and the
   action metrics, the critic, BC's score and labels, the episode and reward series per
   pool and mode, the league's gauge, throughput and memory. Then the rest of the audit;
4. the trigger rules, their state in the checkpoint, and the skipped-step action;
5. the dashboards;
6. the cost measurement and the coverage table.

## Tests

- **Fixed:** `tests/reference/` and the goldens.
- **Spec section 6** is the acceptance list: a test for each item.
- **The tiers:** the `gpu` marker for CUDA (skips cleanly without it); `slow` for a test
  over 10 s; the quick tier stays around 2.5 minutes. The suite's guards fail a non-`gpu`
  test that initialises CUDA, and a test that changes torch's global CPU generator.
- **Break by hand:** for a sample of series and for two trigger rules, introduce a bug,
  watch the test fail, revert, and note it.
- **Everything passes:** the full suite under the lock, once at the end.

## Boundaries

- **May change:** `src/bnb/runtime/metrics.py` (new) and the dashboards module; the
  producers' modules, only to emit their series (learner, BC, rollout analytics, league,
  model heads); `src/bnb/runs/` for the sinks and the rule state in checkpoints; the CLI;
  `pyproject.toml`; tests outside `tests/reference/`; the user docs; your notes.
- **Do not change** what the game, models, learner or league compute. Emitting a series
  must not change training: a test checks that the same seed gives the same update with
  diagnostics on and off.
- **`src/bnb/play/`** is out of scope; the replay of one environment (H1) is written in the
  format `bnb.play.replay` reads.
- **If the spec or audit looks wrong:** take the most literal reading, keep going, and
  write it up as a request. Items marked (proposal) are implemented as written and listed.
- **Names:** full words. **No legacy:** no shims.

## Commits

Commit to `section/diagnostics` and push (`git push -u origin section/diagnostics`).
Author Julian Kizanis <julianzia@gmail.com>; end every commit message with
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Hand back

Write `docs/rewrite/sections/diagnostics-notes.md` with:
- each spec item, with what was done;
- the coverage table;
- the cost measurements, with the load average;
- deviations, requests and open questions;
- a Progress section kept current, so a fresh agent can take over.

This is a long task. Stop at a clean commit if you have been working long. Your final
message summarizes it in a page or less.
