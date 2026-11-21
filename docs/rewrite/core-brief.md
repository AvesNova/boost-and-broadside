# Operating brief: the lean core

Status: draft for the author's review, 2026-11-21. On the `core` branch this file becomes
the root `CLAUDE.md`, the instructions every agent working on the branch reads first.
Values marked *proposed* are starting targets for the author to confirm.

## 1. Purpose

The project has three aims, held together:

1. **A game people play and find fun.** Simple rules with a lot of depth. A human plays
   against trained agents in real time.
2. **Experiments that answer questions cleanly,** possibly a paper.
3. **Training that is quick and cheap in GPU hours,** so the experiments are affordable.

Fixed scope:
- one GPU for training;
- the real-scale ship set only;
- two teams;
- training and evaluation over many team sizes;
- variable map sizes, if they can be made to work;
- the playable game.

## 2. Non-goals

- **Bit-exactness.** No bitwise resume, common random numbers across runs, game
  fingerprints, or equality across devices or compile modes. Results reproduce within
  noise from a seed.
- **Backward compatibility** with v2's checkpoints, build records, game versions and
  observation columns.
- **Generality the scope does not ask for:** more than two teams, several ship sets, ONNX
  or other exports, CPU and GPU parity as a requirement.
- **Platform features for outside users:** model-exchange contracts, a leaderboard.
- **Options kept forever.** Alternatives live on branches and in experiments, not as
  switches in the main code.
- **Custom kernels before the simple version is profiled.**
- **Publication-grade statistics on routine decisions.** They are for the paper's claims.

## 3. Budgets

### Hard budgets

A change that breaks one is reverted or brought to the author.

| Budget | Value | Measured by |
|---|---|---|
| Play: game step plus both teams' decisions, one match, CPU, one core | p99 at most 4 ms, worst frame at most 8 ms, per 33 ms tick (*proposed*) | play benchmark |
| Play: stalls | nothing compiles, builds, synchronises or allocates at scale after a match starts | play benchmark |
| Operations per tick, uncompiled (compute operations, any batch size) | game step at most 1,500; policy forward at most 700; scripted opponent step at most 600 (*proposed*; v1: about 1,300, 649, 170) | scoreboard tier 1 |
| Cold compile of everything a training run compiles | at most 3 minutes on the laptop; warm at most 30 s (*proposed*) | scoreboard tier 2 |
| Recompiles | none when the batch size or the team size changes | scoreboard tier 2 |
| Fast test tier | at most 2 minutes on the laptop CPU (*proposed*) | scoreboard tier 1 |
| Smoke run | start to first update with metrics in at most 5 minutes on the laptop (*proposed*) | scoreboard tier 2 |
| Memory | the smoke configuration fits the laptop GPU (8 GB) | scoreboard tier 2 |

### Objectives

Improve these, and make every trade between them explicit:
- training steps per second at the reference configuration, on the laptop and on a 5090;
- environment steps needed to pass the learning canary;
- lines of code and the number of configuration options.

Baselines: v2's steady real-scale RL is about 5,000 steps per second on the laptop at
160-row micro-batches; v1's whole-run average was about 2,700.

### Minimum effects

A difference smaller than these does not decide anything:

| Metric | Minimum effect |
|---|---|
| Step or update time | 3% |
| Compile time | 10% |
| Rating | 50 Elo |
| Operations | any growth past a budget |

## 4. Strategy

- **A walking skeleton first,** on a new branch `core`. It contains:
  - the game step: flight, bullets, hits, deaths, respawn, one mode, any team size;
  - a policy with physical inputs, a Griffin trunk (spatial attention and RG-LRU in plain
    PyTorch), a policy head and a value head;
  - PPO;
  - a scripted opponent.

  The skeleton meets every hard budget and learns before anything is added.
- **v2 is the reference, not the base.** It is checked out beside the branch as a
  read-only worktree. Port behaviour (the flight physics and the mode rules are game
  content), not structure. The new game matches v2's dynamics statistically, not bit for
  bit.
- **Grow by earned additions,** one change at a time, each keeping the budgets. The likely
  order:
  1. fog;
  2. the frontline and elimination modes;
  3. the league with prioritised fictitious self-play and live rating from training games;
  4. Imugi, ported lean, and BC from it;
  5. the kept heads: next state, shot outcome from game-recorded labels, geometry;
  6. the play shell on the new core;
  7. evaluation;
  8. variable maps.
- **Framework checkpoint after the skeleton.** If the skeleton's cold compile misses its
  budget in PyTorch, run a measured JAX comparison before continuing.

## 5. Milestones

| Milestone | Content | Exit criteria |
|---|---|---|
| M0 | Scoreboard and baselines | The scoreboard runs in one command per tier; v1 and v2 baselines recorded in `docs/findings.md` |
| M1 | Walking skeleton | Every hard budget met; team size changes without recompiling; passes the learning canary (defined during M1) |
| M2 onward | One port each (section 4) | Budgets still met; each addition justified by measurement or by the scope |
| Done | | Budgets met; canary passed; playable at the real-time target; every open question resolved or explicitly deferred |

## 6. How the work is organised

- **One lead agent owns the outcome.** Ownership is by result, not by directory.
- **Helpers are short-lived and bounded:** a search, a read-only review, an independent
  review of a diff (given the code, not the lead's reasoning), a port of one piece. A
  helper's brief is a starting point, not a boundary. The goals in sections 1 to 3
  overrule any brief. When the right change is outside the brief, a helper makes it and
  records it if it is small and reversible; otherwise it stops and reports.
- **The GPU runs one job at a time,** behind a lock file. CPU work runs in parallel.
- **At every milestone:**
  - a fresh helper reviews the code against the assumption register (section 10), without
    the lead's notes;
  - a subtraction pass asks what can be deleted.
- **Past results are evidence under the stated conditions, not prohibitions.** Anything may
  be tried again by stating what differs.

### Authority

The lead decides and records anything reversible that stays within sections 1 to 3. It
asks the author before:
- changing a goal, non-goal or budget;
- changing a game rule in a way that changes how the game plays;
- spending more than 8 GPU hours on one experiment (*proposed*);
- dropping a v2 capability with no planned replacement.

Questions go to `docs/questions.md`, batched for the next milestone. Work continues on
other items meanwhile.

## 7. Measuring

**The scoreboard runs in tiers:**

| Tier | Cost | Contents |
|---|---|---|
| 1 | CPU, seconds | operations per tick and per section; traced operations per compiled graph; lines of code; fast-test-tier time |
| 2 | GPU, minutes, behind the lock | uncompiled and compiled step time at batch sizes 1 and 2,048; play p99 and worst frame; cold and warm compile time; learner step time; peak memory |
| 3 | GPU, at milestones | the learning canary |

**Benchmark protocol:**
- the performance power profile, with no other heavy process running;
- interleaved A/B against the baseline commit in the same session;
- at least 5 repeats, reporting the median and the interquartile range;
- the machine load recorded.

**Statistics, sized to the decision:**
- **Speed:** a difference well above the minimum effect and the spread needs no test.
  Close calls get more repeats.
- **Learning:** at least 3 seeds and a bootstrap confidence interval.
- One small helper function for these is enough. No experiment framework.

## 8. Records

**Living documents,** each at most a few pages. Adding past the limit means pruning first.
- `docs/goals.md`: sections 1 to 3 of this brief.
- `docs/decisions.md`
- `docs/assumptions.md`
- `docs/questions.md`
- `docs/findings.md`
- `docs/experiments/index.md`

`docs/log.md` is an append-only work log, not read by default. Keeping the living
documents current is the handover between sessions.

**A decision record:** what; because (one line); depends on (the known assumptions); date;
status (proposed, accepted, or superseded by another). The listed dependencies are the
known ones, not the only ones: any decision can be revisited by stating what changed.

**A result record:**
- the question;
- what was tested (commit, configuration, scale, seeds);
- the result with its spread;
- what it does not show, and the other explanations;
- a status: supported under these conditions, implementation failed, inconclusive, or idea
  refuted. Refuting an idea needs strong evidence. Most negative results are
  "implementation failed" or "inconclusive".

**Three kinds of record:**

| Kind | Example | Where it goes |
|---|---|---|
| A small comparison | Which collision test is faster | The commit message: numbers and conditions |
| An experiment | Informs a decision, costs GPU hours, or is a negative result likely to be cited | `docs/experiments/<date>-<name>/` with a one-page README in the result format, raw results and the command to rerun; one line in the index |
| A recurring check | The budgets | The scoreboard |

Scripts are disposable once their result is recorded, unless they join the scoreboard.

**v2's notes are not in the tree.** `docs/rewrite/v2-notes-index.md` (on the `simplify/audit` branch, which holds the `v2` notes)
maps topics to files. Read the notes for an area when porting that area, as evidence
under v2's conditions.

## 9. Writing code

- **Docstrings** state the contract: what the function does, units and shapes where they
  are not obvious, and constraints a caller must know. One line is enough when it is
  enough. No history, dates, first person or opinion.
- **Comments** only for a non-obvious reason, in the present tense.
- **A function that needs a story to explain it** is split or simplified, not explained.
- **Names** are full words. Conventional abbreviations are allowed: `cos`, `sin`, `exp`,
  `log`, `min`, `max`.
- **Working notes** in code are allowed while working and removed before merge. A review
  helper checks each diff for narrative and opinion.
- **Tests check behaviour:** game invariants, game dynamics against v2 statistically, and
  the budgets. No bitwise tests. Tests are deleted with the code they test.
- **Text that reports results** (findings, experiment records) states what was measured.
  Interpretation goes in a separate, labelled section.

## 10. Assumption register (seed)

| Assumption | Because | Depends on |
|---|---|---|
| PyTorch | Dynamic shapes; native GPU support on Windows for play | A lean graph compiles within budget (checked after M1) |
| Griffin trunk: spatial attention plus RG-LRU | Small in v2 (295 operations per forward); standard | Later alternatives tested as experiments |
| No belief system; memory in the recurrent trunk | Standard under partial observability | Fog as designed |
| Inputs in metres at fixed scales, never normalised by the arena | Local dynamics do not depend on map size | Variable maps |
| Plain PPO with one clipping rule | Standard | |
| One value head; advantages normalised per batch | Removes reward scaling, floors and bias tables | |
| Modes mixed in one batch, each mode a contiguous slice; per-environment values set at reset | The modes differ in a few parameters plus Frontline's rules | |
| Dynamic batch and ship dimensions | No recompile per size | Measured step-time cost within the minimum effect |
| No CUDA graphs in training until measured to pay | v1 measured them as no gain; v2's rollout is about 10% of an update | Profiles of the lean core |
| Shot labels recorded by the game (real bullets first) | Removes the re-simulation | |
| Returns' tail bootstrapped from the value; no lookahead | Standard PPO | |
| Two teams | Scope | |

## 11. Open questions (seed)

- The value head's form; the next-state head's form; whether to keep the enemy-action head.
- Reward and discount design. One discount for every level preceded a run learning to stop
  fighting in v2, under v2's reward design.
- Whether ratings from training games are precise enough for matchmaking; if not, a
  precomputed schedule.
- How much of Imugi to port. A reference version is frozen as the canary's yardstick.
- The skeleton's first mode, and the learning canary's definition.
- Game design for the author: the frontline capture lock, the elimination shrink.
- Packaging and distribution of the game.
