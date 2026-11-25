# The lean core: goals and standards

Replaces the 2026-11 brief; the details live in core-decisions.md.

Read this first. It holds the aims and the working rules. Everything specific (rules,
physics, model, recipe, numbers) is in `core-decisions.md`, and the plan is in
`core-backlog.md`.

## Goals

- **v2's mathematics in clean code.** The game rules, physics, model, PPO update and
  league from v2, with a short list of deliberate changes, and at about v1's cost: fewer
  operations, short compiles, loose coupling. Done means every v2 feature the author
  wants is back.
- **The project's three aims:** a game people enjoy playing against trained agents in
  real time on an ordinary computer; experiments that answer questions cleanly, possibly
  a paper; training that is cheap in GPU hours.
- **Training targets (author 2026-11-25):** BC reaches 0.40 against the scripted agent
  within 100M steps; RL then takes the averaged agent to 2000 Elo within 100M more. The
  scripted agent is whatever it is at the time and is the 1000 Elo anchor. Training is at
  4v4; the agent is evaluated, and must keep its skill, up to 32v32, with a 99% win rate
  there against the scripted agent. A run that misses a target at its cap stops, and the
  cause is investigated. A publication run follows on a rented GPU, so nothing is tuned
  to one GPU.
- **The averaged agent is the product.** The EMA and uniform weight averages are what
  ships and what is rated, in BC as in RL. The live policy is a training artefact.
- **Shipped so far:** v1 (`main`: run 731, the documentation and the white paper). The
  lean core is the successor and has not shipped.

## Scope

In:
- one GPU for training; two teams;
- elimination without shrink, 4v4, fields drawn per environment, fog;
- one ship (the P-51D) on the core flight model;
- the BFM scripted agent as yardstick and BC teacher;
- the policy (attention plus a recurrent block per token, categorical heads), BC, then
  PPO with half self-play and half league games;
- the evaluation runner and its result store, Elo, the crossover;
- the local web dev tool and sandbox.

Tabled (built so they stay easy to add):
- other modes (frontline, deathmatch), respawn, the shrinking boundary;
- unequal and larger team sizes, variable maps;
- more ship types and other aircraft; cheese agents in the league;
- scripted retreat or extend; the persona population (an experiment arm);
- the game client for players (an ONNX export spike first);
- bit-exact repeatability and backward compatibility with v2: not goals.

## Standards

**Code**
- Full-word names everywhere; only universal abbreviations (`cos`, `exp`, `max`).
- No legacy code, shims or dead options. A temporary compatibility layer is removed once
  native tests replace it. Old checkpoints need not load on new code.
- One source of truth: every tool (training, evaluation, play, sandbox, charts) calls the
  same game code through shared entry points.
- Smooth and repeatable where it is free; never at a real cost in throughput.
  Correctness is within rounding, not bitwise.
- Throughput first: steady speed, then compile time, which is a budget.
- New features are on by default; a setting turns them off for an ablation.
- Libraries are fine; tooling-only ones go in an optional extra.
- Game-feel work goes rough first: the author flies a playable version, then it is
  polished, tested and documented.
- Tests name the assumption or bug they guard; no goldens without a reason.

**Measurement**
- Measure in proportion to the decision: the quick tests on every merge; the compile
  tripwire and one rough speed run only for hot-path changes; a careful quiet-machine
  measurement only when a decision depends on the number.
- Never state a number that was not measured. Mark provisional numbers as provisional.
- Every score names the player (live, EMA or uniform average), the opponent and the
  version (v1, v2 or lean).
- Speed claims use the general path, not a load-tuned one, and compare like with like.
- Ratings use adaptive allocation to a target error, not a fixed number of games.

**Process**
- Small commits, merged and pushed to `core` as work goes. A run traces to one committed
  set of settings.
- One GPU job at a time. Long jobs run as systemd user units with memory caps, write each
  batch to disk, report an ETA as a clock time, and have a bound.
- A decision counts only when the author says so explicitly. A question from the author
  asks for an answer, not a change. Bugs are fixed without asking and reported.
- Before a long run: a proposal saying what it will produce, what stops it and which
  number will be reported; launched after agreement.
- Open questions go to `docs/questions.md`. Anything strange, missing or ambiguous is
  raised, not worked around.
- Each comparison builds on the winner of the last.

**Documentation**
- One short `NOTES.md` per package: what is, what is open. History goes in the worklog.
- Precision and understatement: no hype, no talk about the documentation itself.
- Decisions go into `core-decisions.md` the day they are made.

## Where things live

- `~/avesnova/boost-and-broadside/v2`: the checkout of branch `core`; the code is `lean/`.
  `main` holds v1; the `v2` branch is the read-only reference. `data/` holds run data.
- `lean/README.md`: packages and commands. `lean/CONTRACT.md`: the shared types. Each
  package's `NOTES.md`.
- `docs/rewrite/core-decisions.md`: every decision and deliberate change to v2.
- `docs/rewrite/core-backlog.md`: planned work and status. `core-worklog.md`: what was
  done, when, and how it was checked.
- `docs/questions.md` and `docs/rewrite/author-review-2.md`: what waits on the author.
- `docs/rewrite/scoreboard.md`: measured speed, memory, compile time.
- `docs/provenance.md`: where each component of the model and learner comes from.

## For the author to confirm

1. **Training targets.** Answered 2026-11-25 (above).
2. **Size generalisation.** Answered 2026-11-25: train at 4v4, keep skill up to 32v32 (99%
   win rate against the scripted agent).
3. **Speed floor.** The old brief kept v2's steady training speed as a floor; the core
   game measured below it (question 12, open). Is there a speed requirement?
4. **Parallel work.** "One GPU job at a time" against "agents may run in parallel except
   during careful measurements" (2026-11-24). Is the rule one GPU job, with CPU and
   analysis work in parallel?
5. **Rented GPUs.** The old brief had the author launch rented-GPU runs; the 2026-11-16
   rule lets me restart and choose the next job on my own judgement to keep a rented GPU
   busy. Which holds for lean?
6. **Tests per merge.** core-decisions says the quick suite on every merge; the older v2
   rule was the quick tier on the touched files only. For lean, the whole `lean` quick
   tier?
7. **Where open items live.** `docs/questions.md`, `author-review-2.md` and the morning
   notes all hold items for the author. One place?
8. **Records.** The old brief planned goals, lessons, assumptions, findings and an
   experiments index; only `docs/provenance.md` exists. Keep the rule that every model
   and learner component has a provenance row? Drop the rest?
9. **CLAUDE.md.** The old brief was to become the root `CLAUDE.md` on `core`. Should this
   one?
