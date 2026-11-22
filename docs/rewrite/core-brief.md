# Operating brief: the lean core

Status: second draft, 2026-11-22, revised after the independent review (`review.md`), the
model and learner audit (`model-audit.md`) and the author's answers to both. On the `core`
branch this file becomes the root `CLAUDE.md`, the instructions every agent working on
the branch reads first. Values marked *proposed* are for the author to confirm; values
marked *pending* wait on a named measurement.

## 1. Purpose

The project has three aims, held together:

1. **A game people play and find fun.** Simple rules with a lot of depth. A human plays
   against trained agents in real time, on an ordinary modern computer.
2. **Experiments that answer questions cleanly,** possibly a paper.
3. **Training that is quick and cheap in GPU hours,** so the experiments are affordable.

**The claim the project is built around: zero-shot generalisation across sizes.** A model
trained at about 4v4 or 5v5 plays, with no fine-tuning and no change to its parameters,
at up to 50v50, with unbalanced teams and other zone counts. 50v50 is the goal; 20v20 is
the fallback if 50v50 cannot be reached. It comes in two stages:

- **Stage 1, the first result the core must reproduce: v1's fleet-size curve on a fixed
  map.** v1 (`origin/main`, run 731, trained 4v4) held a break-even of 1.75 scripted
  ships per network ship at the training size, 1.62 at 8, 1.56 at 16, 1.44 at 32 and
  1.19 at 64, staying above parity throughout. Stage 1 uses the mode closest to v1's:
  elimination with the shrinking map switched off.
- **Stage 2: other map sizes.** A model cannot be expected to handle a map size it never
  trained on, so variable maps appear in evaluation only once they are in training. The
  target then is a large fleet on a map scaled to the local density seen in training,
  which is the case the literature supports.

The rewrite is a port, not a redesign. The game's rules and the feel of the ships stay
the same within reason. The model keeps its core (entity attention with rotary position,
a recurrent block per entity, categorical outputs) and loses the parts the audit found
had outlived their reasons. New ideas are tested afterwards, on the cheap base this
produces.

Fixed scope:
- one GPU for training;
- the real-scale ship set only;
- two teams;
- training and evaluation over many team sizes, unequal teams included;
- variable map sizes, in stage 2;
- the playable game.

v2 is set aside while this runs. Its planned experiments are paused, not cancelled, and
the branch is kept as the reference.

## 2. Non-goals

- **Bit-exactness.** No bitwise resume, game fingerprints, or equality across devices or
  compile modes. **A seed fixes what it reasonably can:** the initial weights, the map
  pool, the scenario draws and the order of the data at the start of a run. It does not
  try to match what cannot be matched once two runs diverge. Comparisons between arms use
  the same seeds, because v2 measured the seed's effect as common to every arm.
- **Backward compatibility** with v2's checkpoints, build records, game versions and
  observation columns.
- **Generality the scope does not ask for:** more than two teams, several ship sets, CPU
  and GPU parity as a requirement. An export for play is a non-goal only if the play
  measurement (gate G2) shows it is not needed.
- **Platform features for outside users:** model-exchange contracts, a leaderboard.
- **Options kept forever.** Alternatives live on branches and in experiments, not as
  switches in the main code.
- **Custom kernels before the simple version is profiled.**
- **Publication-grade statistics on routine decisions.** They are for the paper's claims.

## 3. Requirements, tripwires and principles

Requirements are outcomes the project needs, each with a number and where the number
comes from. Tripwires are proxies: they are reported with every change and are not
targets, but crossing one stops the work and raises a question to the author. Neither is
something to fill up to.

### Requirements

A change that breaks one is reverted or brought to the author. "Laptop" is the author's
RTX 4070 Laptop (8 GB); "reference CPU" is that laptop's CPU with the GPU left free for
drawing.

**Play**

| Requirement | Number | Anchor | Status |
|---|---|---|---|
| A game tick plus both teams' decisions, batch size 1 | p99 at most 16 ms; worst frame at most 33 ms | The 30 Hz tick, with half of it spare | Fixed |
| Sizes this holds for | 1v1 to 50v50; 20v20 as the fallback | The claim | Fixed |
| Where play runs | The reference CPU if it meets the numbers above; a GPU, integrated or discrete, is allowed if it does not | "Any reasonably modern computer" | Pending G2 |
| No stall once a match starts | No compile, build, host synchronisation or large allocation during a match; measured as the worst frame over a full match | | Fixed |
| From launch to a playable match | At most 10 s; lower if G2 shows room (importing the framework alone takes seconds) | | Pending G2 |

**Workflows** (laptop, from command to result, compile included)

| Requirement | Number | Anchor | Status |
|---|---|---|---|
| Smoke run to its first update | At most 5 min | The goals section | Proposed |
| Restart after a settings change | At most 1 min | | Proposed |
| Restart after a code change | At most 5 min | | Proposed |
| A 512-game head-to-head evaluation | At most 10 min | | Proposed |
| Evaluating at a team size not yet run in this session | At most 1 min added | See below | Proposed |
| Fast test tier, CPU | At most 2 min | | Proposed |
| A learning comparison worth trusting | Hours, not days | The goals section | Proposed |

How a new size is made cheap is not prescribed. Dynamic dimensions, padding to a fixed
maximum with masks, and a compiler fast enough that a recompile does not matter all meet
it. Which one is used follows from gate G1.

**Training**

| Requirement | Number | Anchor | Status |
|---|---|---|---|
| Steady environment steps per second, laptop, real-scale RL | Not below v2's (about 5,000) | v2's record | Fixed as a floor |
| Memory | The smoke configuration fits 8 GB | The laptop | Fixed |
| Per phase: measured time against its roofline floor | Reported; no number | The learner is GPU-bound, so the gap is meaningful there | Reported |

**Learning.** These exist so that speed cannot be bought with slower learning. The
numbers are recorded from v1's and v2's logs in M0, before M1 starts, and confirmed by
the author.

| Requirement | Number | Anchor | Status |
|---|---|---|---|
| **Clone:** BC from the scripted opponent reaches a set score against it | Steps and laptop GPU hours no worse than v2's | v2's BC records | Set in M0 |
| **RL at the training size:** break-even against the scripted opponent | At least 1.5 scripted ships per network ship, within a GPU-hour budget | v1 1.75 at 4v4; v2 1.49 at 5v5 | Budget set in M0 |
| **Size curve:** v1's crossover protocol at 1, 2, 4, 8, 16, 32 and 64 a side on the training map | Above parity at every size | v1's result | Fixed |

### Tripwires

Reported on the scoreboard with every change. Crossing one needs the author's sign-off
and one line saying what it buys.

| Tripwire | Value | Anchor |
|---|---|---|
| Operations per physics tick of the game, uncompiled | 1,300 | v1 |
| Operations per policy forward | 650 | v1 |
| Operations per call of the scripted opponent | 170, or the simplified opponent's count once set | v1 |
| Traced operations across every graph a run compiles | 10,000 | The audit's estimate for a compile of a few minutes; revisit after G1 |
| Source lines | 33,000 | v1 |

Also reported, with no tripwire: FLOPs and bytes moved per training sample; environment
steps to pass each learning requirement; the number of configuration options; test lines
per source line.

Operation counts matter less if G1 chooses a framework in which the whole loop is
compiled; the tripwires are then restated in that framework's terms.

Each milestone's deletion pass works the numbers down.

### Principles

How speed is usually reached, each for a reason:
- **Fewer, larger operations.** Every operation pays a launch and a dispatch; small ones
  pay it for little work.
- **Batch everything that can be batched,** including every player's rows into one call
  per player.
- **No host synchronisation in a hot loop.** Each one idles the GPU.
- **Let the compiler fuse;** write code it can fuse (no per-component Python loops, no
  branches computed for every row and then selected).
- **Constant work happens once:** at construction or at a reset, not every tick or epoch.
- **bf16 matrix multiplies with fp32 accumulation.**
- **One house data layout:** batch first, contiguous, features last. A local deviation
  needs a measured gain at the top of the profile; consistency is worth more than a
  slice-by-slice optimum.
- **Profile before optimising,** and optimise only where a phase takes a large share of
  the time and sits far from its roofline floor.

### Provenance

Every component of the model and the learner has a row in `docs/provenance.md`: what it
does, the problem it solves, and where it comes from. The source is one of:

- **(a)** the standard choice;
- **(b)** published work shows it beats the standard for a comparable purpose;
- **(c)** measured in this project;
- **(d)** a reasoned choice not yet tested, with the test that would settle it.

A citation counts only when a reviewer has fetched the paper and read the passage; each
carries a label saying so. From memory is not provenance. A paper's result at a fixed
team size is not evidence that something scales, so each row also says how the project's
setting differs. A component with no row does not go into the main code. Rows of kind (d)
carry their reasoning and are the first places to look when something goes wrong; how
they are tested, if at all, follows "Deciding without a training run" in section 4.

### Minimum effects

A difference smaller than these does not decide anything:

| Metric | Minimum effect |
|---|---|
| Step or update time | 3% |
| Compile time | 10% |
| Rating, same-seed pairs | Set at M1 from the measured spread of paired differences |

## 4. Strategy

### Gates before M1

| Gate | Question | How it is settled |
|---|---|---|
| G1 | PyTorch or JAX | Three head-to-heads on the laptop GPU (`sections/framework-spike/`): a policy forward plus a game step, one PPO update, and the same model run at several team sizes. Measured: compile time, steady speed, and what a new size costs |
| G2 | Where play runs | Batch-size-1 latency of a tick plus two decisions on a CPU, in both frameworks, from the same benchmark |
| G3 | The yardstick's numbers | v1's and v2's clone, RL and size-curve results collected into `docs/findings.md`, and the learning requirements' numbers set from them |

### The skeleton

On a new branch `core`. It is built in this order, and each step works before the next
starts:

1. **The scripted opponent and the game step.** The opponent comes first because it is
   the only yardstick v1, v2 and the core share. The game step is flight, bullets, hits
   and deaths, in elimination on one fixed map with no shrink, at any team size.
   Respawn arrives with the other modes. The flight
   physics is lifted from v2 as nearly verbatim as the framework allows and checked by
   running both from the same states and actions and comparing short trajectories within
   a tolerance. The author flies a ship against the scripted opponent at this step.
2. **The observation contract and the model.** Rows of variable count with `exists`,
   `team`, `visible` and `alive`, padded to a maximum and masked. The model is the
   audit's baseline (below).
3. **BC from the scripted opponent.** Passes the clone requirement.
4. **PPO, with team sizes drawn per environment,** unequal teams included, inside one
   tensor shape, and with league self-play in its simplest form: opponents drawn from a
   pool of past snapshots and the scripted opponent. v1 trained with league slots, and
   self-play against past selves is standard, so it belongs here. Passes the RL and
   size-curve requirements. This is stage 1 of the claim.

### The model and learner at M1

From `model-audit.md`, which gives the reasons and the provenance of each line.

- **Inputs:** about 20 state scalars per ship at fixed physical scales, flags and ship
  type. Position in the token as Fourier features on three axes 60 degrees apart (a
  grid-cell layout), at about six scales sized for the maps in use, with the random
  per-match offset kept. Fourier features are the published way to give a network
  coordinates; the hexagonal layout is the same thing with even resolution in every
  direction; and the token code and rotary share one set of wave vectors. Clipped
  distance and direction to the boundary.
- **Attention:** standard attention with a key mask for anything that does not exist,
  and one learned null key. Rotary relative position on queries and keys, with fewer
  bands than v2, sized for the maps in use; rotary on heading kept.
- **Memory:** a Griffin recurrent block per token with the paper's decay initialisation.
  A hidden enemy that has been seen stays as a last-seen token with the time since;
  one never seen is masked. No belief, nothing decoded and fed back.
- **Heads:** a joint categorical action per ship; a per-ship categorical critic per
  reward level with Gaussian-binned targets. One auxiliary head: next state, as an
  open-loop change from the current state. v1 and v2 both trained with one, so it stays
  by precedent. The shot-outcome and geometry heads are left out until something shows a
  need for them.
- **Learner:** PPO with GAE; reward levels with a discount each; every loss a mean in
  nats per softmax group with three coefficients; an entropy term; a decaying
  distillation term toward the cloned policy at the start of RL.

**v2 is the reference, not the base.** It is checked out beside the branch as a read-only
worktree. Lift what is good (the flight physics, the trunk, the mode rules); port
behaviour, not structure.

### Additions after M1

One change at a time, each keeping the requirements, each with a provenance row. The
order follows what stage 2 and the game need:

1. fog;
2. the frontline and deathmatch modes, with respawn, the elimination shrink, and zone
   counts drawn in training;
3. the league's refinements: prioritised fictitious self-play weighting and rating;
4. variable maps in training, then the stage 2 evaluation;
5. the play shell on the new core;
6. the evaluation suite.

### Deciding without a training run

Full training comparisons are expensive and there is no budget for one per component.
Each choice is settled at the cheapest level that can settle it:

1. **Published work or theory.** If a source read in full supports the choice for a
   comparable purpose, or the argument is clear, it is decided and recorded with its
   provenance. Most choices stop here.
2. **A test that needs no training run:** an evaluation of an existing model with one
   thing changed, a probe of what the trunk has learned, a gradient-share measurement, a
   benchmark.
3. **A short BC run,** same seeds, when the question is about what the model can
   represent.
4. **A full training comparison,** only when the choice matters to the claim, the levels
   above cannot settle it, and the author agrees to the cost.

A choice of kind (d) does not owe a test. It owes a stated reason, and it is revisited if
a problem points at it. The stage 1 run is the one full training run the plan commits to.

### Time and fallback

The author sets a time box for M1 before it starts. If M1 has not exited by then, the
work stops and the author decides between extending it and returning to v2 with the
audit's deletions.

## 5. Milestones

| Milestone | Content | Exit criteria |
|---|---|---|
| M0 | Scoreboard, baselines, gates | The scoreboard runs in one command per tier; gates G1 to G3 closed; every *proposed* and *pending* value in section 3 confirmed by the author |
| M1a | Scripted opponent and game step | Trajectories match v2 within the stated tolerance; the author has flown it; play requirements met for the game step |
| M1b | Contract, model, BC | The clone requirement met; play and workflow requirements met; the author has reviewed the interfaces before the bulk is written |
| M1c | PPO with sizes drawn | The RL and size-curve requirements met; every requirement met; the seed spread of paired differences measured |
| M2 onward | One addition each (section 4) | Requirements still met; a provenance row; justified by measurement or by the scope |
| Done | | Requirements met; stage 1 reproduced; stage 2 attempted and reported; playable at the real-time target; every open question resolved or explicitly deferred |

## 6. How the work is organised

- **One lead agent owns the outcome.** Ownership is by result, not by directory.
- **Helpers are short-lived and bounded:** a search, a read-only review, an independent
  review of a diff (given the code, not the lead's reasoning), a port of one piece. A
  helper's brief is a starting point, not a boundary. The goals in sections 1 to 3
  overrule any brief. When the right change is outside the brief, a helper makes it and
  records it if it is small and reversible; otherwise it stops and reports.
- **The GPU runs one job at a time,** behind a lock file. CPU work runs in parallel.
- **At every milestone:**
  - a fresh helper reviews the code against the goals, the provenance table and the
    scoreboard's output, without the lead's notes. It first writes down what it would
    check, then checks it;
  - a subtraction pass asks what can be deleted.
- **Past results are evidence under the stated conditions, not prohibitions.** Anything may
  be tried again by stating what differs.

### Authority

The lead decides and records anything reversible that stays within sections 1 to 3. It
asks the author before:
- changing a goal, non-goal or requirement;
- changing a game rule in a way that changes how the game plays;
- dropping a v2 capability with no planned replacement.

It also asks before crossing a tripwire, and before fixing a core interface, the house
data layout or the observation contract, which are settled with the author inside M1b
before the bulk of the code is written.

Questions go to `docs/questions.md`, batched for the next milestone. Work continues on
other items meanwhile.

**Runs and time.**
- Runs on rented GPUs are proposed by the agent and launched by the author.
- Laptop work is time-boxed by the author's deadline. At the deadline the agent stops,
  leaves the tree committed and clean, and reports what is done, what is not, and what it
  would do next. An unfinished, accurate report is preferred to rushed final checks.
- Development time is a resource like GPU time. Spend rigour where a mistake is expensive
  or hard to undo, and move quickly everywhere else.

## 7. Measuring

**The scoreboard runs in tiers:**

| Tier | Cost | Contents |
|---|---|---|
| 1 | CPU, seconds | operations per tick and per section; traced operations per compiled graph; FLOPs and bytes per training sample; lines of code; fast-test-tier time |
| 2 | GPU, minutes, behind the lock | uncompiled and compiled step time at batch sizes 1 and 2,048; play frame-time distribution; the short workflows' times; learner step time and its roofline gap per phase; peak memory |
| 3 | GPU, at milestones | the learning requirements: clone, RL at the training size, the size curve |

**Benchmark protocol:**
- the performance power profile, with no other heavy process running;
- interleaved A/B against the baseline commit in the same session;
- at least 5 repeats, reporting the median and the interquartile range;
- the machine load recorded.

**Statistics, sized to the decision:**
- **Speed:** a difference well above the minimum effect and the spread needs no test.
  Close calls get more repeats.
- **Learning:** arms are compared on the same seeds, as paired differences. The number of
  seeds follows from the paired spread measured at M1c. BC comparisons are preferred
  where they answer the question, since they are cheaper and v2's clean pairs were there.
- One small helper function for these is enough. No experiment framework.

**Verification in proportion to the stakes:**

| Level | When | What runs |
|---|---|---|
| Inner loop | Every change, seconds to a minute | The tests for what was touched, and the instruments for that area. Most changes stop here |
| Batch | Before merging a set of changes | The fast test tier, the benchmarks for the phases the changes affect, and a short learning check (minutes, one seed, a coarse threshold) that catches gross breakage. Many small commits share one run; a regression is found by bisecting |
| Milestone | At a milestone | The full test suite, the learning requirements and the full scoreboard |

Decisions follow the same split. A **two-way door** (easy to undo; most code changes, since
git can undo them) is decided quickly and recorded. A **one-way door** gets rigour: a core
interface, the house data layout, a game rule, a paper's claim. A standard choice needs no
A/B test. "Good enough, revisit later" is a valid outcome when it is recorded in
`docs/assumptions.md`.

## 8. Records

**Living documents,** each at most a few pages. Adding past the limit means pruning first.
The limit is on what is read by default. Experiment records, the provenance table and the
list of lessons are not pruned.
- `docs/goals.md`: sections 1 to 3 of this brief.
- `docs/provenance.md`: one row per component (section 3).
- `docs/lessons.md`: failures that happened in v1 and v2 and what prevents each, seeded
  from `model-audit.md`.
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
| A recurring check | The requirements and instruments | The scoreboard |

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
- **Text that reports results** (findings, experiment records) states what was measured.
  Interpretation goes in a separate, labelled section.

### Tests

A test earns its place like any other code. Its value is how likely it is to fail when
something is wrong, times what that problem would cost undetected, times how much earlier
it is caught than by anything else. Its cost is writing it, running it on every run, and
maintaining it whenever the behaviour it pins changes on purpose. v2's suite had 1,604 tests
in 44,700 lines, about 0.7 lines of test per line of source, and 13 MB of fixtures.

- **Every test names the failure it catches,** in its name or a one-line docstring: the
  defect, not the implementation.
- **Prefer, roughly in this order:**
  - properties and invariants (a team swap mirrors the outcome; ships stay inside the
    arena; nothing becomes non-finite);
  - behaviour checked statistically (the scripted agent beats random; game lengths in a
    plausible range; a short run reduces its loss);
  - contracts at real boundaries (game to observation, model to actions, saved files);
  - regression tests for bugs that occurred, while the code path exists.
- **A reference implementation** only when it is much simpler than the code it checks.
- **Pinned outputs** (goldens, exact snapshots, exact equality) only where exactness is the
  specification: integer logic, file formats, replays. During a refactor meant to preserve
  behaviour, a pin may be used temporarily and is deleted afterwards.
- **A test that fails on intended changes and is fixed by updating its expected value has
  caught nothing.** When that happens twice, delete the test or rewrite it to check a
  property.
- **Do not duplicate other signals.** The scoreboard covers speed, the learning
  requirements cover training, and runtime checks cover non-finite values.
- **A statistical test fixes its seeds** and is designed to a stated false-failure rate,
  so that it is not deleted as flaky for being statistical.
- **Fixtures are small and generated** by code in the repository, not stored data.
- **A flaky test** is fixed or deleted the same day.
- **Tests are deleted with the code they test,** and the milestone deletion pass includes
  them. The test tiers follow the verification levels in section 7: the fast tier in
  minutes on the CPU, slower and GPU tests at batch and milestone level.

## 10. Assumption register (seed)

Each row moves to `docs/provenance.md` or `docs/assumptions.md` with its source.

| Assumption | Because | Depends on |
|---|---|---|
| The framework | Decided at gate G1 | Compile time, steady speed, and the cost of a new team size, measured |
| Play on the CPU | Reaches ordinary computers; frees the GPU for drawing | Gate G2 |
| Griffin trunk: spatial attention plus RG-LRU, the paper's initialisation | Small in v2 (295 operations per forward); published | Later alternatives tested as experiments |
| No belief system; hidden enemies as last-seen tokens; memory in the recurrent trunk | OpenAI Five reuses the last observation for fogged units; belief against no belief was never measured | A probe of what the trunk remembers about hidden enemies |
| State inputs as scalars at fixed scales | Standard; no source found for binned inputs | |
| Position: Fourier features on a grid-cell layout in the token, with a random offset, plus rotary in attention | Rotary selects which tokens to read; the code carries the numbers to calculate with. Rotary is measured to matter in v2 | Rotating the values as an alternative |
| Attention masks from the start of training, with a null key | Standard masking; a trained model depends on the make-up of its key set | Training and evaluation key sets kept alike |
| Team sizes drawn in training, inside one tensor shape | The best-supported lever for size generalisation | The size curve |
| Variable maps in evaluation only when in training | Rotary and position codes do not extrapolate to unseen separations | Stage 2 |
| PPO with one clipping rule | Standard | |
| Reward levels with a discount each | One confounded run in favour; a published precedent | The single-value experiment |
| Categorical critic with Gaussian-binned targets | Published for actor-critics; not shown with PPO | |
| Losses as mean nats per group, three coefficients, set once from a measured gradient share | The rule is measured; the old numbers are stale | Re-measured on the new model |
| Entropy term, and a decaying distillation term toward the clone | Every BC-then-RL source read keeps both | |
| Modes mixed in one batch, each a contiguous slice; per-environment values set at reset | The modes differ in a few parameters plus Frontline's rules | |
| No CUDA graphs in training until measured to pay | v1 measured them as no gain; v2's rollout is about 10% of an update | Profiles of the lean core |
| Shot labels recorded by the game (real bullets first) | Removes the re-simulation | |
| Returns' tail bootstrapped from the value; no lookahead | Standard PPO | |
| Two teams | Scope | |

## 11. Open questions

- **The value form.** Reward levels and per-level discounts stay for now. One discount for
  every level preceded a run learning to stop fighting in v2, with a long discount and no
  entropy term, so the cause was never isolated. The single-value run is the first
  learner experiment.
- **The advantage normaliser.** DreamerV3's percentile range was a starting point, not a
  measured choice; the alternative is the standard deviation over the update.
- **How loss weights are chosen.**
- **Position codes.** The default is Fourier features on a grid-cell layout, chosen on
  the literature and not by a comparison. v2's colour code is dropped as bespoke. Whether
  rotating the values in attention can replace the code in the token stays open.
- **Nearest-k attention** as an arm for large fleets.
- **The scripted opponent:** how far to simplify it, and whether to broaden it so the
  clone sees more varied behaviour, closer to what human play would give.
- **Why v2's frontline falls with fleet size on the training map.** Four evaluation-only
  tests are listed in `model-audit.md`; they run on a v2 checkpoint when a GPU is free,
  or on the core's first trained model.
- **The league:** whether ratings from training games are precise enough for
  matchmaking; whether one grouped matrix multiply serves several opponents' weights.
- **The next-state head's form; whether to keep the enemy-action head.**
- **Game design for the author:** the frontline capture lock, the elimination shrink.
- **The game side has not had the audit the model has.** A lighter pass (what exists
  only because of something now removed) happens during the port.
- **Packaging and distribution of the game.**
