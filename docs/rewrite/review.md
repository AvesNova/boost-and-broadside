# Review of the lean-core plan (`core-brief.md`)

Reviewer's notes, 2026-11-21. Written in the order the review was done: the first section
was written after reading only `core-brief.md` and the "Goals and non-goals" section of
`simplification.md`, before the review packet, the measurements or any code.

## Initial concerns

Written before reading the packet. Each item says what worries me and what I would want
to check.

### A. The case for a rewrite is not in the plan

1. **The plan never states what v2 fails at, in the plan's own terms.** The requirements
   table has no v2 column. I cannot tell from the brief which requirements v2 misses and
   by how much, so I cannot tell whether a rewrite is needed or whether two or three
   targeted fixes would do. *Verify:* v2's measured value against each requirement.
2. **The reference numbers undercut the headline proxy.** v2 does about 4x v1's
   operations per tick and 8x per forward, yet the brief reports v2 at about 5,000 steps
   per second against v1's 2,700. And the register says v2's rollout is about 10% of an
   update. If the rollout is a tenth of the time, the game step's and the policy
   forward's operation counts (the numbers the audit leads with) bound the training gain
   at about 10%. *Verify:* where a v2 update's wall time goes, and whether operation
   count predicts time on any phase that matters for GPU hours.
3. **No cost, no time box, no stop rule.** There is no estimate of how long M1 takes, no
   point at which the rewrite is abandoned in favour of trimming v2, and "Done" has no
   date or budget. A rewrite that is 80% done and missing the league, BC, evaluation and
   the play shell is worth less than v2.
4. **Knowledge loss.** v2's notes are off-tree and the living documents are capped at a
   few pages with prune-before-add. v2's 1,604 tests encode bugs that happened. The plan
   keeps none of them by default. *Verify:* whether there is a list of v2 defects and
   hard-won behaviours that the port must not reintroduce.

### B. Requirements that cannot fail

5. **Most requirements have no threshold.** "A small share of the machine", "uses the GPU
   efficiently", "minutes, not tens of minutes", "the largest supported size", "the
   laptop" (unspecified hardware). M1's exit is "every requirement met", judged by the
   lead agent that built it. The goals section had numbers the brief dropped or softened:
   p99 under half the frame budget, a smoke run under about 5 minutes, an A/B signal in
   hours, compile under 5% of a run, and an operation *budget enforced in tests* (now an
   instrument with no target). *Verify:* whether those were dropped on purpose.
6. **"Uncompiled within 3x of compiled" is a ratio.** It can be met by a slow compiled
   path, and it pulls against "let the compiler fuse". The need behind it (short jobs
   need no compile) is an absolute time.
7. **The roofline requirement may be the wrong instrument for this model.** For a small
   model at small batch the floor from FLOPs and bytes is microseconds and the measured
   time is launch and dispatch overhead; the gap will be enormous and uninformative.
   Building per-phase FLOP and byte accounting is also real work. No threshold is given.
8. **The ratchet has no teeth.** "Growth carries one line saying what it buys" permits
   unbounded growth one justified line at a time, which is presumably how v2 got here.
9. **The learning canary is defined during M1 by the agent that must pass it.** Same for
   "matches v2's dynamics statistically": which statistics, what tolerance, decided when?

### C. Aims 1 and 2 have no requirements at all

10. **Fun has no measure and comes late.** Every requirement is about performance. The
    play shell is step 6 of 8, so a human first plays the new core after the league, Imugi
    and the heads are ported. If the new physics port feels wrong, that is found late.
11. **The experiments are not named.** Aim 2 is "experiments that answer questions
    cleanly" but the plan does not list the questions. The register removes the belief
    system, multiple discounts, the re-simulation and CUDA graphs. If any planned
    experiment is *about* one of those, the lean core cannot run it. *Verify:* the list
    of arms against what the core removes.
12. **Evaluation is step 7, but every addition from step 1 must be "justified by
    measurement".** Measured with what? The 512-game head-to-head is a required workflow
    at M1, so some evaluation exists early; the plan does not say which part.

### D. Technical bets presented as settled

13. **Dynamic batch and ship dimensions with no recompile, at under 3% cost.** This is the
    hardest technical claim in the plan and it sits in the register as an assumption. In
    my experience dynamic shapes under `torch.compile` are where guards, recompiles and
    lost fusions come from. *Verify:* whether anything has been measured, and what the
    fallback is (padding to a few sizes, which the goals section allowed).
14. **The framework checkpoint is after M1, when it is most expensive to act on.** And
    JAX has no native Windows GPU support and handles dynamic shapes worse, so the
    comparison is close to decided in advance. A one-day spike before M1 (compile time of
    a Griffin block plus a toy step, both frameworks) would be cheaper and real.
15. **Packaging is an open question but constrains the framework.** A game "people play"
    shipped with PyTorch and CUDA is gigabytes. Integrated graphics is a stretch target
    while ONNX export and CPU/GPU parity are non-goals. Those pull in opposite
    directions.
16. **One value head and one discount, with the brief's own open question recording that
    one discount preceded a run that learned to stop fighting.** The register entry and
    the open question disagree about whether this is decided.
17. **"No belief system; memory in the recurrent trunk"** depends on "fog as designed".
    What result would reverse it, and is the reversal cheap once the inputs are fixed?
18. **Variable maps are last,** but the input design that makes them possible (metres,
    fixed scales) is a one-way door taken in M1. If it does not work, that is learned at
    the end.
19. **The scripted opponent in the skeleton versus Imugi at step 4.** The canary's
    yardstick is a frozen Imugi reference, but Imugi is ported after the skeleton must
    pass the canary. Which opponent does M1 train against?

### E. Process

20. **Authority is wide and the check on it is thin.** The lead decides anything
    "reversible", and the brief says most code changes are reversible because git can undo
    them. A thousand individually reversible choices are not reversible together.
    Questions are batched to the next milestone, and M1 is large. Helpers may go outside
    their brief.
21. **Learning regressions are caught only at milestones.** The canary is tier 3. With
    bit-exactness gone and pins discouraged, a subtle learning bug introduced mid-milestone
    is found weeks later and bisected at one canary run per step.
22. **The statistics are thin where they matter.** A bootstrap interval on 3 seeds is not
    an interval. 50 Elo as the minimum effect discards real gains that compound. *Verify:*
    v2's seed-to-seed spread in rating, to see what 3 seeds can resolve.
23. **Statistical tests plus "a flaky test is fixed or deleted the same day"** selects
    against exactly the tests the plan prefers.
24. **M0 measures things that do not exist yet.** A three-tier scoreboard with roofline
    accounting, built before the skeleton, risks being the first piece of machinery the
    plan says it does not want.

### What I intend to spot-check

Chosen before reading the packet; at most five.

- Where a v2 update's time goes (concern 2).
- The operation counts in the reference table, and whether they can be reproduced.
- The "295 operations" Griffin figure against the "5,172 per forward" figure.
- Whether dynamic shapes have been measured (concern 13).
- The v2 steps-per-second figures and their conditions.

## The review

Written after reading the packet, the rest of `simplification.md`, the measurement
records and five spot-checks (listed below). The packet's author-decided items (scope,
no bit-exactness, the dropped heads, no belief at first, metres) are taken as given except
where one conflicts with a stated goal.

**Summary.** The diagnosis is mostly right and the non-goals are the most valuable part
of the plan. The plan's weakness is that its hard requirements cover throughput, latency
and compile time, while the thing aim 3 is about (GPU hours to reach a given playing
strength) and the thing aim 1 is about (agents worth playing against) are not
requirements, have no yardstick shared with v2, and cannot be resolved by the statistics
the plan sets. Second, the evidence in the audit supports *deleting the decided features*
more strongly than it supports *rewriting*; the plan never separates the two, so the
rewrite will be credited with gains that deletion alone would give.

### Spot-checks

| # | Claim | Result |
|---|---|---|
| 1 | Operation counts: 27,019 per v2 rollout tick; 5,172 per forward; trunk 295 | Matches `records/v2_ppo_2048_1024.txt`. Small slips: Imugi is 4,290 in the record against 4,256 in the audit's table. Not material |
| 2 | "About 90% learner", learner GPU-bound | Holds, and is more specific than the packet says. `sections/compile-overhaul-notes.md:500`: 227 ms wall, 220 ms of kernels; Triton reductions 30%, pointwise 28%, GEMMs 19%, attention 4%; "the loss tails of the auxiliary heads' codes ... are a large part". See objection 2 |
| 3 | Seed noise about 110 Elo | It is the gap between *two* seeds of one configuration, from a small number of pairs, one of them confounded by micro-batch size, GPU and commit (`arms-todo.md:108-112`). An order of magnitude, not a measured spread |
| 4 | 6 microseconds per operation; play on the CPU | `sections/play.md:47`: 25-30 ms for the game tick at 10 ships "on the loaded CPU". One measurement, loaded machine, game only. See objection 5 |
| 5 | Static shapes "reach every interface"; test-suite size | `dynamic=False` appears at exactly three `torch.compile` call sites (`runtime/rollout.py:197`, `runtime/flights.py:14`, `runtime/train.py:370`). The flag is trivially changeable; what is pervasive is the compute-then-select coding style. 1,604 tests and 44,703 lines confirmed |

### Objections, most consequential first

#### 1. No requirement says the agents must learn as well per GPU hour, and nothing can compare the new agents with v2's

- **What is wrong.** Every hard requirement is about speed, latency, compile time or
  memory. Learning appears as an instrument with no target ("environment steps to pass
  the learning canary") and a canary that is undefined, set during M1 by the agent that
  must pass it, and run only at milestones. The new game matches v2 "statistically" and
  the observations change, so a v2 champion cannot play a lean-core agent. The only
  common opponent is Imugi, which is itself re-ported "lean" at step 4, after the skeleton
  must already have passed.
- **Consequence.** A core that is 3x faster per step and 3x slower to learn per step
  meets every requirement. The plan removes, at once, the belief, five heads, the value
  form, the reward scaling, the position code and the input codes. Each removal is
  defensible; together they are an untested change to sample efficiency with hard
  pressure on one side of the trade and none on the other. Coding agents optimise what is
  enforced. The packet itself lists this under "not checked": "the learning effects of
  the planned simplifications are unknown".
- **Evidence.** Brief section 3 (requirements against instruments), section 5 (canary
  "defined during M1"), section 11; packet section 5.
- **What would show I am wrong.** A written canary, fixed before M1 starts, of the form
  "reaches win rate W against opponent O within H laptop GPU hours", where O exists today
  and v2's own number for the same test is recorded in M0.
- **What I would change.** Make time-to-strength a requirement. Freeze the yardstick
  first: port Imugi (or a smaller scripted opponent) to the new game *before* the policy,
  and check the port by playing it against itself in both games and comparing outcome
  statistics. Record v2's time-to-beat-the-yardstick in M0. That is the only bridge
  between the versions, so it should not be step 4.

#### 2. The audit's evidence supports deletion; the case for a rewrite over deletion is inferred, and the baseline hides the difference

- **What is wrong.** The packet lists "a lean core is better than simplifying v2 in
  place" under *inferred, not measured*, and rejects the in-place route in one line
  ("the static-shape and code-based design reaches every interface"). But the audit's
  largest costs are things the author has already decided to delete, and deleting them
  needs no rewrite:
  - the belief and correction decode: 3,588 of 5,172 operations per forward, and the
    per-tick loop that is "more than half" of the learner graphs (L1);
  - the shot trajectory head and re-simulation: about 8 of 21 million eager operations
    per update, and the largest single learner kernel;
  - the rating lanes and lookahead: "the rollout plays 75% more than it trains on";
  - the auxiliary heads' loss tails: "a large part" of the 58% of learner kernel time in
    pointwise and reduction kernels (spot-check 2).

  The audit says of its own headline counts that they "explain compile time and eager
  speed, not [the learner's] steady speed", and that the model's arithmetic is within a
  quarter of v1's. So for aim 3, almost all of the identified gain is deletion.
- **Consequence.** M0 records v1 and v2 as baselines. Measured against v2 as it stands,
  the lean core will look like a large win whichever way the work is done. The plan can
  never learn whether the months of re-porting the league, Imugi, BC, the modes, fog,
  evaluation and the play shell bought anything beyond what a week of deletion would
  have. The brief's reference table also repeats the audit's root problem 5 in a new
  unit: all three reference columns are rollout-side counts, for a rollout that is 10% of
  an update.
- **Evidence.** `simplification.md` findings 1, 5, 6, L1, L2, "GPU hours", "Not yet
  examined" item 1; spot-checks 2 and 5.
- **What would show I am wrong.** A time-boxed deletion branch off v2 (remove the belief,
  the dropped heads, the lanes, the lookahead, the bit-exactness tests; set
  `dynamic=None` at the three sites) that still compiles slowly, is still far from real
  time uncompiled, or cannot be made to pass its remaining tests in a few days. That
  result would be the measured version of the packet's one-line rejection.
- **What I would change.** Run that branch first, two or three days, and use it as the
  M0 baseline in place of unmodified v2. If it falls short, the rewrite proceeds with a
  measured reason and an honest baseline. If it gets most of the way, the author has a
  real choice. Either way it also yields the dynamic-shape measurement the plan depends
  on (objection 4). I recognise the author prefers the lean core; this does not overrule
  that, it prices it.

#### 3. The statistics cannot resolve the plan's own minimum effect

- **What is wrong.** The plan sets a 50 Elo minimum effect, "at least 3 seeds and a
  bootstrap confidence interval", and v2's notes put two seeds of one configuration about
  110 Elo apart. Taking that at face value (per-seed standard deviation about 78 Elo),
  the difference of two 3-seed means has a standard error of about 64 Elo. Resolving 50
  Elo at two standard errors needs on the order of 20 seeds per arm. A bootstrap over
  three values is not an interval. v2's notes concluded that "pairwise arms must share
  seeds", but common random numbers are now a non-goal.
- **Consequence.** Every "earned addition" at steps 1 to 8 is to be "justified by
  measurement". With these numbers no learning comparison can pass or fail; additions
  will be admitted on noise or on argument, which is how the audit says v2's features got
  in (arms A13 and A14 "never ran"; the error head kept at +27 under a +40 bar).
- **Evidence.** Brief section 3 (minimum effects), section 7; `evaluation-suite.md:130`;
  spot-check 3.
- **What would show I am wrong.** A measured seed spread on the lean core (say 8 seeds of
  the skeleton) that is several times smaller than v2's. That is plausible, since a
  simpler system may be less seed-sensitive, and it is worth measuring at M1 because
  everything after depends on it.
- **What I would change.** Measure the skeleton's seed spread at M1 as an exit criterion
  and derive the seed count and minimum effect from it. Where it stays large, decide
  additions on cheaper, lower-variance measures (time to a fixed win rate against the
  frozen scripted opponent; head-to-head between checkpoints of the same run family) and
  say so, instead of naming an Elo bar that cannot be met.

#### 4. Dynamic shapes: the plan's enabling technical bet is unmeasured, the brief goes further than the audit, and a cheap test exists today

- **What is wrong.** "No recompile for sizes" is a hard requirement and "dynamic batch
  and ship dimensions" is in the register with the dependency "measured step-time cost
  within the minimum effect" (3%). Nothing has been measured. The audit's own reasoning
  is that batch-like dimensions should be cheap to make dynamic but that "small inner
  dimensions (ships, tokens, bins, heads) gain more from specialisation: keep them
  static, or round them to two or three fixed sizes". The goals section allows rounding.
  The brief dropped that hedge. A second conflict: the audit's fix for root problem 3 is
  "constants as numbers", which bakes them into the graph, while the brief requires that
  a restart after a settings change "recompiles nothing".
- **Consequence.** If a dynamic ship dimension costs more than 3%, or produces guard
  failures and silent recompiles (a stall during play, which is another hard
  requirement), M1 cannot exit as written and the design is re-opened after the skeleton
  is built around it.
- **Evidence.** `simplification.md` "Dynamic shapes" and "Constants as Python values";
  brief sections 3 and 10; packet section 5.
- **What would show I am wrong.** The benchmark the audit already lists as "not yet
  examined" item 3: static, dynamic batch, dynamic ships, on the laptop GPU, for step
  time, compile time and recompile count across four team sizes. It needs only a Griffin
  block and a toy step.
- **What I would change.** Run it before M1, not after. Write the requirement as "no cold
  compile per size" with padding to a few ship counts as an accepted way to meet it.
  State which settings may be baked constants and which must be changeable without a
  compile.

#### 5. The real-time requirement does not say where play runs, and the audit's own arithmetic leaves no margin

- **What is wrong.** The packet's bet is "real-time play on the CPU", inferred from "one
  decision at batch size 1 takes about a millisecond once the operation count is small".
  At the audit's 6 microseconds per operation, a millisecond is about 170 operations; v2's
  trunk alone is 295 and v1's whole forward 649. The audit's budget table puts v1-level
  counts (2,600 operations) at about 16 ms, which is the entire p99 allowance, on an
  average, from one measurement on a loaded CPU. So uncompiled CPU play needs counts well
  under v1's, or play is compiled, or play is on the GPU. The brief does not choose. It
  also requires "uncompiled within about 3x of compiled", measured at batch sizes 1 and
  2,048: at batch size 1 the compiled path avoids nearly all dispatch overhead, so 3x is
  unlikely to hold there and says nothing useful if it does.
- **Consequence.** The choice drives packaging (a CUDA build of PyTorch is gigabytes; the
  integrated-graphics stretch target has no path while ONNX and CPU/GPU parity are
  non-goals), drives whether the "no stall" requirement means shipping a compile cache,
  and is a one-way door by the brief's own definition. It is currently an open question
  ("packaging") and an inference.
- **Evidence.** `simplification.md` "Real time"; `sections/play.md:47`; brief section 3;
  packet section 4 "inferred".
- **What would show I am wrong.** A B=1 CPU timing of a Griffin trunk at the intended
  width with a v1-sized game step, uncompiled and compiled, p99 over a few thousand
  ticks. Half a day, no GPU needed.
- **What I would change.** Decide the play device and whether play is compiled before
  M1, from that timing. Replace the 3x ratio with the absolute times it stands for.

#### 6. The requirements cannot fail, and the builder judges them

- **What is wrong.** "A small share of the machine", "uses the GPU efficiently",
  "minutes, not tens of minutes", "the largest supported size" and "the laptop" have no
  numbers. The goals section had some (p99 under half the frame; smoke under about 5
  minutes; an A/B signal in hours; operation budgets "enforced in tests") and the brief
  dropped or softened each. The audit names "no budgets" as root problem 1 and prescribes
  "the budgets above, as tests"; the brief replaces them with a ratchet in which "any
  growth carries one line saying what it buys". M1's exit is "every requirement met".
- **Consequence.** Seven ports follow M1, and they are where v2's bulk came from (league,
  Imugi, heads, modes, fog). Each will add operations with a one-line reason and still
  "meet" requirements with no thresholds. That is the accumulation the audit describes,
  with better record-keeping.
- **Evidence.** Brief section 3 against `simplification.md` goals and root problem 1.
- **What would show I am wrong.** The author confirming numbers for the *proposed*
  values before M1, and a scoreboard that prints pass or fail against them.
- **What I would change.** Keep the ratchet, which is a good answer to budget-filling,
  and add two things: numeric thresholds on every requirement, fixed by the author before
  M1; and tripwires on the instruments (for example v1's counts per section) that are not
  targets but need the author's sign-off to cross.

#### 7. No time box, no stop rule, and no account of what stops while this runs

- **What is wrong.** The plan has no estimate for M1 or for "Done", no criterion for
  abandoning the core, and does not say what happens to the experiments in
  `arms-todo.md` meanwhile. All v2 results become "evidence under v2's conditions", so
  the paper's baselines restart too. The packet asks "how would this third rewrite fail?"
  but the plan does not say what was learned from the second: v2 was also a rewrite
  "meant to be faster", with a process of briefs, notes and agent reviews much like this
  one.
- **Consequence.** The likely failure is not a bad skeleton. It is a good skeleton and a
  long tail: the league, BC, evaluation and play shell re-ported at 60 to 80% while v2
  is frozen, with neither version able to run the experiments.
- **What would show I am wrong.** A dated budget per milestone and a stated fallback
  ("if M1 has not exited by X, return to the deletion branch").
- **What I would change.** Add both, and order the ports by what the first paper
  experiment needs, not by a general "likely order".

#### 8. Fun has no requirement and the first human play is step 6

- **What is wrong.** Aim 1 is listed first and has no measure. A human first plays the
  new core after fog, the modes, the league, Imugi and the heads. The flight model is
  "game content" matched "statistically", which will not catch a handling difference a
  player feels in seconds.
- **Consequence.** Feel problems in the re-ported physics, and agents that are strong
  but dull, are found late.
- **What I would change.** Put a keyboard-flown ship against the scripted opponent in M1
  (the brief already requires the play benchmark at M1, so the loop must exist), and name
  the statistics and tolerances for "matches v2's dynamics" before the port.

#### 9. Process: wide authority, late checks, capped memory

- **Authority.** "Reversible" covers nearly everything because "git can undo" it, and
  questions are batched to milestones. M1 is the largest milestone and contains the
  one-way doors (interfaces, data layout, input units). I would add an author checkpoint
  inside M1, after the interfaces are sketched and before the bulk is written.
- **The lead's independence.** The milestone review helper gets the code "without the
  lead's notes", which is right. But the lead writes the canary, the requirement
  thresholds and the helper's brief. The reviewer should be handed the goals and the
  scoreboard output, and asked first for what it would check, as this review was.
- **Learning regressions.** The canary runs only at milestones. A cheap version (minutes,
  one seed, a coarse threshold) at batch level would catch gross breakage without the
  bisecting cost.
- **Memory.** Living documents capped at a few pages with prune-before-add, v2's notes
  off-tree, and tests deleted freely will lose negative results. The audit's own process
  cause is "decisions recorded without the conditions they depended on"; pruning does not
  fix that. Keep the cap on what is *read by default* and let the experiment records and
  a list of known v2 defects grow.
- **Tests.** The rules are good. One conflict: statistical behaviour tests are preferred,
  and a flaky test is "fixed or deleted the same day". State the false-failure rate such
  tests are designed to (and fix seeds inside them), or the rule removes them.

### The specific questions

1. **Diagnosis.** Root problems 2 (representation) and 4 (requirements never pruned) are
   well supported; spot-check 2 adds direct GPU evidence for 2 that the packet does not
   cite (the learner's time is in the heads' code losses, not the trunk). Problem 3 is
   half right: the flag is three lines, the style is the cost. Problem 1 is right, and
   the brief then declines its own prescribed fix (objection 6). Problem 5 is right and
   the brief's reference table repeats it (objection 2). Missing: why v2's rewrite went
   this way, beyond process; and that learning quality was never a budget either.
2. **Strategy.** Objections 2 and 7. A skeleton-first build is the right shape *if* a
   rewrite is right. It fails by the long tail, and by losing the bridge to v2
   (objection 1).
3. **Requirements and instruments.** Outcome requirements over proxy caps is the right
   idea. What remains gameable: anything without a number (6); learning, which is
   unconstrained (1); ratios (5); and "meets requirements at M1" judged by the builder.
4. **Process.** Objection 9.
5. **Technical bets.**
   - *Dynamic batch and ships:* batch yes; ships unproven, test first (4).
   - *No belief, recurrent trunk:* a sound default. The +27 Elo result under 110 Elo of
     noise is no evidence either way, so say what result at the fog step would bring a
     belief back as an experiment.
   - *Plain PPO, one value head:* fine as a start, but "advantages normalised per batch"
     does not replace return scaling for the *value loss* across modes and team sizes
     with different return magnitudes; expect to need a standard return normaliser. The
     register treats one discount as settled while section 11 records that it preceded a
     policy that stopped fighting. One of the two entries should change.
   - *Shot labels from the game:* agree. With real bullets only, the head learns hit
     probability given the policy chose to fire; fine as an auxiliary loss, not as a
     firing estimate.
   - *Metres, never normalised by the arena:* agree. It is a one-way door whose payoff
     (variable maps) is tested last; a two-map smoke test at M1 would be cheap.
   - *Modes as contiguous slices:* agree; it is the direct fix for compute-then-select.
     Per-slice graphs need the dynamic batch dimension, so it rests on (4).
   - *Play on the CPU:* undecided in the brief and unsupported by the arithmetic (5).
6. **Missing entirely.** A learning-efficiency requirement and a cross-version yardstick
   (1); the stripped-v2 baseline (2); a schedule and stop rule (7); what the first paper
   experiment is and what it needs from the core; the fate of in-flight v2 experiments;
   a difficulty range for human play (agents trained to win are not automatically fun to
   play against); input latency for the human, which the frame-time requirement does not
   cover.

### What I agree with

- **The non-goals.** Dropping bit-exactness, backward compatibility, N-team generality
  and options-as-switches removes whole categories of machinery, and the audit shows what
  each cost (173 keyed-draw sites, 616 operations per tick of purpose-hashed draws).
- **The representation diagnosis,** now with GPU evidence behind it.
- **Outcome requirements, with proxies reported and not capped.** Right in principle; it
  needs numbers.
- **The workflow requirements** (restart after a fix, a 512-game evaluation, a smoke run).
  Compile time per restart is the cost people feel, and this states it correctly.
- **The result-record format,** especially "implementation failed" as distinct from
  "idea refuted", and "what it does not show".
- **The test rules** and "past results are evidence under stated conditions".
- **The packet's candour.** Section 5 lists most of what I would otherwise have had to
  find, including the two points my top objections rest on.

### What became of the initial concerns

- **Stood and sharpened:** 1-3 (objections 2 and 7), 5 and 8 (6), 6 (5), 9 and 19 (1),
  10 (8), 13 (4), 15 (5), 16, 20-23 (9, 3).
- **Partly withdrawn:** 7. The learner is GPU-bound, so a roofline gap is a sensible
  measure there. It remains uninformative for the rollout and for play, and still has no
  threshold.
- **Withdrawn:** 14, in part. The packet records a prior JAX physics race and flags the
  Windows claim as unverified; the checkpoint is less empty than I assumed. I would still
  move the compile-time comparison before M1, alongside the dynamic-shape test.
- **Not pursued:** 11 (experiments against what the core removes), 12, 17, 18, 24. They
  stand as questions; I did not check them.

## Addendum after the model audit (2026-11-21)

`model-audit.md` followed this review. It corrects three things here:

- **The "+27 Elo" for the error head** (specific question 5, "no belief") came from a
  tournament run on a wrong base and compares a bundle of changes, not the error head
  with its absence. Belief against no belief has never been measured.
- **Objection 3** treated 110 Elo as unpaired seed noise. The seed effect is common across
  arms (112, 115 and 110 Elo in three configurations), so same-seed comparisons are much
  tighter. The objection stands in a weaker form: two seeds, all clean pairs in BC, and
  the pairing needs a seed to fix the initialisation and data order.
- **The headline goal** (zero-shot play at 20v20 and 50v50 from training at 5v5) is not
  in the brief and was not in this review. It should be the first requirement and part
  of the canary; see the audit's last section.
