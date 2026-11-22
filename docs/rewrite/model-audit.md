# Model and learner audit

2026-11-21. An audit of what v2's model ("Yemong") and learner contain, why each part is
there, whether the reason still holds, and what the literature supports, with the aim of
a baseline in which every component has a stated provenance. It follows from the review
in `review.md` and the discussion after it.

The headline goal the audit judges against: **train at about 5v5 and, with no
fine-tuning, play well at 20v20 and 50v50, with unbalanced teams, other zone counts and
other map sizes.**

## How it was made, and how far to trust it

Six passes by separate agents, then a reconciliation by the reviewer. Their full outputs
are in `sections/model-audit/`.

| Pass | What it had | File |
|---|---|---|
| Inventory | Code only, no design notes. Built the champion model on the CPU to check widths and counts | `inventory.md` (135 component rows, 48 size-dependence rows) |
| Blind judgment, model | The inventory and the goals; not the notes | `blind-model.md` |
| Blind judgment, learner | Same | `blind-learner.md` |
| History, model | The notes, decisions and logs | `history-model.md` |
| History, learner | Same | `history-learner.md` |
| Literature on size generalisation | arXiv; 28 papers read in the relevant sections, 19 abstract only, 6 from memory | `literature-size.md` |
| Citation check | The sources the proposed baseline rests on | `citations.md` |

**Limits.**
- The clone is shallow (history starts 2026-11-14), so dates come from the notes.
- "v1" means two things in the notes. `origin/main` (run 731) has scalar value heads and
  no belief, codes or rotary. v2 was ported from a later working branch that already had
  them, and that branch is not in this clone. Why that branch added the belief, the
  categorical codes, rotary, the joint action and the density head is not recoverable
  here.
- The blind passes saw a few designer comments while reading code; each file says which.
- Statements below are the agents' findings unless marked **(checked)**, which means the
  reviewer read the source line or the code. Row numbers such as 3.4 refer to
  `inventory.md`.
- Nothing was run on a GPU. Every recommendation that changes learning is a proposal to
  test, not a result.

## Main findings

1. **The zero-shot claim is not established, and the literature does not hand it over.**
   No paper found shows a centralised model attending over all entities extrapolating 4x
   to 10x in count. v2's own measurements show partial transfer: deathmatch at even
   numbers holds to 40 a side, the margin against a larger scripted fleet collapses at 40,
   frontline falls steadily with size, and larger maps hurt more than larger fleets.
2. **Training has never shown the model any variation in size.** Every batch is 5v5 with
   five zone slots and two arena radii. Size, field-count and map-size sampling exist in
   the code and are off. Unequal teams are never trained. This is the best-supported
   single change for the headline goal.
3. **v1's fleet-size result is the best on record, but v2's deathmatch result is closer
   to it than first stated here.** `origin/main` (trained 4v4, dead keys masked, no
   belief, no rotary) held a break-even of 1.75 scripted ships per network ship at 4 a
   side, 1.56 at 16, 1.44 at 32 and 1.19 at 64. v1's world was one fixed torus and its
   game was deathmatch-like **(checked)**. Read off v2's fixed-odds table for deathmatch
   on its fixed arena, v2's break-even is 1.49 at 5 a side, between 1.5 and 2 at 10 and
   20, and about 1.2 at 40. So at four and eight times the training size the two are of
   the same order; v2 has no measurement at sixteen times. The runs are not comparable
   (different game, opponent and map), and v1's scripted agent may simply have been
   weaker at scale. What v2 adds that v1 never tested is where v2 fails: frontline, and
   maps larger than training's. Nothing in the record shows v2's additions improving
   size transfer.
4. **The owner's question about balancing machinery has a different answer from the one
   expected.** The scalar-era machinery is already gone. Today's loss weights were set
   after everything was categorical, by measured gradient shares, and most were never
   re-measured after later changes. What is stale is the numbers and one normaliser, not
   a layer of leftover machinery.
5. **Large parts of the input encoding have expired reasons.** The 236 km position code,
   the 340 km rotary ladder and the 8x arena code were sized for 62 km maps that are no
   longer played. The random position offset served v1's torus. The blur half of the
   state code presents a belief's uncertainty, and the belief is going.
6. **Removing the belief leaves an unanswered question the plan treats as settled:**
   where a hidden enemy's memory lives. The recurrent block is about to be the only
   memory, and its decay is initialised for about a quarter of a second.
7. **The evidence quoted for and against the error head was misread,** including in
   `review.md`. Belief against no belief has never been tested.
8. **Several defects were found on the way** (listed below). None explains a headline
   result, but two would bite as soon as size sampling is turned on.

## Size generalisation

### What has been measured

**`origin/main` (run 731, trained 4v4).** Scripted ships per network ship at break-even
**(checked** against `origin/main:docs/evaluation.md`**)**: 1.75x at the training size,
falling to 1.19x at 64 a side. That document's own caution: "It does not by itself
establish a universal scale-invariance law."

**v2 champion (trained 5v5, fixed maps), 2026-11-10.** Deathmatch on the training arena,
win rate against the scripted agent, 128 games a cell **(checked,**
`sections/gpu-box-log.md:902-910`**)**:

| Ships a side (network) | Even | Scripted 1.2x | Scripted 1.5x | Scripted 2x |
|---|---|---|---|---|
| 10 | 1.00 | 0.95 | 0.73 | 0.43 |
| 20 | 0.97 | 0.91 | 0.69 | 0.22 |
| 40 | 0.97 | 0.53 | 0.03 | 0.00 |

Frontline rating on the training arena at 5 / 10 / 20 / 40 a side: 1997 / 1875 / 1134 /
526, with the scripted agent at 1000. At 40 a side the champion is below both the
scripted agent and a per-ship nearest-neighbour MLP.

**Map size.** The same 10v10 fleet on a larger, density-scaled map lost about 375 Elo in
frontline and 460 in deathmatch against the training arena. A later agent (rl-v20) won
26 of 64 frontline games at 20v20 on the training map and none of 64 on a 15.6 km map.

**Small and unequal teams.** rl-v20 is only even with the scripted agent at 2v2. Unequal
teams were evaluated through benched slots, never trained.

**What was done about it.** Each time a condition failed it was removed from the
evaluation: density maps dropped, fixed maps made the rule, frontline break-even left out
of the score. A BC run with map-size sampling was started and stopped. No agent has
trained on variable maps or sizes.

### What the literature says

From `literature-size.md`; the numbers are that agent's reading and items 2 and 3 are
re-checked in `citations.md`.

- Every large zero-shot extrapolation in count found (5x and up) is for decentralised,
  cooperative agents that each see a bounded local neighbourhood.
- One study isolates the mechanism: a self-attention policy trained on small counts
  degrades with the full observation at larger counts, and restricting each agent to its
  k nearest targets restores performance to very large counts, **with the map scaled to
  hold density constant** (Hsu et al., arXiv 2011.08055).
- Softmax attention disperses as keys are added; a trained head's accuracy falls as the
  item count grows past training (Veličković et al., arXiv 2410.01104).
- A bounded neighbourhood is not sufficient by itself: one swarm result degraded badly
  zero-shot at 4x and 16x and recovered with a short fine-tune.
- Training over a range of sizes flattens the size curve. Training directly on a very
  wide range did worse than a narrow range plus masking.
- The large precedents did not extrapolate: OpenAI Five pads to a fixed number of unit
  slots, AlphaStar caps and truncates.
- Nothing found covers an adaptive opponent at scale, locality masks inside a multi-block
  centralised attention stack, or categorical value heads under a size shift.

**Reading the two together.** The supported condition is constant local density: 50v50 on
a map sized so that each ship's neighbourhood looks like training's. v2 was tested the
two unsupported ways: more ships on the training arena (crowding, where frontline fails),
and a larger map with absolute-position inputs that training never exercised (where
everything fails). The result is consistent with the literature and does not yet say
which mechanism is responsible.

### Suspects, in the order I would test them

All but the last two are evaluation-only tests on an existing 5v5 model.

| # | Suspect | Rows | Test |
|---|---|---|---|
| 1 | **Map size out of the trained range.** Two radii seen in training; a 25-bin arena code trained at two points; coarse rotary bands that barely turn; absolute position codes. The old rotary ladder aliased exactly at 8,192 m and nobody connected that to the large-map collapse | 1.19, 1.26, 1.28, 1.29, 12.33-12.38 | 5v5 on 0.5x, 2x and 4x arenas, once with the true radius input and once with it held at the training value. Separates "input out of range" from "geometry out of range" |
| 2 | **Unmasked keys that are not there.** Dead ships, never-seen enemies and absent zones are ordinary keys, many stacked at the map centre on the global token | 3.4, 2.5, 12.6, 12.29 | Log attention mass on such keys at 5, 20 and 40 a side. Do not test by masking at evaluation only (see the lesson below) |
| 3 | **Softmax dilution.** 26 keys at 5v5, 56 at 20v20, 116 at 50v50, in one dense softmax with no locality | 12.5 | At 20 and 40 a side, restrict each query to its nearest k ships and compare with dense; log attention entropy against count |
| 4 | **Inputs that scale with fleet size.** The deathmatch score target (25 in training, 250 at 50v50); local presence as an unnormalised count | 1.16, 12.8, 12.9 | Hold each at its training-range value at 40 a side and compare |
| 5 | **Unequal teams.** Own and enemy split by row position; the benched flag always 0 in training | 2.8, 12.2, 12.3 | 5v3 and 3v5 through benched slots, mask on and off |
| 6 | **Crowding itself.** 40 a side on the 5v5 arena is eight times the training density; tactics may simply differ | | The same fleet on a map scaled to training density, after suspect 1 is fixed |
| 7 | **Memory of hidden enemies** once the belief is gone | 3.8, 3.9 | On the rebuilt model: a linear probe from trunk states to hidden enemies' true positions against time since seen, at 5v5 and 20v20 |

Suspects 1 to 5 are cheap and would turn the existing collapse from an observation into a
diagnosis. I would run them on v2 before the port, since they decide what the lean
model's inputs and attention look like.

## The balancing question

"We added machinery to balance rewards and losses when value and next state were scalar.
Now everything is in nats. Is some of it no longer needed?"

**What the record shows** (`history-learner.md` section 1):

- The scalar-era machinery lived on `origin/main`: per-component return scalers in symlog
  space, a Huber value loss, per-component advantage normalisation, two floors, tier
  scales. It was all removed when the later working branch went categorical.
- The current weights were set afterwards, on the stated rule "every term in nats per
  token ... weights are then set against measured trunk-gradient shares, not by hand"
  (`decisions.md:1199-1227`). They are exchange rates between heads with very different
  nats per token.
- Where the numbers came from: policy 27 for a trunk-gradient share of about 0.36 to
  0.40; value 0.007 kept deliberately tiny because raising it to a 0.30 share scored
  530 Elo against 913 (run 756); enemy action 0.11 because halving it did worse (run
  763); entropy -0.0081 as 3e-4 x 27; next state, correction and geometry set in v2 by
  matching gradient norms on short CPU runs.
- Only those last three were re-measured in v2. The rest were ported as "starting
  values" and never re-measured, including after the rewards were multiplied by 10 and
  the value support widened. The measurements they rest on were on a different game,
  head set and team size.

**What cross-entropy does and does not buy** (`blind-learner.md` section 2):

- It bounds each softmax group's logit gradient whatever the target's units. That made
  target normalisation and Huber losses redundant, and they are already gone.
- It does not balance terms against each other. They differ in how many groups are summed
  per token (a next-state row is about 30 nats, a policy token under 3.7), in irreducible
  noise and in denominators.
- The policy surrogate is not a cross-entropy. Its gradient scales with the advantage, so
  an advantage normaliser is still needed.

**So what can go:**

| Piece | Verdict | Why |
|---|---|---|
| Return scale (EMA of the 5th-95th percentile gap, floor 1.0) and policy weight 27 | Test against advantages divided by their standard deviation over the update, coefficient 1 | The form is DreamerV3's, which the citation check confirms divides the baseline-subtracted return by max(1, S), so the project's use is closer to the source than the blind pass assumed. What is not from the source: the floor meant "one win" when the outcome weight was 1 and the weight is now 10; the EMA has no bias correction; and 27 was tuned around it. Never tested against an alternative. `origin/main` had found a floor of 1.0 binding and moved to a standard deviation; the working branch went back without a note. Neither option is "the standard": the PPO paper has no advantage normalisation, and the largest study of PPO choices found the per-minibatch kind "seems not to affect the performance too much" |
| Nine coefficients from 0.0033 to 27 | Collapse to about three: policy 1, value small, one shared auxiliary weight | With every term a mean in nats per softmax group and standardised advantages |
| The rule "set weights from measured trunk-gradient shares" | Keep | It is the one piece with a measurement behind it. Re-run it once on the new model |
| Value weight small | Keep small | Measured (run 756): a large critic share hurt play |
| Rewards x10, 77-bin support, support-clip alert, marginal bias tables | Re-derive | A second layer added because outputs are categorical over a fixed range |
| Gradient clip as a normaliser | Expired | Fires at rate 0 in v2. Keep an ordinary clip as a guard |
| Entropy coefficient | Re-test after the advantage change | See below |

## Reconciliation

Where the blind and history passes agree, the row is settled. Where they disagree, or
where history shows the reason has gone, it is listed for a decision.

### Settled: keep

| Component | Rows | Provenance |
|---|---|---|
| Entity-set model with no slot, index or count parameters; per-kind encoders; a global token | 2.1-2.9 | Standard for entity-attention policies |
| Trunk: spatial attention plus a Griffin recurrent block, RMSNorm, gated MLP; small and cheap (295 of 5,172 operations per forward) | 3.1-3.14 | Published (Griffin); standard parts |
| Rotary relative position in attention | 1.26, 3.3 | Measured in v2: switched off at inference the best model loses every game. Sequence rotary is standard; its use on metric 2D coordinates is the project's own (see `citations.md`) |
| Categorical outputs trained by cross-entropy | 5.x | The owner's stated reason holds. Published support for categorical value targets is for value-based and model-based agents; no experiment with PPO was found, and two-hot (what the code uses) is the weakest variant in the main paper |
| Joint 42-way action per ship, ships sampled independently | 6.2 | Origin not recoverable; nothing against it. Keep, untested |
| PPO, clip 0.2, GAE, Adam 3e-4 | 9.x | Standard |
| Multi-level rewards with a discount per level | 7.1, 7.5 | One confounded run in favour; a published precedent for separate value heads with different discounts. See "decisions" |
| BC from the scripted agent, then RL | 10.x | Standard pattern |

### Settled: drop, because the reason has gone

Each chain is one removal (`history-model.md` section 8).

- **The belief loop** (decided). With it go: the blur half of every input code, the decode
  stack and the spread fit, the replay record with its dtypes and bit-agreement checks,
  enemy-action forecasts as input, enemy-row heads at play, and the non-finite scrub.
- **The delta head inside the loop.** It needed the error head, which needed in turn a
  median, a confidence gain, a rate, a cap and a quantile spread; the extrapolation
  baseline needed held rates and then their decay; plus the acceleration columns in the
  observation, covariance caps, hidden-physics weighting and the marginal bias tables.
  Every one of these stops a closed-loop runaway and has no purpose in an open-loop
  auxiliary head.
- **The 62 km sizing.** The 236 km colour code, the 340 km rotary ladder (and its 3.405
  ratio and 35 degree turn), the 8x arena code and the 64 km geometry bound. The maps
  that set them were dropped on 2026-11-10 and 2026-11-15.
- **Static-shape padding.** Five fixed zone slots, benched slots as shape padding, fake
  absent fields, and dead and unseen ships kept as tokens for shape's sake.
- **Bit-exactness.** Keyed action sampling and keyed initialisation, the fused kernel
  with its tile padding and plain twins.
- **Learner switches with nothing behind them:** the four trust-region variants (never
  run in training), team spirit (computes an identity), the KL early stop (never fires),
  the plateau and score stop rules (stop nothing), the coefficients for removed heads.
- **Enemy-action head, density head, local presence.** No remaining use once the belief
  goes, or no source.

### Settled: replace

| Component | Rows | Replace with | Why |
|---|---|---|---|
| Arena code plus raw arena radius | 1.28, 1.29 | Per-ship distance and direction to the boundary, clipped | A quantity seen at two values acts as a mode flag. Matches the plan's "the arena enters as what it is" |
| Own and enemy split by row position | 2.8 | A team column and an `exists` mask | Structural obstacle to unequal teams |
| Fixed 5v5 in training | 11.7, 12.3, 12.46 | Team sizes, zone and field counts and arena scale drawn per environment, unequal teams included | Finding 2. A modest range first; the literature's one very wide range trained badly |
| Griffin decay initialisation | 3.8, 3.9 | The paper's initialisation | **(checked)** `temporal.py:82` sets the logits to `linspace(0, 4)` with exponent 8, which at a fully open gate gives a memory of at most about 7 ticks (a quarter of a second). A long-memory initialisation was decided in 2026-10 and never built, while `references.md` says the paper's was kept |

### Disagreements and open rows: for the owner

1. **Attention masking.** The blind pass says use a standard key mask for anything that
   does not exist. History has a hard number against a naive version: masking the
   absent-zone tokens on a trained model cost 325 Elo in RL deathmatch and 374 in BC
   **(checked,** `sections/bc-v19/overnight-log.md:361, 393`**)**. The models use the
   placeholder as somewhere to put attention. The two are compatible: that test changed
   the key set between training and evaluation. *Recommendation:* mask from the start of
   training, give the model one always-present learned null key so it has a sink that
   does not depend on what is absent, and never let the key set's make-up differ between
   training and evaluation. `origin/main` masked dead keys.
2. **Hidden enemies.** The decision on record is that hidden enemies leave the
   observation. With a recurrent state per entity, removing the token removes the place
   its memory lives, and what remains is 128 numbers per own ship to remember every
   hidden enemy. History adds that a token with no position is not neutral: it sits at
   the origin. The blind pass proposes keeping a seen-then-hidden enemy as a token with
   its state zeroed, its last-seen position and the time since. That is not a belief: no
   extrapolation, nothing decoded and fed back. *Recommendation:* last-seen tokens in the
   baseline, plain removal as the comparison. This refines a decision of the owner's, so
   it is the owner's call.
3. **Inputs as codes or as scalars.** The 542-wide categorical state code is most of the
   ship token and most of the input cost (1,037 operations per forward). Its reason is
   not recoverable, and half of it presents belief uncertainty. The blind pass would use
   about 20 scaled scalars. The owner's argument for categorical outputs is about losses
   and does not extend to inputs. *Recommendation:* scalars in fixed physical scales; one
   BC comparison against sharp codes if there is doubt.
4. **Absolute position inputs and the random offset.** History: the offset's reason
   (v1's torus, non-memorisable positions) has expired. Blind: keep it and add Fourier
   position features, because rotary puts geometry only in attention weights and not in
   the values. Literature: use relative geometry and avoid absolute coordinates as the
   main positional signal, since absolute metres go out of range on a larger map.
   *Recommendation:* no absolute position input and so no offset; rotary for relations
   between entities; boundary and zone geometry expressed relative to each ship, clipped.
   Test whether the ship token needs explicit relative features on top, which is the
   question the geometry head was added to answer.
5. **Locality in attention.** The literature's best-supported rule is a bounded
   neighbourhood per query. v2's own per-ship nearest-neighbour MLP fails as fleets grow
   (0.89 / 0.31 / 0.00 at 10 / 20 / 40 a side at even numbers), so locality alone is not
   the answer, and no paper tests locality masks inside a centralised multi-block stack.
   *Recommendation:* not in the baseline; decide from suspect test 3.
6. **Value form and discounts.** The plan says one value head. The blind learner pass
   keeps the levels with a discount each; the blind model pass keeps one critic plus an
   outcome head. History: the run that stopped fighting (`rl-check`, 2026-11-13) used one
   discount of 0.9999 and one lambda of 0.99 with no entropy bonus; the evidence that
   per-level discounts fixed it is one sentence about a run on a different game version,
   clone and batch size; the planned single-value arm at 0.995 / 0.95 never ran. Two
   recorded hazards for a merged outcome: bootstrapped at discount 1 it drifted (run
   737), and discounted and bootstrapped in BC it gave negative Brier skill.
   *Recommendation:* keep the levels and per-level discounts in the baseline as the one
   safeguard with any evidence, treat outcome as one of the levels, use one lambda, and
   make the single-value run the first learner experiment. It is cheap to state and
   would settle the brief's own open question.
7. **Entropy in RL.** The RL stage runs with coefficient 0 **(checked:**
   `cli.warm_start_entropy` overrides the default of -0.0081**)**. It was deliberate, on
   one ablation in which the bonus destroyed the clone's play in 24 updates. The blind
   pass's reading is that the bonus was the same order as the range-normalised
   advantages, so the ablation shows mis-scaling, not that entropy hurts. Entropy later
   fell to 0.4 in rl-v20. *Recommendation:* re-test at 0.001 to 0.003 after the advantage
   change, and add a decaying distillation term toward the cloned policy. Every
   BC-then-RL source the citation check read keeps both an entropy term and such a term;
   v2's configuration (entropy 0, no distillation) has no provenance.
8. **RL update size.** By the blind pass's arithmetic from `launch.toml`, RL gets about
   48 PPO updates (10 of them warm-up) at the launch budget, with far fewer optimiser
   steps than BC. Worth one run at a quarter of the update size.
9. **Shot outcome labels.** The re-simulation is 311,664 of the 353,330 operations in
   the once-per-update prepare phase and survives the removal of the trajectory head.
   Labels from fired bullets only, recorded by the game, remove it (already the plan's
   intention). Bullets are not observed at all, which is why the shot heads exist.
10. **Geometry head.** It replaced a rotary read-out from an abandoned design and
    inherited that design's weight. Kept by the owner's decision; no measurement.

## A baseline with provenance

Categories, as the owner set them: **(a)** standard; **(b)** published work supports it
for a comparable purpose; **(c)** measured in this project; **(d)** a reasoned choice, to
be tested. "d" rows are the test list. Sources are in `citations.md` and
`literature-size.md`.

**Model**

| Part | Choice | Cat. |
|---|---|---|
| Observation | Ship, zone and field rows of variable count, with `exists`, `team`, `visible`, `alive`; padded and masked | a |
| Ship inputs | About 20 state scalars at fixed physical scales (symlog where the range is wide); type one-hot; flags | a. No RL source was found for two-hot or binned inputs; DreamerV3 uses symlog scalars for inputs and two-hot only for targets |
| Position | None absolute. Clipped distance and direction to the boundary per ship | d |
| Hidden enemies | Last-seen token: state zeroed, last-seen position, time since seen; never-seen enemies masked | b by precedent: OpenAI Five reuses the last-seen observation for fogged enemy heroes (confirmed) |
| Encoder | Per kind: linear, norm, activation, shared linear to the width | a |
| Spatial attention | Standard scaled dot-product attention with a key mask and one learned null key | a for the mask (hide-and-seek, REFIL); d for the null key (register tokens support it by analogy only) |
| Relative position | Rotary on queries and keys, about six bands sized to the largest intended map, not a power-of-two ladder | c (load-bearing in v2); d for the ladder. No published use of rotary on metric world coordinates of a variable entity set was found, so this is the project's own mechanism |
| Temporal | Griffin recurrent block per token, the paper's decay initialisation | b |
| Normalisation | Per-token only; nothing computed across the entity axis | a |
| Policy head | Joint 42-way categorical per ship, read from that ship's token | d |
| Value | Per-ship categorical critic per reward level, including outcome, with HL-Gauss targets | b for categorical targets (not shown with PPO) and for per-level heads (RND) |
| Auxiliary heads | Next state as an open-loop change from the current state; shot outcome from game-recorded labels; geometry | d each. No source read shows a forward-prediction auxiliary loss helping an on-policy actor-critic, and the planned experiment on the heads never ran |

**Learner**

| Part | Choice | Cat. |
|---|---|---|
| Algorithm | PPO, clip 0.2, 4 epochs, GAE, one lambda | a |
| Discounts | One per reward level | b, with weak c |
| Advantages | Sum over levels, then one normaliser: the standard deviation over the update, or DreamerV3's percentile range with the floor and EMA fixed. Recompute advantages once per pass over the data | d (one comparison run); b for recomputing |
| Losses | Every term a mean in nats per softmax group; policy 1; value small; one shared auxiliary weight, set once from a measured trunk-gradient share | c for the rule and the small value weight |
| Entropy | One coefficient, re-tested after the advantage change | d |
| Returns | Bootstrap from the value at the rollout's end; no lookahead; rewards averaged per ship so targets do not grow with fleet size | a; b for the per-ship averaging |
| Optimiser | Adam 3e-4, short warm-up, ordinary gradient clip, bf16 with fp32 accumulation | a |
| BC | Soft-label cloning from the scripted agent, the same return settings as RL | a |
| BC to RL | A distillation term toward the cloned policy, decayed to zero, alongside an entropy term | b (Kickstarting; every BC-then-RL source read keeps both terms). v2 has neither |
| Averaged weights | At most one average, and only if measured to help | d. No source supports averaged policy weights for evaluation or a league |
| Training distribution | Team sizes, unequal teams, zone and field counts and arena scale drawn per environment | b |
| Evaluation in training | A periodic evaluation at a size never trained on, reported beside the training-size rating | d |

What this keeps from v2: the entity-set formulation, the trunk, rotary, categorical
outputs, the joint action, the reward levels and the gradient-share rule. What it takes
from `origin/main`: masked keys and standard-deviation normalisation. What it drops is
almost all in the inputs, the belief path and the learner's unused switches.

## Citation check

A separate pass fetched and read the sources the baseline rests on (`citations.md`; 21
claims, all against full text except the AlphaStar Nature paper and one other). What it
changed:

- **Griffin's decay initialisation: confirmed, and worse than stated.** The paper
  initialises so that the per-step retention is uniform between 0.9 and 0.999. The
  project's range lies entirely below that: its longest-memory channel (retention 0.865,
  about 7 ticks) forgets faster than the paper's shortest (0.9, about 10 ticks), and the
  paper's longest is about 1,000 ticks.
- **Advantage normalisation is not "the PPO standard".** It is not in the PPO paper. The
  largest empirical study (Andrychowicz et al.) found per-minibatch normalisation makes
  little difference, found value-loss clipping hurts, and recommends recomputing
  advantages once per pass. So replacing the return scale is a test, not a correction;
  the audit's tables above say so.
- **DreamerV3's normaliser** divides the baseline-subtracted return, as the project does.
  Its entropy scale of 3e-4 is presented as working because of that normaliser.
- **Categorical value targets are well supported for actor-critics, with one gap and one
  mismatch.** DreamerV3's critic is a two-hot symlog categorical and the HL-Gauss paper
  has an IMPALA result, so the choice is category (b). The gap: none of the papers read
  runs PPO. The mismatch **(checked):** the project uses HL-Gauss (sigma 0.75 bins) for
  the state, next-state and geometry codes, but the critic in
  `models/yemong/critic.py` uses plain two-hot, which is the weakest variant in the
  HL-Gauss paper. Moving the critic to HL-Gauss is a one-line change with support.
- **Rotary on metric coordinates has no precedent found.** The vision work uses integer
  grid indices. It is the project's own mechanism, measured here to matter.
- **Auxiliary prediction heads have no on-policy support.** The forward-prediction results
  are on value-based agents, and in UNREAL input reconstruction made scores worse.
- **BC then RL with no entropy and no distillation term has no provenance.**
- **Averaged policy weights for evaluation or a league: no support found.**
- **Confirmed as cited:** per-level value heads with separate discounts (RND); the
  k-nearest masking result (Hsu et al.; note the mask is over targets, applied at
  evaluation, with a value-based algorithm); the softmax dispersion numbers (and that
  adaptive temperature did not fix it); OpenAI Five's last-seen observation for fogged
  heroes, per-hero recurrent state and max-pooling over units; attention masking of
  absent entities (hide-and-seek, REFIL; not UPDeT, which masks actions).
- **Papers that could not be read in full:** the AlphaStar Nature paper (its distillation
  term and league weighting are confirmed only second-hand) and Nikishin et al. 2018.
  Those two are the only ones worth the owner downloading.

The net effect on the baseline: fewer rows are category (a) than the blind passes
claimed. PPO, GAE, masking and scalar inputs are standard, and the categorical critic is
supported. The next-state and other auxiliary heads, rotary on world coordinates and
heading, the advantage normaliser and the way loss weights are chosen are the project's
own choices with partial or no outside support, which makes them the test list.

## Open questions agreed with the owner (2026-11-22)

- **Value form:** reward levels and per-level discounts stay for now; the single-value
  run is the first learner experiment.
- **Advantage normaliser:** DreamerV3's form was taken as a starting point, not as a
  measured choice.
- **Loss weights:** how the weights between heads and between value and policy are chosen.
- **Nearest-k attention:** an arm to test.
- **Heading in rotary:** the blind pass dropped the heading harmonics for want of a
  source. The owner's argument for them is sound (two ships side by side are in very
  different situations depending on their attitudes, and an angle is exactly periodic,
  so rotary cannot alias on it). Keep them, as category (d).
- **Settled:** hidden enemies as last-seen tokens with time since seen; Griffin's decay
  at the paper's initialisation; an entropy term in RL; variable maps in evaluation only
  if they are in training.

## Lessons to carry into the rewrite

Each is a failure that happened. Sources are in the two history files.

1. **A decoded forecast fed back with no gradient compounds its own error** (v1 run 734;
   v2's velocity error reaching 1,039 m/s). If a belief ever returns, this is the cost.
2. **The mean and second moment of a wide symlog categorical are tail-dominated.** Use
   medians and quantiles, or narrower supports.
3. **A trained model depends on the make-up of its key set.** Masking a placeholder at
   evaluation cost over 300 Elo. Keep training and evaluation alike.
4. **A token with no position sits at the origin,** it is not absent.
5. **A power-of-two rotary ladder aliases exactly at its coarsest period.**
6. **A large trunk-gradient share for the critic hurt play** (run 756).
7. **A floor that binds silently rescales a signal,** by four orders of magnitude for
   one critic on run 719.
8. **A long-horizon outcome bootstrapped every 128 ticks drifts at discount 1** (run
   737).
9. **An entropy bonus sized for one advantage scale can swamp the policy gradient at
   another.** The advantage scale is about to change.
10. **Seeds move every arm together.** Seed 0 beat seed 1 by 112, 115 and 110 Elo in
    three configurations **(checked,** `sections/gpu-box-log.md:592-594`**)**, attributed
    to the shared initialisation and data order. Compare like seeds only.
11. **Settings that live only in a launch command get lost.** The outcome level's
    discount did, and cost a day of runs on the wrong base.
12. **Short CPU runs misled on gradient shares:** the geometry head's ordering reversed
    in production.

## Defects found

Confirmed in code by the blind learner pass unless noted.

- **Return-scale EMA has no bias correction.** It starts at zero with decay 0.99 and is
  updated once per learner update, so it under-reads throughout a run and the advantage
  divisor is too small. Moot if the normaliser is replaced.
- **GAE bootstraps through values the critic is not trained on:** ships outside the
  valued mask still feed the trace.
- **Benched ships count as valued.** Harmless today; wrong as soon as benching is used
  to sample team sizes.
- **The outcome value changes meaning at the hand-over:** BC uses discount 1 and lambda
  1, RL 0.9997 and 0.97.
- **The density loss divides both sides' cells by the own-ship count.** Moot if the head
  goes.
- **The Griffin decay initialisation** is not the paper's (above).
- **`launch.toml`'s comments disagree with the code** on the mode mix, the rotary ladder
  and the hidden-physics decay, and it still fixes a lineup against the decisions of
  2026-11-14.
- **The critic's targets are plain two-hot** where the review decided HL-Gauss.
- **RL starts from the exponential weight average** where the only measurement favoured
  the uniform one.
- Suspected, not checked in code: the damage trace decays faster than a bullet flies, so
  credit for a shot is heavily attenuated; the outcome discount of 0.9997 at 30 Hz is a
  77 second half-life in a 600 second match.

## Corrections to earlier documents

- **The "+27 Elo" for the error head** (`simplification.md` finding 1; `review-packet.md`;
  repeated in `review.md`). It comes from a tournament the log later says ran on a wrong
  base **(checked,** `sections/gpu-box-log.md:298-311`**)**, and it compares a bundle
  (delta head, extrapolation baseline, split) with the absolute head, not the error head
  with its absence. The history pass reports larger seed-paired gaps for the bundle on
  the corrected base. Every arm in every tournament had a belief. The removal is a
  decision about cost and simplicity; there is no measurement either way.
- **`review.md`, objection 3.** I treated 110 Elo as unpaired seed noise. The record shows
  the seed effect is common across arms, so same-seed comparisons are much tighter. The
  objection stands only in a weaker form: all clean pairs are BC, there are two seeds,
  and the pairing depends on a seed fixing the initialisation and the data order. That
  is far cheaper than bit-exactness, but the plan's non-goal as worded gives it up.
- **`review.md`, the three starting papers.** REFIL is not entity dropout and does not
  test larger counts; UPDeT and the population-curriculum paper are mainly about transfer
  with fine-tuning.

## What I would do next, cheapest first

1. **Run suspect tests 1 to 5 on an existing 5v5 model.** Evaluation only. They decide
   the inputs and the attention of the lean model.
2. **Decide the three owner rows:** hidden enemies as last-seen tokens (2), inputs as
   scalars (3), the value form (6).
3. **Make size and map sampling part of the skeleton, not the last port.** The brief puts
   variable maps at step 8; the evidence says the headline claim depends on training
   seeing variation, and that map size is where v2 fails first.
4. **Put a size curve in the canary:** the policy trained at the small sizes against the
   scripted agent at each of 5, 10, 20 and 40 a side, even and outnumbered, on maps
   scaled to training density and on the training map.
5. **State the claim to fit the evidence.** On what is known today the defensible
   headline is zero-shot within the trained range of local density, with a short
   fine-tune reported separately outside it. Anything stronger is a result to earn, and
   if earned it would be new.
