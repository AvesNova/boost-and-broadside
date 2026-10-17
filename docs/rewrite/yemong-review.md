# Yemong model and training review

Piece-by-piece review of the current model and training code (branch
`feat/counterfactual-shot-labels`), read from the code rather than the docs. Each part
lists what the code does, hidden assumptions, findings (stale or dead pieces), and
questions for the author. Outcomes move into `decisions.md`.

## Part 1: overview and inputs

### Shape of the network (shipped `MODEL_CONFIG`)

- d_model 128; 2 blocks of `spatial, spatial, temporal` (S S T | S S T).
- Spatial = pre-norm transformer block: attention + SwiGLU FFN (4x). Two 64-wide
  heads (`n_spatial_heads=2`), rotary Q/K by world x, y and attitude.
- Temporal = Griffin / RG-LRU with a causal conv (packed hidden state per token).
- Token axis: N ships, then 1 global token, then map tokens (fields, zones).
  - Ships and the global token are queries, carry recurrent state, and go through
    every sublayer.
  - Map tokens are key/value-only memory (`map_read_mode="kv_memory"`, projected to
    64 wide): never queries, never recurrent, never through the FFN.
  - Ships and map tokens share **one softmax** (ships can trade attention between a
    nearby field and a distant ally).
  - Bullets would be a separate K/V cross-attention, but **bullet reading is off**
    (`n_bullet_cross_per_block=0`).
- No attention masks at all: every ship is revealed on spawn and belief validity is
  sticky, so all tokens are always valid (keeps the flash kernel).
- Heads read ship tokens: joint 30-way action, enemy next-command (30-way), next-state
  code (469), per-ship categorical value levels (51 bins). The global token feeds the
  outcome value head (win/tie/loss/unresolved) and the hex density head. Counterfactual
  shot heads are training-only.

### What a token sees

Built by `FeatureCoordinator` (`train/rl/features.py`) from the observation dict, with
a per-type first projection (`encoder_split`) and a shared second projection.

- **Ships:** a 469-wide categorical code (position 9 nested levels of 3x3; velocity 3
  projections x 81 bins; attitude 4 levels x 4; angular velocity 41; shield delay,
  health (= shields), power, local log index 21 each; cooldown 4). Truth gives an exact
  code; a hidden enemy's belief gives a code smoothed by its spreads. Plus: team one-hot
  (3), alive, visible, is_shooting (fired this tick, from truth only), time since
  observation (symlog), respawn time (Fourier), object type one-hot (4), zone role
  one-hot (6), pending action (30-way vector), radius, local index gradient, and two
  local-presence scalars.
- **Fields:** position code, radius, transition width, target log index.
- **Zones:** position code, role, capture progress and direction, offensive and
  defensive distance.
- **Global token:** front position, win threshold, time remaining, game mode one-hot.
- Geometry also enters through rotary rotation of Q/K (absolute world x, y, attitude).

### Hidden assumptions

1. **The policy cannot see bullets.** Bullet reading is off in the shipped config. Any
   dodging has to be inferred from enemy pose, `is_shooting` and the counterfactual
   shot training. Combined with today's ~114 deg a target can turn during bullet
   flight, this shapes the whole combat game.
2. **Two teams are hard-coded in the inputs.** `team_id` is a 3-way one-hot (us, them,
   map) and canonicalization swaps team 0/1. Free-for-all needs relative team IDs.
3. **Game mode is a 2-way one-hot** (`GameMode`: COMBAT, FRONTLINE). The four planned
   modes, capture options and rule settings (lives, respawn) need a richer global
   token.
4. **The action space is baked into inputs and heads:** pending action is a 30-way
   vector and both action heads are 30-way. Stick thirds make it 42.
5. **Position is absolute.** The position code spans the 65,536 m torus with 9 levels of
   3 per axis (finest cell ~3.4 m). Coarse levels are near-constant within one map,
   which is why map scale is randomized in training.
6. **Ship identity/type is not observed** (one ship type); ship-card parameters are not
   in the observation yet.
7. **Feature encoding lives in `train/rl`, and the env wrapper builds a
   `YemongObservation`.** The model's input encoding and the observation format are
   coupled (the seam problem recorded in `decisions.md`).

### Findings (stale or dead)

- `ModelConfig.n_heads` (4) is used only for a divisibility check; every attention
  layer uses `n_spatial_heads` (2). Effectively dead config.
- Comments in `config/core.py` and `config/defaults.py` refer to a value-head
  "TeamPMA" pooling that no longer exists (the per-ship value head is a plain MLP).
- `evaluate_actions(alive_mask=...)` is documented as "used for team pooling in the
  value head" but is unused.
- The attention class docstring says bullets and map memories use separate softmaxes;
  map memory now shares the ship softmax (bullets are still separate).

### Questions for the author

1. Was turning bullet reading off a deliberate gameplay/perf choice you want to keep,
   or a temporary setting? Should agents see bullets in v2?
2. Is a 469-wide categorical code per ship (and the belief smoothing that goes with it)
   something you want to keep as the default input encoding, or one model's choice
   under the new contract (env gives raw physical values; the model encodes)?
3. Global token: should it carry all rule settings (mode, capture rule, lives, respawn
   delay, score/time limits) as observed values, so one agent can play every mode?
4. Absolute positions plus rotary vs. ego-centric inputs: keep absolute (with map-scale
   randomization), or is relative geometry something to revisit?

### Outcomes (author, part 1)

- **Bullets:** not observed by Yemong, but present in the raw observation so other
  models can use them.
- **Teams:** a categorical feature, two teams for now; every agent sees itself as team
  0. **Token type is a separate categorical:** ship, field, zone, super (global) token.
- **Game mode:** categorical for now.
- **Action space:** 42-way joint command (stick thirds) accepted.
- **Positions:** keep absolute positions plus rotary encoding (with map-scale
  randomization). Relative inputs conflict with count-agnostic models.
- **Ship type / ship-card parameters in the observation:** added to the open list.
- **Raw observation vs model input:** the env emits a raw observation only; a
  model-owned adapter (part of the agent) turns it into the model's input (e.g. the
  Yemong categorical codes). Encoding travels with the model's weights.
- **Categorical codes and belief smoothing:** one model's choice, not a contract. The
  author is unsure about the current belief system and wants to experiment later with
  letting the Griffin layers handle memory alone.
- **Per-team differences in the raw observation** are not only team IDs: also
  visibility (which enemy slots are masked), private pending actions (own exact, enemy
  unknown) and time since seen. Map tokens and visible ships' physical values are
  identical across teams. Sharing preprocessing between teams is an optimization inside
  an agent that accepts stacked observers, not part of the contract; measure before
  designing for it (the trunk, which cannot be shared, dominates cost).
- **Masked (unseen) values:** the raw observation carries a mask; a model's adapter
  represents a masked categorical as a **uniform distribution over its categories**
  (author's choice for Yemong). Keep an explicit visible flag as well.
- **Time since last seen** is a feature, Fourier-encoded like the other time features.
- Consequence to settle: rotary encoding has no position for a masked token, so a
  hidden token's rotation should be the identity (no geometry).
- **Full truth plus masks as the observation:** rejected (recommendation). It leaks
  hidden information to any model that ignores or mishandles the mask, silently inflates
  results on a masking bug, and makes a leaderboard unenforceable. The env applies the
  masks; truth reaches only learners, as privileged labels.
- **Masked categorical: zeros plus a visible flag vs uniform.** Zeros plus a flag is the
  standard missing-value pattern and is exactly a learned mask embedding (the flag's
  input column). Uniform is only better when uncertainty is graded (belief smoothing),
  where it is the limit of infinite spread. Recommendation: zeros plus flag if memory
  moves into Griffin; uniform if belief smoothing stays. Adapter choice, not contract.
- **Identity rotation is not "no geometry":** it equals rotating by position (0, 0)
  and heading 0, so a hidden token would look like it sits at the torus origin (a random
  spot relative to the randomly translated map). Today a hidden enemy is rotated by its
  belief mean. Recommendation: zero the rotated dimensions for a masked token so
  attention to it uses only the unrotated (content) dimensions. (Correction, see part
  2: on the 65,536 m Frontline torus rotary uses 48 of the 64 dimensions, leaving 16
  unrotated, not 40 and 24.)
- **Decided:** a visible flag always; masked categoricals are **uniform** for now
  (belief stays in the port). Griffin-only memory is post-transition work; the adapter
  and model code should make swapping the masking/memory scheme cheap.

## Part 2: trunk internals (rotary geometry, Griffin memory, resets)

### Rotary geometry (`models/yemong/rope.py`)

- Q and K of every spatial layer are rotated by world x, world y and attitude. Each
  frequency pair adds `cos(phase + w * (x_query - x_key))` to the score, so relative
  displacement enters attention directly.
- Frequencies are base-2 harmonics of the world period, chosen so the finest period is
  at most 128 m (`POSITION_FINEST_PERIOD`), plus 4 attitude harmonics. Exactly
  periodic over the torus, so wrapping is seamless.
- On the 65,536 m Frontline torus that is **10 frequencies per axis + 4 attitude = 24
  pairs = 48 of the 64 head dimensions**, leaving 16 unrotated. (Docs and config
  comments say 16,384 m / 8 frequencies / 40 dimensions: stale.)
- Tokens without a heading (fields, zones, global token, bullets) use attitude angle 0,
  so their attitude block is not rotated: map objects get an absolute-heading term
  rather than a relative one.
- Rotary and the encoder's position features share one frequency definition
  (`base2_frequencies`) so they cannot disagree.

### Griffin temporal block (`models/yemong/griffin.py`)

- Per token (ships and the global token), per temporal sublayer:
  `norm -> linear1 -> causal depthwise conv (4 taps) -> RG-LRU`, gated by
  `GELU(linear2)`, `-> linear_out`, residual, then a SwiGLU FFN with residual.
- RG-LRU: `a_t = sigmoid(L)^(8 r_t)`, `h_t = a_t h_{t-1} + sqrt(1 - a_t^2) (i_t x_t)`,
  with learned per-channel `L` and input-dependent recurrence and input gates (as in
  the Griffin paper).
- Hidden state per token per temporal sublayer: RG-LRU state (128) plus the conv buffer
  (3 x 128), packed into one tensor.
- Rollout runs one step at a time; the update re-runs the whole rollout with a parallel
  scan (Hillis-Steele, O(T log T) work) from the hidden state stored at the rollout's
  start (truncated backprop at the 128-step rollout boundary).
- Map tokens never enter the temporal path in the shipped config (`kv_memory`).

### Resets

- Episode end: during rollout the trainer zeroes the whole hidden state (RG-LRU and
  conv buffer) of finished environments; in the update, `done_mask` zeroes the decay
  at the next step and masks conv taps across the boundary, so both paths agree.
- **Respawn does not reset memory:** a ship's slot keeps its recurrent state through
  death and respawn ("match memory persists").
- The global token is recurrent exactly like a ship.

### Hidden assumptions

1. **Initial memory horizon is short.** `L` is initialized `linspace(0, 4)`, so
   `sigmoid(L)` spans ~0.5 to ~0.98. With the recurrence gate fully open the per-step
   decay is `sigmoid(L)^8`: at most ~0.85, a time constant of ~6-7 decisions (~0.2 s
   at 30 Hz). Longer memory requires the gate to close. The Griffin paper initializes
   `a^c` in [0.9, 0.999]. Relevant if Griffin is to take over memory from the belief
   system later.
2. **Slot identity is memory identity.** Memory is per token slot, so it relies on a
   ship keeping the same slot for the whole match (it does today).
3. **Truncated backprop at 128 steps** (~4.3 s): gradients never cross a rollout
   boundary, so credit for anything longer must come through the critic.
4. **Rotary resolution is set by the torus, not the arena:** the finest period is 128 m
   regardless of map scale (a 10 m ship spans ~28 deg of the finest phase). Fine
   geometry below that relies on the content dimensions (position code).

### Findings (stale or dead)

- Rotary dimension count in docs/config comments (40 dims, 16,384 m) is stale for the
  65,536 m Frontline world (48 dims).
- `forward_nonrecurrent` and the `field_sub` adapters exist only for
  `map_read_mode="full_attention"`, which the shipped config does not use.
- `YemongBlock` docstring says it uses `n_heads` (it does not).
- The Hillis-Steele scan is work-inefficient; in JAX this becomes
  `lax.associative_scan` for free.

### Questions for the author

1. Should respawn reset a ship's memory, or should memory persist through death (as
   now)?
2. Is ~4.3 s truncated backprop (128-step rollouts) enough, given the goal of longer
   rollouts? (Training-side question; flagged here because memory length depends on it.)
3. Keep the RG-LRU initialization as is, or move toward the Griffin paper's
   long-memory initialization (likely needed for Griffin-only memory later)?
4. Keep map tokens as K/V-only memory (no temporal path), and drop the full-attention
   variant and its adapters from v2?

### Outcomes (author, part 2)

- **Memory goal:** Griffin should remember everything from "that ship just ducked behind
  a field" (seconds) to "this team has played 3 attackers and 2 defenders since the
  start" (minutes; a Frontline match is up to 9,000 decisions = 5 min).
- **Respawn keeps memory** (as today).
- **RG-LRU init:** move to a long-memory initialization (Griffin-paper style).
- **Map tokens:** K/V-only memory; drop the full-attention variant and its adapters.
- **Slot identity = memory identity:** fine, with a test.
- **Backprop length:** truncated backprop already equals the rollout length (128
  decisions, ~4.3 s, ~1.4% of a match). Memory state is carried forward across
  rollouts, but the gradient only teaches retention that pays off within one rollout,
  so minute-scale memory needs much longer rollouts (and the critic) to be learned.
  Longer rollouts cost activation memory (rematerialization and fewer parallel envs)
  and staleness of the stored start-of-rollout state; to be settled in the training
  review.
- **Refinement (what truncation really limits):** the network can *use* memory from
  before the rollout (the carried state is read with gradients), and a keep/accumulate
  rule learned inside the window keeps running for the whole match at inference
  (recurrences length-generalize). Facts that are re-observed often (enemy team
  composition) are learnable as running accumulations from short windows. What
  truncation blocks is learning to *write* a one-shot fact whose payoff comes more than
  one rollout later. Dreamer 4 reportedly alternates long and short training batches
  for long context (secondary summaries; a transformer world model, so only an
  analogy). Existing measurement (`docs/engineering/rl-throughput.md`, "Shorter
  rollouts for speed"): at a fixed token budget, update cost per token is flat from
  T=32 to T=256, so longer rollouts with fewer environments are nearly free in
  compute; the trade is fewer independent environments per update.
- **Decided:** a single rollout length in the initial implementation. Design so that
  later adding multiple rollout/env-count variants (preferred form: separate env pools,
  each with its own compiled update, alternated by the Python loop) is not painful.

## Part 3: heads

All per-ship heads read the final ship tokens; two read the global token. Shipped loss
coefficients are from `profiles/rl.py`.

| Head | Output | Target / loss | Coef | Used at inference? |
|---|---|---|---|---|
| Action | 30-way joint command per ship | PPO clipped objective, entropy bonus, BC cross-entropy | schedule | yes (acting) |
| Enemy action | 30-way, per enemy token | cross-entropy vs the enemy's actual committed command (privileged) | 0.1 | **yes: stored as the enemy's pending-action belief** |
| Next state | 469-wide code logits, residual on the current code | per-group cross-entropy vs the exact next code (alive, non-respawn transitions) | 0.2 | **yes: decoded forecast drives hidden-enemy belief** |
| Per-ship value | 51 symlog bins per reward level (4 local levels) | cross-entropy vs two-hot lambda-return (DreamerV3) | schedule | training only (GAE) |
| Outcome value | 4 classes (win/tie/loss/unresolved), global token | categorical lambda-return, discount as "unresolved" mass | (in value) | training only |
| Hex density | Poisson log-rate per hex cell, ally and enemy, global token | Poisson NLL vs true soft counts (privileged, includes hidden ships) | 1.0 | no |
| Shot trajectory | bullet state code at queried ages | cross-entropy vs ghost/real bullet snapshots | 0.1 | no |
| Shot outcome | 19 classes (hit/near/clear miss, ally vs enemy) | cross-entropy vs ghost/real bullet outcome | 0.1 | no |

### Hidden assumptions

1. **Two "auxiliary" heads are load-bearing at inference.** The next-state and
   enemy-action predictions become the next observation for hidden enemies (belief).
   The model's input at t+1 depends on its own output at t. Stored rollout
   observations mean no gradient flows through that loop, and errors compound for
   enemies hidden a long time. Removing or weakening either head changes what the
   policy sees, not just its training signal.
2. **The critic's shape is the reward function's shape.** One value head per reward
   level (5 levels: outcome, captures, capture progress, kills/deaths, damage). A
   different reward decomposition changes the model's heads, so the agent and the
   reward function are coupled through K.
3. **Correction:** the win/tie/loss/unresolved classes do not assume two teams. The
   two-team assumption is in how the estimate is used: the trainer signs one outcome
   estimate per environment for every ship (`_ship_values`: `sign = +1` on the
   observer's team, `-1` otherwise), i.e. "their win is my loss", which holds only for
   two teams.
4. **The density head is Frontline-specific:** a fixed hex grid over the playable
   circle, scaled per environment.
5. **Ghost bullets live inside gameplay.** Counterfactual shot labels come from ghost
   bullets that share the real bullet ring buffer, with slot-reclaim rules written
   so "gameplay never depends on the ghosts". Training labels are generated by the
   game's own physics state.
6. **Shot labels complete ~29 decisions after launch**, so the last ~29 rows of every
   128-step rollout carry none (~23% dropped); longer rollouts reduce that waste.
7. **Auxiliary weights are uncalibrated:** density 1.0 vs next-state 0.2 vs shot 0.1;
   the docs call the shot weights placeholders.

### Findings (stale or dead)

- The policy module docstring describes the next-state head as "pred next state
  deltas, P from coord"; it outputs 469-wide code logits (a residual on the input code).

### Questions for the author

1. Which auxiliary heads go into the port? Next-state and enemy-action are required
   while belief stays; density and the two shot heads are optional.
2. Ghost bullets: move counterfactual-label generation out of the gameplay bullet
   buffer into a separate privileged-label module (same physics function, its own
   buffer), so gameplay and training labels cannot interfere?
3. Density head: generalize beyond the Frontline circle (other modes and map shapes),
   or keep it Frontline-only?
4. Keep one critic head per reward level (agent's value heads sized by the reward
   function), or a single scalar or summed critic for simpler models?

### Outcomes (author, part 3)

- Auxiliary weights are calibrated on another branch (not visible here).
- **All heads go into the port.**
- **Density head:** centered on the super token, radius set by the local map size (so it
  generalizes beyond the Frontline circle).
- **Critic:** keep the 5-level critic; make a single critic easy to try (agent value
  outputs sized by the reward function's level count).
- **Ghost bullets:** trade-offs written below; author to decide.
- **Depth:** go into more depth on every component (head-by-head from here on).

### Ghost bullets: shared buffer vs separate label module

- *Shared ring buffer (today):* one bullet transport and collision pass serves both
  real and ghost bullets, labels use exactly the gameplay hit code, no extra memory.
  But gameplay state carries training-only data, slot-reclaim rules are needed so
  ghosts never displace real bullets, and labels cannot be switched off without
  touching the bullet logic.
- *Separate label module:* its own small bullet array advanced by the same pure
  transport and swept-hit functions, run only by the training collector. Gameplay
  state is pure gameplay, labels are optional and independently sized, easy to test.
  Cost: a second bullet pass (bullets were ~8.6% of wall clock in the PyTorch profile;
  XLA fuses it into the same step). A cleaner semantics also becomes possible: a
  shadow shot for every eligible decision whether or not the ship really fired, so
  the label always means "what if I shot now".
- Recommendation: separate module. **Overruled (author, decided):** keep one bullet
  operation in the shared buffer (see below).

### Deep dive 3a: action head

- **Distribution:** one categorical over the joint command, `power (3) x turn (5) x
  shoot (2) = 30` today, 42 with stick thirds. Sampled from the full joint (no
  factorization), no temperature.
- **Architecture:** `Linear(128 -> 256) -> RMSNorm -> GELU -> Linear(256 -> 30)`;
  orthogonal init, last layer gain 0.01 (near-uniform initial policy).
- **Ratio:** per ship (each ship's own joint action probability), not a joint team
  ratio. Shared weights across ships (MAPPO-style).
- **PPO loss:** clipped surrogate (clip 0.15), no value clipping. Advantages are the
  sum of the level advantages divided by the EMA 5th-95th percentile return spread
  (floored at one win), with no mean subtraction and no per-minibatch standardization.
- **Masks:** policy gradient counts ships that are the learner's (actor mask), alive,
  and whose decision was committed (a decision cancelled by death or respawn does not
  count). Entropy counts actor and alive (not the commit flag).
- **Entropy:** of the full 30-way joint distribution.
- **BC:** the scripted agent supplies three factor marginals (power, turn, shoot);
  the teacher is their outer product, so BC assumes the scripted policy's factors are
  independent. Loss is cross-entropy between that product and the policy's joint
  log-probabilities.
- **Hidden assumptions:** (1) joint categorical size grows multiplicatively with action
  factors (fine at 42); (2) BC's independent-factor teacher; (3) entropy and policy
  gradient use different masks.

### Outcomes (author, 3a)

- **Bullets stay one operation (decided).** With static tensor sizes every slot is
  transported every tick anyway, so ghosts are essentially free. Checked against the
  code: it takes **two** flags, which already exist. `bullet_active` = can still hit
  (gameplay; a real bullet before it hits). `bullet_flying` = tracked for labels (every
  launched bullet, real or ghost, for its whole lifetime, through any hit). A ghost is
  flying and never active. Every gameplay reader (hits, damage, rendering, the raw
  observation's bullets) must read `active` only, so ghosts never leak.
- **Entropy mask = policy-gradient mask (decided; today's difference is a bug):** exclude
  non-committed decisions from entropy (and BC) too.
- **BC is plain cross-entropy over the 42 joint categories (decided)** against a 42-way
  distribution the scripted agent emits. How the scripted agent builds it (e.g. a
  product of its factor choices) is the scripted agent's business, not the loss's.
- **PPO assessment:** standard core (clipped surrogate, per-agent ratios with shared
  weights as in MAPPO, GAE, target-KL early stopping, no value clipping, which matches
  the "What matters in on-policy RL" findings). Non-standard but principled
  DreamerV3 borrowings: two-hot categorical critic and percentile return
  normalization without per-minibatch standardization. Shipped: clip 0.15, gamma
  0.99 and lambda 0.95 per tick with per-level gammas, 32 minibatches.

### Deep dive 3b: enemy-action head

**What it predicts.** For every enemy ship token, a 30-way distribution over the
command that enemy chose at this same decision (`D_t`). Architecture identical to the
action head (`Linear 128 -> 256, RMSNorm, GELU, Linear 256 -> 30`, gain-0.01 init).

**Timing.** The runtime applies last decision's command (`U_t = Q_t`) and the command
chosen now becomes next decision's pending command (`Q_{t+1} = D_t`). So when the
observation at t+1 is built, the enemy's pending command has not yet moved its ship:
the head is predicting the enemy's **next move** ("intent"), not explaining motion
already visible.

**Target and loss.** Cross-entropy against the enemy's actual committed command (a
privileged label: the learner knows every ship's real action). Mask: enemy team in the
observer's canonical frame, alive, decision committed. **Visibility is not required**:
hidden enemies are in the loss. Coefficient 0.1, applied on every training scale.
Diagnostics log probability of the realized command, entropy, Brier score, accuracy,
and a "persistence" baseline (enemy repeats its previous command).

**How it feeds the next observation** (`train/rl/belief.py`, `write_pending_action_view`):
- `advance` stores `softmax(enemy logits)` as each enemy slot's action belief.
- At the next decision, the pending-action channel (30-wide, per ship token) is:
  allies = exact one-hot of their real pending command; a ship revealed on spawn =
  exact null command; every other enemy = the stored prediction, **whether visible or
  not** (enemy commands are private even when the ship is in sight); a slot never
  observed = uniform.
- In `ego_pass`, each perspective keeps its own belief store and its own predictions;
  only the team-0 perspective's predictions are in the loss (shared weights train both).

**Hidden assumptions**
1. **It models the training opponent mix.** Labels come from whoever controls the enemy:
   the self-play copy, a league checkpoint or the scripted agent. Against a human it
   predicts as if the human were drawn from that mix.
2. **Hidden enemies are trained on** (noisy, high-entropy targets).
3. **One-decision latency semantics depend on action repeat 1** (shipped). With action
   repeat > 1 "pending command" means something different.
4. **Prediction becomes input:** a soft distribution for enemies sits in the same
   channel as hard one-hots for allies; the model separates them by team.
5. **Sized to the action space** (30, becomes 42).

**Findings**
- The pending-action observation key is still named `previous_action`
  (`ObsKey.PREVIOUS_ACTION`) although it means "the command physics consumes next".

**Fit with the new contract.** Raw observation: allies' pending commands exact, enemy
pending commands masked. The Yemong adapter fills masked enemy commands from the
model's own previous prediction, which lives in the agent's carry (the belief store).
Nothing about enemy intent remains in the env or runtime.

**Questions**
1. Keep hidden enemies in this loss, or train only on visible enemies?
2. Keep feeding the prediction back as input (current), or only use it as an auxiliary
   training signal?
3. Rename `previous_action` to `pending_action` everywhere in v2 (assumed yes)?

**Noted for the training review:** in `ego_pass` both perspectives act, but log-probs,
values and next-state predictions are stored from the team-0 half only, and only
team-0 ships enter the policy gradient. In self-play, the team-1 half of the
experience is generated and then not learned from.

### Outcomes (author, 3b)

- **Action repeat is a physics-engine detail.** The head predicts agents' *decisions*,
  one per decision step, regardless of how many physics ticks a decision spans.
- Humans may act more often than the decision rate. Open for the timing contract: how a
  human's inputs map onto decision steps for other agents' observations and labels.
- **Port as is:** keep hidden enemies in the loss and keep feeding the prediction back as
  input. Both are candidates for later experiments.
- **Rename** `previous_action` to `pending_action` in v2.

### Deep dive 3c: next-state head and belief

**What it predicts.** For every ship token, next decision's full physical state as the
469-wide categorical code: 22 softmax groups (position 9 nested levels of 3x3;
velocity 3 projections x 81; attitude 4 levels x 4; angular velocity 41; shield delay,
health (= shields), power, local log index 21 each; cooldown 4).

**Residual form.** `logits = log(input_code + 0.001) + f(h)`, with `f`'s last layer
initialized to zero. At init the head predicts "nothing changes": the next code equals
the code it read. Learning dynamics is learning the residual.

**Target** (`_precompute_ns_labels`): the **exact (sharp) code of the true physical
state one decision later**, from privileged truth, for every ship: allies, visible
enemies and hidden enemies. Absolute, not a delta from the belief (a delta target
let belief error persist unseen; the docstring says that ended an earlier run).

**Loss.** Per-group cross-entropy, averaged over groups. Mask: the slot has been seen
at least once (`belief_valid`), the episode did not terminate, and the transition is
contiguous (no death, downtime or respawn in between). Primary training scale only.
Coefficient 0.2. Logged per group, split visible vs hidden, and against the
"nothing changes" baseline, per population (allies, visible enemies, hidden enemies).

**Why uncertainty grows without being programmed.** Cross-entropy is a proper scoring
rule: for a hidden enemy the best prediction is the *distribution* of where it could
be, given what the model knows. So the forecast widens on its own the longer a ship
stays hidden; the decoder turns that width into spreads.

**Decode to belief** (`decode_logits`, `physical_belief.py`): the head's code is
turned back into **moments**: 11 physical means (position, velocity, attitude,
angular velocity, shield delay, health, power, cooldown, local log index) and 14
spreads (one position sigma, full 2x2 velocity covariance, one sigma each for attitude,
angular velocity and the four scalars, and a cooldown residual distribution). The
baseline floor is removed first so a zero residual round-trips exactly.

**Belief update each decision** (`belief.py`):
1. `observe`: visible ships overwrite the belief with truth and zero spread; a ship
   that just spawned is revealed to both teams and seeds its slot; never-seen slots stay
   invalid (zeros, uniform pending action); `time since observation` counts up while
   hidden.
2. Composition: each slot takes truth if visible, else belief, else nothing; the
   encoder rebuilds the code from those moments and spreads (smoothed code for hidden
   ships).
3. `advance`: after the forward pass, the decoded forecast **replaces** every valid
   slot's belief outright (visible slots get overwritten by truth again next
   `observe`). Non-finite predictions are counted and zeroed.
4. The rollout buffer stores moments, not codes.

**Hidden assumptions**
1. **Belief is unimodal per channel.** A forecast that is genuinely two-peaked ("it
   broke left or right behind that field") is collapsed to one mean and a wide spread:
   a believed position *between* the two possibilities.
2. **Channels are independent** except the velocity covariance; position uncertainty is
   isotropic (one sigma).
3. **Only hidden-enemy forecasts are used**; ally and visible-enemy forecasts are
   training signal only (overwritten by truth next decision).
4. **Code ranges come from the single ship config:** health 0 to `max_health`, power 0
   to `max_power`, angular velocity from the g-limit, slew limit and lowest field
   index, cooldown from `firing_cooldown`. With several ship types these ranges must
   cover every type (or the code becomes per-type).
5. **The codec round trip needs float32** (bf16 matmuls elsewhere).
6. **Feedback without gradient:** forecast at t becomes input at t+1; stored
   observations mean no gradient flows through the loop, and long-hidden ships compound
   forecast error (the docs record a failure, run 734, from an earlier delta target).

**Findings**
- `BeliefTracker.source` still says "instant respawn means a remembered ship is always
  alive"; respawn now has a delay, though a downed ship is public to both teams, so the
  belief is still not fooled.
- Names from an earlier delta-based design survive (`physical_deltas`,
  `PHYSICAL_DELTA_SCALES`, used for scales and a leak threshold).

**Fit with the new contract.** The belief store (means, spreads, action belief, age,
valid) is the Yemong agent's carry; `observe` and composition are the adapter;
`advance` is the end of the agent's step. The env provides the raw masked observation;
the learner provides privileged next-state truth as a label.

**Questions**
1. Accept the unimodal moment belief for the port (assumed yes, since belief stays as
   is)?
2. With ship types: one code range covering every type, or per-type ranges?
3. Keep training the head on allies and visible enemies (pure dynamics learning), or
   restrict it to slots whose forecast is actually used?

### Discussion: HL-Gauss for codes, targets and the critic (author's proposal)

Proposal: every categorical code (observation inputs, next-state targets, value
targets) is an HL-Gauss histogram (Gaussian CDF integrated over each bin). Observed
values use sigma = 0.75 bins; beliefs use wider sigma. Padding bins keep means
consistent at range edges. KL is the quality metric.

**Today's codes** (`categorical_codes.py`): a two-hot of the mean convolved with a
discrete Gaussian of the belief's sigma; edge spill is folded into the end bin and a
closed-form correction step restores the moments (approximate when the end bins hold
most of the mass). Truth is the sigma-0 two-hot.

**Measured (unit bins, numerical check):**

| sigma (bins) | max mean bias over all offsets | max variance error vs sigma^2 + 1/12 |
|---|---|---|
| 0.5 | 2.3e-3 bins | 7.9e-3 |
| 0.75 | 4.8e-6 bins | 3.5e-5 |
| 1.0 | 8.5e-10 bins | 1.1e-8 |

Value exactly on the last real bin, sigma 0.75, spill folded into the last padding bin:

| padding bins | spill | mean bias |
|---|---|---|
| 0 | 25% | -0.28 bins |
| 1 | 2.3% | -0.023 bins |
| 2 | 0.04% | ~0 |
| 3 | 1.5e-6 | ~0 |

**Assessment: adopt, with these details.**
1. *Shift invariance is the real win.* A two-hot's shape (and its own variance,
   0 to 0.25 bins^2) depends on where the value sits between bins, so an observed ship
   looks different to the network depending on offset, and differs in kind from a
   smoothed belief. HL-Gauss at 0.75 has the same shape at every offset (errors ~1e-5)
   and truth and belief differ only in width.
2. *One definition everywhere:* a code is a Gaussian (mean, sigma) in physical units
   projected onto the bins, with an observation floor `sigma_obs = 0.75 x bin width`.
3. *Round-trip corrections or beliefs inflate.* Encode with total width
   `sqrt(sigma_belief^2 + sigma_obs^2)`; decode with
   `sigma_belief^2 = var_histogram - 1/12 bin^2 (Sheppard) - sigma_obs^2`, floored at 0.
   Otherwise every forecast/re-encode cycle adds width, and a hidden ship's belief
   spreads even when the head predicts "nothing changes".
4. *Padding:* 2 bins suffices for observed values; a belief of width sigma needs about
   3 sigma of padding, or the excess is folded (with a known inward bias), or belief
   width is capped. Circular channels (attitude, torus position levels) use a wrapped
   Gaussian and need no padding.
5. *Nested position levels* (3 colours per axis per level): the Gaussian projected onto
   coarse levels is nearly one-hot; `sigma_obs` should be set from the finest level's
   cell (~3.3 m), not per level.
6. *Critic caveat:* value bins are symlog-spaced. HL-Gauss in symlog space with sigma
   0.15 (0.75 x 0.2) makes the raw-space expectation biased by about `exp(sigma^2/2)`,
   ~1.1% too large in magnitude for large returns (convexity of symexp). Two-hot today is
   exact in raw units. Options: accept and measure; HL-Gauss in raw units with a
   local-bin-width sigma; or keep two-hot for the critic only.
7. *KL as the metric:* `KL = CE - H(target)`: identical gradients to cross-entropy, but
   zero means perfect and it stays comparable when target width varies (observed vs
   hidden). Train with cross-entropy, report KL (and KL of the "nothing changes"
   baseline).
8. *Tests:* encode -> decode -> encode stability, zero-residual belief stationarity, and
   edge behavior at full shields, empty cooldown and full power (all very common
   states).

### Outcomes (author, HL-Gauss) and critic analysis

- **Decided: HL-Gauss codes go into the port** (a simplification). Round-trip
  corrections (quadrature floor + Sheppard) confirmed. Circular channels unpadded;
  padding amount is a middle ground to tune (exact means matter less as uncertainty
  grows). Nested position sigma from the finest cell. Train with cross-entropy, report
  KL. Round-trip, stationarity and edge tests required.

**Critic: literature.** Imani & White (2018) introduced the histogram loss with a
Gaussian target (HL-Gauss). Farebrother et al. (2024, "Stop Regressing") found
HL-Gauss with sigma = 0.75 bin widths beats two-hot and regression for value
learning (uniform bins on a fixed range). DreamerV3 uses two-hot on symexp-spaced bins,
expectation in raw units. MuZero applies an invertible squashing transform, two-hot in
transformed space, and inverts the *expectation* (`h^-1(E[h(G)])`).

**The two orders** (51 bins over +/-5 symlog):
- *Gaussian in raw units, then bin on symlog edges:* fixed raw sigma spreads over many
  fine bins near zero and collapses into one coarse bin at large returns, losing the
  same-shape-everywhere property unless sigma scales with local bin width, which is
  the second option in disguise.
- *Symlog, then Gaussian (sigma 0.75 bins), then bin:* same shape everywhere in
  symlog space. Three ways to read a value back (numerical check):

| read-out | single return, round trip | mixture of returns 1 and 9 (true mean 5) |
|---|---|---|
| raw expectation, bin values `symexp(centre)` | up to 1.6% too large (`|G|>5`), 0.026 abs near 0 | 5.078 |
| `symexp(E_symlog)` (MuZero style) | exact to 6e-4 | **3.472** (biased toward the geometric mean) |
| raw expectation, **calibrated bin values** | exact to 2e-4 relative | **5.000** |

  Calibrated bin values: one fixed per-bin table `v`, solved once by least squares so
  that `sum_i HL(mu)_i v_i = symexp(mu)` for every target mean. Recommendation: symlog
  -> Gaussian -> bin, read out with calibrated bin values. It keeps HL-Gauss's benefits
  and two-hot's exactness, at the cost of a precomputed table.
- **Decided:** critic uses symlog -> Gaussian (0.75 bins) -> bin with **calibrated bin
  values**; the velocity code (log-compressed bins) also uses calibrated bin values.
- Still open from 3c: unimodal belief (assumed accepted), code ranges with ship types,
  and whether the next-state loss covers allies and visible enemies.

### Deep dive 3d: critic (value heads, GAE, advantage)

**Reward levels** (`RewardConfig`, shipped weights in win units): outcome 1.0, zone
capture 2.0, zone progress 2.0, kill/death 0.283, damage 0.274. Each event charges the
side it happens to and pays the causing side `payout_ratio x weight`; payout ratios are
2.0 for kills, damage and captures (not zero-sum: an even trade is net positive), and a
step-based "offensive bias" schedule anneals them to 1 between 50M and 300M steps. Team
spirit defaults to 0 on every level (pure per-ship credit).

**Heads.**
- *Per-ship levels* (capture, progress, kill/death, damage): one head, 4 x 51 logits per
  ship over symlog-spaced bins (+/-5 symlog, +/-147 raw), value = expectation in raw
  units.
- *Outcome:* 4 classes off the global token, one per environment; value
  `P(win) - P(loss)`, signed per ship by team (the two-team assumption).

**Discounts** (per decision, 30 Hz), horizons `1/(1-gamma)`:

| level | gamma | lambda | horizon |
|---|---|---|---|
| outcome | 0.9997 | 0.97 | ~3,300 decisions (~110 s) |
| zone capture | 0.999 | 0.97 | ~1,000 |
| zone progress | 0.995 | 0.95 | ~200 |
| kill/death | 0.995 | 0.95 | ~200 |
| damage | 0.991 | 0.90 | ~110 |

The outcome gamma is not 1 on purpose: a 128-step rollout in an ~8,600-step episode
bootstraps ~67 times before a terminal, and at gamma 1 errors compounded (run 737: win
explained variance fell from 0.994 to 0.42). The categorical outcome target models the
discount as mass leaking into "unresolved".

**GAE** (`buffer.compute_gae`): per level, vectorized over the 5 levels, using
rollout-time values. **Truncation is treated as termination** (no bootstrap at the
time limit), because the wrapper auto-resets before returning the observation; the
docstring estimates ~2% of episodes hit the limit.

**Targets and loss** (`_critic_losses`): per-ship levels: cross-entropy vs the two-hot
of the lambda-return (becomes HL-Gauss + calibrated values, decided), over ships alive
**or waiting to respawn** (a downed ship still receives its team's rewards). Outcome:
cross-entropy vs the categorical lambda-return, per environment-step. `vf_loss` = mean
over the 5 levels; coefficient on the schedule.

**Advantage path** (`_precompute_lambda_aggregates`): per level, `adv_i + s_k x mean of
living teammates' adv` (team spirit), summed over levels (weights are already in the
rewards), divided by the return scale `max(1 win, EMA(p95) - EMA(p5))` of the summed
returns of committed, alive actors (EMA decay 0.99, updated from the primary scale only,
applied to all scales). No mean subtraction.

**Hidden assumptions**
1. **Actor and critic share the whole trunk** (only the heads differ). Standard, but the
   value loss shapes the policy's representation; there are gradient diagnostics for
   the actor/critic split because of this.
2. **Truncation = termination** for GAE (slightly pessimistic at the time limit).
3. **Outcome gamma is tied to rollout length** (bootstrap hop count); longer rollouts
   would let it move toward 1.
4. **Two-team zero-sum sign flip** for the outcome (already noted).
5. **The return scale is learned from the primary scale only** and applied to every
   training scale.
6. **Non-zero-sum payouts** (ratio 2) are a deliberate offensive bias, scheduled off by
   300M steps.
7. **Critic levels are averaged equally** in `vf_loss` regardless of reward weight
   (a cross-entropy per level has a comparable scale, so this is reasonable).

**Fit with the new contract.** Reward function: `events -> (rewards per ship per level,
outcome per team)`, with K and the per-level gamma/lambda as part of its config. The
agent's critic outputs K values (K = 1 for a single critic). GAE, team spirit and the
return scale belong to the learner.

**Questions**
1. Keep the shared actor/critic trunk, or make a separate critic trunk an option?
2. Truncation: store the pre-reset final observation so time-limit episodes bootstrap
   correctly? (Cheap in JAX: the step can return both.)
3. Keep team spirit (all zeros today) as a mechanism?
4. Per-level gamma/lambda as reward-function config (assumed yes).

### Outcomes (author, 3d)

- Reward weights, non-zero-sum payouts and return-scale details: **tabled** (already
  changed on an inaccessible branch: recalibrated weights, simple zero-sum).
- **Shared actor/critic trunk: decided** (fewer parameters, the value loss is a good
  prior for the trunk, and privileged critic inputs are not wanted).
- **Time limit as termination: decided and correct.** The time limit is a real game rule
  and `time_remaining` is observed on the global token, so the episode genuinely ends
  there (Pardo et al. 2018, "Time Limits in Reinforcement Learning": bootstrap through a
  limit only when it is artificial and unobserved). Not a bug.
- Per-level gamma/lambda live in the reward-function config: decided.

### Team spirit (explained)

From OpenAI Five. Per level k, a ship's advantage becomes
`own_adv + s_k x mean(advantage of living teammates)`. `s = 0`: each ship is credited
only for events it caused or suffered (low variance, can encourage selfish play such as
kill-stealing or ignoring a teammate under attack). `s = 1`: every ship is equally
credited for the whole team's events (cooperative, but noisy: a ship's own action
explains little of the team total). OpenAI Five annealed it from ~0.3 toward 1 over
training. Here it is 0 on every level; the outcome level is already shared by
construction. It is a cheap mixing matrix in the learner; keeping it as config
(default 0) costs little.

### Gamma, and why it depends on rollout length (explained)

- The return is `G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...`. Gamma < 1 makes a
  reward k steps away worth `gamma^k`, so the agent effectively looks
  `1 / (1 - gamma)` steps ahead. It makes long sums finite and trades bias for
  variance: a short horizon is easier to learn but myopic.
- Rewards beyond the rollout's end are not observed; the critic's own estimate of the
  last state stands in for them (bootstrapping). Inside a rollout, lambda blends
  n-step returns (lambda 0.97 leans on real rewards for roughly 30 steps before the
  estimate).
- A match outcome 8,600 decisions away reaches early states only through a chain of
  bootstraps, one per rollout: ~67 links at 128 steps. Each link copies the critic's
  current error into the next target. With gamma = 1 nothing shrinks those errors along
  the chain, so they accumulate and the critic can drift (run 737). With gamma < 1,
  each link is multiplied by `gamma^128` (0.9997^128 ~ 0.96), which damps the chain.
- Longer rollouts mean fewer links (1,024 steps: ~8), so less accumulated error and room
  for gamma closer to 1. Gamma also changes the objective: at 0.9997 a win 3,300
  decisions away is worth ~0.37 of an immediate one, a mild preference for winning
  sooner.

### Reward function: draft specification

- **Env emits events, not rewards.** Per decision step, fixed-shape arrays accumulated
  over the decision's physics ticks:
  - `damage[i, j]`: shield damage ship i dealt ship j (j may be a zone in damage-capture
    modes: `zone_damage[i, z]`).
  - `death[j]` with a cause (enemy, friendly fire, boundary) and the damage ledger
    needed for kill credit.
  - `capture_progress[z, team]` deltas and `presence[z, i]`; `captured[z, team]`.
  - `outcome[team]` (win / tie / loss) and the terminal reason (score, elimination,
    capture, time limit) on the final step.
  - Lifecycle: `alive`, `downed`, `respawned` per ship; `terminated`.
- **Reward function:** pure `(events, reward_config) -> rewards[N, K]` per ship per
  level in win units, plus `outcome[team]`. Config holds level names, weights,
  attribution rules (who is credited for damage, kills and captures), per-level gamma
  and lambda, and whether it is zero-sum.
- **The game mode defines winning; the reward function defines the learning signal.**
  The leaderboard uses outcomes only.
- **Rules, enforced by tests:** zero-sum per level when configured (sum over teams is
  0 each step); rewards depend only on events (never on hidden state), so the reward
  function changes without touching the env; deterministic; outcome per team
  (ready for more than two teams).
- **Single critic:** a reward config with K = 1 summing all levels. That forces one
  gamma for everything (a real modeling choice, not just a size change).
- Today's code computes rewards from `(prev_state, next_state)` (`env/rewards.py`,
  reading ledgers such as `damage_matrix` and capture ledgers); kill credit is split
  half by damage share and half by the final tick's impact. Those attribution rules
  move into the reward function.

### Naming proposal (to remove "agent" ambiguity)

| Term | Meaning |
|---|---|
| **Ship** | One body in the simulation (a slot index). |
| **Team** | A side; ships share a team id. |
| **Controller** | Anything that picks actions for a set of ships: an NN policy, the scripted controller, random, a human. One controller usually drives a whole team. |
| **Lineup** | The assignment of controllers to ships for a match (data, so "human flies ship 3, NN flies the rest" is just a lineup). |
| **View** | One team's canonical observation (self = team 0). |
| **Carry** | A controller's recurrent state for one view (Griffin state, belief store). |
| **Model / params / checkpoint** | Architecture / weights / saved params plus config. |
| **Learner** | A training algorithm (PPO, BC). |
| **Match**, **decision step**, **tick** | An episode; one controller decision; one physics step. |

"Agent" is avoided in code, except in the JaxMARL adapter where an agent is a team.

### Reward function: revisions (author)

- **Capture by damage and capture by presence are one level** (zone progress): same
  meaning, same distance from the true objective.
- **Events are grouped by level.** One fixed-shape block per level; each level's reward
  is a pure function of its own block. Levels: outcome, zone capture, zone progress,
  deaths, damage.
- **Everything is fixed-shape and dense** (GPU-graph friendly): no variable-length event
  lists, no data-dependent branching; modes and options differ by masks and zeros.

Draft blocks per decision step (accumulated over the decision's physics ticks; N ships,
Z zones, T teams; all static):

| Level | Block | Shapes |
|---|---|---|
| Outcome | result per team (win/tie/loss one-hot), terminal reason, terminated flag | `(T, 3)`, `()`, `()` |
| Zone capture | captures completed per zone and team; credit per ship | `(Z, T)`, `(Z, N)` |
| Zone progress | meter change per zone and team; contribution per ship (presence weight in presence mode, damage dealt to the zone in damage mode) | `(Z, T)`, `(Z, N)` |
| Deaths | died this step, cause (enemy / friendly / boundary), kill credit share per attacker | `(N,)`, `(N,)`, `(N, N)` |
| Damage | shield damage dealt, attacker x target | `(N, N)` |

- *Unified progress:* in both capture modes the progress delta is credited to ships in
  proportion to their contribution; only how contribution is measured differs.
- *Self-contained blocks:* attribution data a level needs travels in its own block
  (kill credit shares are computed by the env at death time), so no level reads another
  level's block.
- *One model across modes:* if a training batch mixes modes, every block uses the
  maximum Z (and T) across those modes with an active mask; a mode without zones has
  all-zero zone blocks.
- *Cost:* the largest blocks are `(N, N)`: 4,096 entries at 64 ships, negligible.

### Deep dive 3e: hex density head

**What it predicts.** Where every living ship is *now*, as a field over a fixed hex
grid: every cell's ally count, then every cell's enemy count (`HEX_DENSITY_DIM` = 2 x
331 = 662). Read from the global token's final embedding only; no ship head reads it
and nothing it predicts re-enters the input. Head: `Linear D -> 2D, RMSNorm, GELU,
Linear 2D -> 662`, orthogonal init, output bias set to the true mean log-rate
(`log(ships per side / 331)`, -4.19 at 5v5).

**Grid** (`train/rl/hex_density.py`). 10 concentric hex rings around a centre cell
(331 cells), stored in units of the playable radius and scaled per environment, so
every map size has the same cells. Centre = `map_center`, unit = playable radius; the
outer ring's corners sit on the boundary. Fixed cell order (ring by ring, from +x,
counter-clockwise). Spacing: 260 m at 5v5, 368 m at 10v10, 581 m at 25v25, 822 m at
50v50 (density-preserving maps).

**Target.** Each living ship deposits exactly one unit, split across the three cell
centres forming its triangle by barycentric weight (the 2D analogue of two-hot:
lossless, no kernel, continuous). The lattice is treated as infinite and off-grid
vertices are dropped, so a ship leaving the circle fades out continuously (gone by
~1.10 radii). Counts all living ships, seen or not (privileged). Observer is always
team 0. Computed on device from pre-step truth each rollout step, stored as a
`(T, B, 662)` buffer tensor.

**Loss.** Poisson NLL on log-rates, `sum_c exp(l_c) - y_c l_c`, averaged over cells
and normalized by step-environment count; no mask (every step has a field). Gradient
`exp(l) - y` is bounded. Log-rate clamped to [-20, 2]. Diagnostics: loss and
**deviance** (loss minus the saturated model's), ally and enemy halves apart.
Coefficient 1.0 (RL), 2.0 (BC); primary scale only.

**History.** Until 2026-09-30 the target was a Gaussian-smoothed field (500 m kernel)
with an MSE loss. The audit (`docs/internal/density-audit-sep2026.md`) found the head
doing real work (R^2 0.925 ally / 0.908 enemy against a 0.167 visible-only baseline,
error growing with hidden enemies beyond the frame-difficulty drift) but the kernel so
wide that two ships read as one until ~2000 m apart. The target was then replaced by
barycentric counts and the Poisson head. **The audit has not been re-run on the count
target**; its numbers describe the old target.

**Hidden assumptions**
1. **Frontline only.** Centre and unit come from `map_center` and
   `playable_boundary_radius`, both zero outside Frontline, where the target is
   meaningless (the code only guards the division against NaN).
2. **Two teams.** Ally/enemy halves; the opening log-rate divides the fleet by two;
   the observer is hard-coded to team 0 (tied to the `ego_pass` finding: team 1's
   view is never a training example).
3. **World-frame grid.** Cells are absolute (map axes), consistent with the absolute
   positions plus rotary decided in part 1. Nothing rotates with a team's attack
   direction.
4. **A fixed resolution in map units, not metres.** Larger maps get coarser cells; the
   field is "where, relative to the map", by design.
5. **"Now", not future.** A nowcast: for allies a restatement of input, for enemies
   belief. It does not ask the token to forecast fleet movement.

**Findings**
- **Clamp at log-rate 2 caps a cell at 7.4 ships and kills the gradient above it.**
  Harmless at the 5v5 primary scale (at most 5 per side), wrong at 10v10 and beyond,
  where a formation can put more ships in one 400-800 m triangle. If the logit ever
  drifts above 2 the clamp also zeroes its gradient, so it cannot come back down. The
  comment's "e2 cells hold every ship in the game" is false above 5v5. In the port:
  clamp only against float overflow (e.g. 15), or clamp at `log(max ships per side)`
  plus margin with a straight-through gradient.
- **Stale comments:** `config/training.py` calls the coefficient an "MSE weight";
  `profiles/rl.py` justifies 1.0 from "a per-cell mean square whose untrained value is
  about 0.1". Both describe the old target. The 1.0 was committed after the Poisson
  switch but its rationale was not re-derived; BC's 2.0 came from run 750's gradient
  norms (target version unclear). Falls under tabled item 13 (weights).
- 62% of cells (rings 7-10) carried 14% of the old target's variance; play
  concentrates near the front. Not a bug, a capacity note.
- The audit's "re-run before trusting the numbers" is still open.

**Port design (proposal)**
1. **Grid placement from the super token.** Every mode gives the super token a
   position and a radius (already decided: centre on the super token, radius from the
   local map size). Frontline: map centre, playable radius (as today).
   **Torus modes without a boundary (TDM):** centre at the super token, radius = half
   the torus diagonal. Minimum-image deltas already fold every ship into the square
   around the centre, so the circle covers the whole torus exactly once; 36% of cells
   (outside the square) never receive mass. One code path, no wrapping grid.
2. **Teams as relative channels.** One field per team in relative order (own team
   first, then the others), static `T_max`, absent teams masked out of the loss. With
   two teams this is exactly today's ally/enemy layout; in FFA it keeps teams apart
   instead of lumping all enemies. Opening log-rate from each team's configured fleet
   size.
3. **Targets derived in the learner, not stored.** The privileged-labels type already
   carries true positions, alive and team per ship; the learner builds the field
   inside the loss. Removes the 662-wide buffer tensor and lets the grid change
   without re-collecting. Each team's view gets its own target once the `ego_pass`
   issue is fixed.
4. **Static and deterministic in JAX.** N ships x 3 vertices, off-grid vertices sent
   to a trash bin, built with a one-hot contraction or `.at[].add`; fixed shapes, no
   branching. Rings fixed (10) in config; cell order defined by the port (no
   backwards compatibility needed).
5. **Clamp fixed** as in the finding above.
6. **Diagnostics:** fraction of deviance explained (Poisson R^2 analog,
   `1 - D_model / D_null` with an EMA per-cell mean as the null) per team channel, and
   the enemy channel split by number of hidden enemies (the audit's belief test, live).

**Questions**
1. Torus placement: half-diagonal circle (proposal) or a wrapping hex grid that tiles
   the torus?
2. Teams: relative per-team channels (proposal) or keep ally/enemy only?
3. Targets built in the learner from privileged truth (proposal)?
4. Keep the target a nowcast, or add a forecast (field at t+k) later as an option?

### Outcomes (author, 3e) and follow-up analysis

**Decided**
- **Every mode has a map centre and a playable radius**; there are no unbounded
  (torus-only) modes. The grid is placed by the super token's centre and the playable
  radius in every mode. (The world itself stays a large torus for translation
  invariance, so deltas keep the minimum image.) The half-diagonal proposal is void.
- **Ally vs enemy only.** An enemy is an enemy whatever its team; two channels in
  every mode, FFA included.
- **A ship outside the playable area appears in no cell.** Hard cut at the boundary
  (the boundary is a game rule), replacing today's fade-out across the rim.
- **No clamp on the log-rate** (see below).
- **Targets built in the learner** from stored privileged truth (the alternative is
  today's: build the field during the rollout and store it).

**Poisson NLL is cross-entropy plus a count term (exact identity).** With per-cell
rates `lambda_c = exp(l_c)`, totals `N = sum y_c`, `Lambda = sum lambda_c`:

    sum_c [lambda_c - y_c log lambda_c]
      = [Lambda - N log Lambda]                       (Poisson NLL of the total count)
      + N * CE(y / N, softmax(l))                     (cross-entropy of the shape)

The shape term is exactly a softmax cross-entropy over cells, weighted by the ship
count; the count term learns the total, unbounded. Gradient per logit
`lambda_c - y_c`, the same form as softmax CE. When a side has no living ships only
the count term acts and drives the rate to zero. Verified numerically. Reporting,
matching the other heads ("KL reported, CE trained"): the shape KL and the count
deviance; their N-weighted sum is the Poisson deviance logged today.

**Clamp.** Not needed. The loss is computed in float32; `exp` overflows only above
88, which only a diverged run reaches (and then NaN is the right signal). Empty cells
drift slowly toward large negative logits with vanishing gradient, which is harmless.
Large overshoots get a large gradient, which global norm clipping bounds. Any clamp
zeroes the gradient beyond it, which is worse than no clamp.

**HL-Gauss on the grid (author's proposal).** The 2D analogue of the 1D codes: each
ship spreads one unit as an isotropic Gaussian with sigma = 0.75 cell spacings,
sampled at cell centres and normalized per ship; the field is the sum over ships.
Barycentric weights are the 2D two-hot, and 3c chose HL-Gauss over two-hot for every
other categorical target, so this makes the family consistent. Measured on the
lattice (spacing = playable radius / 10, as today):

| Target | Cells a ship touches (>10% of peak) | Two ships read as two at | Position bias (interior) |
|---|---:|---:|---:|
| Old Gaussian (removed Sept 30) | ~85 | ~8 spacings | n/a |
| Barycentric (today) | 3 | ~1-2 spacings | 0 |
| HL-Gauss, sigma 0.75 spacing | ~10 | 1.7 spacings | 0 (below 1e-4 spacing) |

So HL-Gauss at 0.75 keeps nearly all of barycentric's resolution (1.7 vs 1-2
spacings, 440 m at 5v5) while giving every cell a smooth, ordinal-aware target. The
Poisson head is unchanged: the targets are still soft counts summing to the living
ship count.

*Edge handling.* Today's grid (hexagon of 10 rings) does not cover the playable
circle: 17% of the circle's area lies outside the hexagon of cell centres, and a ship
there loses the weight of its missing vertices (inside the playable area). Proposal:
cells = every lattice point within `R + 2 sigma` of the centre (a disc, 475 cells at
10 spacings per radius), each in-area ship's weights renormalized over those cells so
it contributes exactly one. That is the 2D version of the padding bins in 3c.

*JAX.* Dense, no scatter: `exp(-|p_i - c_j|^2 / 2 sigma^2)` over ships x cells, mask,
normalize, sum over ships. One einsum-sized computation per sample (10 ships x 475
cells), static shapes, deterministic.

**Present vs next step.** A decision step is 1/60 s; at top speed a ship moves 2.3 m,
under 1% of a 260 m cell, so the field at t+1 is the field at t to any precision
that matters, and a t+1 target adds nothing but deaths and respawns in that step.
Consistency with the next-state head is about form (both are privileged targets
built by the learner), not horizon. A forecast worth having is seconds ahead (a
cell's width at top speed is ~2 s). Since targets are built in the learner from
stored truth, a horizon is just indexing truth at t+k with the usual end-of-episode
and end-of-rollout masks (like shot labels). Proposal: config `density_horizon`,
default 0 (now).

**Pending author confirmation:** HL-Gauss sigma 0.75 on the disc grid with
renormalization; horizon option default 0.
