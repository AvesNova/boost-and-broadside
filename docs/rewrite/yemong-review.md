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
