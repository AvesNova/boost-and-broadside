# JAX rewrite: decision log

Working record of the rewrite discussion, so it survives a lost session. Each item is
**Decided**, **Leaning** (direction agreed, details open), or **Tabled** (deliberately
deferred). Anything not written here has not been decided.

Last updated: 2026-10-16.

## Goals

- **A. A game that is fun to play and watch.** Engaging individual and team strategy,
  polished graphics, live frame rate, playable by humans against trained agents
  (Windows is the priority target; Mac/Android nice to have).
- **B. A training platform.** Easy to train and compare different models (FFN,
  memoryless transformer, next-token predictor, ...) without editing the codebase back
  and forth. Clear configs, fixed contracts between sections, explicit game modes, model
  exchange, possibly a leaderboard. Structured with outside users in mind.
- **C. Very fast GPU-only training.** Fused work, no CPU dispatch bottleneck, long
  rollouts, large parallel batches. CPU only for exceptions like streaming experience to
  host RAM.

Additional motivations: a functional style (easier to reason about and to compile), and
a chance to cut accumulated weight and make the codebase cohesive.

## Evidence that motivated the rewrite

From `docs/engineering/rl-throughput.md` (the existing profile):

- Training is CPU-dispatch bound: ~8,500 kernel launches per rollout step against
  ~148 ms of GPU work, 39-45% GPU occupancy.
- CUDA graphs were implemented and measured as a wash; the idle GPU time sits in the
  eager environment.
- Compiling the environment with `torch.compile` is blocked on correctness: env stages
  write results onto caller-owned objects, and dynamo drops those writes (64 shots
  eager vs 0 compiled). A pure-function environment removes this blocker by design.
- Elo evaluation inside the training loop cost 27-35% of wall clock.

Coupling found in the current code that blocks goal B:

- The env wrapper is `YemongEnvWrapper` and returns a `YemongObservation`.
- Observation composition depends on the policy's belief store, which is fed by the
  next-state head's forecast (env -> obs -> model -> forecast -> obs loop). (The port
  removes this loop; see "Memory and belief".)
- `ppo.py` trains Yemong-specific heads (decomposed value, categorical critic,
  next-state codes, hex density, counterfactual shots) and calls Yemong-specific
  `evaluate_actions` flags.
- `FeatureCoordinator` (feature encoding) lives in `train/rl`, not in the model.

## Decided

### Language and paradigm

- **Pure Python + JAX.** A functional JAX core holds all game rules, physics and agents.
  A thin imperative Python shell handles the window, input, audio, file I/O and
  (later) networking.
- **No game engine and no second implementation.** The game is its own engine. Rust +
  JAX (Rust shell running `jax.export` artifacts via PJRT or IREE) was considered and
  set aside for now. Rust-side runtimes are immature, and Rust would add a second
  language to maintain.
- **Keep every gameplay function pure and exportable** (fixed shapes, no host
  callbacks, plain-array carries and snapshots). This keeps a future shell change, an
  ONNX/StableHLO export, or a Rust client possible without touching the core.
- **Rendering:** pygame for the window, input and audio; ModernGL shaders for graphics
  (instanced draws, glow, trails, particles). Rendering reads immutable render
  snapshots produced by a pure function of state (the pattern already prototyped in
  `ui/gpu_renderer.py` / `ui/gpu_snapshot.py`). Render at display rate and interpolate
  between sim states.
- **Windows play:** JAX on CPU at batch size 1 (30 Hz decisions leave a 33 ms budget;
  the sim + model need a few ms). Packaging is Python + jaxlib via PyInstaller or
  Nuitka. Native-Windows jaxlib CPU wheels are still marked experimental, so **a
  packaged Windows build that plays one match against a trained agent is an early
  milestone, not a late one.**

### Seams (direction decided; exact schemas tabled, see below)

- **Env (rules, pure).** `reset(key, mode) -> state`, `step(state, actions) -> state, events`.
  `observe(state, team)` gives a fixed-schema raw observation, canonicalized per
  team. `privileged(state)` gives truth for training labels and is a separate
  function. The env emits **events** (damage, kills, captures, outcome), not shaped
  rewards.
- **Reward function / game mode (pure).** `events -> rewards`. Reward shaping is a
  training choice, kept outside the env.
- **Agent.** Params plus an **opaque carry** the runtime holds but never inspects.
  - `init_carry`, `step(params, carry, obs, key) -> action, carry, outputs`, and
    `unroll(params, carry0, obs_seq, key)` for training re-evaluation.
  - Generic conformance test for every agent: `unroll == scan(step)`.
  - **Belief in all forms is carried by the model.** Recurrent state and next-state
    belief live in the carry. (Yemong in the port: Griffin state plus, per enemy slot,
    the previous step's forecast mean and sigma for position and attitude, used only
    for damped rotary geometry; see "Memory and belief".)
  - **Feature encoding belongs to the model.** The env provides raw physical
    quantities.
  - Resets and respawns arrive as lifecycle flags in the observation; the agent
    decides what they mean for its carry.
  - Optional `outputs["inspect"]` exposes internals (such as belief) for the dev
    spectator without the runtime knowing their structure. (Yemong: the next-state
    forecasts, residual for visible ships and absolute for hidden ones.)
- **Learner.** Consumes trajectories and owns optimizer state. PPO requires only
  log-prob, entropy and value from an agent. Model-specific auxiliary heads plug in as
  `agent.aux_loss(params, batch, outputs)`. An agent that does not fit PPO comes paired
  with its own learner (explicit in config; no plugin framework).
- **Privileged labels are a separate type**, only reachable from learner batches, so
  they can never be passed to `Agent.step`. Delayed labels (shot outcomes, next-state
  truth) are computed by the trajectory collector.
- **Typed, versioned pytrees** (frozen dataclasses), not loose dicts, for observation,
  action and trajectory schemas.
- **Controllers map to ships, not teams.** "Human flies ship 3, NN flies the rest" or
  "two models on one team" is data. The runtime gathers each ship's action from its
  owning controller.
- **Every controller is an agent:** NN models, the scripted agent, random, and humans
  (a human agent is a pure function of an input array the shell fills from the
  keyboard each frame).

### Raw observation (from the Yemong review, part 1)

- The env emits a **raw, team-canonicalized observation** (self = team 0). A
  model-owned adapter inside the agent builds the model's input; nothing model-specific
  (codes, belief composition, feature transforms) lives in the env or the runtime.
- Raw observation includes bullets even though Yemong does not read them.
- Categorical token type (ship, field, zone, super token) separate from a categorical
  team (two teams for now). Game mode categorical.
- Absolute world positions (models add rotary or other encodings themselves).

### Generality

- **N teams from the start.** Fixed max ships per env with an active mask; team is an
  integer ID per ship; zones are owned by team ID; outcomes are zero-sum across T teams.
  Free-for-all must not require a rewrite.
- **Observation team identity is relative.** "Self team = 0, others relabeled," with a
  permutation-safe relative ID for other teams so a model can tell enemy A from enemy B.
- Counts (ships, zones, fields, bullets) are fixed for a given run; zero-shot transfer
  across counts comes from the model, not from dynamic shapes.

### Game modes

- **Modes:** team deathmatch, king of the hill (single hill), Frontline, and capital
  ship (each team has one zone; whoever takes the enemy zone first wins).
- **Capture rule is an option per mode:** capture by presence in the zone, or capture
  by damaging the zone with guns (a zone becomes an entity with HP, sharing ship hit
  code).
- **Modes are compositions of rule components** (zone set, capture rule, win
  condition, respawn rules), not separate code paths.
- Because one agent may play every mode, the observation carries the objective (zone
  roles and owners, capture progress, scores, time remaining) on the global token.

### Training

- **Loop shape:** jit one update (rollout + learn) and keep a thin Python loop for
  logging, checkpointing and evaluation scheduling. Use buffer donation.
- **Memory sizing:** use XLA's `compile().memory_analysis()` on candidate configs
  instead of porting the PyTorch-allocator VRAM probes. Microbatching becomes a `scan`;
  long BPTT uses `jax.checkpoint` (remat).
- **Precision:** bf16 where safe; fp32 for master params, positions, accumulators,
  recurrent state and anything accumulating small values. Ints/bools where natural.
  JAX disables fp64 and int64 by default; keep fp64 out of the training path (offline
  tools only). Keep the global step counter on the Python side (or uint32 or a pair),
  since 1B+ steps leaves int32 little headroom.
- **Single GPU.** No multi-GPU.
- **BC and RL are separate learners** against the same agent contract. BC can generate
  scripted-expert data inside the jitted rollout. An optional KL penalty toward the
  frozen BC policy, on a step-based decay, can replace today's win-rate-gated BC term.
- **No Elo-gated schedules.** All learning schedules are functions of step count.
  Ratings decide *who you play*, never learning hyperparameters.
- **Evaluation:** heavy evaluation (Elo batches, gradient norms and cosines, etc.) runs
  periodically during training, with an option for more accurate post-training
  evaluation. Prefer blocking every-K-updates evaluation over async (simpler,
  deterministic).
- **Seeding:** one root seed drives everything. Derive streams with
  `fold_in(root, stream_name)` so adding a consumer never shifts other streams.

### Opponents and ratings

- **Only the main scripted agent survives** from the current scripted set.
- **League:** live pseudo-Elo from games that happen naturally during training (live
  agent vs league checkpoints and scripted), plus periodic Elo batches for
  checkpoint-vs-checkpoint ratings. Ratings drive league opponent sampling (consider
  AlphaStar-style prioritized fictitious self-play). League slots are a fixed count of
  stacked param sets; sampling is computed in-graph so it stays deterministic.

### Determinism and debugging

- **Determinism if it is cheap.** Same seed, code, hardware and software version
  should give identical runs, provided it does not cost meaningful speed or
  complexity. Known sources of nondeterminism: atomic scatter-adds and some reductions,
  and XLA autotuning; `--xla_gpu_deterministic_ops=true` fixes both, and its cost
  should be measured once. Prefer dense fixed-shape reductions over scatters where
  natural.
- **Crash reproduction without slowing training:** keep the inputs of the last few
  updates (params, optimizer state, env state, RNG key) in host RAM. On error, dump
  them and replay the single failing update offline with NaN debugging and `checkify`.
  This targets the current "random crash every ~500M steps." Its failure class (CUDA
  error, NaN, hang) is still unknown.

### Configs and tooling

- **Frozen dataclasses for configs.** Lean against Hydra (runtime objects are OmegaConf
  DictConfigs rather than our dataclasses, string overrides, global state). Likely a
  dataclass-driven CLI such as `tyro` plus named presets in Python.
- Static fields (shapes, mode, rollout length) are jit-static. Plain numeric
  parameters (reward weights, learning rates, physics coefficients) can be traced
  arrays, so tuning them avoids recompiles and they can be vmapped.
- **wandb** for logging.
- Drop the current config machinery (fingerprint, drift checks, run-config segments)
  unless a specific need reappears. A frozen config serialized with each checkpoint and
  logged to wandb covers most of it.

### Game scope (v1)

- **Principle: the sim is not redesigned.** Change it only to make it functional, JAX
  and configurable.
- **Kept:** refractive fields; the current perception paradigm; the slip model, g-limit
  and nose dynamics (all made configurable); shields, health and power; respawn delay
  and spawn location rules.
- **Ghost bullets and counterfactual shot labels are training-only**, plus diagnostics
  of trained models (e.g. showing where a real bullet went against where a ship
  predicted it would go).
- **Rules live on the global token.** Respawn rules, mode and other game rules are
  observed through the global/boundary token.
- **Both single-mode and all-modes agents must be supported.** Training one agent for
  one mode and one agent for every mode are both first-class options.

### Physics

- **Ship types:** different ships with different pros and cons in the same match.
  Types can differ in both flight physics and game parameters (e.g. a fast, nimble
  ship with very low health and high damage). Low-level flight constants are derived
  from high-level handling targets. The log-linear solver already exists in
  `config/handling.py`. In JAX the constants become per-ship arrays in state, indexed
  by type.
- **Launch with one type**, possibly a second just to prove heterogeneity works.
- **Per-type, not global:** head-on damage fraction and shield recharge delay can
  differ between ships (in addition to the other combat parameters).
- **Reverse is the exact opposite of boost** (`reverse_thrust = -boost_thrust`). Power is
  an energy store (`E = 0.5 n^2 |v|^2 + power_speed_constant * power`): boost trades
  stored energy for speed and reverse trades speed back into storage, like pitching a
  plane down and up. This removes one free flight constant (16 -> 15).
- **Ship parameters are observable** for all ships, as tuned per type. Idea for later
  (a good prior, not v1): hide enemy parameters and have the agent infer them from
  observed motion, which would need real physical understanding.
- **Parameter randomization during training comes later**, and needs validity checks
  on the free parameters first (see the constraint notes under Tabled).
- **Health model everywhere:** shields plus one exposed-core hit. A hit on a ship that
  entered the tick with zero shields kills it. There is no other health pool
  (`ShipConfig.max_health` goes away).
- **Lives are a configurable rule:** unlimited for Frontline, king of the hill and
  capital ship; no respawns for team deathmatch.
- **Map:** keep a bounded playable arena at a random translation on a large torus (the
  torus exists for the network: translation invariance and uniform location codes). At
  inference, map scale follows ship count (a target density). During training, map
  scale is randomized across the full range, or the coarse position levels go unused.
- **Inventory (fact, from the current code):** the flight model has **16 free
  constants** (`handling.PARAMETERS`; `max_power` is a gauge freedom and held fixed).
  `handling.quantities()` defines 23 closed-form handling quantities of rank 16, so
  exactly 16 independent targets pin a ship's flight. One full-rank set (verified):
  cruise_speed, top_speed, boost_accel, reverse_decel, recharge_time, tank_speed,
  sharp_radius, normal_radius, sharp_sustained_rate, normal_sustained_rate,
  max_turn_rate, onset_time, nose_slew_rate, sharp_gun_offset, fade_speed,
  stall_speed. Combat parameters (collision radius, shield capacity, recharge rate and delay,
  bullet speed, damage, energy cost, cooldown, lifetime, spread, drag, head-on damage
  fraction) are separate from this set.
- **Differentiable physics as far as it is cheap.** Write the continuous flight model
  in naturally smooth forms (use `jnp.where`-safe patterns). Never soften a game rule
  (e.g. bullet hits) for gradients. Use gradients offline only (handling calibration,
  ship-type balance, sensitivity studies), so training speed is unaffected by
  construction.

### Tuning and ship design workflow

- **Port today's values first**, so differential tests against the old physics
  compare like for like. Retuning comes after parity.
- **First planned retune:** halve ship size and gun range, double the other lengths,
  keep speeds (turns 2x slower, gun range ~2.8 turn radii instead of ~11). Numbers in
  `ship-card.md`.
- **Design so rescaling, new ship ideas and training-time randomization are easy**
  (built in from the start, implemented later as needed):
  - Every dimensional config value declares its physical dimension (length, time,
    speed, acceleration, rate, ...), so a similarity rescale (`L`, `T`) is one function
    over the whole config, covering zones, fields, map scale, vision, presence kernel,
    spawn values and bullet drag as well as the ship card.
  - The ship card (high-level handles) is the source of truth; low-level constants are
    always solved, never hand-edited.
  - Per-ship physics and combat parameters are traced arrays in state, not static
    config, so ship types can be mixed in a match, changed without recompiling, and
    sampled per episode.
  - Validity constraints are explicit, so random ship designs can be sampled valid by
    construction.

### Turn levels (decided)

- **Three levels per side, stick thirds:** 1/3, 2/3 and 3/3 of the stall angle,
  constant stick (constant lift coefficient), throttle-independent. **All three are
  in the port.** 1/3 and 3/3 reproduce today's normal and sharp turns, so differential
  tests against the old physics still apply to them; 2/3 is new.
- **Design rule:** every ship's max sustained rate (boosting) sits at 2/3 stick. This
  fixes the drag curve's shape parameter. The shipped ship already satisfies it.
- Turn actions go from 5 to 7 per decision (straight + 3 levels x 2 sides); the joint
  command goes from 30 to 42.
- **One g-limit for all levels**, like a real plane. The hard-coded half-g cap on
  today's normal turn (`0.5 * max_lateral_accel` in `slip_command`) is removed. Parity
  impact: the 1/3 level matches the old normal turn only below ~95 m/s; above that it
  now turns tighter, reaching the full g-limit at ~135 m/s (about top speed).
  Differential tests for that level must stay below ~95 m/s or expect the difference.
- Constant g was rejected (its slow-side equilibrium is unstable). A throttle-dependent
  datum was rejected (under 1% gain). Post-stall remains a later experiment.

### Turn commands: datum design

- **Intent (clarified by the author):** each turn level is a fixed command whose
  equilibrium speed is a datum. Held at its datum speed the ship neither gains nor
  loses speed; faster, it bleeds speed; slower, it gains speed. Normal's datum is the
  max sustained-rate turn; sharp's datum is the min sustained-radius turn.
- **This matches the current mechanism.** Each level asks for a fixed lift; thrust (or
  regenerated power) is fixed while drag grows with speed, so each level has one
  stable equilibrium speed.
- **Sharp already meets the intent.** Max lift is the min radius at any speed below
  corner, and its equilibrium (74 m/s boosting, 40 m/s coasting) is below corner
  speed (100 m/s).
- **Normal does not, today** (computed from the shipped config):
  - Boosting (sustained by regeneration): best sustained rate is ~128 deg/s at slip
    fraction ~0.67 and ~92 m/s. Normal's datum gives ~102 deg/s at ~119 m/s, and the
    sharp datum (114 deg/s) beats it.
  - Coasting (thrust-limited): best is ~79 deg/s at slip fraction ~0.55. Normal gives
    ~70 deg/s.
  - At normal's boosting datum speed the hard-coded half-g cap (active above ~95 m/s)
    is already limiting it, so the closed-form numbers are off there too.
- **Wrinkle:** the max-rate lift depends on throttle. Boosting is power-limited
  (speed ~ C_D^-1/3, optimum maximizes C_L^3 / C_D); coasting is thrust-limited
  (speed ~ C_D^-1/2, optimum maximizes C_L^2 / C_D). One fixed normal lift cannot be
  the datum for both: pick one throttle (boost, like WW2 turn times at combat power),
  or let the normal lift depend on throttle.
- **Consequence for parameters:** normal lift becomes *derived* (the optimum on the
  drag curve), not a free parameter or a global ratio. The half-g cap should go or be
  set by rule. The drag curve shape still needs one parameter, and it must keep the
  optimum strictly inside the stall angle (otherwise normal = sharp).
- **Max deflection (third level):** within the stall angle, sharp already is max
  deflection. A third level only means something past the stall angle (snap shot /
  brake, always bleeding energy, no datum). Optional, after parity.
- **Analysis of the options (shipped config):**
  - *Constant stick vs constant lift:* in this model the lift coefficient depends only
    on the slip, so constant stick offset and constant lift coefficient are the same
    command. The real alternative is constant lift *force* (constant g). Constant g
    has a drag minimum (~100 m/s at half g, ~130 m/s at full g); below it, holding the
    turn loses speed ever faster (the "back side of the power curve"), so its slow-side
    equilibrium is unstable and makes a bad datum. Constant stick gives one stable
    datum, a fixed radius below corner and the g-limit above it, which is how real
    fighters behave.
  - *Throttle-dependent datum:* a single normal lift at slip fraction ~0.61 is within
    ~0.6% of the best sustained rate both boosting (127.0 of 127.7 deg/s) and coasting
    (78.7 of 79.2 deg/s). Throttle dependence buys under 1%.
  - *"Max possible lift" as a third level:* that is already sharp (max lift, g-capped).
    Distinct third levels are a gentle aiming turn, or post-stall (max stick offset).
- **Recommendation (pending the author's choice):** constant stick; throttle-independent
  levels; three levels = gentle (aiming), max sustained rate (normal, compromise lift),
  min radius (sharp, max lift). Post-stall stays an experiment. Joint action space
  becomes 3 x 7 x 2 = 42.
- **Proposal under discussion: stick thirds.** Three levels per side at 1/3, 2/3 and
  3/3 of the stall angle (constant stick, throttle-independent). On the shipped ship:
  - 1/3 is today's normal turn (slip fraction 0.331): r 67 m, 102 deg/s at its boost
    datum (119 m/s).
  - 2/3 lands on the boost max-sustained-rate optimum (127.7 deg/s at 93 m/s; 97.8%
    of the coast optimum): r 42 m.
  - 3/3 is today's sharp turn: r 37 m, 114 deg/s at its boost datum (74 m/s).
  - Thirds of *lift* instead would put the levels at slip 0.18 / 0.42 / 1.0
    (r 111 / 56 / 37 m) and miss the max-rate point.
  - For other ship types the max-rate point need not sit at 2/3. Option: make it a
    design rule ("max sustained rate at 2/3 stick") that fixes the drag-curve shape
    parameter, removing the last free shape constant.
- Making normal the true max-rate datum is a gameplay retune, so it comes after parity.

### Memory and belief (decided 2026-10-21; supersedes parts of review 3b and 3c)

- **One kind of memory: Griffin recurrence only.** Both feedback loops are removed:
  the next-state forecast is no longer decoded into a belief and fed back as hidden
  ships' input, and the enemy-action prediction is no longer fed back as enemies'
  pending-action input. Supersedes "port the single-peak (moment) belief" (3c) and
  "keep feeding the enemy-action prediction back" (3b). Reasons: Griffin already keeps
  per-ship memory per token slot; the loop had no gradient (stored inputs) and could
  compound its own errors (run 734); it removes the decode, re-encode and round-trip
  corrections and most of the carry.
- **Not adopted:** a differentiable belief loop. One step costs about 2x the trunk in
  the update (two passes, both parallel over time), N steps about (N+1)x, all 128 steps
  a sequential update; Griffin gets 128 steps of gradient through memory at 1x because
  its recurrence is linear.
- **Next-state head stays**, as an auxiliary loss only (and for dev spectator display).
  Without feedback, a hidden ship's input code is constant, so a residual head
  (`log(input_code) + f(h)`) is absolute for hidden ships and a small correction for
  visible ones.
- **Next-state head outputs both forms, to see which trains better:** a shared head
  body with two output layers, residual (`log(input_code + eps) + f_res(h)`) and absolute
  (`f_abs(h)`). **Hidden ships are masked from the residual output** (author): the
  residual output trains on allies and visible enemies only; the absolute output trains
  on all three populations. Comparison: residual vs absolute on allies and visible
  enemies; hidden ships use the absolute output (this is option A when read as
  residual-for-visible, absolute-for-hidden, which is also what dev spectator shows).
- Precedent noted: Dreamer's RSSM uses separate posterior (observed) and prior (blind)
  networks and decodes observations absolutely from a bf16 latent; it never predicts
  occluded objects (no privileged labels), which we can.
- **Next-state population weighting: tempered, `weight_p ∝ count_p^alpha`, alpha = 0.5
  (config).** alpha = 1 is today's per-token average, alpha = 0 equal weight per
  population. Per-token amplification of a rare population grows as `1/sqrt(share)`
  instead of `1/share`; an empty population drops out and the rest renormalize. Within a
  population the two outputs split its weight equally (hidden enemies: absolute only).
  Tune alpha with the existing population gradient decomposition (coherent vs noise via
  environment halves). Both outputs feed the trunk; a config option stops either
  output's gradient into the trunk for clean A/B runs.
- **Griffin decay initialization: the paper's**, per-step decay `a^c` in [0.9, 0.999]
  (today's initial time constant is ~0.2 s with the gate open), since Griffin now
  carries all memory.
- **Fog: mixed fog-free and fogged envs**, with a fog flag in the observation; the fogged
  share optionally a step-based schedule; fog is a mode attribute for ratings.
- **Random occlusion augmentation:** post-implementation TODO (not in the port).
- **Critical comparison with the current belief loop (recorded for the post-port
  comparison run).** Gains: one memory, 128 steps of gradient into memory, no compounding
  feedback, far less code and carry. Losses: a hidden ship's position becomes implicit
  in the recurrent state instead of explicit; hidden ships lose geometry in attention
  (rotary encoding, relation bias); memory length must be learned.
- **Geometry for hidden tokens: uncertainty-damped rotary from last step's forecast
  (decided).** The identity rotation means "at world (0, 0)", not "no position", so it
  is not used. Instead:
  - mu and sigma for a hidden ship's position and attitude come from the previous
    step's absolute next-state forecast (decoded moments, with the target smoothing
    sigma_obs and Sheppard's term subtracted, floored at 0);
  - each rotary frequency pair is scaled by `exp(-w^2 sigma^2 / 2)` (exact expectation
    of the score term under a Gaussian position; wrapped normal for attitude). sigma = 0
    is ordinary rotary, large sigma keeps only coarse geometry;
  - the relation bias uses the same (mu, sigma) with closed-form expectations;
  - carry and rollout buffer hold mu and sigma per enemy slot (~5 numbers); no gradient
    through this loop (the update uses the stored values). It is a mild loop: the
    forecast shapes only attention geometry, memory stays in Griffin, the next forecast
    is absolute and trained against truth every step;
  - source is a config choice: `forecast` (default), `mask` (infinite sigma: zero the
    rotated dims and relation bias), `readout` (in-step readout after block 1 used by
    block 2; post-port experiment). Dead reckoning from last-seen state is a cheap
    baseline for the comparison;
  - diagnostic: calibration of the forecast sigma for hidden ships against true error.
  - **Invariant (author):** there are no never-seen ships (every ship is revealed to both
    teams when it spawns), and none are planned. Test it; if it ever changes, `mask` is
    the fallback for such slots.
- **Sweep of consequences (2026-10-21).** Notes added in `yemong-review.md`
  (parts 1, 3, 3b, 3c, HL-Gauss outcomes, 3c revisited, 4a, naming) and
  `diagnostics.md` (groups 7, 8, 9); tabled items 10 and 12 and the agent contract
  updated. What changed:
  - *HL-Gauss codes:* observed-value encoding (sigma_obs = 0.75 bins; padding per the
    rule below, which was only a recommendation when this note was first written)
    unchanged. No belief is encoded into inputs any more,
    so the encode-side quadrature width and the wide-belief padding are moot. The
    decode-side correction stays (forecast moments for the rotary sigma). Tests:
    encode -> decode -> encode stability and zero-residual stationarity are replaced by
    "a sharp target decodes to its mean with sigma ~0"; edge tests stay.
  - *Carry:* Griffin state (RG-LRU plus conv buffer per temporal sublayer) plus the
    previous forecast's mean and sigma for position and attitude per enemy slot. No
    belief store, no action belief, no validity flags.
  - *Enemy pending action:* masked in the raw observation and no longer filled from
    the head; the enemy-action head stays as an auxiliary loss.
  - *Next-state supervision mask:* "has a valid belief" becomes "alive, not terminal,
    contiguous transition" (every ship is seen at spawn).
  - *Diagnostics:* group 7 gains an output slice (residual, absolute) and needs a
    hidden-ship baseline; group 8 now measures the absolute forecast and gains the
    sigma calibration diagnostic; group 9 loses its feedback motivation (its four
    questions remain open).
- **Reopened by the sweep, then decided (author, 2026-10-21):**
  - *Masked categorical encoding: zeros plus the visible flag* (replaces part 1's
    "uniform for now", which assumed belief stayed). Equally expressive with the flag;
    the standard missing-value pattern; implies no graded uncertainty.
  - *Hidden-ship next-state baseline:* group 7 reports only KL for hidden ships; the
    physical-unit comparison against dead reckoning lives in group 8.

### Code padding rule (decided 2026-10-22)

- **Circular codes: no padding** (wrapped Gaussian): attitude, the torus position
  levels, the shot head's travel direction and its displacement code (own torus).
  Recorded in review 3c.
- **Density hex grid: 2 sigma buffer** (367-cell disc, 2.3% worst rim spill). Recorded
  in review 3e.
- **Every other ordered HL-Gauss code: 2 padding bins** per side (2.67 sigma at
  sigma = 0.75 bins, 0.04% spill): observation input codes, next-state targets (same
  codes), shot speed and local-index codes, and **the critic's value bins** (51 symlog
  bins over +/-5, plus 2 per side). Review 3c had recommended 2 bins but left the
  amount "to tune"; now decided.
- Unordered categoricals (actions, outcome classes): no edge, padding does not apply.
- A value beyond the padded range still folds into the last bin; E3 (`critic/support_clip`)
  and `input/envelope_violation_share` measure that.

### Contracts (tabled item 2, in progress from 2026-10-22)

Order: (1) timing, (2) agent interface, (3) action schema, (4) raw observation (items 9
and 10), (5) events, (6) trajectory and labels, (7) versioning.

#### Part 1: timing (decided 2026-10-22)

- **Today (checked):** physics and decisions both run at 30 Hz. `ShipConfig.dt` defaults to
  1/60, but `frontline_ship_config` sets `dt = 1/30` and the RL profile uses
  `action_repeat = 1`. Discounts are per 30 Hz step: gamma 0.99, lambda 0.95.
- **Physics tick: fixed 60 Hz** (not configurable). This is a physics change from today's
  30 Hz and goes into the physics-first work.
- **Decision rate: 30 Hz by default** (repeat 2), set by one config value. The rate must
  divide 60 Hz (60/30/20/15/12/10). A 20 Hz agent is a planned experiment. Agents are
  retrained per rate.
- **Step semantics:** a decision step is `repeat` physics ticks; the action is held for
  every tick, events are summed over the step, the observation is taken after the last
  tick.
- **One clock per match:** every controller, humans included, decides on the same steps.
  The shell keeps the latest input state each frame and the human agent reads it once
  per decision step. This resolves tabled item 11.
- **How settings are expressed (one-change rule):**
  - *Step-native* (integers, often shapes): rollout length (power of 2, 128), minibatch
    and BPTT chunk sizes, conv kernel width, buffer sizes, the repeat. Stay in steps;
    unchanged by the rate (128 steps covers 4.3 s at 30 Hz, 6.4 s at 20 Hz).
  - *Time-native* (game rules): game length, capture time, respawn delay, bullet
    lifetime, cooldown. Written in seconds and converted at resolve time. **Each must be
    an exact multiple of the decision period**, or the config fails (no silent
    rounding), so every rule event lands on a decision step. Cooldown therefore aligns
    with the decision period (0.1 s = 3 steps at 30 Hz; it would have to become a
    multiple of 0.05 s for 20 Hz, e.g. 0.1 s = 2 steps).
  - *Continuous:* gamma and lambda are horizons, converted per step as
    `exp(-dt / tau)`. Defaults keep today's behaviour at 30 Hz: tau_gamma ~ 3.3 s
    (gamma 0.99), tau_lambda ~ 0.65 s (lambda 0.95).
  - Griffin decay init stays at the paper's default (`a^c` in [0.9, 0.999] per step),
    not converted; its memory timescales shift with the rate, which retraining absorbs.
  - Not auto-scaled; retune on a large rate change: entropy coefficient, PPO KL and clip
    ranges, learning rate.
- **Decision period not in the observation** while every run uses a single fixed rate.
  Add it to the global token if mixed-rate training is ever wanted.
- Consequence for tabled item 15: counterfactual shot labels need `action_repeat == 1`
  today; their contract is redefined per decision step (repeat 2).

### Physics changes land in the current env first (planned, not started)

- The flight changes decided here (stick-thirds turn levels, single g-limit, 2/3
  max-rate drag rule, dynamic-pressure two-speed fade, reversal-time handle) are
  implemented in the current PyTorch env (physics, handling solver, action tables)
  before the port. The JAX port is then a pure translation with one-to-one physics
  tests against an env that already has the final semantics.
- Scope: physics, handling solver, action tables, physics tests and flight-envelope
  benchmarks. No backward compatibility: old checkpoints, the old training stack and
  old action layouts may break. No retraining.
- Base: the current working branch, `feat/counterfactual-shot-labels` (not `main`,
  which is behind). `jax-rewrite/design` is that branch plus `docs/rewrite/` only.
- Proposed order, one reviewable step each: (1) stick thirds + single g-limit;
  (2) dynamic-pressure two-speed fade; (3) handling solver: 2/3 drag rule, reversal
  time; (4) re-measure with the flight-envelope harness and update the ship card.
- Added 2026-10-22 (contracts, timing): physics tick from 30 Hz to 60 Hz (`dt = 1/60`)
  with decisions at 30 Hz (repeat 2); time-native rule values must be multiples of the
  decision period.
- **Tick-rate accuracy findings (2026-10-22, not decided).** Scripts in
  `docs/rewrite/physics-checks/`.
  - Today's tick is 30 Hz. The flight update splits forces into two half-steps around
    transport inside one tick (operator splitting); actions, launches, cooldowns,
    collisions and events still happen once per tick.
  - *Flight* (real env physics, no fields, random stick held 0.1 s, against 960 Hz): mean
    position error after 3 s is 11.0 / 7.0 / 3.0 / 1.3 m at 20 / 30 / 60 / 120 Hz (p99
    42 / 25 / 13 / 6 m); heading 3.9 / 2.5 / 1.1 / 0.5 deg. First order (halves with dt).
    Using mid-tick slip for lift and drag helps only ~10%, so the main source is
    elsewhere (not yet found).
  - *Bullet hits:* the swept test sweeps the bullet (start, midpoint, end) but uses the
    target's end-of-tick position. For shots with true closest approach uniform in
    0-20 m (radius 10), hit/miss is wrong for 10.9 / 7.4 / 3.7 / 1.8 % at 20 / 30 / 60 /
    120 Hz (mean distance error 1.9 / 1.4 / 0.7 / 0.4 m). Sweeping the *relative* motion
    (both ends of both bodies, same cost) cuts this to 0.08 / 0.04 / 0.01 / 0.01 %.
  - *Shot-label encounters:* the encounter age is the tick index and the encounter
    snapshot is the bullet's end-of-tick state, up to one tick of bullet travel past the
    contact point (17-23 m at 30 Hz). The swept test's segment parameter gives the exact
    sub-tick time and location for free.
  - **Decided (author, 2026-10-22):** relative swept collision test (sweep bullet minus
    ship, both bodies' start and end). Record the closest-approach time and position from
    the swept parameter (free: the test already computes it).
  - *Error source (2026-10-22, `variants.py`, `flight_var.py`, `flight_sub.py`):* the
    nose. Making the nose instant cuts 30 Hz error 7.0 -> 2.4 m; turn-only schedules carry
    most of the error. The nose is stepped once per tick (exact linear step, then a rate
    clamp) and its end-of-tick slip is held for the whole tick; the force order is also
    not symmetric (thrust, drag, lift | drift | thrust, drag, lift).
  - Restructuring as symmetric Strang splitting (nose + forces over each half tick, the
    second half mirrored, one drift) at no extra cost: 7.0 -> 3.7 m at 30 Hz (60 Hz: 2.9
    -> 1.7 m), close to today's 60 Hz. Substepping the per-ship forces K times inside each
    half (elementwise, no pairwise work, fields untouched): 2.8 m (K=2), 2.5 m (K=4). Still
    first order; the rest is likely discontinuities (power running out mid-tick, passive
    regen added once per tick, stall switching, command evaluated at start-of-tick
    speed).
  - *Energy:* already exact by construction. Drag is an exact decay, lift an exact
    work-free rotation, thrust exchanges exact work with power, and field transport
    projects onto constant proper speed. The error is in trajectory phase, not energy.
    Proposed: a per-tick energy-budget test (change = thrust work - drag loss + regen, to
    round-off).
  - *Representation (2026-10-22, `repr_bench.py`, JAX CPU, 40,960 ships, indicative
    only):* complex64 716 us/step, separate real x/y arrays 627, 2x2 matrices 2525.
    Discussed: complex and real pairs are the same arithmetic (XLA lowers complex to real
    ops); 2D PGA motors unify rotation and translation but cost more per operation, have
    no mature JAX library and gain nothing for point/circle bodies on a torus.
  - *PGA (2026-10-22, `pga.py`, `pga_derive.py`, `repr_bench2.py`):* hand-expanded 2D PGA
    motor helpers (rotor, translator, product, apply to point/direction, normalize) in the
    same struct-of-arrays style. JAX CPU, min of 15: complex64 698, real pairs 555, PGA
    596, PGA with composed motors 585, 2x2 matrices 2194 us/step; all agree to 0.02 m.
  - *Integrator benchmark (2026-10-22, `integrators.py`, `integ_bench.py`, `integ_time.py`;
    real env physics, CPU torch, 1280 ships, random stick held 0.1 s, 3 s, reference =
    symmetric split at 3840 Hz).* Mean position error at 3 s, 30 / 60 Hz:
    - No fields: today 7.0 / 3.2; symmetric split 3.8 / 1.8; + K=2 2.8 / 1.4; + K=4 2.5 /
      1.2; exact combined flow (closed-form speed, Gauss quadrature, power events, regen in
      the flow) K=1 6.4 / 2.9, K=2 2.9 / 1.4, K=4 1.4 / 0.8; with midpoint slip K=2 2.4 /
      1.2, K=4 1.2 / 0.7.
    - Fields on (10 fields, 37% of ship-ticks inside one): today 8.8 / 4.2; today with
      midpoint field integrator 8.0 / 3.7; symmetric split 5.0 / 2.4; + K=2 3.9 / 2.0;
      exact flow K=2 (midpoint slip) 2.9 / 1.5.
    - Every method stays first order, even with the nose rate limit removed: the limit is
      the nose model itself (slip held per piece, arrival snap, speed-dependent command,
      lift kink), not the force integration. The exact flow is only as good as its number
      of nose pieces.
    - Energy (drag and regen off, fields on): every split scheme conserves the generalized
      energy to float32 round-off (|dE|/E ~ 1e-6 after 3 s). The exact flow needs a
      small-drag series (it divides by the drag coefficient; NaN at zero drag here).
    - Cost (CPU eager torch, 10,240 ships, per tick): today 12.6 ms, split 26, split K=2 38,
      exact flow K=1 180, K=2 361; whole env tick 181 ms with fields. Eager CPU counts
      every op as a memory pass; in a fused JAX kernel the per-ship arithmetic is small
      next to pairwise collisions and fields. The exact flow as written does ~2,000
      transcendentals per ship-tick (40 bisection steps; Newton would cut this).
  - *Integrator-friendly flight design (2026-10-22, `nose_alt.py`; each model against its
    own 3840 Hz reference).* Three first-order sources found: the nose (rate clamp,
    arrival snap, slip held per piece), the turn command depending on speed through the
    lift cap (evaluated once per piece), and hard power limits (empty store, full store
    on reverse, regen once per tick). Removing the lift cap alone: 1.1 / 0.45 / 0.14 m at
    30 / 60 / 120 Hz.
    - Smoother nose alone (linear ramp at the slip-rate limit, or a tanh soft rate limit)
      stays first order (tanh 3.4 / 1.5 / 0.6 m).
    - **g-command design** (all three made smooth): the stick commands a lateral-g
      fraction q (sharp 1, normal 0.5) that follows the command through a tanh soft rate
      limit (same full-swing reversal time, small-signal time constant 0.08 s); lateral
      acceleration = smooth min of q x max_lateral_accel and the stall cap C_Lmax u^2;
      slip (gun offset, drag) derived algebraically from the lift curve; thrust fades
      over the last 2% of the power store (and reverse recovery over the top 2%); regen
      inside each half step. Symmetric split, no fields: 0.53 / 0.20 / 0.07 m (K=1), 0.21
      / 0.07 / 0.05 m (K=2), heading 0.06 deg at 30 Hz; about second order. With 10
      fields: 0.54 / 0.20 / 0.07 m (K=1). Energy with drag and regen off: 5e-7.
      Cost (CPU eager): 24.6 ms per tick (K=1) vs today 9.6 and split 17.4.
    - Gameplay changes to check: slip follows speed instantly (no lag on speed changes);
      soft stall cap near corner speed; boost fades in the last 2% of power; the
      slow-speed command fade is not modelled; handling numbers need re-solving (drag loss
      in the test was ~9% vs ~2% today with the same coefficients).
    - **Decided (author, 2026-10-22): soft limits** on power (fade over a band at empty and
      full) and on the g limit (smooth min with the stall cap). The nose redesign itself is
      still open.
  - *Helper size (2026-10-22, `xy.py` vs `pga.py`):* 14 vs 23 non-blank lines, 6 functions
    each; call sites are about the same length.
  - Proposed (pending the author): symmetric splitting with the nose stepped per half;
    optional K force substeps; event-exact power cutoff; re-measure with fields (Frontline
    uses the first-order `two_step` field integrator with 1 substep).
- Status: planning only. No implementation until the author says so.

### Presentation

- Polished graphics, spectator mode, dev spectator mode (privileged truth plus agent
  `inspect` outputs), and replays.
- **Replays record state keyframes plus actions**, not only seed plus actions. GPU
  recording vs CPU playback diverges numerically in a chaotic sim; keyframes bound
  drift and make seeking trivial.

### Ecosystem

- **Native batched-array API** as the core, plus a thin **JaxMARL adapter with teams as
  agents**. Checked Oct 2026: JaxMARL is maintained (0.2.0 released 2026-09-10, ~857
  stars). Its API is dicts keyed by agent name, and SMAX is its closest sibling
  environment.

### Repository

- **Design branch:** `jax-rewrite/design` holds the old code plus `docs/rewrite/`, so
  design discussions have full context. Specs and contracts are drafted here.
- **Option 3:** same repo, fresh start on a branch.
  - Tag the current state (proposed tag: `v1-torch-final`) on the working branch
    (`feat/counterfactual-shot-labels`, after the physics-first changes), not `main`.
  - Start a `v2` branch with the old tree removed.
  - Merge to `main` once v2 is playable. Note: `main` is behind the working branch
    today.
  - The old code stays reachable via the tag / a `git worktree`, granted per milestone.
- Move checkpoints and artifacts out of git (wandb artifacts or the Hugging Face Hub).
- The top-level layout should mirror the seams: contracts/types, env (physics, rules,
  modes), agents (scripted, human, models), learners (PPO, BC), league/eval, and the
  play shell (render, input, replay, spectator).

### Old code and docs

- **Old docs are not a spec.** Many are stale; treat them as a source of ideas only.
- **Old code is ground truth only for physics behavior** (useful for differential
  tests). The training stack, wrapper and evaluation stack should not be ported.
- The new spec is written fresh and owned by the project author.

## Tabled (deliberately deferred)

1. **Remaining flight-parameterization questions** (the card itself is in
   `docs/rewrite/ship-card.md`):
   - **Stall angle (decided):** shared default (15 deg today), first candidate to
     promote to a per-type handle. It does not change radius, rates, drag or datums;
     it sets the gun offset in a turn (5 / 10 / 15 deg at stick thirds) and how far
     the nose travels in a reversal.
   - **Low-speed control (decided):** two speeds, `stall_speed` (no control) and
     `full_control_speed` (full control), faded by **dynamic pressure**:
     `clamp((u^2 - stall^2) / (full^2 - stall^2), 0, 1)`. `min_speed` stays a global
     numerical floor. Retune target from the P-51 comparison: stall ~0.22 x top speed
     (~30 m/s), full control ~1.3 x stall (~39 m/s). Note: no setting reproduces
     today's linear fade exactly.
   - **Nose response (decided):** per-type handle is **reversal time** (the roll-rate
     analog), which sets the slew limit (`max_slip_rate`); the card defines it as
     `2 * stall_angle / max_slip_rate` and shows the simulated value. The spring
     frequency (onset) becomes a shared, defaulted parameter that a ship may override.
     WW2 basis: fighters differed most in roll rate (at ~250 mph roughly Fw 190
     ~150-160 deg/s, P-47 ~85-100, P-51 ~90, Bf 109 ~75-80, Spitfire V ~40-65; forum
     citations of NACA 868 / RAE, approximate). Today: onset 0.167 s, reversal 0.233 s
     simulated (0.20 s by the card definition). Later idea: speed-dependent roll rate.
   - Confirm the validity constraints and the log-gap parameterization for sampling.
   - Pick the similarity-ratio suite shown on the card (current draft: 10 ratios).
2. **Concrete contracts:** observation schema, action schema (42-way joint command,
   accepted in review part 1; this item used to ask "keep the 30-way joint command?"),
   decision rate and action repeat, event list, trajectory/label format, and versioning
   rules.
3. **Leaderboard and model exchange:** submission format (exported `act` via
   `jax.export`?), rating method, fixed baselines, how a submission pins a mode version.
   Needs real thought.
4. **Testing strategy:** contract and conformance tests, differential tests against
   old physics, performance budgets, who writes and approves which tests.
5. **Agent workflow:** milestone granularity, review protocol, files that agents may
   not change without asking (specs, contracts, tests), per-milestone access to old
   code, keeping testing/benchmark effort proportionate. Key constraint from the author:
   no silent changes to the codebase; work step by step without babysitting every
   decision.
6. **JAX library deep dive (discussed 2026-10-21, re-tabled by the author):** NN
   library (Flax NNX vs Linen vs Equinox), optax, checkpointing (Orbax vs plain pytree
   save). Findings so far:
   - Versions checked 2026-10-20: jax 0.12.0, flax 0.12.10, equinox 0.13.8, optax 0.2.8,
     orbax-checkpoint 0.12.7. NNX API still changing in 2026 (deprecations and renames in
     0.12.1, 0.12.7, 0.12.8; tree mode added in 0.12.5-0.12.6); pin versions.
   - Mockups of the Griffin temporal block in plain JAX, Linen, NNX and Equinox:
     `docs/rewrite/library-mockups/` (all four agree numerically, rollout equals
     sequence, stacked league slots, grad + optax, export).
   - In flax 0.12.10 NNX modules are pytrees: plain `jax.jit`/`grad`/`vmap` and optax take
     them directly (no split/merge needed); in-module mutation inside plain `jax.jit` is
     dropped (Yemong has none).
   - `jax.export` serialization needs params in plain containers; native NNX and Equinox
     params fail until converted.
   - Proposed, not decided: rule A (models are pure functions over an opaque params
     pytree; "any model" means any architecture, one library for all our models) and
     rule B (plain dicts only when saving and exporting); optax; Orbax vs plain
     safetensors + JSON checkpoints; `jaxtyping`. Assistant's last lean: NNX with plain
     JAX transforms, pinned.
8. **In-depth review of the current Yemong model and training.** Walk through the
   policy architecture (encoder, spatial attention, Griffin recurrence, heads,
   belief, categorical codes) and the training stack (PPO, critic, auxiliary losses,
   BC, opponents, rating, rollout and update loop) piece by piece, so the author
   understands everything and incorrect or hidden assumptions surface before the new
   model and learner contracts are written. Review the code itself, not just the
   docs, since many docs are stale.
9. **Ship type and ship-card parameters in the observation:** how ship types and their
   (observable) parameters are presented in the raw observation and to models.
10. **Hidden-entity convention in the raw observation:** what an unseen enemy's slot
    contains (mask only, or last-seen values and time since seen), given that belief
    moves into models and may be replaced by Griffin memory.
    *Update 2026-10-21:* Yemong no longer needs last-seen values (Griffin-only memory;
    "Memory and belief"). Already decided in part 1: a mask, a visible flag always, and
    time since last seen as a feature; every ship is revealed at spawn, so there are no
    never-seen slots. Left for contracts (item 2): whether the raw observation carries
    last-seen values at all (lean: no; models remember for themselves; the dead-reckoning
    baseline can be computed from stored observations by diagnostics).
11. ~~**Human input timing:**~~ decided 2026-10-22 (contracts, timing: one clock per
    match; the human agent reads the latest input once per decision step). Humans may act more often than the decision rate; define how
    their inputs map onto decision steps (for other agents' pending-action views and
    for training labels if human games are ever learned from).
12. ~~Unfinished from the Yemong review 3c~~: decided. ~~Single-peak belief for the port;~~
    (superseded 2026-10-21: no belief loop, see "Memory and belief");
    codes in absolute units with global maximums from the design envelope (arbitrary
    parameters within reason); next-state loss on allies, visible and hidden enemies,
    averaged per population.
13. **Reward weights, zero-sum and the return-scale details:** changed on a branch not
    visible here (recalibrated weights, simple zero-sum env). Revisit with access.
14. ~~Naming convention~~: decided (ship, team, controller, lineup, view, carry, model /
    params / checkpoint, learner, match / decision step / tick; "agent" only in the JaxMARL
    adapter, where an agent is a team). See the review, 3d.
15. **Counterfactual shot labels (review 3f), pending benchmarks:** env ghosts in the
    shared bullet buffer (current decision) vs learner-side re-simulation from stored
    truth (labels every decision, pure env); HL-Gauss trajectory codes; per-type
    collision radius in hits and near-miss scale; end-of-rollout label loss (23% at 128
    steps); damage in the outcome. Port the heads as they are until benchmarked.
    Added 2026-10-22: labels must be defined per decision step (2 physics ticks at the
    decided 60 Hz physics / 30 Hz decisions); today they require `action_repeat == 1`.
    **Decided (author, 2026-10-22): learner-side re-simulation.** No ghosts in the env
    (pure gameplay, plain bullet ring buffer). The learner re-flies a counterfactual shot
    from every stored decision with the same pure bullet functions, against stored ship
    states (interpolated between decisions, using stored velocities), with the relative
    swept test. Exact closest-approach time and position; sample times continuous (exact
    closed-form drag flight without fields, interpolation through start/mid/end with
    fields). About 3x today's label coverage; estimated cost well under 1% of an update
    (~1.6e9 swept tests and field evaluations per 128 x 4096 x 10 update vs ~1e14 model
    flops). Still open from this item: HL-Gauss trajectory codes, per-type radius,
    end-of-rollout loss, damage in the outcome.
16. **Privileged (fog-free) critic, after the port:** V(h, s) design (view features plus a
    fog-free summary), either a feedforward encoder or the same trunk on fog-free input
    (needs its own carry: a full second trunk pass). Values computed in the learner.
    See the review, 4a discussion.
7. ~~One model for all modes, or one per mode~~: decided, both supported (see Game
   scope).

## Units (decided)

- **World units are metres.** Every quantity in the simulation, the contracts, the
  model's physical codes, configs and diagnostics is in SI-style units: m, m/s, m/s^2,
  rad, rad/s, s. "px" in today's code is a legacy name for a world unit (1 px = 1 m
  numerically, so no value changes).
- **Pixels exist only in the renderer**, inside its camera transform (world metres to
  screen pixels, with zoom, window size and fullscreen as camera parameters; today's
  renderer already has such a scale). Nothing outside the renderer may depend on screen
  size.
- Renames in the port, for example `position_px` -> `position_m`,
  `velocity_px_s` -> `velocity_m_s`, `boundary_damage_per_pixel_second` ->
  `boundary_damage_per_m_s`, "px/s" in config comments -> "m/s".

## Post-implementation TODO

Checks and revisits deliberately deferred until the JAX port runs.

- [ ] Carry-staleness diagnostic (burn-in check, review 4d): policy KL and value gap
      between stored and rebuilt carries against step index.
- [ ] League slots: revisit fused opponents + drain (A1 + B1) and the slot count with
      profiles (review 4b).
- [ ] Verify the env step and network forward overlap in the profile (review 4a).
- [ ] Shot-label design benchmarks (tabled item 15).
- [ ] Privileged critic experiment (tabled item 16).
- [ ] Belief experiments, e.g. direct code feedback (review 3c). Note: the belief loop
      is removed in the port (Griffin-only memory); compare against it after the port.
- [ ] Random occlusion augmentation (randomly hide visible enemies for a few steps) as an
      optional knob, if hidden-ship learning looks data-starved.

## Next steps (resume point)

Design discussion is on branch `jax-rewrite/design`. Documents:
`docs/rewrite/decisions.md` (this file), `docs/rewrite/ship-card.md`,
`docs/rewrite/yemong-review.md` (the model and training review: parts 1-3 model,
part 4 training 4a-4e), `docs/rewrite/diagnostics.md` (diagnostics audit).

- [x] Flight parameterization (only constraints and the ratio suite left to confirm).
- [x] Yemong review part 1 (inputs), part 2 (trunk), part 3a-3d (action, enemy-action,
      next-state/belief, critic).
- [x] Yemong review 3e: hex density head (decided: HL-Gauss, 367-cell disc, 2 sigma buffer,
      Poisson head, no clamp, ally/enemy, learner-built targets, k = 0).
- [x] Yemong review 3f: counterfactual shot heads (reviewed; design tabled as item 15
      pending benchmarks).
- [x] Naming convention confirmed; team spirit kept as config, default 0.
- [x] Training review 4a: rollout loop (decided: train every learner view, drop
      `shared_pass`, drop multi-scale, env pools, env step and forward in parallel).
- [x] Training review 4b: opponents and league (decided: no dedicated evaluator, ratings
      from live league games via periodic joint fit, time-based snapshots, milestones only
      after training, one pool with p(1-p) + uniform floor, self-play fixed share, avg and
      rungs kept, A1 + B1 slots revisited after implementation, periodic diagnostics).
- [x] Training review 4c: BC (decided: scripted agent in JAX with 42 actions, BC alone then
      natural RL from BC weights, BC rollouts use the pool, entropy through teacher targets,
      no fixed BC cutoff yet).
- [x] Training review 4d: PPO update mechanics (decided: wide envs with observation chunks
      streamed to host while the next chunk collects, small per-step arrays and GAE on
      device, buffer residency as config; non-finite steps skipped, counted, dumped for
      replay, abort on a budget; entropy E1 fixed coefficient; SIGReg dropped; clip and
      weight retuning tabled to the other branch).
- [x] Evaluation 4e (decided: offline BT pipeline kept, torch parts rewritten on the JAX
      match engine; leaderboard keyed by game version x mode x fleet size, scripted at
      1000; diagnostics every 10 updates).
- [x] **Diagnostics audit (complete 2026-10-22, `docs/rewrite/diagnostics.md`):**
      generic BC/RL checklist; method rules (avoid syncs, raw values only, no
      expected-behaviour pruning, "derivable" is not a drop reason, quantity-first
      naming); groups 1-20; checklist items without a group (`repr/`, `input/`, `bc/`,
      batch composition, tests for A6/A8/A9); final trigger rules (8 alerts, 13
      warnings, dashboard only, only the skipped-step rule acts); three dashboard views
      (12-panel headline, ~141-panel curated, automatic everything). Threshold tuning is
      a post-implementation TODO.
- [ ] JAX library choice (tabled item 6): discussed, re-tabled by the author; findings
      recorded under item 6.
- [x] Memory and belief redesign (2026-10-21): Griffin-only memory, dual-output
      next-state head, tempered weights, damped rotary from last step's forecast, mixed
      fog; consequences swept through the docs. Reopened points decided: masked values
      are zeros plus the visible flag; hidden ships report KL only in group 7.
- [ ] Revisit review parts 1-2 at the deeper level used from part 3 on.
- [ ] **Next (author, 2026-10-22): contracts** (tabled item 2, absorbing 9, 10, 11).
      Inputs from the memory redesign: opaque carry with `init_carry` per model (Yemong:
      Griffin state plus last forecast mean/sigma per enemy slot), optional auxiliary
      losses via `agent.aux_loss`, masked values as zeros plus visible flag, enemy
      pending action masked, whether the raw observation carries last-seen values (lean
      no), fog flag in the observation, 42-way joint action, code padding rule.
      Progress: part 1 timing [x] (2026-10-22); next part 2, agent interface.
- [ ] Then: agent workflow, then the remaining tabled items.
- [ ] Physics-first changes in the PyTorch env (planned, not started).
- [ ] Repo setup (option 3), after the physics-first changes.
