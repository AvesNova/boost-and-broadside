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
- *Superseded 2026-10-23 by `flight-model.md` (boost/reverse are dive/climb energy
  exchange commands; the engine is separate, constant power, along the nose; the store
  is in J).* **Reverse is the exact opposite of boost** (`reverse_thrust = -boost_thrust`). Power is
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
- *Superseded 2026-10-23 for the new model (new parameters listed in
  `flight-model.md`, Open 2).* **Inventory (fact, from the current code):** the flight
  model has **16 free constants** (`handling.PARAMETERS`; `max_power` is a gauge freedom and held fixed).
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

*Under review 2026-10-23:* the new flight model's prototype uses thirds of the lift
budget, not of the stall angle; see `flight-model.md`, Open 1.

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
  **Superseded (author, 2026-10-22):** with the new flight model (rigid-body nose,
  exponential kick, exact arc; ~100x more accurate per step) the physics tick equals the
  decision tick (repeat 1): 30 Hz now, 20 Hz for a 20 Hz agent. Time-native rule values
  are multiples of the decision period, as before.
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

**Updated 2026-10-23: the new flight model replaces the earlier change list.** The
specification is `docs/rewrite/flight-model.md` (prototype `physics-checks/flight8.py`,
checks `final_check.py`). Author: induced drag x3 adopted; fold the model into this plan.

- Scope: implement the proposed flight model in the current PyTorch env (physics,
  actions, energy store in J, exchange commands, engine, lift budget, rigid-body nose,
  exponential-kick integrator at the decision tick, curved-space fields, relative swept
  hits, learner-side shot labels, bullet cost), then the handling solver and ship card.
  No backward compatibility: old checkpoints, the old training stack's assumptions and
  old action layouts may break. No retraining in this step.
- Base: `feat/counterfactual-shot-labels` (not `main`). `jax-rewrite/design` is that
  branch plus `docs/rewrite/` only.
- Proposed order, one reviewable step each: (1) state and units (1 kg ship, energy store
  in J, map-local frame without wrap, physics tick = decision tick); (2) flight model and
  integrator (lift budget, nose, engine, exchange, drag), checked against the prototype;
  (3) fields as curved space, with the env's union composition; (4) bullets: relative
  swept hits, cost per bullet, proposed ballistics; shot labels re-simulated by the
  learner; (5) handling solver and ship card rewritten for the new parameters; (6)
  re-measure with the flight-envelope harness.
- Open before step 2: the turn-level definition (flight-model.md, Open 1). Open for the
  plan: implement in PyTorch first (as decided earlier) or directly in JAX with the
  prototype checks as the specification (flight-model.md, Open 5).
- Status: planning only. No implementation until the author says so.
- *Superseded list (2026-10-22, kept for history):* stick thirds + single g-limit;
  dynamic-pressure two-speed fade; handling solver 2/3 drag rule and reversal time;
  re-measure. Earlier text follows.
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
  - *2D rigid-body prototype (2026-10-22, `rigid2d.py`, `rigid2d_bench.py`,
    `rigid2d_cost.py`, `today_jax.py`; JAX, no fields, 2048 ships, random stick and power
    held 0.1 s, 2 s, reference = screw scheme at 3840 Hz float64).* The nose is a rigid
    body in the plane: `r' = w(u)^2 (alpha_cmd - alpha) - 2 zeta w(u) alpha'`, `w(u)` =
    12.3 rad/s x u / u_corner, zeta 0.7; stick levels ask for a lift fraction, a soft g
    limit caps it, alpha_cmd inverts today's lift curve; drag polar `C_D0 + k C_L^2`;
    thrust along the nose; soft power limits; no roll axis.
    - Integrators: (a) x/y split (exact nose step with the path frozen, then path
      forces with the nose frozen): 7.1 / 4.5 / 2.5 m at 30 / 60 / 120 Hz; the nose and
      the path are stiffly coupled, so splitting them is poor. (b) screw scheme (RK2 kick
      of the body-frame twist: nose rate and body velocity together; then an exact screw
      motion of the pose; then the second kick), written in x/y and in 2D PGA: identical
      results (3e-11 m apart), **0.055 / 0.009 / 0.002 m**, heading 0.11 deg at 30 Hz.
      About 70x more accurate than today at 30 Hz and better than second order.
    - Energy (drag and regen off, 30 Hz): 1e-4 for the screw scheme (the RK2 kick is not
      exactly work-free for lift; fixable), 2e-3 for the split.
    - Cost (JAX CPU, f32, 40,960 ships, per step, after replacing a pow and a cos):
      screw in x/y **6.8 ms**, screw in PGA 9.4 ms (extracting, wrapping and rebuilding the
      motor each step), split x/y 13.4 ms, **today's model ported to JAX 7.6 ms**.
    - Feel (constant speed, full stick): corner speed 100 m/s; turn rate 62 / 92 / 130 /
      109 / 86 deg/s and radius 37 / 37 / 44 / 74 / 120 m at 40 / 60 / 100 / 140 / 180
      m/s; reversal (90%) 0.54 / 0.36 / 0.21 / 0.15 / 0.12 s (0.21 s at corner speed,
      today's value); overshoot 4.6% (a brief accelerated stall at low speed).
    - Bug found while testing: an angle-wrap error in the first x/y version (fixed); PGA
      sign conventions also needed a cross-check earlier.
    - *Wrap (2026-10-22, `wrap_cost.py`, per step, 40,960 ships):* the earlier PGA wrap was
      a naive extract-and-rebuild. Wrapping by composing a translator, `M' = T(-kW) M`,
      changes only the translation components (a few multiply-adds; checked). Without
      wrap / with wrap: today (JAX port) 7.7 / 8.8 ms; rigid body x/y 6.5 / 6.7 ms;
      rigid body PGA 7.4 / 9.3 ms naive, **7.2 ms with the translator wrap**. PGA and x/y
      are within ~10%.
  - **Decided (author, 2026-10-22): no wrapping in gameplay.** Physics runs in a
    map-local frame (map centre at 0) with no wrap or minimum-image arithmetic; the
    per-episode random offset (and orientation, handedness) is applied only when building
    observations, so models still see uniform absolute positions. Contracts part 4 states
    how positions are reported.
  - *Map origin (author's idea, 2026-10-22; decided above):* today the map centre is
    already drawn uniformly on the torus per episode (`frontline.py`, plus random
    orientation and handedness). Proposal: run physics in a map-local frame (centre at 0)
    and add the per-episode offset only when building observations, so models still see
    uniform absolute positions while physics keeps small coordinates (float32 spacing
    0.5 mm at 4 km vs 7.8 mm at 65 km) and, if the torus is never gameplay, drops wrapping
    and minimum-image arithmetic (including in pairwise collisions and visibility).
  - *Rigid body v2 with fields (2026-10-22, `rigid2d_v2.py`, `rigid2d_v2_bench.py`,
    `v2_cost.py`).* Clean rewrite: forces computed in the body frame (no world round
    trips), identical math in x/y and PGA (agree to 1e-9 m).
    - **Fields fit as curved space:** n(x) gives the metric n^2 I. The ship flies the
      unchanged flight model in proper time (d tau = dt/n) with proper velocity w = n v
      (this is exactly today's model: thrust / n^2, lift and drag on world speed, nose
      frequency / n); the field adds only geodesic bending of w at rate
      `|v| (v_hat x grad ln n)`, which keeps |w| (generalized energy) fixed. In the
      kick it is one extra rotation rate; the drift uses n at the step midpoint from
      `ln n + 1/2 grad ln n . displacement` (free). One field evaluation per step.
    - **Exponential kick:** the fast nose / angle-of-attack oscillator is solved exactly
      (closed-form 2x2 step, lift slope linearized, coefficients at the predicted
      midpoint speed); the velocity direction is set exactly so lift and bending do no
      work; speed and power by RK2. Needed because the RK2 kick goes unstable when the
      nose frequency per step is large (high proper speed in a low-index region).
    - Design rule added: reverse thrust fades out smoothly between 30 and 10 m/s (today
      disables it below `min_speed`); without it a ship can reverse to zero speed, where
      its direction is undefined.
    - Accuracy (2 s, 2048 ships, 24 fields, 27% of ships ending inside one), 30 / 60 /
      120 Hz: no fields 0.087 / 0.022 / 0.005 m, heading 0.03 deg at 30 Hz; with fields
      0.36 / 0.08 / 0.02 m (p99 4.5 m at 30 Hz, near-grazing refraction). Second order.
      Energy (drag and regen off, 30 Hz): 7e-6 without fields, 2e-5 with fields.
    - Cost (JAX CPU, f32, 40,960 ships): x/y 12.5 ms, PGA 12.9 ms without fields; 12.9
      / 13.6 ms with 10 fields (PGA within 3-5% of x/y). Today's model (JAX port, no
      fields) 7.9 ms: the exponential kick costs ~1.6x on CPU (both cos and cosh branches
      evaluated, an atan2 and a sqrt per kick). Possible trim: keep the body velocity in
      polar form (speed, alpha) as state, removing the atan2 and sqrt.
  - *Rigid body v3 (2026-10-22, `rigid2d_v3.py`, `v3_bench.py`, `climb_table.py`).*
    - **Author: boost and reverse stand for diving and climbing in gravity** (the power
      store is altitude; today's code mostly did this). Reverse must not work when
      stalled, like turning near stall. Prototype: thrust acts along the flight path
      (velocity) rather than the nose (switch `THRUST_ALONG`); reverse (climb) draws on
      the same lift budget `L(u) = softmin(C_Lmax u^2, a_max)` as turning, turn first:
      `climb <= softmin(|T_rev|, L sqrt(1 - (a_lat/L)^2))`. Replaces the 10-30 m/s reverse
      fade. Induced drag from climb lift is an option (`CLIMB_DRAG`, off).
      Climb available (m/s^2, reverse 80) straight / 2/3 stick / full stick: 10 m/s 2.7 /
      2.0 / 0; 30 m/s 24 / 18 / 0; 60 m/s 73 / 63 / 0; 100 m/s 80 / 78 / 0; 180 m/s 80 /
      36 / 0. A full-stick turn uses the whole budget, so no climb while turning hard.
    - **Storage variants, same physics** (agree to 1e-10 m, same accuracy: 0.084 / 0.021
      / 0.005 m without fields, 0.356 / 0.081 / 0.021 m with fields at 30 / 60 / 120 Hz).
      Cost (JAX CPU, 40,960 ships, ms per step, 0 / 10 fields): heading (cos, sin) +
      velocity as (speed, angle of attack) **7.2 / 9.2**; PGA motor + (speed, angle of
      attack) **7.3 / 9.7**; heading angle + (speed, angle of attack) 10.7 / 11.4; heading
      (cos, sin) + cartesian velocity 17.2 / 16.0. Today's model (no fields) 7.9.
  - **Decided (author, 2026-10-22):** engine thrust along the nose; dive and climb along
    the flight path; one lift vector shared by turning and climbing/diving, coupled
    (p = 2, no knob); symmetric elevator (any direction), asymmetry only from energy.
  - *v4 prototype (2026-10-22, `flight4.py` shared scalar physics, `ship_xy.py` complex
    paradigm, `ship_pga.py` PGA paradigm: motor pose, body twist bivector, rotor and motor
    sandwiches, motor exponential; `v4_bench.py`, `v4_check.py`).* Controls: stick sets the
    lateral lift fraction; coast holds level, boost pushes, reverse pulls; the commanded
    vector is scaled into the budget along its direction. Holding level costs `g_hold`
    (assumed 27 = 10% of a_max, flagged); exchange `a_x = k (g_hold - L_vert)` with k =
    0.3; engine 8 m/s^2 along the nose, free; induced drag on the whole lift vector.
    - Agreement x/y vs PGA 1e-10 (no fields) / 8e-10 m (fields). Accuracy 30 / 60 / 120
      Hz: 0.039 / 0.010 / 0.003 m without fields; 0.39 / 0.14 / 0.04 m with fields.
      Energy drift (drag, regen, engine off): 4e-6 / 7e-6.
    - Cost (JAX CPU, 40,960 ships): x/y 12.5-14.1 ms, PGA 12.8-14.5 ms (within noise of
      each other); today's model 7.9 ms. About 3 ms is the atan2 for the angle of attack
      (removed by storing velocity as speed and angle, as in v3).
    - Feel (single ship, no fields): level-flight minimum speed 31.6 m/s; cruise (coast)
      106 m/s; coast full stick settles at 56 m/s, 72 deg/s; boost straight 128 m/s (store
      empty); reverse straight 46 m/s (store full). Stall from 25 m/s recovers to 40 m/s in
      2.7 s, and the store still gains (regen 10/s outweighs the gravity sink of ~0.4/s);
      with an empty store, engine alone reaches 41 m/s in 3 s; a full-stick turn at 45 m/s
      does not spiral down. With today's regen there is no low-and-slow trap; making stall
      bite needs less regen or a larger g_hold or k.
  - **Decided (author, 2026-10-22):** continue with x/y only for now; store state in
    whatever form is fastest. Use SI terminology. Regen and the store size are arbitrary
    for now.
  - *SI and P-51D scale (2026-10-22, `p51_scale.py`).* The game is in metres and seconds,
    per unit mass: forces N/kg (= m/s^2), the "power" store is specific energy (J/kg,
    20,000 J/kg max = psc x 100), regen is specific power (2,000 W/kg). Taking the game's
    level-hold lift (27 m/s^2) as 1 g and the minimum level speed (31.6 m/s) as the
    P-51D's estimated stall (47 m/s, C_Lmax ~1.5 assumed) gives speed x1.48, acceleration
    x0.363, **time x4.1, length x6.1**: the game runs ~4x faster and is ~6x smaller.
    Scaled game vs P-51D (approximate references, several from forums): cruise 157 vs 162
    m/s; corner 149 vs ~134 m/s; max lateral 10 g vs 8 g; instantaneous turn 32 vs ~34
    deg/s; sustained turn 18 vs ~18 deg/s; corner radius 269 vs ~231 m; lift per metre
    4.4e-3 vs 4.5e-3. Far off: regen as climb 110 vs 16 m/s (~7x), dive/climb exchange
    2.8 g / 2.2 g along the path vs at most 1 g, drag per metre 2.7x, engine accel ~2.4x
    (balancing to the same speeds), bullet range 3 km vs ~0.3-0.7 km, collision radius 61
    m vs a 5.6 m half-span, firing cooldown 0.41 s vs ~80 rounds/s for six guns.
  - *v5 retune (2026-10-22, `flight5.py`, `v5_bench.py`; x/y only, velocity stored as
    speed and angle of attack).* **Author: ship mass 1 kg; an engine with constant power
    split between thrust and charging the energy store; retune energy, power, bullets and
    hitbox roughly toward a P-51 at game scale; the fire rate must be a multiple of the
    decision rate.** Proposed values (game units, SI): g 27 m/s^2; C_Lmax 0.027 1/m (stall
    31.6 m/s); a_max 216 m/s^2 = 8 g (corner 89 m/s); cd0 2.9e-4 1/m; k_ind 3.8; engine
    400 W = 320 W thrust (force P/u, capped at 10 N) + 80 W store; store 10 kJ (370 m game
    altitude, ~2.2 km real); dive/climb exchange k_x 0.1 (~1 g at full push). Bullets (not
    in the flight prototype): 600 m/s (real ~890), lifetime 0.4 s = 12 decisions (240 m,
    ~1.5 km real), one volley per decision (30/s, each ~11 real rounds), hitbox radius 3 m
    (~3x the scaled half-span of 0.92 m), 12 bullet slots per ship.
    - Feel (real-scale in brackets): cruise 105 m/s (156); coast full-stick sustained 59
      m/s, 75 deg/s (87 m/s, 18.4 deg/s; P-51 ~18); store charge as climb 3 m/s (4.4 m/s;
      P-51 16.3); full dive from a full store peaks at 118 m/s (175; P-51 dive limit ~225);
      zoom climb from 150 m/s stores 7.3 kJ before 46 m/s. Stall from 25 m/s recovers to
      40 m/s in 2.3 s (9.4 s real) even with an empty store (static thrust 10 N): stall is
      still forgiving.
    - Accuracy 30 / 60 / 120 Hz: 0.024 / 0.006 / 0.002 m (no fields), 0.22 / 0.06 / 0.016
      m (fields); energy drift with drag and engine off 6e-7. Cost (JAX CPU, noisy ~20%):
      12-15 ms per step vs today's 7.9.
  - *Cost of the v5 step (2026-10-22, `opcount.py`, `flight5_fast.py`, `v5f_bench.py`).*
    Expensive operations per step (divisions, square roots, trig, exp): today 37, v5 204.
    Optimized v5 (reciprocals, one lift split and thrust per half kick, polynomial 2x2
    exponential, small-angle series; same accuracy to 3 digits): 74. JAX CPU, 40,960 ships:
    today 7.9 ms; v5 15.3 / 12.8 ms (0 / 10 fields); optimized 13.5 / 10.9 ms (field
    evaluation alone 2.0 ms). On CPU the time follows total arithmetic more than the
    expensive operations. Per-ship work is a few hundred flops; on a GPU the step is
    bound by memory traffic and pairwise work (collisions, visibility, fields), not this.
    Accuracy per cost: today at 30 Hz has ~3.9 m error after 2 s; the new step 0.024 m.
    Physics at the decision rate (30 Hz, repeat 1) with the new step is ~100x more
    accurate than today's at 60 Hz, at less total cost than the decided 60 Hz tick.
  - *v6 memoryless pitch (2026-10-22, `flight6.py`, `v6_sweep.py`, `v6_fire.py`; author
    asked to prototype it and try values; shooting must cost more than passive gain, no
    shooting forever).* All engine power is thrust along the nose; coast / boost / reverse
    select a flight-path pitch (+gamma0 / -gamma_max / +gamma_max, gamma_max 30 deg);
    gravity along the path trades speed with the store (dE/dt = m g u sin gamma); pitch is
    scaled by the lift left after turning and forced down when lift cannot hold it; the
    store's ends fade the pitch (ceiling, floor). Drag set so coasting cruises at 105 m/s;
    static thrust P / 40 m/s. One bullet per step (30/s). Results at the 30 Hz tick:
    | engine W / neutral deg / store kJ | 400/3/10 | 800/3/10 | 1200/3/10 | 1200/6/10 | 1600/5/10 | 1600/5/6 |
    |---|---|---|---|---|---|---|
    | coast: fill from empty (s game) | 66 | 66 | 66 | 33 | 40 | 24 |
    | top level speed when full (m/s) | 124 | 113 | 110 | 116 | 111 | 111 |
    | climb (reverse): fill from empty (s), at speed | 20 at 29 | 10.5 at 65 | 9.3 at 74 | 9.0 at 77 | 8.6 at 82 | 4.9 at 87 |
    | dive (boost) from full: peak (m/s) | 170 | 156 | 146 | 152 | 141 | 136 |
    | coast full stick: speed, turn, store | 63, 80 deg/s, -102 W | 73, 91, -114 | 80, 96, -123 | 81, 97, -73 | 86, 98, -94 | 86, 98, -94 |
    | stall 25 -> 40 m/s (s) | 1.9 | 0.9 | 0.6 | 0.6 | 0.4 | 0.4 |
    Firing: a bullet cost set above the *neutral* charge still allows endless fire while
    climbing (climb charges 4-8x faster than neutral). Set above the *best climb* instead
    (1.5 x 1,350 W / 30 = 67 J per bullet), continuous fire from a full 10 kJ store lasts
    5.6 s coasting (167 bullets), 11 s climbing (329, speed falls to 77 m/s), 2.7 s diving
    (81), 4.7 s in a hard turn (140). A weak engine (400 W) cannot hold a 30 deg climb and
    slows to near stall.
  - *v7 commanded exchange, no implied angle (2026-10-23, `flight7.py`, `v7_sweep.py`,
    `v7_fire.py`).* **Author: no implied pitch angle; reverse must slow the ship.** Each
    power action commands an exchange acceleration along the path (coast +a0, reverse
    +a_climb = 27, boost -a_dive = 27 m/s^2); dE/dt = m u a_x. Static thrust per kg (15-20
    N) stays below a_climb, so reverse always decelerates. Lift coupling, forced sink and
    store-end fades as in v6. Sweep (drag set for a 105 m/s cruise):
    - Reverse from 105 m/s: 81-83 m/s after 1 s, 62-67 after 2 s; it then settles into a
      sustained climb near the best-climb speed (~60-75 m/s) and keeps charging.
    - Coast from empty fills 10 kJ in 31-48 s; boost from full peaks at 156-167 m/s and
      empties in ~2.6 s; full-stick coast turn 79-87 m/s, 95-98 deg/s, store -250 to -280
      W; stall 25 -> 40 m/s in 0.8-1.0 s.
    - Firing: a sustained climb can always charge at the engine's **maximum excess power**
      (557 / 804 / 847 W for 800 / 1200 / 1600 W engines, at ~60 m/s). With the bullet cost
      tied to the coast charge, fire while climbing never runs dry. **Invariant: continuous
      fire drains more than the maximum excess power.** At 1.2x (22-34 J per bullet), a
      full store lasts 14-22 s coasting, 38-59 s climbing, ~2.2 s diving, 8-10 s in a hard
      turn; a larger margin shortens these.
  - *Why climbing can always fund fire (2026-10-23, `v7_stallcycle.py`).* Author's point:
    reverse is a 1 g climb and with thrust/weight < 1 it should run to stall. Correct; v7's
    near-stall blend (fading the climb into a sink in proportion to the lift shortfall)
    acted as a best-climb autopilot. But total energy (kinetic + store) always changes at
    the engine's excess power (T - D) u, whatever the path, so any way of holding speed
    steady (the stall cycle, alternating coast and reverse, or settling) puts that power
    into the store. Measured with an unbounded store (1200 W engine, max excess 804 W at 62
    m/s): blend, hold reverse: settles at the stall, 31 m/s, +522 W; no blend, hold
    reverse: 63-64 m/s, +802 W; alternating 0.5 s reverse / 0.5 s coast: +774 W (blend),
    +799 W (no blend). So "no shooting forever" still needs continuous fire to drain more
    than the maximum excess power (or stall to dissipate energy, which alternation avoids).
  - *Follow-up (author, 2026-10-23).* Net positive energy gain is intended unless the ship
    is too fast or shooting. v7 (with its near-stall blend) is the natural behaviour:
    holding reverse decelerates monotonically to the stall speed (31 m/s) and settles there
    (the earlier "best-climb autopilot" reading came from runs that ended while still
    decelerating). The no-blend variant's settling at 63 m/s was an artifact (the soft
    budget makes the hold fraction 0.999, flipping a threshold); dropped. No extra
    parameters are needed: the bullet cost (which exists anyway) is set above the maximum
    excess power. Best climb speed (max excess power) for the 1200 W engine: 62 m/s = 2.0x
    stall, 804 W, 30 m/s of game climb (real-scale 92 m/s and 44 m/s; P-51 best climb is
    also well above stall, at a lower rate). Excess power: 517 W at 32, 758 at 50, 804 at
    62, 751 at 75, 576 at 90, 285 at 105, ~0 at 116 m/s. Holding reverse banks only the
    stall-speed excess (~520 W); flying best climb (e.g. alternating coast and reverse)
    banks ~800 W, which rewards skill.
  - *Headline stats, game-scale SI (2026-10-23, `old_stats.py`, `new_stats.py`).* P-51D
    converted with speed / 1.485, rates x 4.087, lengths / 6.07 (approximate references).
    | | today | v7 (1200 W, 20 N, a0 3) | P-51D at game scale |
    |---|---|---|---|
    | stall / min level speed | none (control fades below 35 m/s) | 31.6 m/s | 31.6 (47 est.) |
    | cruise (coast) | 100 m/s | 104 (charging), 116 store full | 109 (162 cruise) |
    | top level speed | 136 sustained boost (store empty) | 116 | 119-131 (177-195) |
    | dive / burst peak | 229 (boost from 160, full store) | 162 (boost from full) | ~152 (225 limit) |
    | corner speed | 100 m/s | 89 | 90 (134, forum) |
    | max instantaneous turn | 151 deg/s at 100 m/s (cap 155) | 98 measured (cap 115) at 89 | ~137 (33.6, 8 g at 134) |
    | min instantaneous radius | 37 m | 37 | ~38 (231 at corner) |
    | max sustained turn, coasting | 70 deg/s at 81 m/s (normal), r 67 | 94 deg/s at 76 m/s (full), r 46 | ~74 at 76 (18 at 113, forum) |
    | min sustained radius, coasting | 35 m (sharp, 39 m/s) | 46 m (full stick) | ~59 (360) |
    | sustained turn boosting / diving | 119 deg/s at 73 m/s, r 35 (power empty) | 85 deg/s at 121, r 82, while the store lasts (4.9 s) | n/a |
    | nose response (90%) at corner | 0.07 s | 0.33 s | no sourced roll figure |
    | best climb | n/a | 62 m/s, 30 m/s climb | ~51, 11 (76, 16.3) |
    v7 is close to the P-51 in stall, cruise, corner, radius and sustained turn; slower to
    respond and lower in instantaneous rate than today (8 g vs today's 10 g and its
    near-instant nose); today's boost is a free sustained top speed that v7 does not have.
  - *Raising the instantaneous turn rate (author, 2026-10-23; `flight8.py`, `v8_rate.py`,
    `v8_pick.py`).* Measured on the flight path (heading minus angle of attack; an
    earlier pass wrongly read the nose rotation rate, which overshoots while the nose
    swings in). The peak path rate is set by the sharpness of the soft g limit (soft-min
    exponent p = 4 / 8 / 16: 98 / 115 / 126 deg/s); the nose speed only sets how fast it
    arrives (time to 90% 0.43 / 0.27 / 0.17 / 0.10 s at nose x1 / 1.5 / 2 / 3).
    | | peak path | t90 | sustained coast | accuracy 30/60/120 Hz |
    |---|---|---|---|---|
    | v7 (p 4, nose x1, 8 g) | 98 deg/s | 0.43 s | 98 deg/s @ 85 m/s, r 50 | 0.033 / 0.009 / 0.003 m |
    | p 16, nose x2, 8 g | 127 | 0.13 | 103 @ 69, r 38 | 0.168 / 0.043 / 0.011 |
    | p 16, nose x2, 8.6 g | 131 | 0.13 | 103 @ 69, r 38 | 0.187 / 0.048 / 0.012 |
    | p 16, nose x2, 9 g | 134 | 0.13 | 103 @ 69, r 38 | 0.200 / 0.051 / 0.013 |
    P-51 at game scale ~137 deg/s. The sharper limit costs accuracy (still second order and
    ~20x better than today's ~4 m).
  - *Turn curves vs a P-51 model (2026-10-23, `turn_curves.py`).* Both are the same
    physics family (point mass: lift-limited turn rising with speed up to the g limit, then
    falling as 1/speed; power-limited thrust; parasitic + induced drag). P-51D sea-level
    energy-maneuverability model (m 4,500 kg, S 21.8, C_Lmax 1.5, C_D0 0.0163, k 0.068,
    1,490 hp x 0.8, 8 g) in game units: stall 31.6, corner 89.4 m/s.
    - Instantaneous curve: ours (p 16, 8 g) matches the P-51 at every speed (31 / 61 / 80 /
      98 / 114 / 129 / 131-136 / 117 / 107 / 98 deg/s at 35 / 45 / 55 / 65 / 75 / 85 /
      90 / 105 / 115 / 125 m/s); p 4 loses up to ~16% around the corner.
    - Sustained curve: matches below ~50 m/s, then ours is far higher (93 vs 49 deg/s at
      75 m/s; 48 vs 1 at 105). Best sustained: P-51 70 deg/s at 49 m/s (real 17 deg/s at
      73 m/s, consistent with the forum ~18); ours 101 deg/s at 67 m/s. Cause: our excess
      power is ~2.7x the P-51's (the faster climb the author asked for), and sustained
      turn and climb both draw on excess power. A larger induced-drag factor would lower
      sustained turn while barely changing the straight-line climb.
    - Not modelled: altitude (one air density), compressibility, the P-51's own pitch
      dynamics; our energy store replaces vertical flight.
  - *Sustained turn by energy condition and induced drag (2026-10-23, `v8_induced.py`; v8
    p 16, nose x2, 8 g, 1200 W).* Full stick, steady state: coast (store mid or full) 69
    m/s, 103 deg/s, r 38 (coast climb is cut to zero by the turn; store -35 W); boost with
    an unlimited store 83 m/s, 123 deg/s, r 38 (store -731 W); boost with an empty store
    68 m/s, 102 deg/s; engine off, coast 36 m/s, 43 deg/s (gliding, -152 W); engine off,
    boost 57 m/s, 82 deg/s (-602 W).
    Induced-drag factor (cd0 re-tuned for a 105 m/s cruise), x1 / 1.5 / 2 / 3 / 4: best
    coast sustained 103 / 91 / 82 / 70 / 61 deg/s at 69 / 61 / 55 / 47 / 42 m/s; max excess
    804 / 784 / 765 / 728 / 692 W (best climb 29.8 / 29.0 / 28.3 / 27.0 / 25.6 m/s); top
    speed 116-117; instantaneous peak 127 / 126 / 125 / 124 / 122 deg/s; speed lost in 1 s
    of full stick from 95 m/s 10 / 16 / 22 / 30 / 37 m/s. x3 matches the P-51's best
    sustained (70 deg/s at 49 m/s). Sustained curve, P-51 vs ours x3 (analytic, deg/s):
    35 m/s 31/31, 45 61/61, 50 69/61, 55 65/60, 65 57/57, 75 49/52, 85 40/45, 95 28/38,
    105 1/28 (ours keeps more at high speed: higher top speed).
  - *Tests before deciding (author, 2026-10-23: test, don't decide first; `flight9.py`,
    `v9_levels.py`, `v9_bleed.py`, `v9_gravity.py`, `v9_gravity2.py`; final config).*
    - **Turn levels.** Coast datum (same from 50 and 120 m/s), rate, radius, gun offset;
      boost datum with an unlimited store:
      lift thirds: 1/3 87 m/s, 44 deg/s, r 113, 2.7 deg (boost runs away to 207 m/s);
      2/3 58, 61, r 54, 6.3 deg (boost 89 deg/s at 87); 3/3 47, 70, r 39, 10.3 deg (boost
      85 at 57). Stall-angle thirds: 1/3 66 m/s, 58 deg/s, r 65, 5.0 deg (boost 85 at
      98); 2/3 47, 65, r 41, 9.2 deg (boost **87 at 62, the best: the "max sustained at
      2/3" rule holds**); 3/3 as above. Instantaneous peak by start speed 40 / 60 / 90 /
      120 / 140 m/s: stall-angle 1/3 41 / 54 / 75 / 95 / 95 (a hard turn at high speed,
      constant lift coefficient), lift 1/3 25 / 34 / 44 / 36 / 31. Full stick stops at
      10.3 deg because part of the lift holds level.
    - **Turn bleed source** (full stick from 90 m/s, store full): from speed: 62 / 49 / 47
      m/s at 1 / 3 / 6 s, store untouched, 3.0 kJ total lost by 3 s, 80 deg/s; half each:
      75 / 63 / 60 m/s, store 5.7 / 3.0 kJ, 6.4 kJ lost, 102 deg/s; from the store: 96 /
      66 / 47 m/s, store empty by 3 s, 11.9 kJ lost, 119 deg/s. Paying from the store keeps
      the ship fast, where drag is higher, so it burns ~4x the energy: it is a descending
      turn. Boost + turn already gives that choice explicitly.
    - **Climb/dive vs a 3D point mass in gravity** (u' = T - D - g sin gamma, gamma' =
      (L_v - g cos gamma)/u, h' = u sin gamma, horizontal speed u cos gamma; a pilot law
      tracks the commanded angle with the lift budget). The energy law matches exactly
      (coast at 6 deg agrees to <0.3 m/s). Differences: (1) a = g means a vertical climb or
      dive, yet the model kept full horizontal speed; (2) holding a climb needs g cos
      gamma, not g; (3) horizontal turn rate is lateral lift / (u cos gamma); (4) the
      commanded angle is reached instantly, while a real pull-up or push-over takes time
      and costs induced drag. Adding (1)-(3) with sin gamma = a/g and a 60 deg cap fixes
      horizontal distance (30 deg climb 240 vs ref 240 m; 60 deg climb 121 vs 130; 60 deg
      dive 157 vs 158). (4) remains: after 3 s of a full 60 deg climb from 105 m/s, the
      reference is at 54 m/s with +0.2 kJ total, the model at 62 m/s with +2.1 kJ (the
      reference's 8 g pull-up spends ~1.5 kJ in induced drag).
  - **Decided (author, 2026-10-23): no pitch angle.** Climb and dive stay pure energy
    exchange; only the energy law must match flight in gravity, and it does (checked
    against the 3D point mass). The climb-geometry variant (`geom` in `flight9.py`) is
    not adopted.
  - *Turn levels, four options (2026-10-23, `v9_levels2.py`).* Coast datum (stable from 50
    and 120 m/s unless noted), rate, radius, gun offset; boost datum (unlimited store);
    from 120 m/s: peak rate and speed after 2 s.
    - Lift thirds (fraction of the lift budget): 1/3 87 m/s, 44 deg/s, r 113, 2.7 deg;
      boost 20 deg/s at 206 m/s; from 120: 36 deg/s, 107 m/s. 2/3 58, 61, r 54, 6.3; boost
      89 at 87; from 120: 76, 81. 3/3 47, 70, r 39, 10.3; boost 85 at 57; from 120: 118, 59.
    - Angle thirds (stall angle; lift coefficient 0.56 / 0.89 / 1): 1/3 66, 58, r 65, 5.0;
      boost 85 at 98; from 120: 95, 80. 2/3 47, 65, r 41, 9.2; boost 87 at 62 (best);
      from 120: 117, 61. 3/3 as above.
    - Coefficient thirds (lift coefficient 1/3, 2/3, 1: fixed radius below the g limit):
      1/3 86, 45, r 110, 2.8; boost 67 at 131; from 120: 60, 98. 2/3 58, 61, r 54, 6.3;
      boost 89 at 85; from 120: 104, 72. 3/3 as above. Same as lift thirds below corner
      speed; differs only above it (the g limit caps each level separately).
    - Energy-neutral middle (2/3 = hardest turn that keeps the current speed): degenerate.
      Above cruise there is no spare thrust, so it flies straight, accelerates to top speed
      and never turns. Dropped.
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
   `docs/rewrite/ship-card.md`). *Superseded 2026-10-23 in large part by the new flight
   model (`flight-model.md`): stall speed now follows from g and C_Lmax; the nose is a
   rigid body (frequency and damping, no slew limit); low-speed control comes from lift
   ∝ u² and the forced sink, not a separate fade. The card is rewritten in the
   physics-first plan, step 5.*
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
- [x] Flight model redesign (2026-10-22/23): `docs/rewrite/flight-model.md`.
- [ ] Physics-first changes in the PyTorch env (planned, not started; plan updated
      2026-10-23 for the new flight model).
- [ ] Repo setup (option 3), after the physics-first changes.
