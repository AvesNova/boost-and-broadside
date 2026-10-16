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
  next-state head's forecast (env -> obs -> model -> forecast -> obs loop).
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
    belief live in the carry.
  - **Feature encoding belongs to the model.** The env provides raw physical
    quantities.
  - Resets and respawns arrive as lifecycle flags in the observation; the agent
    decides what they mean for its carry.
  - Optional `outputs["inspect"]` exposes internals (such as belief) for the dev
    spectator without the runtime knowing their structure.
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
- Making normal the true max-rate datum is a gameplay retune, so it comes after parity.

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
  - Tag the current state (proposed tag: `v1-torch-final`).
  - Start a `v2` branch with the old tree removed.
  - Merge to `main` once v2 is playable, so the public README keeps showing working
    results meanwhile.
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
   - Confirm the global shape rules: normal-turn lift fraction, normal-turn lateral
     fraction (today a hard-coded 0.5), induced-drag tie between normal-slip and stall
     drag, fade behavior tied to the control-loss speed, stall angle, nose slew
     derived from onset.
   - Confirm the validity constraints and the log-gap parameterization for sampling.
   - Pick the similarity-ratio suite shown on the card (current draft: 10 ratios).
2. **Concrete contracts:** observation schema, action schema (keep the 30-way joint
   command?), decision rate and action repeat, event list, trajectory/label format,
   and versioning rules.
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
6. **JAX library deep dive:** NN library (Flax NNX vs Linen vs Equinox), optax,
   checkpointing (Orbax vs plain pytree save).
8. **In-depth review of the current Yemong model and training.** Walk through the
   policy architecture (encoder, spatial attention, Griffin recurrence, heads,
   belief, categorical codes) and the training stack (PPO, critic, auxiliary losses,
   BC, opponents, rating, rollout and update loop) piece by piece, so the author
   understands everything and incorrect or hidden assumptions surface before the new
   model and learner contracts are written. Review the code itself, not just the
   docs, since many docs are stale.
7. ~~One model for all modes, or one per mode~~: decided, both supported (see Game
   scope).

## Next steps

- [ ] Set up the repo per option 3: tag `v1-torch-final`, create the `v2` branch.
- [ ] Then work through the tabled items, starting with game scope (1) and contracts
      (2), which most other items depend on, plus the agent workflow (5) before any
      code is written.
