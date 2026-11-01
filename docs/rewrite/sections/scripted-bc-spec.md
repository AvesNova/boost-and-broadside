# Scripted agent and BC spec (sections 4 and 7)

Status: spec, 2026-11-01. Decisions: `scripted.md` (section 4) and `bc.md` (section 7).
**(v2)** marks a decided change from v1. **(proposal)** marks a value or choice the spec
makes, implemented as written and listed for the author.

Two sections, one spec: BC's labels, temperature and stop rule depend on the scripted
agent's interface. They can be built by one agent in two stages (section 4 first, its
acceptance passing, then section 7), after section 6 merges.

# Part A: the scripted agent (section 4)

## A1. Interface

- **`bnb/controllers/scripted.py`** gains `Scripted` beside `Random` and `Chaser`. It is a
  `Controller`, and memoryless (carry `None`).
- **The pure function:** `distribution(obs, characteristics, parameters) → (R, S, 42)`,
  the joint action probabilities in `to_joint` order, from the observation alone.
  - `step` samples from it with the controller's generator stream.
  - BC calls it directly on stored observations (B3).
  - `reads = ("ships", "zones", "globals")`; it never reads `Privileged` or game state
    **(v2)**.
- **The joint:** the outer product of three marginals, power (3), stick (7) and shoot (2),
  as in v1.
- **A dead ship** (or one waiting to respawn) gets the no-op action with probability 1.
- **Modes:** from the global mode one-hot. Team deathmatch runs the combat layer (A4);
  Frontline runs the Frontline layer on top (A5). Sandbox runs the combat layer and is
  not tested.

## A2. Characteristics: the ship card's quantities (v2)

Every threshold is a dimensionless parameter times a characteristic quantity of the
observing ship's type (or the target's, where stated). The type is a visible column;
the card is public game knowledge.

- **Measured with the ship card's tool:** `bnb/game/handling.py` (section 1b) measures
  them per ship type, through the game's own flight and bullet code (a short batched
  simulation at construction, cached), at the standard field index. If section 1b has
  not merged, this section creates the module with what it needs, and 1b extends it:
  - `cruise_speed`: the no-input terminal speed;
  - `top_speed`: the boost terminal speed, straight;
  - `corner_speed`: the slowest speed where the g limit caps the full-stick turn;
  - `turn_rate[level]` for the three stick levels, at cruise: the steady flight-path rate;
  - `stall_angle` (the gun's offset scale; per type when the ship card makes it so);
  - `gun_range`: how far a bullet fired at cruise travels in its lifetime, with drag;
  - `time_of_flight(distance)`: the bullet's time to a distance, from the drag law
    (closed form: `t = (exp(k s) − 1) / (k w₀)` for quadratic drag at index 1);
  - `max_energy`, `bullet_energy_cost`, `max_shield`, `hit_radius` (card values);
  - `vision_range`.
- Nothing here may duplicate the flight equations: the measurement runs the game's
  functions.

## A3. Parameters

- **`ScriptedParameters`:** a frozen dataclass of dimensionless numbers, the ramps
  `(low, high)` and their `(probability at low, probability at high)`, with names that
  say what each scales by (for example `shoot_distance_ramp_in_gun_range`).
- **The starting values** are v1's, converted through v1's own characteristics where a
  conversion exists (for example v1's shoot ramp 100–200 px over its 200 px range is
  0.5–1.0 gun ranges). Where none exists, a value the agent chooses and lists.
- **Retuning (author: keep it short):**
  - the scripted agent's primary job is a sensible action distribution, for BC and as an
    opponent, not peak strength;
  - one small search over the parameters, in batched games, scored against the chaser and
    against the v1-converted starting point, in both modes, fog as in training, capped at
    about an hour of GPU time;
  - record the chosen values and the search in the notes.
- **v1's team-target blend is not ported:** v1's only shipped configuration disables it
  (probability 0 at both ends of its ramp), so it is dead code **(v2)**.
- **v1's `prob_offset`** was never read; not ported.

## A4. The combat layer (v1's structure, retuned)

Per own ship, against its **target**: the nearest visible, living enemy (ties: lowest
row).
- **Lead (v1's interception, first order):**
  - `t = time_of_flight(distance)`, using the shooter's type;
  - the aim point is the relative displacement at `t`: `(p_target + v_target t) −
    (p_own + v_own t)`, since the bullet inherits the shooter's velocity;
  - no field bending.
- **Stick (v2: three levels per side):**
  - the error is the signed angle from the **nose** (`heading`) to the aim point, positive
    to the left (the stick's `+` side);
  - `p_turn` ramps with |error| (v1's turn ramp); the turn's level is chosen by two
    further ramps, level 2 over level 1 and level 3 over level 2. Both are in units of
    `stall_angle`, so the levels follow the gun's offset scale;
  - the turn's side follows the error's sign. Probabilities for the 7 stick indices
    follow `STICK_LEVELS`' layout (0 straight; odd right, even left; levels 0.4, 0.75,
    1.0).
- **Power (coast, boost, reverse):**
  - **boost** when slow (a ramp on speed in units of `corner_speed`) or when the energy
    store is full enough for the distance (v1's boost metric: energy fraction against
    distance in gun ranges);
  - **reverse** when close (a ramp on distance in gun ranges);
  - **coast** otherwise.
- **Shoot:**
  - aligned: the nose's angle to the aim point, as a ratio to the target's angular size
    `2 atan(hit_radius_target / distance)`, clamped to [1°, 45°] (v1);
  - in range: a ramp on distance in gun ranges;
  - `p_shoot = p_aligned · p_in_range`;
  - when the gun is cooling or the store cannot pay for a shot, `p_shoot` is 0: such a
    shot is a no-op, and a 0 keeps the teacher's labels honest about what it does.
- **The arena boundary (v2, new):**
  - v1's world wrapped; v2's arena has a boundary that damages. A ship whose distance from
    the centre (the world origin) exceeds the arena radius minus a margin, and whose
    flight path points outward, steers toward the centre;
  - the margin is a parameter times the ship's g-limited turn radius at its speed,
    `speed² / max_acceleration`;
  - the steer direction blends into the aim direction by a ramp on the depth into the
    margin.
- **No visible enemy (v2, new):** v1 idled (the no-op). Under fog that drifts into the
  boundary. Instead the ship heads for the arena centre at cruise, holding course once
  within a parameter times the arena radius **(proposal)**. It does not fire.

## A5. The Frontline layer (v1's `frontline_strategy`, retuned)

The same terms and their combination, read from the observation:
- **Strength:** v1's Gaussian kernels over visible, living ships, weighted by health
  `0.2 + 0.8 · shield / max_shield`. The combat radius is in gun ranges.
- **Zone need and preference:**
  - over own-defense and enemy-defense zones (zone roles 1 and 3);
  - the margins, v1's defensive floor, and the offensive margin `exp(aggression)`;
  - the defense urgency from the zone's meter (`progress`, `direction`) where it moves
    against the viewer;
  - support radius in units of the zone's radius (v1's default, two zone radii, followed
    the map scale the same way).
- **Separation** from allies (radius in gun ranges), **recovery** toward own spawn (role
  0) at low shield when threatened (v1's recovery health and threat), and the force
  combination.
- **Blending with combat:** v1's log-probability blend between the combat layer's
  marginals and the strategy's, by the nearest enemy's distance from the inner shooting
  range to the combat radius. Recovery overrides power and stick.
- **The boundary term** of A4 applies to the result.

## A6. Rungs

- **One controller for the whole stationary ladder (author):** random, every rung and the
  scripted agent are one controller with a per-view probability `p` (a tensor: 0 is
  random, 1 is the scripted agent).
- **One pass each:** a single scripted pass and a single uniform draw over all its views,
  then per ship and decision the scripted action where `u < p`, else the uniform one. Its
  distribution is `p · scripted + (1 − p)/42`.
- The league (section 8) sets each view's `p`.

## A7. Acceptance

1. **Behavioural tests** on constructed observations, for both ship types:
   - the stick level rises with the angle error, and the side follows its sign;
   - no shot when unaligned, out of range, cooling or short of energy; a likely shot when
     aligned and in range;
   - boost when slow or far with a full store; reverse when close;
   - the lead moves the aim point along a crossing target's velocity, by its time of
     flight;
   - near the boundary and heading out, it turns inward;
   - with no visible enemy, it heads for the centre and does not fire;
   - in Frontline, it goes to an own-defense zone whose meter moves against it, and
     retreats toward its spawn when hurt and threatened.
   - **Invariance:** the distribution is unchanged by the seat (the team-swap and mirror
     seat tests), by permuting own or enemy rows (permuted accordingly), and by the
     values of hidden rows' zeroed columns.
2. **Against random:** at most 2 non-wins (losses or draws) in 2048 games per mode, both
   seats, fog as in training. A steady 99% fails.
3. **Against the chaser:** a target, not a blocker. Aim for a score of 0.70 per mode over
   1024 games, both seats, fog as in training. It must beat the chaser (above 0.5); if it
   falls short of 0.70 after the capped search, report the score and move on.
4. **Speed:** its cost per tick, compiled within the rollout, is at most 10% of the
   game's tick at the 5v5 bench cells **(proposal)**. Report it.
5. **Characteristics:** each measured quantity matches an independent run of the game
   (for example `gun_range` against a fired bullet's flight), and they move with the
   card (a perturbed card moves them in the right direction).

# Part B: BC (section 7)

## B1. The learner

- **`bnb/learn/bc.py`:** BC over section 6's machinery: the same segments, streaming,
  `unroll`, labels, micro-batching and mixed batches.
- **The loss:** the cross-entropy of the student's joint 42-way distribution against the
  teacher's (B3), in nats per token, on the policy's mask (own ships alive), coefficient
  0.025.
- **Also trained:** the critic on the student's own returns (section 6's targets), next
  state, enemy action, density and the shot heads, at section 6's coefficients (value
  0.007 per level, next state 0.0041, enemy action 0.11, density 0.021, shot trajectory
  0.007 averaged over ages, shot outcome 0.019).
- **Nothing else:** no policy gradient and no entropy bonus. Adam at 3e-4 with the 1M-step
  warmup; 4 epochs of 32 minibatches; no KL early stop (it is a PPO trust region).
- **The return scale** (section 6's EMA) runs during BC, and is handed to RL.

## B2. Rollouts

- **The student plays the learner views,** sampling from its own policy.
- **Opponents (proposal):**
  - self-play 0.5 and the scripted agent 0.5 at its natural distribution;
  - no rungs in BC;
  - section 8's pool replaces this when it exists.
- **The mix:** Frontline 0.5, team deathmatch squad 0.375, team deathmatch battle map
  0.125; fog per pool as in training.

## B3. Teacher labels

- **At update time, per shard, on the GPU:** `Scripted.distribution` on the stored
  observations of the learner views. The teacher is memoryless and reads only the
  observation, so nothing more is stored.
- **The temperature** applies to the labels only:
  - `q_T ∝ q^(1/T)` over the 42-way joint, with a uniform floor
    `(1 − ε) q_T + ε / 42`;
  - `T` defaults to 1 and `ε` to 0; the opponent always plays `q` itself;
  - structural zeros (for example turning away from the target) stay zero under `T`
    alone. `ε` exists for the case where the target entropy cannot be reached otherwise.
- **Calibration:** a tool that bisects `T` (with `ε` fixed) to a target mean entropy, in
  nats per living ship-step, over a recorded set of teacher observations from the
  training mix. It reports the natural entropy (`T = 1`) and the chosen `T`.

## B4. Stopping and hand-off

- **The stop rule:** whichever fires first; the report names it.
  - **Score against the scripted agent:** at least 0.45 over the last 1000 finished games
    against it **(proposal: window)**, from the training games themselves;
  - **KL plateau:** the BC KL `KL(teacher ‖ student) = CE − H(teacher)` per token, as an
    EMA. It plateaus when the EMA has improved by less than 1% over the last 20M
    environment steps **(proposal)**;
  - **Budget:** 500M environment steps **(proposal; v1's upper bound)**.
- **Ratings come from the training games** (the decided rule). There are no evaluation
  rollouts. Diagnostic rollouts, if any, are rare and separate.
- **The hand-off:** a checkpoint with the weights, the return scale's state and the step
  count. RL loads it with a fresh optimizer and no cloning term.

## B5. Metrics

- The BC KL, the teacher's entropy and the student's (per pool);
- explained variance per level;
- the auxiliary losses;
- the live score per opponent kind and per pool.

## B6. Acceptance

1. **The BC loss** matches an independent computation on a stored segment, masks
   included, and accumulates exactly over micro-batches (section 6's rule).
2. **Labels:** the update-time teacher distribution equals the one the scripted agent
   computed at rollout time on the same observation, exactly.
3. **Temperature:** calibration hits a target entropy within 0.01 nats; `T = 1, ε = 0`
   leaves the labels unchanged; the opponent's play is unchanged by `T` and `ε`.
4. **The stop rule** fires correctly on synthetic score and KL series, each rule alone.
5. **The toy test:** the toy model (section 6's relational MLP) trained by BC on the
   scripted agent at 2v2 team deathmatch:
   - scores at least 0.45 against its teacher over 512 games;
   - its score against the chaser is within 0.10 of the teacher's **(proposal)**;
   - within a budget the section measures and records.
6. **The hand-off:** RL started from a BC checkpoint begins with BC's weights and return
   scale, a fresh optimizer, and no cloning term.
7. **Throughput:** BC's environment steps per second end to end at the 5v5 training cells,
   against section 6's PPO.
