# Sections 4 and 7 (the scripted agent and BC): notes

Branch `section/scripted`. The spec is `scripted-bc-spec.md`; the user pages are
`docs/explanation/scripted-agent.md`, `docs/explanation/behaviour-cloning.md` and
`docs/guides/behaviour-cloning.md`.

## Progress (working state)

PROGRESS_LINE

## Design

### Part A: the scripted agent

| Module | Job | Lines |
|---|---|---:|
| `game/handling.py` | `Characteristics`, `measure` (the game's `flight_tick`, `launch`, `bullet_motion` at n = 1), `time_of_flight`, the cached `characteristics` | 194 |
| `controllers/scripted_parameters.py` | `Ramp`, `ramp`, `ScriptedParameters` (the searched defaults), `pack`/`unpack`/`table` (per-view parameters) | 112 |
| `controllers/scripted_combat.py` | `Context` (what both layers read), `lead`, the stick/power/shoot marginals, the boundary steer, `combat`, `blend` | 226 |
| `controllers/scripted_frontline.py` | strength kernels, the defense/attack assignment by rank, separation, recovery, the blend with combat | 141 |
| `controllers/scripted.py` | `marginals`, `joint`, `distribution`, `Scripted` (sampling, rungs) beside `Random` and `Chaser` | 161 |
| `runtime/evaluate.py` | `first_results` (fresh games to their end, each view's first result), `score`, `shares`; `win_rate` uses it | 44 |

- **Characteristics.** Measured once per card and rules (about 5 s on the CPU, cached):
  cruise is coasting straight from corner speed with a full store until a second changes
  the speed by less than 1e-3 m/s (the true no-input terminal speed, store full); top
  speed is the peak of a straight boost from that state (the store empties first); turn
  rates hold each stick level at cruise speed (the speed reset each tick) for 1 s and read
  the path rate; gun range flies a no-spread bullet launched from the cruise state for its
  12 moves. Corner speed is the card's `sqrt(a_max / C_max)`. The default card gives
  cruise 103.7 / 106.0 m/s, top 162.1 / 171.3 m/s, gun range 220.8 / 204.7 m, turn rates
  1.12 / 1.96 / 2.00 and 0.94 / 1.44 / 1.45 rad/s (fighter / heavy).
- **One pass, fixed shapes.** Both layers are computed for every view; the mode one-hot
  selects per view (`torch.where`), so a mixed batch is one graph and nothing branches on
  data. Pair work is `(R, S, E)` and `(R, S, S)`.
- **The Frontline assignment (v2).** v1's per-zone need with every ship choosing the
  neediest zone does not survive memorylessness: measured on the first build, all five
  ships reached the attack zone, each then saw it covered without itself, and all left
  together (oscillation; 4.7% draws against random, mean game 220 s). The layer now
  assigns: the defense zone wants max(floor, enemy strength near it + margin), plus the
  urgency while its meter moves against the viewer; ships are ranked by (distance to the
  defense zone − distance to the attack zone), ties by row, and fill the want in that
  order by health-weighted strength; the rest attack unless their strength times
  exp(aggression) is below the enemy's at the attack zone. Every ship of a view computes
  the same assignment. Against random this gave 512/512 wins and a mean game of 86 s.
- **Sampling.** Inverse CDF per marginal from the default generator (in the captured
  graph); a zero-probability category is never drawn. Rungs: one uniform draw for all
  views, `u < p` keeps the scripted action.
- **Per-view parameters.** A parameter is a float (a compile-time constant) or a column
  view `(R, 1)` of a table; the search ran 32 candidates side by side in one compiled
  rollout this way.

### Part B: BC

| Module | Job | Lines |
|---|---|---:|
| `learn/bc.py` | `config`, `TeacherConfig`, `labels`, `entropy`, `teacher_distribution`, `BehaviourCloning`, `mean_entropy`, `calibrate`, `StopConfig`, `StopRule` | 210 |
| `runtime/cloning.py` | `LINEUP`, `scores` (per opponent kind), `clone` (the loop with the stop rule), `warm_start`, `teacher_distributions` | 107 |

- **The learner.** `BehaviourCloning(Learner)` overrides two methods: `_policy` (the
  cross-entropy against the labels over own ships alive, the student's entropy as a
  coefficient-0 term, and per-pool sums of KL, both entropies and tokens as the
  statistics) and `report` (per-pool means). Everything else is section 6's update:
  returns, the critic on the student's returns, the auxiliary heads, micro-batching,
  the return scale. `bc.config()`: policy coefficient 0.025, entropy 0, `target_kl=None`.
- **Labels at update time.** `teacher_distribution` flattens the micro-batch's stored
  observations `(L, n)` into views and runs `Scripted`'s `distribution` inside the
  compiled loss. The rollout records the teacher's parts with `extra_reads`.
- **The loop.** `clone` runs `Trainer.update` and feeds the stop rule with the games
  against the scripted agent from `pool_metrics` and the update's BC KL.

## Decisions and deviations

1. **v1's code was not read.** The brief puts the v1 tree out of bounds, so "v1's numbers
   converted through v1's characteristics" could not be done beyond what the spec quotes
   (the shoot ramp, 0.5–1.0 gun ranges, is the one starting value taken from it). The
   other starting values are mine (listed below) and the search moved all of them; the
   search's second opponent is therefore my starting point, not a v1-converted one.
2. **Frontline assignment by rank** instead of each ship choosing the neediest zone (see
   Design): the literal port oscillated without memory. The spec's quantities remain
   (strength kernels, the defensive floor, the margins, exp(aggression), the urgency from
   the meter's direction, the support radius in zone radii); the zone-distance cost was
   dropped with the per-ship choice.
3. **The lead's distance is capped** (`lead_cap_in_gun_range`, searched to 1.16): the
   drag law's time of flight grows exponentially with distance (about 2.4 s at 1000 m),
   and an uncapped lead points seconds ahead at targets that will turn.
4. **No target, power:** "at cruise" is coasting, with the slow-boost rule kept (a ship
   spawns at 30 m/s).
5. **Top speed** is the peak of a full-store boost from cruise, not a held-store terminal
   speed (the held-store terminal under boost, ~197 m/s, is out of reach of the store).
   Cruise is the settled speed with a full store, matching the reference flight's
   60 s coast (`top_full_store`) to 2e-4.
6. **`turn_rate` and `top_speed` are measured but the agent does not read them** (its
   turn ramps are in stall angles, as the spec says). They are part of the
   characteristics A2 lists and the ship card (1b) will extend this module.
7. **Labels are exact per tick, not across batch shapes.** The update labels a segment
   as one batch; the CPU's vectorized kernels may round the last bit differently from the
   rollout's per-tick batch (measured: 1.2e-7 at most, 0.15% of entries). The test
   checks per-tick equality exactly and the batched labels to 2.5e-7.
8. **Labels per micro-batch, every epoch**, not per shard once: the teacher is cheaper to
   rerun (part A's cost) than its `(L, V, S, 42)` labels are to store and stream.
9. **The student's entropy is detached.** It is reported, not trained; its gradient at an
   underflowed probability is 0 × ∞ (found as NaN at update 53 of the first toy run;
   `test_the_gradient_stays_finite_for_a_confident_student`).
10. **The rollout records the learner's parts plus `extra_reads`**, in `PARTS` order; the
    learner's controller then also receives those parts.
11. **`Trainer.ppo` holds BC's learner too.** `bench/cell.py` (fixed) reads
    `trainer.ppo`, so the attribute keeps its name.
12. **Order of work.** Part B's code was written while part A's parameter search held the
    GPU; part A's code and CPU acceptance were committed first (e03361d8), its GPU
    acceptance after the search (scores below).
13. **The hand-off** is in memory (`warm_start`: weights into the learner and its league
    copy, in place, and the return scale's EMAs; the optimizer fresh; RL's own warmup
    counter starts at 0). Section 11 has not merged, so the policy-file round trip is the
    open item (below).

## Measurements

All on the development laptop (RTX 4070 Laptop GPU, 8 GB). Another agent's test suite
shared the GPU during part of them; timings are provisional, scores are not timings.

### The parameter search

`/tmp/bnb-scratch-scripted/search.py`: 32 candidates side by side (one parameter-table
row per view) in one compiled rollout per (mode, opponent): against the chaser and against
the starting point, team deathmatch and Frontline presets, fog on, seat drawn per game.
The objective is the mean of the four scores. Each round perturbs the round's best
(log-normal on every scalar and ramp end, aggression additive; ramp probabilities fixed).

| Stage | Rounds | Games per candidate per match | sigma | GPU time |
|---|---|---:|---|---:|
| 1 | 3 | 128 | 0.3, x0.6 per round | 203 s |
| 2 (from stage 1's best) | 8 | 256 | 0.2, x0.8 per round | 846 s |

The best combined score per round: 0.807, 0.900, 0.910; 0.903, 0.931, 0.939, 0.953,
0.962, 0.969, 0.965, 0.964. Re-evaluating each round's carried best at the next round
gives 0.03 less on average (selection on noise). About 20 minutes of the hour's GPU cap.

The starting point and the chosen values (rounded to three figures; ramps as low, high):

| Parameter | Start | Chosen |
|---|---|---|
| `turn_ramp_in_stall_angle` | 0.03, 0.15 | 0.0185, 0.160 |
| `level_two_ramp_in_stall_angle` | 0.2, 1.0 | 0.153, 0.400 |
| `level_three_ramp_in_stall_angle` | 1.0, 3.0 | 1.04, 2.09 |
| `boost_slow_ramp_in_corner_speed` (1 → 0) | 0.8, 1.1 | 0.471, 0.532 |
| `boost_far_ramp_in_store_gun_ranges` | 1.5, 3.0 | 2.91, 4.91 |
| `reverse_ramp_in_gun_range` (1 → 0) | 0.25, 0.5 | 0.368, 0.512 |
| `aim_ramp_in_target_size` (1 → 0) | 0.5, 1.5 | 0.335, 1.09 |
| `shoot_ramp_in_gun_range` (1 → 0) | 0.5, 1.0 (v1's) | 0.773, 0.876 |
| `lead_cap_in_gun_range` | 2.0 | 1.16 |
| `boundary_margin_in_turn_radius` | 3.0 | 2.30 |
| `boundary_steer_ramp_in_margin` | 0.0, 0.5 | 0.0133, 0.479 |
| `centre_hold_in_arena_radius` | 0.3 | 0.480 |
| `combat_radius_in_gun_range` | 3.0 | 2.33 |
| `support_radius_in_zone_radius` | 2.0 (v1's two zone radii) | 1.00 |
| `defensive_floor` | 0.5 | 0.810 |
| `defensive_margin` | 0.5 | 0.496 |
| `aggression` | 0.0 | 0.628 |
| `defense_urgency` | 1.0 | 1.82 |
| `separation_radius_in_gun_range` | 0.5 | 0.853 |
| `separation_weight` | 0.5 | 0.444 |
| `recovery_shield_fraction` | 0.3 | 0.185 |
| `recovery_threat` | 0.5 | 0.945 |

### Scores (acceptance A7.2, A7.3)

The chosen defaults, fresh games to their end, 5v5 presets, fog on, seat drawn per game
(`/tmp/bnb-scratch-scripted/acceptance.py`):

| Preset | vs random (2048) | vs chaser (1024) | vs start (1024) |
|---|---|---|---|
| `deathmatch` | 2048 wins (0 non-wins) | 1.000 | 0.907 |
| `frontline` | 2048 wins (0 non-wins) | 0.994 (0.988 wins, 0.012 draws) | 0.926 |
| `deathmatch_wide` | 2048 wins | 1.000 | not run (dynamo's recompile limit in the script) |

The starting parameters scored 0.997 (deathmatch) and about 0.94 (Frontline, 512 games)
against the chaser before the search, and 1.0 / 1.0 against random after the Frontline
assignment fix.

### Cost per tick (A7.4)

`/tmp/bnb-scratch-scripted/speed.py`: the 5v5 training mix at 4096 environments (bench's
ship budget), compiled rollout with CUDA graphs, best of 5 segments of 128 ticks after a
warm-up: random vs random 1.052 ms per tick, random vs scripted 1.111 ms. The scripted
agent costs 0.059 ms per tick for 4096 views, **5.6% of the game's tick** (the 10% bar
holds). Provisional (shared GPU). A first version of the script made the scripted agent
the rollout's learner, which also adds its observation records; that measured 18%,
mostly recording.

### The teacher's natural entropy

NATURAL_ENTROPY

### The toy BC result and budget (B6.5)

TOY_RESULT

### BC throughput (B6.7)

THROUGHPUT

## Changes outside `bnb.controllers` and `bnb.learn`

- **`bnb.game.handling`** (new): the characteristics, measured through the game's code.
  Game behaviour is unchanged.
- **`bnb.runtime.evaluate`** (new): `first_results`, `score`, `shares`.
- **`bnb.runtime.cloning`** (new): BC's lineup, loop, warm start and teacher sampling.
- **`bnb.runtime.rollout`**: `extra_reads`; the learner's view carries `self.reads`.
- **`bnb.runtime.train`**: `SCRIPTED`, `OBJECTIVES`, `TrainConfig.objective` and
  `.teacher`; the trainer builds only the controllers its pools seat (plus the learner
  and its league copy), measures the characteristics when the scripted agent plays or
  teaches, and builds `BehaviourCloning` for `objective="bc"`; `win_rate` uses
  `first_results`.
- **`bnb.learn.ppo`** (inside `bnb.learn`, listed for the reviewer): the policy terms are
  `Learner._policy` with `STATISTICS` and `report`; `target_kl=None` disables the early
  stop. PPO's numbers are unchanged (section 6's tests pass).
- **`bnb.cli`**: `--objective`, `--teacher-temperature`, `--teacher-floor`,
  `--bc-budget-steps`, `--calibrate-entropy`.
- **Tests:** `tests/sections/test_scripted.py`, `scripted_cases.py`, `test_handling.py`,
  `test_bc.py`, `test_bc_toy.py` (gpu, slow).
- **Docs:** new `docs/explanation/scripted-agent.md`, `docs/explanation/behaviour-cloning.md`,
  `docs/guides/behaviour-cloning.md`; the two in-progress pages removed; `mkdocs.yml` nav;
  `docs/roadmap.yml` (sections 4 and 7: merged); the training loop, learner, training
  guide and glossary (a "Rung" entry) updated; `reviewed:` dates on the pages that cover
  the rollout, the CLI and the training module after rereading them.

## Requests and open questions

1. **Open: the hand-off through policy files.** Section 11 has not merged.
   `tests/sections/test_bc.py::test_the_hand_off_round_trips_through_a_policy_file` is
   written against `bnb.runs.checkpoint.save_policy` / `load_policy` and skips until the
   module exists. The in-memory hand-off (`bnb.runtime.cloning.warm_start`) is built and
   tested.
2. **Request (runs spec 3):** the BC → RL hand-off needs the return scale's state (two
   EMAs) and BC's step count. A policy file holds the weights and step counters; the
   return scale is listed only in a training checkpoint. Either the policy file carries
   the return scale, or RL warm-starts from BC's training checkpoint (weights and return
   scale, discarding the optimizer).
3. **Request (spec A2):** `stall_angle` is one shared flight constant today; it enters the
   characteristics per type so a per-type card value drops in.
4. **Question:** should RL's learning-rate warmup restart after BC (it does now, with the
   fresh optimizer)?
5. **Question (spec A5):** the literal per-ship "neediest zone" choice oscillates without
   memory; the rank assignment is my replacement (decision 2).
6. **Proposal check:** the natural entropy (above) is low; a target entropy for RL needs
   a temperature well above 1 or a floor (`--calibrate-entropy` reports the reachable
   range).

## Size

Part A: 834 lines of source (handling 194, the four controller modules 640) and 44 for
`evaluate`; part B: 210 (`bc.py`) + 107 (`cloning.py`); edits to `ppo.py`, `rollout.py`,
`train.py` and `cli.py`: +143 / −58. Tests: 844 lines in five files.
