# Sections 4 and 7 (the scripted agent and BC): notes

Branch `section/scripted`. The spec is `scripted-bc-spec.md`; the user pages are
`docs/explanation/scripted-agent.md`, `docs/explanation/behaviour-cloning.md` and
`docs/guides/behaviour-cloning.md`.

## Progress (working state)

- **Done (first round):** parts A and B, their acceptance runs and measurements, the docs.
- **Fix round (2026-11-02):** items 1-4, 6 and 7 of the review's fix list done and pushed;
  see "Fix round" at the end.
- **Open:** item 5, the policy-file hand-off: section 11 (`bnb.runs.checkpoint`) has not
  merged into `origin/v2`. When it does: merge v2, make `bnb train --objective bc` save
  the clone as a policy file (weights, return scale, step counts) when it stops, and
  unskip `test_the_hand_off_round_trips_through_a_policy_file`.

## Design

### Part A: the scripted agent

| Module | Job | Lines |
|---|---|---:|
| `game/handling/characteristics.py` | `Characteristics` from section 1b's measurements (`measure`'s handles, `shown.top_speed`, `shown.settled`), `time_of_flight`, the cached `characteristics` | 98 |
| `controllers/scripted_parameters.py` | `Ramp`, `ramp`, `ScriptedParameters` (the searched defaults), `pack`/`unpack`/`table` (per-view parameters) | 112 |
| `controllers/scripted_combat.py` | `Context` (what both layers read), `lead`, the stick/power/shoot marginals, the boundary steer, `combat`, `blend` | 226 |
| `controllers/scripted_frontline.py` | strength kernels, the defense/attack assignment by rank, separation, recovery, the blend with combat | 141 |
| `controllers/scripted.py` | `marginals`, `joint`, `distribution`, `Scripted` (sampling, rungs) beside `Random` and `Chaser` | 161 |
| `runtime/evaluate.py` | `first_results` (fresh games to their end, each view's first result), `score`, `shares`; `win_rate` uses it | 44 |

- **Characteristics.** Read from section 1b's measurements once per card and rules
  (about 12 s on the CPU, cached): cruise and gun range are the card's handles (cruise
  with a full store; the gun range at cruise plus muzzle speed); top speed is 1b's shown
  peak of a full-store boost from cruise; the turn rates are 1b's coast datum (settled
  coasting turns with the store held at a quarter); stall angle, vision range and bullet
  drag are the card's per-type scaled values. Before section 1b merged this section flew
  its own measurement (`game/handling.py`, removed in the merge); cruise, top speed and gun
  range agree with it to 1e-5, and the turn rates changed definition (below). The default
  card gives cruise 103.7 / 106.0 m/s, top 162.1 / 171.3 m/s, gun range 220.8 / 204.7 m,
  datum turn rates 0.80 / 1.04 / 1.22 and 0.76 / 1.09 / 1.26 rad/s (fighter / heavy).
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
2. **Frontline assignment by rank** (decided by the author, 2026-11-02) instead of each
   ship choosing the neediest zone (see Design): the literal port oscillated without
   memory. The spec's quantities remain
   (strength kernels, the defensive floor, the margins, exp(aggression), the urgency from
   the meter's direction and progress, the support radius in zone radii); the zone-distance cost was
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
   characteristics A2 lists. Since 1b merged, `turn_rate` is 1b's coast datum (the settled
   turn) rather than A2's "at cruise": one shared definition, and no reader depends on it.
14. **Merged with section 1b.** 1b's `bnb.game.handling` package took the module's name;
    the characteristics became `bnb.game.handling.characteristics` on top of 1b's
    functions, with no flight code of their own. The search, the scores and the toy run
    were measured before the merge; the default card's flight is the same after it
    (cruise, top speed, gun range unchanged), and the scores were re-run after the merge
    (below).
7. **Labels are exact per tick, not across batch shapes.** The update labels a segment
   as one batch; the CPU's vectorized kernels may round the last bit differently from the
   rollout's per-tick batch (measured: 1.2e-7 at most, 0.15% of entries). The test
   checks per-tick equality exactly and the batched labels to 2.5e-7.
8. **Labels per micro-batch, every epoch**, not per shard once: the teacher is cheaper to
   rerun (part A's cost) than its `(L, V, S, 42)` labels are to store and stream.
9. **The student's entropy uses PPO's formula** (fix round), `-(exp(log_softmax) ·
   log_softmax)`, not detached. The first version computed `xlogy` of the probabilities
   and detached it, since that gradient at an underflowed probability is 0 × ∞ (found as
   NaN at update 53 of the first toy run;
   `test_the_gradient_stays_finite_for_a_confident_student`, which fails under `xlogy`).
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

`bench/scripted_search.py` (committed in the fix round; first run from a scratch copy
of the same procedure): 32 candidates side by side (one parameter-table
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
(`/tmp/bnb-scratch-scripted/acceptance.py`, the first round; the fix round's scores, on the
acceptance tests' seeds and with the progress urgency, are under "Fix round", item 2):

| Preset | vs random (2048) | vs chaser (1024) | vs start (1024) |
|---|---|---|---|
| `deathmatch` | 2048 wins (0 non-wins) | 1.000 | 0.907 |
| `frontline` | 2048 wins (0 non-wins) | 0.994 (0.988 wins, 0.012 draws) | 0.926 |
| `deathmatch_wide` | 2048 wins | 1.000 | not run (dynamo's recompile limit in the script) |

After the merge with section 1b (386db109), the same runs with a fresh seed: deathmatch
2048/2048 against random and 1.000 against the chaser; Frontline 2048/2048 against random
and 0.993 against the chaser (0.985 wins, 0.015 draws).

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

`bnb --preset mixed --num-envs 256 --calibrate-entropy 0.5` (and
`/tmp/bnb-scratch-scripted/reach.py`): the scripted agent plays itself in the 5v5 training
mix (staggered clocks, 4 segments of 128 ticks, 1.27M living own ship-steps). **Natural
entropy 0.396 nats per living ship-step.** The distribution's support averages 2.4 joint
actions (mean log-support 0.753), so the temperature alone reaches 0.753 at most:

| floor | T = 1 | T = 2 | T = 5 | T = 100 |
|---:|---:|---:|---:|---:|
| 0 | 0.396 | 0.595 | 0.719 | 0.753 |
| 0.05 | 0.745 | 0.932 | 1.050 | 1.081 |
| 0.1 | 1.026 | 1.202 | 1.312 | 1.342 |
| 0.2 | 1.515 | 1.668 | 1.765 | 1.791 |
| 0.3 | 1.943 | 2.074 | 2.157 | 2.180 |

Calibration to 0.5 nats at floor 0 gives T = 1.40 (twice: 1.4011 and 1.3949 on two
recordings). The calibration test hits its targets to 0.01 nats on synthetic
distributions.

### The toy BC result and budget (B6.5)

`tests/sections/test_bc_toy.py` (gpu, slow): the relational model (hidden 256), 2v2 team
deathmatch (4 fields, arena 1200 m), 256 environments x 128 ticks per update, BC's lineup,
`bc.config()`, the default stop rule with a 100M-step budget, seed 0
(`/tmp/bnb-scratch-scripted/toy_bc.py`, curves in `toy_bc.jsonl` and `toy_bc_30M.jsonl`):

| env steps | BC KL (EMA) | student entropy | live window score vs teacher |
|---:|---:|---:|---:|
| 2.0M | 0.98 | 1.18 | - |
| 5.9M | 0.58 | 0.89 | - |
| 9.8M | 0.38 | 0.71 | - |
| 15.7M | 0.26 | 0.55 | 0.02 |
| 19.7M | 0.22 | 0.51 | 0.06 |
| 29.5M | 0.15 | 0.44 | 0.23 |
| 61.6M | 0.089 | 0.35 | 0.42 |
| 65.4M | 0.087 | 0.35 | 0.45 (stop: score) |

The teacher's label entropy is 0.25-0.27 nats in this game. Stopped by the score rule
after 65.4M environment steps, 792 s including compilation. Then 512 fresh games each:
**the clone scores 0.478 against its teacher, 1.000 against the chaser; the teacher scores
1.000 against the chaser** (B6.5's 0.10 bar holds). A 30M-step run without the stop rule
ended at 0.348 against the teacher, so the score lags the KL by a wide margin: the
window passes 0.45 only once the KL is below about 0.09 nats.

### BC throughput (B6.7)

`/tmp/bnb-scratch-scripted/throughput.py`: Yemong, the 5v5 training mix, 1024
environments, float32, the bench cell's protocol (compile update, one episode of rollouts,
best of 3 timed collect + update). 4096 environments (bench's budget) ran out of memory with
another agent's process on the GPU.

| | rollout | update | env steps / s end to end | peak VRAM |
|---|---:|---:|---:|---:|
| PPO, `bnb.runtime.train.LINEUP` | 3.47 s | 47.0 s | 2596 | 3.53 GB |
| BC, `bnb.runtime.cloning.LINEUP` | 4.34 s | 63.8 s | 1923 | 3.51 GB |

**BC runs at 0.74x PPO's rate here, provisionally.** Other agents' benchmarks and tests ran
on the machine during both, and the gap does not match its known parts: the teacher's
labels cost 0.7 ms per 16-view x 128-tick micro-batch (`teacher_cost.py`), against about
113 ms for a micro-batch's gradient, and BC computes no extra unroll. A quiet
re-measurement should settle it.

## Changes outside `bnb.controllers` and `bnb.learn`

- **`bnb.game.handling.characteristics`** (new, in section 1b's package): the
  characteristics from the card's measurements. Game behaviour is unchanged.
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
3. **Resolved by 1b:** the stall angle, vision range and bullet drag are per type now and
   the characteristics read them per type.
4. **Question:** should RL's learning-rate warmup restart after BC (it does now, with the
   fresh optimizer)?
5. **Resolved (spec A5):** the rank assignment replaces the per-ship "neediest zone"
   choice, decided by the author (decision 2).
6. **The clone's entropy:** the natural entropy (above) is low. The floor now defaults
   to 0.01 (author; fix round item 8). If the clone's entropy is still too low with it,
   the author's next levers are the temperature on the labels (built) or a separate
   entropy loss during BC (not built).

## Size

Part A: 738 lines of source (characteristics 98, the four controller modules 640) and 44 for
`evaluate`; part B: 210 (`bc.py`) + 107 (`cloning.py`); edits to `ppo.py`, `rollout.py`,
`train.py` and `cli.py`: +143 / −58. Tests: 844 lines in five files.

## Fix round (2026-11-02)

The review's fix list, each with the test that now guards it. Every behaviour mutation
the review named was applied by hand and fails the named test (and passed the old suite).

1. **Behaviour tests that can fail** (`tests/sections/test_scripted.py`):
   - defense urgency removed (`under_attack` forced False):
     `test_frontline_a_meter_moving_against_it_calls_the_attacker_back` (no enemies; two
     ships; meter still or moving the viewer's way: one defends, one attacks; moving
     against: both defend);
   - the meter's `progress` is now read (spec A5): the urgency is
     `defense_urgency · (1 + progress)` while the meter moves against the viewer.
     `test_frontline_the_urgency_grows_with_the_enemys_progress` (four ships, floor 0.5,
     margin 0, urgency 1: two defend at progress 0, three at 0.9) fails when progress is
     dropped;
   - the shooter's velocity dropped from the lead: the lead test is parametrized with a
     moving shooter (80, −40) m/s; it also fails
     `test_near_the_boundary_and_heading_inward_it_does_not_turn_back`;
   - the outward condition removed from the boundary steer:
     `test_near_the_boundary_and_heading_inward_it_does_not_turn_back` (20 m inside the
     edge, heading 120°, a target dead ahead: flies straight);
   - the handover replaced by all-combat when a target is visible:
     `test_frontline_hands_over_to_combat_from_the_combat_radius_in` (beyond the combat
     radius: strategy; inside the shoot ramp's start: combat; half way: both sides
     between 0.25 and 0.75);
   - the zone-ranking tie-break removed: `test_frontline_equally_placed_ships_split_by_row`
     (two ships at exactly equal order: row 0 defends, whichever ship holds it).
2. **Win rates as tests:** `tests/sections/test_scripted_acceptance.py` (`gpu`, `slow`):
   against random, at most 2 non-wins in 2048 games; against the chaser, above 0.5 over
   1024 games (warns below 0.70); at `deathmatch`, `deathmatch_wide` and `frontline`, fog
   on, seat drawn per game, seeds 20261102 + 10 · preset + opponent, all at or above
   `bench.scripted_search.SEED_LIMIT` (checked by `test_the_seeds_are_not_the_searchs`).
   7 passed in 111 s. With the progress urgency, on these seeds: 2048/2048 against random
   at every preset; against the chaser 0.999 (deathmatch, one loss), 1.000 (wide, one
   draw), 0.991 (Frontline, 1.9% draws); against the starting parameters (seeds
   20261202 + preset) 0.893, 0.867, 0.934.
3. **The search is auditable:** `bench/scripted_search.py` holds the procedure, the
   starting values (`STARTING_PARAMETERS`, the table above) and its seeds: the
   perturbations from `--seed` (default 7), round `r`'s games from
   `game_seed(seed, r) = 1000 · seed + r < SEED_LIMIT = 10⁶`. The run that chose the
   defaults was the scratch copy with perturbation seed 7; its games drew from torch's
   default generators unseeded (their fixed start-up seed), not from any acceptance seed.
   Rerunning the tool will not reproduce those exact games, and its "start" opponent now
   uses the progress urgency. The scripted-agent page points at the tool.
4. **Student entropy, PPO's formula:** `bnb.learn.bc.BehaviourCloning._policy` computes
   `-(exp(log_softmax) · log_softmax)` with no detach.
   `test_the_gradient_stays_finite_for_a_confident_student` now checks the gradient at
   entropy coefficients 0 and −0.0081 and fails under `xlogy` of the probabilities
   (applied by hand). ε's default then changed by author decision (item 8).
5. **Open: the policy-file hand-off.** Section 11 has not merged into `origin/v2` (checked
   2026-11-02; `section/runs` has `save_policy`/`load_policy` with the return scale). The
   guide says the CLI's clone is not yet saved.
6. **Spec deviations:**
   - `characteristics.py` reads `Airframe.corner_speed` through
     `bnb.game.physics.flight.airframe_of`; `test_card_values_and_constants_are_read_as_given`
     compares against the game's value with `torch.equal`;
   - `bnb --calibration-segments` (default 4) sets `teacher_distributions`' segments;
   - full words for `p`, `u`, `q` in `scripted.py` (and `q` in `cloning.py`);
   - `StopRule` keeps per-update `(games, points)`; it drops the oldest updates whole
     while the rest fill the window and counts the oldest kept one by the exact share
     still needed. `test_the_score_window_trims_whole_updates_without_favouring_a_result`
     (30 wins + 30 losses, then 60 draws, window 100: 0.50; the old per-game deque gave
     0.40).
7. **The label rounding:** `test_the_update_time_labels_are_the_rollout_time_distribution`
   states that per-tick labels are exact, that a whole segment labelled as one batch
   differs at the ulp level (at most 1.2e-7 measured), and the 2.5e-7 tolerance.

8. **The teacher never gives an action probability 0** (author decision, after the review):
   `bnb.learn.bc.FLOOR = 0.01` is the default floor of `TeacherConfig`, `calibrate` and
   `bnb --teacher-floor`, spread over the 42 joint actions as before.
   `test_every_label_gives_every_action_a_probability_above_zero` (the stored segment's
   labels, dead ships included, at T = 1 and T = 0.25: every entry ≥ 0.99 · 0.01/42);
   the loss test's independent computation applies the floor; the temperature tests pin
   floor 0 where they test a structural zero or T = 1 leaving labels unchanged. The BC
   page states the default and its reason (the clone can sample every action in RL). The
   toy BC run, the natural entropy and the entropy table were measured at floor 0; the
   toy test was rerun at the new default (below). Next lever if the clone's entropy is
   still too low: a separate entropy loss during BC (not built).

Changes outside `bnb.controllers` and `bnb.learn` in this round: `bench/scripted_search.py`
(new), `bnb.game.handling.characteristics`, `bnb.runtime.cloning`, `bnb.cli`, the tests
above, and the docs (`scripted-agent.md`, `behaviour-cloning.md` in explanation and
guides, `benchmarking.md`; `reviewed:` dates on `training.md`, `designing-a-ship.md`,
`ship-card.md`, `benchmark.md` after rereading them).
