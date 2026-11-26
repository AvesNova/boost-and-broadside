# Scripted look-ahead agent: progress (branch scripted-lookahead)

The design the author approved on 2026-11-26 (brief from the coordinator), built in
`lean/scripted/` (paths.py, opponent.py, parameters.py). Milestone notes: tried / worked /
didn't / next. Numbers only as measured; laptop RTX 4070, performance profile.

## Author additions (2026-11-26, during the build)

1. No enemy known (start, or all dropped): no goal point (no centre, no enemy side: a
   convention between both teams' agents to exploit). Only the own-ship terms: speed, a
   disengaged energy preference (`disengaged_energy_weight`), the boundary, and a gentle seek
   toward the freshest dropped last-seen position (`seek_weight`, eager to patient).
2. Hidden enemies are dead-reckoned from the last sighting by the time since seen (last speed
   and nose rate for up to 2 s, then straight; `opponent.dead_reckon`); their 21 paths start
   from there. The roam discount and the cutoff still apply. No worst case for the hidden
   stretch (beta covers the worst case inside the look-ahead).

## Step 1: paths, pair tables, collapse (2026-11-26)

- Tried: candidate paths from `core.steady_flight` (new: `steady_rates` plus the steady angle)
  stepped over 4 segments (0.25, 0.25, 0.5, 1 s), the nose spring in closed form (offset ODE
  e'' + 2 sigma e' + k omega^2 e = 0, under/critical/over-damped), the path rate linear in the
  angle between now (`core.current_path_rate`: the angle's own rate is not in the observation)
  and the steady state; chord per segment; medium and gradient bend per segment start.
- Worked: `path_check.py` against the real tick holding each of the 21 actions for 2 s, 64
  random 1v1 deals after 10 random ticks: no fields, median / 90th-percentile position error
  4.2 / 17.0 m at 1 s and 10.1 / 27.1 m at 2 s, nose 1.3 / 7.4 deg at 1 s, speed 2.4 / 12.3 m/s
  at 1 s; Snap is the worst level (median 13-14 m, 4-5 deg at 1 s). Fields: 4.7 / 22.9 m at 1 s,
  12.9 / 51.5 m at 2 s. (The 0.25 s point is not a whole tick count: 8 ticks = 0.267 s were
  compared, which accounts for most of its 4.5-4.7 m median.)
- Decided myself: the agent reads the team view state (what `observe` is built from), only
  the columns Obs carries (positions, headings, proper speed = world speed x n, angle of
  attack, nose rate, store, shield, cooldown, pending of own ships, time since seen; ln n and
  its gradient are functions of position and the fields, which Obs carries). Stickiness reads
  the pending action (in Obs), so the agent has no memory and its labels are a function of the
  observation; `fresh_memory` and the callers' memory plumbing were removed (step 2).
  Persona drawn per ship at each deal (today's mechanism). Speed kept above 1/4 stall speed in
  the paths (a never-seen enemy's row is all zeros).
- Next: game measurements, then the teacher interface, tests, docs.

## Step 2: speed (2026-11-26, laptop 4070, performance profile, 2,048 envs of 4v4, fields, fog)

`speed.py` (compiled with `compile_dynamic`, 20 timed calls after 3 warm), `profile_call.py`
(torch.profiler, compiled), `stage_times.py`.

- Tried: everything in PyTorch under compile. Didn't: 309.9 ms per call, peak 2,753 MiB above
  the state, cold first call 322 s (the rule-based agent 0.34-0.35 ms, 3 MiB, 15 s; the policy's
  forward on 3,072 seats 23.0 ms, 417 MiB, 30 s). Eager: the pair tables were 970 of 1,126 ms.
- Tried: the pair stage and the reply collapse as one Triton kernel (`kernels.pair_scores`, a
  21 x 21 tile per (ship, enemy) in registers; the PyTorch path kept as its twin, a test holds
  them equal). Worked: 38.4 ms, 347 MiB, cold 256 s.
- Tried: the kernel's tile (benchmark on random inputs: 32 x 32 at 8 warps 21.3 ms; 8 of my
  candidates x 32 at 2 warps 7.7 ms; 4 x 32 at 1 warp 7.4 ms, chosen), reciprocals for
  divisions; cover and his own-only terms moved into the kernel; `core.steady_flight` over all
  three powers at once; both views' paths in one pass and materialised once (inductor had
  recomputed the path chains inside each consumer: two fused kernels of 4.4 ms each). Worked:
  **16.6 ms per call, peak 273 MiB, cold first call 90.5 s**. Profile per call: the pair kernel
  9.7 ms, everything else about 6.7 ms.
- Didn't reach the estimate: 16.6 ms is 72% of the policy's forward (23.0 ms), against the
  estimate of about 13%. The kernel alone is 42% of the policy's forward; its 21 x 21 tables are
  padded to 21 x 32 lanes (b) and 24 rows (a). Not done: chunking his candidates with an online
  log-sum-exp (21 -> 24 lanes, at most about 1.3x on the kernel), sharing one exp between the
  lead's flight time and the kinetic worth.

## Step 3: defaults (2026-11-26)

- `energy_weight` 1.0 -> 0.25 by reasoning: the card's `store_turn` (2.09 rad on the P-51D) is
  the turn a full store buys beyond the sustained rate, about 0.75 of the angles term's unit, so
  a store is worth about 0.75 x `angles_weight` (0.3). At 1.0 a ship 566 m away 45 deg left,
  crossing, was turned toward with probability 0.58 (energy outweighed the gun's 0.11 score
  seconds); at 0.25, 0.92.
- Against the rule-based agent, 4v4 with fields, seats alternated, before (energy 1.0, eager,
  128 envs x 1,800 ticks): 113 wins, 0 draws, 16 losses; after (energy 0.25, compiled, 512 envs
  x 3,600 ticks): 1,404 wins, 9 draws, 68 losses of 1,481 games.

## Step 4: path prediction against the tick (coordinator's request, 2026-11-26)

The path's lag behind the nose and the fields were in the paths from step 1: the path rate is
linear in the angle between now and the steady state, and its time integral is the closed form
of the nose response's angle (`paths.nose_response` returns the integral of the offset), so
the lag in the first fraction of a second is carried; the medium (`core.medium`: ln n and its
gradient) is evaluated at every segment's start, map speed is proper speed / n, every rate is
scaled by 1 / n and the path bends with the gradient across it, as the tick does. Cost
(`paths_cost.py`, both views in one pass, compiled, 2,048 envs of 4v4): 7.69 ms with the
default fields, 6.51 ms without (the medium per segment: 1.2 ms).

`path_check.py`: 64 ships per speed, flying straight at 0.6 / 1.0 / 1.5 x corner speed (half
store), each of the 21 actions held for 2 s in the real tick; the flown state interpolated to
the exact look-ahead times. Median / 90th percentile errors, both sides and all powers pooled
per level (position m, flight-path heading deg, nose deg):


### Without fields

| speed / corner | level | error | 0.25 s | 0.5 s | 1.0 s | 2.0 s |
|---|---|---|---|---|---|---|
| 0.6 | straight | position m | 0.0 / 0.0 | 0.0 / 0.0 | 0.1 / 0.2 | 0.3 / 1.2 |
| 0.6 | straight | path deg | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| 0.6 | straight | nose deg | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| 0.6 | Sustain | position m | 0.4 / 0.4 | 1.5 / 1.7 | 5.8 / 7.7 | 17.9 / 27.5 |
| 0.6 | Sustain | path deg | 0.5 / 0.6 | 1.8 / 2.4 | 3.8 / 5.4 | 4.9 / 5.4 |
| 0.6 | Sustain | nose deg | 0.4 / 0.5 | 1.7 / 1.7 | 3.4 / 3.6 | 4.2 / 4.6 |
| 0.6 | Max | position m | 0.4 / 0.4 | 1.5 / 2.1 | 4.9 / 7.8 | 11.5 / 22.9 |
| 0.6 | Max | path deg | 0.9 / 1.4 | 2.2 / 3.6 | 2.4 / 4.3 | 3.1 / 7.1 |
| 0.6 | Max | nose deg | 0.9 / 1.6 | 2.4 / 3.9 | 2.4 / 4.2 | 3.2 / 7.4 |
| 0.6 | Snap | position m | 3.0 / 3.1 | 9.5 / 9.7 | 18.9 / 18.9 | 21.8 / 30.7 |
| 0.6 | Snap | path deg | 4.5 / 5.3 | 6.1 / 6.7 | 5.7 / 6.2 | 6.1 / 6.5 |
| 0.6 | Snap | nose deg | 3.6 / 5.8 | 6.4 / 7.2 | 7.4 / 7.4 | 23.9 / 23.9 |
| 1.0 | straight | position m | 0.0 / 0.0 | 0.0 / 0.0 | 0.1 / 0.1 | 0.4 / 0.8 |
| 1.0 | straight | path deg | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| 1.0 | straight | nose deg | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| 1.0 | Sustain | position m | 0.4 / 0.4 | 0.7 / 0.8 | 2.2 / 2.4 | 5.1 / 8.3 |
| 1.0 | Sustain | path deg | 0.2 / 0.2 | 0.4 / 0.6 | 0.4 / 0.8 | 1.0 / 1.3 |
| 1.0 | Sustain | nose deg | 0.1 / 0.2 | 0.3 / 0.3 | 0.3 / 0.5 | 1.3 / 3.0 |
| 1.0 | Max | position m | 1.1 / 1.1 | 2.7 / 2.8 | 6.8 / 7.3 | 13.7 / 14.7 |
| 1.0 | Max | path deg | 1.8 / 1.9 | 1.8 / 1.9 | 1.3 / 1.3 | 0.1 / 1.0 |
| 1.0 | Max | nose deg | 1.8 / 1.8 | 2.3 / 2.3 | 2.9 / 2.9 | 1.5 / 1.5 |
| 1.0 | Snap | position m | 1.2 / 1.2 | 1.8 / 1.8 | 4.1 / 4.1 | 15.0 / 15.0 |
| 1.0 | Snap | path deg | 1.4 / 1.6 | 0.1 / 0.2 | 0.1 / 0.1 | 8.6 / 8.6 |
| 1.0 | Snap | nose deg | 1.5 / 1.5 | 1.2 / 1.2 | 5.9 / 5.9 | 24.3 / 24.3 |
| 1.5 | straight | position m | 0.0 / 0.0 | 0.0 / 0.0 | 0.1 / 0.1 | 2.8 / 4.9 |
| 1.5 | straight | path deg | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| 1.5 | straight | nose deg | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| 1.5 | Sustain | position m | 0.0 / 0.0 | 0.0 / 0.0 | 0.1 / 0.9 | 2.8 / 4.9 |
| 1.5 | Sustain | path deg | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.2 | 0.0 / 2.0 |
| 1.5 | Sustain | nose deg | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.9 | 0.0 / 3.4 |
| 1.5 | Max | position m | 1.7 / 1.7 | 1.8 / 1.8 | 3.0 / 3.4 | 5.4 / 10.4 |
| 1.5 | Max | path deg | 0.2 / 0.2 | 0.2 / 0.3 | 0.3 / 0.6 | 0.2 / 1.4 |
| 1.5 | Max | nose deg | 0.2 / 0.2 | 0.1 / 0.6 | 0.0 / 1.0 | 0.9 / 1.8 |
| 1.5 | Snap | position m | 2.0 / 2.0 | 2.7 / 2.8 | 4.8 / 5.7 | 7.8 / 15.1 |
| 1.5 | Snap | path deg | 0.2 / 0.2 | 0.6 / 0.8 | 0.6 / 1.1 | 0.1 / 1.7 |
| 1.5 | Snap | nose deg | 0.0 / 0.2 | 0.6 / 1.0 | 0.2 / 1.4 | 0.8 / 2.1 |

### With fields

| speed / corner | level | error | 0.25 s | 0.5 s | 1.0 s | 2.0 s |
|---|---|---|---|---|---|---|
| 0.6 | straight | position m | 0.0 / 0.2 | 0.0 / 0.6 | 0.1 / 5.8 | 0.3 / 28.8 |
| 0.6 | straight | path deg | 0.0 / 0.2 | 0.0 / 0.8 | 0.0 / 2.4 | 0.0 / 5.5 |
| 0.6 | straight | nose deg | 0.0 / 0.2 | 0.0 / 0.5 | 0.0 / 2.0 | 0.0 / 5.4 |
| 0.6 | Sustain | position m | 0.4 / 1.5 | 1.5 / 5.0 | 5.8 / 15.3 | 17.9 / 40.5 |
| 0.6 | Sustain | path deg | 0.5 / 1.8 | 1.8 / 3.9 | 3.8 / 6.2 | 4.9 / 12.0 |
| 0.6 | Sustain | nose deg | 0.4 / 1.6 | 1.7 / 3.0 | 3.4 / 4.6 | 4.2 / 12.8 |
| 0.6 | Max | position m | 0.4 / 1.7 | 1.5 / 4.1 | 4.9 / 13.3 | 11.5 / 36.6 |
| 0.6 | Max | path deg | 0.9 / 2.7 | 2.2 / 4.2 | 2.4 / 5.4 | 3.1 / 13.1 |
| 0.6 | Max | nose deg | 0.9 / 2.8 | 2.4 / 4.3 | 2.5 / 5.6 | 3.2 / 13.2 |
| 0.6 | Snap | position m | 3.0 / 5.7 | 9.5 / 14.3 | 18.9 / 25.7 | 21.8 / 44.7 |
| 0.6 | Snap | path deg | 4.5 / 5.4 | 6.1 / 6.7 | 5.9 / 8.1 | 6.1 / 12.0 |
| 0.6 | Snap | nose deg | 3.6 / 5.8 | 6.4 / 7.4 | 7.4 / 9.3 | 12.2 / 23.9 |
| 1.0 | straight | position m | 0.0 / 0.4 | 0.0 / 5.6 | 0.1 / 14.7 | 0.4 / 64.4 |
| 1.0 | straight | path deg | 0.0 / 0.9 | 0.0 / 2.7 | 0.0 / 4.4 | 0.0 / 6.3 |
| 1.0 | straight | nose deg | 0.0 / 0.3 | 0.0 / 1.6 | 0.0 / 4.1 | 0.0 / 6.8 |
| 1.0 | Sustain | position m | 0.4 / 2.4 | 0.7 / 5.0 | 2.2 / 23.1 | 5.8 / 76.9 |
| 1.0 | Sustain | path deg | 0.2 / 0.7 | 0.4 / 2.5 | 0.4 / 6.0 | 1.0 / 15.1 |
| 1.0 | Sustain | nose deg | 0.1 / 0.5 | 0.3 / 1.8 | 0.4 / 5.2 | 1.3 / 15.7 |
| 1.0 | Max | position m | 1.1 / 4.7 | 2.7 / 7.2 | 6.8 / 18.1 | 13.7 / 57.0 |
| 1.0 | Max | path deg | 1.8 / 2.3 | 1.8 / 3.6 | 1.3 / 9.0 | 1.0 / 14.8 |
| 1.0 | Max | nose deg | 1.8 / 2.2 | 2.3 / 3.1 | 2.9 / 8.5 | 1.5 / 14.1 |
| 1.0 | Snap | position m | 1.2 / 5.4 | 1.8 / 6.1 | 4.1 / 17.0 | 15.0 / 65.8 |
| 1.0 | Snap | path deg | 1.4 / 1.6 | 0.2 / 2.0 | 0.1 / 9.0 | 8.6 / 23.2 |
| 1.0 | Snap | nose deg | 1.5 / 1.5 | 1.2 / 4.1 | 5.9 / 12.5 | 18.3 / 24.3 |
| 1.5 | straight | position m | 0.0 / 2.3 | 0.0 / 13.5 | 0.1 / 38.3 | 4.9 / 132.1 |
| 1.5 | straight | path deg | 0.0 / 2.1 | 0.0 / 4.3 | 0.0 / 6.3 | 0.0 / 13.0 |
| 1.5 | straight | nose deg | 0.0 / 1.6 | 0.0 / 3.6 | 0.0 / 6.5 | 0.0 / 13.4 |
| 1.5 | Sustain | position m | 0.0 / 2.4 | 0.0 / 13.6 | 0.1 / 38.7 | 4.9 / 125.7 |
| 1.5 | Sustain | path deg | 0.0 / 2.1 | 0.0 / 4.3 | 0.0 / 6.4 | 1.1 / 14.6 |
| 1.5 | Sustain | nose deg | 0.0 / 1.6 | 0.0 / 3.7 | 0.0 / 6.6 | 1.5 / 15.3 |
| 1.5 | Max | position m | 1.7 / 6.5 | 1.8 / 14.1 | 3.0 / 53.4 | 10.4 / 124.4 |
| 1.5 | Max | path deg | 0.2 / 1.6 | 0.2 / 5.5 | 0.3 / 10.6 | 1.4 / 22.8 |
| 1.5 | Max | nose deg | 0.2 / 0.8 | 0.2 / 5.1 | 0.5 / 10.7 | 1.8 / 23.0 |
| 1.5 | Snap | position m | 2.0 / 5.9 | 2.7 / 13.4 | 4.8 / 50.3 | 15.1 / 115.1 |
| 1.5 | Snap | path deg | 0.2 / 1.6 | 0.6 / 5.5 | 0.6 / 10.7 | 1.5 / 23.2 |
| 1.5 | Snap | nose deg | 0.1 / 1.3 | 0.6 / 5.1 | 0.6 / 10.6 | 2.1 / 22.9 |

Where it is large: **Snap at low speed** (0.6 x corner: 18.9 m and 7.4 deg of nose at 1 s; the
nose 23.9 deg off at 2 s) and **Snap's nose at 2 s** at corner speed (24.3 deg; path 8.6 deg):
the spring's target is Snap's deep angle at the starting speed, while the game closes the deep
angle toward the stall angle as the speed bleeds (`_holding_angle`), and the path rate is linear
in the angle, which past the stall it is not. With fields the 90th percentiles grow at 2 s (up to
132 m at 1.5 x corner, straight): the medium is sampled only at segment starts and the last
segment is 1 s long. Sustain and Max stay within about 7 m and 3 deg at 1 s without fields. Not
tuned (reported first, as asked).

## Step 5: games (2026-11-26; compiled; fog on; finished games re-dealt)

`measure.py`, 256 environments x 3,600 ticks per self-play setup.

| setup | games | draws | kills per env-minute | mean speed / corner | mean store | teacher entropy (nats) |
|---|---|---|---|---|---|---|
| look-ahead 1v1, no fields | 522 | 386 | 0.16 | 0.96 | 0.45 | 2.335 |
| look-ahead 1v1, fields | 538 | 328 | 0.34 | 0.98 | 0.52 | 2.324 |
| look-ahead 4v4, no fields | 881 | 14 | 9.72 | 0.87 | 0.21 | 1.949 |
| look-ahead 4v4, fields | 725 | 2 | 8.41 | 0.90 | 0.25 | 2.025 |
| rule-based 1v1, no fields | 703 | 40 | 1.15 | 0.48 | 0.27 | - |
| rule-based 4v4, fields | 525 | 15 | 5.66 | 0.51 | 0.17 | - |

New against old (seats alternated, 512 environments x 3,600 ticks), wins / draws / losses by
the look-ahead agent: 4v4 with fields 1,404 / 9 / 68 (1,481 games); 1v1 without fields 409 /
686 / 89 (1,184 games).

Usage of the 42 actions over living ship-ticks of self-play (no fire / fire):

| stick | power | selfplay_1v1_f0 (no fire / fire) | selfplay_1v1_f10 (no fire / fire) | selfplay_4v4_f0 (no fire / fire) | selfplay_4v4_f10 (no fire / fire) |
|---|---|---|---|---|---|
| straight | coast | 10.78% / 0.03% | 11.31% / 0.05% | 10.23% / 0.24% | 9.54% / 0.33% |
| straight | dive | 7.21% / 0.03% | 7.25% / 0.04% | 9.84% / 0.22% | 8.66% / 0.29% |
| straight | climb | 11.30% / 0.02% | 12.11% / 0.03% | 9.81% / 0.22% | 9.52% / 0.32% |
| Sustain R | coast | 8.75% / 0.02% | 8.62% / 0.04% | 7.47% / 0.26% | 7.58% / 0.33% |
| Sustain R | dive | 6.24% / 0.02% | 5.81% / 0.04% | 7.91% / 0.25% | 7.29% / 0.30% |
| Sustain R | climb | 7.27% / 0.01% | 7.67% / 0.03% | 5.29% / 0.23% | 5.78% / 0.27% |
| Sustain L | coast | 8.92% / 0.02% | 8.73% / 0.05% | 7.53% / 0.26% | 7.57% / 0.32% |
| Sustain L | dive | 6.31% / 0.02% | 5.80% / 0.04% | 7.90% / 0.24% | 7.32% / 0.29% |
| Sustain L | climb | 7.34% / 0.01% | 7.70% / 0.03% | 5.31% / 0.22% | 5.76% / 0.28% |
| Max R | coast | 3.09% / 0.01% | 2.80% / 0.01% | 3.24% / 0.08% | 3.30% / 0.09% |
| Max R | dive | 3.01% / 0.00% | 2.63% / 0.01% | 4.10% / 0.06% | 3.93% / 0.07% |
| Max R | climb | 3.09% / 0.01% | 2.82% / 0.01% | 3.23% / 0.08% | 3.32% / 0.08% |
| Max L | coast | 3.11% / 0.01% | 2.80% / 0.01% | 3.29% / 0.08% | 3.27% / 0.08% |
| Max L | dive | 3.01% / 0.00% | 2.59% / 0.01% | 4.19% / 0.07% | 3.91% / 0.07% |
| Max L | climb | 3.10% / 0.00% | 2.80% / 0.01% | 3.29% / 0.08% | 3.27% / 0.08% |
| Snap R | coast | 1.07% / 0.00% | 1.30% / 0.00% | 0.55% / 0.02% | 0.91% / 0.02% |
| Snap R | dive | 1.43% / 0.00% | 1.48% / 0.00% | 1.22% / 0.02% | 1.52% / 0.03% |
| Snap R | climb | 1.09% / 0.00% | 1.31% / 0.00% | 0.55% / 0.02% | 0.90% / 0.02% |
| Snap L | coast | 1.10% / 0.00% | 1.29% / 0.00% | 0.56% / 0.02% | 0.89% / 0.02% |
| Snap L | dive | 1.48% / 0.00% | 1.48% / 0.01% | 1.24% / 0.03% | 1.52% / 0.03% |
| Snap L | climb | 1.11% / 0.00% | 1.29% / 0.00% | 0.55% / 0.02% | 0.88% / 0.02% |

selfplay_1v1_f0: entropy 2.335 nats, kills per env-minute 0.16, unused actions 0, under 0.1% 21, games 522 (draws 386)

selfplay_1v1_f10: entropy 2.324 nats, kills per env-minute 0.34, unused actions 0, under 0.1% 21, games 538 (draws 328)

selfplay_4v4_f0: entropy 1.949 nats, kills per env-minute 9.72, unused actions 0, under 0.1% 12, games 881 (draws 14)

selfplay_4v4_f10: entropy 2.025 nats, kills per env-minute 8.41, unused actions 0, under 0.1% 12, games 725 (draws 2)

Every action is used in every setup. Power is close to indifferent: dive and climb are about as
frequent as coast at every level, and the replays show the power changing from tick to tick
(the 2 s look-ahead barely separates the powers, and tau_me 0.05 with stickiness 0.05 does not
hold them).

## Step 6: merge turn (2026-11-26)

`merge_turn.py` (the event definition of branch merge-turn; 256 envs 1v1, 64 envs 4v4, 1,800
ticks, seed 0, no re-deal). Median / 75th / 90th percentile seconds from the closest approach;
slow = time to 30 deg over 1.5 s or not reached in 10 s.

| setup | agent | aircraft | to 30 deg | to 90 deg | slow |
|---|---|---|---|---|---|
| 1v1, no fields | look-ahead | 1,476 | 1.17 / 2.03 / 2.80 | 2.97 / 4.03 / 5.17 | 554 |
| 1v1, no fields | rule-based (this tree) | 794 | 0.53 / 1.07 / 1.62 | 2.00 / 3.87 / 5.51 | 95 |
| 1v1, fields | look-ahead | 894 | 0.92 / 1.73 / 2.84 | 2.67 / 3.98 / 5.75 | 268 |
| 1v1, fields | rule-based (this tree) | 590 | 0.57 / 0.97 / 1.62 | 2.17 / 3.14 / 4.93 | 66 |
| 4v4, fields | look-ahead | 1,564 | 0.83 / 1.47 / 2.50 | 2.37 / 3.67 / 6.13 | 362 |
| 4v4, fields | rule-based (this tree) | 734 | 0.67 / 1.30 / 2.00 | 2.70 / 4.23 / 6.25 | 143 |

(The brief's 0.80 / 2.40 s for the rule-based agent, 1v1 without fields, was measured on branch
merge-turn before the post-stall flight; on this tree it measures 0.53 / 2.00 s.)

## Step 7: the 1v1 stalemate (2026-11-26) — a design problem, reported, not patched

1v1 self-play draws 74% of games without fields (386 of 522) and 61% with fields; 0.16 kills per
env-minute against the rule-based agent's 1.15. In the replays (`scripted1v1_nofields_seed0`,
`_seed1`, both drawn at the clock) the two ships circle 0.8-1.2 km apart and do not commit. In
neutral geometry nothing in the score rewards closing: the angles term is zero-sum ((cos ATA_me -
cos ATA_him) / 2 is 0 with both noses on), the gun is zero beyond the range band, and with
caution (beta -2) and a symmetric threat an approach scores below holding off. In 4v4 the
nearest enemies are close enough that fights start anyway.

Diagnostic (1v1 without fields, 256 envs x 1,800 ticks, `measure.py --set`):

| change | games | draws | kills per env-minute |
|---|---|---|---|
| defaults (3,600 ticks) | 522 | 386 | 0.16 |
| caution 0 | 266 | 157 | 0.32 |
| caution +2 | 273 | 131 | 0.51 |
| angles_weight 1.0 | 282 | 127 | 0.57 |
| caution 0 and angles_weight 1.0 | 288 | 114 | 0.63 |
| rule-based agent (3,600 ticks) | 703 | 40 | 1.15 |

Parameters move it but do not close the gap (40% draws at best). Options for the author: a range
term (closing toward a desired range while not defensive, as in McGrew's exp(-|R - R_d| / k),
docs/rewrite/sections/scripted-research/references.md), a longer horizon for the angles and
energy terms only, or an engagement drive outside the look-ahead. The quick-tier test
`lean/learn/tests/test_rewards.py::test_the_real_game_gives_zero_sum_levels` (2v2, 300 ticks,
expects some damage) fails for the same reason and was left failing.

## Replays (2026-11-26)

Five games in `~/avesnova/boost-and-broadside/data/replays-2026-11-26-lookahead/` (the
look-ahead yardstick on both teams, CPU, batch 1, fog on): `scripted4v4_seed0..2.mp4` (default
map with fields; all three Blue wipes at 27-38 s) and `scripted1v1_nofields_seed0..1.mp4` (both
drawn at the 60 s clock, the stalemate of step 7). Watch with any player (`mpv FILE`, VLC); each
label is the slot, the flight action flown (level, side L/R, power) and the trigger's chance
when above 5%; the line goes to the ship's nearest scored enemy. Re-render:
`CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv run --no-sync python
docs/rewrite/sections/replay/replay.py --seeds 0 1 2 --out DIR` (add `--ships-per-team 1
--no-fields` for 1v1). Seen in them: 4v4 fights resolve; the power label changes nearly every
tick (step 5); in 1v1 the two circle 0.8-1.2 km apart.

The three 4v4 replays were all Blue wins; a seat check (self-play 4v4 with fields, 512 envs x
3,600 ticks): team 0 won 712, team 1 won 703, 28 draws of 1,443 games. No seat bias.

## Step 8: compile check (2026-11-26, laptop 4070, performance profile, load 1.5)

`docs/rewrite/sections/scoreboard/compile_check.py`: **fails**. `act` (the collect half that
holds the scripted call) 5,396 traced operations against the baseline's 1,821 (+196%); total
10,075 (baseline 6,503); cold 289.4 s against the 260 s budget (baseline 231.7 s); `act`'s
backend 82.1 s; 4 graphs, no recompiles, no graph breaks. The baseline was not updated.
