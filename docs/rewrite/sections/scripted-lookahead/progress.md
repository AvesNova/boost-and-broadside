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

# Round 2 (author decisions 2026-11-26)

Laptop RTX 4070, performance profile, nothing else on the GPU. 2,048 environments of 4v4,
fields and fog, after 30 scripted ticks, unless stated.

## R2 step 1: profile (2026-11-26)

`profile_kernels.py` (torch.profiler over 20 compiled calls: wall, GPU kernel time, kernels per
call, each kernel's share), `bench_kernel.py` (the pair kernel alone on captured real inputs,
`triton.testing.do_bench`, registers and spills from the compiled kernel). Nsight Compute is
installed but the driver restricts counters to admin (`RmProfilingAdminOnly: 1`, no sudo), so
occupancy, memory traffic and the transcendental share are **inferred**, not counted.

Measured, round-1 tree:

| stage | wall per call | GPU kernel time | kernels per call |
|---|---|---|---|
| paths (both views, compiled alone) | 7.66 ms | 7.61 ms | 62 |
| whole call | 16.76 ms | 16.66 ms | 71 |
| pair kernel alone (fields 10 / 0) | 9.82 / 7.26 ms | | 1 |

- Launch overhead is not it: wall = GPU time within 1%, 62-71 kernels per call.
- The pair kernel: 168 registers, 4 spills. The cover loop is 2.56 ms of its 9.82 (26%): it ran
  per chunk of 4 of my candidates on a 32-lane warp (7 of 8 lanes idle), 6 times per point.
  Inferred: at 168 registers x 32 threads a SM holds 12 one-warp programs (25% of 48 warps);
  the 21 x 21 table is padded to 24 x 32 lanes (57% useful).
- The paths: the 7.7 ms is inductor's layout, not arithmetic. The stacked tensor had the point
  and the field axes innermost ((E, N, C, P, 12)); inductor wrote it with cat kernels whose
  lanes interleave the 4 points (each warp runs every point's masked branch) and stores at a
  stride of 12 floats. Example: one kernel that only takes cos and sin of 2.75 M angles took
  0.82 ms (inferred about 200 MB of sector traffic at the laptop's ~256 GB/s, for 22 MB of
  useful writes). The useful output is 132 MB (12 fields x 2.75 M), a floor of about 0.5 ms.
  The per-ship work is small; the paths are per ship, but every quantity is per candidate and
  point (32,768 view-ships x 21 x 4) and the medium is evaluated with 10 fields at each point.

Fixed (cheap, obvious):

1. The cover on all candidates at once (one pass per point on the 32 lanes, before the tile
   loop): kernel 9.82 -> 7.29 ms, cover 2.56 -> 0.40 ms, registers 168 -> 128, no spills.
2. Paths laid out (P, E, views x N, fields, C) (every write a run of 21 floats; the kernel's
   field offsets compile-time multiples of C): kernel 7.29 -> 6.50 ms. A field-major layout
   (fields, P, E, views x N, C) was tried first: 8.09 ms, 255 registers (runtime field
   strides), dropped.
3. Tried and dropped: approximate division (`tl.math.fdiv`, ieee off) for the three per-element
   divisions: 7.29 -> 7.24 ms (noise).

After: **whole call 12.96 ms** wall (12.79 GPU, 74 kernels; was 16.76), the pair kernel 6.10 ms
inside it. The paths are still about 6.5 ms inside the call: step 3 (7 sticks at the current
power) cuts their size by 3 and the table by 9, so they were not tuned further at 21 candidates.

## R2 step 2: engagement range term, caution 0 (2026-11-26)

- Tried: McGrew's form in both pair tables (`aggression` x 1/2 (1 + angles) x exp(-|d - R_d| /
  w)): McGrew's angular score in cosines (1 on his tail nose on, 1/2 neutral, 0 he on mine)
  times a range peak; R_d = `desired_range` 0.75 gun range (523 m, inside the gun's full band),
  w = `range_width` 1 gun range (697 m); his mirror (1/2 (1 - angles), his gun range) in S_him,
  so neutral geometry rewards closing for both. Not zero-sum on purpose. `caution` default 0
  (author). Plain twin and kernel alike (the twin test holds).
- Measured (1v1 without fields, self-play, 256 envs x 1,800 ticks, compiled, `measure.py --set`):

| aggression (caution 0) | games | draws | draw share | kills per env-minute |
|---|---|---|---|---|
| 0 (1e-4) | 265 | 153 | 58% | 0.31 |
| **0.5 (default)** | 300 | 106 | 35% | 0.70 |
| 1.0 | 281 | 117 | 42% | 0.57 |
| 2.0 | 289 | 112 | 39% | 0.63 |

  (Round 1, caution -2, no range term: 74% draws over 3,600 ticks, 59% at caution 0.)
- Didn't: the draws fall from 58% to 35-42% and stop there; more aggression does not lower them
  further (0.5 to 2.0 are within the noise of 300 games). The rule-based agent draws 6% here.
  Default 0.5 (LOG law, width 0.5). The rest is looked at again after the power split.

## R2 step 3: power split (2026-11-26)

- Built: the pair table covers the 7 sticks at each ship's table power (7 x 7 per pair). The
  own ship's table power is its pending power (in its observation). An enemy's pending power is
  not observable, so (decided myself) his table power is the power his own outlook scores best
  (from straight flight, engaged, with the yardstick's rows: none of his persona is read). The
  power is chosen per ship from the own-ship terms over the power look (`paths.POWER_LOOK` 1, 2,
  3, 4, 5 s; `paths.power_outlook`: 3 per-ship paths of speed and store only, holding the
  pending stick, from `core.steady_flight` per segment): minus the speed term, plus energy at
  the energy weight while engaged and the disengaged energy weight while not, discounted by
  `power_horizon` 3 s; its own softmax (`power_temperature` 0.05) and stickiness
  (`power_stickiness` 0.3 = 6 x the temperature: indifferent, a switch on about 0.5% of ticks).
  Joint = stick (7) x power (3) x fire (2). Both views are always predicted (with fog off too:
  the enemies' table power differs from the pending one). The seen view's boundary margin is
  the yardstick's (round 1 read the enemy's persona row there).
- Look-ahead points now 0.25, 0.5, 1, 1.5, 2 s (a module setting, `paths.LOOK_AHEAD`, read at
  call time; step 4 and 5 measure it).
- Measured, 1v1 without fields, self-play, 256 envs x 1,800 ticks: **power switches 15.9 ->
  0.31 per ship-second**, 300 games, 97 draws (32%), 0.75 kills per env-minute, teacher entropy
  1.233 nats (the power is now decisive), every one of the 42 actions used (fire 0.00-0.06% of
  ship-ticks per action, as in round 1's 1v1); power shares coast 27%, dive 50%, climb 22%.
- Speed (`speed.py`, compiled, 2,048 envs 4v4, fields, fog): **5.77 ms per call** (was 12.96
  after step 1), peak 135 MiB, cold first call 130.5 s (the extra point and the outlook).
  Profile: 5.66 ms wall, 5.11 GPU, 111 kernels; the pair kernel 1.29 ms, the paths about 3.5 ms.

### The plain PyTorch pair stage against the kernel (coordinator, 2026-11-26)

Standing decision: keep a custom kernel only if a profile shows it pays. Measured at 7 x 7:

| pair stage | call ms | note |
|---|---|---|
| Triton kernel | 5.77 | kernel 1.29 ms of the call |
| plain, points as a Python loop (round 1's twin) | 19.27 | inductor splits the table into 3 kernels of 4.0-4.4 ms that write many per-point intermediates |
| plain, points as a tensor axis summed at the end | 27.50 | one 19.9 ms kernel (worse; reverted) |
| plain, his paths gathered per field (no stacked copy) | (in both rows above) | kept |

The plain pair stage costs about 14.9 ms of a 19.3 ms call against the kernel's 1.3 ms: the
kernel pays clearly (about 11x on the stage, 3.3x on the call) and stays, with its twin test.
Inferred cause: inductor caps a fusion's size (`max_fusion_size`), so the long per-element
expression (two gun solutions, angles, energy, range, contact per point) is cut into kernels
that pass intermediates through memory; not changed (a global inductor setting).

### CUDA graphs (coordinator, 2026-11-26; a measurement, not the default)

`cuda_graph.py`, the scripted call (`act`, both teams) at 2,048 envs 4v4, fields, fog, 50 calls
after 5 warm; two runs (the second on a warm compile cache):

| variant | run 1 ms | run 2 ms | first call (run 1 / run 2) |
|---|---|---|---|
| `compile_dynamic` (lean's), no graph | 5.73 | 5.99 | 115.2 / 46.6 s |
| `torch.compile(dynamic=False)`, no graph | - | 4.09 | - / 46.3 s |
| `mode="reduce-overhead"`, static shapes (CUDA graph) | 3.41 | 3.91 | 60.6 / 6.8 s |

- Most of the gain is the static shapes (5.99 -> 4.09 ms in the same process), the graph
  itself only 4.09 -> 3.91 (launch overhead is small: 111 kernels, wall about GPU time). Its
  output equals the dynamic call's exactly.
- Constraints hit: a hand capture (`torch.cuda.graph` around the dynamic compiled call) fails:
  the call builds small constants with `torch.tensor(..., device=cuda)` (the look-ahead times,
  the level indices), a CPU -> GPU copy that capture refuses unless pinned; reduce-overhead
  lifts them. reduce-overhead needs static shapes (a recompile per environment count / team
  size), its outputs live in the graph's pool (must be cloned before the next call), and the
  uniforms are inputs, so no RNG inside the graph. Not tried on the whole act step.

## R2 step 4: prediction fixes (2026-11-26)

`path_check.py` (the 7 sticks predicted at each power against the real tick holding each of the
21 flight actions for 2 s; 64 ships per speed, straight at 0.6 / 1.0 / 1.5 x corner, half
store; median / 90th percentile), four variants: **A** round 1's prediction (the spring's
target the steady angle at the speed now; points 0.25, 0.5, 1, 2 s), **B** A with the target
recomputed per segment from the segment's start speed (`paths.RETARGET`), **C** B with a point
at 1.5 s (segments at most 0.5 s), **D** C with a point at 0.75 s. (The per-segment spring
restarts from the carried offset and rate; with a fixed target that is the same closed form.)
Full tables: `data/lookahead-r2/path_{A,B,C,D}.md`. Excerpt (1 s and 2 s; without fields every
level but straight, with fields straight and Snap):

| fields | speed | level | error | A 1 s | A 2 s | B 1 s | B 2 s | C 1 s | C 2 s | D 1 s | D 2 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.6 | Sustain | position m | 5.8 / 7.6 | 17.9 / 26.7 | 5.8 / 6.8 | 17.9 / 21.9 | 5.8 / 6.8 | 18.3 / 21.5 | 6.2 / 7.0 | 18.4 / 18.8 |
| none | 0.6 | Sustain | nose deg | 3.4 / 3.5 | 4.2 / 4.4 | 3.4 / 3.4 | 4.2 / 4.2 | 3.4 / 3.4 | 4.2 / 4.2 | 3.4 / 3.4 | 4.2 / 4.2 |
| none | 0.6 | Max | position m | 4.9 / 8.1 | 11.5 / 23.5 | 4.9 / 8.2 | 11.4 / 24.1 | 4.9 / 8.2 | 11.3 / 22.9 | 5.0 / 8.3 | 11.7 / 23.1 |
| none | 0.6 | Max | nose deg | 2.4 / 4.2 | 3.2 / 7.7 | 2.4 / 4.2 | 3.2 / 7.5 | 2.4 / 4.2 | 3.3 / 6.7 | 2.6 / 4.3 | 3.5 / 6.7 |
| none | 0.6 | Snap | position m | 18.6 / 18.6 | 18.4 / 29.6 | 18.6 / 18.6 | 19.6 / 29.6 | 18.6 / 18.6 | 20.0 / 28.7 | 18.3 / 18.3 | 19.7 / 28.3 |
| none | 0.6 | Snap | nose deg | 6.3 / 6.3 | 29.9 / 29.9 | 6.3 / 6.3 | 2.1 / 4.9 | 6.3 / 6.3 | 11.2 / 11.2 | 6.9 / 6.9 | 16.7 / 16.7 |
| none | 1.0 | Sustain | position m | 1.8 / 2.1 | 4.1 / 6.8 | 1.7 / 2.0 | 5.3 / 5.7 | 1.7 / 2.0 | 5.2 / 5.5 | 1.8 / 2.0 | 4.5 / 5.4 |
| none | 1.0 | Sustain | nose deg | 0.5 / 0.8 | 1.7 / 3.6 | 0.6 / 0.7 | 1.5 / 3.3 | 0.6 / 0.7 | 0.4 / 1.9 | 0.3 / 0.5 | 0.2 / 1.6 |
| none | 1.0 | Max | position m | 6.8 / 7.4 | 13.7 / 14.9 | 7.5 / 7.5 | 15.0 / 15.1 | 7.5 / 7.5 | 14.6 / 15.1 | 7.6 / 7.6 | 14.9 / 14.9 |
| none | 1.0 | Max | nose deg | 2.9 / 2.9 | 1.5 / 1.5 | 2.2 / 2.2 | 0.0 / 0.2 | 2.2 / 2.2 | 1.5 / 1.5 | 2.1 / 2.1 | 2.0 / 2.0 |
| none | 1.0 | Snap | position m | 4.2 / 4.2 | 15.9 / 15.9 | 5.2 / 5.2 | 11.2 / 11.2 | 5.2 / 5.2 | 14.8 / 14.8 | 5.4 / 5.4 | 14.5 / 14.5 |
| none | 1.0 | Snap | nose deg | 5.7 / 5.7 | 24.1 / 24.1 | 4.3 / 4.3 | 14.0 / 14.0 | 4.3 / 4.3 | 6.7 / 6.7 | 1.1 / 1.1 | 7.2 / 7.2 |
| none | 1.5 | Sustain | position m | 0.1 / 0.9 | 2.8 / 4.9 | 0.1 / 0.5 | 4.9 / 12.6 | 0.1 / 0.5 | 1.3 / 13.1 | 0.1 / 0.6 | 1.2 / 7.8 |
| none | 1.5 | Sustain | nose deg | 0.0 / 0.9 | 0.0 / 3.4 | 0.0 / 1.4 | 0.0 / 4.8 | 0.0 / 1.4 | 0.0 / 3.2 | 0.0 / 0.8 | 0.0 / 2.0 |
| none | 1.5 | Max | position m | 1.8 / 2.6 | 2.8 / 8.4 | 1.8 / 2.7 | 2.5 / 8.8 | 1.8 / 2.7 | 2.4 / 7.8 | 1.8 / 2.7 | 3.0 / 7.1 |
| none | 1.5 | Max | nose deg | 0.4 / 0.8 | 1.2 / 1.6 | 0.2 / 0.7 | 0.9 / 1.4 | 0.2 / 0.7 | 0.3 / 0.7 | 0.0 / 0.5 | 0.2 / 0.6 |
| none | 1.5 | Snap | position m | 4.7 / 5.8 | 7.6 / 15.4 | 4.6 / 5.9 | 6.9 / 15.7 | 4.6 / 5.9 | 6.6 / 14.1 | 4.5 / 5.9 | 7.3 / 13.4 |
| none | 1.5 | Snap | nose deg | 0.2 / 1.4 | 0.9 / 2.1 | 0.3 / 1.3 | 0.6 / 1.9 | 0.3 / 1.3 | 0.1 / 1.2 | 0.5 / 1.1 | 0.2 / 1.1 |
| fields | 0.6 | straight | position m | 0.1 / 7.1 | 0.3 / 28.6 | 0.1 / 7.1 | 0.3 / 28.6 | 0.1 / 7.1 | 0.3 / 20.4 | 0.0 / 6.1 | 0.2 / 19.7 |
| fields | 0.6 | straight | nose deg | 0.0 / 2.5 | 0.0 / 6.0 | 0.0 / 2.5 | 0.0 / 6.0 | 0.0 / 2.5 | 0.0 / 2.4 | 0.0 / 1.3 | 0.0 / 3.8 |
| fields | 0.6 | Snap | position m | 18.6 / 25.6 | 18.4 / 41.1 | 18.6 / 25.6 | 19.6 / 41.1 | 18.6 / 25.6 | 20.0 / 40.6 | 18.3 / 25.3 | 19.7 / 36.9 |
| fields | 0.6 | Snap | nose deg | 6.3 / 9.5 | 15.5 / 29.9 | 6.3 / 8.4 | 4.9 / 15.9 | 6.3 / 8.4 | 11.2 / 27.5 | 6.9 / 14.1 | 16.7 / 20.5 |
| fields | 1.0 | straight | position m | 0.1 / 17.8 | 0.4 / 64.6 | 0.1 / 17.8 | 0.4 / 64.6 | 0.1 / 17.8 | 0.4 / 53.4 | 0.1 / 14.0 | 0.3 / 41.8 |
| fields | 1.0 | straight | nose deg | 0.0 / 5.0 | 0.0 / 9.3 | 0.0 / 5.0 | 0.0 / 9.3 | 0.0 / 5.0 | 0.0 / 7.4 | 0.0 / 2.1 | 0.0 / 4.9 |
| fields | 1.0 | Snap | position m | 4.2 / 16.2 | 15.9 / 68.9 | 5.2 / 17.7 | 11.2 / 66.2 | 5.2 / 17.7 | 14.8 / 50.9 | 5.4 / 12.2 | 14.5 / 45.6 |
| fields | 1.0 | Snap | nose deg | 5.7 / 12.3 | 19.1 / 26.9 | 4.3 / 10.3 | 14.0 / 21.8 | 4.3 / 10.3 | 6.7 / 14.9 | 1.1 / 5.9 | 7.2 / 12.1 |
| fields | 1.5 | straight | position m | 0.1 / 41.9 | 4.9 / 123.3 | 0.1 / 41.9 | 4.9 / 123.3 | 0.1 / 41.9 | 2.2 / 81.6 | 0.1 / 24.3 | 2.2 / 63.7 |
| fields | 1.5 | straight | nose deg | 0.0 / 6.2 | 0.0 / 13.4 | 0.0 / 6.2 | 0.0 / 13.4 | 0.0 / 6.2 | 0.0 / 10.9 | 0.0 / 2.3 | 0.0 / 9.9 |
| fields | 1.5 | Snap | position m | 4.7 / 47.5 | 15.4 / 126.1 | 4.6 / 49.9 | 15.7 / 126.1 | 4.6 / 49.9 | 13.7 / 102.0 | 4.5 / 30.8 | 11.2 / 66.0 |
| fields | 1.5 | Snap | nose deg | 0.5 / 10.2 | 2.1 / 21.6 | 0.6 / 10.3 | 1.9 / 24.3 | 0.6 / 10.3 | 1.2 / 17.9 | 0.7 / 6.2 | 1.1 / 15.0 |

Cost (`speed.py`, compiled, 2,048 envs 4v4, fields, fog, 200 calls):

| variant | ms per call | peak MiB |
|---|---|---|
| A (round 1's) | 4.72 | 80 |
| B (retarget) | 4.98 | 103 |
| C (retarget, 5 points) | 5.90 | 135 |
| D (retarget, 6 points) | 7.17 | 163 |

- Retarget (A -> B, +0.26 ms): Snap's nose at 2 s, 0.6 x corner 29.9 -> 2.1 deg, corner speed
  24.1 -> 14.0 deg; Snap's position at 2 s at corner 15.9 -> 11.2 m. Snap at 0.6 x corner is
  still 18.6 m and 6.3 deg off at 1 s in every variant: the first segment's error (the path
  rate taken linear in the angle, which past the stall it is not), not the target. **Kept.**
- Shorter steps with fields (B -> C, +0.92 ms, +18%): the 90th percentile position at 2 s,
  straight at 1.5 x corner 123 -> 82 m, at corner 65 -> 53 m, Snap 126 -> 102 m; Snap's nose at
  2 s at corner 14.0 -> 6.7 deg, but at 0.6 x corner 2.1 -> 11.2 deg (the extra segment boundary
  re-linearises the path rate). D (+1.27 ms more): 82 -> 64 m, 53 -> 42 m. **C kept** (decided
  myself: at most 0.5 s per segment, the same rule the horizon sweep uses); D's further 20 m at
  the 90th percentile is not worth another 22%.

## R2 step 5: horizon sweep (2026-11-26)

The stick look-ahead's last point, segments at most 0.5 s: 1 s (0.25, 0.5, 1), 1.5 s (+1.5),
2 s (+1.5, 2), 3 s (+1.5, 2, 2.5, 3); the discount `horizon` stays 1.5 s. Every 1v1 run of
this table is **fog on** (the author's fog-off rule came after it; the chosen horizon's 1v1
numbers are re-run fog off in step 6). Self-play 1v1 without fields 256 envs x 1,800 ticks;
against the rule-based agent 512 envs x 1,800 ticks, seats alternated (W / D / L by the
look-ahead agent; the rule-based agent is mistuned for Sustain / Max / Snap, a weak
reference); merge turn 1v1 without fields (`merge_turn.py --only 1v1-nofields`, seconds from
the closest approach); cost `speed.py` 100 calls, 2,048 envs 4v4, fields, fog. Duels: the 2 s
agent against each other horizon on the other team, seats alternated, 512 envs x 1,800 ticks.
Raw: `data/lookahead-r2/sweep/`.

| look-ahead | 1v1 self-play games / draws (share) | kills per env-min | vs rule-based 4v4 fields W / D / L | vs rule-based 1v1 W / D / L | merge t30 (med / 75 / 90) | t90 | slow | ms per call | peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| 1 s | 306 / 106 (35%) | 0.77 | 630 / 4 / 39 | 192 / 287 / 145 | 0.87 / 1.33 / 1.74 | 2.13 / 2.70 / 3.53 | 143 of 896 | 3.87 | 78 |
| 1.5 s | 305 / 97 (32%) | 0.78 | 636 / 2 / 41 | 210 / 316 / 82 | 0.70 / 1.10 / 1.57 | 1.87 / 2.40 / 3.17 | 103 of 918 | 5.15 | 107 |
| 2 s | 300 / 97 (32%) | 0.75 | 642 / 3 / 33 | 258 / 309 / 44 | 0.63 / 1.00 / 1.47 | 1.83 / 2.33 / 3.00 | 76 of 860 | 6.33 | 135 |
| 3 s | 313 / 97 (31%) | 0.80 | 611 / 3 / 37 | 256 / 307 / 21 | 0.63 / 0.90 / 1.40 | 1.83 / 2.30 / 2.85 | 69 of 966 | 7.98 | 188 |

| duel (2 s against) | 4v4 fields W / D / L (2 s's) | 1v1 no fields W / D / L |
|---|---|---|
| 1 s | 395 / 1 / 390 | 225 / 186 / 183 |
| 1.5 s | 388 / 5 / 416 | 226 / 185 / 193 |
| 3 s | 485 / 1 / 334 | 181 / 194 / 226 |

**Chosen: 2 s** (the default already). Against 1.5 s it loses fewer games to the rule-based
agent (33 against 41 of about 680 in 4v4; 44 against 82 in 1v1), turns sooner after the merge
(0.63 against 0.70 s median to 30 deg; 76 against 103 slow), for +1.2 ms; their 4v4 duel is
388 / 5 / 416 (about one standard error, 14 games). 3 s beats 2 s in the 1v1 duel (226 to 181)
but loses the 4v4 duel clearly (334 to 485), costs +1.65 ms and 40% more memory, and its merge
turn and draw rate are the same as 2 s. 1 s is cheapest but slowest at the merge and loses the
most 1v1 games to the rule-based agent. The 1v1 draw share does not move with the horizon
(31-35%): the stalemate is not a horizon problem.

## R2 step 6: re-measure (2026-11-27, after merging core: hit damage halved 21 -> 10.5)

Everything in this section is on the merged tree (hit damage 10.5) with the defaults: 2 s
look-ahead (0.25, 0.5, 1, 1.5, 2), nose retarget, power look 1 / 2.5 / 5 s (cut from 5 points
to 3 after the first compile check: two fewer steady-flight traces in the graph, decided
myself). 1v1 runs are **fog off, no fields** (author); 4v4 fog on. Raw:
`data/lookahead-r2/final/` (a first pass with the old damage and the 5-point power look is
kept in `data/lookahead-r2/final_before_damage_halved/`; every number before this section was
taken with hit damage 21).

Cost (`speed.py`, 2,048 envs 4v4, fields, fog, 100 calls, warm cache):

| call | ms per call | peak MiB above the state |
|---|---|---|
| look-ahead (both teams) | 5.70 | 134 |
| rule-based | 0.35 | 3 |
| policy forward (3,072 seats) | 21.69 | 417 |

The look-ahead is 26% of the policy's forward (round 1: 16.6 ms, 72%).

Compile check (`scoreboard/compile_check.py`, cold, baseline not updated): **fails**. `act`
7,864 traced operations against the baseline's 1,821 (round 1: 5,396; before the power look
was cut 8,856); total 12,545; cold **333.0 s** against the 260 s budget (round 1 289.4 s;
before the cut 357.4 s); `act`'s backend 100.8 s; 4 graphs, no recompiles, no graph breaks.
The cause: every steady-flight evaluation is about 500 traced operations and the call now
has 8 (5 path segments, 3 power-look segments; round 1 had 4). Not fixed: reported.

Games (256 envs x 3,600 ticks self-play, re-dealt; spread 1 = personas drawn per ship at
every deal, the full persona box; training's `persona_spread` default is 0):

| run | games | draws | kills per env-min | entropy | speed/corner | store | power sw/s | stick sw/s | TV joint / stick / power per tick |
|---|---|---|---|---|---|---|---|---|---|
| selfplay_1v1_f0_fogoff | 796 | 8 | 1.34 | 1.008 | 0.75 | 0.06 | 0.26 | 11.65 | 0.156 / 0.147 / 0.011 |
| selfplay_1v1_f0_fogoff_spread1 | 910 | 17 | 1.62 | 1.022 | 0.76 | 0.11 | 0.56 | 11.32 | 0.155 / 0.138 / 0.021 |
| selfplay_4v4_f0 | 961 | 2 | 10.83 | 0.799 | 0.84 | 0.09 | 0.26 | 9.41 | 0.149 / 0.137 / 0.011 |
| selfplay_4v4_f10 | 802 | 2 | 9.29 | 0.884 | 0.86 | 0.12 | 0.25 | 10.39 | 0.158 / 0.147 / 0.010 |
| selfplay_4v4_f10_spread1 | 760 | 4 | 9.02 | 0.916 | 0.86 | 0.16 | 0.53 | 10.39 | 0.158 / 0.140 / 0.020 |
| old_1v1_f0_fogoff | 612 | 36 | 0.89 | 0.000 | 0.47 | 0.22 | 4.80 | 9.10 | - |

versus_4v4_f10: W/D/L 1433 / 5 / 31 of 1469

versus_1v1_f0_fogoff: W/D/L 2298 / 0 / 139 of 2437

(TV: the mean total variation between a living ship's consecutive-tick distributions;
switches: sampled action changes per ship-second.) The rule-based row: its own self-play, 1v1
fog off. Against the rule-based agent (seats alternated, 512 envs x 3,600 ticks; the rule-based
agent is mistuned for Sustain / Max / Snap, a weak reference): **4v4 fields 1,433 / 5 / 31**
(W / D / L of 1,469), **1v1 fog off 2,298 / 0 / 139** (of 2,437).

Merge turn (`merge_turn.py`, seconds from the closest approach, median / 75th / 90th; slow =
over 1.5 s to 30 deg or not reached):

| setup | agent | aircraft | to 30 deg | to 90 deg | slow |
|---|---|---|---|---|---|
| 1v1 fog off, no fields | look-ahead | 1,828 | 0.73 / 1.17 / 1.60 | 2.03 / 2.53 / 3.33 | 231 |
| 1v1 fog off, no fields | rule-based | 1,130 | 0.53 / 1.10 / 1.67 | 2.13 / 3.80 / 5.10 | 148 |
| 4v4 fields, fog | look-ahead | 1,632 | 0.70 / 1.20 / 2.10 | 2.10 / 3.37 / 5.48 | 267 |
| 4v4 fields, fog | rule-based | 818 | 0.67 / 1.27 / 2.02 | 2.93 / 4.63 / 7.06 | 157 |

Usage of the 42 actions (share of living ship-ticks, no fire / fire):

| stick | power | selfplay_1v1_f0_fogoff (no fire / fire) | selfplay_4v4_f0 (no fire / fire) | selfplay_4v4_f10 (no fire / fire) |
|---|---|---|---|---|
| straight | coast | 6.64% / 0.04% | 9.91% / 0.29% | 8.73% / 0.33% |
| straight | dive | 8.18% / 0.07% | 16.01% / 0.51% | 14.28% / 0.55% |
| straight | climb | 0.78% / 0.00% | 1.87% / 0.04% | 2.44% / 0.08% |
| Sustain R | coast | 6.57% / 0.08% | 5.84% / 0.28% | 5.79% / 0.34% |
| Sustain R | dive | 9.42% / 0.10% | 11.65% / 0.53% | 10.92% / 0.53% |
| Sustain R | climb | 1.05% / 0.01% | 0.74% / 0.03% | 1.31% / 0.07% |
| Sustain L | coast | 6.71% / 0.07% | 5.80% / 0.28% | 5.87% / 0.31% |
| Sustain L | dive | 9.58% / 0.11% | 11.43% / 0.49% | 10.79% / 0.54% |
| Sustain L | climb | 1.00% / 0.01% | 0.75% / 0.03% | 1.29% / 0.09% |
| Max R | coast | 6.01% / 0.07% | 3.35% / 0.13% | 3.34% / 0.13% |
| Max R | dive | 10.24% / 0.11% | 7.42% / 0.26% | 6.93% / 0.24% |
| Max R | climb | 1.52% / 0.01% | 0.70% / 0.02% | 0.85% / 0.03% |
| Max L | coast | 6.12% / 0.06% | 3.38% / 0.13% | 3.45% / 0.13% |
| Max L | dive | 10.64% / 0.10% | 7.23% / 0.26% | 7.05% / 0.25% |
| Max L | climb | 1.53% / 0.01% | 0.72% / 0.02% | 0.86% / 0.03% |
| Snap R | coast | 1.99% / 0.15% | 1.26% / 0.10% | 1.68% / 0.11% |
| Snap R | dive | 3.57% / 0.22% | 2.97% / 0.21% | 3.56% / 0.22% |
| Snap R | climb | 0.59% / 0.03% | 0.38% / 0.02% | 0.66% / 0.03% |
| Snap L | coast | 2.01% / 0.14% | 1.31% / 0.10% | 1.68% / 0.11% |
| Snap L | dive | 3.65% / 0.23% | 2.95% / 0.21% | 3.50% / 0.22% |
| Snap L | climb | 0.55% / 0.03% | 0.40% / 0.02% | 0.66% / 0.03% |

selfplay_1v1_f0_fogoff: entropy 1.008 nats, kills per env-minute 1.34, unused actions 0, under 0.1% 14, games 796 (draws 8)

selfplay_4v4_f0: entropy 0.799 nats, kills per env-minute 10.83, unused actions 0, under 0.1% 8, games 961 (draws 2)

selfplay_4v4_f10: entropy 0.884 nats, kills per env-minute 9.29, unused actions 0, under 0.1% 7, games 802 (draws 2)

- Every action is used in every setup; the climb is the least used power (3-6% of
  ship-ticks), dive the most (55-65%), and the mean store sits low (0.06-0.16 of full; round
  1: 0.21-0.52): the power look spends the store on speed. Teacher entropy 0.80-1.01 nats
  (round 1 1.95-2.34: the power is now decided, not near uniform).
- Power: the distribution barely moves (TV 0.010-0.021 per tick) and samples switch 0.25-0.56
  times per ship-second (round 1: 15.9). **The stick distribution moves about 0.14-0.15 TV per
  tick** (the stick marginal shifts by 14% of its mass every tick on average): flagged per the
  author's rule (the distribution itself jumps, not only the samples); not tuned.
- 1v1 fog off: 1% draws (8 of 796), yardstick mirror; 2% with personas (17 of 910). The fog-on
  draws of steps 2-5 (31-35%) were mostly lost sight, not a refusal to engage.
- Personas (spread 1): kills per minute 1.62 vs 1.34 in 1v1, 9.02 vs 9.29 in 4v4 fields; power
  switches double (0.53-0.56 per second), still far below round 1.

## R2 step 7: replays (2026-11-27)

`~/avesnova/boost-and-broadside/data/replays-2026-11-26-lookahead-r2/` (look-ahead on both
teams, CPU, batch 1, merged tree): `scripted1v1_nofields_fogoff_seed0..2.mp4` (the author's
rule: every 1v1 fog off and no fields, so the brief's "1v1 with fields" replay is a third
fog-off no-fields seed instead), `scripted4v4_seed0..1.mp4` (default map, fields, fog). Results:
1v1 seed 0 Blue wipe at 11.6 s, seed 1 to the clock (Blue on shield), seed 2 Red wipe at 46.8 s;
4v4 seed 0 Red wipe at 36.7 s, seed 1 Blue wipe at 39.1 s. Re-render:
`CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv run --no-sync python
docs/rewrite/sections/replay/replay.py --seeds 0 1 2 --ships-per-team 1 --no-fields --no-fog
--out DIR` (new `--no-fog`). Seen on contact sheets (a frame every 3-5 s; I cannot play the
video): the power label holds (mostly dive) instead of changing every tick; in 1v1 seed 1 the
two open to about 2 km for the first ~20 s before closing, then fight slowly at 0.4-0.6 x
corner speed near the stall with the store empty; 4v4 fights resolve in pairs and threes.

## R2 step 8: the zero-sum test

`lean/learn/tests/test_rewards.py::test_the_real_game_gives_zero_sum_levels` passes on this
tree (since step 3; also after the merge), with the quick tier of lean/scripted and lean/learn.

## R2 step 5b: 4 s and 6 s horizons (author, 2026-11-27; merged tree, damage 10.5)

The author's points (segments up to 2 s): 4 s = 0.25, 0.5, 1, 2, 4; 6 s = ..., 4, 6. The GPU
time per horizon was about 17 minutes, so only **6 s** was run (the author's rule for over 30
minutes); 4 s has path errors only. 1v1 fog off, no fields; 4v4 fields, fog; 512 envs x 1,800
ticks (the 2 s rows from step 6, 3,600 ticks).

| | 2 s (default) | 6 s |
|---|---|---|
| vs rule-based 4v4 fields W / D / L | 1,433 / 5 / 31 (2.1% lost) | 556 / 1 / 10 (1.8% lost) |
| vs rule-based 1v1 W / D / L | 2,298 / 0 / 139 (5.7% lost) | 860 / 0 / 43 (4.8% lost) |
| duel, 2 s's W / D / L, 4v4 fields | | 514 / 7 / 195 (2 s wins 72%) |
| duel, 2 s's W / D / L, 1v1 | | 781 / 2 / 160 (2 s wins 83%) |
| merge turn 1v1, to 30 deg (med / 75 / 90) | 0.73 / 1.17 / 1.60 | 0.77 / 1.20 / 1.63 |
| to 90 deg | 2.03 / 2.53 / 3.33 | 2.07 / 2.59 / 3.33 |
| slow | 231 of 1,828 | 238 of 1,932 |
| ms per call (100 calls) / peak MiB | 5.70 / 134 | 6.87 / 161 |

Path prediction at 2, 4 and 6 s (`path_check.py --look-ahead 0.25,0.5,1,2,4,6`, median /
90th percentile; `data/lookahead-r2/path_E.md` (4 s), `path_F.md` (6 s)):

| fields | speed / corner | level | error | 2 s | 4 s | 6 s |
|---|---|---|---|---|---|---|
| no | 0.6 | straight | position m | 0.3 / 1.2 | 16.3 / 27.1 | 13.0 / 98.7 |
| no | 0.6 | straight | nose deg | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| no | 0.6 | Sustain | position m | 17.9 / 21.9 | 25.9 / 37.8 | 40.5 / 41.1 |
| no | 0.6 | Sustain | nose deg | 4.2 / 4.2 | 4.9 / 5.4 | 4.9 / 5.7 |
| no | 0.6 | Max | position m | 11.4 / 24.1 | 21.0 / 52.8 | 22.0 / 47.5 |
| no | 0.6 | Max | nose deg | 3.2 / 7.5 | 4.1 / 8.9 | 4.9 / 11.9 |
| no | 0.6 | Snap | position m | 19.6 / 29.6 | 66.6 / 66.6 | 91.4 / 91.5 |
| no | 0.6 | Snap | nose deg | 2.1 / 4.9 | 11.5 / 11.5 | 50.7 / 50.7 |
| no | 1.0 | straight | position m | 0.4 / 0.8 | 33.0 / 44.2 | 111.9 / 125.1 |
| no | 1.0 | straight | nose deg | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| no | 1.0 | Sustain | position m | 5.3 / 5.7 | 35.3 / 60.0 | 75.8 / 186.1 |
| no | 1.0 | Sustain | nose deg | 1.5 / 3.3 | 2.5 / 4.5 | 5.2 / 21.5 |
| no | 1.0 | Max | position m | 15.0 / 15.1 | 36.2 / 36.6 | 39.8 / 86.2 |
| no | 1.0 | Max | nose deg | 0.0 / 0.2 | 5.1 / 5.1 | 2.8 / 30.9 |
| no | 1.0 | Snap | position m | 11.2 / 11.2 | 110.4 / 110.4 | 216.9 / 266.5 |
| no | 1.0 | Snap | nose deg | 14.0 / 14.0 | 77.1 / 77.1 | 172.5 / 176.8 |
| no | 1.5 | straight | position m | 2.8 / 4.9 | 24.0 / 34.2 | 42.3 / 59.2 |
| no | 1.5 | straight | nose deg | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| no | 1.5 | Sustain | position m | 4.9 / 12.6 | 34.2 / 50.0 | 59.2 / 71.9 |
| no | 1.5 | Sustain | nose deg | 0.0 / 4.8 | 0.0 / 0.6 | 0.0 / 4.3 |
| no | 1.5 | Max | position m | 2.5 / 8.8 | 5.8 / 39.9 | 41.3 / 112.9 |
| no | 1.5 | Max | nose deg | 0.9 / 1.4 | 5.3 / 5.3 | 11.7 / 12.6 |
| no | 1.5 | Snap | position m | 6.9 / 15.7 | 12.9 / 48.9 | 53.0 / 123.0 |
| no | 1.5 | Snap | nose deg | 0.6 / 1.9 | 5.5 / 5.5 | 13.0 / 13.0 |
| yes | 0.6 | straight | position m | 0.3 / 28.6 | 16.3 / 135.0 | 39.8 / 284.8 |
| yes | 0.6 | straight | nose deg | 0.0 / 6.0 | 0.0 / 10.7 | 0.0 / 20.1 |
| yes | 0.6 | Sustain | position m | 17.9 / 32.8 | 25.9 / 106.6 | 40.5 / 202.2 |
| yes | 0.6 | Sustain | nose deg | 4.2 / 10.6 | 4.9 / 21.7 | 5.0 / 30.5 |
| yes | 0.6 | Max | position m | 11.4 / 30.0 | 21.0 / 95.0 | 39.2 / 165.2 |
| yes | 0.6 | Max | nose deg | 3.2 / 11.7 | 4.7 / 31.7 | 6.6 / 38.7 |
| yes | 0.6 | Snap | position m | 19.6 / 41.1 | 66.6 / 131.4 | 91.4 / 196.2 |
| yes | 0.6 | Snap | nose deg | 4.9 / 15.9 | 11.5 / 39.1 | 34.5 / 52.4 |
| yes | 1.0 | straight | position m | 0.4 / 64.6 | 44.2 / 257.4 | 117.1 / 451.6 |
| yes | 1.0 | straight | nose deg | 0.0 / 9.3 | 0.0 / 21.9 | 1.3 / 42.9 |
| yes | 1.0 | Sustain | position m | 5.3 / 71.4 | 40.5 / 257.2 | 84.1 / 449.3 |
| yes | 1.0 | Sustain | nose deg | 1.5 / 15.1 | 4.4 / 36.7 | 9.4 / 52.0 |
| yes | 1.0 | Max | position m | 15.0 / 50.0 | 36.2 / 143.1 | 41.3 / 231.7 |
| yes | 1.0 | Max | nose deg | 0.2 / 14.4 | 5.1 / 38.9 | 6.6 / 51.9 |
| yes | 1.0 | Snap | position m | 11.2 / 66.2 | 110.4 / 202.4 | 216.9 / 340.6 |
| yes | 1.0 | Snap | nose deg | 14.0 / 21.8 | 58.1 / 77.1 | 93.8 / 176.3 |
| yes | 1.5 | straight | position m | 4.9 / 123.3 | 39.7 / 448.7 | 120.6 / 919.6 |
| yes | 1.5 | straight | nose deg | 0.0 / 13.4 | 0.5 / 41.8 | 4.1 / 53.1 |
| yes | 1.5 | Sustain | position m | 12.6 / 128.5 | 50.0 / 443.9 | 143.0 / 843.7 |
| yes | 1.5 | Sustain | nose deg | 1.4 / 15.7 | 2.2 / 45.2 | 5.2 / 60.8 |
| yes | 1.5 | Max | position m | 8.8 / 120.0 | 39.9 / 349.4 | 112.9 / 497.5 |
| yes | 1.5 | Max | nose deg | 1.4 / 22.9 | 5.3 / 50.6 | 12.6 / 78.5 |
| yes | 1.5 | Snap | position m | 15.7 / 126.1 | 48.9 / 355.1 | 123.0 / 471.9 |
| yes | 1.5 | Snap | nose deg | 1.9 / 24.3 | 5.5 / 50.3 | 13.0 / 86.0 |

As expected, a held Snap is far off by 4-6 s (corner speed: nose 77 deg at 4 s, 173 deg at 6 s;
position 110 / 217 m), and with fields the 90th percentiles reach 450-920 m at 6 s.
**2 s stays the default**: 6 s loses both head-to-head duels heavily (2 s wins 72% in 4v4 and
83% in 1v1), is no better against the rule-based agent (within noise) or at the merge, costs
+1.2 ms, and its later points are mostly prediction error.

## Round 3: planned, on hold (coordinator, 2026-11-27)

The author watched the round-2 replays ("looked really good", damage halved) and approved a round 3
in principle; it waits while the hit model (hit radius, core, dispersion, a possible deflection
field) is discussed. Round 3, with a fresh agent:
1. A steady-flight lookup table built once from the game's own functions (speed x stick level x
   power; the agent interpolates, as the Sustain table does), to cut the compile (cold 333 s
   against the 260 s budget; `act` 7,864 traced operations) and the path cost.
2. The stick distribution's tick-to-tick change (total variation 0.14-0.15) measured after
   grouping actions by effect (left / straight / right, steady turn rate), to tell harmless
   sliding between equivalent actions (Sustain and Max mid-turn) from real jumping.
3. Power and energy: the store runs nearly empty (mean 0.06-0.16 of full; dive 55-65% of
   ship-ticks, climb 3-6%): weight the store more, weaken the preferred-speed term, compare the
   store with round 1 (0.21-0.52); check the slow near-stall empty-store 1v1 (round-2 replay
   1v1 seed 1).
4. Final pass: remove the rule-based agent, tests, compile check, docs; merge into core after the
   author watches the replays.
Author notes for it: 1v1 mirror draws are not a target (near-optimal play may draw); judge flicker
on the distribution, not on sampled actions; 1v1 tests run with fog off and no fields.

# Round 3 (author's go 2026-11-27 ~01:30; overnight plan docs/rewrite/overnight-2026-11-27.md, stream A)

Laptop RTX 4070, performance profile; another agent shares the machine overnight (one GPU job at a
time via the lock, but its CPU work overlaps: load averages are given with the compile times).

## R3 step 1: steady-flight table (2026-11-27 01:35-02:05)

- Tried: `core.steady_flight` tabulated once per ship type with the card
  (`core.steady_characteristics`, `_steady_table`): (type, level 4, power 3, store 17, speed 400,
  4 outputs) = rate, d speed / dt, d store / dt, angle. Speeds on half Sustain's step (1.47 m/s)
  from a quarter of stall speed, nodes on Sustain's; the store enters only through its edge rooms
  (`core.store_room`, factored out of `_setting`), so its axis is 9 sink-room nodes then 9
  top-room nodes (shared middle). Read by `core.steady_lookup` (bilinear, one gather of the 4
  corner rows). The table is per type, not per slot: a new `Constants.per_type` dict (1.3 MB per
  type; dealt per slot it would be 1.3 MB x every slot). Ship type from `State.type`. The 5 path
  segments and the 3 power-look segments read it; no steady-flight evaluation is left in the graph
  (`current_path_rate` still runs the flight functions once).
- Table against the direct `steady_flight` (200,000 random points, speeds 30-550 m/s, a quarter
  of them in the empty band): median errors ~1e-6 rad/s (rate), ~3e-4 m/s^2 (speed rate); 99th
  percentiles: rate 0.6-2.0e-3 rad/s, speed rate 0.02-0.28 m/s^2, store rate 0.8-2.2 /s, angle
  1.3e-4 rad (Snap 2.6e-3). Worst points: Snap diving below stall speed inside the empty band
  (angle up to 0.16 rad, speed rate 6.4 m/s^2), where the steady angle jumps with the room. With
  5 rooms per band instead of 9 the worst angle was 0.28 rad; 17 rooms 0.07 rad.
- Path prediction (`path_check.py`, 64 deals, against the tick): unchanged within 0.3 deg / 0.1 m
  in every cell (`data/lookahead-r3/path_before.md` vs `path_step1.md`).
- Compile check (cold, baseline not updated; load average 2.9 at start, 5.4 at the end): `act`
  **4,494** traced operations (round 2: 7,864), backend 73.7 s (100.8 s); cold **311.8 s**
  (333.0 s). Still over the 260 s budget: **fails**. Traced operations of `scripted.decide` by
  innermost function after a second trim of the lookup (4v4, fields; 3,368 in all): medium 456,
  predict 425, steady_lookup 321, nose_response 302, smootherstep 238 (medium's and the store
  rooms'), _contact 135, _trigger 104. The flight functions are out; what is left is spread over
  the unrolled segment loop (5 x medium, 5 x nose spring, 8 x lookup). Reaching 260 s needs a
  structural change (the paths in one kernel, as the pair stage is, or fewer segments): not done,
  a question for the author.
- Next: step 2 (grouped distribution check).

## R3 step 2: the stick distribution grouped by effect (2026-11-27 ~02:00)

`measure.py` now also reports, per living ship and tick, the total variation of the stick
distribution after grouping by side (left / straight / right), and the 1-Wasserstein distance
between the two ticks' stick distributions on the 7 sticks' signed steady turn rates (at the
ship's speed, store and pending power now, from the game's steady table; in deg/s, and over the
span of the 7 rates, a mass fraction times a rate fraction comparable to the total variation).
Tree 0b7bad03 (step 1 + the hit model through `core.hit_damage`), yardstick, 256 envs x 1,800
ticks, compiled, finished games re-dealt; 1v1 fog off no fields, 4v4 fields and fog. Raw:
`data/lookahead-r3/runs/A_*.json`.

| setup | stick TV (raw) | side TV | rate W1 deg/s | rate W1 of span |
|---|---|---|---|---|
| 1v1 | 0.146 | 0.068 | 4.41 | 0.042 |
| 4v4 | 0.148 | 0.089 | 5.13 | 0.045 |

- About half of the raw 0.15 is mass sliding between actions of the same side (Sustain, Max,
  Snap of one direction); 0.07-0.09 of the mass changes side per tick. Weighted by what the
  actions do, the distribution's expected-turn movement is 4-5% of the rate span per tick
  (4.4-5.1 deg/s). Reported only; nothing tuned for it (author: judge flicker on distributions).
- Other numbers of these runs (for step 3's baseline): 1v1 367 games, 5 draws, 1.31 kills per
  env-minute, entropy 1.022, store 0.07, coast / dive / climb 37 / 57 / 6%; 4v4 363 games, 0
  draws, 9.26 kills per env-minute, entropy 0.887, store 0.12, 31 / 60 / 8%.

## R3 step 3: power and energy (2026-11-27 02:00-02:50)

- Tried: a new row `store_weight` (the store itself in full stores per second, in the power
  choice, on top of the energy term: diving trades store for speed at almost no change in total
  energy, so the energy term never held the store) and a weaker speed term (`speed_weight` 0.3
  -> 0.15). Self-play 256 envs x 1,800 ticks, compiled; duels 512 envs x 1,800 ticks, seats
  alternated (W / D / L by the first agent). 1v1 fog off no fields, 4v4 fields fog. Raw:
  `data/lookahead-r3/runs/`, table `data/lookahead-r3/step3_table.md`.

| run | games | draws | kills / env-min | entropy | speed / corner | store | coast / dive / climb |
|---|---|---|---|---|---|---|---|
| A 1v1 (store 0, round 2) | 367 | 5 | 1.31 | 1.022 | 0.76 | 0.07 | 37 / 57 / 6% |
| B 1v1 (store 0.25) | 446 | 1 | 1.63 | 1.021 | 0.72 | 0.44 | 53 / 25 / 22% |
| C 1v1 (store 0.5, speed 0.15) | 408 | 9 | 1.57 | 1.148 | 0.56 | 0.82 | 37 / 0 / 63% |
| D 1v1 (store 0.25, speed 0.15) | 475 | 6 | 1.82 | 1.121 | 0.61 | 0.66 | 53 / 6 / 41% |
| A 4v4 | 363 | 0 | 9.26 | 0.887 | 0.86 | 0.12 | 31 / 60 / 8% |
| B 4v4 | 350 | 1 | 8.70 | 0.896 | 0.76 | 0.51 | 49 / 23 / 28% |
| C 4v4 | 310 | 1 | 8.11 | 1.026 | 0.63 | 0.78 | 37 / 3 / 61% |
| D 4v4 | 320 | 0 | 8.37 | 0.977 | 0.68 | 0.64 | 45 / 11 / 43% |

| duel | 1v1 W / D / L | 4v4 W / D / L |
|---|---|---|
| B against A | **584 / 5 / 267 (68%)** | **384 / 3 / 285 (57%)** |
| D against A | 423 / 2 / 535 (44%) | 325 / 1 / 368 (47%) |

- Worked: `store_weight` 0.25 alone brings the mean store to 0.44-0.51 (round 1: 0.21-0.52;
  round 2: 0.07-0.12), dive 23-25% of ship-ticks, climb 22-28%, and beats the round-2 power
  choice in both setups. Kills per env-minute 1v1 1.31 -> 1.63, 4v4 9.26 -> 8.70.
- Didn't: weakening the speed term (D, C) makes the ships slow (0.56-0.68 corner) and climb on
  41-63% of ticks; D loses to A in both duels. Not taken (the author's suggestion tried and
  measured: the speed term stays 0.3).
- The round-2 slow 1v1 (replay seed 1, fog off, no fields; `data/lookahead-r3/seed1_check.py`):
  on the step-1 tree (store weight 0) the two drain the store by 10 s (store under 0.05 of full
  on 76% of ticks; 7% of ticks under 0.6 corner speed) and Red wins at 33.3 s; with store weight
  0.25 the stores stay at 0.5-1.0 (never under 0.05), 2.9% of ticks under 0.6 corner, and the
  game ends at 41.5 s with both ships gone (a draw). (Round 2 went to the clock: the merged
  hit model and the step-1 tree already changed this seed.)
- Around 0.25 (02:28-02:55; duels 512 envs x 1,800 ticks; self-play 256 x 1,800):

| duel | 1v1 W / D / L | 4v4 W / D / L |
|---|---|---|
| 0.15 against 0.25 | 382 / 5 / 510 (43%) | 327 / 0 / 351 (48%) |
| 0.4 against 0.25 | 533 / 6 / 388 (58%) | 327 / 2 / 326 (50%) |
| 0.6 against 0.4 | 546 / 5 / 335 (62%) | 302 / 2 / 303 (50%) |

| self-play | kills / env-min | speed / corner | store | coast / dive / climb |
|---|---|---|---|---|
| 0.4, 1v1 | 1.90 | 0.68 | 0.68 | 56 / 7 / 37% |
| 0.4, 4v4 | 8.13 | 0.71 | 0.67 | 52 / 9 / 39% |
| 0.6, 1v1 | 1.67 | 0.64 | 0.79 | 45 / 1 / 54% |

- In 1v1 the duel keeps rewarding a fuller store (0.4 over 0.25, 0.6 over 0.4); in 4v4 0.25,
  0.4 and 0.6 are even, while kills per env-minute fall (8.70, 8.13; 0.6's duel 7.92) and the
  dive goes out of use (23%, 9%, 1-5%). **Chosen: `store_weight` 0.25** (decision-safe: the
  smallest change that fixes the empty store, the store back at round 1's top (0.44-0.51), all
  three powers in use, and it beats round 2 in both setups). The 1v1 preference for more is a
  question for the author (docs/rewrite/questions.md).
