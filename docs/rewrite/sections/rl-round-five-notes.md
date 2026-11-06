# RL round five: notes

Brief: `rl-round-five-brief.md`. Branch `feat/rl-round-five`, CPU only.

## 1. League turnover: the clean cut

- `LeagueConfig.cut_every` (2), `cut_slots` (1), `carry_at_cut` (`"reset"` or `"keep"`),
  `ladder_in_pool` (false). The drain, the incoming stacked set, `OpponentCarry.incoming`
  and the draining slot are gone (`bnb.runtime.opponents`, `bnb.league.league`).
- A cut runs in `League.updated`, after the update's games are read and the weights are
  reweighed, so it draws from this update's weights. A slot whose draw equals what it holds
  is left alone (its games stay credited), as round 2's simulation did.
- Uncredited games: the cut clears the slot's rows' `whole` flags, the same flag that leaves
  out each row's first game; the tally skips each row until its next reset. The learner's
  side of those games trains as usual.
- `"keep"` keeps the old occupant's carry only between two networks (`same_build`). In the
  training league every network is a snapshot or an average of the live policy in the one
  stacked template, so the build check reduces to "both are networks"; a ladder row's
  carry is the scripted agent's, which runs on every row throughout.
- The ladder out of the pool also removes the seat's scripted pass over every opponent row,
  and one network pass per tick (no incoming set): slots passes instead of slots + 1.
- **Open:** without rating lanes and without the ladder in the pool, nothing connects the
  league's entries to the scripted agent, so the fit rates nothing and the gauge keeps its
  starting ratings (every p = 0.5, so sampling is uniform by the floor and p(1 - p)). Real
  runs have lanes; the toy tests do not.
- The gauge's guard: `tests/sections/test_league_ratings.py` (the run's 28-of-29 case under
  the one fit: finite, within 300 of the maximum) and the alert `gauge_runaway` on
  `pool/gauge_anomalies/<mode>` (an entry that has played, more than 1,500 from live or with
  an infinite error).
- Series: `pool/cut/{draws,cuts,cut_rows,uncredited_games,carries_reset}`,
  `pool/slot_share/<kind>`, `pool/distinct_held`; `pool/draining` is gone (dashboard panel
  updated).
- The resume test (`tests/runs/resume_worker.py`) now cuts two slots at every update, so a
  cut falls right before the checkpoint and more follow the resume.

## 2. The memory hand-over test

`bench/memory_handover.py` (CPU, no training): 16 fresh games of calm-fern's 73M live policy
(A) against itself, the opponent seat switched at 60 s to B, measured for 32 s after. Each
B in three versions on the same observations: (a) its own carry from the game's start (the
truth; the near B's drives the seat), (b) A's whole carry, (c) A's carry with the trunk
zeroed and the belief kept (`carry_at_cut="reset"`). KL(a || x) per own living ship over the
42 joint actions, mean per window; "gap" is KL(a || A), no hand-over at all. Near: B =
88M (`step_88080384-live`); far: B = 7.3M (`step_7340032-live`). Every model draws the same
position offset at a reset; a first version that let them draw their own measured a floor
that never decayed (KL 0.11 near), the policy's sensitivity to its frame, not memory.

The policies were trained at commit 724153e7, before the game's later changes; the test
loads them as they are (their build fills the newer Yemong fields with defaults).

| Seconds after | Frontline near: gap / (b) / (c) | Frontline far | Deathmatch near | Deathmatch far |
|---|---|---|---|---|
| 0-0.25 | 0.150 / 0.043 / 0.579 | 1.14 / 0.138 / 0.451 | 0.075 / 0.029 / 0.622 | 1.63 / 0.173 / 0.490 |
| 0.25-0.5 | 0.146 / 0.016 / 0.135 | 1.13 / 0.057 / 0.149 | 0.080 / 0.006 / 0.102 | 1.41 / 0.058 / 0.132 |
| 0.5-1 | 0.135 / 0.007 / 0.075 | 1.17 / 0.021 / 0.059 | 0.071 / 0.003 / 0.032 | 1.63 / 0.027 / 0.054 |
| 1-2 | 0.136 / 0.004 / 0.062 | 1.24 / 0.006 / 0.016 | 0.075 / 0.002 / 0.015 | 1.65 / 0.010 / 0.019 |
| 2-4 | 0.150 / 0.002 / 0.044 | 1.24 / 0.002 / 0.004 | 0.076 / 0.001 / 0.004 | 1.60 / 0.002 / 0.004 |
| 4-8 | 0.135 / 0.001 / 0.005 | 1.18 / 0.001 / 0.001 | 0.069 / 0.001 / 0.002 | 1.50 / 0.001 / 0.001 |
| 8-16 | 0.141 / 0.001 / 0.006 | 1.20 / 0.000 / 0.001 | 0.078 / 0.001 / 0.002 | 1.45 / 0.001 / 0.001 |
| 16-32 | 0.129 / 0.002 / 0.003 | 1.22 / 0.001 / 0.001 | 0.073 / 0.001 / 0.001 | 1.50 / 0.001 / 0.001 |

- **(b) beats (c)** at every horizon until both reach the floor (about 0.001, the bf16
  matmuls and the belief's own drift), in both modes and both pairs, by 4-14x in the first
  half second and 2-20x over 0.5-2 s.
- **How fast:** (b) is within 0.01 of the truth by 0.5-1 s for the near pair and by 1-2 s
  for the far one. (c) starts at KL 0.45-0.62, several times the near pair's whole gap (a
  reset opponent briefly plays less like B than A does), is under 0.02 by 1-2 s except
  frontline near (0.06, and 0.04 at 2-4 s), and is at the floor from 4-8 s.
- **What it means for the cut:** a cut opponent's play is off for about the first second
  under `"reset"` and a fraction of that under `"keep"`; against games of about 150-300 s,
  either is a small share of the cut game's ticks. Even the far pair (66M steps apart) hands
  over well, so "keep" is safe across the pool's age range. `"reset"` stays the default, as
  the brief says; the measurement favours `"keep"`.
- Raw: `/tmp/bnb-scratch-rl-five/handover-{frontline,deathmatch}.json` at the time of
  writing; rerun with

```bash
P=../runs-v2/players-calm-fern
uv run --no-sync python -m bench.memory_handover $P/step_73400320-live \
    $P/step_88080384-live $P/step_7340032-live --mode frontline --threads 8
```

## 3. DPPO and clip-higher

- `LearnerConfig.trust_region` (`"clip"`, `"clip_higher"`, `"dppo_tv"`, `"dppo_binary_tv"`),
  `clip_high` (0.28), `dppo_delta` (0.15). The brief's `clip_low` is the existing `clip`
  (0.2), not a new field.
- **The brief's "product of the three heads" does not apply to Yemong:** its policy is one
  42-way categorical per ship (`policy_logits (..., 42)`), so the exact TV is over that
  softmax directly. The divergence is per ship (a sample is a ship's decision), not over the
  team's product.
- Storage: under `"dppo_tv"` only, the rollout records each own ship's joint distribution as
  float16 (`Outputs.probabilities` -> `Trajectory.probabilities`): 420 bytes per learner
  view per tick at 5v5, about a tenth of the segment store's per-tick records (about 4.2 kB
  at 5v5, measured on the tiny model's store with the production record layout). At the
  default 2048 environments x 128 ticks x 4 shards that is about 0.6-0.7 GB more pinned host
  memory per update (the learner's views, about 1.5 per environment). Binary TV needs
  nothing new (it reads the stored log-probability).
- `"dppo_tv"` requires `old_log_probabilities = "rollout"` (its mu is the rollout's).
- Statistics (every mode): `policy/clip_fraction` is the trust region's rate (clipped or
  masked), with `policy/clip_fraction/entropy_{low,middle,high}` by the joint entropy in
  thirds of ln 42; `policy/divergence/{mean,above_delta,p50,p90,p99}` (exact TV under
  `dppo_tv`, else the binary form), from a 32-bin log histogram over [1e-4, 1].
- Not done (not in the brief, flagged by the notes): a cap on r for unmasked rare actions.

## 4. The representation tap

`bnb.learn.gradients.block_taps` hooks the block's last sublayer (`last_sublayer`): the
temporal one in `spatial_first`, the last spatial one in `temporal_first`. The dormant share
reads that sublayer's feedforward hidden units.
