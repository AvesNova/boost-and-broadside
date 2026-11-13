# feat/lethality progress (game v12, gpubox4)

## Playable now
Game v13: the real-scale set at x2 native speed, with the lethality rules. Fly it with:

    uv run --no-sync bnb play --preset yongma-jujak --fly

## Questions (the author's call; I continue with the default shown)
- Yongma's shield (142) is tougher than the others' (80, 100), so it takes 8-9 core hits
  where Jujak and Vizia take 5-7. Default: keep the shields (they are the archetypes' and
  the game loop's tuning) and give every type the same centre hit times its archetype
  delta. The alternative is pulling Yongma's shield down towards 100.
- Shield timing: back to the old 6 s delay + 12/s (author's call after flying v12). Note
  that empty to full now takes 6 + 6.6-11.8 s = 13-18 s, slower than deathmatch's 8 s
  respawn and about frontline's ~18 s, which is what the brief's constraint ruled out.
  Confirm this is intended.
- Reverse power also restarts the shield wait, as before (only coasting recharges).
- Native x2: the shield is now 3 s + 24/s native (the old 6 s + 12/s equivalent, as
  flown). The capture lock is unchanged, to recalibrate once the ships are final.
- `real_scale.solve_set` (the researched-aircraft builder) still builds an x1 set; a fresh
  build has to go through native_x2.py again. Should the builder scale itself?
- The other 27 real-scale types went through the exact similarity but were not
  re-solved (their bullet life snaps to 10 ticks where a solve gives 11).
- Ratings: `bnb tournament` and `bnb head-to-head` now default to frontline + a new
  standard `elimination` preset (5v5, 10 fields, 2600 m). The in-training league lanes
  still rate on the training games (deathmatch included); I left them alone.

## Log (KST)
### 13:54 start
Branch feat/lethality from integrate/next-runs (fd918a0a). The GPU is running bc-mb512
until ~14:30, so code and CPU tests come first.

### 14:00 milestone 1: damage profile, lethality, armour, shield (items 1-4)
- New card field and handle `core_radius` (all ship files; 0 keeps the classic
  (1-(b/R)^2)^2 falloff, so every other set and the fingerprint are unchanged).
  `combat.closeness`: 1 inside the core, then a smoothstep to 0 at the hit radius (one
  torch.where, fixed shapes). The hit test still uses hit_radius (now the outer radius).
- Yongma, Jujak and Vizia (real-scale), set through their handles and re-armed with
  `real_scale.rearmed` (closed-form combat card, flight card kept):
  hit radius 12 m, core 5 m; centre hit 21 x archetype delta (Vizia 21.0, Yongma 22.98,
  Jujak 19.19); head-on fraction 0, half damage at 45 deg, sharpness 3; shield delay 3 s,
  rate max_shield / 4.5 s.
- Core hits to kill (shooter rows, target columns; the kill is the hit after the break):

  | shooter \ target | Yongma | Jujak | Vizia |
  |---|---|---|---|
  | Yongma | 8 | 5 | 6 |
  | Jujak  | 9 | 6 | 7 |
  | Vizia  | 8 | 5 | 6 |

  At 15 shots/s that is 0.33-0.6 s of core fire.
- Item 4 needed no new logic: `combat.quiet` already restarts the wait on any boost or
  reverse, any shot, any hit, and any stick past the fine level (QUIET_STICKS); only the
  delay and the rate changed.
- VERSION 12 (the fingerprint is unchanged: its presets fly none of the three types).
- Tests (CPU): test_game_section, test_tuning, test_time_scale, test_tuning_solve,
  test_ship_card, test_game_version pass, except
  test_eager_cuda_graph_replays_draw_fresh_randomness, which needs the GPU (busy). I
  recheck it after 14:30. `bnb play --preset yongma-jujak --fly` renders offscreen.

### 14:10 milestone 2: ratings (item 5)
- Elimination at the clock (`modes.phase.clock_result`): the team with more health left
  wins, its living ships' shield fractions (shield / the type's full shield) summed;
  equal is a draw. The ship count no longer comes first. step.py passes the fractions.
- The RL outcome reward reads the same result (`learn.rewards.outcome_events` takes
  `events.result`, which is this rule at the clock), so a run-away draw earns nothing.
- Rating games: `tournament.PRESET_NAMES` and the CLI's `RATED_PRESETS` (tournament,
  head-to-head) = frontline, elimination. The training mix is unchanged.
- Tests (CPU): test_elimination (clock case rewritten for the new rule), test_modes,
  test_mode_rules, test_league_tournament, test_game_api_display, test_map_knobs,
  test_game_version pass; tests/docs has 5 failures that the base branch also has (stale
  references, not this work).

### 14:25 author feedback: shield back to the old timing
The flying and the time to kill feel right. Yongma, Jujak and Vizia's shields went back to
a 6 s delay and 12/s (rearmed); the restart-on-action rule stays.

### 14:45 game v13: x2 native speed (the author's top priority)
- 6459e6d0: every real-scale card and handle plus rules.toml went through the exact
  similarity (length 1, time 1/2; native_x2.py). Yongma, Jujak and Vizia were re-solved
  from the scaled handles (flown error ~1e-7; resolve_native.py). The bullet life follows
  the kept gun range (11 ticks; the similarity alone snaps to 10). The mode clocks keep
  their native seconds. The scripted agent's memory fade is /2 and switch rates x2
  (scripted_x2.py). It now tells a real-scale card by the nearer gravity, since x2 gravity
  is 39.24 > 27. The chaser was solved at x2 (chaser.toml). Fingerprint 00177471050a7b4b.
- Merged the laptop's four play fixes (origin/integrate/next-runs).
- f03ebf3d: the tuning panel's time-scale section is now a "Real-world equivalent" toggle
  (off by default; a read-only table labelled "equivalent"; edits and saves stay native).
- Tests: test_play_interface passes except the frame-rate one, which already failed.
  Flight tests' hard-coded x1 speeds and accelerations were scaled.

Native table (`docs/rewrite/sections/lethality/native_table.py`; time to kill: core hits on
each target type, first hit at t = 0):

| | native | equivalent |
|---|---|---|
| Yongma stall speed | 96 m/s | 48 m/s |
| Yongma corner speed | 288 m/s | 144 m/s |
| Yongma cruise speed | 335 m/s | 167 m/s |
| Yongma top_boost speed | 450 m/s | 225 m/s |
| Yongma settled fine | 294 m/s at 14.1 deg/s | 147 m/s at 7.0 deg/s |
| Yongma settled sustained | 161 m/s at 26.1 deg/s | 80 m/s at 13.0 deg/s |
| Yongma settled max | 141 m/s at 34.0 deg/s | 70 m/s at 17.0 deg/s |
| Yongma core time to kill | yongma 8 hits 0.23 s, jujak 5 hits 0.13 s, vizia 6 hits 0.17 s | yongma 0.47 s, jujak 0.27 s, vizia 0.33 s |
| Jujak stall speed | 83 m/s | 42 m/s |
| Jujak corner speed | 249 m/s | 125 m/s |
| Jujak cruise speed | 316 m/s | 158 m/s |
| Jujak top_boost speed | 434 m/s | 217 m/s |
| Jujak settled fine | 281 m/s at 20.4 deg/s | 141 m/s at 10.2 deg/s |
| Jujak settled sustained | 161 m/s at 39.4 deg/s | 80 m/s at 19.7 deg/s |
| Jujak settled max | 137 m/s at 43.8 deg/s | 68 m/s at 21.9 deg/s |
| Jujak core time to kill | yongma 9 hits 0.27 s, jujak 6 hits 0.17 s, vizia 7 hits 0.20 s | yongma 0.53 s, jujak 0.33 s, vizia 0.40 s |
| vizia stall speed | 89 m/s | 45 m/s |
| vizia corner speed | 268 m/s | 134 m/s |
| vizia cruise speed | 325 m/s | 163 m/s |
| vizia top_boost speed | 442 m/s | 221 m/s |
| vizia settled fine | 287 m/s at 17.0 deg/s | 143 m/s at 8.5 deg/s |
| vizia settled sustained | 160 m/s at 32.0 deg/s | 80 m/s at 16.0 deg/s |
| vizia settled max | 139 m/s at 38.6 deg/s | 69 m/s at 19.3 deg/s |
| vizia core time to kill | yongma 8 hits 0.23 s, jujak 5 hits 0.13 s, vizia 6 hits 0.17 s | yongma 0.47 s, jujak 0.27 s, vizia 0.33 s |

## Next
- Item 6, league: no-flee scripted agents and the fast-chaser ship. ETA ~15:45.
- Item 7: doctrine round 1 running in tmux `lethality-r1` since 14:48 (7 candidates x 1/2/5,
  versus r9m and r9-press). Then Yongma v Jujak 5v5.
