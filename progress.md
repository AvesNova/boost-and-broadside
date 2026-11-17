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
- Ratings: `bnb tournament` and `bnb head-to-head` default to frontline, a new standard
  `elimination` preset (5v5, 10 fields, 2600 m) and kill-lead deathmatch (the author's
  three modes; deathmatch_wide is no longer rated). The in-training league lanes
  still rate on the training games (deathmatch included); I left them alone.

- **Balance under v13 is far off: Jujak beats Yongma.** Yongma's score energy v turn is
  0.11-0.14 at 1v1, 0.31-0.34 at 2v2 and 0.20-0.24 at 5v5 (doctrine round 1, below). A
  card change is the author's call (Yongma's damage or turn, Jujak's shield?). I have not
  started a card round.
- The search harness (bench.game_loop) still plays deathmatch with respawns, scored by
  the kill lead. The rating games are now frontline and elimination; should the doctrine
  search move to elimination?

- Coming from the laptop (feat/elimination-shrink, not here): elimination's boundary
  shrinks over the match, and the real-scale map defaults change (field band 160, radius
  120-1200 lognormal median 400, deathmatch arena 4200 m). Rounds 2-3's elimination balance
  is therefore provisional; re-check the final card on the shrinking elimination. I don't
  touch modes, boundary damage, observations, scripted boundary avoidance or rendering.

- Kill-lead deathmatch at the v15 clock (600 s, 4200 m) is near-deterministic: one side
  wins 90-100% of games from a 55-60% kill share. For ratings and the search, score it by
  kill share, or shorten its clock?

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
each target type, first hit at t = 0). **The settled turns below are at a held store of
0.25 (`shown`'s TURNING_STORE), where the store feeds the turn, so full stick out-turns the
sustained level. Level turns with an empty store (LEVEL_STORE, v11's condition) are
slower at full stick, as intended**, equivalent: Yongma 147 at 7.0 / 80 at 13.0 / 57 at
10.2 deg/s; Jujak 141 at 10.2 / 81 at 19.7 / 59 at 17.2; Vizia 143 at 8.5 / 80 at 16.0 / 57
at 13.4. Checked at 15:05: v11 (fd918a0a, x1) and v13 (x2, halved) agree to 0.1 deg/s at
both store levels, so the x2 conversion left the flight unchanged.


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

### 15:00 doctrine round 1 on the native v13 game (`docs/rewrite/sections/lethality/round1/`)
512 games a cell (SE about ±0.02), the ship files as they are. Score: the candidate's
Yongma side in energy v turn; against r9m and r9-press at 5v5, mixed / reversed.

| Candidate | EvT 1v1 / 2v2 / 5v5 | 5v5 v r9m | 5v5 v r9-press |
|---|---|---|---|
| r9m | 0.14 / 0.32 / 0.20 | 0.52 / 0.51 | 0.68 / 0.62 |
| r9m2 | 0.12 / 0.32 / 0.20 | 0.55 / 0.51 | 0.72 / 0.64 |
| DE Y 0.7 / J 0.25 | 0.12 / 0.34 / 0.24 | 0.53 / 0.53 | 0.72 / 0.63 |
| DE Y 0.5 / J 0.1 | 0.12 / 0.31 / 0.20 | 0.51 / 0.51 | 0.72 / 0.59 |
| r9m2, no crowd extension | 0.14 / 0.15 / 0.06 | 0.32 / 0.35 | 0.50 / 0.43 |
| r9m2, no r9 extend | 0.11 / 0.32 / 0.21 | 0.58 / 0.52 | 0.73 / 0.58 |
| press (no extension) | 0.13 / 0.11 / 0.05 | 0.30 / 0.34 | 0.46 / 0.49 |

- The duel extension's gates matter little now: shrinking them (Y 0.7 / J 0.25, Y 0.5 /
  J 0.1) stays level with r9m2 against r9m. The 1v1 duel is decided by the matchup
  (Jujak), not by when to leave.
- r9's own extension element (`extend`) is not needed: without it r9m2 is as strong or
  stronger (0.58 / 0.52 against r9m).
- **The crowd extension still wins**: without it a team loses to r9m (0.30-0.35) and only
  ties the no-flee r9-press. Leaving the crowd still pays at 2v2 and 5v5 under the new
  lethality. Running away in a team fight is still rewarded. Next levers are the crowd
  gates (energy, shield, threat count), or the game: the elimination score in the harness.
- The mirrors (1v1) are even (0.47-0.53).

### 15:17 round 2: round 1's key comparisons scored on elimination and frontline (`round2/`)
Elimination: 180 s clock, 512 games (SE ~0.02). Frontline: 1200 s, 256 games (SE ~0.03).
Score = the candidate's Yongma side in energy v turn; against r9m and r9-press at 5v5,
mixed / reversed. Provisional (the shrinking elimination and new map defaults are coming).

| Candidate | Elim EvT 1/2/5 | Elim v r9m | Elim v r9-press | Front EvT 2/5 | Front v r9m | Front v r9-press |
|---|---|---|---|---|---|---|
| r9m | 0.23 / 0.38 / 0.38 | 0.48 / 0.48 | 0.52 / 0.55 | 0.02 / 0.27 | 0.51 / 0.50 | 0.55 / 0.52 |
| r9m2 | 0.22 / 0.36 / 0.37 | 0.52 / 0.53 | 0.54 / 0.56 | 0.04 / 0.30 | 0.55 / 0.54 | 0.59 / 0.53 |
| DE Y 0.7 / J 0.25 | 0.18 / 0.37 / 0.40 | 0.54 / 0.50 | 0.56 / 0.55 | 0.02 / 0.32 | 0.49 / 0.51 | 0.52 / 0.54 |
| r9m2, no crowd ext | 0.22 / 0.41 / 0.48 | 0.44 / 0.47 | 0.48 / 0.45 | 0.02 / 0.16 | 0.47 / 0.44 | 0.49 / 0.48 |
| press | 0.20 / 0.39 / 0.43 | 0.43 / 0.47 | 0.47 / 0.45 | 0.01 / 0.14 | 0.46 / 0.48 | 0.42 / 0.46 |

- **Most of round 1's crowd-extension win was the kill-lead scoring.** Against r9m,
  removing the crowd extension cost 0.15-0.18 under kill lead. Under elimination it costs
  0.03-0.06, and under frontline 0.03-0.07. r9m2 beats the no-flee r9-press by only
  0.04-0.09 (0.58-0.72 before). Leaving a crowd still pays a little, about one SE to two.
- The duel gates: shrinking them is level with r9m2 within SE in both modes. r9m2 stays
  the doctrine to beat.
- **Balance is off in both rating modes.** Yongma: elimination 0.18-0.23 at 1v1 and
  0.36-0.48 at 2v2/5v5; frontline 0.01-0.04 at 2v2 (Jujak wins almost every 2v2 front)
  and 0.14-0.32 at 5v5. Round 3 (the card levers, elimination) is running.

### 15:26 round 3: single card levers, elimination, 256 games (SE ~0.03; `round3/`)
Both sides r9m2; the score is Yongma's, energy v turn. Provisional (the shrinking
elimination is coming).

| Card | 1v1 | 2v2 | 5v5 |
|---|---|---|---|
| base | 0.24 | 0.39 | 0.35 |
| Y lift x1.1 / 1.2 / 1.3 | 0.27 / 0.26 / 0.33 | 0.38 / 0.41 / 0.45 | 0.36 / 0.40 / 0.34 |
| Y power x1.15 / 1.3 | 0.27 / 0.30 | 0.47 / 0.44 | 0.45 / 0.56 |
| Y drag x0.8 | 0.24 | 0.42 | 0.43 |
| Y damage x1.25 / 1.5 | 0.29 / 0.38 | 0.40 / 0.36 | 0.45 / 0.46 |
| J shield x0.8 / 0.65 | 0.25 / 0.29 | 0.44 / 0.54 | 0.48 / 0.58 |

No single lever reaches 0.45 at 1v1. The duel is the hardest. Damage helps it most, then
lift. Power and Jujak's shield fix the team sizes but barely move the 1v1.
Round 4 (running in tmux `lethality-r4` since 15:26): the best per lever and one step
further (lift 1.3 / 1.5, power 1.3 / 1.5, damage 1.5 / 2, shield 0.65 / 0.5). It runs
frontline 2/5 first, then kill-lead deathmatch 1/2/5, then elimination 1/2/5.
Likely next: a two-lever card, damage (the duel) plus power or shield (the teams).

### 15:47 round 4: the levers further, on all three rating modes (`round4/`)
256 games a cell (SE ~0.03). Yongma's score, energy v turn, both sides r9m2. Elimination
is provisional.

| Card | Front 2v2 | Front 5v5 | DM 1v1 | DM 2v2 | DM 5v5 | Elim 1v1 | Elim 2v2 | Elim 5v5 |
|---|---|---|---|---|---|---|---|---|
| base | 0.04 | 0.29 | 0.15 | 0.33 | 0.27 | 0.17 | 0.42 | 0.39 |
| Y lift x1.3 | 0.08 | 0.43 | 0.23 | 0.49 | 0.28 | 0.36 | 0.38 | 0.32 |
| Y lift x1.5 | 0.07 | 0.39 | 0.31 | 0.53 | 0.32 | 0.38 | 0.42 | 0.48 |
| Y power x1.3 | 0.07 | 0.47 | 0.22 | 0.59 | 0.78 | 0.32 | 0.49 | 0.53 |
| Y power x1.5 | 0.09 | 0.61 | 0.23 | 0.73 | 0.90 | 0.29 | 0.51 | 0.63 |
| Y damage x1.5 | 0.05 | 0.39 | 0.23 | 0.50 | 0.45 | 0.25 | 0.45 | 0.48 |
| Y damage x2 | 0.07 | 0.45 | 0.31 | 0.56 | 0.59 | 0.39 | 0.47 | 0.51 |
| J shield x0.65 | 0.05 | 0.40 | 0.22 | 0.55 | 0.48 | 0.37 | 0.52 | 0.58 |
| J shield x0.5 | 0.08 | 0.53 | 0.29 | 0.60 | 0.69 | 0.36 | 0.55 | 0.70 |

- **No single lever balances all three modes.** The 1v1 duel tops out at 0.31-0.39
  (lift x1.5, damage x2). Power is very strong in kill-lead deathmatch 5v5 (0.78-0.90)
  but barely moves the duel.
- **Frontline 2v2 is 0.04-0.09 for every card.** A diagnostic on the base card
  (`diag/`, 256 games) shows the harness is sound: mirrors are even (energy 0.56, turn
  0.45, mixed 0.49) while energy v turn is 0.04. Some frontline 2v2 mechanism favours
  Jujak whatever the card: captures by the better sustained turner, or the scripted
  agent's frontline play for Yongma. Worth a look at the scripted frontline goals before
  more card search.
- Closest single cards: damage x2 (DM 0.31 / 0.56 / 0.59, elim 0.39 / 0.47 / 0.51, front
  0.07 / 0.45) and Jujak shield x0.5 (DM 0.29 / 0.60 / 0.69, elim 0.36 / 0.55 / 0.70, front
  0.08 / 0.53).

### 16:00 frontline 2v2 diagnosis (closed: frontline is played at 5v5 and up only)
CPU run (`diag/zones.log`, `zones.py`: 16 games of 600 s, 2v2, base card, r9m2 on both
sides). The turn-circle hypothesis (a 626 m zone against Yongma's 707 m sustained circle and
Jujak's 468 m) is mostly not it. Yongma spends 0.44 of its living time inside a capturable
zone, Jujak 0.52 (within 1.5 radii: 0.59 and 0.68). The gap is the fight: Jujak killed
Yongma 251 times to Yongma's 125 on Jujak, and frontline's all-or-nothing result magnifies
that 2:1. The author dropped frontline below 5v5 (16:00), so it is judged at 5v5 with a
10v10 spot-check.

### 16:00 merged game v15 (origin/integrate/next-runs c35ef260: the shrinking elimination, band sight, new map defaults)
Clean merge; the x2 rules survived; the fingerprint is v15's. bench/state_deltas_bounds.py
now reads boundary_max_damage and boundary_ramp_width. Round 5's frontline (v13, `round5/`)
finished before the merge; its deathmatch and elimination were stopped and moved to round 6.
From here, elimination results count. The harness uses each preset's clock (deathmatch
600 s, elimination 180 s, frontline 600 s); the frontline runs before this used 1200 s.

### 16:23 round 6 on game v15 (`round6/`): finalists and two-lever cards, all three modes
256 games a cell (SE ~0.03). Yongma's score, energy v turn, both sides r9m2. Elimination
counts now. Deathmatch is a 600 s clock on the 4200 m map: its kill-lead result
saturates (base 1v1 0.002, most buffs 0.91-1.0 at 5v5, from 15-120 kills a game), so the
table shows its kill share (Yongma's share of the kills), which reads 0.5 at balance.

| Card | Elim 1v1 | Elim 2v2 | Elim 5v5 | DM share 1v1 | DM share 2v2 | DM share 5v5 | Front 5v5 |
|---|---|---|---|---|---|---|---|
| base | 0.21 | 0.38 | 0.40 | 0.22 | 0.46 | 0.49 | 0.29 |
| Y damage x2 | 0.36 | 0.41 | 0.48 | 0.27 | 0.53 | 0.55 | 0.42 |
| Y power x1.3 | 0.24 | 0.48 | 0.54 | 0.20 | 0.53 | 0.56 | 0.50 |
| J shield x0.5 | 0.38 | 0.63 | 0.65 | 0.27 | 0.54 | 0.56 | 0.52 |
| Y damage x2 + power x1.3 | 0.47 | 0.50 | 0.61 | 0.27 | 0.60 | 0.61 | 0.63 |
| **Y damage x2 + J shield x0.65** | 0.49 | 0.55 | 0.64 | 0.30 | 0.57 | 0.59 | 0.53 |
| Y damage x1.5 + J shield x0.65 | 0.41 | 0.51 | 0.63 | 0.29 | 0.54 | 0.57 | 0.51 |
| Y damage x1.5 + power x1.15 | 0.35 | 0.48 | 0.45 | 0.24 | 0.54 | 0.56 | 0.54 |

- Closest overall: Yongma damage x2 + Jujak shield x0.65. Elimination is within 0.45-0.55 at
  1v1 and 2v2 but 0.64 at 5v5, frontline 5v5 is 0.53, and the deathmatch 1v1 share only
  0.30.
- Every card has the same shape: the duel (1v1) too low and 5v5 overshooting. Damage,
  power and shield all help the teams more than the duel. In round 4 lift x1.5 lifted
  the 1v1 (elim 0.38) without the 5v5 (0.48), so round 7 pairs lift with smaller
  damage / shield steps.
- Deathmatch's kill-lead score is too deterministic at 600 s to rate with. Rate on the kill
  share, or a shorter clock? (Question for the author.)

### 16:40 round 7 (`round7/`): lift with smaller damage and shield steps, v15, 256 games
Yongma's score (deathmatch: its kill share), energy v turn, both sides r9m2.

| Card | Elim 1v1 | Elim 2v2 | Elim 5v5 | DM share 1v1 | DM share 2v2 | DM share 5v5 | Front 5v5 |
|---|---|---|---|---|---|---|---|
| base | 0.19 | 0.36 | 0.47 | 0.22 | 0.46 | 0.49 | 0.32 |
| Y damage x2 + J shield x0.65 | 0.41 | 0.56 | 0.65 | 0.30 | 0.57 | 0.59 | 0.53 |
| Y lift x1.5 | 0.40 | 0.45 | 0.47 | 0.26 | 0.52 | 0.51 | 0.37 |
| **Y lift x1.5 + damage x1.5** | **0.48** | **0.52** | **0.55** | 0.27 | 0.57 | 0.54 | 0.47 |
| Y lift x1.3 + damage x1.5 | 0.43 | 0.43 | 0.50 | 0.27 | 0.55 | 0.54 | 0.48 |
| Y lift x1.3 + damage x1.5 + J shield x0.8 | 0.42 | 0.47 | 0.56 | 0.28 | 0.57 | 0.56 | 0.48 |
| Y lift x1.5 + J shield x0.8 | 0.43 | 0.49 | 0.61 | 0.28 | 0.55 | 0.53 | 0.50 |

- **Yongma lift x1.5 + damage x1.5 is the first card inside 0.45-0.55 on all three
  elimination sizes.** Frontline 5v5 is 0.47 and the deathmatch share at 2v2 / 5v5 is
  0.57 / 0.54. Lift balances the duel without the 5v5 overshoot that shield and power give.
- **The deathmatch 1v1 share stays at 0.26-0.30 for every card** (base 0.22). The respawn
  duel on the small map (1878 m) favours the turn fighter whatever the card, like frontline
  2v2 did. Is deathmatch 1v1 a rating size? (Question for the author.)
- Round 6's leader read 0.49 at elimination 1v1 there and 0.41 here: 256 games is coarse.
  Round 8 confirms at 512.

### 16:50 round 7 spot-checks (`round7-spot/`, run on the laptop): 10v10 and 5v5 at 0.5x / 2x arenas
256 games; Yongma's score (deathmatch: its kill share; the .md files hold the saturated
kill-lead score).

| Card | Elim 10v10 | Elim 0.5x | Elim 2x | DM share 10v10 | DM share 0.5x | DM share 2x | Front 10v10 | Front 0.5x | Front 2x |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.45 | 0.52 | 0.32 | 0.48 | 0.52 | 0.44 | 0.12 | 0.46 | 0.18 |
| Y damage x2 + J shield x0.65 | 0.77 | 0.65 | 0.61 | 0.57 | 0.62 | 0.55 | 0.40 | 0.83 | 0.42 |
| Y damage x1.5 + J shield x0.65 | 0.62 | 0.64 | 0.58 | 0.55 | 0.60 | 0.53 | 0.36 | 0.81 | 0.36 |

- Round 6's damage + shield cards overshoot elimination at 10v10 (0.62-0.77) and frontline
  on the half arena (0.81-0.83), but stay low on frontline at 10v10 and on the double arena
  (0.36-0.42). Map size swings frontline the most: base 0.46 on the half arena, 0.18 on the
  double. Round 8 checks the lift card the same way.

**Why deathmatch 1v1 resists every card (the laptop's point 1).** At 5v5, map size favours
Yongma when small, as the laptop says. But a 1v1 is a lone duel, not a crowd. Both 1v1 modes
are mostly sustained turning fights (two-circle merges 0.72 in elimination, 0.83 in
deathmatch), and Jujak takes 78-81% of the kills at base, mostly from the flank. In
elimination 1v1, Yongma's kill share follows its score (lift x1.5 + damage x1.5: score 0.48,
kill share 0.46), so the card's gain there is kills. In deathmatch 1v1 the same card gets a
0.27 share. The two differ in arena (3488 m elimination, 1878 m deathmatch: 2.7 of
Yongma's 707 m sustained circles) and in respawns. My guess is the small arena: in a lone
duel it leaves Yongma no room to extend and come back. In a crowd it gives Yongma
head-on passes. The disentangling run is queued after round 8 (`round8b/`): deathmatch 1v1
on elimination's 1v1 arena and elimination 1v1 on deathmatch's.

**Elimination on the half arena is decided at the first crossing (the laptop's point 2).**
`modes.spawns.side_placement` puts each team in a disc of 0.15 arena radii (585 m on the
half arena), centres 0.8 radii off the middle (6240 m apart), noses at the enemy's centre
within 15 degrees. Both blocks fly straight at each other; at cruise they meet in about 9
s (first kill median 8.7 s, against 32 s on the double arena). The first crossing decides
it: 0.96 of kills come in the first pass of a pair, 0.49 in the first merge, and 0.49 of
deaths are trades. The armour removes the dead-ahead joust (head-on aspect kills 0.01 of
damage), but in a 5v5 block crossing, ships are hit obliquely (0.33) and from the flank
(0.53) by enemies other than the one ahead, within 0.13-0.27 s. So it is the packed,
facing spawn plus the lethality, not head-on damage. Possible rules (the laptop's area,
not touched here): spawn areas spread wider, or a spawn offset that grows on small maps,
or a heading jitter that spreads the blocks.

### 17:05 round 8 on game v15 (`round8/`): the lift leader confirmed at 512 games
Yongma's score (deathmatch: kill share). Sizes 1 / 2 / 5 at 512 games (SE ~0.02); the spot
columns at 256 (SE ~0.03). Stopped after the deathmatch 10v10 block: v16 changes the
deathmatch arenas (round 9 reruns them).

| Card | Elim 1/2/5 | DM share 1/2/5 | Front 5v5 | Elim 10v10 | Elim 0.5x | Elim 2x | DM share 10v10 |
|---|---|---|---|---|---|---|---|
| base | 0.21 / 0.35 / 0.40 | 0.21 / 0.46 / 0.49 | 0.31 | 0.40 | 0.49 | 0.29 | 0.48 |
| **Y lift x1.5 + damage x1.5** | **0.49 / 0.49 / 0.54** | 0.31 / 0.56 / 0.54 | 0.47 | 0.46 | 0.59 | 0.44 | 0.52 |
| Y damage x2 + J shield x0.65 | 0.41 / 0.51 / 0.63 | 0.30 / 0.57 / 0.59 | 0.53 | 0.64 | 0.64 | 0.62 | 0.57 |

The lift leader holds at 512 games: elimination 0.49 / 0.49 / 0.54 and 0.46 at 10v10;
frontline 5v5 0.47; deathmatch share 0.56 / 0.54 / 0.52 at 2v2 / 5v5 / 10v10. Map size
still matters (elimination 0.59 on the half arena, 0.44 on the double). The deathmatch 1v1
share (0.31, on the 1878 m arena) is the one miss. v16's 3000 m floor tests the arena
explanation directly (round 9).

### 17:10 merged game v16 (origin/integrate/next-runs 7545c16b, with the test sweep)
Clean merge; the fingerprint is v16's (71982b10a86b586f). The search harness (game_loop),
density_sweep and gun_lethality take arena radii from `api.arena_radius_for` (the density
rule, the scale, then the mode's floor: deathmatch 1v1 / 2v2 now 3000 m). Tests pass:
arena_floor, game_version, elimination, boundary_shrink, ship_type_tournament,
league_tournament, game_loop. Round 9 (v16; tmux `lethality-r9` since 17:10): base and the
two leaders at 512 games on elimination and deathmatch 1/2/5, then the three modes'
spot-checks (10v10, 0.5x, 2x; frontline for the lift leader is new there).

### 17:53 round 9 on game v16 (`round9/`): the leaders on the floored deathmatch and the closing elimination
Yongma's score (deathmatch: kill share). Sizes 1 / 2 / 5 at 512 games (SE ~0.02); the spot
columns at 256 (SE ~0.03-0.04). Deathmatch 1v1 / 2v2 now play at the 3000 m floor.

| Card | Elim 1/2/5 | Elim 10v10 | Elim 0.5x / 2x | DM share 1/2/5 | DM 10v10 | DM 0.5x / 2x | Front 10v10 | Front 0.5x / 2x |
|---|---|---|---|---|---|---|---|---|
| base | 0.24 / 0.38 / 0.44 | 0.38 | 0.54 / 0.39 | 0.26 / 0.47 / 0.50 | 0.48 | 0.50 / 0.45 | 0.16 | 0.41 / 0.23 |
| **Y lift x1.5 + damage x1.5** | **0.49 / 0.46 / 0.52** | 0.60 | 0.57 / 0.50 | **0.41** / 0.56 / 0.55 | 0.53 | 0.56 / 0.50 | 0.27 | 0.87 / 0.31 |
| Y damage x2 + J shield x0.65 | 0.46 / 0.52 / 0.62 | 0.73 | 0.59 / 0.58 | 0.40 / 0.57 / 0.59 | 0.57 | 0.60 / 0.56 | 0.41 | 0.88 / 0.42 |

- **The deathmatch 1v1 gap was mostly the small arena.** At v16's 3000 m floor the leader's
  1v1 kill share rose from 0.31 (1878 m, round 8) to 0.41, and the base card's from 0.21 to
  0.26. The rest of the gap is the duel itself.
- **The lift leader stays close in elimination and deathmatch at every size and arena**:
  elimination 0.46-0.57 except 10v10 (0.60); deathmatch share 0.50-0.56 from 2v2 up.
- **Frontline is the outlier and swings with map size**: the leader is 0.47 at 5v5
  (round 8), 0.27 at 10v10, 0.87 on the half arena and 0.31 on the double. The base card is
  just as steep (0.41 on the half, 0.16-0.23 large). It is a property of frontline's maps,
  not the card. A smaller map packs the five zones closer for the faster ship. A card cannot
  fix both ends; a frontline rule might (the author's planned minimum zone size and zone
  travel distance), or the lock and wave doctrine (round 10, now running).

### 18:11 round 10 (`round10/`): frontline lock and wave doctrines against r9m2
Frontline (600 s), the lift x1.5 + damage x1.5 card, each candidate against r9m2 in the mixed and
mixed-reversed lineups: 5v5 at 512 games a lineup, 10v10 at 256. Elo against r9m2 =
400 log10(s / (1 - s)) of the two lineups' mean score, SE by the delta method.

| Candidate | 5v5 score (mixed / reversed) | 5v5 Elo | 10v10 score (mixed / reversed) | 10v10 Elo |
|---|---|---|---|---|
| r9m2 (control) | 0.48 / 0.47 | -19 ± 11 | 0.49 / 0.52 | +5 ± 11 |
| r9m2 + lock 0.5 | 0.73 / 0.77 | +189 ± 12 | 0.67 / 0.70 | +137 ± 12 |
| **r9m2 + lock 1** | **0.75 / 0.81** | **+222 ± 13** | **0.78 / 0.77** | **+218 ± 13** |
| r9m2 + wave 30 s | 0.39 / 0.41 | -73 ± 10 | 0.20 / 0.18 | -256 ± 12 |
| r9m2 + wave 10 s | 0.32 / 0.35 | -116 ± 11 | 0.23 / 0.22 | -215 ± 12 |
| r9m2 + lock 1 + wave 30 s | 0.48 / 0.55 | +11 ± 10 | 0.27 / 0.26 | -178 ± 11 |

- **The lock-aware goal wins big: about +220 Elo at both sizes** (lock_press 1; 0.5 gives
  +140-190). Attacking while the enemy is locked out of the zone it lost, and defending
  while the own team is locked out, until a travel time before the lock ends.
- **Holding for the wave loses**, -70 to -260 Elo, worse at 10v10, and it cancels the
  lock's gain when combined. Waiting at the defense for the respawns hands the enemy the
  lock window and the initiative. Not recommended as it stands; a shorter hold or one only
  for a large missing share might differ.
- Next: lock_press into the default doctrine (both types), then a re-check of the card
  balance on frontline with it, since frontline balance moves with doctrine.

### 18:45 round 11 (`round11/`, game v19): balance with r9m3 (r9m2 + lock_press 1)
Self-play energy v turn on the lift x1.5 + damage x1.5 card; Yongma's score (deathmatch:
kill share, every death counted). Frontline 5v5 at 512 games, the others at 256 (10v10,
arenas) or 512 (elimination, deathmatch 5v5). The 0.5x frontline arena is at v19's 4200 m
floor.

| Doctrine | Front 5v5 | Front 10v10 | Front 0.5x (4200 m) | Front 2x | Elim 5v5 | DM share 5v5 |
|---|---|---|---|---|---|---|
| r9m2 | 0.46 ± 0.02 | 0.24 ± 0.03 | 0.70 ± 0.03 | 0.29 ± 0.03 | 0.49 ± 0.02 | 0.543 |
| r9m3 | 0.40 ± 0.02 | 0.32 ± 0.03 | 0.73 ± 0.03 | 0.29 ± 0.03 | 0.51 ± 0.02 | 0.544 |

- **The doctrine change leaves elimination and deathmatch where they were** (0.51 against
  0.49, 0.544 both).
- **Frontline stays unbalanced, and as map-size sensitive as before.** Yongma 0.40 at 5v5
  and 0.32 at 10v10 (Jujak favoured), 0.73 on the smallest frontline map and 0.29 on the
  double. r9m3 moves 5v5 down 0.06 and 10v10 up 0.08 (each about 2-3 SE). Lock play does not
  remove the map-size effect: on big frontline maps the better sustained turner (Jujak) wins
  the zone fights, on small ones the faster Yongma does.
- So frontline needs a frontline lever, not a card one. Options, all the author's: a zone
  size or count that grows slower than the map; a frontline arena that stays nearer 5v5's
  size at 10v10; or a Yongma-only frontline aid. A card shift toward Yongma (say lift x1.6)
  would help frontline 5v5 / 10v10 but push elimination and deathmatch past 0.55.

### 19:35 frontline geometry exploration (`geometry/`, `geometry.py`; r9m3 both sides, lift x1.5 + damage x1.5, 256 games)
Yongma's frontline score. A today's; B zones and ring x (3 / map_scale)^0.5 (5v5 unchanged,
so B-5v5 = A-5v5); C the arena x (n / 5)^0.25 with fleet size (only 10v10 changes); D both.

| Variant | 5v5 | 10v10 | 0.5x | 2x |
|---|---|---|---|---|
| A | 0.49 ± 0.03 | 0.41 ± 0.04 | 0.79 ± 0.03 | 0.28 ± 0.03 |
| B | 0.49 (=A) | 0.39 ± 0.04 | 0.78 ± 0.03 | 0.32 ± 0.03 |
| C | 0.49 ± 0.03 (=A) | 0.29 ± 0.04 | 0.79 ± 0.03 (=A) | 0.28 ± 0.03 (=A) |
| D | 0.49 (=A) | 0.51 ± 0.04 | 0.78 ± 0.03 (=B) | 0.32 ± 0.03 (=B) |

- No variant flattens the map-size effect at 5v5 on a smaller or bigger map: 0.5x stays
  0.78-0.79 and 2x 0.28-0.32 under every geometry. B alone barely moves anything.
- At 10v10, D (both) gives 0.51, against A's 0.41; C alone drops it to 0.29 (a denser map
  with today's zones favours Jujak). So the 10v10 imbalance is the zones and arena
  together, and D fixes 10v10 without moving 5v5. The 0.5x / 2x imbalance is the arena
  size itself (the speed against the turn), and these geometry levers do not touch it.
- B at 0.5x reaches about 46 m past the boundary with its outer zones (ring 0.838 at the
  4200 m floor).

### 19:40 WIP: Baekho (r9m3 as the one scripted doctrine): see the WIP commit's message.

## Next
- Item 6, league: no-flee scripted agents and the fast-chaser ship. ETA ~15:45.
- Item 7: round 6 (v15) is running in tmux `lethality-r6`; then 10v10 and 0.5x / 2x arena
  spot-checks for the finalists. Item 6's chaser needs a third kind in bench.game_loop to play.

## 2026-11-16 21:27 integration for tonight's launch (feat/imugi-boundary)
- Tried: merged lethality-sticky (game v23, sticky Imugi build 2) and balance-eval; both clean, quick tier 95 passed.
- Worked: game v24, no fog in team deathmatch: GameConfig.fogged_share by mode, resolved per pool in Lineup (every Rollout path); balance_sweep fog follows the mode; ae3ea958.
- Next: the whole fast suite (author asked), the CPU smoke check of `bnb train --preset real_scale_all_modes` pools.
- 22:12 tried: the whole fast suite (`not slow and not gpu`): 2613 passed, 54 failed. 39 were Yemong
  model tests whose team deathmatch fixtures needed hidden ships: fixed with an explicit
  `fog_share=1.0` (160 passed). Smoke: real_scale_all_modes PPO and BC pools give deathmatch 0/981
  fogged, elimination 981/981, Frontline 1110/1110. Next: rerun the rest for the other 15 failures.
- 22:47 the rest of the fast suite: 2500 passed, 7 failed; fixed 5 (boundary 216 expectation, ghost test
  fogged, arena window 30 s since the clear arena's first kill is later, a docstring, formatting).
  Two fail at the base a3e7b09a too and are left: test_layering (set_num_threads in league/design.py,
  models/initialise.py) and test_stacked[False] (1/900 elements over 1e-4). Next: the launch.

## 2026-11-16 21:45 diag/boundary-deaths (agent)
- Tried: bench/boundary_deaths.py (eager, wraps api.resolve + Imugi internals) on Y0.46, 4 conditions x 256 games; per-tick CPU traces of 1v1.
- Found: boundary deaths happen with the edge steer at full weight (w=1) for seconds; the stick flips full-right/full-left in 4-5 tick runs, each flip on a doctrine "sense" tick (post-merge turn sense following the target's turn, applied after the edge steer). Heading stays outward, boosting, ~2 s to death at 216/s.
- Next: fix = scale the maneuvers' geometry gate by (1 - boundary_weight) (BOUNDARY_OVER_MANEUVERS); before/after on the same 4 conditions.

## 2026-11-16 21:55 diag/boundary-deaths (agent)
- Tried: baseline Y0.46, 256 games each (DM 300 s): deaths by cause per ship.
- Found: DM 1v1 99% of deaths are boundary (Y 2.0/game, J 0.6/game; hits 0.01); 2v2 Y 60% boundary, J 36%; 5v5 Y 22%, J 8%; elim 5v5 Y 18%, J 5%. Every crossing has the edge steer at w=1 and a target, ~370 m/s, turn radius ~400 m (margin is ample).
- Next: after-fix run (BOUNDARY_OVER_MANEUVERS on), ETA 22:10 KST (worst 22:15).

## 2026-11-16 23:30 feat/imugi-initiative (agent)
- Step 0, Imugi's action mix (DM 5v5, 128 games, v24 card, before): Yongma stick straight 7.6%, R/L 0.4 8.5/8.5, 0.75 5.6/5.6, 1.0 32.4/31.8; power coast 3.7, boost 88.5, reverse 7.8; fire 5.4%. Jujak straight 6.8, 0.4 8.8/8.7, 0.75 7.8/7.4, 1.0 30.6/30.0; coast 3.7, boost 87.8, reverse 8.5; fire 7.3. Every action is used, but full stick (~63%) and boost (~88%) dominate; coast under 4%. Elim/FL 5v5 running.
- Tried: commitment at the merge + initiative (BNB_TURN_COMMIT) and stick by energy (BNB_STICK_BY_ENERGY), both on; turn onset 0.30 s at corner speed (3.3 nose periods, turn_onset.py); dead-on tie: map centre, then left. Imugi tests 104 pass. Commit bf0c1fa1.
- Next: 1v1 turn_fight before/after (128 games, ETA 23:50), then balance+mix 512 games x 4 conditions before/after (ETA ~01:30), stills on the CPU.

## 2026-11-16 23:45 feat/imugi-initiative (agent)
- Tried: DM 1v1, 128 games, before vs after (both rules on). Scissors windows 2752 -> 33 (one circle 1096, two circle 883); reversals per post-merge window at 15 deg/s 36.8 -> 1.0; sustained turn (>15 deg/s held 1 s) 14% -> 52%; stick side flips 414-459/min -> 95-145/min.
- Found: the duel is now decided: Yongma kill share 0.59 -> 0.07 (deaths 1.66 -> 2.47 a game, Yongma's 205 by hits + 86 boundary): in a held turning fight the turn fighter wins, as the card says. Energy-advantaged Yongma takes the initiative into a fight it loses.
- Next: balance+mix 512 games x4 conditions before/after (running, ETA ~01:30); 1v1 ablations commit-only and ENERGY_BREAKS_RATE_TIES=1; stills on the CPU.

## 2026-11-16 23:50 feat/imugi-initiative (agent): step 0, the action mix (before, 128 games each, v24 card)
| mode | ship | straight | R/L 0.4 | R/L 0.75 | R/L 1.0 | coast | boost | reverse | fire |
|---|---|---|---|---|---|---|---|---|---|
| DM 5v5 | Yongma | 7.6 | 8.5/8.5 | 5.6/5.6 | 32.4/31.8 | 3.7 | 88.5 | 7.8 | 5.4 |
| DM 5v5 | Jujak | 6.8 | 8.8/8.7 | 7.8/7.4 | 30.6/30.0 | 3.7 | 87.8 | 8.5 | 7.3 |
| Elim 5v5 | Yongma | 13.0 | 12.0/12.0 | 5.8/5.7 | 25.6/25.8 | 3.8 | 90.0 | 6.2 | 2.6 |
| Elim 5v5 | Jujak | 12.0 | 12.4/12.4 | 6.9/7.1 | 24.8/24.4 | 3.7 | 89.8 | 6.5 | 3.5 |
| FL 5v5 | Yongma | 13.5 | 11.9/11.9 | 3.6/3.6 | 27.5/28.0 | 56.0 | 30.9 | 13.1 | 3.6 |
| FL 5v5 | Jujak | 12.6 | 12.8/12.8 | 4.6/4.6 | 26.3/26.4 | 49.8 | 36.0 | 14.2 | 4.8 |
- Found: all 7 sticks and 3 powers used everywhere; full stick 51-64%, the sustained level (0.75) the rarest at 7-16%; DM/Elim boost ~88-90% with coast <4% (FL coasts ~50-56%).
- Next: balance+mix 512 games (4 jobs at a time under one GPU lock hold), 1v1 ablations.

## 2026-11-16 23:57 feat/imugi-initiative (agent)
- Tried: 1v1 ablations (128 games): commit only (stick by energy off): Yongma kill share 0.012, 11.5 merges/game, sustained turn 68%/62%; both rules (default): 0.068, 15.7 merges/game; with BNB_ENERGY_BREAKS_RATE_TIES=1: 0.036. Stick by energy adds merges and kills; rate-first does not help Yongma (left off).
- Found: DM 2v2 (512 games) Yongma kill share 0.421 -> 0.466 after (band .481-.521): toward balance. Before DM 5v5 0.489.
- Next: the other 5 balance jobs (ETA 00:25, worst 00:40).

## 2026-11-17 00:16 feat/imugi-initiative (agent)
- Tried: balance spot check, 512 games each, eager, v24 card (Y0.46, head-on 0.05), before vs after (both rules on).
- Found: Yongma kill share DM 2v2 0.421 -> 0.466 (band .481-.521), DM 5v5 0.489 -> 0.514 (.494-.507), Elim 5v5 0.544 -> 0.592 (.498-.545, SE 0.013), FL 5v5 0.481 -> 0.507 (.491-.513). Every team mode moves +0.025..+0.048 toward Yongma (the 1v1 the other way); the 5v5 stick/power mix barely moves (the sense manoeuvre is rarely on in a crowd).
- Next: per-rule ablations on the 4 conditions (commit only, stick only), ETA 00:45 (worst 01:00).

## 2026-11-17 00:52 feat/imugi-initiative (agent)
- Tried: per-rule ablations, 512 games x 4 conditions (commit only / stick only).
- Found: Yongma kill share commit only: DM 2v2 0.471, DM 5v5 0.509, Elim 0.592, FL 0.528; stick only: 0.415, 0.493, 0.539, 0.483 (= before within error). The commitment carries the whole balance shift; stick by energy alone is neutral in team modes and pulls FL back (0.528 -> 0.507 with both).
- Next: final report; results in docs/rewrite/sections/imugi-initiative/results.md.

## 2026-11-17 02:35 integ/v25 (agent, re-trim)
- Tried: 3 shield trims of v24's card with the committed-turn Imugi, 512 games x DM 2v2/5v5, Elim 5v5, FL 5v5 (retrim-v25/results.md).
- Found: each trim lands 2 of 4 in band; Y0.42-J1.22 has the smallest worst miss (DM 2v2 0.449, 0.032 under; Elim 0.556, 0.011 over). Applied as game v25, reference regenerated, quick tier + ruff pass.
- Next: 10v10 at 256 games for Y0.42-J1.22 and Y0.38 (running, laptop); dry run 2 from feat/imugi-boundary.
## 2026-11-16 22:45 trail stills (feat/trail-stills)
- Built: bnb.figures.match (one whole game headless via the play shell's MatchLoop, Imugi or a policy per side, events + ghosts per tick) and bnb.figures.trails (renderer); wired as set "trails".
- Worked: Imugi vs Imugi real-scale elimination 5v5 (205 s game, 3 min CPU) and a 1v1 duel render cleanly; the duel's busiest window is a two-circle fight.
- Next: policy fog ghosts (no v9 checkpoint exists; capturing rl-v20 v8 in a detached worktree at c262a32b), deathmatch 5v5, tests + docs.

## 2026-11-16 23:10 trail stills
- Done: CLI (`bnb figures trails --scripted --match.* --still.*`), docs section, 6 quick tests (16 s), ruff clean; pushed.
- Worked: real belief ghosts from rl-v20-seed1 (v8) captured in a detached worktree at c262a32b with match.py copied in; early-game window is too crowded (5 hidden ships), so the fog window caps at 2 hidden.
- Next: whole-game policy elimination capture (running), deathmatch 5v5 Imugi (running), pick and commit the example PNGs.

## 2026-11-16 23:35 trail stills: examples
- Done: four examples in docs/images/stills-examples/ (deathmatch 5v5 opening, light and dark; 1v1 two-circle duel; elimination 5v5 fog with rl-v20-seed1's belief ghosts).
- Learned: a 10-min 5v5 game in one still is texture only (ticks now off past 90 s); the fog still needs at most 2 ghosted ships and a fight window to read.
- Open: no v9 policy exists, so the fog example comes from a v8 policy captured at c262a32b (game v19) with match.py copied in.

## 2026-11-16 23:55 trail stills: author feedback
- Done: rebased on feat/imugi-boundary (bbb52a66; the final card commit had not landed); stills now draw trails + death X only, shots/hits/ends/fog opt-in, never all three in a fleet still.
- Re-recorded the 1v1 duel (120 s) and the 5v5 deathmatch opening (60 s) on this tree: identical paths to before, as the earlier base already held sticky Imugi build 2.
- Duel shape: tight circles (~400-500 m across), straight under 3 deg/s only 8-12% of the time; no racetrack legs. PNGs in /home/vizia/avesnova/review/stills/.

## 2026-11-16 results charts (branch feat/results-charts, worktree bnb-charts)
- Tried: read ladder, crossover, rating_pool, evaluation, metrics keys of rl-v20/bc-v21; v1 viz on main.
- Found: ladder.break_even_points reads `presets` at the result's top level, but write_result nests it under `payload` (real results would give no points); no per-game length record exists anywhere.
- Next: src/bnb/figures/results/ (fan, break-even, arms, team size, health, game length, values file) + `bnb figures results`.
- 2026-11-16 23:40 Worked: 5247438a, `bnb figures results {fan,break-even,arms,team-size,health,game-length}` + values file; 10 CPU tests, ruff clean; previews checked; health drawn from real bc-v20 + rl-v20 records.
- Didn't: no per-game length record exists (game-length needs one); no driver plays the team-size lineups (n, k n) yet; the W&B path is untested against a live run.
- Next: real data after tonight's run: crossover per snapshot into $OUT/break-even, then fan / break-even; arms.json as decisions land.

## feat/unified-eval (2026-11-17, laptop 4070)
- 03:55 tried: one continuously refilled engine per size bucket (players, seats, team sizes, lineups, map scale as pending data; modes as static slices); planner over Elo / break-even / win-rate / decision targets.
- worked: CPU tests (18 quick); laptop smoke 256 envs, 3 players, games tally and refill with no missed games; null false-positive 0.050 (no draws) / 0.054 (20% draws, 213-game max), 0.045/0.048 at 850.
- next: sizing probe + throughput vs the box queue's suite (6656 games in 32 min), concurrency, 5090 check script, docs.
- 04:15 tried: VRAM sizing by eager probes; worked after a throwaway probe + gc (76 MB + 1.12 MB/env, stable); didn't: first probes unstable (0.8-5.3 MB/env) and a 6336-env batch OOMed; /tmp quota hit by the inductor cache (this session's entries deleted, runs now cache under ueval-runs/cache).
- next: compiled per-env factor from a 1024-env run, then games/hour and GPU use vs the box suite; results.md.
- 04:45 worked: laptop bucket-5 run, 1024 envs: 37-45k games/h, GPU 100%, the same 6973 games took the 5090 queue 1198 engine-s vs 675 s here (laptop 1.8x the 5090 queue). didn't: filler games skewed the Elo allocation (3037 rl_20m-bc games) -> replaced by overtime; a bucket stopped before its in-flight games ended -> fixed.
- next: benched (bucket 5) vs native (bucket 1) 1v1 scores, the v25-vs-v24 1v1 gap (DM 1v1 0.29 here vs 0.61 on the box), results.md.
- 05:08 worked: v24 tree reproduces the box suite's 1v1 scores (0.625 vs 0.612, 0.238 vs 0.219); benched = native within 1.7 SE; v25 Imugi is far stronger at 1v1 (DM 0.29 vs 0.62). Per-machine remembered sizes; planner 3.8 ms/game at 3x41-player fields. didn't: bucket 5 plays 1v1s 2.7x slower than a 1-a-side engine (add small buckets when 1v1-heavy).
- next: concurrent two-bucket test, final report.
- 05:25 worked: two buckets concurrently (5 and 10, 640 + 832 envs, GPU 100%); results.md and the 5090 checklist written. next: the 5090 check (box/unified_check.sh) once the box is free.

## feat/unified-eval: Elo curves to precision (2026-11-17 morning)
- 08:40 tried: dry run 1's curves left late RL snapshots as bounds; the unified EloTarget had the same rule (a bound "settled" after 4/goal^2 games, unrated players planned at the anchor's rating).
- worked: bounds/undetermined never a result; their finished games carry no planning information across fitted components (prior variance 1000 Elo, so they go first); planning ratings from a penalised fit with a neighbour prior (run steps, matched steps across runs, BC last to RL first); prior games from rating pools (`--prior`); loud WARNING when a safety bound stops a target. Offline (bench/elo_curve_planner.py, dry run 1's table as truth, 1024 in flight, from scratch): elimination 5,280 games, deathmatch 13,680, frontline 16,960; every player rated, worst SE <= 60.
- didn't: ~40% of frontline games sit beyond p = 0.95 (the A-optimal long hops of a 5,400-Elo span; information per game peaks near 400 Elo hops); ~4% beyond 0.99 after the first wave.
- next: laptop run of dry run 1's curves on the v24 tree with the old pool's 36,672 games as prior (ueval-runs/curves-v24).
- 09:20 worked: merged feat/v26-imugi-y046 (game v26) for dry run 3; break-even at the trained size (policy_count 5, Imugi 1-20); map-scale targets dropped from unified_check.sh (author); probe cap (8 a pair, 32 a player in flight) for players without a rating: offline 4,940 / 11,776 / 15,408 games (elimination / deathmatch / frontline).
- didn't: first laptop launch OOM-killed at the 8 GB host cap while compiling 4,480 environments; relaunched at 2,048 environments, 2 compile threads. The laptop run (pre-probe planner) plays 6-13k games/h (long 5v5 deathmatch and frontline games); test_play_sides' 2 failures predate the merge (model_draws).
- next: the laptop run to precision (ueval-runs/curves-v24, safety 3 h), then the final table.
- 09:40 diagnosis (laptop, curves-v24, 52 network players, 2048 envs): 613 game-seconds a wall second (18,400 env-steps/s at 30 ticks a game second) against 1,450-1,680 (44,000-50,000 env-steps/s at 1,024-1,344 envs) in the earlier one-network-player checks: 2.2x the network rows a tick (lanes [(26, 232), (26, 114)] = 8,996 padded rows for 4,096; before 2,730 for 2,688) and a further ~1.9x from the 5v5 frontline/deathmatch games themselves; games/h falls again by game length (frontline ~220 s, deathmatch ~370 s of game time against elimination 5v5's ~60 s). Host planning 0-1%, GPU 100%, no recompiles.
- tried: a resident set (12 network players at once, lanes 1.15 x rows / 12 = 393, swapped by the planner's value once drained, weights copied in place): 4,524 padded rows a tick for 4,096. Restarted the laptop run on it (resumes games.jsonl + prior).
- 09:55 worked: resident set (12 lanes x 393 rows, 4,716 padded rows a tick) with sticky next games: 52,600-57,400 env-steps/s on the laptop for the 52-player field, GPU 100%, warm compile 63 s, parked 0 (without the sticky rule 222 parks in 10 min: current + next games both hold lane rows). Before: about 27,000 env-steps/s (905 game-seconds a second from the finished games, 09:15-09:34). Paused at 09:55 for the author's matchup run (GPU handed over); resumes from games.jsonl.
- next: resume, confirm swaps happen and the swapped-in players' results agree with their prior ratings.
- 10:45 resumed at 10:31 on the quiet GPU (swap ratio 1.25): 52,600-53,600 env-steps/s, first 3 swaps at 10:41; parks rise while leaving residents drain (1,100 by 10:42), worth a look. A direct A/B of the 52-seated layout failed: its cold compile was OOM-killed at the 8 GB host cap. Status at 10:43: frontline 2,369 new games, all 53 rated, 28 goals unmet (worst 146); deathmatch 4,306 new, 38 rated, 15 bounds above left, worst 205; elimination 0 new, met from the prior. The run continues (ueval-runs/curves-v24, safety 3 h to 13:31, 60,000 games a mode).
- 11:25 worked: two runs in one field (author): per-agent Imugi rules (a CPU test: a flags-off view plays exactly the process flags off, a flags-on view the flags on, side by side in one pass); v24 games admitted into v26 fields (scripted -> imugi_v24, logged); unified run dirs as prior; box/unified_spec.py for several chains; planner 68 -> 8 ms a choice on a 108-player field. CPU eager smoke of the engine with both Imugis: 40 blocks, no error.
- 12:10 tried: the box's park storm (dry-v26-v24-unified: parked 3150 > 2395 games, 0 swaps), rebuilt on CPU (bench/unified_park_sim.py; 108 players, 16,128 environments; it reproduces 61-71% parked and 0 swaps). Cause: residents are seated by value, the most valuable are unrated players, and the probe caps hold them to 32 games in flight. The v24 chain is unrated at first (its pool links it to imugi_v24, not the anchor). So 10 of 12 lanes were at ~96 of 3,092 rows; 3 residents marked leaving at once, each draining over 2 long games.
- worked: filler tier (Elo pairs with the probe caps lifted) before parking; a spare lane (the incoming player seated at once); leaving residents take short games while a longer one plays; the choice reads only the seated players' cells; filler refreshed every 64 games. Sim, 600 game-s: parked 63% -> 0%, real games 30,693 -> 49,555 (0.0032 -> 0.0051 a game an environment-second), host 1,030 -> 364 s. GPU smoke: spare-lane swaps, no overflow. progress.log now prints the parked share (environments now, ended ticks).
- didn't: swaps are still about one per drain (a leaving resident finishes its current and next long games); its unstarted next games are not re-planned (the one-block lag).
- next: relaunch the box run on this commit (13 lanes: one cold compile); watch parked share 0% and swaps > 0.

## feat/unified-eval: the box's two-run curve diagnosed (2026-11-17 afternoon)
- tried: dry-v26-v24-unified's games, spec and priors copied; the planner rebuilt on CPU from them; the box's last session replayed (engine assignment, 3,000 environments).
- found: 15 players never seated (2 swaps in 15 min: one lane per drain at ~70 s of wall time a game; 62% of that session's games were scripted vs imugi_v24); champion and rl_148897792 the same weights as two players with no game between them (1536 vs 1726); a resume credited 1,583 games of step_98566144 (session 1's champion) to step_148897792 (names only in games.jsonl); the late-RL dip = two islands (115-135M with the champion, 140-149M with 20-35M); early BC a bound (lost every game to the anchor, never met dry run 1's BC); win-rate reports had no `met`.
- worked: identities in games.jsonl (resume and unified priors by weights); same weights = one player with aliases; ROTATING 4 residents draining at once plus FRESH_GAMES 16 (newcomers stay until their probes return); met/stopped in every report. Re-fit of the stamped games: late block 1673-1772, champion = snapshot 1702 ± 13.
- next: resume the box run from ueval-runs/dry-v26-v24-unified/games.stamped.jsonl on this commit; ~6-10k games, 15-20 min, for the 15 unplayed and the early BC.
- 14:50 (agent, static curves) worked: static Elo-curve design (static_design.py, static_target.py; spec design = "static", the curves' default in box/unified_spec.py): prior 100 Elo a step, Imugi at the first RL checkpoint; pairs +-reach on the chain and the anchor near the transition; A-optimal counts (greedy + trim, full-rank check); stage 1, fit, stage-2 top-up at the fitted ratings (to 0.95 x goal), stop, WARNING per player above the goal. The engine's resident window streams along the training order (BucketEngine._stream). CPU sim on dry run 1's elimination: spec (reach 2, pilot 1) 15.6k games, all <= 60; reach 4 pilot 2 ~7k; adaptive 4,940.
- didn't: deathmatch and frontline (large fitted gaps): place-based pairs either explode (377k games) or leave bounds. They need stage-2 pairs by fitted rating.
- next: the box run on one run (dry run 3 elimination, then rel-jujak-s1); then the linking stage for several runs.
- 15:20 (agent, static curves) worked: rebased on ace2295d/1637b67b (aliases, hashed identities); laptop GPU smoke end to end (stage 1, stage 2, window swaps, stop, WARNING); dry run 3 alone with the stamped prior: 15.4k games at the spec's reach 2 / pilot 1, 3.4-4.5k at reach 4 / pilot 1.5-2 (CPU); eval/rel-jujak-unified = feat/unified-eval + feat/release-jujak-elim (the hemisphere preset) for the release run's curves.
- next: the author picks reach/pilot; box runs: dry run 3 elimination alone, then rel-jujak-s1 once its training ends.
- 15:25 (agent, windows -> all loaded) tried: fixed overlapping windows (spec `windows = {size, overlap[, capacity]}`; 16/8 on dry run 3 elimination, reach 4, pilot 2, 1024 environments, laptop). With a barrier: window 1 (606 games) 133 s of play, window 2 (91 games) 348 s, 99% of it the idle tail. Cause: the windows are tiny against 1024 environments (stage 1 is 1,481 games in all) and mid-BC games run 150-220 game seconds, so each window lasts its slowest game; the other ~95% of environments play parked games. Then without a barrier (the next windows seated in spare lanes once a window's last cell started): the windows cascade, but the tick slowed 2.6x.
- found: on the laptop a tick's cost is the network's padded rows (lanes x capacity), not the environments: env-steps/s x padded rows = 122M at 3.2k, 4.8k and 8.3k padded rows (38k, 25.6k, 14.6k env-steps/s at 1024 environments).
- worked: windows kept as an option (pipelined, `WINDOW_AHEAD` = 1, capacity from the spec). The author's call (15:25): the whole field seated, each lane sized by its player's share of the design's games (`BucketEngine._design_shares`, `lane_runs`), no swaps, stage 2 on the same lanes; launched with `residents=None`, MemoryMax=16G.
- 16:15 (agent) worked: dry run 3 elimination curve complete on the laptop (all 62 checkpoints rated, worst SE 58.5, no WARNING; 3,029 new games; results.md). The whole-field engine: compile 676 s, 10.8 GB host peak (16 GB cap). Stage 1 remainder 6.5 min, stage 2 28 min.
- didn't: stage 2 mostly parked: (1) every environment's next game was parked when stage 2 was designed (two parked games of wait, ~3 min) and (2) the top-up's 1,562 games fell on early BC players held to 43-row lanes (lanes sized by stage 1). Both fixed (re-choose parked next games and reseat lanes by the new stage's games, in place), not yet run on a GPU.
- next: the release run's curve on the box (rel-jujak-s1, whole field, RESIDENTS above the field).
