# feat/lethality progress (game v12, gpubox4)

## Playable now
Rules items 1-4 are in (rough). Fly them with:

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

## Next
- Item 6, league: no-flee scripted agents and the fast-chaser ship. ETA ~15:45.
- Item 7 (GPU, after bc-mb512 ends ~14:30): the doctrine search, then Yongma v Jujak 5v5.
