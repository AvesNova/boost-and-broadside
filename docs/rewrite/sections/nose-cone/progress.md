# Nose cone and collisions: progress

- 22:47 worktree ../v2-nose (branch nose-cone-collisions from core eba3e474). Before:
  game_ops step 1161 ops no fields, 1499 with fields; scripted act 393/386.
- 22:47 game change committed (be94f237): card nose_inner/outer_angle, core.aspect_factor
  (cosine smootherstep), core.collisions in the step; tests a-f pass. Quick tier: 1 failure,
  test_every_persona_is_competent_at_spread_one (fired share of long-lived ships < 0.97) — the
  nose cone holds fire within 30 deg. Next: scripted miss offset, then re-check.
- 22:59 scripted miss offset committed (30b88fc4); quick tier on lean 332 passed. Compile
  check (after): 4 graphs, 5,918 operations (baseline 5,721: act 1,746->1,819, advance
  2,220->2,273, minibatch_loss 1,009->1,080, slot_actions 746), 0 breaks, cold 252.3 s,
  "within the baseline". The core-before compile check (for a like-for-like before count) was
  stopped by the coordinator (pause: fan must stay quiet). Docs, replay collision marks and
  sanity.py written; NOT run.
- PAUSED. Remaining: rerun compile check on core-before
  (scratchpad/core-before worktree, eba3e474); game_ops.py after (before: step 1,161/1,499,
  act 386); sanity.py on both trees (1v1, 4v4, fields off/on, 256 games); 2 replays into
  data/replays-2026-11-25-nose/; fill numbers into NOTES and core-decisions.
- 00:57 RESUMED. Compile check on core eba3e474 (before): 4 graphs, 5,703 operations (act
  1,704, advance 2,173, minibatch_loss 1,080, slot_actions 746), cold 234.3 s. After (22:55 run,
  before the gun fix): 5,918 (act 1,819 +115, advance 2,273 +100, the others equal), cold 252.3
  s; within the baseline (5,721, budget 260 s). The game step's +100 and the scripted act's +115
  are the collision test and `merge_miss`.
- game_ops.py after (CPU, one thread, E = 256, 5v5): step 1,243 compute ops no fields (1,161
  before, +82), 1,581 with fields (1,499, +82); scripted act both teams 459 (386, +73), one team
  466 (393).
- Found and fixed (6192df9f): the gun's kill case fired nose to nose at a target with 0 shield
  (0 >= 0), but inside the cone the hit does nothing. Now it needs a hit worth more than 0. Before
  the fix 45% of real hits on enemies landed inside 30 deg (smoke run, 1v1).
- Sanity (sanity.py, the yardstick both teams, GameConfig(), 256 games each, seed 0;
  sanity-before.json / sanity-after.json): see the table in lean/game/NOTES.md "Numbers".
- Once-per-pass check: of overlap runs (scored or ending a tick inside the two hit radii) not
  cut by a death, 1v1 0-2 of 144-183 and 4v4 9-10 of 475-535 hold 0 or 2+ contacts. Inspected
  (scratchpad debug4.py, 4v4, 256 games each): most are team-mates in formation (closing under
  100 m/s, overlapping for many ticks: a slowly turning relative velocity makes several closest
  approaches, each a small damage) or spawned overlapping (never scored). Fast (over 100 m/s)
  passes scored twice: 1 (no fields) and 4 (fields) per 256 games; fast passes never scored: 1
  and 2. The cause: both ships turning, so the relative velocity rotates between ticks and the
  closest approach of the straight moves lands in two ticks or in neither. Not changed (the rule
  is the author's); a state-free alternative is to score on the tick of first touch (the start
  gap at least R, the entry inside the tick), which is once per overlap by construction.
- Replays: data/replays-2026-11-25-nose/scripted1v1_nofields_seed0.mp4 (one collision, tick
  601, 46 / 49 damage) and scripted4v4_seed0.mp4 (three: tick 80, 0.03 each, team-mates; tick
  205, 40 / 45; tick 589, 54 / 54).
