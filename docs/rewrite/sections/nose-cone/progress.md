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
