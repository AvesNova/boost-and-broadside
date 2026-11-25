# Nose cone and collisions: progress

- 22:47 worktree ../v2-nose (branch nose-cone-collisions from core eba3e474). Before:
  game_ops step 1161 ops no fields, 1499 with fields; scripted act 393/386.
- 22:47 game change committed (be94f237): card nose_inner/outer_angle, core.aspect_factor
  (cosine smootherstep), core.collisions in the step; tests a-f pass. Quick tier: 1 failure,
  test_every_persona_is_competent_at_spread_one (fired share of long-lived ships < 0.97) — the
  nose cone holds fire within 30 deg. Next: scripted miss offset, then re-check.
