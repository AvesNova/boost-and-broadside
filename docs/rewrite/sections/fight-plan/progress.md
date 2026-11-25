# Fight plan (one- or two-circle, held; hard turn after the pass): progress

- 01:15 worktree ../v2-fightplan (branch scripted-fight-plan from core 17e6cf3b). Read the agent,
  the ghosts notes (the head-on stalemate diagnosis: the plan flips every pass, the two-circle
  cap folds full stick into middle). Plan: memory column `plan` (plan x angles off at its choice),
  held for the engagement (same target), switched only when angles off grew by `switch_loss`
  since the plan was set; a per-match lean drawn in `fresh_memory` (like `side`) added to the
  energy advantage, so the yardstick keeps a mix too; no stick cap while the committed sense is
  held. Next: sibling sanity bench, before numbers on core. ETA before numbers 01:40.
- 01:18 change committed (baca2b9a), quick tier scripted+game 142 passed, ruff clean; pushed.
- 01:25 sanity before/after (sanity.py, 256 games each). Worked: plan switches per engagement
  1v1 1.75 -> 0.17, 4v4 0.22 -> 0.09; one-circle plans chosen 3-13% -> 22-28%. Didn't: 1v1
  without fields stalls more (draws 41 -> 78, head-on passes per kill 2.35 -> 5.19, kills per
  environment-minute 1.66 -> 0.99); 4v4 about unchanged.
- Ablations (scratchpad copies, same bench): hold only (old two-circle cap kept): 1v1 draws 31,
  kpm 1.93; full stick only (plan flipping as before): draws 27, kpm 1.84. Each alone helps;
  together they stall. Trying a stall rule (switch after N passes without gain) as a measured
  option, not as the default.
- 01:30 Diagnosis (scratchpad stalldiag.py, 1v1 no fields, 256 games): in the 78 drawn games
  98% of the ticks with both ships planned had both two-circle (decided games 57%), mean full
  stick share 0.79: two equal ships, same plan, full stick, meet head-on after every turn and
  the losing rule never fires (the pass is neutral). Stall rule prototypes (switch after N
  passes without a 10 deg gain): N=1 draws 27, N=2 30, N=3 29 (1v1 no fields).
- 01:33 Stall rule in the branch (f7be7c68): `stall_passes` 3, `switch_gain` 0.17 rad (back,
  as the stall's gain), memory `plan_passes`. The no-stall after numbers kept as
  sanity-no-stall.json. Seeds 1 and 2, 1v1 (draws before / after): no fields 40/25, 32/29;
  fields 33/28, 28/31; kills per env-minute 1.78/1.64, 1.79/1.50 (no fields).
- 01:33 Compile check, core 17e6cf3b (before, worktree v2-fightplan-before): 4 graphs, 5,930
  operations (act 1,821, advance 2,283, minibatch_loss 1,080, slot_actions 746), 0 breaks,
  cold 222.4 s, within the baseline. After: running, ETA 01:41.
- 01:41 Compile check after (f7be7c68): 4 graphs, 5,952 operations (act 1,836 +15, advance
  2,289 +6, minibatch_loss 1,081 +1, slot_actions 746), 0 breaks, cold 222.7 s (core 222.4 s),
  within the baseline; performance profile, load ~2. game_ops (CPU, one thread, E = 256, 5v5):
  scripted act both teams 461 -> 475, one team 468 -> 484.
- 01:42 Replays: data/replays-2026-11-26-fightplan/scripted1v1_nofields_seed0.mp4 (1,104
  ticks, Red wins at 0:36.8) and scripted4v4_seed0.mp4 (918 ticks, Red wins at 0:30.6).
  replay.py reads `reasons.plan` now. Temporary worktree v2-fightplan-before removed.

## Results (sanity.py, the yardstick both teams, GameConfig(), 256 games each, seed 0)

Before = core 17e6cf3b (sanity-before.json); after = the branch with the stall rule
(sanity-after.json); "hold only" = the branch without the stall rule (baca2b9a,
sanity-no-stall.json). Kills per minute are deaths per environment-minute of play.

| | 1v1 | 1v1 fields | 4v4 | 4v4 fields |
|---|---|---|---|---|
| draws, before / after (hold only) | 41 / 29 (78) | 13 / 20 (37) | 3 / 0 (1) | 4 / 4 (4) |
| mean game length, s | 32.1 / 35.2 (38.0) | 30.5 / 30.4 (31.1) | 27.1 / 28.0 (28.0) | 33.6 / 33.2 (33.3) |
| head-on passes per kill | 2.35 / 3.35 (5.19) | 1.43 / 1.52 (1.74) | 0.67 / 0.69 (0.70) | 0.59 / 0.57 (0.58) |
| kills per minute | 1.66 / 1.42 (0.99) | 1.80 / 1.81 (1.60) | 12.07 / 11.75 (11.74) | 9.63 / 9.81 (9.77) |
| one-circle share of plans chosen | 0.03 / 0.22 | 0.13 / 0.27 | 0.10 / 0.25 | 0.13 / 0.28 |
| plan switches per engagement | 1.75 / 0.66 (0.17) | 1.22 / 0.38 (0.20) | 0.22 / 0.10 (0.09) | 0.22 / 0.10 (0.09) |
| passes (sense commits) per engagement | 3.19 / 4.00 | 2.79 / 2.91 | 1.42 / 1.48 | 1.42 / 1.46 |
| turning ticks in one-circle geometry | 0.12 / 0.25 | 0.20 / 0.30 | 0.33 / 0.38 | 0.34 / 0.39 |

Seeds 1 and 2, 1v1, before / after: draws without fields 40 / 25 and 32 / 29, with fields 33 /
28 and 28 / 31; kills per minute without fields 1.78 / 1.64 and 1.79 / 1.50, with fields 1.73 /
1.49 and 1.73 / 1.59. So in 1v1 the draws fall or stay level, but kills come about 10-15% slower
and take more head-on passes; 4v4 is unchanged within the noise. Skill (panel) not measured.
