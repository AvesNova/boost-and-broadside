# Two-level league: progress

Author 2026-11-27 ~20:15: "Let's implement and test out the two level league". Branch
`league-pool` off core 2770a376. Laptop RTX 4070. Bound: everything committed by 08:30.

## Plan (20:30)

1. pool.py: the pool (members, tallies), level 1 (slot loads between updates), level 2 (per
   game, uniform over loaded slots with a free row), draining, overflow; checkpoint state.
2. rollout.py: per-row scripted parameters (yardstick + personas) in the one scripted call; the
   merge takes network slots only.
3. train.py: the league opens at RL start (EMA + personas present from tick 0), level 1 after
   every update, resume.
4. Tests (CPU): share tracking (simulated), duplicates, drain, persona rows, no recompile,
   resume. Rewrite the old capacity tests.
5. GPU: tick at 1 and 12 snapshots, scripted share early, a short RL smoke run with resume.
6. Quick tier, slow compiled tests, ruff, compile check; NOTES, core-decisions PROPOSAL.

ETA: code + CPU tests 01:30 (worst 03:00); GPU measurements 03:30 (worst 05:00); finish 06:00
(worst 08:00).

## Best-judgement calls (one line each)

## Log
