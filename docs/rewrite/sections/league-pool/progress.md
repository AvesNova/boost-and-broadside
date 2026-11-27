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

- Level 1 rounding: a carried debt per member (deterministic error diffusion) instead of random systematic rounding: exact long-run share, no RNG to checkpoint, bounded error.
- Swaps only while some member is short by more than one slot against another (v_u - v_o > 1): each swap strictly lowers the squared debt, so no churn loop; throttled by the free slots.
- The slot unloaded is the over-served member's loaded slot with the fewest games (drains first); a slot with no games frees at once.
- A member dropped by retention unloads all its slots (they drain, games RETIRED); no game is ever re-dealt, so `abandon` and the retiring-slot re-deal are gone.
- An overflow slot (a capacity-free yardstick slot, index K + R) catches a game only if no loaded slot has a free row (cannot happen at K C >= 1.25 L unless retention drains 3+ slots at once with no free slot); counted in `overflow_games`, 0 in every simulation.
- Personas sit after the snapshots (member indices 0/1/2/3.. unchanged); placeholders persona14, persona21, persona27, persona09 from round 1 (skill.json values, 4 significant figures, all 21 sampled rows) in `LearnConfig.league_personas`.
- The per-row parameter table holds the sampled rows plus any row a persona names (not all 37): fewer tensors in the scripted call.
- Per-row scripted parameters in both stages (BC reads row 0 only): one scripted graph for BC and RL instead of two in a `--stage all` process (the compile check counts graphs).
- The EMA joins at RL start (`League.open`, before the first deal), the uniform average once it holds an update (about 16 updates in, as before); the BC stage's league stays the yardstick only.
- Slot weight buffers are allocated once per slot and reused (copy_ on load); snapshots' weights kept in host memory (CPU) plus once on disk.
- `dealt` (undecayed games dealt per member) added to the league's device state: the realised shares, printed as `scripted_dealt` on each update line.
- K 12 + R 4 kept (the brief's numbers); the simulation's K/R variants are recorded for the author.

## Log

- 21:05 Code + CPU tests done (commits ff302665..4fdc9c74): pool.py two levels, rollout per-row
  scripted parameters + network-only merge, train.py opens the league at RL start; test_league
  rewritten (27 pass, quick tier of lean/learn 62 + train 5 pass), slow test_compiled 5 pass
  (the static test needed its ticks under no_grad, a test bug). Simulation (CPU, 300 updates):
  numbers below. Next: GPU bench (core vs pool, 0/1/12 snapshots) and the smoke run, queued on
  the GPU lock behind the persona agent's jobs. ETA 22:30 (worst 00:30).
- 22:00 Bound: the 08:30 target dropped (coordinator); finish when adequately done. GPU bench
  done (below); smoke run 10 updates done, its resume step exited at once in the job (debugging
  now). Next: resume fix, BC-stage per-row cost bench, compile check, docs. ETA 23:30 (worst
  01:30).
