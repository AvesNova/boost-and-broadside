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

## Share tracking (CPU simulation, docs/rewrite/sections/league-pool/simulate.py, 300 updates)

The real League (two levels) against core 2770a376's capacity rule on its own membership; L
1,024, parity schedule, synthetic win rates and game lengths (the script's docstring). Error
of the realised share of games dealt against the deal-weighted target, mean over windows:
total variation / largest member error.

| rule | 1 update | 10 updates | 100 updates | network calls (mean to 100M / max) | loads |
|---|---|---|---|---|---|
| core capacity rule | 0.309 / 0.147 | 0.300 / 0.140 | 0.246 / 0.137 | - | - |
| two levels K 12, R 4 (default) | 0.266 / 0.088 | 0.106 / 0.029 | 0.010 / 0.004 | 6.3 / 12 | 324 |
| K 12, R 8 | 0.250 / 0.078 | 0.051 / 0.015 | 0.009 / 0.003 | 8.9 / 13.75 | 623 |
| K 16, R 4 | 0.223 / 0.078 | 0.102 / 0.028 | 0.009 / 0.003 | 8.1 / 15 | 337 |
| K 24, R 8 | 0.146 / 0.048 | 0.032 / 0.010 | 0.006 / 0.002 | 13.0 / 22 | 669 |

One update cannot track closer than a slot (1/K); the spare slots (drain rate) set the
10-update error; over 100 updates the target is met. Core's error stays (the cap is
systematic). No overflow game in any run. The CPU test asserts the shares within 0.01 over 100
updates and 0.05 total variation over 10 (fixed win rates, L 512).

## Tick times (GPU, laptop RTX 4070, performance profile, load 1.1-1.7, E 2,048, static compile)

One locked job 21:14-21:19, core and pool back to back per row (bench_collect.py in the
scratchpad: the league allocated as with every slot free, then every league game dealt).

| setting | core 2770a376 | league-pool | calls x rows (core / pool) | scripted league rows (core / pool) |
|---|---|---|---|---|
| RL, no snapshot | 24.14 ms (no network) | 27.89 ms (EMA) | 0 / 2 x 107 | 1,024 / 852 |
| RL, 1 snapshot | 26.37 ms | 28.65 ms | 2 x 86 / 3 x 107 | 852 / 757 |
| RL, 12 snapshots | 32.51 ms | 33.06 ms | 11 x 86 / 9 x 107 | 95 / 252 |
| BC (22:48, twice each) | 24.19, 24.18 ms | 25.51, 25.51 ms | 0 / 0 | 1,024 / 1,024 |

The per-row scripted parameters cost 1.3 ms a tick (BC row: the only change there), about 0.7
s of a 127 s BC update (0.5%); kept, since it gives one scripted graph for both stages. The
rest is the rows run (107-row slots against 86). With 12 snapshots the pool shows more scripted
rows than core because the 4 personas are members.

## Smoke run (GPU, RL from random weights: no BC checkpoint on the current game)

`python -m lean.train --stage rl --updates 10 --out data/league-pool/smoke` then `--resume
... --updates 12` (21:30-21:52, resume 22:28-22:33). 8,100-8,140 steps/s after the first
update (7,276 with its compile; the resumed first update 7,190). Snapshots at updates 4 and 9;
`draining` 3-4 from update 1 on (the spare slots in use), overflow 0, loads 33 in 10 updates.
Realised shares of the 5,949 games dealt in the first 10 updates: yardstick 17.7%, the 4
personas 16.6-18.7% each, EMA 8.8%, snapshot 3.3%; scripted in all 87.9% (core: 100% until the
first snapshot at 5M, then 83%). The learner (random) loses every scripted game, so PFSP itself
puts most slots on scripted members. The resume found a core bug: a GPU resume failed setting
the generators (load_checkpoint maps the RNG states to the GPU); fixed (`set_random_states`
moves them to the CPU), then the resume ran 2 updates without a recompile.
