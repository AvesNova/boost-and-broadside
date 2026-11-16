# Balance map: progress (2026-11-16)

Branch `feat/balance-map`: c4ede6c8 (feat/lethality-sticky, game v23) + origin/feat/balance-eval
(5bcb20bb, merged clean), then **Imugi build 2, ede616c4** merged clean (coordinator, 18:40:
offset approach, lead turn, one-or-two-circle choice; the cards are unchanged, VERSION 23).
The search uses build 2. Code: `bench/balance_map.py`; tests `tests/sections/test_balance_map.py`.

## 18:10-18:50 set-up, search space, the solver's cost

- Built: worktree /home/vizia/avesnova/bnb-map (`.venv` linked), `ribs` 0.12.1 in the `tools`
  extra (the PyPI name is `ribs`; `pyribs` is a stub). On box5090 installed additively into
  /workspace/venv with `uv pip install ribs==0.12.1` (torch unchanged, 2.15.0+cu132); no
  `uv sync` there, since the main checkout's lock is another branch's.
- Search space: 13 handles, each a coordinate in [0, 1] of real-WWII-type bounds (log where
  ratios matter): sustained turn 0.50-0.85 rad/s, nose reversal 0.25-0.55 s, low-speed
  acceleration 8.8-20, climb time at the v23 store 200-500 s, cruise 260-360 m/s, stall 75-106
  m/s, g limit 6-9, shield 344-803, hit damage 10.3-24.0, gun range 400-700 m, store x0.5-2,
  shot cost x0.5-2, dive acceleration 38-50. Every other handle is v23 Jujak's; 30 Hz fire.
- Time to kill (lethality-sticky's definition): the reference's 0.6-1.4 s on the candidate
  sets the shield bounds, the candidate's on the reference the damage bounds, and the damage
  is clipped so the candidate's own mirror is inside the band too. **Not guaranteed between
  two candidates**: the band holds for a pair only if (shield spread) x (damage spread) <=
  1.4 / 0.6 = 2.33; checked when picks are chosen.
- Didn't: the handle solver on free handle sets. 22 of 30 uniform random cards fail, all on
  the low-speed acceleration (the engine's power, which cruise, zero-lift drag and climb set,
  over the stall speed caps it), and a solve is slow on the CPU: 39 s for Jujak (3
  iterations, 1 card), 627 s for 8 cards (16 iterations: a batch runs until its last card
  converges or fails).
- Worked: the cap's closed form (`low_speed_limits`, from `solver.first_guess`; Jujak 21.0
  against its 13.5, Yongma 12.7 against 8.8): the low-speed acceleration is clipped to 85% of
  it, and a card whose cap is under the real floor 8.8 is dropped before the solver (counted).
- Next: the solve's speed on the box (GPU and CPU), then the pilot.

## 18:50-19:00 the solver made affordable, the run's shape

- Coordinator, 18:40-18:58: Imugi build 2 (ede616c4) merged (used for the whole search);
  author: no archetype constraints, the deliverable is (1) a pair of balanced cards from far
  apart cells, (2) ideally 3+ mutually far-apart elites, (3) the full map; picks confirmed
  against each other (pairwise, all 8 conditions, 512+ games). balance-eval round 2
  (7148b709) merged: workers at 16384 environments on the 5090 (4096 is launch-bound).
- Tried: the solver on the 5090's GPU: 64 cards did not finish in 13 min (small tensors,
  launch-bound; killed). Box CPU, eager, 8 processes: 64 cards near the seeds (sigma 0.15)
  in 544 s; 55 solved, 6 no convergence (low-speed acceleration 6-20% off, a nose reversal
  1.6% off), 3 under the low-speed floor.
- Worked: the compiled flight tick (`bnb.runtime.flights`) in a persistent pool, each process
  solving PART = 16 cards padded to one shape: 32 cards in 125-148 s on 2 laptop processes
  (eager: 627 s for 8 cards). The solver's checks run card by card first, so one infeasible
  card does not fail a batch. Low-speed margin 0.85 -> 0.8 (the remaining failures).
- The run: 8 emitters x 32 = 256 cards a generation, as two halves of 128 (4 emitters each, one
  ribs Scheduler each, one shared archive): one half's games play while the other half is
  asked and solved. Each half fills one worker batch per condition (128 variants x 128
  games = 16384 environments). Rechecks: up to 16 a half (each cell's best with objective
  >= 38 and under 512 games) get 384 more games at seed 1. Search games are seed 0 (common
  spawns for every card).
- Doctrine: every card (and the reference) flies v23 Jujak's stored doctrine; the card-derived
  rules (extend-and-return, circle choice, lead) adapt it to each card.
- Next: the pilot (2 generations a half, 512 cards + 6 seeds), launched 19:01.

## 19:00-19:40 the pilot on box5090

- Workers at 16384 environments, 128 variants: cold compile 308-340 s each, one at a time
  (19:01-19:22 KST), RSS about 3.6 GB each after.
- Seeds (512 games a condition, win-rate equivalent against the v23 Jujak team):
  | seed | dm 2v2 | dm 5v5 | elim 5v5 | Frontline 10v10 | objective |
  |---|---|---|---|---|---|
  | v23 Jujak (the mirror) | 52.9 | 48.9 | 54.1 | 52.0 | 45.9 |
  | v23 Yongma | 4.7 | 0.2 | 0.8 | 0.4 | 0.2 |
  | E Yongma (turn x1.12, shield x0.55) | 12.2 | 3.3 | 4.8 | 7.8 | 3.3 |
  | E Jujak (turn x0.95) | 24.0 | 12.7 | 11.7 | 12.9 | 11.7 |
  | G Yongma | 13.8 | 6.6 | 10.1 | 6.8 | 6.6 |
  | G Jujak (turn x0.95, shield x1.16) | 26.2 | 18.9 | 13.5 | 22.0 | 13.5 |
  - The sustained turn rate dominates under build 2 + v23 lethality: Jujak's own card with 5%
    less turn falls to a 39% kill share at 2v2 and 44% at 5v5 (12-24 win-equivalent).
  - The mirror itself reads 45.9: at 512 games the worst of four noisy conditions sits ~4
    points off even, so 45 is at the noise floor of a 512-game check (confirm at 2048).
- Doctrine check (512 games, v23 Yongma card against the Jujak team): its own stored doctrine
  23.5% / 31.0% kill share (dm 2v2 / 5v5) against 23.1% / 30.6% for the seed's Yongma card on
  Jujak's doctrine (the seed's card also takes Jujak's unsearched handles: explosive share 0,
  zero-lift drag). One doctrine for every card does not handicap the energy-style card.
- Throughput: a half (128 asked) solves in 122-156 s on 8 CPU processes (83-87% solved;
  failures: 27 no convergence, 12 under the low-speed floor of 262 asked), plays its 4
  conditions in about 7 min, GPU 94-98%. About 900 solved cards an hour, GPU-bound.
- Search so far (g0000a): median objective 2.2; the worst condition is deathmatch 5v5 for
  56% of cards, Frontline 10v10 for 30%. Best new card 44.2 (dm 2v2 46.8, 5v5 48.1, elim
  55.8, Frontline 10v10 44.5).
- Conversion refit on the pilot (all cards, mostly far from even): chi^2/dof 3-13, so not
  used yet; balance-eval's slopes kept until there are enough near-even cards.
- Laptop: `screen.py` plays the four unsearched conditions for cards at objective >= 40 on
  512 games (laptop_job.sh), from 19:26.
- Next: the long run continues from the pilot's state (no recompile): 40 generations a half
  or 12 h, started automatically when the pilot's search ends.
