# Third ship: progress (2026-11-16)

Goal (author, 2026-11-16): a third ship type for the real-scale game, balanced against Yongma
and Jujak (each team-of-type pairing within about 40-60% at 5v5 deathmatch, also at 2v2 and in
Frontline/elimination) and flying differently from both; every card value inside the real WWII
types' range; its own Imugi doctrine. Branch `tune/third-ship` (never merged into
feat/imugi-boundary). Power profile `performance`.

## 06:36 start

- Worktree /home/vizia/avesnova/bnb-third-ship from origin/feat/imugi-boundary (6cb7acff).
- Read: archetypes.md, imugi.md, the design-rule notes, the balance agent's progress (Yongma
  team v Jujak team 0.22 on game v22; the balance moves with the sustained turn, the power,
  damage/shield, not the instantaneous turn; Baekho's running-away elements, extend, the crowd
  extension and the duel extension, stay off by the author's 2026-11-14 decision).
- Harness: `third.py` puts the candidate C into one slot of `bench.game_loop`'s two-type
  variant table (no harness change): variants (C, Jujak) and (Yongma, C), plus the release pair
  as the reference. `cards.py` solves a candidate's handles into a card (`api.solve_handles`).
- Screen 1: three concepts, each a few damage/shield brackets, on the stored doctrines
  (C flies Jujak's stored doctrine, whose weights are near zero: the base pursuit):
  - Z, the radius fighter (A6M5/Ki-43): the lowest stall speed (75), the best sustained turn
    (0.785), slow (cruise 275), draggy, slow to charge, 15 Hz cannon, fragile;
  - K, the power fighter / cannon brawler (Bf 109K-4, La-7, Spitfire XIV): the best
    low-speed acceleration (18.97) and climb (290 s), fast (330), high stall speed (94), a
    middling turn (0.63), a short-range heavy cannon (muzzle 1040, 400 m, 15 Hz, explosive);
  - P, the heavy boom-and-zoom (Tempest/P-47): the worst turn (0.52), low drag, strong dive,
    heavy 30 Hz guns, tough.

## 07:30 screens 1 and 3: the power fighter K, and gun reach

- The handle solves are slow (30 min for 14 in processes while other agents' tests hold the
  CPUs); every Z (radius) and P (heavy boom-and-zoom) set failed on the low-speed
  acceleration (misses 25% and 40%; ill-conditioned with the cruise and the sustained turn);
  K solved. `cards.py` now has a batched solve and a combat-only mode (`--combat-from`: the
  combat handles recomputed on a solved flight, seconds).
- Deathmatch 5v5, 256 games a pairing, C flying Jujak's stored doctrine (release Y v J 0.26):
  - K with its short-range cannon (400 m, 15 Hz): C v J 0.01-0.03, C v Y 0.02-0.24 at
    damage x1.0-1.3 (of Jujak's DPS), shield 90-120. Kill shares 0.35-0.46: with ~40 kills a
    game, a few points of kill share are tens of points of win rate.
  - The same K flight with Jujak's gun: d1.15/s110 0.16 / 0.49; d1.3/s120 0.33 / 0.65.
    With a long 15 Hz cannon (Spitfire's: 593 m, explosive 0.5): d1.3/s120 0.23 / 0.57. A
    480 m gun at d1.3/s120: 0.09 / 0.35.
  - **Gun reach is a strong axis under Imugi** (every threshold is in own gun ranges, and the
    longer gun shoots first in a turning fight): the real types' 400-600 m spread is worth
    more than 30% damage. The set's guns (585-590 m) sit at the real top, so a third ship can
    only be shorter-ranged, and must pay for it elsewhere.
- The balance agent adopted Jujak sustained turn x0.93 (game v23, 47.7 ± 1.1% Y v J); from
  screen 4 on the Jujak card is its v23 card (`candidates/jujak-v23-card.json`).

## 07:40 MILESTONE: a candidate within 40-60% of both (screen 4, Jujak v23)

- Release Y v J with the v23 Jujak: 0.511 ± 0.021 (512 games).
- **K-jgun-d1.15-s110 (the power-fighter flight with a Jujak-type gun, damage x1.15 of
  Jujak's, shield 110): C v J 0.469 ± 0.030, C v Y 0.488 ± 0.029** (256 games each). K with a
  long 15 Hz cannon (Spitfire's, explosive 0.5) at d1.3/s120: 0.516 / 0.559.
- But the profile (`profile.py`) shows K is a middle ship: its turn, corner speed, cruise and
  climb power all lie between Yongma's and Jujak's; only its top speed (lowest), drag
  (highest) and low-speed acceleration (highest) stand out. Balanced, not yet different.
  Next: the radius fighter Z (solved at low-speed acceleration 12: the smallest circle, the
  slowest cruise, 45 deg/s max turn) and the heavy P (solved at 6.0, below the real 8.87:
  out of range), screen 5; radius-not-rate Z variants (sustained turn 0.62-0.71) solving.

## 07:55 screen 5 and the Z doctrines: the radius fighter does not work; P is a stronger Yongma

- Z (radius fighter: 45 deg/s max and 36.6 sustained, out-turning Jujak at every stick
  level, but cruise 275, corner 198, climb power 6516 against Jujak's 9178): C v J 0.06-0.09,
  C v Y 0.09-0.20 at damage x1.0-1.2, shield 70-85, Zero or Jujak gun. Out-turning Jujak does
  not pay in a 5v5 melee when the ship is slow and charges slowly.
- Z doctrines (256 games, Z-zgun-d1.2-s80): the base 0.13 / 0.18; one-circle against every
  type 0.05 / 0.05, two-circle 0.08 / 0.13, + bleed 0.11 / 0.08, + deny 0.03 / 0.06, slash
  0.09 / 0.15. No element helps.
- Radius-not-rate Z (stall 75, sustained 0.62-0.71) does not solve: the low-speed
  acceleration misses by 19-139% (it and the sustained turn are both near-stall excess
  thrust).
- P (heavy boom-and-zoom, sustained 0.52, top 448, solved only at low-speed acceleration
  6.0, below the real 8.87): C v J 0.07-0.17, C v Y 0.30-0.68 (d1.0/s130 to d1.3/s150). A
  stronger Yongma that still loses to Jujak: no cycle, and out of range.
- Running: a Z card-factor screen (which deficit sinks Z: cruise, charge, shield, drag, g,
  power, damage), K doctrines, K with a 30 Hz cannon (tight spread, explosive), K with the
  Fw 190's roll (nose reversal 0.25) solving.

## 08:15 harness validated; doctrines; guns; the roll

- **Sanity (box, 512 games):** C = a copy of Jujak: C v J 0.500 ± 0.021, C v Y 0.801 (the
  release Y v J v22 0.242); C = a copy of Yongma: C v J 0.242 (= the release exactly), C v Y
  0.498. The slot trick is sound.
- Z card factors (from Z-zgun-d1.2-s80, base ~0.11 / 0.18): shield 80->110 0.28 / 0.46; g
  7->9 0.24 / 0.33; engine power x1.3 0.24 / 0.31; damage x1.3 0.21 / 0.28; cruise, charge,
  drag 0.13-0.24. A fragile ship dies; the radius fighter is dropped.
- K doctrines (K-jgun-d1.15-s110, 256 games; base 0.48 / 0.46): slash 0.46 / 0.42, bleed
  0.41 / 0.51, deny 0.36 / 0.49, two-circle against all 0.34 / 0.43, one-circle 0.24 / 0.32,
  lag 0.10 / 0.09 (lag_in_gun_range 0.5 aims far behind). No element beats the base pursuit
  at full weight; a doctrine search with small weights comes later.
- K with a tight-spread explosive 30 Hz cannon (spread 1.5, 1600 m/s, drag 7e-4, 560 m):
  d1.15/s110 0.26 / 0.24, d1.25/s110 0.32 / 0.36: worse than the .50-style gun. The gun
  factors one at a time are on the box (`kgun2`).
- **The roll helps: K with the Fw 190A-8's nose reversal (0.25 s, the real fastest):
  0.58 / 0.61 (g 8); with g 7 0.43 / 0.54**; + sustained turn 0.66 0.75 / 0.76; + drag
  7.6e-5 and cruise 322 0.57 / 0.66. The identity: **the power-and-roll fighter** (Fw 190A
  roll, Bf 109K-4 low-speed acceleration, the draggiest airframe and the lowest top speed,
  g 7). Screen 9: K-roll and K-roll-g7 at damage x1.1-1.2, shield 100-110, 384 games.
- The coordinator flagged the box idle: its lock was held by another agent's compile-memory
  measurement (CPU-bound, GPU at 0%); my kgun2 job is queued there under its lock.

## 08:20 the power-and-roll fighter; its doctrine

- Screen 9 (384 games a pairing, Jujak v23): K-roll-g7 (nose reversal 0.25, g 7) at damage
  x1.2 of Jujak's (23.0 a bullet, Yongma's), shield 110: **C v J 0.496 ± 0.024, C v Y 0.542 ±
  0.024**; d1.15/s110 0.447 / 0.518; with g 8, d1.15/s100 0.479 / 0.531. Shield 100-105 costs
  ~10-15 points against Jujak.
- Doctrine search round 1 (48 random allowed-element doctrines over Jujak's stored one, 128
  games a pairing + 128 mixed against the base doctrine), the top 7 re-checked at 384 + 512
  mixed: the base 0.499 / 0.547, mixed 0.503; **r1-040 mixed 0.575 ± 0.021, r1-034 0.564**
  (both with their elements gated shut: the gain is the sense, two-circle instead of
  one-circle in the mirror); r1-020 (bleed 0.62, deny 0.63, lag 0.18, gated on the speed and
  turn ratios) 0.555 with C v Y 0.586. Round 3 (structured around the two-circle mirror and
  r1-020) running.
- Range: the solved K flights keep Jujak's dive acceleration (40.9, below the real 43.6-44.3;
  Jujak's own value is out of range too). Re-solved at 43.6 (K-roll-g7-dive; with sustained
  0.66 too); sustained 0.593 and cruise 320 do not solve. Screen 11 on the laptop.
- The box: its lock has been held 30+ min by another agent's compile-memory measurement
  (CPU-bound, GPU at 0%); my kgun2 and the mode checks (front1: elimination, Frontline 5v5,
  deathmatch 2v2) are queued behind it and the balance agent's jobs.

## 08:40 doctrine chosen, card trimmed, the ship is in the set

- Doctrine round 3/4 (structured, then 512 + 1024 mixed on the in-range card d1.2/s115):
  base mixed 0.503; **no sense 0.542 ± 0.015**; the two-circle mirror 0.530; r1-020 without
  lag 0.535; p020-12 0.533. Chosen: Jujak's stored doctrine with `sense` 0 (Bonghwang never
  commits to a one-circle after a merge). The release mirrors are one-circle.
- **Fight shape:** the roll makes scissors. Post-merge fight shares (one-circle, two-circle,
  scissors): Bonghwang mirror 0.22 / 0.29 / 0.49 with the base doctrine, 0.20 / 0.42 / 0.38
  with its own; Yongma mirror 0.37 / 0.48 / 0.15, Jujak mirror 0.37 / 0.43 / 0.20; Bonghwang
  against either 0.18-0.23 scissors against 0.12 in Yongma v Jujak. Without the roll (screen
  4) the mirror is 0.21 scissors.
- Trim grid (640 games a pairing, no-sense doctrine, dive 43.6): d1.2/s115 0.473 / 0.601,
  **d1.17/s115 0.504 / 0.531**, d1.15/s117 0.416 / 0.548, d1.15/s120 0.488 / 0.577,
  d1.12/s120 0.447 / 0.558, d1.18/s112 0.457 / 0.530. Neighbours differ by more than their
  SEs: the seed-to-seed noise is about ±0.03 here (as the balance agent saw).
- **Applied** (456a1003, game v23 on this branch): `bonghwang.toml` (d1.17/s115), its
  doctrine, Jujak's v23 card, maps bonghwang-mirror / -jujak / -yongma / three-ships. Quick
  tests on the touched files pass (the game reference re-recorded) except
  `test_ship_sets::test_training_takes_its_games_set_card`, which fails on the base tree too.
  `bnb play --preset three-ships` renders offscreen.
- Running: the tight 5v5 run (laptop: d1.17/s115 and d1.2/s115, 1536 games a pairing, the
  release pair and all three mirrors); the box: elimination 5v5, deathmatch 2v2, Frontline
  5v5, deathmatch 10v10 on d1.2/s115 (queued behind the balance agent's jobs).

## 09:15 modes and sizes; the balance agent's option C

- Against the v23 pair (Bonghwang d1.2/s115; table in results.md): deathmatch 5v5 (1536)
  **0.490 ± 0.012 / 0.567 ± 0.012** (Y v J 0.495); 1v1 0.33 / 0.57 (Y v J 0.31); 2v2 0.31 /
  0.67 (0.28); 10v10 0.60 / 0.46 (0.73); elimination 5v5 0.40 / 0.68 (0.36); Frontline 5v5
  0.33 / 0.64 (0.34). Bonghwang always lands between the other two and moves less than
  they do; where the pair is far apart (small fights, elimination, Frontline: Jujak; 10v10:
  Yongma) it cannot be within 40-60% of both.
- The balance agent's option C (`options/jt0.91-yt1.08-ys0.6`: Jujak turn x0.91, Yongma turn
  x1.08, Yongma shield x0.6, flat over sizes): Y v J 0.51 (2v2) / 0.61 (5v5) / 0.61 (10v10)
  here. Bonghwang d1.2/s115 against it: **2v2 0.36 / 0.36, 5v5 0.57 / 0.56, 10v10 0.74 /
  0.76**: level at 5v5 but steep over sizes (with the others' shields at 80-85 its 115 is
  the tank; a high shield wins the big fights). d1.1/s105: 0.34 / 0.31, 0.45 / 0.42, 0.56 /
  0.56. Next: an option-C card for Bonghwang with a lower shield (90-95) and the level from
  damage or the 0.66 turn (optC2: 5v5 and 10v10 on the laptop, 2v2 on the box).
- Elimination and Frontline against option C queued on the box.

## 09:45 an option-C Bonghwang (lower shield, the 0.66 turn)

- Option-C grid (5v5 256 games, 10v10 192, no-sense doctrine): with the others' shields at
  80-85, Bonghwang needs a 90-95 shield; the level then comes from damage or the turn.
  Bt63 (sustained 0.63) d1.3/s95: 5v5 0.49 / 0.46, 10v10 0.59 / 0.55, elimination 0.50 /
  0.45, but **Frontline 0.37 / 0.25**. **Bt66 (sustained 0.66) d1.1/s95: 5v5 0.53 / 0.52,
  10v10 0.63 / 0.49, elimination 0.52 / 0.54, Frontline 0.50 / 0.41** (Y v J in those runs
  0.61 / 0.61 / 0.49 / 0.64). The 2v2 grid is on the box; a 1024-game 5v5 trim around Bt66
  d1.1/s95 on the laptop.
- Bonghwang d1.2/s115 (the v23 card) against option C: elimination 0.47 / 0.54.
- Box practice (coordinator, ~09:30): its CPU compiles for ~10 min a launch, so box jobs are
  now one mode/size per process with 3-4x the games; small quick jobs stay on the laptop.

## 10:05 the balance agent's options; a persistent box worker

- The balance agent now offers flat-over-sizes options (its results.md): C' (Jujak turn
  x0.91, Yongma turn x1.08, shield x0.55), D (x0.93, x1.08, x0.6), E (x0.95, x1.12, x0.55),
  besides A (v23, in src). Their turns converge (Yongma 0.640-0.664, Jujak 0.696-0.726) and
  the shields equalise (78-85), so Bonghwang's card depends on the author's pick.
- Against option C (x0.91, x1.08, x0.6) the 0.63-turn Bonghwang is steep over sizes (d1.2/s115
  2v2 0.36 / 0.36, 10v10 0.74 / 0.76) and weak in Frontline; the 0.66-turn one with shield
  90-95 is flat: Bt66 d1.1/s95 2v2 0.45 / 0.48, 5v5 0.56 / 0.55 (1024), 10v10 0.63 / 0.49,
  elimination 0.52 / 0.54, Frontline 0.50 / 0.41.
- Against C' (768): ref Y v J 0.52; Bt66 d1.1/s90 **0.514 / 0.541**, d1.05/s95 0.54 / 0.58,
  d1.1/s95 0.60 / 0.58. Against E: Bt66 d1.1/s95 0.38 / 0.41 (E's Yongma turns 0.664, as well
  as Bonghwang): needs a 0.68-0.72 turn (solving).
- `third.py --queue DIR --pull --lock /tmp/gpu.lock`: a persistent worker (compiles once,
  runs job files from its queue, takes the GPU lock only per sweep). On the box: worker
  `dm10` (deathmatch 10v10) with the options job queued; more workers as needed
  (`/workspace/third-queue/<name>`).

## 11:25 finals: a Bonghwang per balance option, every condition

- `finals1.json`: Bonghwang trimmed for A (v23: t0.63 d1.2 s115), C' (t0.66 d1.1 s90), D (t0.66
  d1.1 s95), E (t0.70 d1.1 s90; also d1.0 s95), each with the option's own Y v J reference
  (`finals_table.py` prints the tables). B v J / B v Y:
  - deathmatch 5v5 (1536): A 0.48 / 0.56, C' 0.54 / 0.57, D 0.47 / 0.52, **E 0.50 / 0.50**;
  - 2v2 (1024): A 0.31 / 0.66, C' 0.42 / 0.50, D 0.37 / 0.47, E 0.38 / 0.42;
  - 10v10 (384): A 0.65 / 0.47, C' 0.57 / 0.60, D 0.53 / 0.56, E 0.57 / 0.59;
  - elimination 5v5 (768): A 0.43 / 0.67, C' 0.53 / 0.61, D 0.48 / 0.56, E 0.46 / 0.57;
  - Frontline 5v5 (768): A 0.28 / 0.55, C' 0.49 / 0.47, D 0.37 / 0.45, E 0.50 / 0.48;
  - 1v1 (1024, mostly draws): A 0.34 / 0.59, C' 0.38 / 0.43, D 0.36 / 0.43, E 0.37 / 0.41.
- Under the flat options Bonghwang keeps a slope of its own (weaker at 2v2, stronger at
  10v10). The balance agent's finding holds for it: the shield sets the slope over sizes. The
  slope round (shield 85-90 with more turn or damage) runs on the laptop for E and C' (2v2,
  5v5, 10v10) and as a 24-card grid on the box workers (2v2, 10v10).
- Box: the persistent workers sat behind the balance agent's in-lock compiles; jobs that
  waited too long were moved to the laptop (renamed `.moved` in the queue).
- Fight shape holds in every option: Bonghwang's mirror 0.39-0.45 scissors (Yongma mirror
  0.16-0.21), its cross fights 0.18-0.24 (the pair 0.09-0.13); mutual head-on kills 3-4%.

## 12:25 slope rounds; small fights under E

- Slope round (laptop) and the 24-card slope2 grid (box workers, 2v2 and 10v10; 5v5 queued
  there). Best flat cards by the worst distance from 0.5 over sizes (`retrim.py pick`):
  - **C': Bt66 d1.15/s85: 2v2 0.43 / 0.46, 5v5 0.51 / 0.52, 10v10 0.49 / 0.55**; Bt68
    d1.0/s85: 2v2 0.45 / 0.48, 10v10 0.53 / 0.56 (5v5 pending).
  - E: every card is weak at 2v2 against E's Jujak (0.30-0.39) while level at 5v5 and 10v10;
    best Bt70 d1.2/s85 0.39 / 0.42, 0.48 / 0.52, 0.56 / 0.51.
- Sense variants under E (Bt70 d1.1/s90; 2v2 / 5v5 against Jujak): no sense 0.39 / 0.50,
  Jujak's stored 0.37 / 0.49, two-circle against all 0.34 / 0.38, one-circle against all
  0.13 / 0.25. The doctrine is not the 2v2 gap; the no-sense doctrine stays.
- g 8 (the real maximum; Bonghwang had 7) for E and C' running (2v2, 5v5 laptop; 10v10 box):
  the instantaneous turn is what a duel uses.
- `retrim.py`: a grid of Bonghwang's three balance numbers against any pair's ship files and
  the pick over sizes, so a later balance decision needs one grid, not a search.
