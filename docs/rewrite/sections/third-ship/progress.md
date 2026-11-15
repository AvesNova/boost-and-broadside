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
