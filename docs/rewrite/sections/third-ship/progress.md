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
