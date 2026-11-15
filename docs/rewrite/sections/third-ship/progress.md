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
