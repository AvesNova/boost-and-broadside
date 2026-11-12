# Tank armour and dive strength: progress

Branch `tune/tank-armour` (from `origin/tune/balance-sizes` 7cf7f846). The task: end the
mirror joust and balance the duels through the ship card, not the agent (duel-deny's
finding: turning off a head-on line loses to a ship that holds it, so the fix must make the
joust a bad trade). Hard stop 01:30 KST. Power profile: **performance** (read 23:03). Every
GPU job a `systemd-run --user` unit with `MemoryMax=8G`, one at a time (two orphaned
multiprocessing workers of 128 MiB each, more than a day old, sat on the GPU throughout;
left alone on the coordinator's word).

## Code

- **The armour cone's sharpness** (`head_on_sharpness`, a card field, the target's): the
  aspect is now `1 - (1 - f0) exp(-ln2 (theta / theta_h)^p)` (`physics/combat.py`), 2 by
  default (the old Gaussian cone, every ship file at 2.0). A high `p` is a tank's nose:
  flat near `f0` inside the cone and full damage soon past `theta_h`. It is a card tensor
  (`per_ship`), so a sweep never recompiles; it is a handle too (`Handles.head_on_sharpness`,
  carried through `weapons.combat_card` / `combat_handles` unchanged), so the tuning panel
  and the solver keep working. Test: `test_head_on_sharpness_is_the_exponent_of_the_targets_cone`.
- **Fight statistics** in `bench/game_loop.py` (`fight_choices` in every job summary),
  from the post-merge window (`CIRCLE_WINDOW`, 3 s after a head-on merge ends):
  - `two_circles`: each two-circle fight's turn circles, from both ships' mean turn
    centre (position + speed / turn rate along the left normal of the flight path) and
    radius over the window's ticks where both turn: **overlapping** when the centres are
    nearer than the radii's sum, **side by side** otherwise; with the share that ends in a
    kill in the pair before its next merge ends (or either ship dies or respawns).
  - `post_merge_choices`, by the chooser's type: **same sense** (turning in the bandit's
    rotation sense, the two-circle choice), **opposite sense** (one-circle), **extend**
    (turning less than half the window), **turn v extender** (the bandit extended); each
    with its outcome until the pair's next merge ends: the chooser kills, is killed,
    neither. A choice still open when the game ends is not counted.

## Tools

`sizes.py`, `trajectories.py` (duel-deny's, unchanged), `table.py` (all the goals in one
row; `--fights` for the new statistics), `make_map.py` (balance-sizes'), `launch.sh`.
Candidates set the armour on both types with `{"set": value}`: `f0` as
`head_on_fraction_scale` (of 0.3), `theta_h` as `head_on_angle_scale` (of 42.27 degrees;
today's cards fly 0.5, 21.1 degrees), `p` as `head_on_sharpness`. The dive exchange as
`dive_exchange_scale` (of `flight.dive_exchange` 9.81 m/s^2: today Yongma 0.626 = 6.1,
Jujak 0.314 = 3.1).

## 23:07-23:22 KST: round 1 (`round1/`): the cone grid and the dive, on frozen, C and D2

512 games; the laptop GPU. The 5v5 and the strength against r9 did not finish: the unit was
stopped at 23:22 for the author's playtest, so this round has 1v1 and 2v2 only (tables in
`round1/table.md`, `round1/fights.md`). Armour on both types: `f0` 0.01 (two at 0.001),
`theta_h` 6 / 9 / 12 degrees, `p` 4 / 6 / 8. Dive: Yongma at 7.8 or 9.8 m/s^2, both at 9.8.

| Card | JJ 1v1 mutual, life | YY 1v1 mutual, life | E v T 1v1 / 2v2 | Mixed 1v1 kills m / o / f / t |
|---|---|---|---|---|
| frozen: r9 | 0.80, 151 s | 0.31, 853 s | 0.00 / 0.36 | 0.14 / 0.13 / 0.42 / 0.31 |
| frozen: armour 12 deg p8 | 0.04, 1633 s | 0.00, 1489 s | 0.00 / 0.30 | |
| C: r9 | 0.78, 117 s | 0.65, 198 s | 0.42 / 0.52 | 0.55 / 0.35 / 0.09 / 0 |
| C: armour 6 deg p4-p8 | 0.26-0.33, 640-680 s | 0.16, 490 s | 0.28 / 0.43-0.47 | 0.23 / 0.65 / 0.12 / 0 |
| C: armour 12 deg p8 | 0.03, 1024 s | 0.01, 633 s | 0.27 / 0.44 | 0.02 / 0.86 / 0.13 / 0 |
| D2: r9 | 0.65, 259 s | 0.36, 285 s | 0.50 / 0.45 | 0.24 / 0.68 / 0.08 / 0 |
| D2: armour 6 deg p4-p8 | 0.23-0.28, 690-820 s | 0.15, 460-520 s | 0.52 / 0.48-0.52 | 0.20 / 0.72 / 0.08 / 0 |
| D2: armour 9 deg p4-p8 | 0.10-0.16, 980-1090 s | 0.07, 510 s | 0.53-0.55 / 0.46-0.50 | 0.10 / 0.82 / 0.08 / 0 |
| D2: armour 12 deg p4-p8 | 0.02-0.06, 1290-1450 s | 0.01-0.04, 560-620 s | 0.50-0.52 / 0.48-0.52 | 0.02 / 0.88 / 0.10 / 0 |
| D2: armour 0.001, 12 deg p8 | 0.01, 1249 s | 0.00, 545 s | 0.50 / 0.51 | 0.01 / 0.91 / 0.07 / 0 |
| D2: Y dive 7.8 / 9.8 | 0.64, 255 s / 0.63, 262 s | 0.25, 363 s / 0.17, 750 s | 0.41 / 0.32; 0.26 / 0.20 | |
| D2: both dive 9.8 | 0.11, 725 s | 0.21, 841 s | 0.40 / 0.41 | 0.10 / 0.60 / 0.29 / 0 |
| C: Y dive 9.8; both 9.8 | 0.77; 0.40 | 0.46; 0.49 | 0.25 / 0.21; 0.35 / 0.37 | |

- **The cone ends the mirror joust**, in proportion to its width: `theta_h` matters, `p`
  and `f0` (0.01 against 0.001) barely do. At 12 degrees the mutual head-on share is
  0.01-0.06 on every card.
- **And the duels stop ending.** Mirror 1v1 life goes from 120-290 s to 500-2100 s. The
  kills move to the offset head-on (10-45 degrees: 0.82-0.91 of D2's mixed 1v1 kills): r9
  still flies the joust, only fewer of its hits count. A narrow cone takes the pay out of
  the joust without giving anything else the pay.
- **D2's mixed balance holds** (1v1 0.50-0.55, 2v2 0.46-0.52). C's duel falls to 0.27-0.29:
  C's short-ranged Yongma lived on the head-on.
- **A stronger dive loses the Yongma's duel** (D2 1v1 0.50 to 0.41 at 7.8 m/s^2 and 0.26 at
  9.8; C 0.42 to 0.25). The dive exchange turns store into speed; under r9 (no zoom and
  boom doctrine) the faster Yongma flies wider circles and arrives with less store. Both
  types at 9.8 cut the Jujak joust (0.11) through the Jujak's own speed, with 725 s lives.
- **The fight statistics** (`round1/fights.md`): no ship ever extends after a merge (0.00
  everywhere: r9 has no post-merge extension). 71-92% of post-merge turns are same sense
  (two-circle). In mixed fights the opposite-sense (one-circle) fights go to the Jujak
  (D2: Jujak kills 0.19, Yongma 0.01); the two-circle fights are even (0.12 / 0.15). The
  armour cuts the kill rate of both kinds of two-circle fight (D2 JJ side by side 0.19 to
  0.01).

## 23:25- KST: round 2 (`round2/`, gpubox7): pace compensators, milder and wider cones

512 games on gpubox7 (RTX 5060 Ti, shared with a respawn-waves agent). On C and D2:
the 12-degree p8 cone with damage x1.5 / x2 or shield x0.5; `f0` 0.05 and 0.1; 18 and 24
degrees with damage x2; the cone with both dives at 9.8; and round 1's D2 armour and dive
arms again for their 5v5.

1v1 first (the crowds below when they land):

| D2 + | JJ 1v1 mutual, life; merges m / med / first / p90 | YY 1v1 mutual, life | E v T 1v1 |
|---|---|---|---|
| r9 | 0.63, 253 s; 3.2 / 2 / 0.39 / 7 | 0.39, 289 s | 0.50 |
| 0.01, 12 deg p8 | 0.03, 1557 s; 1.6 / 1 / 0.63 / 4 | 0.02, 580 s | 0.50 |
| ... damage x1.5 | 0.05, 1031 s; 1.5 / 1 / 0.69 / 6 | 0.01, 441 s | 0.56 |
| ... damage x2 | 0.05, 1024 s; 1.8 / 1 / 0.71 / 7 | 0.02, 364 s | 0.51 |
| ... shield x0.5 | 0.05, 915 s; 1.7 / 1 / 0.70 / 6 | 0.02, 395 s | 0.52 |
| 0.05, 12 deg p8 | 0.11, 1095 s | 0.02, 514 s | 0.54 |
| ... damage x1.5 | 0.18, 704 s; 2.6 / 1 / 0.52 / 7 | 0.03, 449 s | 0.51 |
| 0.1, 12 deg p6, damage x1.5 | 0.58, 233 s; 3.0 / 2 / 0.49 / 7 | 0.14, 315 s | 0.54 |
| 0.01, 18 deg p8, damage x2 | 0.03, 1286 s | 0.00, 522 s | 0.54 |
| 0.01, 24 deg p8, damage x2 | 0.04, 1852 s | 0.00, 572 s | 0.50 |
| 0.01, 12 deg p8, damage x1.5, both dive 9.8 | 0.01, 522 s; 1.7 / 2 / 0.44 / 3 | 0.00, 677 s | 0.40 |

**Why the Jujak mirror never ends: a side-by-side two-circle stalemate.** On D2 with the
12-degree cone, the Jujak mirror plays 3,823 two-circle fights in 512 games (7.5 a game),
83% of them side by side, and 1% of those end in a kill (r9 on D2: 21%). The pair circles
apart, passes nose to nose once a lap (harmless now), and nobody gains: in a mirror under
one agent the circles are equal, and r9 has no move that breaks the symmetry (no ship ever
extends, yo-yos or reverses after a merge). Engagements by the two-turn-diameter
definition fall from 1,201 to 219: the two circles sit farther apart than two turn
diameters. Today the joust *is* the stalemate breaker, which is why taking its pay away
costs the pace. More damage does not help (x2: 1557 to 1024 s), because almost no shots
land in a side-by-side fight. The mixed duel keeps its pace (E v T 1v1 life 230 s at damage
x2: the Yongma and Jujak circles differ, so one side gains). The trade-off between the mirror
joust and the mirror pace is set by `f0` alone (0.01: 0.03, 1557 s; 0.05: 0.18, 704 s; 0.1:
0.58, 233 s); round 3 maps it between 0.05 and 0.1.

## Sketch (not built): a scripted yo-yo stalemate breaker

The stalemate above is the case it is for. As a doctrine element beside extend / slash /
bleed / lag / deny (`scripted_doctrine.Doctrine`, the weight multiplied into the combat
layer's aim point and power like the others):

- **Trigger** (a logistic gate, per pair, like `deny_in_crowd`): the pair is in a
  two-circle fight (both turning, opposite world sides of the merge line, the same sense)
  for longer than `yoyo_after_laps` (say 1.5 laps, counted from the pair's merges in the
  tracker the combat layer already keeps for deny) and the nose is far off the target
  (`yoyo_in_stall_angle`, above about 3 stall angles: no shot coming).
- **High phase**: ease the stick toward `yoyo_ease` of full (say 0.5), which widens the
  circle and stops the bleed, and set the power to the store-filling setting (the climb
  into the store: speed traded for store at the climb exchange) for `yoyo_hold` seconds or
  until the store is above `yoyo_in_store_fraction`.
- **Low phase**: full stick back toward the target's lead point with the dive (store
  spent for speed at the dive exchange), so the ship comes back inside the bandit's circle
  with more energy and a better angle: the vertical's geometry folded into the store.
- **Phase memory**: two per-ship scalars in the agent's carry (phase, ticks in phase);
  the doctrine table gains the six fields above, per-type tensors, so the ES and the game
  loop sweep them without a recompile, and `deny_in_crowd`'s crowd gate keeps it a duel
  move.
- **Its test** is duel-deny's: the yo-yo ship against r9 in the 1v1 mirror must score at
  least 0.5 (a best response, not an avoidance), and the mirror life must fall back under
  300 s with the tank cone on. A stronger dive exchange is what would make the low phase
  pay; on its own (round 1) it costs the Yongma, because r9 never climbs into the store.

**Round 2 crowds** (`round2/table.md`): E v T 1v1 / 2v2 / 5v5, strength against r9 (5v5 mixed /
reversed):

| Card | E v T | v r9 |
|---|---|---|
| D2: r9 | 0.50 / 0.46 / 0.54 | 0.511 / 0.488 |
| D2: 0.01, 12 deg p8 | 0.50 / 0.52 / 0.61 | 0.482 / 0.504 |
| D2: ... damage x1.5; x2; shield x0.5 | 0.56 / 0.52 / 0.61; 0.51 / 0.53 / 0.68; 0.52 / 0.50 / 0.66 | 0.475-0.502 / 0.508-0.521 |
| D2: 0.05, 12 deg p8; damage x1.5 | 0.54 / 0.53 / 0.57; 0.51 / 0.48 / 0.64 | 0.507 / 0.474; 0.522 / 0.506 |
| D2: 0.1, 12 deg p6, damage x1.5 | 0.54 / 0.51 / 0.66 | 0.503 / 0.509 |
| D2: 18 / 24 deg, damage x2 | 0.54 / 0.50 / 0.64; 0.50 / 0.52 / 0.66 | 0.476-0.527 |
| D2: cone + damage x1.5 + both dive 9.8 | 0.40 / 0.40 / 0.39 | 0.519 / 0.493 |
| D2: Y dive 9.8; both dive 9.8 | 0.23 / 0.18 / 0.07; 0.41 / 0.40 / 0.31 | 0.445 / 0.487; 0.515 / 0.506 |
| C: r9; C + any cone | 0.38 / 0.53 / 0.62; 0.21-0.30 / 0.39-0.45 / 0.54-0.59 | 0.46-0.54 |

- **Strength holds** on every arm (0.46-0.54 against r9): the card is shared, so the
  scripted agent loses nothing to it.
- **The cone leans the 5v5 toward the Yongma** (D2 0.54 to 0.57-0.61 alone, 0.64-0.68 with
  more damage or less shield): in a crowd the Yongma's flank and tail shots are what kill,
  and the armour removes the head-on hits the Jujak trades with.
- **A stronger dive is a Yongma loss at every size** (5v5 0.07 with the Yongma's alone).
- C does not take the cone (1v1 0.21-0.30): its Yongma's short guns need the head-on.

## 23:47-00:05 KST: round 3 (`round3/`): the band between f0 0.03 and 0.085, on D2

512 games; 1v1 to 10v10 (10v10 at 256). The cone at 9, 12 and 15 degrees (p8), damage x1.5
or x2, and split cones (a stronger Yongma nose):

| D2 + | JJ 1v1 mutual, life | YY 1v1 mutual, life | E v T 1v1 / 2v2 / 5v5 | v r9 |
|---|---|---|---|---|
| r9 | 0.64, 260 s | 0.36, 283 s | 0.50 / 0.48 / 0.55 | 0.506 / 0.493 |
| 0.03, 12 deg, damage x2 | 0.13, 899 s | 0.03, 340 s | 0.56 / 0.54 / 0.67 | 0.457 / 0.508 |
| 0.05, 9 deg, damage x1.5 | 0.26, 569 s | 0.09, 383 s | 0.50 / 0.52 / 0.67 | 0.473 / 0.513 |
| 0.07, 15 deg, damage x1.5 | 0.19, 687 s | 0.04, 418 s | 0.53 / 0.49 / 0.67 | 0.499 / 0.472 |
| 0.07, 12 deg, damage x1.5 | 0.23, 546 s | 0.05, 340 s | 0.52 / 0.53 / 0.64 | 0.517 / 0.513 |
| 0.07, 9 deg, damage x1.5 | 0.32, 445 s | 0.08, 331 s | 0.50 / 0.49 / 0.66 | 0.511 / 0.505 |
| **0.085, 12 deg, damage x1.5** | **0.45, 288 s** | **0.07, 344 s** | 0.52 / 0.53 / **0.65** | 0.520 / 0.465 |
| 0.07, 12 deg, damage x2 | 0.51, 239 s | 0.09, 318 s | 0.52 / 0.52 / 0.67 | 0.464 / 0.497 |
| 0.085, 12 deg, damage x2 | 0.61, 209 s | 0.19, 276 s | 0.51 / 0.47 / 0.69 | 0.483 / 0.518 |
| Y 0.01 / J 0.07, 12 deg, x1.5 | 0.23, 539 s | 0.02, 421 s | 0.55 / 0.54 / 0.68 | 0.488 / 0.484 |
| Y 0.03 / J 0.085, 12 deg, x1.5 | 0.52, 311 s | 0.02, 429 s | 0.55 / 0.56 / 0.68 | 0.518 / 0.504 |

- **One number sets both the joust and the pace: what a dead-on hit does** (f0 times the
  damage factor). At 0.06-0.075 of a full hit the Jujak mirror is 0.13-0.26 mutual and
  570-900 s long; at 0.105, 0.23 and 546 s; at 0.1275, 0.45 and 288 s; at 0.14-0.17, 0.51-0.61
  and 210-240 s (D2 today: 0.15, 0.64, 260 s). The cone's width and sharpness move it far
  less. **Inside the 300 s band, the card takes the Jujak mirror's joust from 0.64 to about
  0.45**, and the Yongma mirror's from 0.36 to 0.07 (344 s).
- **The 5v5 leans to the Yongma under every armour with more damage** (0.64-0.69 against
  0.55). Round 4 looks for a crowd-only counter on the Yongma (its damage, range, shield).
- The mixed duel stays at 0.50-0.56 and strength at 0.46-0.52.

## 00:02-00:12 KST: round 4 (`round4/`): crowd-only counters for the Yongma

512 games, 1v1 to 10v10, the round-3 pick (f0 0.085, 12 degrees, p8, damage x1.5) with the
Yongma's damage, range or shield cut back:

| D2 + 0.085 / 12 deg / p8 + | JJ 1v1 mutual, life | YY 1v1 mutual, life | E v T 1v1 / 2v2 / 5v5 / 10v10 | v r9 |
|---|---|---|---|---|
| (r9, no armour) | 0.65, 259 s | 0.35, 304 s | 0.51 / 0.46 / 0.59 / 0.47 | 0.498 / 0.490 |
| damage x1.5 | 0.49, 300 s | 0.07, 345 s | 0.52 / 0.48 / 0.69 / 0.62 | 0.498 / 0.487 |
| damage Y x1.25, J x1.5 | 0.48, 288 s | 0.05, 437 s | 0.50 / 0.48 / 0.61 / 0.54 | 0.488 / 0.503 |
| damage x1.5, Y bullet life x0.85 | 0.49, 295 s | 0.07, 399 s | 0.54 / 0.44 / 0.60 / 0.60 | 0.481 / 0.504 |
| damage x1.5, Y bullet life x0.7 | 0.49, 296 s | 0.06, 435 s | 0.55 / 0.46 / 0.53 / 0.50 | 0.517 / 0.482 |
| damage x1.5, Y shield x0.8 | 0.46, 285 s | 0.16, 297 s | 0.50 / 0.47 / 0.61 / 0.60 | 0.488 / 0.523 |
| (0.07 / 9 deg) damage x1.5, Y bullet life x0.85 | 0.32, 432 s | 0.11, 380 s | 0.54 / 0.48 / 0.64 / 0.50 | 0.472 / 0.507 |
| (0.07 / 9 deg) damage Y x1.25, J x1.5 | 0.31, 439 s | 0.08, 409 s | 0.51 / 0.44 / 0.58 / 0.51 | 0.505 / 0.531 |

- The Yongma's range (bullet life x0.7, gun range about 340 m) and its damage are crowd
  levers here as on balance-sizes: the duel stays at 0.50-0.55 and the crowds come back to
  0.50-0.61. Both lengthen the Yongma mirror (435 s); the Yongma's shield x0.8 shortens it
  (297 s) and also pulls the crowds down a little (0.61 / 0.60). The confirm combines them.

## 00:12-00:23 KST: confirm (`confirm/`): 1024 games, 1v1 to 10v10

1024 games (10v10 512; SE about ±0.01 at 1v1, ±0.014 at 2v2 and 5v5, ±0.02 at 10v10 and
the strength). Every arm is D2 with the cone f0 0.085, 12 degrees, p8 on both types:

| D2 + cone + | JJ 1v1 mutual, life; merges m / med / first / p90 | YY 1v1 mutual, life | E v T 1v1 / 2v2 / 5v5 / 10v10 | v r9 | Life; kills/min/ship (all lineups) | Trades max | Mixed 1v1 kills m / o / f / t |
|---|---|---|---|---|---|---|---|
| (r9, no cone) | 0.65, 258 s; 3.4 / 3 / 0.39 / 7 | 0.38, 282 s | 0.49 / 0.49 / 0.56 / 0.47 | 0.518 / 0.496 | 103-285 s; 0.20-0.53 | 0.08 | 0.22 / 0.70 / 0.08 / 0 |
| damage x1.5, Y bullet life x0.7 | 0.45, 289 s; 3.6 / 2 / 0.48 / 9 | 0.05, 453 s | 0.55 / 0.46 / 0.56 / 0.47 | 0.478 / 0.506 | 95-453 s; 0.13-0.57 | 0.09 | 0.09 / 0.83 / 0.08 / 0 |
| **B: damage Y x1.25, J x1.5** | 0.49, 298 s; 3.7 / 2 / 0.47 / 9 | 0.06, 415 s | 0.52 / 0.46 / 0.59 / 0.51 | 0.483 / 0.480 | 93-415 s; 0.14-0.58 | 0.09 | 0.10 / 0.83 / 0.07 / 0 |
| **A: damage x1.5, Y bullet life x0.8, Y shield x0.8** | **0.50, 306 s; 3.7 / 2 / 0.48 / 10** | **0.11, 327 s** | **0.51 / 0.44 / 0.53 / 0.48** | **0.512 / 0.480** | 91-327 s; 0.18-0.59 | 0.10 | 0.11 / 0.80 / 0.08 / 0 |
| damage Y x1.25, J x1.5, Y shield x0.8 | 0.48, 295 s; 3.7 / 2 / 0.46 / 9 | 0.09, 329 s | 0.48 / 0.40 / 0.52 / 0.44 | 0.511 / 0.452 | 91-329 s; 0.18-0.59 | 0.10 | 0.11 / 0.80 / 0.08 / 0 |

Fight statistics on A (`confirm/fights.md`), 1v1:

| Lineup | Two-circle fights: overlapping / side by side (share; ends in a kill) | Post-merge choice (share; chooser kills / killed) |
|---|---|---|
| JJ, r9 on D2 | 0.25; 0.20 / 0.75; 0.19 | same sense 0.80; 0.10 / 0.10, opposite 0.20; 0.05 / 0.05, extend 0 |
| JJ, A | 0.17; 0.14 / 0.83; 0.15 | same 0.82; 0.07 / 0.07, opposite 0.18; 0.04 / 0.04, extend 0 |
| YY, r9 on D2 | 0.59; 0.22 / 0.41; 0.30 | same 0.84; 0.13 / 0.13, opposite 0.16; 0.07 / 0.07, extend 0 |
| YY, A | 0.60; 0.14 / 0.40; 0.25 | same 0.86; 0.09 / 0.09, opposite 0.14; 0.02 / 0.02, extend 0 |
| E v T, r9 on D2 | 0.67; 0.27 / 0.33; 0.28 | Yongma: same 0.86; 0.15 / 0.13, opposite 0.14; 0.01 / 0.14 |
| E v T, A | 0.56; 0.33 / 0.44; 0.33 | Yongma: same 0.83; 0.20 / 0.13, opposite 0.17; 0.01 / 0.24 |

## Final state (00:30 KST)

Ship files and the play default: **unchanged** (every card at sharpness 2, today's cone).
Two opt-in maps (`src/bnb/game/maps/`), flown by r9:

- **`armour-tank`** (A; `bnb play --preset armour-tank`): D2 with a tank nose on both
  types (f0 0.085: a dead-on hit does 8.5% instead of D2's 15-21%; half strength at 12
  degrees instead of 21; sharpness 8, flat across the cone, full damage by about 16
  degrees), every gun x1.5, and the Yongma at bullet life x0.8 (gun range about 390 m)
  and shield x0.8 (57).
- **`armour-tank-b`** (B): the same nose, Jujak's guns x1.5 and Yongma's x1.25 (shield and
  range as D2). Duels as A, the Yongma mirror longer (415 s), 5v5 0.59.

**How A plays.** A head-on pass is a bad trade: bullets that meet the nose do 8.5% of a hit,
while anything 16 degrees or more off the nose does a full hit, 1.5 times D2's (on D2 a
hit 20 degrees off the nose does about half). So a merge
no longer decides the fight on its own. The ships pass, turn (four times in five the same
sense, a two-circle fight), and the kill comes on a later pass from a slightly offset
angle (offset head-on 0.80 of the mixed duel's kills; mutual head-on 0.11, from 0.22). The
Jujak mirror still meets nose to nose often (0.50 of its kills, from 0.65), because in a
side-by-side circle the only shots are at the pass. The Yongma mirror nearly stops jousting
(0.11 from 0.38). The Yongma is a shorter-ranged, softer, harder-hitting ship than on D2:
in a crowd its flank shots are what kill (5v5 0.53), and in a duel it still holds even
(0.51). Duels are 2 to 4 merges long (median 2, first-pass 0.48, p90 10); the mirrors run
306 and 327 s a life (D2 258 / 282), crowds 91-104 s.

Images (`images/`): before (D2, r9) `D2-before-jj-0.png` (a seven-merge loop fight, a
head-on kill at 156 s), `D2-before-evt-*.png`; after, A: `D2-tank-jj-*.png` and
`D2-tank-evt-*.png`. (The cone without the crowd counters, the round-3 pick, is
`D2-armour-*.png`: a 12-merge, 9.9-circle Jujak mirror, killed at 271 s.)

**Against the goals** (A, 1024 games): mirror joust JJ 0.65 to 0.50 (not ended), YY 0.38 to
0.11; E v T 0.51 / 0.44 / 0.53 / 0.48 (2v2 0.01 under the 0.45 line, within its SE); strength
0.512 / 0.480 (in 0.47-0.53); life 91-327 s (the mirrors 6 and 27 s over 300); kills a
minute a ship 0.18-0.59 (in band); trades at most 0.10 (D2 0.08).

**What moved what:**

1. **The damage of a dead-on hit (f0 x damage) is the joust lever, and it is also the
   mirror pace lever.** No cone shape separates them: at 0.06-0.075 of a full hit the
   Jujak mirror stops jousting (0.13-0.26) and lasts 570-1550 s. The narrow tank nose
   (high p, 9-12 degrees) is worth it only as a way of keeping flank and offset damage
   high while the dead-on hit is cut.
2. **The mirror stalemate is the agent's, not the card's**: a symmetric side-by-side
   two-circle fight, which r9 never breaks (zero post-merge extensions). More damage, less
   shield and wider cones do not end it; the joust did. An agent move that breaks a
   symmetric circle (the yo-yo sketch above) is what lets the cone go lower.
3. **The cone plus damage leans the crowds to the Yongma**; the Yongma's range, damage and
   shield take it back without touching the duel.
4. **A stronger dive exchange is harmful under r9** (round 1 and 2): Yongma alone at 9.8
   m/s^2 is 0.23 / 0.18 / 0.07 at 1v1 / 2v2 / 5v5; both types at 9.8 cut the Jujak joust
   (0.08-0.11) but pull every size to 0.31-0.41 and stretch the mirrors to 750 s. r9 has no
   climb-and-dive use for the store, so a faster dive only makes wider circles. It belongs
   with the yo-yo, not before it. No card was re-solved and no ship file was written (no
   freeze solve was needed: every candidate is card factors).
5. C (the range card) does not take the cone: its short-gunned Yongma lives on the
   head-on (1v1 0.38 to 0.21-0.30).

Next:
1. Build the yo-yo (or any symmetric-circle breaker) and re-run the cone at f0 0.03-0.05,
   where the mirror joust is gone if the mirror pace can be had another way.
2. A per-type pass-through (the shooter's bullets that cross the nose cone ricochet:
   no damage, no store refund) would be the same lever as f0; the closure-scaled
   head-on damage (duel-deny's next step 2) is a different one and untested.
3. 20v20 was not run on A (every card fails it at the density rule; see balance-sizes).
