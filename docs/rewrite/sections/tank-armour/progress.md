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
