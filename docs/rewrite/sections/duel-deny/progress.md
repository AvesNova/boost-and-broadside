# Duel-only deny: progress

Branch `tune/duel-deny` (from `origin/tune/balance-sizes` 7cf7f846). The task: stop the
same-type duels from jousting (64-80% of Jujak-mirror kills are mutual head-on on every card)
without the deny element's strength cost (about -150 Elo, 0.25-0.32 against r9 at 5v5), by
switching the deny on only in duel-like situations. Hard stop 23:00 KST. Power profile:
**performance** (read 21:44). The laptop GPU was this job's alone; every run a
`systemd-run --user` unit with `MemoryMax=8G`.

## The gate (code)

`Doctrine` has two new fields (`src/bnb/controllers/scripted_doctrine.py`), and the combat
layer multiplies the deny's weight by the gate (`scripted_combat.doctrinal`):

- `deny_crowd_in_gun_range` (a falling logistic, default centre 2, scale 0.1): how near an
  enemy is, in own gun ranges, from the nearer of the ship and its committed target;
- `deny_in_crowd` (a falling logistic, default centre 1e4, scale 0.2): the gate, in the
  crowd, the soft count of the known enemies other than the target near either ship of the
  pair. The default centre is far above any count, so the gate is open and every stored
  doctrine flies as before.

Both are doctrine-table columns, so they are per-group tensors in the game loop (no
recompile between candidates) and the doctrine ES (`bench/doctrine_arena.py`) now searches
them too. Test: `test_a_crowd_gated_deny_denies_in_a_duel_only`. The searched gate was
`deny_in_crowd` [0.5, 0.1]: one other enemy in range closes it to 0.001. (At scale 0.2 the
residual 0.03 still tipped the stick near a dead-ahead target, so the search started over
at 0.1.)

The deny itself is the balance-sizes one (weight 0.8, angle 0.39 rad, on-nose 0.625 stall
angles, within 3.49 gun ranges, the doctrine opened by `in_advantage` [0.5, 0.1] with the
other elements off), on both types unless named.

## Tools

`sizes.py` and `trajectories.py` are the balance-sizes tools (copied; `sizes.py` gained
`--mirror-sizes`, so the mirrors play at 1v1 only). `table.py` prints the table below from
a result. `launch.sh` starts a unit. Cards: frozen, `balance-c` (Ys 0.45, Yd 1.6, Yb 0.45 |
J induced 1.5, Js 0.8) and `balance-d2` (Ys 0.5, Yb 0.75, Y head-on fraction 0.7, Yd 1.1 | J
induced 1.6, J head-on fraction 0.5), all flown by r9.

Columns: the mirrors' 1v1 mutual head-on kill share, life and merges a kill (two turn
diameters); energy v turn (Yongma's score) at 1v1 / 2v2 / 5v5; strength, the candidate
against r9 at 5v5 (mixed / mixed reversed). SE at 512 games: ±0.015 1v1 mirrors, ±0.02
2v2 and the strength.

## 21:45-21:53 KST: search (`search/`): crowd radius 1.5 and 3 gun ranges, weight 0.8 and 0.4

512 games:

| Candidate | JJ 1v1 mutual, life, merges | YY 1v1 mutual, life, merges | E v T 1v1 / 2v2 / 5v5 | v r9 (5v5 mixed / reversed) |
|---|---|---|---|---|
| frozen: r9 | 0.80, 155 s, 2.2 | 0.32, 907 s, 3.4 | 0.01 / 0.35 / 0.50 | 0.505 / 0.491 |
| frozen: deny 0.8 | 0.06, 533 s, 5.6 | 0.00, 1249 s, 2.9 | 0.00 / 0.27 / 0.49 | 0.330 / 0.275 |
| frozen: duel deny 0.8, crowd 1.5 gun ranges | 0.06, 536 s, 5.8 | 0.00, 1534 s, 2.9 | 0.00 / 0.31 / 0.51 | 0.410 / 0.377 |
| frozen: duel deny 0.4, crowd 1.5 gun ranges | 0.44, 745 s, 5.2 | 0.03, 1368 s, 2.8 | 0.00 / 0.30 / 0.52 | 0.438 / 0.430 |
| frozen: duel deny 0.8, crowd 3.0 gun ranges | 0.07, 549 s, 5.7 | 0.00, 1326 s, 2.8 | 0.00 / 0.29 / 0.52 | 0.448 / 0.462 |
| frozen: duel deny 0.4, crowd 3.0 gun ranges | 0.44, 700 s, 5.4 | 0.01, 1132 s, 2.9 | 0.00 / 0.32 / 0.53 | 0.443 / 0.458 |
| C: r9 | 0.77, 117 s, 1.4 | 0.64, 210 s, 1.4 | 0.40 / 0.53 / 0.63 | 0.475 / 0.513 |
| C: deny 0.8 | 0.03, 324 s, 3.7 | 0.02, 565 s, 2.2 | 0.39 / 0.48 / 0.65 | 0.188 / 0.231 |
| C: duel deny 0.8, crowd 1.5 gun ranges | 0.04, 332 s, 3.8 | 0.01, 590 s, 2.2 | 0.36 / 0.47 / 0.58 | 0.344 / 0.331 |
| C: duel deny 0.4, crowd 1.5 gun ranges | 0.44, 397 s, 3.8 | 0.14, 524 s, 2.0 | 0.40 / 0.44 / 0.56 | 0.396 / 0.362 |
| C: duel deny 0.8, crowd 3.0 gun ranges | 0.02, 327 s, 3.9 | 0.01, 584 s, 2.2 | 0.39 / 0.50 / 0.60 | 0.421 / 0.411 |
| C: duel deny 0.4, crowd 3.0 gun ranges | 0.44, 408 s, 3.7 | 0.11, 508 s, 2.0 | 0.46 / 0.50 / 0.60 | 0.431 / 0.447 |
| D2: r9 | 0.63, 256 s, 3.3 | 0.38, 275 s, 2.1 | 0.51 / 0.46 / 0.57 | 0.485 / 0.485 |
| D2: deny 0.8 | 0.01, 429 s, 4.2 | 0.00, 575 s, 2.3 | 0.61 / 0.58 / 0.64 | 0.314 / 0.273 |
| D2: duel deny 0.8, crowd 1.5 gun ranges | 0.01, 448 s, 4.5 | 0.01, 565 s, 2.3 | 0.60 / 0.56 / 0.63 | 0.456 / 0.409 |
| D2: duel deny 0.4, crowd 1.5 gun ranges | 0.29, 899 s, 3.4 | 0.04, 529 s, 2.0 | 0.67 / 0.50 / 0.64 | 0.463 / 0.416 |
| D2: duel deny 0.8, crowd 3.0 gun ranges | 0.02, 447 s, 4.4 | 0.01, 484 s, 2.2 | 0.61 / 0.56 / 0.61 | 0.464 / 0.439 |
| D2: duel deny 0.4, crowd 3.0 gun ranges | 0.28, 924 s, 3.7 | 0.05, 635 s, 2.1 | 0.61 / 0.51 / 0.66 | 0.464 / 0.447 |

- **The gate recovers part of the strength, not all of it.** Ungated deny scores 0.19-0.33
  against r9; gated at 1.5 gun ranges 0.33-0.46, at 3 gun ranges 0.41-0.46. The wider the
  "someone else is near" radius, the less often it fires in a team fight, the more strength
  comes back.
- At 1v1 the gate is always open (no other enemy), so every gated arm is the full deny
  there: JJ mutual 0.01-0.07 at weight 0.8 (0.28-0.44 at 0.4, with duels 400-900 s long).
- Weight 0.4 is worse on both counts: it neither ends the joust nor costs less.

## 21:54-21:58 KST: round 2 (`round2/`): radius 5 and 8, weight 0.6, Jujak only

512 games:

| Candidate | JJ 1v1 mutual, life, merges | YY 1v1 mutual, life, merges | E v T 1v1 / 2v2 / 5v5 | v r9 (5v5 mixed / reversed) |
|---|---|---|---|---|
| frozen: r9 | 0.78, 152 s, 2.2 | 0.40, 1007 s, 3.6 | 0.01 / 0.39 / 0.52 | 0.480 / 0.504 |
| frozen: duel deny 0.8, crowd 5 gun ranges | 0.06, 543 s, 5.6 | 0.01, 1176 s, 3.1 | 0.00 / 0.28 / 0.55 | 0.476 / 0.446 |
| frozen: duel deny 0.8, crowd 8 gun ranges | 0.07, 531 s, 5.7 | 0.01, 1281 s, 3.0 | 0.00 / 0.34 / 0.53 | 0.477 / 0.463 |
| frozen: duel deny 0.6, crowd 5 gun ranges | 0.21, 589 s, 5.6 | 0.01, 1303 s, 2.7 | 0.00 / 0.33 / 0.53 | 0.457 / 0.472 |
| frozen: Jujak duel deny 0.8, crowd 5 gun ranges | 0.07, 480 s, 5.3 | 0.31, 1025 s, 3.6 | 0.01 / 0.46 / 0.58 | 0.507 / 0.476 |
| C: r9 | 0.77, 116 s, 1.4 | 0.63, 185 s, 1.4 | 0.39 / 0.54 / 0.62 | 0.521 / 0.487 |
| C: duel deny 0.8, crowd 5 gun ranges | 0.03, 352 s, 3.9 | 0.01, 559 s, 2.3 | 0.37 / 0.48 / 0.62 | 0.400 / 0.407 |
| C: duel deny 0.8, crowd 8 gun ranges | 0.03, 327 s, 3.9 | 0.00, 629 s, 2.2 | 0.33 / 0.56 / 0.63 | 0.406 / 0.401 |
| C: duel deny 0.6, crowd 5 gun ranges | 0.14, 349 s, 4.1 | 0.06, 514 s, 2.1 | 0.41 / 0.51 / 0.63 | 0.422 / 0.451 |
| C: Jujak duel deny 0.8, crowd 5 gun ranges | 0.03, 345 s, 3.8 | 0.63, 193 s, 1.3 | 0.56 / 0.68 / 0.68 | 0.490 / 0.387 |
| D2: r9 | 0.64, 256 s, 3.3 | 0.39, 287 s, 2.1 | 0.51 / 0.47 / 0.56 | 0.490 / 0.493 |
| D2: duel deny 0.8, crowd 5 gun ranges | 0.02, 439 s, 4.5 | 0.01, 484 s, 2.2 | 0.60 / 0.53 / 0.62 | 0.448 / 0.461 |
| D2: duel deny 0.8, crowd 8 gun ranges | 0.01, 420 s, 4.4 | 0.01, 603 s, 2.3 | 0.60 / 0.53 / 0.63 | 0.474 / 0.479 |
| D2: duel deny 0.6, crowd 5 gun ranges | 0.07, 558 s, 4.6 | 0.02, 565 s, 2.1 | 0.63 / 0.50 / 0.65 | 0.483 / 0.465 |
| D2: Jujak duel deny 0.8, crowd 5 gun ranges | 0.01, 445 s, 4.5 | 0.38, 273 s, 2.2 | 0.60 / 0.57 / 0.60 | 0.463 / 0.473 |

- **Radius 8 gun ranges holds strength on two cards of three**: frozen 0.477 / 0.463, D2
  0.474 / 0.479 (within about 0.03 of 0.50 at ±0.02), but C 0.406 / 0.401.
- **Jujak-only deny** ends the Jujak mirror joust as well (0.01-0.07) and leaves the Yongma
  mirror as it was, but in mixed fights the denying Jujak loses to the Yongma that holds its
  line: E v T 2v2 rises by 0.07-0.14 (C 0.54 to 0.68).
- The cost the gate cannot remove: a ship that turns off the line against an opponent who
  does not is shot. Deny wins nothing against a jouster; it only ends the joust when both
  sides deny. Every residual loss against r9 is that.

## 21:59-22:04 KST: confirm (`confirm/`): 1024 games, radius 8 and 12

1024 games (SE ±0.01 1v1 mirrors, ±0.014 2v2 and the strength):

| Candidate | JJ 1v1 mutual, life, merges | YY 1v1 mutual, life, merges | E v T 1v1 / 2v2 / 5v5 | v r9 (5v5 mixed / reversed) |
|---|---|---|---|---|
| frozen: r9 | 0.79, 154 s, 2.2 | 0.35, 1042 s, 3.5 | 0.01 / 0.37 / 0.50 | 0.494 / 0.491 |
| frozen: duel deny 0.8, crowd 8 gun ranges | 0.07, 528 s, 5.6 | 0.01, 1219 s, 3.0 | 0.00 / 0.30 / 0.55 | 0.460 / 0.432 |
| frozen: duel deny 0.8, crowd 12 gun ranges | 0.06, 530 s, 5.6 | 0.00, 1419 s, 2.9 | 0.00 / 0.33 / 0.52 | 0.457 / 0.447 |
| C: r9 | 0.76, 115 s, 1.4 | 0.65, 195 s, 1.3 | 0.40 / 0.51 / 0.63 | 0.517 / 0.494 |
| C: duel deny 0.8, crowd 8 gun ranges | 0.03, 332 s, 3.8 | 0.01, 565 s, 2.2 | 0.37 / 0.53 / 0.63 | 0.420 / 0.449 |
| C: duel deny 0.8, crowd 12 gun ranges | 0.03, 333 s, 3.8 | 0.01, 616 s, 2.2 | 0.38 / 0.51 / 0.62 | 0.423 / 0.416 |
| D2: r9 | 0.64, 257 s, 3.3 | 0.37, 285 s, 2.1 | 0.50 / 0.48 / 0.55 | 0.496 / 0.495 |
| D2: duel deny 0.8, crowd 8 gun ranges | 0.01, 434 s, 4.3 | 0.01, 502 s, 2.3 | 0.61 / 0.57 / 0.62 | 0.462 / 0.466 |
| D2: duel deny 0.8, crowd 12 gun ranges | 0.01, 447 s, 4.4 | 0.00, 552 s, 2.4 | 0.61 / 0.55 / 0.64 | 0.435 / 0.455 |

- **Strength is not held at 1024 games.** Against r9 the duel deny scores 0.43-0.47 (frozen
  0.460 / 0.432, C 0.420 / 0.449, D2 0.462 / 0.466 at radius 8), 0.03-0.08 under 0.50, about
  -20 to -45 Elo, against -150 ungated. Round 2's D2 0.474 / 0.479 at 512 was at the edge of
  its noise.
- **Radius 12 is no better than 8** (0.416-0.457). At 12 gun ranges the gate is shut
  whenever any other enemy is anywhere near the fight, so the remaining loss is from real
  duels inside the 5v5 (a ship isolated with one enemy, the last survivors), where the
  denier meets r9, which holds its line and shoots.
- **The joust does end.** Mutual head-on in the 1v1 mirrors: JJ 0.79 / 0.76 / 0.64 to
  0.07 / 0.03 / 0.01 (frozen / C / D2); YY 0.35 / 0.65 / 0.37 to 0.00-0.01.
- **At a price in pace and in the mixed duel.** Mirror 1v1 life: JJ 154 / 115 / 257 s to
  528 / 332 / 434 s, YY to 502-1419 s (the 300 s band broken everywhere), merges a kill
  2.2 / 1.4 / 3.3 to 5.6 / 3.8 / 4.3. On D2, energy v turn moves from 0.50 / 0.48 / 0.55 to
  0.61 / 0.57 / 0.62 (1v1 / 2v2 / 5v5): when both types deny, the Yongma wins more of the
  mixed duels, and D2's balance is lost. On C and the frozen card E v T barely moves.

## 22:09-22:11 KST: why the residual cost (`versus1/`): the duel deny against r9, 1v1

1024 games; the candidate's score against r9 (the candidate's ship first):

| Card | JJ | YY | Y (candidate) v J (r9) | J (candidate) v Y (r9) |
|---|---|---|---|---|
| frozen: r9 (control) | 0.500 | 0.494 | 0.003 | 0.991 |
| frozen: duel deny | **0.132** | 0.467 | 0.001 | 0.984 |
| C: r9 (control) | 0.499 | 0.509 | 0.402 | 0.608 |
| C: duel deny | **0.060** | **0.306** | **0.150** | 0.450 |
| D2: r9 (control) | 0.499 | 0.500 | 0.499 | 0.511 |
| D2: duel deny | **0.133** | **0.382** | 0.459 | **0.381** |

- **A denier loses the duel to a jouster**: a denying Jujak scores 0.06-0.13 against r9's
  Jujak. Turning off the line while the other ship holds it gives the other ship the shot.
  The deny ends the joust only when both ships deny; it is not a best response to a ship
  that does not.
- So no gate on the situation can hold strength while the deny fires in duels: every duel
  the gate opens in, against r9, costs the denier. In 5v5 the duels are rare, so the cost
  is 0.03-0.08 there, but in a 1v1 against r9 it is decisive.

## Trajectories (`images/`)

D2, the Jujak-mirror 1v1, 32 games each (`traj.sh`):

- before, r9: `D2-jj-r9-0.png` (one merge, a loop and a head-on kill at 58 s), `-1.png`;
  merges a kill 3.65, a kill in 87 s;
- after, the duel deny (crowd 8): `D2-jj-duel-deny-0.png` (four merges, the two ships in
  opposing circles, a figure-eight, an offset kill at 116 s), `-1.png`; merges a kill 5.47,
  a kill in 136 s.

## Result (22:12 KST)

**No: the duel-only deny ends the mirror joust but does not hold strength.** Best
parameters: deny 0.8 on both types (angle 0.39 rad, on-nose 0.625 stall angles, within
3.49 gun ranges), gated by `deny_crowd_in_gun_range` [8, 0.1] and `deny_in_crowd` [0.5,
0.1]. At 1024 games:

| Card | JJ 1v1 mutual, life (r9 → duel deny) | YY 1v1 mutual, life | E v T 1v1 / 2v2 / 5v5 | v r9, 5v5 |
|---|---|---|---|---|
| frozen | 0.79, 154 s → 0.07, 528 s | 0.35, 1042 s → 0.01, 1219 s | 0.01 / 0.37 / 0.50 → 0.00 / 0.30 / 0.55 | 0.460 / 0.432 |
| C | 0.76, 115 s → 0.03, 332 s | 0.65, 195 s → 0.01, 565 s | 0.40 / 0.51 / 0.63 → 0.37 / 0.53 / 0.63 | 0.420 / 0.449 |
| D2 | 0.64, 257 s → 0.01, 434 s | 0.37, 285 s → 0.01, 502 s | 0.50 / 0.48 / 0.55 → 0.61 / 0.57 / 0.62 | 0.462 / 0.466 |

Strength 0.42-0.47 against r9 (target 0.47-0.53); the ungated deny was 0.19-0.33. The
mirror duels also last 1.3-3.4x longer (out of the 300 s band) and D2's mixed duel tips to
the Yongma (0.61). Nothing is changed in the stored doctrines: the gate's default is open
and every stored deny weight is 0, so the play agent and r9 are unchanged.

What to try next:
1. **A best response to the joust, not an avoidance**: a lead turn (start the turn before
   the merge, toward the side that puts the nose on the other ship's tail after the pass),
   or a reversal or bleed at the merge to force the overshoot. The test is the 1v1 against
   r9 (`versus1/`): it must score ≥ 0.5 there, which the deny cannot.
2. **Make the joust a bad trade on the card**, so the jouster stops winning: the head-on
   fraction lever (D, D2: JJ 0.64) further, or head-on damage that scales with the closure
   so a pass at the merge is less lethal than a shot from behind.
3. **The doctrine ES with 1v1 against r9 in its fitness** plus a mirror-joust penalty
   (balance-sizes' path 4): the ES would find whether any deny or lag setting is
   individually rational. The new gate fields are already in its search space.
