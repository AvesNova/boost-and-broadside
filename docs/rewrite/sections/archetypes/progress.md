# Archetype pairs: progress

Branch `tune/archetypes` (from `origin/tune/tank-armour` 010a0931). The task: pairs of
different ship types, each with its own way to fly, where all three matchups (A v A, B v B,
A v B) are fair and fun at every fleet size (1v1 to 10v10, 20v20 a spot check). Fun: more
decisions and more control; not a tank that wins by wearing the opponent down. Hard stop
08:00 KST. GPU: gpubox7 (RTX 5060 Ti 16 GB, shared with other agents' benches).

## Code

- **Arming distance** (`arming_distance`, a card field, the shooter's; 0 = off, every ship
  file): a hit's damage times a smoothstep of the distance the bullet has travelled over
  `[0, X]`, the distance being its moves so far (this one included) at its present speed
  (`physics/combat.py`, beside the closeness and the aspect). A card tensor, so sweeps never
  recompile; a handle too (`Handles.arming_distance`, through `weapons.combat_card` /
  `combat_handles`). Test: `test_arming_distance_ramps_the_shooters_damage_over_the_first_metres`.
  The researched types' notes now carry the head-on sharpness and the arming distance (the
  notes test had failed since the sharpness landed).

## Tools

- `pairs.py`: a pair as `sizes.py` candidates. Ship A sits in the card table's first slot
  (the Yongma row: lineup "energy"), ship B in the second (the Jujak row: "turn"). A hull sets
  a slot's flight fields: `Y`, `J` (the frozen files), `M` (their geometric mean in flight,
  shield, damage and size: one neutral ship in both slots), then factors.
- `sizes.py` (tank-armour's, unchanged): the candidates in self-play per size, one compiled
  loop a size.
- `score.py`: every goal in one row per candidate, and a `badness` (the summed misses against
  the goals, 0 when all are met) to sort by.
- `make_map.py`: a candidate as an opt-in play map.

Columns: A v B is ship A's score (win = more kills, draws half). AA / BB are the 1v1 mirrors:
mutual head-on kill share, life (s), merges a kill mean / median / p90, H the post-merge choice
entropy in bits (same sense, opposite sense, extend, turn v extender; r9 never extends, so
the practical ceiling is 1 bit: same v opposite sense evenly).

## 00:39-00:52 KST: round 1 (`round1/`): 20 pairs, the arming sweep (1v1, 2v2, 5v5)

512 games a job; mirrors at 1v1. Balance (A's score) and the 5v5 kill mix:

| Pair (A v B) | A v B 1v1 / 2v2 / 5v5 | AA 1v1 mutual, life; merges m / med / p90 | BB 1v1 | A v B 1v1 kills m / o / f / t |
|---|---|---|---|---|
| D2 (Yongma v Jujak, balance-d2) | 0.48 / 0.46 / 0.55 | 0.38, 298 s; 2.2 / 2 / 4 | 0.65, 252 s; 3.3 / 3 / 7 | 0.23 / 0.68 / 0.08 / 0 |
| D2, arming 50 m | 0.50 / 0.46 / 0.59 | 0.35, 290 s | 0.62, 274 s | 0.23 / 0.69 / 0.08 / 0 |
| D2, arming 150 m | 0.47 / 0.42 / 0.52 | 0.16, 448 s | 0.31, **873 s** | 0.17 / 0.76 / 0.07 / 0 |
| D2, arming 300 m | 0.45 / 0.39 / 0.42 | 0.02, 953 s | 0.29, **2371 s** | 0.05 / 0.84 / 0.10 / 0 |
| D2 + cone 0.085 12° p8 dmg x1.5, arming 150 | 0.53 / 0.52 / 0.67 | 0.05, 451 s | 0.17, 668 s | 0.07 / 0.85 / 0.09 / 0 |
| D2 + the same cone, arming 300 | 0.52 / 0.51 / 0.64 | 0.02, 465 s | 0.11, 959 s | 0.04 / 0.85 / 0.10 / 0 |
| frozen, arming 300 | 0.00 / 0.35 / 0.54 | 0.03, 3481 s | 0.27, 2123 s | 0.03 / 0.04 / 0.44 / 0.49 |
| same ship M (control) | 0.53 / 0.47 / 0.50 | 0.66, 261 s; 3.2 / 3 / 6 | 0.65, 232 s | 0.66 / 0.31 / 0.03 / 0 |
| same ship M, arming 150 | 0.50 / 0.45 / 0.52 | 0.43, 590 s; 4.3 / 3 / 9 | 0.46, 486 s | 0.47 / 0.50 / 0.03 / 0 |
| slasher v dogfighter (M) | **0.26 / 0.69 / 0.88** | 0.61, 495 s | 0.67, 214 s | 0.33 / 0.58 / 0.09 / 0 |
| gunner v brawler (M) | 0.14 / 0.08 / 0.04 | 0.74, 466 s; 5.1 / 5 / 11 | 0.70, 222 s | 0.60 / 0.37 / 0.03 / 0 |
| gunner arming 300 v brawler | 0.07 / 0.04 / 0.01 | 0.20, **19 188 s** | 0.72, 219 s | |
| zoomer v level (D2 hulls) | 0.19 / 0.02 / 0.00 | 0.26, 1735 s | 0.75, 252 s | |
| hit-and-run v tough (M) | 0.24 / 0.12 / 0.06 | 0.71, 172 s | 0.61, 385 s | |
| scout v brawler (M) | **0.74 / 0.55 / 0.41** | **0.99, 116 s** | 0.46, 870 s | 0.56 / 0.33 / 0.11 / 0 |
| armoured nose v agile (M) | 0.02 / 0.24 / 0.18 | 0.02, 679 s | 0.66, 198 s | |
| rate (CL x1.3, induced x1.6) v sustained (M) | 0.49 / 0.48 / 0.46 | 0 kills, 13 952 s | 0.65, 537 s | all "tail" |

The pairs (factors on the neutral ship M unless named): slasher = damage x2 at 7.5 Hz
(energy a shot x2), drag x0.8, lift x0.85; dogfighter = damage x0.5 at 30 Hz, lift x1.1,
drag x1.1. Gunner = bullet life x1.6 (912 m), damage x0.6; brawler = life x0.5 (314 m),
damage x1.6. Zoomer = dive x2.5, climb exchange x1.5, store x1.3; level = engine at full
store x1.1, dive x0.5. Hit-and-run = recharge x4 after 0.4 of the delay, shield x0.7; tough
= shield x1.3, recharge x0.25. Scout = vision x1.6, shield x0.8; brawler = vision x0.7,
damage x1.25. Armoured nose = cone 0.03 12° p8, induced x1.3; agile = lift x1.1.

- **Arming distance works like the armour cone: it takes the pay out of the joust, and the
  duels stop ending.** 50 m does nothing; 150 m halves the mirrors' mutual head-on (D2 BB
  0.65 to 0.31) and triples the Jujak mirror's life (252 to 873 s); 300 m ends the joust
  (0.02-0.29) at 950-2400 s. The kills move to the offset head-on, not to the flank or tail:
  in a joust most hits land from range, so the ramp mostly cuts the late, close hits.
- **On the frozen card arming 300 m kills the duel pace** (the Jujak's tail kills, from close
  behind, are the hits it removes): AA 3481 s, BB 2123 s.
- **The size split is each pair's main fault**: slasher v dogfighter goes 0.26 to 0.88 from
  1v1 to 5v5 (the dogfighter's 30 Hz stream wins duels, the slasher's bursts win crowds),
  scout v brawler 0.74 to 0.41 (sight wins duels). Gunner v brawler, zoomer, hit-and-run and
  armoured nose lose at every size as first set.
- **Vision shapes the mirror more than any other field seen**: the long-sighted scout
  mirror is a 99% mutual head-on joust in 116 s; the short-sighted mirror runs 870 s (the
  ships lose each other).
- The rate fighter at lift x1.3 and induced drag x1.6 cannot hold level flight
  (`characteristics.py` gives no cruise); its mirror never kills and its kills are the
  boundary's. Milder rate fighters (lift x1.2, induced x1.3: sustained 0.230 rad/s, cruise
  149 m/s, corner 122 m/s) fly.

## 00:47-00:52 KST: round 2 (`round2/`): the mirror fix, arming with damage (1v1)

| Card | A v B 1v1 | AA mutual, life; merges | BB mutual, life; merges |
|---|---|---|---|
| D2 | 0.49 | 0.36, 274 s; 2.1 | 0.62, 256 s; 3.3 |
| D2, arming 100, dmg x1.25 | 0.51 | 0.34, 235 s; 1.9 | 0.64, 262 s; 3.4 |
| D2, arming 150, dmg x1.25 | 0.51 | 0.31, 302 s | 0.38, 568 s; 3.7 |
| D2, arming 150, dmg x1.5 | 0.52 | 0.41, 217 s | 0.59, 266 s |
| D2, arming 200, dmg x1.25 | 0.48 | 0.21, 424 s | 0.28, 899 s |
| D2, arming 200, dmg x1.5 | 0.52 | 0.33, 282 s | 0.35, 587 s |
| D2, arming 200, dmg x2 | 0.50 | 0.50, 183 s | 0.63, 247 s |
| D2, cone 0.085 12° p8, arming 100, dmg x2 | 0.50 | 0.10, 329 s | 0.43, 285 s; 3.5 |
| D2, cone 0.15 12° p8, arming 100, dmg x1.5 | 0.54 | 0.24, 255 s | 0.61, 218 s |
| M | 0.54 | 0.66, 260 s | 0.65, 236 s |
| M, arming 150, dmg x1.5 | 0.50 | 0.66, 220 s | 0.65, 205 s |

- **Damage buys the pace back and the joust with it.** Every arming x damage cell lies on
  tank-armour's cone curve: at ~260 s of Jujak-mirror life the mutual share is ~0.6 whatever
  the mix; at ~570 s ~0.35. Cone + arming + damage x2 gives BB 0.43 at 285 s, the cone's
  own best (0.45 at 288 s). **Arming distance is a second handle on the same trade-off, not
  a way off it.** With the scripted agent the joust is the only kill a mirror has, so any
  lever that devalues it lengthens the duel in proportion.
- On the neutral ship M arming 150 with damage x1.5 leaves the joust untouched (0.66): M's
  mirror is less sensitive to the ramp than D2's Jujak mirror (why is not measured; a guess:
  its hits land from farther out).

## 00:54-00:59 KST: the mirror study (`mirror/`): what shapes a 1v1 mirror

One ship played against itself at 1v1, 512 games: the neutral ship M with one change, and the
D2 Yongma with its parts undone. Kills m / o / f / t, life, merges a kill mean / median / p90,
choice entropy H:

| Ship | Kills m / o / f / t | Life s | Merges | H |
|---|---|---|---|---|
| M | 0.66 / 0.31 / 0.03 / 0 | 262 | 3.2 / 3 / 6 | 0.61 |
| **M, vision x0.7** | **0.49 / 0.25 / 0.25 / 0** | **1526** | 3.0 / 2 / 7 | **0.94** |
| M, vision x1.3 | 0.90 / 0.10 / 0.01 / 0 | 175 | 2.7 / 3 / 6 | 0.71 |
| M, drag x0.8 (faster) | 0.66 / 0.31 / 0.03 / 0 | 264 | 3.1 / 3 / 6 | 0.64 |
| M, lift x1.15 | 0.71 / 0.26 / 0.03 / 0 | 188 | 2.7 / 3 / 5 | 0.75 |
| M, induced x0.8 / x1.25 | 0.76 / 0.22 / 0.02 / 0; 0.60 / 0.36 / 0.04 / 0 | 328; 214 | 3.3; 2.7 | 0.53; 0.65 |
| M, bullet life x0.7 / x1.4 | 0.62 / 0.36 / 0.02 / 0; 0.68 / 0.28 / 0.04 / 0 | 360; 225 | 3.5; 2.9 | 0.54; 0.66 |
| M, damage x1.5 / shield x0.6 | 0.72 / 0.23 / 0.05 / 0; 0.71 / 0.24 / 0.04 / 0 | 158; 143 | 1.8; 1.6 | 0.77; 0.84 |
| M, 30 Hz at damage x0.5 | 0.47 / 0.52 / 0.02 / 0 | **1574** | 4.3 / 3 / 12 | 0.36 |
| M, 7.5 Hz at damage x2 | 0.74 / 0.23 / 0.03 / 0 | 247 | 3.2 | 0.63 |
| M, spread x3; explosive 0.5; hit radius x1.5; engine x1.3 | 0.64-0.67 / 0.28-0.33 / 0.03-0.05 / 0 | 203-330 | 2.5-3.5 | 0.60-0.72 |
| M, head-on fraction x0.5 | 0.19 / 0.80 / 0.02 / 0 | 1137 | 4.6 / 4 / 9 | 0.41 |
| M, head-on angle x2; stall angle x1.2 | 0.72-0.74 / 0.23-0.25 / 0.02-0.03 / 0 | 308-327 | 3.2-3.4 | 0.51-0.54 |
| M, bleed exponent 1.5 | no kills (cannot hold level flight) | 7866 | | |
| D2 Yongma | 0.36 / 0.54 / 0.10 / 0 | 291 | 2.2 / 2 / 4 | 0.63 |
| D2 Yongma, explosive 0; bullet life x1.33 | 0.39-0.41 / 0.48-0.50 / 0.10-0.11 / 0 | 251-305 | 2.0-2.1 | 0.64 |
| D2 Yongma, head-on fraction x1.43 (undone) | 0.52 / 0.37 / 0.11 / 0 | 211 | 1.6 | 0.75 |
| D2 Yongma, shield x2 (undone) | 0.12 / 0.74 / 0.14 / 0 | 2428 | 3.1 | 0.35 |
| frozen Yongma | 0.35 / 0.54 / 0.11 / 0 | 1018 | 3.6 / 3 / 9 | 0.38 |

- **No flight field moves the mirror** off the joust (speed, lift, bleed, engine, stall:
  0.60-0.76 mutual). Toughness (damage, shield) and the frontal fraction slide it along the
  joust-pace curve. The fire rate does too: a 30 Hz stream at half damage is a 1574 s mirror.
- **Short sight is the only change that brings flank kills into a mirror**: at vision x0.7
  a quarter of the kills are from the flank, the choice entropy is the highest seen (0.94),
  but the ships lose each other and the duel lasts 1526 s. Mirror study 2 (running) looks
  for the vision between x0.75 and 0.9 that keeps the flank kills at a shorter life.
- The D2 Yongma's low mutual share (0.36) is its frontal armour (x0.7) and its halved shield
  together: shield x2 back makes a 2428 s mirror.

## 01:00-01:05 KST: mirror study 2 (`mirror2/`): sight between x0.75 and x0.9

1v1, 512 games; the mirrors and the pair's own A v B:

| Card | AA mutual, life; kills f | BB mutual, life; kills f | A v B 1v1; kills m / o / f / t |
|---|---|---|---|
| M, vision x0.9 | 0.59, 345 s | 0.59, 310 s | 0.51; 0.61 / 0.34 / 0.06 / 0 |
| M, vision x0.85 | 0.54, 386 s | 0.55, 362 s | 0.52; 0.54 / 0.37 / 0.10 / 0 |
| M, vision x0.85, dmg x1.3 | 0.55, 312 s | 0.56, 301 s | 0.51; 0.55 / 0.32 / 0.13 / 0 |
| M, vision x0.8, dmg x1.3 | 0.42, 556 s | 0.39, 482 s | 0.51; 0.40 / 0.40 / 0.20 / 0 |
| M, vision x0.75, dmg x1.3 | 0.47, 698 s | 0.41, 670 s | 0.50; 0.45 / 0.32 / 0.23 / 0 |
| D2, vision x0.9 | 0.44, 597 s | 0.54, 261 s | 0.45; 0.20 / 0.67 / 0.13 / 0 |
| D2, vision x0.8 | 0.29, 1057 s | 0.51, 291 s | 0.35; 0.16 / 0.57 / 0.27 / 0 |
| D2, vision x0.7 | 0.38, 1239 s | 0.13, 716 s | 0.37; 0.25 / 0.29 / 0.46 / 0 |

- Short sight trades along the same joust-pace curve, but it **adds flank kills** that no
  other lever gives (up to 0.2-0.46 of the duel's kills at x0.7-0.8), and the most varied
  post-merge choices (H 0.85-0.92). Damage x1.3 recovers about 70 s of the pace.
- **D2 does not like short sight**: the Yongma's duel falls (0.49 to 0.35 at x0.8): its long
  approaches need to see the Jujak coming.

## 00:53-01:09 KST: round 3 (`round3/`): balance grids at 1v1-10v10

512 games (10v10 256); A v B score by size and the 1v1 mirrors:

| Pair | A v B 1v1 / 2v2 / 5v5 / 10v10 | AA mutual, life | BB mutual, life |
|---|---|---|---|
| slasher (vision x1.4, shield x0.65) v dogfighter | 0.15 / 0.46 / 0.58 / 0.71 | 0.95, 102 s | 0.69, 221 s |
| slasher (vision x1.4, shield x0.8) v dogfighter | 0.22 / 0.62 / 0.78 / 0.83 | 0.92, 157 s | 0.67, 218 s |
| slasher (shield x0.65) v dogfighter (induced x1.3) | 0.46 / 0.20 / 0.10 / 0.06 | 0.67, 218 s | 0.67, 195 s |
| gunner (dmg x0.9) v brawler (dmg x1.2) | 0.24 / 0.48 / 0.53 / 0.54 | 0.72, 241 s | 0.64, 377 s |
| gunner (dmg x1.0) v brawler (dmg x1.3) | 0.26 / 0.50 / 0.56 / 0.63 | 0.72, 208 s | 0.68, 305 s |
| gunner (life x1.3, dmg x0.85) v brawler (life x0.6, dmg x1.3) | 0.24 / 0.30 / 0.27 / 0.25 | 0.65, 288 s | 0.71, 222 s |
| scout (vision x1.3, shield x0.9) v brawler (vision x0.8, dmg x1.15) | 0.73 / 0.51 / 0.41 / 0.41 | 0.91, 146 s | 0.36, 575 s |
| scout (vision x1.3) v brawler (dmg x1.25) | 0.50 / 0.42 / 0.34 / 0.34 | 0.91, 175 s | 0.70, 186 s |
| rate (lift x1.2, induced x1.3) v sustained (lift x0.9, induced x0.8) | 0.04 / 0.05 / 0.01 / 0.02 | 0.68, 177 s | 0.69, 428 s |
| rate, dmg x1.25 | 0.07 / 0.13 / 0.06 / 0.02 | | |
| sniper (spread x0.3, 7.5 Hz, dmg x2, life x1.4) v shotgun (spread x3, 30 Hz, dmg x0.6, life x0.6) | 0.30 / 0.50 / 0.59 / 0.58 | 0.71, 226 s | 0.69, 276 s |

- **Every pair has a duel-crowd split**, and the levers fall in two kinds: sight, range and
  the fire stream move the duel; damage and shield move the crowds. The weapon pairs
  (gunner v brawler, sniper v shotgun) balance the crowds and lose the duel to the
  short-range ship (0.24-0.30): in a duel the brawler gets inside.
- **The rate fighter loses everywhere** (0.01-0.13): lower corner speed and more lift do not
  make up for a worse sustained turn and a slower cruise. Under this agent the sustained
  turn is the fighter's turn.
- A bleeding dogfighter (induced x1.3) makes the A v B crowds nearly killless (life up to
  4866 s, 0.01 kills a minute): not followed.
- Round 4 (running) adds the sustained turn as the duel lever (the brawler's or shotgun's
  induced drag up, or the gunner's down) and sight as the gunner's.
