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

## 01:10-01:19 KST: mirror studies 3-4 (`mirror3/`, `mirror4/`): the mirror kit

The neutral ship M against itself (both 1v1 mirror slots, 512 games), with short sight, the
12-degree p8 armour cone at f0, and damage:

| M + | AA mutual, life; merges m / med / p90 | BB mutual, life | A v B kills m / o / f / t |
|---|---|---|---|
| nothing | 0.66, 262 s; 3.2 / 3 / 6 | 0.65, 236 s | 0.66 / 0.31 / 0.03 / 0 |
| cone 0.15, dmg x2 | 0.59, 208 s | 0.58, 196 s | 0.58 / 0.36 / 0.06 / 0 |
| cone 0.1, dmg x2 | 0.38, 349 s; 3.4 / 2 / 9 | 0.35, 286 s | 0.38 / 0.53 / 0.08 / 0 |
| cone 0.07, dmg x2.5 | 0.20, 370 s; 3.0 / 2 / 8 | 0.22, 362 s | 0.19 / 0.70 / 0.12 / 0 |
| vision x0.9, cone 0.1, dmg x2 | 0.26, 386 s | 0.32, 333 s | 0.30 / 0.59 / 0.10 / 0 |
| vision x0.9, cone 0.1, dmg x2.5 | 0.34, 273 s; 2.7 / 1 / 7 | 0.41, 273 s | 0.41 / 0.45 / 0.13 / 0 |
| **vision x0.9, cone 0.08, dmg x2.5** | **0.25, 376 s; 3.0 / 2 / 8** | **0.29, 315 s** | 0.29 / 0.56 / 0.14 / 0 |
| vision x0.85, cone 0.08, dmg x3 | 0.29, 331 s; 2.6 / 1 / 7 | 0.31, 300 s | 0.31 / 0.52 / 0.17 / 0 |
| vision x0.8, cone 0.1, dmg x2 | 0.09, 664 s | 0.10, 690 s | 0.08 / 0.66 / 0.26 / 0 |
| D2 + vision x0.9, cone 0.1, dmg x2 | 0.28, 415 s | 0.57, 195 s | 0.13 / 0.73 / 0.14 / 0 |

- **On the neutral ship the cone and short sight move the joust-pace curve, not just slide
  along it**: at ~300-330 s the M mirror is 0.29-0.31 mutual head-on (D2's Jujak mirror on
  the cone alone: 0.45 at 288 s), with a tenth to a sixth of the kills from the flank and 3
  merges a kill (p90 7-8). This is the **mirror kit** (vision x0.9, cone f0 0.08 12° p8,
  damage x2.5) carried into round 5.
- **The D2 Jujak mirror resists every kit** (0.57-0.66): the bleeding rate fighter jousts
  whatever its armour. The Yongma mirror takes it (0.26-0.36).

## 01:10-01:18 KST: round 4 (`round4/`): two levers a pair

512 games (10v10 256). A v B by size, and the A v B duel's kills and life:

| Pair | A v B 1v1 / 2v2 / 5v5 / 10v10 | A v B 1v1 kills m / o / f / t; life | AA; BB mutual, life |
|---|---|---|---|
| gunner (dmg x0.9, induced x0.85) v brawler (dmg x1.2) | 0.50 / 0.50 / 0.48 / 0.49 | 0.45 / 0.40 / 0.15 / 0; **4866 s** | 0.76, 298 s; 0.65, 378 s |
| gunner (dmg x0.9) v brawler (dmg x1.2, induced x1.25) | 0.63 / 0.43 / 0.35 / 0.27 | 0.16 / 0.05 / 0.79 / 0; 1276 s | |
| gunner (dmg x0.9) v brawler (dmg x1.2, induced x1.45) | 0.16 / 0.05 / 0.00 / 0.00 | | |
| gunner (dmg x0.9, vision x1.25) v brawler (dmg x1.2) | 0.28 / 0.52 / 0.59 / 0.57 | 0.60 / 0.38 / 0.02 / 0; 196 s | 0.89, 166 s; 0.66, 387 s |
| sniper (induced x0.85) v shotgun | 0.67 / 0.54 / 0.53 / 0.58 | 0.34 / 0.36 / 0.30 / 0; 466 s | 0.71, 262 s; 0.69, 266 s |
| sniper v shotgun (induced x1.25) | 0.66 / 0.36 / 0.22 / 0.23 | 0.18 / 0.03 / 0.78 / 0.01; 1024 s | |
| scout (vision x1.2) v brawler (vision x0.85, dmg x1.1) | 0.64 / 0.54 / 0.50 / 0.57 | 0.58 / 0.37 / 0.05 / 0; 249 s | 0.83, 187 s; 0.55, 319 s |
| scout (vision x1.15) v brawler (dmg x1.15) | 0.52 / 0.40 / 0.38 / 0.36 | 0.71 / 0.27 / 0.03 / 0; 195 s | 0.77, 203 s; 0.69, 195 s |
| scout (vision x1.3, shield x1.1) v brawler (vision x0.8, dmg x1.15) | 0.76 / 0.62 / 0.58 / 0.65 | 0.41 / 0.45 / 0.13 / 0; 276 s | 0.87, 218 s; 0.38, 543 s |
| slasher (induced x0.8, shield x0.65) v dogfighter | 0.60 / 0.51 / 0.62 / 0.72 | 0.41 / 0.49 / 0.09 / 0; 506 s | 0.70, 291 s; 0.71, 225 s |
| slasher (induced x0.7, shield x0.65) v dogfighter | 0.50 / 0.26 / 0.23 / 0.19 | 0.35 / 0.35 / 0.29 / 0; 3521 s | |

- **A flat 0.50 can be a stalemate.** The gunner with the better sustained turn scores
  0.50 / 0.50 / 0.48 / 0.49, but its duel with the brawler has 0.1 kills a game (life 4866
  s): the gunner keeps the brawler off and cannot hurt it. The same happens whenever the
  sustained turns of two different ships come near parity with one side unable to reach the
  other (slasher induced x0.7, brawler or shotgun induced x1.25: 1000-3500 s duels, flank
  kills from the turning fight). **The duel lever's middle is a stalemate band**, and the
  A v B duel's life is now a column of the score table.
- The sustained turn moves these pairs far and non-linearly (brawler induced x1.25 to x1.45
  takes the gunner from 0.63 to 0.16 in the duel and from 0.27 to 0.00 at 10v10).
- **Scout v brawler is the steadiest family**: a 250-280 s duel decided at the merge (kills
  0.4-0.7 mutual), the band within reach (0.64 / 0.54 / 0.50 / 0.57). Sight is its duel
  lever, the brawler's damage its crowd lever.

## 01:21-01:27 KST: screen 1 (`screen1/`): new pair ideas with the mirror kit (1v1, 5v5)

Every ship M plus its factors plus the kit (vision x0.9, cone f0 0.08 12° p8, damage x2.5),
512 games:

| Pair (A v B) | A v B 1v1 / 5v5 | A v B duel kills m / o / f / t; life | AA mutual, life | BB mutual, life |
|---|---|---|---|---|
| interceptor (drag x0.8, engine x1.15, vision x1.2, dmg x0.8) v brawler (drag x1.15, vision x0.85, dmg x1.2) | 0.78 / 0.63 | 0.21 / 0.60 / 0.19 / 0; 317 s | 0.19, 448 s | 0.14, 630 s |
| **big gun, big target (radius x1.4, dmg x1.4) v small, light (radius x0.75, dmg x0.85)** | **0.52 / 0.50** | 0.30 / 0.56 / 0.14 / 0; 273 s | 0.43, 224 s | 0.16, **512 s** |
| glass cannon (shield x0.6, dmg x1.5) v steady (shield x1.3, dmg x0.8) | 0.49 / 0.39 | 0.35 / 0.50 / 0.14 / 0; 251 s | 0.51, 170 s | 0.12, 621 s |
| armed gunner (arming 150 m, dmg x1.4, life x1.3) v close fighter (life x0.8) | 0.44 / 0.66 | 0.25 / 0.60 / 0.15 / 0; 305 s | **0.24, 319 s** | **0.24, 360 s** |
| armed gunner, dmg x1.2 | 0.44 / 0.57 | 0.23 / 0.61 / 0.15 / 0; 327 s | 0.15, 405 s | 0.24, 415 s |
| (kit cone over) induced x1.1 v lift x1.1 (the "wide nose" arm: the kit's cone replaced its own) | 0.13 / 0.38 | | 0.25, 320 s | 0.33, 258 s |
| hungry 30 Hz stream (dmg x0.55, cost x1.5) v efficient 10 Hz bursts (dmg x1.5) | 0.23 / 0.08 | | 0.13, 893 s | 0.37, 292 s |
| fast recharge (x3 after half the delay, shield x0.85) v big shield (x1.15, recharge x0.5) | 0.45 / 0.38 | | 0.34, 278 s | 0.17, 398 s |

- **Big gun on a big target v a small, light ship is even at both sizes on its first try**
  (0.52 / 0.50), with a 273 s duel. The small ship's own mirror is long (512 s): a small
  target and light guns. Round 6 tunes the small ship's damage and size.
- **The arming distance makes a real archetype as one ship's trait**: the armed gunner
  (no damage inside 150 m, more damage and range) against a close fighter gives the best
  mirror shapes seen with kills in band (0.24 / 0.24 mutual at 319 / 360 s), but leans
  to the gunner in crowds (0.57-0.66) and away in the duel (0.44).

## 01:20-01:37 KST: round 5 (`round5/`): refinements, and the kit on live pairs

512 games (10v10 256); strength = the candidate pair flown by r9 against r9 itself on the
same card (5v5 mixed / reversed), a sanity check that the card does not break the agent:

| Pair | A v B 1v1 / 2v2 / 5v5 / 10v10 | A v B duel life | AA; BB mutual, life | v r9 |
|---|---|---|---|---|
| scout (vision x1.15) v brawler (vision x0.9, dmg x1.1) | **0.57 / 0.53 / 0.46 / 0.44** | 240 s | 0.78, 196 s; 0.57, 281 s | 0.50 / 0.52 |
| scout (vision x1.1) v brawler (vision x0.85, dmg x1.1) | 0.62 / 0.52 / 0.50 / 0.45 | 267 s | 0.73, 210 s; 0.53, 357 s | 0.50 / 0.49 |
| scout (vision x1.15) v brawler (vision x0.85, dmg x1.1) | 0.61 / 0.52 / 0.49 / 0.54 | 277 s | 0.77, 210 s; 0.54, 313 s | 0.48 / 0.49 |
| the same, kit | 0.77 / 0.68 / 0.62 / 0.71 | 334 s | 0.33, 311 s; 0.12, 743 s | 0.48 / 0.52 |
| sniper (induced x0.9) v shotgun | 0.68 / 0.58 / 0.61 / 0.60 | 320 s | 0.70, 246 s; 0.65, 263 s | 0.50 / 0.49 |
| the same, kit | 0.72 / 0.57 / 0.61 / 0.56 | 331 s | **0.34, 323 s; 0.25, 344 s** | 0.52 / 0.52 |
| sniper (induced x0.85, dmg x0.9) v shotgun | 0.62 / 0.46 / 0.41 / 0.44 | 648 s | 0.71, 320 s; 0.67, 268 s | 0.49 / 0.53 |
| slasher (induced x0.8, shield x0.55) v dogfighter | 0.57 / 0.32 / 0.46 / 0.49 | 403 s | 0.69, 258 s; 0.69, 214 s | 0.53 / 0.54 |
| the same, kit | 0.65 / 0.49 / 0.70 / 0.84 | 293 s | 0.35, 351 s; 0.30, 275 s | 0.51 / 0.52 |
| gunner (dmg x0.9, induced x0.95, vision x1.25) v brawler (dmg x1.2) | 0.47 / 0.57 / 0.60 / 0.58 | 260 s | 0.90, 170 s; 0.64, 364 s | 0.52 / 0.47 |

- **Scout v brawler reaches the band without the kit**: 0.57 / 0.53 / 0.46 / 0.44, a 240 s
  duel, both mirrors in pace (196 / 281 s), strength even. Its fault is the scout mirror:
  a 0.78 mutual joust (long sight finds the other ship nose-on).
- **The kit fixes the mirrors and moves the balance**: every pair's mirrors drop to 0.12-0.37
  mutual, and the pair leans to the ship with the sight or the speed (scout 0.62-0.77,
  slasher 0.70-0.84 in crowds). The kit's x0.9 sight hurts the short-sighted ship most (its
  mirror: 743 s).
- Strength holds everywhere (0.47-0.54 against r9).

## 01:30-01:43 KST: round 6 (`round6/`): big v small, the armed gunner, the interceptor (kit)

512 games (10v10 256), every ship M + factors + the kit:

| Pair | A v B 1v1 / 2v2 / 5v5 / 10v10 | Duel kills m / o / f / t; life | AA; BB mutual, life |
|---|---|---|---|
| big (radius x1.4, dmg x1.4) v small (radius x0.75, dmg x0.85) | 0.57 / 0.50 / 0.49 / 0.47 | 0.32 / 0.54 / 0.14 / 0; 282 s | 0.40, 209 s; 0.17, 502 s |
| **big (x1.3, x1.3) v small (x0.8, x0.9)** | **0.55 / 0.48 / 0.47 / 0.48** | 0.32 / 0.55 / 0.13 / 0; 277 s | 0.39, 231 s; 0.23, 468 s |
| big (x1.4, x1.4) v small (x0.8, x0.9) | 0.54 / 0.45 / 0.50 / 0.48 | 0.32 / 0.56 / 0.12 / 0; 270 s | 0.41, 217 s; 0.23, 420 s |
| big (x1.5, x1.5) v small (x0.75, x0.9) | 0.58 / 0.51 / 0.45 / 0.49 | 254 s | 0.44, 193 s; 0.19, 432 s |
| big (x1.4, x1.4) v small (x0.75, x0.95) | 0.54 / 0.46 / 0.40 / 0.41 | 269 s | 0.44, 213 s; 0.24, 427 s |
| armed gunner (dmg x1.0, vision x1.2) v close fighter | 0.52 / 0.58 / 0.59 / 0.54 | 0.24 / 0.67 / 0.09 / 0; 316 s | 0.13, 485 s; 0.22, 393 s |
| armed gunner (dmg x1.1, vision x1.15) v close fighter | 0.53 / 0.58 / 0.62 / 0.59 | 300 s | 0.16, 449 s; 0.21, 353 s |
| interceptor (vision x1, dmg x0.8) v brawler | 0.60 / 0.51 / 0.52 / 0.55 | 0.15 / 0.63 / 0.22 / 0; 456 s | 0.15, 482 s; 0.16, 580 s |

- **Big v small holds 0.47-0.58 at every size in every variant**: the size of the target and
  the weight of the gun trade evenly in duels and in crowds. The small ship's mirror is its
  one fault (420-500 s: small targets and light guns).
- The armed gunner's mirrors are the least jousty of all (0.13-0.22) but slow (350-485 s).

## 01:37-01:48 KST: round 7 (`round7/`): the kit at full sight (KIT2: cone f0 0.08, dmg x2.5)

| Pair | A v B 1v1 / 2v2 / 5v5 / 10v10 | Duel life | AA; BB mutual, life |
|---|---|---|---|
| **sniper (induced x0.95, dmg x0.95) v shotgun** | **0.50 / 0.55 / 0.53 / 0.43** | 326 s | 0.40, 287 s; 0.34, 359 s |
| sniper (induced x0.95) v shotgun | 0.55 / 0.56 / 0.59 / 0.50 | 313 s | 0.43, 265 s; 0.38, 342 s |
| sniper (induced x0.9) v shotgun | 0.74 / 0.62 / 0.63 / 0.59 | 250 s | 0.43, 290 s; 0.33, 340 s |
| **scout (vision x1.05) v brawler (vision x0.9, dmg x1.1)** | **0.58 / 0.57 / 0.52 / 0.51** | 270 s | 0.38, 366 s; 0.35, 289 s |
| scout (x1.1) v brawler (x0.95, dmg x1.1) | 0.57 / 0.55 / 0.58 / 0.58 | 272 s | 0.38, 315 s; 0.36, 263 s |
| scout (x1.15) v brawler (x0.9, dmg x1.1) | 0.62 / 0.64 / 0.65 / 0.68 | 269 s | 0.45, 330 s; 0.35, 287 s |
| slasher (induced x0.8, shield x0.45) v dogfighter (kit) | 0.62 / 0.44 / 0.64 / 0.78 | 267 s | 0.39, 306 s; 0.35, 273 s |

- **At full sight the kit keeps the mirrors' pace** (263-366 s against 290-743 s at x0.9)
  with 0.34-0.45 mutual: the sniper v shotgun and scout v brawler pairs now have all three
  matchups near the band. Sniper = tight spread, 7.5 Hz, damage x2, range x1.4, a slightly
  better sustained turn; shotgun = spread x3, 30 Hz, damage x0.6, range x0.6.
- Slasher v dogfighter keeps a 1v1-2v2-10v10 zigzag (0.62 / 0.44 / 0.78): not followed.

## 01:43-01:51 KST: round 8 (`round8/`): the small ship's mirror; the armed gunner

| Pair | A v B 1v1 / 2v2 / 5v5 / 10v10 | Duel life | AA; BB mutual, life |
|---|---|---|---|
| big (x1.3, x1.3) v small (x0.8, x0.9), kit2 | 0.57 / 0.48 / 0.50 / 0.47 | 250 s | 0.49, 198 s; 0.27, 405 s |
| **big v small, kit, the small ship's cone at f0 0.12** | 0.65 / 0.57 / 0.51 / 0.49 | 242 s | **0.37, 230 s; 0.44, 288 s** |
| big v small, kit2, small f0 0.12 | 0.71 / 0.56 / 0.53 / 0.51 | 194 s | 0.51, 201 s; 0.51, 241 s |
| big (x1.4) v small (x0.75, dmg x1.0) | 0.54 / 0.48 / 0.38 / 0.37 | 268 s | 0.40, 226 s; 0.28, 393 s |
| **armed gunner (dmg x0.9, vision x1.2) v close fighter, kit** | **0.53 / 0.53 / 0.52 / 0.46** | 330 s | 0.12, **475 s**; 0.22, 372 s |
| armed gunner (dmg x0.95, vision x1.2), kit2 | 0.49 / 0.57 / 0.49 / 0.49 | 324 s | 0.19, 623 s; 0.26, 349 s |

- A softer nose on the small ship (f0 0.12 against the kit's 0.08) brings its mirror into
  pace (468 to 288 s) and costs it the duel (0.55 to 0.65 for the big ship); round 9 gives
  the small ship agility or sight to win the duel back.
- **The armed gunner v close fighter is in band at every size** (worst 0.04) with the least
  jousty mirrors seen (0.12 / 0.22), but the gunner mirror is slow (475 s).
- Merged `origin/tune/maneuvers` (ad2dff4d) for the doctrine re-evaluation once its confirm
  lands; r9's own play is unchanged by the merge (the new elements are off in r9).

## 01:52-02:03 KST: round 9 (`round9/`): the finalists' last knobs

512 games (10v10 256); strength against r9 at 5v5 (mixed / reversed):

| Pair | A v B 1v1 / 2v2 / 5v5 / 10v10 | Duel kills m / o / f / t; life | AA; BB mutual, life; merges mean / median / p90 | v r9 |
|---|---|---|---|---|
| **sniper (induced x0.95, dmg x0.92) v shotgun, kit2** | **0.52 / 0.52 / 0.52 / 0.45** | 0.29 / 0.63 / 0.08 / 0; 316 s | 0.37, 306 s, 3.5 / 2 / 9; 0.35, 334 s, 3.2 / 2 / 8 | 0.52 / 0.51 |
| sniper (dmg x0.97) v shotgun, kit2 | 0.51 / 0.54 / 0.54 / 0.44 | 321 s | 0.35, 268 s; 0.34, 346 s | 0.45 / 0.51 |
| **big v small, kit, small cone f0 0.10** | 0.59 / 0.52 / 0.50 / 0.47 | 0.34 / 0.49 / 0.17 / 0; 256 s | 0.38, 239 s; 0.38, 328 s | 0.50 / 0.50 |
| big v small, small f0 0.12, small vision x1.1 | 0.59 / 0.47 / 0.46 / 0.49 | 213 s | 0.43, 233 s; 0.50, 253 s | 0.53 / 0.51 |
| big v small, small f0 0.10, small induced x0.95 | 0.35 / 0.43 / 0.43 / 0.40 | 231 s | 0.38, 240 s; 0.36, 331 s | 0.52 / 0.49 |
| **armed gunner (arming 100 m, dmg x0.9, vision x1.2) v close fighter, kit** | 0.52 / 0.55 / 0.52 / 0.47 | 0.23 / 0.68 / 0.10 / 0; 311 s | **0.17**, 471 s; **0.21**, 390 s | 0.50 / 0.49 |
| armed gunner (arming 120 m) | 0.55 / 0.54 / 0.57 / 0.50 | 326 s | 0.15, 510 s; 0.22, 395 s | 0.48 / 0.51 |
| **scout (vision x1.05) v brawler (vision x0.92, dmg x1.12), kit2** | 0.60 / 0.56 / 0.53 / 0.51 | 0.30 / 0.58 / 0.12 / 0; 265 s | 0.33, 324 s; 0.36, 270 s | 0.52 / 0.49 |

- **Sniper v shotgun meets nearly every goal at 512 games**: within 0.45-0.55 at every size,
  all three matchups 0.35-0.37 mutual head-on with 3.2-3.5 merges a kill (p90 8-9), lives
  306-334 s, strength even. Confirming at 1024 games with 20v20 (`confirm1/`, running).
- The small ship's sustained turn is a steep duel lever (induced x0.95: 0.59 to 0.35).

## 02:03-02:20 KST: confirm 1 (`confirm1/`): the four finalists at 1024 games, 1v1-20v20

1024 games a lineup (10v10 512, 20v20 256), r9; SE about ±0.015 (1v1-5v5), ±0.02 (10v10),
±0.03 (20v20):

| Pair | A v B 1v1 / 2v2 / 5v5 / 10v10 / 20v20 | Duel kills m / o / f / t; life | AA; BB mutual, life; merges | Trades max | v r9 |
|---|---|---|---|---|---|
| sniper v shotgun, kit2 | 0.49 / 0.49 / 0.52 / **0.39 / 0.38** | 0.34 / 0.58 / 0.08 / 0; 333 s | 0.37, 308 s, 3.4; 0.32, 339 s, 3.2 | 0.18 | 0.50 / 0.52 |
| **big v small, kit, small f0 0.10** | 0.59 / 0.51 / 0.49 / 0.50 / **0.47** | 0.35 / 0.50 / 0.14 / 0; 261 s | 0.36, 236 s, 2.4; 0.38, 326 s, 3.1 | 0.20 | 0.51 / 0.49 |
| armed gunner v close fighter, kit | 0.56 / 0.56 / 0.57 / 0.47 / 0.51 | 0.24 / 0.67 / 0.09 / 0; 310 s | 0.15, 458 s, 3.1; 0.22, 376 s, 3.0 | 0.19 | 0.51 / 0.49 |
| **scout v brawler, kit2** | 0.56 / 0.57 / 0.53 / 0.52 / **0.54** | 0.33 / 0.57 / 0.10 / 0; 284 s | 0.35, 322 s, 3.2; 0.36, 279 s, 2.9 | 0.20 | 0.50 / 0.52 |

- **Big v small and scout v brawler hold at 20v20** (0.47 and 0.54), where every Yongma v
  Jujak card failed (0.07-0.27): pairs on one hull do not carry the arena-size lean.
- Sniper v shotgun loses the big crowds to the shotgun at 1024 games (0.39 / 0.38): the 512
  game 10v10 (0.45) was high. Not carried forward.

## 02:04-02:18 KST: round 10 (`round10/`): the finalists under the maneuvers doctrines

The maneuvers branch's elements (merged), on both types in self-play: "CE" = the crowd
extension and the 0.5 s lead turn; "DCL" = CE plus the duel one-circle sense (mirror only).
The players beat r9 0.80-0.85 at 5v5 on every one of these cards.

| Pair | Doctrine | A v B 1v1 / 2v2 / 5v5 / 10v10 | AA; BB mutual, life, merges, H | Duel life |
|---|---|---|---|---|
| big v small (f0 0.10) | r9 (confirm 1) | 0.59 / 0.51 / 0.49 / 0.50 | 0.36, 236 s, 2.4, 0.66; 0.38, 326 s, 3.1, 0.53 | 261 s |
| big v small | CE | 0.57 / 0.56 / 0.50 / 0.51 | 0.31, 264 s, 2.2, 0.12; 0.28, 373 s, 2.9, 0.13 | 278 s |
| big v small | DCL | 0.57 / 0.51 / 0.53 / 0.48 | 0.51, 142 s, 2.4, 0.51; 0.60, 186 s, 3.6, 0.39 | 270 s |
| sniper v shotgun | CE | 0.53 / 0.57 / 0.60 / 0.61 | 0.25, 382 s; 0.25, 430 s | 395 s |
| sniper v shotgun | DCL | 0.53 / 0.56 / 0.56 / 0.55 | 0.67, 206 s, 4.3; 0.70, 165 s, 3.6 | 371 s |
| armed gunner v close | CE | 0.57 / 0.66 / 0.67 / 0.68 | 0.19, 455 s; 0.16, 441 s | 382 s |
| armed gunner v close | DCL | 0.55 / 0.62 / 0.65 / 0.59 | 0.66, 398 s, 6.1; 0.56, 190 s | 335 s |
| scout v brawler | CE | 0.57 / 0.61 / 0.64 / 0.68 | 0.24, 400 s; 0.26, 328 s | 327 s |
| scout v brawler | DCL | 0.55 / 0.62 / 0.63 / 0.63 | 0.71, 186 s, 4.0; 0.56, 169 s, 3.2 | 322 s |
| big v small, small induced x0.98 | r9 | **0.54 / 0.50 / 0.49 / 0.47** | 0.38, 228 s; 0.35, 351 s | 261 s |
| D2 (the merge check) | r9 | 0.52 / 0.46 / 0.56 / 0.41 | 0.41, 284 s; 0.65, 254 s | 273 s |

- **Big v small is the pair that keeps its balance under the stronger doctrine** (0.48-0.57
  under CE and DCL); the others lean 0.6-0.68 to the ship with range or sight once ships
  extend out of crowds (the extension favours the ship that shoots first on the way back in).
- The doctrines reshape the mirrors: CE lowers the mutual share (0.16-0.31) but lengthens the
  mirror and narrows the post-merge choices (H 0.09-0.13: the extension is nearly the only
  choice made); DCL's one-circle brings the joust back (0.51-0.71) at a quicker pace.
- D2 under r9 after the merge matches round 1 (0.48 / 0.46 / 0.55) within noise: the merge
  leaves r9 as it was.

## 02:14-02:20 KST: screen 2 (`screen2/`): recharge, the store, energy v turn on one hull

512 games, kit2 on both, r9 and CE; A v B 1v1 / 5v5, mirrors' mutual and life:

| Pair | r9 | CE | AA; BB (r9) |
|---|---|---|---|
| fast recharge (x3, half delay, shield x0.85) v big shield (x1.15, recharge x0.5) | 0.42 / 0.36 | 0.46 / 0.40 | 0.42, 254 s; 0.23, 382 s |
| hit-and-run (recharge x4 at 0.4 delay, shield x0.8) v standard | 0.43 / 0.37 | 0.46 / 0.41 | 0.46, 232 s; 0.37, 291 s |
| **zoomer (dive x2, climb x1.5, store x1.3) v level (engine x1.1, dive x0.6)** | **0.55 / 0.47** | 0.54 / 0.41 | **0.18, 311 s**; 0.61, 249 s |
| energy (drag x0.85, engine x1.1, induced x1.1) v turn (lift x1.1, induced x0.9, drag x1.1) | 0.02 / 0.29 | 0.03 / 0.32 | 0.30, 289 s; 0.45, 269 s |

- **The zoomer v level pair is near even on its first try** with kit2, and the zoomer mirror
  is the least jousty mirror in band seen (0.18 at 311 s); the level ship's mirror jousts
  (0.61). Round 11 (running) works on it. The extension does not make the store pay.
- A recharging ship is 0.05-0.1 short and the extension barely helps it (it leaves, but its
  enemies recharge too). Round 11 gives it damage.
- On one hull, the classic energy v turn pair is the frozen card's: the turn fighter takes
  the duel (0.02).

## 02:22-02:27 KST: round 11 (`round11/`): zoomer v level, hit-and-run

512 games (10v10 256), kit2 on both:

| Pair | A v B 1v1 / 2v2 / 5v5 / 10v10 | Duel life | AA; BB mutual, life | v r9 |
|---|---|---|---|---|
| zoomer v level | 0.58 / 0.36 / 0.47 / 0.24 | 204 s | 0.20, 310 s; 0.60, 248 s | 0.47 / 0.50 |
| zoomer (dmg x1.05) v level (cone f0 0.12) | 0.60 / 0.43 / 0.49 / 0.29 | 186 s | 0.18, 273 s; 0.68, 171 s | 0.52 / 0.48 |
| hit-and-run (dmg x1.1) v standard | 0.50 / 0.47 / 0.45 / 0.40 | 254 s | 0.50, 206 s; 0.35, 288 s | 0.52 / 0.48 |
| **hit-and-run (recharge x4 after 0.4 of the delay, shield x0.8, dmg x1.2) v standard** | **0.49 / 0.51 / 0.49 / 0.49** | 245 s | 0.53, 186 s; 0.34, 313 s | 0.49 / 0.50 |

- **Hit-and-run v standard is flat across sizes at 512 games** (0.49-0.51): a fragile ship
  that hits harder and recharges four times faster after a short break, against the plain
  ship. Its own mirror is the more jousty (0.53). Confirming (`confirm3/`, with both
  doctrines and 20v20).
- The zoomer zigzags with the fleet size (2v2 and 10v10 to the level ship): not carried.

## 02:28-02:38 KST: screens 3-4 (`screen3/`, `screen4/`): flight-distinct pairs on one hull

Kit2 on both, 512 games, A v B 1v1 / 5v5:

| Pair | A v B | AA; BB mutual, life |
|---|---|---|
| jouster (armoured nose f0 0.03 at 20°, dmg x1.2) v turner (induced x0.9) | 0.22 / 0.56 | 0.05, 751 s; 0.39, 346 s |
| jouster (induced x1.1, dmg x1.5, 20°) v turner | 0.09 / 0.50 | 0.06, 649 s; 0.37, 327 s |
| jouster (dmg x1.3, cone 0.05 at 30°) v turner | 0.32 / 0.70 | 0.08, 1326 s; 0.37, 316 s |
| hunter (induced x1.05, vision x1.15, drag x0.9) v turner | 0.35 / 0.54 | 0.47, 314 s; 0.35, 303 s |
| hunter (induced x1.1, vision x1.35, drag x0.9) v turner | 0.25 / 0.54 | 0.64, 337 s; 0.34, 287 s |
| hunter (induced x1.05, vision x1.2) v turner (induced x0.95) | 0.19 / 0.43 | 0.48, 330 s; 0.35, 321 s |

- **A sustained-turn gap of 5-10% decides the duel** whatever is set against it: an armoured,
  harder-hitting nose (0.07-0.32) or the longest sight (0.19-0.35). The crowds stay near even.
  Under r9 a flight-distinct pair balances in the duel only the D2 way, where the better
  instantaneous turner bleeds to below the other's sustained turn.
- An armoured nose makes its own mirror a stalemate (650-1330 s).

## 02:22-02:40 KST: confirm 2 (`confirm2/`): 1024 games, 1v1-20v20

| Pair | Doctrine | A v B 1v1 / 2v2 / 5v5 / 10v10 / 20v20 | AA; BB mutual, life, merges | Duel life | v r9 |
|---|---|---|---|---|---|
| big v small (small induced x0.98, f0 0.10) | r9 | 0.54 / 0.49 / 0.47 / 0.43 / **0.36** | 0.37, 234 s, 2.3; 0.36, 335 s, 3.2 | 263 s | 0.49 / 0.50 |
| the same | CE | 0.49 / 0.49 / 0.47 / 0.44 / 0.43 | 0.31, 273 s; 0.28, 393 s | 274 s | 0.80 / 0.81 |
| the same | DCL | 0.51 / 0.50 / 0.49 / 0.48 / 0.38 | 0.49, 140 s; 0.60, 184 s | 271 s | 0.82 / 0.81 |
| **scout (x1.03) v brawler (x0.92, dmg x1.12)** | r9 | **0.56 / 0.56 / 0.55 / 0.54 / 0.55** | 0.36, 328 s, 3.3; 0.35, 276 s, 2.9 | 283 s | 0.47 / 0.50 |
| armed gunner (dmg x0.85) v close fighter | r9 | 0.52 / 0.54 / 0.52 / 0.44 / 0.47 | 0.17, 511 s; 0.22, 408 s | 317 s | 0.49 / 0.48 |

- The small ship's x0.98 induced drag also helps it in the crowds (10v10 0.50 to 0.43, 20v20
  0.47 to 0.36): the agility is a lever at every size, not a duel-only one.
- **Scout v brawler is flat at 0.54-0.56 from 1v1 to 20v20**: one uniform lean to take out.
  Confirm 4 (running) moves the brawler's damage and the scout's sight.

## 02:28-02:45 KST: confirm 3 (`confirm3/`): hit-and-run v standard, 1024 games, every doctrine

| Doctrine | A v B 1v1 / 2v2 / 5v5 / 10v10 / 20v20 | AA (hit-and-run); BB (standard) mutual, life, merges mean / median / p90 | Duel kills m / o / f / t; life | Trades max | v r9 |
|---|---|---|---|---|---|
| r9 | **0.49 / 0.48 / 0.47 / 0.49 / 0.44** | 0.55, 189 s, 2.3 / 1 / 6; 0.35, 292 s, 3.3 / 2 / 8 | 0.46 / 0.46 / 0.08 / 0; 241 s | 0.22 | 0.50 / 0.50 |
| CE | **0.50 / 0.50 / 0.51 / 0.49 / 0.55** | 0.46, 215 s; 0.22, 359 s | 0.36 / 0.55 / 0.09 / 0; 265 s | 0.17 | 0.81 / 0.80 |
| DCL | **0.49 / 0.49 / 0.51 / 0.49 / 0.50** | 0.67, 128 s; 0.69, 182 s | 0.34 / 0.58 / 0.08 / 0; 267 s | 0.17 | 0.83 / 0.83 |

- **The most even pair of the night, under every doctrine and at every fleet size** (0.44-0.55
  at 20v20, 0.47-0.51 elsewhere). Its weakness is fun, not balance: the hit-and-run mirror is
  a quick joust (0.55 mutual, a median of one merge a kill), and trades reach 0.22 at 20v20.
  Round 12 (running) arms the hit-and-run ship's nose. Play map `arch-hit-and-run`.

## 02:40-02:46 KST: confirm 4 (`confirm4/`): scout v brawler without the lean

1024 games (10v10 512, 20v20 256), r9:

| Pair | A v B 1v1 / 2v2 / 5v5 / 10v10 / 20v20 | AA; BB mutual, life, merges | Duel life | v r9 |
|---|---|---|---|---|
| scout x1.03 v brawler (x0.92, dmg x1.16) | 0.57 / 0.56 / 0.51 / 0.52 / 0.53 | 0.34, 328 s; 0.38, 265 s | 259 s | 0.52 / 0.51 |
| **standard (vision x1) v brawler (vision x0.92, dmg x1.12)** | **0.54 / 0.54 / 0.49 / 0.52 / 0.54** | 0.33, 335 s, 3.2 / 2 / 8; 0.34, 272 s, 2.8 / 1 / 7 | 280 s | 0.50 / 0.49 |
| standard v brawler (x0.92, dmg x1.15) | 0.53 / 0.52 / 0.51 / 0.52 / 0.45 | 0.29, 331 s; 0.38, 260 s | 269 s | 0.50 / 0.50 |
| big v small (small induced x0.99) | 0.57 / 0.51 / 0.47 / 0.47 / 0.42 | 0.37, 233 s; 0.38, 319 s | 250 s | 0.52 / 0.48 |

- With the scout's sight back at x1 the pair is the standard ship against a short-sighted
  brawler (vision x0.92, damage x1.12): within 0.49-0.54 at every size, both mirrors 0.33-0.34
  mutual at 272-335 s, 3 merges a kill.
- **A three-ship set?** The hit-and-run ship and the brawler are each even against the same
  standard ship. The triangle's third side, hit-and-run v brawler, is running (`triangle/`).

## 02:44-02:52 KST: round 12 (`round12/`): the hit-and-run ship's nose

512 games (10v10 256), r9; the hit-and-run ship's own cone f0 (kit2's is 0.08):

| Hit-and-run + | v standard 1v1 / 2v2 / 5v5 / 10v10 | AA (hit-and-run) mutual, life, merges | BB mutual, life | Trades |
|---|---|---|---|---|
| (confirm 3: f0 0.08, dmg x1.2) | 0.49 / 0.48 / 0.47 / 0.49 | 0.55, 189 s, 2.3 / 1 / 6 | 0.35, 292 s | 0.22 |
| f0 0.06 | 0.57 / 0.53 / 0.52 / 0.51 | **0.39, 264 s**, 2.9 / 1 / 7 | 0.37, 296 s | 0.13 |
| f0 0.05 | 0.58 / 0.52 / 0.54 / 0.49 | 0.24, 336 s, 3.0 / 1 / 8 | 0.35, 278 s | 0.14 |
| f0 0.05, dmg x1.3 | 0.58 / 0.55 / 0.56 / 0.50 | 0.31, 302 s | 0.38, 312 s | 0.14 |
| f0 0.04, dmg x1.3 | 0.59 / 0.57 / 0.54 / 0.53 | 0.12, 366 s | 0.40, 293 s | 0.14 |

- A harder nose fixes the hit-and-run mirror (0.55 to 0.39 mutual at f0 0.06, life in band)
  and costs the standard ship the duel (0.49 to 0.57): confirm 5 trims the damage back.

## 02:47-02:55 KST: the triangle (`triangle/`): hit-and-run v brawler, 1024 games

| Doctrine | A v B 1v1 / 2v2 / 5v5 / 10v10 / 20v20 | AA; BB mutual, life | Duel life | v r9 |
|---|---|---|---|---|
| r9 | **0.53 / 0.52 / 0.52 / 0.50 / 0.50** | 0.56, 186 s; 0.36, 278 s | 228 s | 0.51 / 0.50 |
| CE | 0.54 / 0.55 / 0.55 / 0.56 / 0.61 | 0.46, 229 s; 0.28, 323 s | 267 s | 0.79 / 0.79 |
| DCL | 0.54 / 0.55 / 0.55 / 0.58 / 0.61 | 0.67, 128 s; 0.58, 164 s | 242 s | 0.82 / 0.81 |

- **Under r9 the three ships (standard, hit-and-run, brawler) are pairwise even at every
  size** (all nine 1v1-20v20 cells of the three pairs within 0.44-0.54). Under the
  maneuvers doctrines the hit-and-run ship gains on the brawler in big crowds (0.56-0.61):
  the extension out of a furball is where a fast recharge pays.

## 02:51-03:09 KST: confirms 5-6 (`confirm5/`, `confirm6/`): the armoured-nose hit-and-run ship

1024 games (10v10 512, 20v20 256), r9 unless named. "Hit-and-run 2" = recharge x4 after 0.4
of the delay, shield x0.8, damage x1.15, its own cone at f0 0.06:

| Pair | A v B 1v1 / 2v2 / 5v5 / 10v10 / 20v20 | AA; BB mutual, life, merges | Duel life | v r9 |
|---|---|---|---|---|
| hit-and-run 2 v standard (confirm 5) | 0.56 / 0.48 / 0.50 / 0.48 / 0.48 | 0.37, 277 s, 2.9; 0.36, 298 s, 3.2 | 262 s | 0.48 / 0.50 |
| hit-and-run 2 v standard (confirm 6, the same card) | 0.53 / 0.49 / 0.44 / 0.47 / 0.43 | 0.34, 266 s; 0.39, 304 s | 265 s | 0.49 / 0.50 |
| hit-and-run 2 v brawler | 0.60 / 0.53 / 0.49 / 0.46 / 0.53 | 0.36, 261 s; 0.36, 281 s | 256 s | 0.48 / 0.48 |
| hit-and-run 2 (vision x0.96) v standard | 0.50 / 0.45 / 0.44 / 0.42 / 0.41 | 0.34, 277 s; 0.38, 293 s | 274 s | 0.50 / 0.50 |
| hit-and-run 2 (vision x0.96) v brawler | 0.55 / 0.49 / 0.47 / 0.41 / 0.43 | | 270 s | 0.52 / 0.47 |
| hit-and-run 2 (vision x0.96) v standard, CE | 0.51 / 0.44 / 0.44 / 0.43 / 0.40 | 0.19, 320 s; 0.23, 363 s | 296 s | 0.79 / 0.82 |
| hit-and-run 3 (f0 0.06, dmg x1.12) v standard | 0.54 / 0.49 / 0.46 / 0.45 / 0.39 | 0.34, 270 s | 273 s | 0.49 / 0.50 |
| hit-and-run 3 (f0 0.05, dmg x1.12) v standard | 0.55 / 0.50 / 0.49 / 0.44 / 0.46 | 0.18, 364 s | 280 s | 0.49 / 0.48 |
| standard v brawler, CE | 0.54 / 0.57 / 0.58 / 0.55 / 0.59 | 0.20, 401 s; 0.27, 331 s | 322 s | 0.81 / 0.83 |
| standard v brawler, DCL | 0.53 / 0.57 / 0.58 / 0.61 / 0.56 | 0.68, 186 s; 0.55, 166 s | 317 s | 0.83 / 0.84 |

- **The run-to-run spread is larger than the quoted SE.** The same card in confirm 5 and
  confirm 6 differs by 0.034 at 1v1 (SE 0.010) and 0.061 at 5v5 (SE 0.015). A cell's real
  uncertainty at 1024 games is nearer ±0.03-0.04 than ±0.015; set 4 repeats one card to
  measure it. Differences under ~0.05 between candidates are not resolved.
- With the hit-and-run ship's nose at f0 0.06, all of its matchups have mirrors at 0.33-0.39
  mutual and 260-305 s, with 3 merges a kill.
- Standard v brawler leans to the standard under the doctrines (0.53-0.61): sight pays more
  when ships extend and come back.

## 02:55-03:08 KST: roster 1 (`roster1/`): every ship against the standard

512 games (10v10 256), r9, every ship M + kit2 + its own factors, against the standard:

| Ship v standard | 1v1 / 2v2 / 5v5 / 10v10 | Its mirror: mutual, life, merges | Duel life |
|---|---|---|---|
| big (radius x1.3, dmg x1.3) | 0.59 / 0.54 / 0.53 / 0.49 | 0.48, 199 s, 2.4 | 224 s |
| **small (radius x0.8, dmg x0.9, f0 0.10)** | **0.45 / 0.50 / 0.49 / 0.52** | 0.37, 308 s, 3.2 | 268 s |
| armed gunner (arming 100 m) | 0.51 / 0.60 / 0.52 / 0.48 | 0.17, 625 s | 287 s |
| close fighter (range x0.8) | 0.54 / 0.52 / 0.47 / 0.45 | 0.29, 450 s | 330 s |
| sniper | 0.63 / 0.53 / 0.49 / 0.39 | 0.37, 325 s | 296 s |
| shotgun | 0.59 / 0.50 / 0.46 / 0.43 | 0.32, 366 s | 282 s |
| zoomer | 0.43 / 0.37 / 0.38 / 0.17 | 0.20, 309 s | 203 s |
| level | 0.37 / 0.46 / 0.38 / 0.31 | 0.60, 266 s | 217 s |
| glass cannon (shield x0.6, dmg x1.5) | 0.43 / 0.48 / 0.46 / 0.40 | 0.63, 116 s, 1.2 | 171 s |

- The small ship is the next set member (within 0.45-0.52 of the standard). Set 4 (running)
  plays every pair of standard, hit-and-run 2, brawler and small.

## 03:10-03:24 KST: set 4 (`set4/`): every pair of a four-ship set

Every ship M + kit2 + its factors: **standard** (nothing else), **hit-and-run** (recharge x4
after 0.4 of the delay, shield x0.8, damage x1.15, cone f0 0.06), **brawler** (vision x0.92,
damage x1.12), **small** (hit radius x0.8, damage x0.9, cone f0 0.10). 1024 games (10v10 512,
20v20 256), r9; A's score:

| A v B | 1v1 / 2v2 / 5v5 / 10v10 / 20v20 | AA; BB mutual, life, merges | Duel life | v r9 |
|---|---|---|---|---|
| standard v hit-and-run | 0.44 / 0.50 / 0.54 / 0.57 / 0.51 | 0.30, 319 s, 3.2; 0.37, 240 s, 2.9 | 268 s | 0.51 / 0.48 |
| standard v brawler | 0.52 / 0.52 / 0.52 / 0.53 / 0.52 | 0.34, 330 s, 3.3; 0.35, 270 s, 2.9 | 289 s | 0.51 / 0.50 |
| standard v brawler (repeat in the same run) | 0.54 / 0.52 / 0.53 / 0.52 / 0.46 | 0.33, 327 s; 0.37, 282 s | 274 s | 0.50 / 0.49 |
| standard v small | 0.54 / 0.50 / 0.47 / 0.48 / 0.47 | 0.33, 320 s; 0.42, 281 s, 3.3 | 276 s | 0.49 / 0.53 |
| hit-and-run v brawler | 0.59 / 0.53 / 0.50 / 0.49 / 0.49 | 0.37, 263 s; 0.36, 270 s | 258 s | 0.49 / 0.52 |
| hit-and-run v small | 0.61 / 0.51 / 0.47 / 0.45 / 0.41 | 0.35, 264 s; 0.43, 301 s | 242 s | 0.51 / 0.52 |
| brawler v small | 0.51 / 0.47 / 0.43 / 0.47 / 0.46 | 0.33, 303 s; 0.44, 294 s | 259 s | 0.50 / 0.51 |

- **A four-ship set within 0.39-0.61 on all 30 cells** (six pairs, five sizes), every mirror
  0.30-0.44 mutual at 240-330 s with about 3 merges a kill, every duel 240-290 s, strength
  even. The in-run repeat agrees to 0.02 at 1v1-10v10 (0.06 at 20v20, 256 games).
- The hit-and-run ship is a duellist (0.56-0.61 at 1v1) and fades in big crowds; the small
  ship is the reverse. Set 4b (running) eases the hit-and-run nose (f0 0.07) and trims the
  small ship's damage (x0.88).

## 03:26-03:34 KST: set 4b (`set4b/`): the hit-and-run nose eased, the small ship trimmed

Hit-and-run at damage x1.2 with its cone at f0 0.07; small at damage x0.88. 1024 games, r9:

| A v B | 1v1 / 2v2 / 5v5 / 10v10 / 20v20 | AA; BB mutual, life | Duel life |
|---|---|---|---|
| **standard v hit-and-run** | **0.48 / 0.51 / 0.50 / 0.49 / 0.51** | 0.34, 324 s; 0.51, 209 s | 245 s |
| standard v brawler | 0.54 / 0.53 / 0.52 / 0.48 / 0.55 | 0.32, 323 s; 0.35, 274 s | 282 s |
| standard v small | 0.56 / 0.52 / 0.47 / 0.51 / 0.53 | 0.32, 320 s; 0.42, 294 s | 276 s |
| hit-and-run v brawler | 0.58 / 0.54 / 0.54 / 0.55 / 0.48 | 0.46, 223 s; 0.35, 284 s | 241 s |
| hit-and-run v small | 0.63 / 0.50 / 0.48 / 0.48 / 0.38 | 0.46, 219 s; 0.42, 291 s | 228 s |
| brawler v small | 0.50 / 0.48 / 0.46 / 0.46 / 0.43 | 0.33, 297 s; 0.43, 293 s | 259 s |

- **Hit-and-run v standard at f0 0.07 is flat from 1v1 to 20v20 (0.48-0.51)**, the evenest pair
  of the night, at the price of a jousty hit-and-run mirror (0.51 mutual, 209 s).
- **The four ships are not transitive**: the hit-and-run ship beats the small ship in the
  duel (0.61-0.63) and loses to it at 20v20 (0.38-0.41), while both are even with the
  standard. Each pair has to be balanced on its own; the three-ship set {standard,
  brawler, small} (set 4's cards) is the tight one: 0.43-0.54 on every cell.

## 03:35-03:45 KST: roster 2 (`roster2/`): flight-distinct ships against the standard

512 games (10v10 256), r9, M + kit2 + the flight change (characteristics: lift x1.2 keeps the
sustained turn at 0.273 rad/s and lowers the corner speed from 134 to 122 m/s; drag x0.85 lifts
cruise 163 to 171 m/s and top 385 to 407 m/s):

| Ship v standard | 1v1 / 2v2 / 5v5 / 10v10 | Its mirror mutual, life | Duel life |
|---|---|---|---|
| rate (lift x1.2) | **0.71** / 0.50 / 0.45 / 0.43 | 0.43, 226 s | 214 s |
| rate (lift x1.2, induced x1.05) | 0.48 / 0.43 / 0.35 / 0.29 | 0.40, 223 s | 228 s |
| rate (lift x1.3, induced x1.1) | 0.26 / 0.40 / 0.27 / 0.17 | 0.43, 213 s | 213 s |
| **speed (drag x0.85)** | **0.53 / 0.50 / 0.55 / 0.57** | 0.34, 338 s | 299 s |
| speed (drag x0.85, induced x1.1) | 0.18 / 0.41 / 0.41 / 0.42 | 0.34, 298 s | 194 s |
| power (engine x1.15, induced x1.15) | 0.15 / 0.37 / 0.41 / 0.42 | 0.32, 309 s | 200 s |
| sustained (lift x0.85, induced x0.9) | 0.60 / 0.59 / 0.69 / 0.73 | 0.23, 552 s | 321 s |
| sustained (lift x0.85) | **0.23** / 0.46 / 0.59 / 0.61 | 0.21, 445 s | 234 s |

- **The lift coefficient trades the duel for the crowds** at sustained-turn parity: more
  lift (a lower corner speed, the slow lift-limited fight the maneuvers branch measured)
  wins duels (0.71) and loses crowds (0.43); less lift the reverse (0.23 against 0.61).
  This is the lever the flight-distinct pairs lacked: round 13 (running) pairs a rate
  fighter (lift up, damage up) with a speed fighter (drag x0.85).
- A low-drag ship at equal turn is close to even with the standard everywhere (0.50-0.57);
  any bleed bought for speed or power loses the duel (0.15-0.18).

## 03:36-03:45 KST: the set under the crowd extension (`set3ce/`)

1024 games, every ship flying CE (crowd extension + lead turn):

| A v B | 1v1 / 2v2 / 5v5 / 10v10 / 20v20 | AA; BB mutual, life | v r9 |
|---|---|---|---|
| **hit-and-run (set 4b) v standard** | **0.53 / 0.48 / 0.50 / 0.50 / 0.49** | 0.38, 243 s; 0.24, 370 s | 0.79 / 0.81 |
| standard v brawler | 0.53 / 0.57 / 0.58 / 0.61 / 0.59 | 0.21, 394 s; 0.26, 311 s | 0.81 / 0.82 |
| standard v small | 0.54 / 0.53 / 0.49 / 0.46 / 0.49 | 0.19, 380 s; 0.32, 349 s | 0.85 / 0.84 |
| brawler v small | 0.51 / 0.46 / 0.41 / 0.43 / 0.37 | 0.24, 350 s; 0.33, 365 s | 0.83 / 0.82 |

- **Hit-and-run v standard stays flat under the stronger doctrine** (0.48-0.53): with r9
  (set 4b) and CE it is within 0.48-0.53 at every size; confirm 3 had it even under the
  duel one-circle doctrine too (at f0 0.08). The brawler (short sight) loses ground when
  ships extend (0.37-0.43 against the small ship, 0.39-0.47 against the standard).

## 03:46-03:51 KST: round 13 (`round13/`): rate v speed, flight-distinct on one hull

512 games (10v10 256), r9, kit2 on both. Rate = lift up (lower corner speed, the same
sustained turn), damage up; speed = drag x0.85 (cruise 171, top 407 m/s):

| Pair | A v B 1v1 / 2v2 / 5v5 / 10v10 | AA (rate); BB (speed) mutual, life | Duel life |
|---|---|---|---|
| rate (lift x1.05) v speed | 0.55 / 0.50 / 0.40 / 0.42 | 0.31, 288 s; 0.37, 288 s | 269 s |
| rate (lift x1.05, dmg x1.1) | 0.61 / 0.55 / 0.48 / 0.39 | 0.41, 246 s; 0.38, 294 s | 254 s |
| **rate (lift x1.05, dmg x1.2)** | **0.63 / 0.56 / 0.52 / 0.50** | 0.46, 232 s; 0.36, 306 s | 235 s |
| rate (lift x1.1, dmg x1.1) | 0.65 / 0.51 / 0.49 / 0.46 | 0.46, 242 s; 0.35, 300 s | 229 s |
| rate (lift x1.1, dmg x1.2) | 0.68 / 0.55 / 0.53 / 0.45 | 0.49, 214 s; 0.37, 311 s | 209 s |
| rate (lift x1.15, dmg x1.2) | 0.73 / 0.55 / 0.50 / 0.43 | 0.50, 208 s; 0.35, 293 s | 201 s |

- **The first flight-distinct pair on one hull with the crowds even**: the rate fighter's
  lift buys the duel, its damage the crowds, and the speed fighter's low drag holds both.
  The duel still leans to the rate fighter (0.63); round 14 (running) gives the speed fighter
  sight or more speed, or the rate fighter less lift.

## 03:51-03:58 KST: round 14 (`round14/`): rate v speed, the duel lean

512 games (10v10 256), r9:

| Pair | A v B 1v1 / 2v2 / 5v5 / 10v10 | AA; BB mutual, life | Duel life |
|---|---|---|---|
| rate (lift x1.05, dmg x1.2) v speed (vision x1.05) | 0.59 / 0.51 / 0.46 / 0.41 | 0.47, 222 s; 0.39, 282 s | 238 s |
| rate (x1.05, x1.25) v speed (vision x1.08) | 0.58 / 0.48 / 0.51 / 0.47 | 0.47, 215 s; 0.47, 300 s | 219 s |
| rate (x1.03, x1.2) v speed | 0.59 / 0.54 / 0.50 / 0.50 | 0.44, 237 s; 0.35, 290 s | 243 s |
| rate (x1.05, x1.2) v speed (drag x0.82) | 0.60 / 0.58 / 0.52 / 0.51 | 0.47, 243 s; 0.37, 280 s | 244 s |
| **rate (x1.03, x1.25) v speed (vision x1.04)** | **0.58 / 0.51 / 0.49 / 0.52** | 0.46, 221 s; 0.40, 277 s | 235 s |
| rate (x1.05, x1.2) v speed (drag x0.8, vision x1.04) | 0.59 / 0.53 / 0.49 / 0.44 | 0.46, 232 s; 0.37, 289 s | 231 s |

- The crowds come even; the duel holds at 0.58-0.60 for the rate fighter under every
  sight and speed tried. Round 15 tries the joust (the duel's kills are 0.4 mutual).

## 03:51-03:59 KST: confirm 7 (`confirm7/`): hit-and-run v standard again

| Doctrine | Hit-and-run's score 1v1 / 2v2 / 5v5 / 10v10 / 20v20 | AA (hit-and-run); BB mutual, life, merges | v r9 |
|---|---|---|---|
| r9 (set 4b, first run) | 0.52 / 0.49 / 0.50 / 0.51 / 0.49 | 0.51, 209 s; 0.34, 324 s | 0.51 / 0.51 |
| r9 (second run) | 0.53 / 0.49 / 0.51 / 0.48 / 0.47 | 0.47, 221 s, 2.6; 0.36, 284 s, 3.1 | 0.50 / 0.48 |
| CE (set3ce) | 0.53 / 0.48 / 0.50 / 0.50 / 0.49 | 0.38, 243 s; 0.24, 370 s | 0.79 / 0.81 |
| DCL | 0.52 / 0.49 / 0.50 / 0.50 / 0.48 | 0.66, 140 s; 0.68, 185 s | 0.82 / 0.83 |

- **Hit-and-run v standard is within 0.47-0.53 in all 20 cells**: four runs, three doctrines,
  1v1 to 20v20. Not flagged PERFECT: the hit-and-run mirror is still a 0.47-0.51 joust under
  r9 (merges a kill 2.6, median 1), and the two ships fly alike (they differ in shield,
  recharge, damage and nose, not in flight).

## 03:59-04:03 KST: round 15 (`round15/`): rate v speed through the joust

512 games (10v10 256), r9. Rate (lift x1.03, dmg x1.25) v speed (drag x0.85, vision x1.04),
kit2 on both, with one ship's cone f0 changed (kit2's is 0.08):

| Nose | A v B 1v1 / 2v2 / 5v5 / 10v10 | AA (rate); BB (speed) mutual, life | Duel kills m / o / f / t; life |
|---|---|---|---|
| (round 14) | 0.58 / 0.51 / 0.49 / 0.52 | 0.46, 221 s; 0.40, 277 s | 0.40 / 0.54 / 0.06 / 0; 235 s |
| **speed f0 0.06** | **0.49 / 0.49 / 0.51 / 0.48** | 0.48, 233 s; 0.14, **485 s** | 0.33 / 0.60 / 0.07 / 0; 306 s |
| speed f0 0.05 | 0.46 / 0.48 / 0.50 / 0.46 | 0.49, 234 s; 0.11, 465 s | 303 s |
| rate f0 0.10 | 0.55 / 0.54 / 0.53 / 0.48 | 0.58, 180 s; 0.39, 293 s | 223 s |
| rate f0 0.12 | 0.47 / 0.50 / 0.47 / 0.45 | 0.63, 148 s; 0.43, 309 s | 199 s |
| speed f0 0.06, rate f0 0.10 | 0.45 / 0.50 / 0.51 / 0.44 | 0.60, 191 s; 0.13, 412 s | 237 s |

- **The nose is the duel lever this pair needed**: a harder nose on the speed fighter takes
  the rate fighter's duel edge away (0.58 to 0.49) and leaves the crowds (0.48-0.51). It
  slows the speed fighter's own mirror (485 s); round 16 looks between.

## 04:03-04:08 KST: round 16 (`round16/`): rate v speed, the noses between

512 games (10v10 256), r9; rate (lift x1.03, dmg x1.25) v speed (drag x0.85, vision x1.04):

| Noses | A v B 1v1 / 2v2 / 5v5 / 10v10 | AA (rate); BB (speed) mutual, life, merges | Duel life |
|---|---|---|---|
| **speed f0 0.07** | **0.53 / 0.48 / 0.53 / 0.49** | 0.47, 224 s, 2.7; 0.23, 343 s, 3.2 | 257 s |
| speed f0 0.07, rate f0 0.09 | 0.51 / 0.51 / 0.51 / 0.48 | 0.54, 197 s; 0.30, 399 s | 257 s |
| speed f0 0.065, rate f0 0.09 | 0.50 / 0.49 / 0.50 / 0.50 | 0.52, 205 s; 0.19, 424 s | 270 s |
| rate f0 0.11 | 0.52 / 0.50 / 0.47 / 0.45 | 0.62, 162 s; 0.38, 274 s | 213 s |
| speed f0 0.06, dmg x1.1; rate dmg x1.35 | 0.49 / 0.52 / 0.51 / 0.49 | 0.51, 207 s; 0.17, 430 s | 260 s |

- Several arms are within 0.45-0.53 at every size; the trade left is the two mirrors: the
  harder the speed fighter's nose, the slower its mirror. Confirm 9 (running) takes the
  speed nose at f0 0.07 to 1024 games, 20v20 and the crowd extension.
