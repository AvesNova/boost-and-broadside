# r9m2 on the play maps: progress

Branch `eval/r9m2-maps` (from `origin/tune/maneuvers-2` d253044e, merged with
`origin/integrate/next-runs` 102f3255; no conflicts). The task: how the best scripted doctrine
r9m2 (`maneuvers-2/champion2.json`) plays on the ship candidates the author playtests, against
r9 on the same maps. Hard stop 07:50 KST. GPU: the laptop RTX 4070 (alone), power profile
`performance`.

## Method

- `make.py`: each map's Yongma and Jujak card fields as absolute card values on the bench's
  base card (the real-scale rows; the map's unlisted fields are the ship files', as in the
  play shell), flown by r9 and by r9m2 (r9 with every doctrine field and scripted parameter of
  `champion2.json`; checked equal to `player_from(champion2.json)`).
- maneuvers-2's `sizes.py`, unchanged: self-play, 1024 games a cell (10v10 512), 300 s games,
  fog on, the density rule's arena. Lineups: A v B (ship A in the Yongma slot) at 1v1 / 2v2 /
  5v5 / 10v10, the A v A and B v B mirrors at 1v1. r9 and r9m2 run in the same compiled
  loops (one per size), so the r9 numbers here are a same-run baseline.
- maneuvers-2's `table.py` for the post-merge choice entropy.

Columns: A v B is ship A's score (the Yongma slot; win = more kills, draws half), SE about
±0.010 at 1v1, ±0.014 at 2v2 and 5v5, ±0.02 at 10v10 (the real run-to-run spread is about
±0.03-0.04, archetypes confirms 5-6). AA / BB: the 1v1 mirrors, mutual head-on kill share,
life (s), merges a kill mean / median / first-pass share / p90, H the post-merge choice
entropy in bits (same sense / opposite / extend / turn v extender; max 2). Pace: A v B life a
ship and kills a minute a ship at 1v1 / 5v5 / 10v10, trades at 5v5.

## 05:40-05:49 KST: run 1 (`../maneuvers-2/r9m2-maps/`): six maps x {r9, r9m2}, 1024 games

`table.md` there. The r9 rows reproduce the archetypes and tank-armour summaries within
±0.04 on every cell (gunner-close 10v10 0.51 here, 0.44 there).

## 05:50-05:54 KST: run 2 (`../maneuvers-2/r9m-maps/`): the same under r9m

r9m (maneuvers' champion: the duel one-circle and the crowd extension, no duel extension)
splits r9m2's change into what r9m brought and what the duel extension adds.

## Result (1024 games; 10v10 512)

| Map | A v B 1v1 / 2v2 / 5v5 / 10v10: r9 → r9m → r9m2 | AA mutual, life, H: r9 → r9m → r9m2 | BB mutual, life, H: r9 → r9m → r9m2 | AA merges m/med/first/p90 r9 → r9m2 | BB merges r9 → r9m2 | 5v5 life s, kills/min/ship, trades: r9 → r9m2 | 5v5 H r9 → r9m2 |
|---|---|---|---|---|---|---|---|
| hit-and-run v standard | 0.53/0.47/0.46/0.42 → 0.54/0.48/0.47/0.46 → 0.54/0.47/0.52/0.54 | 0.36, 266 s, 0.60 → 0.61, 148 s, 0.12 → 0.31, 169 s, 1.58 | 0.37, 294 s, 0.56 → 0.69, 167 s, 0.12 → 0.49, 181 s, 0.57 | 2.8/1/0.53/7 → 1.9/2/0.45/4 | 3.2/2/0.48/8 → 3.6/2/0.40/8 | 81, 0.66, 0.07 → 101, 0.54, 0.05 | 0.92 → 1.79 |
| rate v speed | 0.57/0.51/0.48/0.46 → 0.57/0.52/0.48/0.46 → 0.56/0.53/0.48/0.47 | 0.34, 287 s, 0.58 → 0.68, 166 s, 0.11 → 0.32, 176 s, 1.59 | 0.31, 333 s, 0.53 → 0.71, 186 s, 0.11 → 0.51, 193 s, 0.54 | 3.1/1/0.51/8 → 2.1/2/0.40/4 | 3.4/2/0.47/8 → 4.0/3/0.39/9 | 81, 0.66, 0.07 → 103, 0.53, 0.05 | 0.92 → 1.77 |
| standard v brawler | 0.53/0.51/0.52/0.51 → 0.55/0.56/0.58/0.60 → 0.53/0.59/0.63/0.69 | 0.34, 320 s, 0.54 → 0.69, 176 s, 0.12 → 0.35, 188 s, 1.59 | 0.36, 274 s, 0.58 → 0.58, 155 s, 0.14 → 0.43, 173 s, 0.66 | 3.2/2/0.47/8 → 2.3/2/0.33/4 | 2.8/2/0.50/7 → 3.1/2/0.43/7 | 82, 0.65, 0.07 → 103, 0.53, 0.05 | 0.92 → 1.79 |
| armed gunner v close | 0.54/0.58/0.53/0.51 → 0.55/0.60/0.68/0.65 → 0.53/0.61/0.69/0.70 | 0.18, 484 s, 0.51 → 0.73, 307 s, 0.07 → 0.32, 196 s, 1.59 | 0.22, 406 s, 0.51 → 0.56, 178 s, 0.12 → 0.40, 198 s, 0.64 | 3.2/2/0.41/7 → 2.4/2/0.28/4 | 2.9/2/0.49/7 → 3.5/2/0.40/8 | 85, 0.63, 0.07 → 109, 0.50, 0.04 | 0.92 → 1.73 |
| armour-tank (Yongma v Jujak) | 0.49/0.44/0.54/0.47 → 0.52/0.42/0.52/0.38 → 0.52/0.42/0.58/0.48 | 0.12, 363 s, 0.55 → 0.55, 152 s, 0.12 → 0.34, 214 s, 1.60 | 0.46, 289 s, 0.69 → 0.70, 248 s, 0.10 → 0.54, 249 s, 0.41 | 2.0/2/0.46/4 → 2.1/2/0.41/4 | 3.6/2/0.48/9 → 4.2/2/0.47/10 | 91, 0.60, 0.05 → 113, 0.49, 0.04 | 0.92 → 1.75 |
| balance-d2 (Yongma v Jujak) | 0.49/0.47/0.57/0.47 → 0.50/0.44/0.56/0.45 → 0.49/0.44/0.57/0.53 | 0.38, 286 s, 0.62 → 0.74, 134 s, 0.12 → 0.61, 189 s, 1.59 | 0.66, 261 s, 0.70 → 0.83, 209 s, 0.10 → 0.67, 223 s, 0.41 | 2.2/2/0.42/4 → 2.0/2/0.35/3 | 3.4/3/0.38/7 → 4.1/5/0.38/8 | 103, 0.53, 0.05 → 133, 0.42, 0.03 | 0.92 → 1.74 |

(1v1 A v B life under r9 → r9m2: 254 → 250, 268 → 267, 280 → 289, 322 → 314, 220 → 212,
289 → 295 s; unchanged. 10v10 life rises like 5v5's: 89-118 → 115-157 s.)

**What r9m2 does on these maps, against r9:**

1. **The mirrors: against r9 they joust as much or more, never clearly less.** The A
   (Yongma-slot) mirror holds r9's level on hit-and-run, rate v speed and standard v brawler
   (0.36 / 0.34 / 0.34 → 0.31 / 0.32 / 0.35) and rises on gunner v close (0.18 → 0.32),
   armour-tank (0.12 → 0.34) and D2 (0.38 → 0.61). The B (Jujak-slot) mirror rises on every
   map (0.37 → 0.49, 0.31 → 0.51, 0.36 → 0.43, 0.22 → 0.40, 0.46 → 0.54, 0.66 → 0.67).
   maneuvers-2's "r9m2 cuts the joust" is against r9m, whose one-circle near doubles it
   (r9m: 0.55-0.83 on every mirror here); r9m2 takes most of that back in the A mirror and
   only part in the B mirror, whose doctrine extends only below 0.4 of the target's shield
   and so flies r9m's one-circle most of the time.
2. **Mirrors are shorter and decided sooner**: life 266-484 s → 169-249 s (every mirror now
   within the 300 s line; the gunner mirror 484 → 196 s), AA p90 merges a kill 7-8 → 4. A
   tail-kill share appears (0.08-0.18 of mirror kills: the extender caught leaving).
3. **Choice entropy**: the A mirror 0.5-0.6 → 1.58-1.60 bits on every map; the B mirror
   stays 0.4-0.7 (and falls 0.69-0.70 → 0.41 on the two Yongma/Jujak cards); the 5v5 crowd
   0.92 → 1.73-1.79 (extension and turn-v-extender each ~0.38; it came with r9m's crowd
   extension, r9m 1.76-1.81).
4. **Pace**: crowds slow by about a quarter (5v5 life 81-103 → 101-133 s, kills a minute a
   ship 0.53-0.66 → 0.42-0.54; D2 0.42 at 5v5 and 0.36 at 10v10); trades fall
   (0.05-0.07 → 0.03-0.05). This is r9m's crowd extension (r9m: 97-123 s).
5. **Balance**: hit-and-run v standard, rate v speed, armour-tank and D2 stay inside
   0.42-0.58 at every size (hit-and-run's 10v10 0.42 → 0.54, D2's 0.47 → 0.53).
   **Standard v brawler and gunner v close tip to ship A in the crowds**: 2v2 / 5v5 / 10v10
   0.59 / 0.63 / 0.69 and 0.61 / 0.69 / 0.70 (r9: 0.51 / 0.52 / 0.51, 0.58 / 0.53 / 0.51).
   r9m already tips them (0.56-0.60, 0.60-0.68: the crowd extension, as archetypes' CE rows
   0.57-0.61 and 0.57-0.68 showed); r9m2's duel extension adds 0.03-0.09 on the brawler pair.

## 05:55-06:00 KST: gate diagnostic (`../maneuvers-2/r9m2-gates/`): one duel gate on both slots

On the two pairs that tip, r9m2 with one shield gate for the duel extension on both slots
(r9m2 has Yongma 1.0, Jujak 0.4); 1024 games:

| Map, gate | A v B 1v1 / 2v2 / 5v5 / 10v10 | AA mutual, life, H | BB mutual, life, H |
|---|---|---|---|
| standard v brawler, r9m2 (1.0 / 0.4) | 0.53 / 0.59 / 0.63 / 0.69 | 0.35, 188 s, 1.59 | 0.43, 173 s, 0.66 |
| standard v brawler, 0.4 both | 0.54 / 0.57 / 0.58 / 0.60 | 0.48, 182 s, 0.56 | 0.42, 170 s, 0.67 |
| standard v brawler, **1.0 both** | 0.56 / 0.53 / 0.56 / 0.60 | 0.35, 187 s, 1.59 | **0.32, 183 s, 1.60** |
| gunner v close, r9m2 (1.0 / 0.4) | 0.53 / 0.61 / 0.69 / 0.70 | 0.32, 196 s, 1.59 | 0.40, 198 s, 0.64 |
| gunner v close, 0.4 both | 0.54 / 0.63 / 0.66 / 0.69 | 0.48, 284 s, 0.39 | 0.41, 196 s, 0.65 |
| gunner v close, **1.0 both** | 0.54 / 0.61 / 0.63 / 0.64 | 0.33, 197 s, 1.59 | **0.29, 215 s, 1.60** |

- **The mirror's joust and entropy follow the gate, not the ship**: at 1.0 either mirror is
  0.29-0.35 mutual with 1.6 bits; at 0.4 either is 0.41-0.48 with 0.4-0.7 bits. On the
  one-hull pairs the slot (Yongma / Jujak doctrine) is arbitrary, so r9m2's split gate gives
  ship B the worse mirror. The 1.0 gate on both is r9's level in both mirrors (brawler
  0.34 / 0.36 under r9; the gunner pair a little above r9's 0.18 / 0.22) with the shorter life.
- **The crowd tilt stays**: 1.0 on both brings standard v brawler back to r9m's tilt (0.53-0.60)
  and gunner v close to 0.61-0.64; the rest is the crowd extension, which no duel gate fixes.

## State (06:00 KST)

Nothing stored: no doctrine, ship or map file changed. For the author:
1. On the four maps that hold balance under r9m2 (hit-and-run, rate v speed, armour-tank,
   D2) r9m2 adds decisions (A mirror and crowds ~1.6-1.8 bits) and shortens mirrors, but
   jousts at r9's level or more in the A mirror and more in the B mirror.
2. Standard v brawler and gunner v close lean to ship A in crowds under any doctrine with the
   crowd extension (0.56-0.70); a crowd-side counter or a card retune is needed if those
   ships are played with r9m-family scripted agents.
3. For the one-hull pairs a single duel gate (1.0) on both slots is the better r9m2 variant
   (both mirrors 0.29-0.35 mutual, 1.6 bits). It was not run on the other maps or against
   the league.
