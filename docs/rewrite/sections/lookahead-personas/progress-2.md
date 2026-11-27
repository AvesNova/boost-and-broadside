# Look-ahead personas, round 2: progress

Author 2026-11-27 ~20:15: the persona test as it will be evaluated (1v1, 4v4, 16v16, always
fields and fog; longer clocks allowed). Branch personas-2 from core 2770a376, laptop RTX 4070,
performance profile. Data: ~/avesnova/boost-and-broadside/data/lookahead-personas-2/.

## Best-judgement calls (one line each)

- 20:25 Step 1 pairings: yardstick mirror + 6 round-1 personas (persona14, 21, 27, 00, 53, 44:
  best/worst of round 1 at 1v1 and 4v4, one near the median); caps 18,000 ticks (1v1) and 9,000
  (4v4, 16v16); a game shorter than the cap ended by a wipe.

## Milestones
- 20:40 Study clocks (p90 of wipe times rounded up to 10 s): 1v1 400 s (12,000 ticks), 4v4 90 s
  (2,700), 16v16 80 s (2,400); 2v2 and 8v8 provisional 90 s / 80 s until measured.
- 20:40 1v1 cannot reach 90% for the mirror at any practical clock (86.5% by 450 s, 88.5% by
  600 s, 11.5% still running at 600 s); 400 s is the pooled p90 (91% by 450 s), kept for the study.
- 20:40 Crowd rows added as FIXED rows (SAMPLED, `personas`, box_point unchanged; `vector` and
  `PARAMETERS` longer by 12): 10 CROWDED rows x `_crowd` ratio, `crowd_measure`, `crowd_half`.
  Added before step 2 so every round-2 game shares one game version (they are off by default).

### 20:40 step 1 done
- Tried: long caps (1v1 18,000 ticks, 4v4/16v16 9,000), yardstick mirror + 6 personas,
  fields + fog; 7,168 / 3,584 / 2,688 games.
- Measured: wiped before 60 s (today's clock): 1v1 39.7%, 4v4 72.3%, 16v16 67.7%. p50 / p90:
  1v1 75 / 395 s (6.5% at the 600 s cap), 4v4 49 / 81 s, 16v16 53 / 76 s. Draw wipes 1v1 5.1%.
- Didn't: 1v1 mirror reach 90% (see above).
- Next: step 2 skill at the study clocks (106 agents x 3 sizes, target SE 0.015); ETA 22:00
  (worst 23:00).

### 21:31 step 2 at 1v1 and 4v4 done; 2v2, 8v8 clocks
- Measured: 2v2 p50 / p90 60 / 131 s (wiped before 60 s 50.1%), 8v8 48 / 71 s (79.0%).
  Study clocks 2v2 140 s (4,200 ticks), 8v8 80 s (2,400).
- Skill (131,072 games per size, every pairing SE <= 0.015): population sd 71 Elo (1v1),
  57 (4v4); 1v1 against 4v4 r = +0.08. Sign flips 1v1 -> 4v4 (z > 6): angles weight +138 ->
  -39, aggression +86 -> -55, energy weight -37 -> +62; large shifts: speed weight +17 ->
  +101, horizon +12 -> -69, fire tolerance 0 -> +71, reply temperature -67 -> -2.
- Re-check of inert rows: boundary and seek inert at both sizes; disengaged energy weight is
  not inert in 1v1 with fog (-43 +- 11 per box), 0 in 4v4.
- Next: 16v16 skill (ETA 22:10, worst 22:40), crowd measures job.
- 22:08 16v16 skill target SE 0.02 (about 14 Elo) instead of 0.015: 16v16 plays about 22
  games/s here (16,384 games in 812 s with compile), so 0.015 (about 131k games) would take
  about 100 minutes; 0.02 about 74k games. Pure persona teams kept (no mixed lineups).
- 22:08 Coordinator: no hard 08:30 bound; finish when adequately done, no rabbit holes.

### 22:08 step 2 16v16 running
- Next: 16v16 to SE 0.02, ETA 23:00 (worst 23:30); the crowd-measure job is on the GPU now.

### 23:10 step 2 done; step 3 crowd measures; screen launched
- Measured (106 agents against the yardstick; 1v1, 4v4 131,072 games, SE <= 0.015; 16v16
  65,536 games, SE <= 0.02): population sd without error 71 / 57 / 155 Elo (1v1 / 4v4 /
  16v16); range 801-1090 / 812-1103 / 497-1350. Personas' Elo r: 1v1-4v4 +0.08, 1v1-16v16
  -0.27, 4v4-16v16 +0.90. Linear fit R^2 0.85 / 0.85 / 0.88 (leave-one-out 0.72 / 0.75 / 0.79).
- Size-dependent (linear slope 1v1 -> 16v16 per box, z > 4): angles weight +138 -> -231,
  aggression +86 -> -182, energy weight -37 -> +193, speed weight +17 -> +179, horizon +12 ->
  -205, fire tolerance 0 -> +200, reply temperature -67 -> +87, preferred speed -39 -> +86,
  store weight -27 -> -98, threat weight +9 -> -66. 4v4 sits about 40-50% of the way.
  Not size-dependent: temperature (colder wins at all sizes: -104 / -50 / -86).
- Crowd measures (yardstick mirror; mean weight at crowd_half 1, 1v1/2v2/4v4/8v8/16v16): near
  0 / .08 / .27 / .43 / .53; known 0 / .12 / .46 / .76 / .87; own 0 / .43 / .68 / .82 / .90.
- Surrogate (additive quadratic fits; magnitudes extrapolated, not trusted): crowd candidates
  beat the plain best point on the mean, own and known at half 0.5-1 the highest.
- 23:10 decision: screen 6 candidates (plain, plain-half, crowd-own/known/near at half 1,
  crowd-own-half) with fresh games at 1, 2, 4, 8, 16 per side, SE 0.02, before any CMA;
  store_crowd kept at 1 (surrogate gain 6 Elo, under the 8 Elo bar).
- Next: screen ETA 00:00 (worst 00:30; the GPU is shared with the league agent).
