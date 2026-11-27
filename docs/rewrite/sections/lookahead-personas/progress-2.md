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

### 23:30 screen done (fresh games, SE 0.02, BT over all pairings incl. head-to-head)
- Elo against the old yardstick (1v1 / 2v2 / 4v4 / 8v8 / 16v16): plain 845 / 967 / 1276 /
  1666 / 2078; crowd-own 989 / 1131 / 1310 / 1643 / 2019; crowd-known 1084 / 1146 / 1299 /
  1617 / 2020; crowd-near 1139 / 1122 / 1262 / 1571 / 1975; crowd-own-half 1179 / 1113 / 1191
  / 1403 / 1712; plain-half 977 / 1035 / 1207 / 1400 / 1671.
- Worked: every crowd candidate beats plain at 1v1 and 2v2 by 120-290 Elo and keeps 4v4-16v16
  within 60 Elo of it. Mean over 1v1/4v4/16v16: known 1468, near 1459, own 1439, plain 1400.
- Didn't: the old yardstick saturates at 8v8/16v16 (scores of 95-99.7%); Elo there rests on
  the head-to-head games.
- 23:30 decision: one shared measure, **known enemies** (crowd_measure 1, crowd_half 1): best
  mean over the three study sizes and over all five.
- 23:30 decision: CMA rated against crowd-known (at 1000 in the search), not the old yardstick;
  space: 19 rows (boundary, seek dropped) + 5 crowd ratios (angles, aggression, speed,
  preferred speed, reply temperature) + store_crowd; popsize 12, 192 games per size, sigma 0.15.
- Next: CMA, bound 60 generations or 05:30; first generation ETA 23:50.
- 23:40 CMA running, about 3 minutes per generation; generations 1-3 mean Elo against
  crowd-known 982 / 1028 / 1058. ETA 60 generations 02:40 (worst 05:30, the loop's bound).
- 00:17 CMA generations 1-16 against crowd-known: population mean 982 -> 1176 (1v1 1087, 4v4
  1135, 16v16 1308 at generation 16); sigma 0.138 -> 0.172.
- 00:17 decision: re-reference the search to its generation-16 mean (candidates/cma-mean-g16.json):
  at 16v16 the population scored about 85-90% against crowd-known, where ranking resolution
  falls. Journal lines from generation 17 carry the reference id. Loop bound unchanged
  (60 generations or 05:30); ETA 02:40 (worst 05:30).

### 00:45 redirect: CMA stopped
- 00:42 CMA stopped after 24 generations on the author's redirect (coordinator 00:35: generation
  1 was already good enough; over-optimising risks a solution brittle to the yardstick and the
  conditions). Journal kept (cma/cma.jsonl). No re-centred yardstick from it.
- 00:45 decision for the redirected study: centre = crowd-known (the screen's configuration),
  spread-1 boxes around it. Skill: 48 personas + every sampled row at u = -0.5 and 1.5 (half a
  box past each edge) against crowd-known at 1v1/4v4/16v16, SE 0.03. Behaviour: a state set
  per size around crowd-known, Saltelli (base 32) JSD to the centre, label blur at 0.25/0.5/1.
  The old-yardstick data of step 2 (106 agents, three sizes) is reused for the interaction fit.

### 01:08 revised brief (coordinator 00:55): one skill surface, robust centre, ladder test
- Pool (round 2 only, one game version, study clocks; round 1 left out: other conditions and
  version, and its personas are round 2's population): 492 agents, 1,398 pairings, 559,104
  games before the around games finish.
- Surface: Bradley-Terry on outcomes, f quadratic (linear, square, all 210 products) in the
  per-size effective unit coordinates (crowd rows folded in by the size's mean crowd weight),
  ridge. 5-fold CV over agents (references kept in training): held-out deviance explains 84.0%
  of the coin-to-saturated gap per size (lambda 100), 82.0% with a shared f + per-size linear
  and square terms, 75.9% additive (no products). Chosen: per size, with products. No GP needed.
- Decisions: around-crowd-known 16v16 at SE 0.04 (loose Elo is fine); the screening Saltelli /
  blur around crowd-known dropped in favour of the same around the robust centre.
- Next: around 16v16 (ETA 01:15), active rounds 1-2 (ETA 01:40), ladder + round robin (ETA
  02:15, worst 02:45), behaviour around the centre (ETA 02:40, worst 03:15), write-up.
- 01:50 active rounds 1 and 2 done (24 points x 2 references x 3 sizes x 128 games). The r0
  surface predicted round 1's scores with r 0.81 / 0.95 / 0.93 (1v1 / 4v4 / 16v16), RMSE
  0.08 / 0.07 / 0.12 against a binomial SE of 0.044. Final fit (517 agents, 627,456 games):
  82.7% of the coin-to-saturated deviance held out, per size, lambda 100.
- 01:50 decision: robust centre = the centroid of a coordinate-wise Metropolis sample of the
  region within 50 Elo (mean over sizes) of the peak, inside the data's 2-98% support per size.
  Seven rows sit at their support edge there (threat, collision, horizon, temperature, speed
  weight, power temperature, fire tolerance): the surface would go further than the data.
- 01:50 round robin 1v1 done (6 agents, 1,024 games per pair): no cycles at 2 SE.
- Next: ladder (16 agents, SE 0.05, three sizes) ETA 02:20 (worst 02:50), behaviour around the
  centre ETA 02:45 (worst 03:15), write-up ETA 03:15.
- 02:10 ladder done (16,384 games per size), behaviour around the centre done (states,
  Saltelli base 32, blur, league JS). GPU for the revised brief: about 01:20-02:09 (active
  rounds, ladder, round robin, behaviour), within the 2 h budget.
- 02:15 decision: league set = league1, 2, 3, 5, 6 (league4 dropped: closest to the centre and
  to league5 in JS). results-2.md written (summary at the top). Final checks: quick tier lean/scripted + lean/eval + lean/game 179 passed, 1 skipped; ruff clean on lean and this folder.
