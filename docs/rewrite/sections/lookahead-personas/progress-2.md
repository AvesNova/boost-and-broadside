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
