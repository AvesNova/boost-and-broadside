# Clock: scripted wipe times (2026-11-25)

BFM-CMA2 (the default) against itself and against the 8 screening panel members (panel.json,
by row name onto today's rows), elimination with the clock at 30,000 ticks, standard spawn,
fog and fields at the defaults. Store: data/scripted-screening/clock-2026-11-25.sqlite; script
and numbers: sections/clock-measurement/ (clock.py, report.json). Pooled games; ticks at 30 Hz.

| condition | games | p50 | p80 | p90 | p95 | at the cap | before 1,800 |
|---|---|---|---|---|---|---|---|
| 4v4, 4,200 m (main) | 2,048 | 1,058 (35.2 s) | 1,621 (54.0 s) | 2,107 (70.2 s) | 2,784 (92.8 s) | 0.05% | 84.8% |
| 3v3, 4,200 m | 512 | 980 (32.7 s) | 1,608 (53.6 s) | 2,191 (73.0 s) | 3,661 (122.0 s) | 0.2% | 82.8% |
| 5v5, 4,200 m | 512 | 1,170 (39.0 s) | 1,722 (57.4 s) | 2,036 (67.9 s) | 2,678 (89.3 s) | 0.2% | 83.6% |
| 4v4, 3,000 m | 512 | 938 (31.2 s) | 1,502 (50.1 s) | 2,004 (66.8 s) | 2,710 (90.3 s) | 0% | 86.5% |
| 4v4, 6,000 m | 512 | 1,182 (39.4 s) | 1,678 (55.9 s) | 2,090 (69.7 s) | 2,670 (89.0 s) | 0.6% | 83.8% |

Main condition split: the mirror (1,024 games) p80 1,622 ticks, the panel (1,024) p80 1,618.
The p80's bootstrap 95% interval in the main condition is 1,557 to 1,679 ticks. Measured on
core 48795d87 (after C5's shared tick and the scripted trim); an earlier run on 257ffd10 gave a
main p80 of 1,584 ticks (52.8 s), the same clock.

**Chosen clock: 1,800 ticks (60 s)**, the main p80 (54.0 s) rounded up to a multiple of 10 s.
85% of main-condition games end by a wipe before it; under the old 5,400 ticks, 98.1% did.

Spread: the p80 runs from 50.1 s (3,000 m) to 55.9 s (6,000 m), and 53.6 s (3v3) to 57.4 s
(5v5). Every condition's p80 is under 60 s, and 83 to 87% of games end before it in each. The
tail is long: 5% of 4v4 games last past 93 s, so about 15% of games at the default will be
decided by the clock rule rather than a wipe.

For comparison, the screening's skill games (18 conditions, 5,400-tick clock, older flight and
agents) averaged 1,252 ticks for BFM-CMA2; the main mean here is 1,459 ticks, median 1,058.
