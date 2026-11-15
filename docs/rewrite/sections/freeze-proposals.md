# Game freeze: proposals and expected numbers (2026-11-15)

**Locked in (author, 2026-11-15, later the same day):** deathmatch has a **10-minute** clock with 4·N^(4/3) tickets rounded to 5 at every team size (5, 10, 15, 25, 35, 85, 215 for N = 1, 2, 3, 4, 5, 10, 20). Elimination shrinks linearly over 4 minutes as below. Frontline is **±2** with a 10-minute clock, 3v3 at the smallest and the lock at 6 s/√km, with no centre zone and no healing for now. The proposals below are kept as they were made.

The author's current proposals for the three modes, with what we expect them to give. Nothing here is in the game yet; the values go in together, in one game-version bump, once the author locks them. Every number is Imugi against Imugi (explore search with the wide spacing unless noted), 5v5 unless noted. Measurements: [elimination-shrink/results.md](elimination-shrink/results.md), [mode-timing/results.md](mode-timing/results.md), [frontline-lock/results.md](frontline-lock/results.md).

## Deathmatch: an 8-minute clock, tickets 4·N^(4/3) rounded to the nearest 5

Tickets per team by ships per team N: 5, 10, 15, 25, 35, 85 and 215 for N = 1, 2, 3, 4, 5, 10 and 20 (today: 5 kills per ship, so 5N).

Read from the measured kill times (tickets off, 600 s games; 1v1, 2v2 and 10v10 with the earlier waypoint search, 5v5 with explore-wide):

| teams | tickets | games ended by tickets within 8 min | median game | tickets for 80% within 8 min |
|---|---|---|---|---|
| 1v1 | 5 | 30% | 8:00 (the clock) | 3 |
| 2v2 | 10 | 77% | 5:52 | 9 |
| 5v5 | 35 | 59% | 7:23 | 30 |
| 10v10 | 85 | 53% | 7:54 | 67 |

- The 4·N^(4/3) rule was fitted to 80% of games ending by tickets within **10** minutes. With 8 minutes about 55–60% end by tickets; the rest end on the clock and are decided by the kill lead.
- For 80% within 8 minutes the measured counts fit **3.2·N^(4/3)** (3, 8, 27, 69). Rounded to 5 that is 5, 10, 25, 70: right from 2v2 up, too many at 1v1 (it needs 3).
- **Options:** 3.2·N^(4/3) rounded to 5 from 3v3 up (exact below); 4·N^(4/3) with about 60% ending by tickets; or 4·N^(4/3) with a 10-minute clock.

## Elimination: a linear shrink from 0 over 4 minutes (clock 248 s)

Not measured directly. Estimated from explore-wide without a shrink (48% of games over by 2:00, 64% by 2:40, 70% by 3:00, 79% by 4:00) and from how the 120 s shrink compressed games. The boundary reaches 2.5 km, below which it starts to matter, at about 2:43, and 2 km at 3:00.

| | expected |
|---|---|
| median game | about 2:05 (hardly moved by the shrink) |
| p80 | about 3:10–3:20 |
| p90 | about 3:30–3:40 |
| games on at the clock | none: the boundary closes 8 s before it |
| deaths by the boundary | about 8–12% |
| last deaths by the boundary | about 15–20% |

1v1 stays mostly decided by the boundary (about half of its games with a 2-minute shrink, likely more with 4). A measured run is about 10 minutes of CPU.

## Frontline: a neutral centre zone first, and full health and energy on capture (to try on another machine)

**The ideas (the author, 2026-11-15):**
1. **A neutral centre zone.** The match starts with one neutral zone in the middle. Capturing it moves the front to ±1 for the capturing team, and the game then continues as today's Frontline from there. The win is still a net 3 zones, the centre counting as one.
2. **Full health and energy on capture.** The capturing team's ships are restored when they take a zone.
3. **Longer matches** are acceptable.

**Expectations (Claude's, before any test):**
- **The centre start barely changes the game's length.** Today's first capture already moves the front to ±1. With the front a random walk (lock 6 s/√km, P(reversal) 0.49), the expected captures to ±3 from ±1 are (3 − 1)(3 + 1) = 8, so 1 + 8 = 9, the same as from 0 (3 × 3). The centre's winner wins 2/3 of games, as today's first capturer does. What it changes is the opening: both teams have one shared objective at equal distance, so contact is quick and symmetric. That is an objective, not an agreed meeting point. It probably helps small teams, whose captures are rare because they meet late.
- **Healing on capture adds momentum.** The team that won the fight is restored for the next push, as the long lock did (19 s/√km gave P(reversal) 0.29, 14 s/√km 0.38). Assumed P(reversal) 0.33–0.42. That shortens games but makes the front persistent and comebacks rarer, and the first capturer wins more than 2/3.
- **Open points:**
  - the lock after the centre falls;
  - each team's spawn before and after it;
  - the observation's zone rows through the change of phase;
  - a new phase for Imugi's Frontline layer;
  - a possible standoff at an evenly held centre.

**Modelled shares of games ending by capture.** The model is a correlated random walk with a per-game capture speed (exponential intervals, mean 120 s, log-normal spread σ 0.5 across games), fitted to the measured ±3 (47% within 10 min, 53% within 12, median 10.7 min). It under-predicts ±2 a little (80% within 12 min against 84% measured). The centre start is modelled as a first capture 20% sooner; healing only as a lower P(reversal).

| variant | 10 min | 12 min | 15 min | 20 min | 25 min | median |
|---|---|---|---|---|---|---|
| ±3 today | 45% | 52% | 61% | 72% | 79% | 11.4 min |
| ±3, centre start | 46% | 53% | 62% | 73% | 80% | 11.0 |
| ±3, heal (P(reversal) 0.38) | 54% | 62% | 71% | 81% | 87% | 9.0 |
| ±3, centre + heal (0.38) | 56% | 63% | 72% | 81% | 88% | 8.6 |
| ±3, centre + heal (0.33) | 61% | 68% | 76% | 85% | 90% | 7.7 |
| ±2 today (reference; measured 79% / 84%) | 74% | 80% | 87% | 92% | 95% | 5.0 |

- At ±3 even centre + heal needs about a 20-minute match for 80% of games to end by capture; ±2 gets there in about 12 minutes.
- **To measure on the other machine:**
  - P(reversal) with healing on;
  - the share of games the first capturer wins;
  - the time to the centre's capture;
  - the share of games ending by capture at 12, 15 and 20 minutes.

  These say whether the snowball is acceptable.

**The other machine:** this needs the game change, a phase for Imugi, and runs of `bench.mode_dynamics frontline` and `bench.front_reversal`. Nothing for it is in the code yet.
