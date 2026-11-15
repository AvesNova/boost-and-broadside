# League snapshot schedule from rl-v20's Elo curve (2026-11-15, proposal)

Data: the live-Elo pools of rl-v20 (game v20), exponential-average checkpoints (the averaged agent), anchored at Imugi = 1000: frontline and elimination at 17 steps, deathmatch scored win/loss at 5 steps (wide errors). Step 0 is bc-v20's last rated checkpoint, rated in its own pool. Each mode gets a monotone (isotonic) fit; the curve is their mean. Run `analyse.py`, then `rules.py` / `rules2.py` / `plot.py`, in this directory.

![schedule](league_schedule.png)

- 10M to 25M: Elo is about linear in steps, roughly 100 Elo per 5M.
- 25M to 90M: Elo is about linear in log(steps); ×1.17 in steps is about 100 Elo.
- After 95M the mean flattens: elimination is flat from 95M, and deathmatch has only two points there (95M and 150M).

**Proposed rule:** the first member at 5M, then next = previous + max(5M, 0.17 × previous). That gives 5, 10, 15, 20, 25, 30, 35, 41, 48, 56, 66, 77, 90, 105, 123, 144M, then 169, 197, 231, 270, 316, 370 and 433M (23 members to 500M). Elo gaps 10M to 90M: 101–124. Gaps 5M to 10M: 33. After 90M: 21–62.

## Simpler rules, multiples of 5 (`doubling.py`; the plot shows the first)

Every 5M up to a first stage, then the interval doubles each time the steps double: k members per doubling, a geometric rule of ratio 2^(1/k) snapped to multiples of 5.

| rule | members to 160M | Elo gaps 10–90M | members to 500M |
|---|---|---|---|
| every 5M to 50M, then 10, 20, 40, 80 (5 per doubling, 2^(1/5) ≈ 1.15) | 18 | 71–127, mean 101 | 26 |
| every 5M to 40M, then 10, 20, 40, 80 (4 per doubling, 2^(1/4) ≈ 1.19) | 16 | 85–154, mean 116 | 22 |
| every 5M to 30M, then 10, 20, 40 (3 per doubling) | 13 | 115–193 | 18 |
| next = previous + max(5M, 17%) (above) | 16 | 101–124 | 23 |

The golden ratio (1.618) would be about one member per 300 Elo here.
