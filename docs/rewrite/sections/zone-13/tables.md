### yardstick (mirror), fields and fog on: item 12 (60 s clock) vs the zone

| measure | 1v1 item 12 | 1v1 zone | 4v4 item 12 | 4v4 zone |
|---|---|---|---|---|
| games | 4096 | 2048 | 2048 | 1024 |
| game length mean, s | 51.2 | 62.8 | 50.8 | 54.2 |
| game length median, s | - | 59.8 | - | 52.3 |
| game length p90, s | - | 111.3 | - | 76.2 |
| game length max, s (bound 131.1) | - | 121.6 | - | 118.4 |
| games ending by elimination | 37.0% | 100.0% | 63.5% | 100.0% |
| one team wiped (a win) | - | 99.8% | - | 99.9% |
| both wiped (a draw) | - | 0.20% | - | 0.10% |
| kills per game-minute | 0.43 | 0.96 | 5.68 | 6.14 |
| deaths by bullet | 99.1% | 84.2% | 96.9% | 92.9% |
| deaths by collision | 0.7% | 0.8% | 3.1% | 3.5% |
| deaths by boundary | - | 15.0% | - | 3.6% |
| tail-chase share of bullet hits | 75.7% | 75.9% | 53.4% | 54.2% |
| mean hit range, m | 121 | 116 | 157 | 152 |
| bullet hits per ship-minute | 3.94 | 7.06 | 15.77 | 16.38 |
| shots per ship-minute | 13 | 23 | 59 | 61 |

Map-edge target (living ships every 15th tick, m): quantiles 0 / 0.1% / 1% / 50% / 99% / 99.9% / 1
- 1v1: -4,176 / -3,753 / -3,256 / -829 / -12 / 217 / 555
- 4v4: -4,176 / -3,836 / -3,422 / -1,417 / -138 / 127 / 361

Notes: item 12's lengths and kill rates are of games cut at the 60 s clock (its 37.0% / 63.5%
are the shares ended by a wipe before the clock), so its means are bounded by 60 s and not
like for like. No zone game reached full depth (128.6 s): the longest ended at 121.6 s, when
the radius was -56 m (every ship already on the ramp). The map-edge head's range (-4,200 to
300 m) still holds 99.9% of the scripted agent's rows: unchanged.
