# Final freeze check: results (2026-11-15)

Branch `check/freeze-final`, **game version 22**. The mode conditions were measured on
6c75ff8f (the game as on 0b675320; later commits change no game code); the head-on passes on
06544dc7 (that tree with `feat/imugi-boundary`'s split critic merged in, which changes
training only; the game version test passes at 22). Laptop GPU, power profile
`performance`. The author's review is in [review.md](review.md).

## What changed on this branch

- **vizia dropped** from the real-scale set (`order.toml`, `vizia.toml`). Yongma's and
  Jujak's files hold absolute handles, so neither depends on it; no game number changed.
- **Imugi's doctrine lookup** chose its stored doctrines by the card's type count alone.
  With vizia gone the real-scale set has two types, like the standard set, so a standard
  card would have flown Yongma's and Jujak's doctrines. It now matches the type count and
  the scale (the card's gravity): `imugi_doctrine.for_card`, `rows_by_kind`,
  `ImugiStep.table`. Real-scale Imugi flies the same doctrines as before.
- **The shield fix (game v22).** `combat.apply_damage` restarted the recharge wait only on
  damage above zero. With a head-on fraction of 0 and a sharpness of 3, a bullet within
  about 0.2 degrees of the nose deals exactly 0.0 in float32, so such a hit let the shield
  recharge that tick. A hit now restarts the wait whatever its damage
  (`test_shield_recharge.py`). The reference's 48 random ticks had no such hit, so its
  numbers are unchanged.
- **A bench fix (`bench.mode_dynamics`, eeb73d99).** The trace's score misses the death
  that ends an elimination game (it lands on the reset tick), so every 1v1 wipe was
  recorded as a draw and some larger games too. Elimination's result now comes from the
  survivors. The records here were made before the fix; `freeze_report.py` recomputes the
  result the same way. Only the energy-against-turn elimination scores were affected (the
  partial tables in 0b675320 had 1v1 50.0%, 2v2 27.7%, 5v5 27.0%, 10v10 33.0%); every recomputed
  score still meets its SE target of 2 points.

## Setup

Imugi on both sides (the stored doctrines and each mode's parameter set), fog on, fields
on, each preset's own map (deathmatch 4200 m, elimination and Frontline 7800 m at every
team size), the default search (explore), T = 1, seed 22. A game is an environment's first
game. Each condition runs batches of 256 games until its SE targets are met, at most 2048:
deathmatch, ended by tickets SE ≤ 3 points; elimination, the three shares SE ≤ 3 points;
Frontline, the three shares SE ≤ 3 points; energy against turn, the Yongma team's score SE ≤
2 points. "Mixed" lineups draw each slot Yongma or Jujak with equal chance, both teams
flying the same lineup; "energy against turn" puts a pure Yongma team against a pure Jujak
team, both seat orders.

Deathmatch is recorded with tickets off and replayed with the frozen tickets 4·N^(4/3) to the
nearest 5 and the 600 s clock; the definitions are in `freeze_report.py`'s docstring. SEs
are binomial for shares over games, sd/√n for means, a game bootstrap for medians and
quantiles, and clustered by game for pooled ratios. Kill-aspect SEs treat kills as
independent, so they are a little optimistic.

## Deathmatch (mixed lineups)

| ships | games | tickets | ended by tickets (%) | median | p80 | kills/min |
|---|---|---|---|---|---|---|
| 1v1 | 256 | 5 | 0.4 ± 0.4 | 10:00 ± 0 s | 10:00 ± 0 s | 0.16 ± 0.01 |
| 2v2 | 512 | 10 | 43.4 ± 2.2 | 10:00 ± 0 s | 10:00 ± 0 s | 1.59 ± 0.03 |
| 5v5 | 256 | 35 | 80.5 ± 2.5 | 7:42 ± 7 s | 9:56 ± 12 s | 8.41 ± 0.13 |
| 10v10 | 256 | 85 | 83.6 ± 2.3 | 7:43 ± 7 s | 9:48 ± 18 s | 20.71 ± 0.26 |

Contact in the small games (first enemy hit after a spawn, from the bench's per-life
histogram; lives still running at the clock are not in it):

| ships | first hit of the game, median | games with no hit (%) | spawn to first hit: q25 / median / q80 | life: q25 / median / q80 | deaths a game | first deaths by the boundary (%) |
|---|---|---|---|---|---|---|
| 1v1 | 40 s | 17.6 | 17 / 39 / 119 s | 68 / 154 / 324 s | 1.6 | 31 |
| 2v2 | 17 s | 0 | 15 / 30 / 63 s | 48 / 89 / 173 s | 15.5 | 8 |
| 5v5 | 4 s | 0 | 11 / 15 / 26 s | 31 / 49 / 80 s | 83.5 | 5 |

At 1v1 the ships find each other at the start in most games (a hit in 82%, median 40 s),
but the duel rarely ends in a kill: 1.6 deaths in 10 minutes, about a third of them by the
boundary. Since mixed lineups mirror the teams, every 1v1 here is a same-type duel.

## Elimination (mixed lineups)

| ships | games | median | p80 | p90 | on at the clock (%) | deaths by the boundary (%) | last death by the boundary (%) |
|---|---|---|---|---|---|---|---|
| 1v1 | 768 | 2:33 ± 2 s | 3:11 ± 2 s | 3:31 ± 3 s | 0.0 ± 0.0 | 46.7 ± 1.8 | 46.5 ± 1.8 |
| 2v2 | 256 | 1:55 ± 4 s | 2:45 ± 5 s | 3:00 ± 4 s | 0.0 ± 0.0 | 20.3 ± 1.9 | 25.0 ± 2.7 |
| 5v5 | 256 | 1:35 ± 3 s | 2:18 ± 5 s | 2:38 ± 4 s | 0.0 ± 0.0 | 9.9 ± 1.1 | 19.1 ± 2.5 |
| 10v10 | 256 | 1:26 ± 3 s | 2:03 ± 6 s | 2:34 ± 9 s | 0.0 ± 0.0 | 4.3 ± 0.6 | 10.2 ± 1.9 |

## Frontline (mixed lineups)

| ships | games | ended by capture (%) | median | P(reversal) | first capturer wins (%) | moves a game |
|---|---|---|---|---|---|---|
| 3v3 | 256 | 64.8 ± 3.0 | 6:44 ± 23 s | 0.402 ± 0.020 | 79.7 ± 2.7 | 2.84 ± 0.09 |
| 5v5 | 256 | 78.5 ± 2.6 | 5:07 ± 35 s | 0.467 ± 0.019 | 78.4 ± 2.7 | 3.31 ± 0.11 |
| 10v10 | 256 | 75.4 ± 2.7 | 4:37 ± 30 s | 0.520 ± 0.020 | 80.1 ± 2.7 | 3.55 ± 0.14 |

A random walk has P(reversal) 0.5. 3v3 is persistent (5 SE below), 5v5 is 1.7 SE below,
10v10 is 1 SE above. The first capturer wins about 80% at every size; with a ±2 front, a
first capture that is never reversed ends the game, so this is close to what a random walk
gives (from one step ahead, 3/4) plus the persistence.

## Yongma team against Jujak team

| condition | games | Yongma team's score (%) |
|---|---|---|
| deathmatch 1v1 | 256 | 21.7 ± 1.9 |
| deathmatch 2v2 | 256 | 7.8 ± 1.6 |
| deathmatch 5v5 | 512 | 21.8 ± 1.8 |
| deathmatch 10v10 | 768 | 48.2 ± 1.8 |
| elimination 1v1 | 512 | 10.0 ± 1.3 |
| elimination 2v2 | 512 | 18.6 ± 1.7 |
| elimination 5v5 | 512 | 21.9 ± 1.8 |
| elimination 10v10 | 512 | 29.9 ± 2.0 |
| Frontline 3v3 | 256 | 11.9 ± 1.8 |
| Frontline 5v5 | 256 | 13.1 ± 1.9 |
| Frontline 10v10 | 512 | 18.2 ± 1.6 |

Kills by killer and victim type in these games (pooled, friendly fire is on):

| condition | Yongma on Jujak | Jujak on Yongma | Yongma on Yongma | Jujak on Jujak |
|---|---|---|---|---|
| deathmatch 2v2 | 1,360 | 2,771 | 29 | 181 |
| deathmatch 5v5 | 17,252 | 22,705 | 781 | 3,211 |
| deathmatch 10v10 | 66,388 | 77,138 | 4,320 | 14,318 |
| elimination 5v5 | 1,017 | 2,064 | 40 | 306 |
| Frontline 5v5 | 2,951 | 4,792 | 143 | 523 |

Jujak kills more at every size. Its share of its own team's kills is larger too: in
deathmatch 10v10, 16% of the Jujak team's kills are of its own ships (6% for Yongma), and
each gives the Yongma team a point. Without them the Jujak team leads on kills (77,138
against 66,388); with them the points are close to level (about 80,700 against 81,500,
boundary deaths aside), which is where the Yongma team's 48% at 10v10 comes from.
The game-loop freeze (2026-11-12, round 12) had 5v5 at 53.8 ± 1.0% and 2v2 at 28%, on an
earlier card with 300 s games; the cards, flight model and maps have changed since.

## Kills by aspect (mixed lineups)

| condition | kills | mutual head-on (%) | offset head-on (%) | flank (%) | tail (%) | Y on Y / Y on J / J on Y / J on J |
|---|---|---|---|---|---|---|
| dm-1v1 | 269 | 1.9 ± 0.8 | 8.2 ± 1.7 | 43.1 ± 3.0 | 46.8 ± 3.0 | 127 / 0 / 0 / 142 |
| dm-2v2 | 7282 | 3.1 ± 0.2 | 9.9 ± 0.3 | 45.5 ± 0.6 | 41.5 ± 0.6 | 2264 / 639 / 1374 / 3005 |
| dm-5v5 | 20665 | 3.3 ± 0.1 | 12.5 ± 0.2 | 51.4 ± 0.3 | 32.7 ± 0.3 | 4582 / 3728 / 5130 / 7225 |
| dm-10v10 | 51374 | 3.0 ± 0.1 | 13.2 ± 0.1 | 52.0 ± 0.2 | 31.7 ± 0.2 | 10781 / 10723 / 13360 / 16510 |
| elim-1v1 | 411 | 1.5 ± 0.6 | 8.8 ± 1.4 | 40.1 ± 2.4 | 49.6 ± 2.5 | 207 / 0 / 0 / 204 |
| elim-2v2 | 491 | 2.2 ± 0.7 | 7.9 ± 1.2 | 45.6 ± 2.2 | 44.2 ± 2.2 | 158 / 66 / 85 / 182 |
| elim-5v5 | 1610 | 4.3 ± 0.5 | 11.2 ± 0.8 | 48.1 ± 1.2 | 36.3 ± 1.2 | 358 / 270 / 423 / 559 |
| elim-10v10 | 3816 | 3.6 ± 0.3 | 12.4 ± 0.5 | 49.2 ± 0.8 | 34.9 ± 0.8 | 718 / 750 / 1076 / 1272 |
| front-3v3 | 5461 | 3.0 ± 0.2 | 15.6 ± 0.5 | 50.7 ± 0.7 | 30.7 ± 0.6 | 1628 / 742 / 1281 / 1810 |
| front-5v5 | 9761 | 3.6 ± 0.2 | 15.8 ± 0.4 | 53.0 ± 0.5 | 27.6 ± 0.5 | 2175 / 1647 / 2645 / 3294 |
| front-10v10 | 22902 | 3.9 ± 0.1 | 15.2 ± 0.2 | 50.6 ± 0.3 | 30.2 ± 0.3 | 4798 / 4461 / 6325 / 7318 |

Mutual head-on is under 10 degrees off the victim's nose, offset head-on 10–45, flank 45–135,
tail beyond. Kills with a shooter only.

## Head-on passes before a kill

`bench.game_loop` (its merge definition: a pair inside the longer gun range with both noses
within the merge cone; a pair's merges count from when both last spawned; the histogram's
last bin holds 7 or more), 5v5, the stored doctrines and each mode's Imugi parameters,
mixed lineups in both seat orders, fields on, the preset's map; 512 games a row, deathmatch
300 s, elimination 248 s, Frontline 600 s. The bench gives no SE for the mean; the shares'
SEs are binomial over kills.

| mode | lineup | kills a game | passes before a kill (mean) | kills with no merge (%) | first-merge kills (%) | mutual / offset / flank / tail (%) |
|---|---|---|---|---|---|---|
| deathmatch | mixed | 39.6 | 0.38 | 68.0 ± 0.3 | 12.5 ± 0.2 | 3.4 / 12.1 / 58.9 / 25.6 |
| deathmatch | mixed reversed | 39.2 | 0.41 | 66.7 ± 0.3 | 13.2 ± 0.2 | 3.3 / 12.7 / 59.2 / 24.7 |
| elimination | mixed | 14.5 | 0.55 | 58.8 ± 0.6 | 11.5 ± 0.4 | 3.4 / 10.7 / 59.0 / 26.9 |
| elimination | mixed reversed | 15.1 | 0.53 | 60.3 ± 0.6 | 12.3 ± 0.4 | 3.5 / 11.1 / 59.1 / 26.4 |
| Frontline | mixed | 67.9 | 0.41 | 68.3 ± 0.2 | 12.2 ± 0.2 | 3.6 / 15.2 / 60.0 / 21.3 |
| Frontline | mixed reversed | 69.7 | 0.43 | 67.2 ± 0.2 | 12.9 ± 0.2 | 4.0 / 16.2 / 58.7 / 21.2 |

Most kills (59–68%) come with no head-on merge between the pair at all; the mean is 0.4–0.55
passes, against a target of 3–4. With the head-on fraction at 0 (physics v12), a hit on the
nose does no damage, so the card's closed-form head-on time to kill is infinite; the game
loop's rounds measured 0.3–0.6 on every card since 2026-11-12. The bench's aspect shares
differ from mode_dynamics' above (flank 59% against 51–53%); the two benches run different
clocks and game loops, and I have not traced the difference further.

## Respawn (report only, nothing changed)

`tune/respawn-waves` (last commit 972fd4dc, branch deleted) recommended on 2026-11-13:
Frontline a minimum 8 s then waves every 20 s, no capture effect; deathmatch a fixed 8 s, no
waves. At 5v5 waves leave P(same team next) at 0.47–0.53, shorten Frontline games (median
506 against 570 s) and in deathmatch cost 7–14% of the kill rate.

The game at v22 (`api.rules_of` of each preset):

| preset | respawn wait | waves |
|---|---|---|
| real_scale_deathmatch | 10 s fixed | none (waves are Frontline only) |
| real_scale_frontline | 8 s minimum | every 20 s per team |
| real_scale_elimination | none (no respawn) | none |

Frontline follows the recommendation. Deathmatch stays at 10 s, the author's choice of
2026-11-13 recorded in `modes/rules.py`; the branch measured 8 s at +3% kills per minute
against 10 s, with life and trades unchanged.

## Reproduce

```
cd <worktree>
# every condition, then the passes (skips what data/ already holds; about 40 min from empty)
systemd-run --user --unit=freeze-final -p MemoryMax=8G --working-directory=$PWD \
    --setenv=PYTHONPATH=$PWD/src:$PWD .venv/bin/python docs/rewrite/sections/freeze-final/measure.py
# the tables
PYTHONPATH=src:. uv run --no-sync python docs/rewrite/sections/freeze-final/freeze_report.py
```

Data: `data/modes.jsonl` (one JSON line per batch of 256 games), `data/passes.json`.
