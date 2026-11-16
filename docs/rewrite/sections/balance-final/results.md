# Balance final: results (2026-11-16)

Branch `tune/balance-final` (not merged). Game version 22 -> **23**. Imugi on both sides,
fog and fields on, each preset's own map, `sweep.py` (the game-loop harness: deathmatch 300 s
with no early win, the result the sign of the points; elimination and Frontline their own
clocks and results). Laptop RTX 4070 (power profile `performance`) and a rented RTX 3070.
The working notes, rounds and dead ends are in [progress.md](progress.md).

## Summary

| | before (v22) | after (v23) |
|---|---|---|
| Yongma team v Jujak team, deathmatch 5v5 (2048 games) | **24.8 ± 0.9%** | **47.7 ± 1.1%** |
| kill share of the Yongma team, same games | 46.2% | 49.8% |
| mutual head-on kills (E v T / mixed lineups) | 3.5% / 3.4% | 3.3% / 3.4% |
| offset head-on / flank / tail, mixed lineups | 12.3 / 58.8 / 25.5% | 12.8 / 59.3 / 24.5% |
| kills a minute, mixed lineups 5v5 | 7.91 | 7.83 |

The freeze measured 21.8 ± 1.8% (512 games, its own seed); this run's before is 24.8 ±
0.9%, and the four other before measurements here are 20.5-26.0%.

**The change is one handle**: Jujak's sustained turn rate x0.93, 0.7647 -> 0.7111 rad/s
(43.8 -> 40.7 deg/s at the set's x2 native scale), the card re-solved from the handles.
Jujak's sustained turn is then 1.20 times Yongma's instead of 1.29. Nothing else in either
ship file changes by hand; **Imugi's doctrines are unchanged** (the doctrine search could not
do it within the author's rules; below).

## What changed (exact values)

`src/bnb/game/ships/real-scale/jujak.toml` (the before file is `before/jujak.toml`, the after
`after/jujak.toml`):

| | before | after |
|---|---|---|
| **[handles] sustained_turn_rate** | 0.7646540 | **0.7111282** (x 0.93) |
| [card] induced_drag_scale | 0.266357 | 0.419743 |
| [card] static_thrust | 33.452 | 64.818 |
| [card] engine_power | 11269.71 | 11290.69 |
| [card] engine_power_full_store | 2091.71 | 2112.69 |
| [card] corner_nose_frequency | 11.5599 | 11.3800 |
| [card] dive_exchange_scale | 4.14380 | 4.12781 |
| [card] climb_exchange_scale | 3.48530 | 3.44689 |
| [card] sustained_stick | 0.741061 | 0.696955 |
| [card] fine_stick | 0.222318 | 0.209086 |

The card fields are the solver's (`api.solve_handles`, `handle_cards.py`), holding every other
handle (low-speed acceleration, climb, cruise, stall, dive, ...) at its value. Yongma's file
and `imugi_doctrines.json` are byte-identical to before. `bnb.game.VERSION` 22 -> 23 with its
note; `tests/sections/game_reference.pt` re-recorded (the cards changed).

## Range check (the author's rule: within the real WWII types)

The real spread is over the 16 real types in `src/bnb/reference/real-plane-types` (A6M5,
Bf 109 K-4, F4U-1D, F6F-5, Fw 190 A-8 and D-9, Ki-84, La-7, Mosquito, P-38L, P-47D, P-51D,
Spitfire IX and XIV, Tempest V, Yak-3; the Me 262 and Me 163 left out), same units:

| handle | real range | before (Y / J) | after (Y / J) | verdict |
|---|---|---|---|---|
| sustained turn rate (rad/s) | 0.503 (P-47D) - 0.785 (A6M5); max/min 1.56 | 0.593 / 0.765 (J/Y 1.29) | 0.593 / **0.711** (J/Y 1.20) | inside |
| low-speed acceleration | 8.87 (P-47D) - 18.97 (Bf 109 K-4) | 8.79 / 13.54 | unchanged | Yongma 1% under the floor (as before) |
| max shield ratio Y/J | real max/min 1.44 | 1.78 | unchanged | outside (as before, not touched) |
| damage per second ratio | real max/min 1.42 | 1.20 | unchanged | inside |

Two values were already outside before this work and are left as they were: Yongma's
shield over Jujak's (1.78 against the real 1.44), and Yongma's low-speed acceleration 1%
under the P-47D's. A Yongma shield rise is therefore not an option; a Yongma shield cut
toward the real spread is in round 12 (open below).

## The other conditions

Yongma team's score (wins + half the draws) and its kill share:

| condition (games) | before | after: Jujak turn x0.93 | B: Jujak turn x0.95 + Yongma climb x0.88 |
|---|---|---|---|
| deathmatch 1v1, 300 s (1024) | 30.1 ± 0.9% (kills 30.6%) | 31.3 ± 0.8% (31.8%) | 30.7 ± 0.9% (31.2%) |
| deathmatch 2v2, 300 s (512) | 14.7 ± 1.4% (33.3%) | 29.9 ± 1.9% (41.4%) | 26.2 ± 1.8% (39.8%) |
| **deathmatch 5v5, 300 s (2048)** | **24.8 ± 0.9% (46.2%)** | **47.7 ± 1.1% (49.8%)** | 51.2 ± 1.1% (50.1%) |
| deathmatch 10v10, 300 s (512) | 44.9 ± 2.1% (49.5%) | 71.9 ± 1.9% (51.8%) | 75.1 ± 1.8% (52.1%) |
| elimination 2v2 (1024) | 19.0 ± 1.2% (27.6%) | 24.4 ± 1.3% (33.5%) | 28.1 ± 1.4% (34.6%) |
| elimination 5v5 (1024) | 20.4 ± 1.3% (37.7%) | 38.1 ± 1.5% (45.6%) | 38.4 ± 1.5% (46.6%) |
| elimination 10v10 (512) | 25.0 ± 1.9% (43.1%) | 50.2 ± 2.2% (50.9%) | 42.0 ± 2.2% (50.2%) |
| Frontline 5v5 (1024) | 12.3 ± 1.0% (39.2%) | 30.5 ± 1.4% (46.6%) | 26.3 ± 1.3% (44.7%) |
| Frontline 10v10 (512) | 13.8 ± 1.4% (44.2%) | 46.1 ± 2.1% (50.1%) | 42.4 ± 2.1% (49.1%) |

Aspects stay as they were at every size (mutual head-on 2-3.5% of kills; mixed 5v5 offset
12-13%, flank 59%, tail 24-25%), and kills a minute within 0.1-0.2 of before. Mixed
lineups both teams on the same card; Imugi's strength cannot change with a card-only change
(the same doctrines on both sides), and the mixed job "after v before" read 48.8-50.3% as it
should.

Notes:
- Elimination and Frontline at 10v10 land on even with A (50.2%, 46.1%, from 25% and 14%),
  and A is nearer even there than B.
- **The size spread in deathmatch is the open problem.** One card moves deathmatch 2v2 from
  15% to 30% and 10v10 from 45% to 72%. In kill share the 10v10 change is small (49.5% -> 51.8%): in a 10v10 the win
  goes to the side with more kills almost every time, so a 2-point kill edge is a 72% win
  rate. The 5v5 target puts 10v10 Yongma-favoured. Round 12 tests a flatter pair (more
  Jujak turn, less Yongma shield).
- 1v1 is mostly draws in 300 s (as in the freeze) and the card does not move it.
- B (two ships changed, smaller each) centres deathmatch 5v5 better (51.2%) but is further
  off at deathmatch 2v2 and 10v10 and at elimination/Frontline 10v10; A changes one number,
  so it is the after. B's files are not written; its card is
  in `candidates/final.json`.

## Flatter cards (rounds 12-16, 512 games)

The v23 card centres deathmatch 5v5 but leaves the sizes apart (2v2 29%, 10v10 71%). Yongma
team's score, deathmatch:

| card | 2v2 | 5v5 | 10v10 |
|---|---|---|---|
| before | 14-15 | 22-25 | 41-45 |
| **v23: Jujak turn x0.93** | 28.6 | 51.8 | 71.3 |
| Jujak turn x0.91 + Yongma shield x0.8 | 27.7 | 42.4 | 58.3 |
| Jujak turn x0.91 + Yongma shield x0.85 | 29.5 | 48.3 | 69.9 |
| Yongma turn x1.08 | 36.3 | 58.0 | 84.7 |
| Jujak turn x0.95 + Yongma turn x1.08 | 52.1 | 80.3 | 94.4 |
| Jujak turn x0.91 + Yongma turn x1.08 + shield x0.8 | 62.3 | 82.0 | 92.6 |

- A Yongma shield cut takes more off the 10v10 than the 2v2, so turn closer together with
  less Yongma shield flattens the spread; shield x0.8 (113.6, 1.43 times Jujak's) also brings
  the shield ratio inside the real spread.
- Jujak turn x0.91 is the solver's limit with the other handles held (x0.88-0.90 miss the
  low-speed acceleration or the nose reversal by 1-9%).
- Yongma's sustained turn is the stronger lever (x1.08 alone: 58% at 5v5), and with it
  2v2 reaches even. Round 15 tries it with deeper shield cuts (x0.6-0.8) and smaller turn
  steps; progress.md has the latest.
- Frontline 5v5 is the hardest mode: 12% before, 31% with v23 (1024 games); 38% with x0.91 +
  shield x0.85 (512).

## The doctrine search (done first; why the cards)

- The before doctrines' weight (`strength x in_advantage`) is near zero in every matchup
  (Yongma's advantage is (turn ratio)^1.86 = 0.62 against Jujak, the centre 1.3; Jujak's 0.91,
  the centre 1.68), so Imugi flies the maneuvers and the base pursuit only.
- Turning Yongma's elements on against Jujak (slash, lag, deny) made Yongma worse (5-20%);
  64 + 64 random Yongma doctrines topped out at +3-4 points after a re-check (selection noise
  at the top). Turning Jujak's doctrine on against Yongma flips the matchup (87%) but makes
  Imugi's Jujak weaker (mixed 25-33% against the before): rejected.
- The one doctrine that helped (r3-041, +3-5 points, and stronger on mixed lineups, 58%)
  uses extend, the crowd extension and the duel extension, which the author turned off on
  2026-11-14 (they ran ships out of bounds and taught the BC student to flee;
  `test_play_doctrines`): not used. With those held at 0 (round 10), nothing moved it.

## How to test

On `tune/balance-final` (game v23, the after card):

```
# play the after card (Imugi both sides; the project map flies 2 Yongma + 3 Jujak a team)
uv run --no-sync bnb play --preset real_scale_deathmatch
uv run --no-sync bnb play --preset yongma-jujak

# switch to the before card (copies before/'s yongma.toml, jujak.toml and doctrines into src/)
PYTHONPATH=$PWD/src:$PWD uv run --no-sync python docs/rewrite/sections/balance-final/apply.py --restore
uv run --no-sync bnb play --preset real_scale_deathmatch
# option C, the flat card (below): writes its handles and solved card into src/
PYTHONPATH=$PWD/src:$PWD uv run --no-sync python docs/rewrite/sections/balance-final/apply.py \
    --candidates options.json jt0.91-yt1.08-ys0.6
uv run --no-sync bnb play --preset real_scale_deathmatch
# back to the after card (A, v23)
git checkout -- src/bnb/game/ships/real-scale src/bnb/controllers/imugi_doctrines.json
```

The files themselves: `before/` (v22), `after/` (A, as committed in src/), `options/<name>/`
(C, C', D, E: the ship files and doctrines each would put in src/).

(With the before files in src/ the game reference test fails, since VERSION says 23; that is
expected while testing.) To measure both side by side (energy against turn and mixed
lineups; the base card is before/'s whatever src/ holds; about 4 minutes on the 4070):

```
PYTHONPATH=$PWD/src:$PWD uv run --no-sync python docs/rewrite/sections/balance-final/sweep.py \
    --mode deathmatch --candidates docs/rewrite/sections/balance-final/candidates/options.json \
    --out /tmp/final-check.jsonl --games 1024 --mixed-games 256 --variants 4
uv run --no-sync python docs/rewrite/sections/balance-final/table.py /tmp/final-check.jsonl
```

`--mode elimination` / `--mode frontline` and `--ships N` for the other conditions. The
records behind every table here are in `data/` (`table.py` prints them).

## Option C: the flat card (found after the after was committed; not in src/)

Jujak sustained turn x0.91 (0.7647 -> 0.6958), Yongma sustained turn x1.08 (0.5930 -> 0.6404),
Yongma max shield x0.6 (141.9 -> 85.2, 1.07 times Jujak's 79.7). Every value inside the real
range (turn 0.503-0.785; shield ratio inside 1.44, which the before's 1.78 is not). Deathmatch,
512 games, two seeds (rounds 15 and 16):

| | 2v2 | 5v5 | 10v10 |
|---|---|---|---|
| before | 15-17% | 20-21% | 43-44% |
| A (v23) | 29% | 48-52% | 71-72% |
| **C** | **50.4 / 48.5%** | **54.4 / 60.2%** | **62.6 / 61.8%** |

Round 17 (512 games) found three cards even at every size; the 1024-game check (seed 200,
`data/options-deathmatch-*.jsonl`, SE 1.0-1.5 points) holds it:

| card (deathmatch, 1024 games) | 2v2 | 5v5 | 10v10 |
|---|---|---|---|
| before | 14.7% | 22.6% | 45.5% |
| A (v23): Jujak turn x0.93 | 28.7% | 48.2% | 74.4% |
| B: Jujak turn x0.95, Yongma climb x0.88 | 27.1% | 52.2% | 76.0% |
| C: Jujak turn x0.91, Yongma turn x1.08, Yongma shield x0.6 | 51.2% | 57.2% | 65.0% |
| **C': Jujak turn x0.91, Yongma turn x1.08, Yongma shield x0.55** | **46.0%** | **49.6%** | **54.2%** |
| D: Jujak turn x0.93, Yongma turn x1.08, Yongma shield x0.6 | 42.3% | 49.0% | 54.2% |
| **E: Jujak turn x0.95, Yongma turn x1.12, Yongma shield x0.55** | **45.9%** | **47.7%** | **50.1%** |

C' and E are inside 55/45 at every deathmatch size. Elimination and Frontline 5v5 (1024
games, `data/o2-*-laptop.jsonl`):

| card | elimination 5v5 | Frontline 5v5 | Frontline aspects E v T: mutual / offset / flank / tail |
|---|---|---|---|
| before | 21.0 ± 1.3% | 10.7 ± 0.9% | 3.6 / 15.9 / 64.8 / 15.7 |
| C' | 45.6 ± 1.6% | 58.0 ± 1.5% | 4.1 / 21.4 / 62.3 / 12.2 |
| D | 41.7 ± 1.5% | 54.9 ± 1.5% | 3.9 / 21.0 / 62.8 / 12.3 |
| E | 45.7 ± 1.6% | 55.2 ± 1.5% | 4.0 / 20.9 / 62.5 / 12.5 |

So E is at or inside 55/45 in every mode measured at 5v5 and every deathmatch size. With the
closer turns, more kills are offset head-on (about 21% against 16%) and fewer from the tail
(12% against 16%) in the energy-turn games; mutual head-on stays about 4%; kills a minute as
before. Kill aspects and kills a minute stay as
before in all of them (the records). Shield x0.55 is 78.1, a little under Jujak's 79.7;
x0.6 is 85.2. The turn ratio Jujak over Yongma: C' 1.09, D 1.11, E 1.09 (before 1.29, A 1.20).
To play them: `apply.py --candidates options.json jt0.91-yt1.08-ys0.55` (C'),
`... jt0.93-yt1.08-ys0.6` (D), `... jt0.95-yt1.12-ys0.55` (E). Their elimination and Frontline
5v5 checks are running on the laptop (`data/o2-*-laptop.jsonl`), and 2048-game deathmatch
runs of every option on the 3070 (`/workspace/out/big-deathmatch-*.jsonl` on the box).

C was the first card within about 13 points from 2v2 to 10v10, near even at every size. The
cost: the archetypes move closer (Jujak out-turns Yongma by 1.09 instead of 1.29, and the
shields are nearly equal; Yongma keeps its speed, climb, dive and 1.2x damage). It is a bigger
change than "relatively similar"; it is for the author to fly. Its elimination and Frontline
checks are running (progress.md). If C is chosen it would be v23's card instead of A (the
reference re-recorded again).

## Open for the author

1. A (one handle, v23 on the branch: 5v5 even, 2v2 29%, 10v10 72%) or a flat card (C', D:
   three handles, even from 2v2 to 10v10, the archetypes closer: Jujak out-turns Yongma by
   1.09-1.12 instead of 1.29, and the shields about equal). My pick if the sizes matter: C'
   or E (inside 55/45 at 2v2, 5v5 and 10v10 at 1024 games), pending their elimination and
   Frontline checks; E moves Jujak least, C' moves Yongma's turn least.
2. The card or the controller: as the freeze review said, this is Imugi's balance. The
   trained agent's own Yongma-v-Jujak score after the first arm is the check that matters.
3. The running-away elements stay off; if the author wants the energy fighter to extend,
   r3-041 is the measured doctrine (+3-5 points, stronger on mixed lineups).
4. Frontline stays Jujak-favoured with A (5v5 31%, 10v10 46%).
