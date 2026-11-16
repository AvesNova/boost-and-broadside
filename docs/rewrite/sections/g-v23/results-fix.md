# Option G on v23 and its shield trims, with the boundary fix

The balance check of 2026-11-16, 22:10 KST, on the RTX 5090 box, repeated with the scripted
agent's boundary fix (commit 1d5cbcc3, cherry-picked: the edge steer fades the doctrine's sense
and extension manoeuvres, `BOUNDARY_OVER_MANEUVERS`). `run-fix.sh candidates.json trims.json`,
2048 games per entry and condition, `--variants 2 --environments 4096`; deathmatch without fog,
elimination and Frontline with fog. Lines: /workspace/out/g-v23-fix/<condition>.jsonl.

Each cell: Yongma's (side A) kill share / point share (the share of all deaths, the game's score);
2v2 and 1v1 add deaths a game (both sides, every cause). Standard errors as in results.md
(about 0.001 in deathmatch, 0.002 Frontline, 0.005-0.007 elimination). Bands (kill share for a
45-55% win rate): DM 2v2 .481-.521, DM 5v5 .494-.507, DM 10v10 .497-.504, elimination 5v5
.498-.545, elimination 10v10 .492-.526, Frontline 5v5 .491-.513, Frontline 10v10 .493-.508; the
deathmatch bands were fitted with fog on, so they are approximate. 1v1 has no band.

| Candidate | DM-5v5 | DM-10v10 | Elim-5v5 | Elim-10v10 | FL-5v5 | FL-10v10 | DM-2v2 | DM-1v1 | in band | worst miss |
|---|---|---|---|---|---|---|---|---|---|---|
| v23 | 0.297 / 0.353 | 0.313 / 0.385 | 0.113 / 0.177 | 0.191 / 0.276 | 0.332 / 0.340 | 0.340 / 0.356 | 0.225 / 0.274 (12.88 d/g) | 0.031 / 0.074 (0.21 d/g) | 0 | 0.385 |
| G-v23 | 0.512 / 0.516 | 0.519 / 0.524 | 0.579 / 0.564 | 0.571 / 0.564 | 0.502 / 0.504 | 0.534 / 0.535 | 0.435 / 0.437 (9.41 d/g) | 0.391 / 0.318 (0.15 d/g) | 1 | 0.046 |
| Y0.50 | 0.505 / 0.507 | 0.511 / 0.515 | 0.549 / 0.542 | 0.540 / 0.537 | 0.493 / 0.494 | 0.522 / 0.523 | 0.424 / 0.425 (9.83 d/g) | 0.358 / 0.305 (0.15 d/g) | 2 | 0.057 |
| Y0.46 | 0.497 / 0.499 | 0.502 / 0.506 | 0.546 / 0.537 | 0.507 / 0.506 | 0.482 / 0.484 | 0.507 / 0.507 | 0.419 / 0.420 (9.89 d/g) | 0.356 / 0.286 (0.15 d/g) | 4 | 0.062 |
| J1.25 | 0.509 / 0.511 | 0.513 / 0.518 | 0.561 / 0.551 | 0.551 / 0.547 | 0.491 / 0.493 | 0.523 / 0.524 | 0.426 / 0.428 (9.46 d/g) | 0.340 / 0.279 (0.15 d/g) | 0 | 0.055 |
| Y0.50-J1.22 | 0.496 / 0.500 | 0.505 / 0.510 | 0.549 / 0.539 | 0.526 / 0.524 | 0.482 / 0.484 | 0.514 / 0.515 | 0.418 / 0.418 (9.62 d/g) | 0.308 / 0.262 (0.16 d/g) | 1 | 0.063 |

"in band" counts the seven banded conditions on kill share; "worst miss" is the largest distance
outside a band.

**Y0.46 is the closest card.** In band on 4 (DM 5v5, DM 10v10, elimination 10v10, Frontline
10v10); elimination 5v5 misses by +0.001 (point share 0.537, inside), Frontline 5v5 by -0.009,
and DM 2v2 by -0.062. Point share follows kill share within 0.01 everywhere except elimination
5v5 and 1v1.

What the fix changed (Y0.46, against results.md): the large modes hardly move (DM 5v5 0.506 ->
0.497, DM 10v10 0.506 -> 0.502, elimination 0.550/0.505 -> 0.546/0.507, Frontline
0.485/0.508 -> 0.482/0.507). DM 2v2 drops by 0.04-0.05 for every card (Y0.46 0.464 -> 0.419, with
more kills: 7.4 a game -> 9.2), so the old boundary deaths had favoured Yongma there;
2v2 is now every card's worst miss and no shield trim reaches it. 1v1 deaths rise from
about 0.02 to 0.15 a game: still no duel.

## Head-on damage on Y0.46

Y0.46 with a head-on damage fraction of 0.05 and 0.1 for both ships (`head_on_fraction_scale`
set to 0.05/0.3 and 0.1/0.3 of `rules.head_on_fraction` = 0.3; the cards have 0), boundary fix
on, the same settings: `run-headon.sh headon.json`, lines in /workspace/out/g-v23-headon/. The
first row is Y0.46 from the table above.

| Head-on | DM-5v5 | DM-10v10 | Elim-5v5 | Elim-10v10 | FL-5v5 | FL-10v10 | DM-2v2 | DM-1v1 | in band | worst miss |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.497 / 0.499 | 0.502 / 0.506 | 0.546 / 0.537 | 0.507 / 0.506 | 0.482 / 0.484 | 0.507 / 0.507 | 0.419 / 0.420 (9.89 d/g) | 0.356 / 0.286 (0.15 d/g) | 4 | 0.062 |
| 0.05 | 0.493 / 0.494 | 0.501 / 0.503 | 0.537 / 0.529 | 0.513 / 0.510 | 0.488 / 0.489 | 0.508 / 0.509 | 0.425 / 0.425 (10.26 d/g) | 0.583 / 0.574 (1.72 d/g) | 3 | 0.056 |
| 0.1 | 0.487 / 0.490 | 0.497 / 0.500 | 0.549 / 0.541 | 0.504 / 0.502 | 0.482 / 0.483 | 0.511 / 0.511 | 0.421 / 0.421 (10.85 d/g) | 0.627 / 0.620 (2.94 d/g) | 1 | 0.060 |

A small head-on fraction breaks the 1v1 deadlock: deaths a game 0.15 -> 1.72 (0.05) -> 2.94
(0.1), and the duels go to Yongma (kill share 0.58, 0.63). 2v2 gains a few deaths (9.9 -> 10.3
-> 10.9) and stays about 0.42. The 5v5 and 10v10 shares move by at most 0.01 (0.05: DM 5v5
-0.004, elimination 5v5 -0.009 into its band, Frontline 5v5 +0.006; 0.1: DM 5v5 -0.010 out of
its band, elimination 5v5 +0.003 out of its band).

## Why the 1v1 duels rarely end (Y0.46, boundary fix, no head-on damage)

`bench/duel_anatomy.py` (an extension of `bench/boundary_deaths.py`), deathmatch-1v1, 128 games
eager on the laptop GPU; numbers in duel-anatomy-Y0.46.json. 0.14 deaths a game (15 by hits, 3
boundary, all Yongma's), both ships alive 4.98 of the 5 minutes.

| Per ship | Yongma | Jujak |
|---|---|---|
| quiet flight (share of time alive) | 0.9% | 0.9% |
| longest quiet streak (mean of games / max) | 0.07 s / 0.10 s | 0.07 s / 0.10 s |
| shield restored a game | 0 | 0 |
| shield lost a game (of max 353 / 498) | 45 | 55 |
| shots a game / a merge | 523 / 22.5 | 378 / 16.2 |
| hits a game / a merge | 315 / 13.5 | 235 / 10.1 |
| opponent on the nose (line within its hit radius) and in range | 5.8% | 4.9% |
| within 15 degrees and in range / in range at all | 6.8% / 16.9% | 6.9% / 16.7% |
| sustained hard turn (>= half the sustained rate, one sense, >= 3 s) | 0.5% | 1.3% |
| extending | 10.1% | 0% |

Merges: 23.3 a game (one every 13 s). The game loop's post-merge classification: 3014 of 3017
windows are scissors (both reverse sense at least twice in the 3 s after the pass), 3 one-circle,
0 two-circle. Both ships in a sustained turn: 0.2% of the time (same sense), ~0% opposite; one
ship extending 10% (always Yongma), both 0%.

Reading: there is no circle stalemate and no recharge. The duel is a stream of head-on passes in
which the hits land: 550 a game, but at about 0.18 shield each (100 lost a game against shields of
353 and 498), because with a head-on fraction of 0 a hit near the nose does almost nothing. The
shield never recharges (no quiet streak longer than 0.1 s), it is simply barely touched. The
head-on runs above confirm it: a fraction of 0.05 takes deaths from 0.15 to 1.7 a game.
