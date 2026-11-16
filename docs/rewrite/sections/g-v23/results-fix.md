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
