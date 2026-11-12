# Morning follow-ups (2026-11-13): progress

Branch `eval/morning` (from `origin/eval/r9m2-maps` 22764d67), worktree
`/home/vizia/avesnova/bnb-morning`. Hard stop 11:00 KST. GPU: the laptop RTX 4070, one job at
a time as `systemd-run --user` units with MemoryMax=8G (`queue.sh`, which pauses while a
`bnb play` process runs). Power profile `performance` (read 08:24).

## Item 1: the BC Elo anchor

**Sources.** gpubox4 `/workspace/calib/` (read only): the ratings rounds `ratings/round-0*`
(copied to `runs/calib/ratings/`, gitignored), the run `bc-mb256` (`config.json`, `run.json`,
`metrics.jsonl`, the 10M and 30M policy files).

**Anchor and teacher are the same agent.**
- The ratings: `tournament.py round` runs `bnb ratings <run> --no-league --no-ladder
  --lineup Yongma Yongma Jujak Jujak Jujak ...`. The field is the policy files plus the
  ladder's "scripted" (`bnb.runtime.lanes.SCRIPTED_PLAYER`, pinned at 1000:
  `bnb.league.tournament.ANCHOR`). In `bnb.league.matches` a scripted row plays
  `Scripted(characteristics, S, MODE_DEFAULTS, temperature 1, share 1)` with the default
  `doctrines=STORED` (the stored per-type doctrines of the card's ship set), flying the
  lineup.
- The teacher: the run's `config.json` has `teacher.temperature 1.0`, `teacher.floor 0.0`
  and lineup Yongma Yongma Jujak Jujak Jujak. `bnb.learn.bc.teacher_agent` is
  `Scripted(characteristics, S, MODE_DEFAULTS, 1.0)` with the same `STORED` doctrines. BC's
  scripted pools play `TeacherOpponent`, which is that agent at floor 0.
- Same parameters, doctrine, temperature, lineup and code tree (the box repo is
  `calib/overnight` 0f3ec4b4). This is not an anchor mismatch.

**The stored ratings put bc-mb256 30M below the scripted agent, not above it.** Frontline,
calibrated (scripted = 1000):

| bc-mb256 EMA | 5M | 10M | 15M | 20M | 25M | 30M |
|---|---|---|---|---|---|---|
| round 1 | -217 ± 33 | -314 ± 32 | -98 ± 29 | 410 ± 23 | 539 ± 22 | **631 ± 17** |
| round 3 (reuses round 1) | -152 ± 27 | -268 ± 28 | -67 ± 25 | 428 ± 21 | 552 ± 21 | **644 ± 15** |

The tournament's own direct games against the scripted agent for that snapshot, over every
round's records (frontline): **29 W / 145 D / 850 L** (score 0.085, about -410 Elo). Every
BC snapshot up to 20M lost all its direct games against the scripted agent (0 / 0-3 / 253-256).
"1631" is in no record or log on the box. It is 1000 + 631: the leaderboard's
`calibrated_elo` already carries the 1000 anchor, and adding the anchor again gives 1631.
This is a reading error. The sign is right (the fit gives scripted 0 and the snapshots
negative before the shift).

**Direct games (08:27-08:32 KST, 512 games).** `runs/h2h_bc.sh`: `bnb ratings bc-mb256
--no-league --no-ladder --presets real_scale_frontline --budget 512 --blocks 2
--block-environments 256 --lineup Yongma Yongma Jujak Jujak Jujak --with <30M EMA>` on the
`calib/overnight` tree (bnb-calib's venv, the code the policy was trained on). The 30M EMA
against the scripted agent, frontline 5v5: **20 W / 59 D / 433 L, score 0.097 ± 0.013**
(-387 Elo; the fit: 613 ± 26). By seat: 10 / 29 / 217 and 10 / 30 / 216. This agrees with the
box's 29 / 145 / 850 (0.085). **BC is not beating its teacher. It loses about 85% of frontline
games to it.**

**The dip at 10M.** In the BC-against-BC games the 10M EMA loses to the 5M EMA: 32 W /
120 D / 104 L (bc-mb256; mb64 and mb32 show the same). The training log's frontline games
against the teacher (`metrics.jsonl`, per update, learner side) show why:

| Steps | KL | shots | hits | combat deaths | boundary deaths | life s |
|---|---|---|---|---|---|---|
| 3-5M | 0.65-0.78 | 2,400 | 50-70 | 0.2-0.3 | 5-8 | 75-111 |
| 6.3-7.3M | 0.50-0.62 | 190-610 | 60-90 | 0.3-0.5 | 0.0-3.4 | 118-129 |
| 9.4-11M | 0.38-0.41 | 1,450-1,800 | 660-810 | 5.4-7.8 | 0.5-0.6 | 108-127 |
| 20M | 0.41 | 4,700 | 2,100 | 21 | 0 | 161 |
| 30M | 0.39 | 6,100 | 2,900 | 31 | 0 | 132 |

Up to about 7M the clone sprays fire, rarely fights (0.2-0.5 combat deaths a game) and
mostly dies at the boundary. Between 7 and 10M it learns to close and engage (hits x10,
combat deaths x15) before it wins the engagement. Against an earlier clone that does not
fight, that loses: it trades into the 5M policy's fire instead of drawing. The KL falls
steadily through the dip (0.68 to 0.39), so this is a phase of imitation (approach learned
before the fight), not a training fault. Every snapshot up to 20M lost every direct game
against the scripted agent (0 W). The ratings below about 400 rest on BC-against-BC games
only, so their place against the anchor is weakly set (the prior and the chain through
20-30M).

## Item 2: what "extend when outgunned" adds (r9m2 minus r9m)

Nothing run. maneuvers-2 `confirm2/` (1024 games, SE about ±0.010 at 1v1, ±0.015 at 2v2 and
5v5) and `ten/` (512, ±0.021): r9m2 (`champion2.json`) against r9m on the four cards, r9m2's
score, both orders (1v1: the Yongma mirror / the Jujak mirror; 2v2, 5v5, 10v10: mixed /
mixed reversed), as Elo (400 log10 s/(1-s)); the mean of the two orders with its SE:

| Card | 1v1 | 2v2 | 5v5 | 10v10 |
|---|---|---|---|---|
| D2 | 0.562 / 0.532: +43 / +22 (+33 ± 5) | 0.533 / 0.539: +23 / +27 (+25 ± 7) | 0.586 / 0.532: +60 / +22 (+41 ± 7) | 0.558 / 0.607: +40 / +76 (+58 ± 11) |
| cone 0.085 | 0.555 / 0.531: +38 / +22 (+30 ± 5) | 0.534 / 0.541: +24 / +29 (+26 ± 7) | 0.591 / 0.554: +64 / +38 (+51 ± 7) | 0.552 / 0.592: +36 / +65 (+50 ± 11) |
| big v small | 0.563 / 0.577: +44 / +54 (+49 ± 5) | 0.530 / 0.519: +21 / +13 (+17 ± 7) | 0.554 / 0.556: +38 / +39 (+38 ± 7) | 0.559 / 0.565: +41 / +45 (+43 ± 10) |
| frozen | 0.591 / 0.571: +64 / +50 (+57 ± 5) | 0.552 / 0.519: +36 / +13 (+25 ± 7) | 0.560 / 0.517: +42 / +12 (+27 ± 7) | 0.575 / 0.606: +53 / +75 (+64 ± 11) |
| **all four** | **0.560: +42 ± 3** | **0.533: +23 ± 4** | **0.556: +39 ± 4** | **0.577: +54 ± 5** |

The same runs' r9m-against-r9m control rows lie within 0.460-0.544 (worst ±31 Elo at 10v10,
±20 at 1v1-5v5). The run-to-run spread is about twice the binomial SE, so a single cell is
1-2 spread units. The pooled per-size means are clear. These are the maneuvers-2 cards (D2,
cone, big v small, frozen), not the play maps: the r9m2-maps runs played self-play, and r9m2
against r9m on the play maps was not recorded.

## Item 4: the four-ship set in mixed 4v4 (`fourship.py`, 08:37-08:41 KST)

**Harness.** `bench.game_loop` flew two types a game (a Yongma slot and a Jujak slot per card
variant) and tallied kills by those two kinds. Added (tested in
`tests/sections/test_game_loop.py`): `ROW_LINEUPS`, lineups that name each ship's row of
the variant table, so a game can fly all 2K rows; and `kills_by_row` (killer side x
killer row x victim row). `fourship.py` builds the 4-type table with K = 2: rows standard
and close fighter (Yongma slot, the Yongma doctrine, as on their maps) and brawler and small
(Jujak slot, the Jujak doctrine; on `arch-brawler-small` the brawler flies the Yongma slot).
The cards come from the play maps (`arch-standard-brawler`, `arch-close-standard`,
`arch-standard-small`). Every field that differs between the two ship files is set on
these maps, so each type is one ship whichever slot it fills. Self-play, 4v4 team
deathmatch, 300 s, fog, the density rule's arena, 1024 games a lineup (SE ±0.016; the
symmetric lineup is the noise check). Per side and type: share of all kills, K/D, kills a
minute a ship. Each type has about 3,000 kills a lineup, so K/D is good to about ±0.04.

| Lineup (A v B) | r9: A score | r9m2: A score |
|---|---|---|
| one of each v one of each (control) | 0.509 | 0.527 |
| 2 standard + 2 brawler v 2 small + 2 close | 0.485 | **0.436** |
| 2 standard + 2 small v 2 brawler + 2 close | 0.527 | 0.537 |
| 2 standard + 2 close v 2 brawler + 2 small | 0.496 | 0.544 |
| 4 standard v one of each | 0.490 | 0.530 |
| 4 brawler v one of each | 0.486 | **0.428** |
| 4 small v one of each | 0.529 | 0.495 |
| 4 close v one of each | 0.506 | 0.505 |

One of each on both sides, kill share and K/D per type (A side; B agrees within 0.07):

| Type | r9 | r9m2 |
|---|---|---|
| standard | 0.13, 1.03 | 0.13, 1.04 |
| close fighter | 0.12, 0.97 | 0.12, 1.01 |
| brawler | 0.13, 1.07 | 0.13, 1.04 |
| small | 0.12, 0.97 | 0.12, 0.97 |

(full per-side rows in `fourship.md`; kills a minute a ship 0.60-0.69 under r9, 0.44-0.54
under r9m2: r9m2's slower crowds.)

- **Under r9 every type is viable and none dominates.** In the one-of-each game every type
  takes 0.12-0.13 of the kills at K/D 0.97-1.07. Every unequal composition is within
  0.485-0.529 (at most 1.8 SE off even), and so is every four-of-a-kind against a mixed team.
  Composition does not matter under r9.
- **Under r9m2 the brawler is the weak stack.** Two brawlers + two standards lose to small +
  close (0.436, -45 Elo), and four brawlers lose to one of each (0.428, -50 Elo). In both,
  the brawlers' K/D is 0.95. A single brawler in a mixed team holds (K/D 1.04 / 0.97). This
  is archetypes' "under CE the brawler slips against the small ship" (0.37-0.43): the
  crowd extension takes the close committed fight away from the short-sighted ship. The
  standard + close and standard + small teams lean 0.54 (+28 Elo), about 2 SE. The
  control's 0.527 says that is near the spread.

## Item 3: the lead turn on top of r9m2 (`lead.py`, `leadtable.py`, 08:40-08:53 KST)

Re-scoped by the coordinator at 08:40 (the author: 0.5 s may be early). The lead turn
(two-circle sense, full stick, crossing gate 0.5 rad as in maneuvers' round 3) at 0.1 /
0.2 / 0.3 / 0.5 s before the merge, on top of r9m2, on balance-d2, the frozen card (the
ship files) and hit-and-run v standard. Each was played against r9m2 at 1v1 / 2v2 / 5v5 in
four lineups (E v T, T v E, mixed, mixed reversed), plus self-play E v T. 512 games a cell.
r9m2 against itself is the same-run control. (A first launch at 08:34 failed on shell
quoting; a first item-3 run, 0.5 s only against r9 and r9m2, was stopped at the re-scope.)
Table in `lead/table.md`. The mean of the four lineups as Elo:

| Card: doctrine | 1v1 | 2v2 | 5v5 | self-play E v T 1v1 / 2v2 / 5v5 |
|---|---|---|---|---|
| D2: r9m2 (control) | +4 ± 5 | -12 ± 7 | -11 ± 7 | 0.505 / 0.456 / 0.619 |
| D2: lead 0.1 s | -2 | -1 | -27 | 0.569 / 0.427 / 0.556 |
| D2: lead 0.2 s | +7 | -12 | -31 | 0.572 / 0.424 / 0.557 |
| D2: lead 0.3 s | +1 | +2 | -23 | 0.543 / 0.444 / 0.559 |
| D2: lead 0.5 s | +2 | -9 | -20 | 0.562 / 0.439 / 0.548 |
| frozen: r9m2 (control) | -4 ± 4 | -3 ± 7 | +3 ± 7 | 0.013 / 0.438 / 0.353 |
| frozen: lead 0.1 s | -12 | +6 | -10 | 0.014 / 0.426 / 0.322 |
| frozen: lead 0.2 s | +2 | -13 | -27 | 0.009 / 0.411 / 0.334 |
| frozen: lead 0.3 s | -3 | -11 | -29 | 0.010 / 0.414 / 0.357 |
| frozen: lead 0.5 s | +0 | -9 | -23 | 0.008 / 0.451 / 0.338 |
| hit-and-run: r9m2 (control) | -4 ± 5 | -7 ± 7 | +3 ± 7 | 0.514 / 0.447 / 0.536 |
| hit-and-run: lead 0.1 s | -2 | +2 | -29 | 0.522 / 0.443 / 0.516 |
| hit-and-run: lead 0.2 s | **+22** | -16 | -26 | 0.505 / 0.431 / 0.512 |
| hit-and-run: lead 0.3 s | +4 | -1 | -16 | 0.497 / 0.474 / 0.524 |
| hit-and-run: lead 0.5 s | +13 | -16 | -30 | 0.500 / 0.480 / 0.507 |

- **No lead time helps r9m2.** At 1v1 and 2v2 every setting is within the noise of the
  control (one exception: hit-and-run 0.2 s at 1v1, +22 ± 5, not repeated on the other
  cards or at 2v2). **At 5v5 every setting loses**: -10 to -31 Elo, against controls of -11
  to +3, so about -10 to -30 net on every card. Shorter is no better than 0.5 s. Turning
  before a crowd merge seems to cost what r9m2's crowd extension gains.
- **The tilt**: on D2 the lead turn moves the 1v1 E v T toward the Yongma (0.505 to
  0.54-0.57) and the 5v5 toward even (0.62 to 0.55-0.56). On frozen and hit-and-run the
  balance moves by at most 0.03.
- None is at least neutral overall. The least bad is 0.3 s (even at 1v1 and 2v2 on all
  three, -16 to -29 at 5v5), and it is not recommended.
