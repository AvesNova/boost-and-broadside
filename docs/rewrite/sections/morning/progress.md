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

Direct games (this morning, 512): see below.

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
