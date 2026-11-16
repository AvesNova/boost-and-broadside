# Balance final: progress (2026-11-16)

## RESUME NOTE 2 (handover, 2026-11-16 ~13:10 KST; read first, then the 11:05-12:45 sections at the end)

- Laptop: the author's (no heavy jobs). All work is on the box (`ssh -p 12149 root@115.73.222.54`,
  /workspace/bnb fixed at 8488056e, do not pull src). Workers (worker.py, logs
  /workspace/worker-<mode>-<size>.log, queues /workspace/queue/<mode>-<size>/): deathmatch 2/5/10,
  elimination 5/10, frontline 5/10. Queued: probe2 (dm10, el5, el10, dm2), probe3 in all seven
  (outputs /workspace/out/probe{2,3}-*.jsonl). Collect: `rsync -a -e "ssh -p 12149"
  root@115.73.222.54:/workspace/out/ data/`, then `probe_table.py data/probe3-*.jsonl`.
- Finding: Frontline 10v10's offset is a kill-share effect, not a steeper win curve. Turn and
  shield/damage cannot hold FL10 with the rest; the energy economy (Yongma store, shot cost) is
  Frontline-light and Jujak's shield moves FL10 twice FL5. probe3 tests the model's cards (Q1-Q5).
- Next: pick the best probe3 card, express its turn change as a handle (Yongma
  sustained_turn_rate solved via handle_cards.py instead of the raw induced drag factor), range-check,
  confirm at 1024-2048 games in every mode, write it into results.md as option G.

## RESUME NOTE (handover, 2026-11-16 ~10:55 KST; read first)

**State of the branch** (`tune/balance-final`, not merged): game v23 = A (Jujak sustained turn
x0.93) is committed in src/ with the game reference re-recorded. results.md is the
deliverable (before/after, range check, how to test, options A-F). The before files are in
before/, A in after/, other options in options/<name>/ (`apply.py --candidates options.json
<name>` writes one into src/; `apply.py --restore` puts before/ back; `git checkout -- src/...`
returns to A). sweep.py always plays before/'s card as its base (`--base before`), so every
candidate's card is relative to v22 whatever src/ holds; handle_cards.py solves from before/.

**Best options so far** (Yongma team's score against a Jujak team):

| card | dm 2v2 | dm 5v5 | dm 10v10 | elim 5v5 | Frontline 5v5 | turn J/Y |
|---|---|---|---|---|---|---|
| before | 15 | 22.6 (2048) | 42.8 (2048) | 21 | 11 | 1.29 |
| A (v23): Jujak turn x0.93 | 29 | 50.0 (2048) | 74.2 (2048) | 38 | 31 | 1.20 |
| C': Jujak x0.91, Yongma turn x1.08, shield x0.55 | 46 | 50.9 (2048) | 52.9 (2048) | 45.6 | 58.0 | 1.09 |
| E: Jujak x0.95, Yongma turn x1.12, shield x0.55 | 46 | 50.1 (2048) | 49.6 (2048) | 45.7 | 55.2 | 1.09 |
| F: Yongma turn x1.12, shield x0.75 (Jujak untouched) | 40 | 54 | ? | 43 | 43 | 1.15 |

(1024 games unless marked; F from grid1 at 256-384.) The open problem: Frontline 10v10 has a
steeper slope than deathmatch (before 17%; C'/E-like cards about 76-80% at 256 games); F was
the best all-mode compromise in grid1 (Frontline 10v10 57).

**Running at handover**
- Laptop (4070; jobs via `laptop_sweep.sh` = systemd-run units under
  flock /tmp/claude-1000/bnb-gpu.lock, code from this worktree with PYTHONPATH=$PWD/src:$PWD):
  - fin3 (`candidates/fin3.json`: before, A, C', E, F, and two grid1 neighbours) in Frontline
    and elimination at 10v10 (512 games, 2048 environments) then 5v5 (1024) ->
    `data/fin3-{frontline,elimination}-{10v10,5v5}.jsonl`; progress in the scratchpad log of
    the previous agent (units `balance-fin3-*`; `journalctl --user -u balance-fin3-frontline-10`).
  - then grid2 (`candidates/grid2.json`, 41 cards around F: Jujak turn 1 / 0.97, Yongma turn
    1.10-1.16, shield 0.65-0.8, damage 0.9) in Frontline 10v10, Frontline 5v5, elimination 5v5
    -> `data/grid2-*.jsonl` (units `balance-grid2-*`).
- Box (RTX 3070, `ssh -p 12149 root@115.73.222.54`, repo /workspace/bnb): three persistent
  workers (`worker.py`, logs /workspace/worker-deathmatch-{5,2,10}.log) watching
  /workspace/queue/deathmatch-{5,2,10}/; each job file holds /tmp/gpu.lock while it plays.
  Queued in each: 01-grid1 (60 cards, 384 games), 02-fin3 (1024), 03-grid2 (384); outputs in
  /workspace/out/<name>.jsonl (`big-*` = the options at 2048, done for 5v5 and 10v10; 2v2
  running). Collect: `rsync -a -e "ssh -p 12149" root@115.73.222.54:/workspace/out/ data/`.
  To add work: drop a JSON job file ({"candidates": path or list, "out", "games",
  "mixed_games", "seed"}) into a queue directory; new workers compile before taking the
  lock (`worker.py` docstring).

**Box caveats**: slow Xeon, 10-20 min compile per new process (hence the workers; never start
one-shot sweeps there); 8 GB VRAM shared with the third-ship agent's workers
(/workspace/third-queue/), check nvidia-smi before adding workers; `git pull` in /workspace/bnb
aborts if untracked files collide with committed data (move them to /workspace/boxdata
first); never `pgrep -f`/`pkill -f` a pattern your own ssh command contains (it kills the
ssh shell); the box can be interrupted.

**Next on the list**
1. Collect fin3 (all modes at 5v5 and 10v10) and the box's 2048 2v2, grid1 and fin3 deathmatch;
   put the numbers into results.md (the options table and the second-pass section).
2. From grid2 + grid1, pick the most distinct card (largest Jujak/Yongma turn ratio) inside
   55/45 at dm 2v2/5v5/10v10, elimination 5v5 and Frontline 5v5, with Frontline 10v10 as
   close as possible; confirm it at 1024-2048 games in every mode at 5v5 and 10v10, with
   aspects and kills/min (`table.py`, `grid_table.py --band`).
3. Other slope handles not yet tried with Yongma turn: Yongma power (climb x0.88-0.93 raises
   10v10 more than 2v2), damage alone at 0.8-0.9, Jujak shield up.
4. Keep both GPUs busy (the author's standing instruction): small jobs on the laptop, big
   batches through the box workers. Don't merge; the author decides A vs a flat card.


Goal (author, 2026-11-15): Yongma team v Jujak team in deathmatch 5v5 within 55/45 (soft), from
21.8 ± 1.8% on game v22; doctrines first, cards only if needed; keep the kill-aspect mix,
mutual head-on near zero, Imugi's strength (mixed lineups v the before doctrines), Frontline
as a check. Power profile `performance`.

## 04:15 start

- Merged origin/feat/imugi-boundary (83404bbf) into tune/balance-final.
- Reading of the before doctrines: both types' doctrine weight (`strength` x `in_advantage`)
  is near zero in every matchup. Yongma's advantage is (turn ratio)^1.86: 0.62 against Jujak,
  1 in the mirror, against a centre of 1.3; Jujak's is (turn ratio)^-0.385:
  0.91 against Yongma, 1 in the mirror, against a centre of 1.68.
  So the before Imugi flies only the maneuvers (sense, the crowd extension, lock press) and
  the combat layer's base pursuit; the energy fighter never extends or slashes.
- Round 1 (deathmatch 5v5, 256 games E v T, 128 + 128 mixed): turn Yongma's doctrine on
  against Jujak only (turn_exponent -1.86 keeps the mirror off), with and without extend,
  deny, the duel extension gated on the lift ratio, and the sense against every type.

## 04:20 rounds 1-2 (deathmatch 5v5, 256 games E v T; a batch of 16 candidates takes ~2 min)

- Before: 0.26 ± 0.03 and 0.19 ± 0.02 (two repeats). Mixed before-v-before 0.51 / 0.47.
- **Turning Yongma's doctrine on against Jujak makes Yongma worse**: on 0.05, + extend 0.00
  (kills 0.20, tail 77%), + deny 0.04, slash only 0.20, lag only 0.07. The Baekho elements
  as tuned do not help Yongma against Jujak. The crowd extension 0.04-0.07; the sense
  against every type 0.14 (one-circle) / 0.20 (two-circle). Nothing beat the before.
- **Turning Jujak's doctrine on against Yongma flips the matchup**: 0.87 (bleed, deny, lag
  all on; without bleed 0.89, without deny 0.84). But mixed v before falls to 0.25-0.33, so
  that is a weaker Jujak, not a better Yongma: rejected (Imugi strength).
- The duel extension in the mirror (lift-gated, Yongma) leaves E v T unchanged (0.20) and is
  0.56 ± 0.03 against the before on mixed lineups: a possible strength gain, noted.
- Next: a random search of Yongma's doctrine (round 3, 64 candidates, 128 + 128 games),
  and a card round ready (sweep.py now takes a candidate "card" of field factors).

## 04:27 round 3: random Yongma doctrines (64, 128 + 128 games)

- Best E v T 0.34 ± 0.04 (r3-041), 0.34 (r3-032), 0.31 (r3-007); the median about 0.2. With
  64 draws at ±0.04, the top is mostly selection: a re-check at 512 games is round 5.
  Common to the top three: `lag` 0. No doctrine came near 0.45.
- The GPU lock is shared with another agent's timing jobs (04:28-04:40 waiting).

## 04:59 round 4: card field factors (stored doctrines, 256 + 256 games; the lock was busy 04:27-04:59)

Card fields (not handles), E v T: base 0.26; Yongma damage x1.15 0.27, x1.3 0.37, x1.5 0.51;
Yongma shield x1.2 0.33, x1.4 0.47; Jujak shield x0.85 0.40; Jujak damage x0.85 0.31; Jujak
induced drag x1.3 0.31, x1.6 0.49 (the sustained turn); Yongma engine power x1.2 0.43;
Yongma induced x0.8 0.34; Yongma explosive 1 0.30; max lift (instantaneous turn) Yongma x1.1
0.22, Jujak x0.93 0.20 (no effect). Mixed v before 0.46-0.54 everywhere (strength unchanged).
Aspects unchanged (mutual 3-4%). So the balance moves with the sustained turn, the power, and
damage/shield; not with the instantaneous turn.

Author's go-ahead (via the coordinator, ~04:45): cards may change, modestly, every value inside
the real types' range; prefer handles moving one ratio each. The real-type spread
(`src/bnb/reference/real-plane-types`, the 16 real WWII types, Me 262/163 excluded): sustained
turn 0.503-0.785 rad/s, low-speed acceleration 8.87-18.97, damage per bullet ratio 1.57, dps
ratio 1.42, shield ratio 1.44. Yongma's shield over Jujak's is already 1.78 (outside the real
spread), so a Yongma shield rise is out. Round 6 is handle candidates re-solved
(`handle_cards.py`).

## 05:09 round 5: the top random doctrines re-checked (512 E v T + 256 mixed)

Before 0.242 ± 0.018; r3-007 0.285 ± 0.019, r3-033 0.275 (mixed 0.40: weaker), r3-032 0.271,
r3-041 0.269 ± 0.019 with mixed 0.580 ± 0.030 (a stronger Imugi), the rest 0.20-0.24. So the
random search's top was mostly selection: doctrines move E v T by +3-4 points at most.
**Doctrines alone cannot reach 45%; the card is needed** (the author's go-ahead came in at
~04:45). r3-041 is the doctrine to carry: E v T +2.7 and stronger on mixed lineups (extend
0.93 gated on the energy ratio, lag 0, the duel extension in every matchup, wider fight range).

## 05:21 rounds 6-7: handle solves; r3-041 (d041) with card factors

- Handle solves (`handle_cards.py`, 7 processes, 14 min): Jujak sustained turn x0.95 and x0.91
  solve (the solver moves the induced drag x1.39 / x1.79, the static thrust, the sticks);
  x0.87 does not. **Yongma's low-speed acceleration does not solve at x1.25 or x1.5** (misses
  by 8-23%; ill-conditioned with the sustained turn and the cruise), and x1.15 moves the static
  thrust 49 -> 474: dropped. Yongma climb time x0.88 (333.6 -> 293.6 s, the Tempest's 290 the
  real floor) is the engine power x1.107. Damage is direct.
- Round 7 (d041 + card fields, 256 games): base 0.36, Jujak induced x1.3 0.43, x1.45 0.47;
  Yongma power x1.2 0.44; damage x1.3 0.43; induced x1.3 + power x1.1 0.52; induced x1.3 +
  damage x1.15 0.49 (mixed 0.59). The d041 base here (0.36 ± 0.03) is above its 512-game
  check (0.27 ± 0.02): noise between seeds is larger than I like; final numbers at 1024+.
- Round 8: d041 on the solved handle cards (Jujak turn, Yongma climb, Yongma damage, combos),
  512 games.

## 05:40 import check (coordinator's question)

Every sweep unit (rounds 1-8 and the finalists) runs `sweep.sh`: `systemd-run ...
-E PYTHONPATH=src:. --working-directory=/home/vizia/avesnova/bnb-balance-final`. Each unit's
journal shows its tracebacks/warnings from `/home/vizia/avesnova/bnb-balance-final/src/bnb`
(rounds 1, 3, 4, 5, 7, 8 checked), and `PYTHONPATH=src:. python -c "import bnb"` gives this
worktree's file. The card candidates are passed to the bench as explicit field values
(`{"set": ...}` or factors on the loaded card), and the handle solves ran with
`PYTHONPATH=src:.`. No round ran against the main worktree's code. From here on the scripts
use `PYTHONPATH=$PWD/src:$PWD`.

## 06:15 the d041 doctrine is out; finalists again on the before doctrine

- Finalists (d041 + handle cards, 1024 games, deathmatch 5v5): before 0.219 ± 0.012 (the
  freeze's 21.8 reproduced); d041 + Jujak turn x0.95 0.452 ± 0.015; + Yongma damage x1.08
  0.519 ± 0.015; + Yongma climb x0.88 0.557 ± 0.015. At 10v10 (300 s, no tickets) before
  0.41, d041 + Jujak turn x0.95 0.73: the card change overshoots at 10v10.
- **The quick tests caught it**: `test_play_doctrines` holds the author's decision of
  2026-11-14: Baekho's running-away elements (extend, the crowd extension, the duel
  extension) stay off (they ran ships out of bounds and taught the BC student to flee).
  d041's gain comes from exactly those (extend 0.93, crowd_sense 0.09, duel_extend 0.22; its
  turn_exponent 1.86 leaves its doctrine weight off against Jujak). So d041 is not allowed;
  the d041 finalist runs on the box (elimination, Frontline, 2v2) were stopped. Of the
  allowed elements (slash, bleed, lag, deny, the sense), rounds 1-2 found nothing that helps.
- Now: finalists2 = the before doctrine on handle cards (Jujak turn x0.97-0.91, Yongma damage
  x1.08/1.18, Yongma climb x0.88), on the rented 3070 (dm 5v5 1024 games, then 10v10 and
  2v2); a constrained random doctrine search (round 10: extend, crowd and duel extension held
  at 0) on the laptop when the lock frees.
- `test_ship_sets.py::test_training_takes_its_games_set_card` fails on the base tree too (a
  'trial' set has no drawn lineup types): not this branch's.

## 06:40 finalists2 (before doctrine, handle cards, deathmatch 5v5, 1024 games, the 3070)

| card | Yongma team | 
|---|---|
| before | 0.205 ± 0.012 |
| Jujak turn x0.97 + Yongma damage x1.18 | 0.417 ± 0.015 |
| Jujak turn x0.95 | 0.397 ± 0.015 |
| Jujak turn x0.95 + Yongma damage x1.08 | 0.440 ± 0.015 |
| Yongma damage x1.18 + climb x0.88 (Jujak untouched) | 0.457 ± 0.015 |
| Jujak turn x0.95 + Yongma damage x1.18 | 0.481 ± 0.015 |
| **Jujak turn x0.93** (one handle) | **0.485 ± 0.015** |
| **Jujak turn x0.95 + Yongma climb x0.88** | **0.494 ± 0.015** |
| Jujak turn x0.93 + Yongma damage x1.08 | 0.558 ± 0.015 |
| Jujak turn x0.91 | 0.597 ± 0.015 |

- Round 10 (64 random Yongma doctrines with extend, the crowd and duel extension held at 0,
  128 games): the top 0.31 ± 0.04, the rest at the before's level: the allowed doctrine
  elements do not move the matchup.
- The two finalists (A: Jujak turn x0.93; B: Jujak turn x0.95 + Yongma climb x0.88) go to the
  final measurement (`candidates/final.json`: deathmatch 5v5 2048 games, elimination and
  Frontline 5v5 1024, on the 3070; deathmatch 1v1 and elimination 2v2 on the laptop). More
  handle variants solving (round 11: Jujak low-speed acceleration, climb, drag, cruise, stall;
  Yongma climb x0.93).
- The box's job script now takes `flock /tmp/gpu.lock` (shared with the third-ship agent).

## 07:20 the after is in (v23), results.md written; round 11; more checks running

- **Final, deathmatch 5v5, 2048 games (3070, seed 100): before 24.8 ± 0.9%, A (Jujak turn
  x0.93) 47.7 ± 1.1%, B (Jujak turn x0.95 + Yongma climb x0.88) 51.2 ± 1.1%.** 1v1 30-31%
  for all three (mostly draws); elimination 2v2 19.0 / 24.4 / 28.1%. A is applied to src/
  (game v23, the reference re-recorded, quick tests on the touched files pass) and committed:
  c875bb84. results.md has the tables, the range check and how to test.
- sweep.py now plays before/'s card by default (`--base before`), so a sweep means the same
  thing whatever src/ holds; laptop jobs started before that ran from a detached worktree at
  the base (/home/vizia/avesnova/bnb-balance-sweep).
- Round 11 (512 games, other Jujak/Yongma handles alone): Jujak low-speed acceleration x0.85
  24.6%, x0.75 18.2% (the solver lowers its induced drag with it); Jujak climb x1.05 20.2%;
  drag x1.1 24.3%; cruise x0.97 22.7%; stall x1.05 20.7%; Yongma climb x0.93 30.0%; Jujak
  turn x0.94 45.9%; turn x0.95 + Yongma climb x0.93 50.5%; turn x0.94 + climb x0.93 48.0%.
  **Only the sustained-turn ratio (and a little Yongma power) moves the matchup.**
- Running: Frontline and elimination 10v10 (laptop), elimination and Frontline 5v5 (3070,
  waiting on the shared lock), round 12 (the size spread: Jujak turn x0.91-0.93 with Yongma
  shield x0.8-0.9) at 5v5, 2v2, 10v10 on the laptop.

## 07:50 round 12: the size spread (deathmatch 2v2 / 5v5 / 10v10, 512 games, laptop)

| card | 2v2 | 5v5 | 10v10 |
|---|---|---|---|
| before | 13.9 | 22.8 | 40.8 |
| Jujak turn x0.93 (the v23 after) | 28.6 | 51.8 | 71.3 |
| Jujak turn x0.91 | 34.1 | 54.6 | 81.0 |
| Jujak turn x0.91 + Yongma shield x0.8 | 27.7 | 42.4 | 58.3 |
| Jujak turn x0.91 + Yongma shield x0.85 | 29.5 | 48.3 | 69.9 |
| Jujak turn x0.91 + Yongma shield x0.9 | 30.7 | 48.9 | 73.9 |
| Jujak turn x0.93 + Yongma shield x0.9 | 26.9 | 38.6 | 62.0 |
| Jujak turn x0.93 + Yongma shield x0.85 | 19.7 | 36.8 | 55.2 |

(SE 1.4-2.1 points.) A Yongma shield cut takes more off the 10v10 than the 2v2, so more
Jujak turn with less Yongma shield flattens the spread: x0.91 + shield x0.8 spans 28-58
(30 points) against x0.93's 29-71 (43). Shield x0.8 is 113.6, 1.43 times Jujak's: inside the
real 1.44, which the before's 1.78 is not. Next: Jujak turn x0.88-0.90 (solving) with Yongma
shield x0.75-0.8, and these in elimination and Frontline 5v5.
- handle_cards.py now reads before/'s ship files (src/ holds v23 since c875bb84; a solve
  started at 07:50 from src/ was stopped before it finished, none of its numbers used).

## 08:12 elimination 5v5 final; round 12 in elimination and Frontline; Jujak turn's solver limit

- Final elimination 5v5 (1024, 3070): before 20.4 ± 1.3%, A 38.1 ± 1.5%, B 38.4 ± 1.5%.
- Round 12 in the other modes (512 games, laptop): elimination 5v5 before 26.0, x0.93 37.9,
  x0.91 + shield x0.8 36.1, x0.91 + shield x0.85 39.3; Frontline 5v5 before 8.4 (!), x0.93
  33.3, x0.91 + shield x0.8 33.2, x0.91 + shield x0.85 38.1. Frontline is the most lopsided
  mode and stays Jujak-favoured with every card so far.
- Jujak turn x0.88, 0.89, 0.90 do not solve (the low-speed acceleration or the nose reversal
  missed by 1-9%): x0.91 is the limit with the other handles held. Round 13 (3070): x0.91 with
  Yongma shield x0.75-0.85 and Yongma climb x0.88/0.93 or damage x1.08. Round 14 (laptop):
  Yongma sustained turn x1.08 with them.
- An elimination run on the laptop failed once on "Disk quota exceeded" in /tmp (the tmpfs
  was full for a moment; 8.9 of 23 GB used after); re-run, fine.

## 08:40 rounds 14-15: Yongma's turn and a deep shield cut (deathmatch 2v2 / 5v5 / 10v10, 512)

| card | 2v2 | 5v5 | 10v10 |
|---|---|---|---|
| Yongma turn x1.08 | 36.3 | 58.0 | 84.7 |
| Jujak turn x0.95 + Yongma turn x1.08 | 52.1 | 80.3 | 94.4 |
| Jujak x0.91 + Yongma turn x1.08 + shield x0.8 | 62.3 | 82.0 | 92.6 |
| Yongma turn x1.08 + shield x0.8 | 28.1 | 39.4 | 60.1 |
| Yongma turn x1.08 + shield x0.7 | 25.9 | 28.2 | 43.5 |
| Yongma turn x1.08 + shield x0.6 | 21.5 | 18.2 | 20.7 |
| Jujak x0.95 + Yongma turn x1.08 + shield x0.6 | 31.0 | 41.4 | 45.4 |
| **Jujak x0.91 + Yongma turn x1.08 + shield x0.6** | **50.4** | **54.4** | **62.6** |

The shield sets the slope over sizes (a shield cut takes most off the 10v10; at x0.6 the
spread even inverts) and the turn ratio sets the level. Jujak x0.91 + Yongma turn x1.08 +
shield x0.6 is the first card within about 13 points from 2v2 to 10v10 and centred. It is a
larger change: the turn ratio 1.29 -> 1.09, the shields nearly equal (85.2 against 79.7);
every value inside the real spread. Round 16: smaller steps around it (Yongma turn x1.04-1.08
solved, shield x0.6-0.7).
- The 3070 has been held since ~08:00 by another agent's compile-memory measure (the lock
  held, the GPU at 0%); my Frontline 5v5 final and round 13 wait behind it.

## 09:05 round 13, round 16, option C in results.md

- Round 13 (3070, deathmatch 5v5, 512): Jujak x0.91 + Yongma shield x0.8 43.8; + Yongma
  climb x0.88 54.3; + climb x0.93 50.5; shield x0.85 + climb x0.93 49.3; shield x0.75 + climb
  x0.88 50.3; shield x0.8 + damage x1.08 47.7; shield x0.85 + climb x0.88 55.1. Its 10v10 and
  2v2 wait on the box's lock.
- Round 16 (laptop, 512): C = Jujak x0.91 + Yongma turn x1.08 + shield x0.6 repeats at
  48.5 / 60.2 / 61.8 (2v2 / 5v5 / 10v10; round 15 had 50.4 / 54.4 / 62.6). Smaller steps:
  Jujak x0.91 + Yongma turn x1.05 + shield x0.65 40.3 / 49.2 / 59.3; Jujak x0.93 + Yongma turn
  x1.08 + shield x0.65 45.4 / 57.2 / 64.7; Jujak x0.91 + Yongma turn x1.04 + shield x0.7 40.1 /
  52.8 / 61.9; with shield x0.65 / x0.7 at turn x1.08 the 10v10 climbs to 71 / 83.
- results.md now offers C (options.json, apply.py --candidates options.json) beside A (v23 in
  src/). Next: round 17 (C with shield x0.5-0.55, Yongma turn x1.12), C at 1024 games in all
  deathmatch sizes (laptop), C in elimination and Frontline 5v5 (3070, queue7).

## 09:20 round 17; the box re-planned for compile cost

- Round 17 (laptop, 512): three cards even at every deathmatch size: C' (Jujak x0.91, Yongma
  turn x1.08, shield x0.55) 49.4 / 49.2 / 47.8; D (Jujak x0.93, Yongma turn x1.08, shield
  x0.6) 43.9 / 50.1 / 53.1; Jujak x0.95, Yongma turn x1.12, shield x0.55 46.9 / 51.7 / 48.3.
  Shield x0.5 overshoots the other way (10v10 40.5); Jujak x0.91 + Yongma turn x1.05 + shield
  x0.6 37.8 / 41.6 / 46.7. results.md offers C' and D.
- The box compiles ~10 min per launch on its slow CPU, so it now runs big batches only
  (queue9: all seven options at 2048 games, one process per deathmatch size; the small
  queues 6-8 were stopped before they started). The laptop takes the small jobs: options at
  1024 in all deathmatch sizes (running), then options2 in Frontline and elimination 5v5.

## 10:05 option checks in the other modes; grid1; the box worker

- C', D, E in elimination and Frontline 5v5 (1024, laptop): elimination 45.6 / 41.7 / 45.7,
  Frontline 58.0 / 54.9 / 55.2 (before 21.0 / 10.7). E is at or inside 55/45 in every mode
  measured at 5v5 and every deathmatch size. With the turns closer, offset head-on kills rise
  (about 21% against 16%) and tail kills fall (12% against 16%) in the energy-turn games;
  mutual head-on about 4%. In results.md.
- grid1: 60 cards = Jujak turn {1, 0.95, 0.91} x Yongma turn {1, 1.04, 1.08, 1.12} x the
  slope handle {none, shield x0.75, x0.6, damage x0.8, shield x0.75 + damage x0.9}.
  Frontline 5v5 (384 games, laptop) is even near a turn ratio of 1.13-1.15 (e.g. Jujak x0.95 +
  Yongma turn x1.08 + shield x0.75 50.8; Yongma turn x1.12 + damage x0.8 48.8); at 1.09 it is
  Yongma-favoured (64-82) unless the shield is cut. Raising Yongma's turn moves it more than
  lowering Jujak's at the same ratio. `grid_table.py` lists a card per row over the files.
  The deathmatch sizes of grid1 are on the box; elimination 5v5 on the laptop.
- **The box worker** (`worker.py`, the coordinator's request): one process per mode and size
  builds and compiles the loop once and then plays job files from
  /workspace/queue/<mode>-<ships>/ (JSON: candidates file or list, out, games, mixed_games,
  seed), moving each to done/ or failed/; each job takes /tmp/gpu.lock while it plays and
  releases it between jobs. Running: deathmatch 5, 2 and 10 (4096 environments, 8 variants);
  queued in each: the options at 2048 games, then grid1 at 384. To add work: drop a job file
  in the directory. The pattern fits any `bench.game_loop.Loop` bench (third.py too): build
  the loop once, then call the per-batch function per job.

## 11:05 (new agent) Frontline 10v10: what drives it; probe1

- **Win rate follows the kill share in every mode**; what differs is the kill share each
  mode gives a card. Fitting score = Phi(slope x (kill share - 0.5)) over grid1: the 50%
  point sits at kill share 0.500-0.505 in deathmatch and Frontline at every size, 0.52-0.53
  in elimination 5v5. The slopes (per unit kill share): dm2 6, el5 5, FL5 11, FL10 16,
  dm5 18, dm10 34. So 55/45 is a kill-share band of about ±2.1 points at dm2, ±2.4 el5,
  ±1.2 FL5, ±0.8 FL10, ±0.7 dm5, ±0.4 dm10. Frontline 10v10 is *not* steeper per kill
  than deathmatch 10v10 (16 against 34): its problem is the kill share itself.
- Kill share per card (grid1, 60 cards) regressed on the log factors (Jujak turn, Yongma
  turn, Yongma shield, Yongma damage): the shield and damage move every mode about alike
  (10-20 kill points per log unit), but the turn moves the small and dispersed fights far
  more (per log unit of Yongma turn: dm2 169, el5 174, FL5 117, FL10 99, dm5 76), and the
  modes start from different Yongma shares (dm2 31, el5 36, FL5 39, FL10 44, dm5 46 at the
  before card). Each mode's line crosses dm5's at a turn gap T = log(J ratio) + log(Y ratio):
  dm2 0.16, FL5 0.15, el5 0.13 (at its 52.8 target), but **FL10 at 0.075**. E has
  T = 0.165 (flat over dm2/dm5/dm10, FL5, el5), A has T = 0.073 (flat over dm5 and FL10:
  fin3 FL10 at 512 games: A 52.2, E 69.6, C' 72.6, before 16.1). So with turn and
  shield/damage only, FL10 and the small modes want different turn gaps: Frontline 10v10
  carries about 2.5 kill points of extra Yongma advantage at E's turn gap, and the two
  knob kinds cannot remove it (shield and damage move FL10 like dm5).
- So a third kind of knob is needed, one that moves FL10 differently from the rest.
  **probe1** (`candidates/make_probe.py` -> `candidates/probe1.json`): 24 raw card-field
  perturbations around E (Yongma shield, recharge rate and delay, vision, power, zero-lift
  drag, damage, gun range, muzzle speed, hit radius, explosive share, store, spawn energy,
  shot cost, max lift, induced drag; Jujak shield, damage, vision, power, recharge, range),
  512 games each in FL10, FL5, dm5, el5, dm10, el10 on the laptop and dm2 at 1024 on the box
  (worker queue; grid1's dm2 job held in queue/deathmatch-2/held, the laptop has it).

## 12:00 fin3 (A, C', E, F at 5v5 and 10v10, Frontline and elimination); probe1

fin3 (laptop; Frontline/elimination 10v10 512 games, 5v5 1024), Yongma team's score / kill share:

| card | FL5 | el5 | FL10 | el10 |
|---|---|---|---|---|
| before | 9.3 / 39.5 | 20.5 / 37.3 | 16.1 / 44.2 | 29.1 / 43.9 |
| A (Jujak turn x0.93) | 31.3 / 46.6 | 36.8 / 45.9 | 52.2 / 50.5 | 50.0 / 51.2 |
| C' | 58.6 / 51.9 | 46.5 / 49.7 | 72.6 / 53.4 | 45.3 / 50.0 |
| E | 55.2 / 51.3 | 46.5 / 50.0 | 69.6 / 52.8 | 48.8 / 50.5 |
| F (Yongma turn x1.12, shield x0.75) | 47.3 / 49.3 | 41.8 / 48.8 | 56.7 / 51.3 | 51.6 / 52.3 |
| Jujak x0.95, Yongma turn x1.08, shield x0.75, damage x0.9 | 48.3 / 49.7 | 43.2 / 48.0 | 62.6 / 52.3 | 48.8 / 50.9 |
| Yongma turn x1.12, shield x0.75, damage x0.9 | 41.0 / 48.4 | 39.8 / 47.0 | 56.8 / 51.1 | 46.7 / 50.3 |

Elimination 10v10 is even for every option; Frontline 10v10 is the outlier, as predicted.

probe1 (24 single-field steps around E, 512 games; `probe_table.py data/probe1-*.jsonl`):
kill-share change from E, in kill points (SE of a difference about 0.3; el5 about 1):

| probe | dm10 | dm5 | el5 | FL10 | FL5 |
|---|---|---|---|---|---|
| Yongma shield x0.9 | -1.1 | -1.5 | -2.1 | -1.0 | -1.1 |
| Yongma turn (induced drag x0.85, about turn x1.02) | +0.5 | +1.2 | +1.2 | +2.0 | +3.0 |
| Yongma store x1.3 | +0.2 | +0.4 | +0.9 | **-0.5** | +0.1 |
| Yongma shot cost x0.7 | +0.9 | +1.4 | +3.9 | **+0.3** | +0.4 |
| Yongma vision x1.3 | +0.3 | +0.7 | +1.2 | +0.6 | +2.1 |
| Yongma muzzle speed x1.15 | +1.9 | +2.8 | +4.4 | +2.3 | +3.1 |
| Yongma hit radius x0.85 | +2.3 | +3.2 | +4.1 | +3.3 | +3.0 |
| Yongma spawn energy x1.5 | +1.7 | +1.7 | +2.6 | +1.7 | +1.7 |
| Jujak shield x1.15 | -1.4 | -1.9 | -3.9 | -1.3 | -0.7 |
| Jujak vision x0.8 | +0.9 | +1.8 | +2.1 | +1.9 | +3.4 |
| Jujak gun range (bullet life) x0.85 | +0.9 | +1.1 | +3.1 | **+2.0** | +2.0 |

(the rest in the data: recharge rate/delay, explosive share, max lift near 0; damage, power,
drag, range small and flat.) **Three kinds of profile**: flat (shield, damage, spawn energy,
hit radius: all modes alike), Frontline-heavy (turn, vision, Jujak's gun range: FL moves 2-4x
deathmatch 10v10), and Frontline-light: **the energy economy** (Yongma's store and shot cost
move deathmatch and elimination but not Frontline). A third kind is what the turn+shield cards
lacked. So: give back some of Yongma's turn (or Jujak's range/vision edge) to pull Frontline
down, and pay Yongma in the energy economy (cheaper shots, a bigger store) to hold deathmatch
and elimination. That also keeps the archetypes apart (a larger turn gap; the energy fighter
gets more energy). The store and shot cost are game knobs, not WWII values.
probe2 (`candidates/make_probe2.py`): the least-squares combinations (store + vision + less
Yongma turn; store + shot cost + Jujak range; shot cost + vision + Jujak range) at 0.7/1/1.3 of
the fitted step plus single checks, in every mode (laptop) and dm2 (box, 1024).

## 12:45 probe2 (part); the laptop is the author's now

- The author needs the laptop (coordinator, ~12:40): no heavy laptop jobs until told. The
  probe2 chain was stopped after FL10, FL5 and dm5; dm10, el5 (1024) and el10 are queued on
  the box (dm10 worker; new elimination-5 and elimination-10 workers), dm2 behind probe1's dm2.
  The box stays at commit 8488056e (no src pulls: a warm relaunch is ~30 s, a pull costs a
  recompile).
- probe2 so far, kill share from E (score):

| probe | dm5 | FL10 | FL5 |
|---|---|---|---|
| E | 50.3 (51) | 52.5 (67) | 51.4 (57) |
| store x1.35 + vision x1.25 + Yongma induced x1.18 (x0.7) | -0.0 (53) | -1.1 (58) | -1.6 (46) |
| **store x1.53 + vision x1.37 + induced x1.27 (x1.0)** | -0.5 (48) | **-2.4 (47)** | -2.5 (44) |
| store x1.74 + vision x1.51 + induced x1.36 (x1.3) | -1.1 (43) | -3.4 (43) | -4.3 (34) |
| shot cost x0.86 + vision x1.71 + induced x1.40 | -0.8 (45) | -2.7 (51) | -3.2 (42) |
| Yongma induced x1.27 alone | -2.3 (39) | -2.6 (50) | -3.9 (37) |
| Yongma store x1.6 / x2 alone | +0.9 / +0.6 | -0.4 / -0.7 | -0.2 / -1.5 |
| Yongma vision x1.5 alone | +1.5 | +1.5 | +2.7 |
| store/shot cost + Jujak range up (3 scales) | +1.1..+1.4 | -0.5..-0.7 | -0.6..-1.5 |

  Jujak's longer gun range does not mirror probe1's shorter one (no Frontline gain), so the
  Jujak-range combinations fail; the store + vision + less Yongma turn line works: at x1.0 it
  puts dm5 and FL10 near even together for the first time (48 / 47), FL5 a little low (44).
- A per-mode linear model on the log factors (probe1 + probe2 points; rms 0.1-0.4 kill
  points) finds no store/vision/turn/shot-cost card that holds FL5 and FL10 together (FL5 is
  more turn-sensitive and FL10 less). **Jujak's shield** is the missing handle: it moves FL10
  twice as much as FL5 (-8.0 against -4.3 per log unit) and deathmatch about like FL10. With
  it the model has cards inside every band (dm10, dm5, el5, el10, FL10, FL5; dm2 unmodelled).
  probe3 (`candidates/make_probe3.py`): five such cards (Q1-Q5: Yongma store x1.7-3.7,
  vision x1.0-1.5, induced drag x1.06-1.22 (less turn), shot cost x0.42-0.61; Jujak shield
  x1.16-1.32, Q3 also Jujak vision x0.77) at 0.8/1/1.2 of the step, queued in all seven
  mode-sizes on the box (new frontline-10 and frontline-5 workers; probe3 dm2 at 1024).

## 13:28 (new agent, third) box reordered; probe3's raw cards measured; probe4 = the handle versions

- Box: the dm10, el10 and el5 workers were waiting on the lock with probe2 announced; stopped
  them (not playing), renamed their probe2 jobs to sort after probe3 (dm10 07-probe2, el 03-probe2),
  restarted with the same arguments (warm, ~1-2 min). dm2's stale 02-fin3 and 03-grid2 moved to
  queue/deathmatch-2/held (fin3/big already cover them). probe3 now runs first in every mode.
- `measure_cards.py` (new; flies a candidate file's cards with `api.measure_handles`, CPU, 1.5 min
  for 18 cards): **probe3's raw induced-drag step is a small turn change and a real
  low-speed-acceleration loss**: Yongma turn x1.08-1.11 (from E's 1.12), but low-speed
  acceleration x0.85-0.975, with Yongma already 1% under the real floor (P-47D 8.87): Q1x1.0
  would be 8.03, 9% under. So the raw cards break the range rule; the handle version is needed
  anyway, and it may play differently (the acceleration loss was part of the probe's effect).
- The store changes the climb-time handle as a side effect (climb_time = time to fill the store:
  Q1 x1.93, Q5 x3.7); the rate of climb at a given energy (the engine power) is unchanged.
  Jujak's shield at Q1-Q3 x1.0 is 98-105 against Yongma's 78 (the flag in the brief).
- probe4 (`candidates/make_probe4.py`): each probe3 card with Yongma's turn as the handle at
  the measured value (E's other handles), the store/shot cost/vision and Jujak's shield/vision
  raw on top. Solving on the box CPU now; queued behind probe3 once solved.
- 13:45 probe4 solved (16 solves, box CPU, ~14 min, all feasible: Yongma induced drag
  0.296-0.371 against the raw cards' 0.32-0.34 and E's 0.2805; static thrust 19.7-23.7, the
  sticks 0.70-0.73), measured: turn as intended, low-speed acceleration x1.000. Queued in all
  seven queues right after probe3 (dm2/el5 1024 games, the rest 512; outputs
  /workspace/out/probe4-*.jsonl), probe2's dm10/el leftovers after it. ETA for probe3 everywhere
  about 17:00 KST (worst 18:00; 7 jobs of 30-45 min sharing the lock with the third-ship agent),
  probe4 about 3.5 h after that.

## 14:20 the box GPU at 0%: card characteristics measured inside the lock (fixed in worker/sweep)

- The coordinator saw the GPU at 0% for ~35 min with a worker holding the lock at 100% CPU.
  Cause: `Loop.set_types` measures each chunk's card characteristics on the CPU
  (`measure_characteristics`), and `top_speed` (src/bnb/game/handling/shown.py) flies a boost
  until every store is spent, with a `bool()` stop each tick: its length grows with
  `max_energy`. Laptop CPU: 6.6 s for E and Q2 (store unchanged), >5 min for Q1 (store x1.93);
  on the box's Xeon minutes a chunk, all inside the GPU lock. No recompiles (every card field
  is a tensor). probe2's store cards were slow for the same reason.
- Fix (worker.py/sweep.py only, no src change, so the workers stay warm): `sweep.prewarm`
  measures every chunk's characteristics in spawned CPU processes (12 by default) into the
  disk cache under the loop's device key *before* the lock is taken; `set_types` inside the
  lock then loads them. `sweep.chunk_cards` is the one chunking both use. Checked on the
  laptop CPU (2 cards: the cache entry the loop reads is there).
- **src follow-up for later (not now)**: `top_speed`'s per-tick `bool()` stop scales with the
  store; the play shell must never stall on a new card, so a vectorized or analytic stop is
  worth doing after the balance.

## 14:45 the Frontline 10v10 recompile inside the lock: a carry stride (warm-up fixed in worker/sweep)

- The coordinator saw frontline-10 recompile the tick ([0/1]) on its first job, inside the lock.
  `recompile_check.py` (new; the worker's start-up at 64 environments, then a real job's ticks,
  TORCH_LOGS=recompiles; box CPU, outside the lock) gives the guard:
  `tensor 'carry.boundary_progress' stride mismatch at index 0. expected 1, actual 15`.
  `imugi_memory.track_boundary` returns `boundary_progress = obs.globals[:, episode_progress]`,
  a column view (stride 15, the global columns), while `init_carry` makes it contiguous. So the
  *second* tick of any game recompiles once per process; the filler start-up ran only one tick.
  It is a one-time ~3-5 min compile per worker process, not per job (no [0/2]); deathmatch
  workers show none.
- Fix (no src change): `sweep.warm`, called by worker.py after the filler tick, plays 3 ticks of
  a job-like batch (a card dict on every variant, the three lineups, a candidate and the before
  doctrines) and restores the base card. With it the recompile happens during start-up and the
  job's ticks compile nothing (check log: [0/1] before "warm done", none after).
- **src follow-up for later**: make the carry's `boundary_progress` contiguous (or the obs
  column a copy) in `track_boundary`, so the second tick never recompiles.

## 15:05 probe3 (dm2, dm5, FL5, FL10 in; dm10 done; el5/el10 playing); confirm1 queued

Kill share minus E's (score), 512 games (dm2 1024):

| probe3 | dm 2v2 | dm 5v5 | FL 10v10 | FL 5v5 |
|---|---|---|---|---|
| E | 44-48 | 51-53 | 69-71 | 54-60 |
| Q1x0.8 | +5.0 (60) | +0.4 (57) | -2.9 (50) | -2.3 (46) |
| Q2x1.0 | +5.7 (62) | +0.0 (52) | -3.3 (55) | -2.7 (46) |
| Q3x0.8 | +4.1 (55) | -0.1 (49) | -3.0 (49) | -2.1 (45) |
| **Q4x0.8** | **+2.1 (50)** | **-0.4 (52)** | **-3.0 (52)** | **-2.6 (47)** |
| Q4x1.0 | +1.7 (49) | +0.1 (53) | -3.5 (47) | -3.4 (42) |
| Q5x1.0 | - | -0.4 (50) | -4.3 (32) | - |

- The cards pull FL10 and FL5 down together as modelled, but **dm2 (unmodelled) rises 4-6
  kill points with vision** (Q1, Q2, Q5 have Yongma vision x1.4-1.5): those are 55-62 at 2v2.
  Q4 (no vision: store x1.69, shot cost x0.57, Yongma turn x1.110, Jujak shield x1.23) is the
  only line even at 2v2. Q4x0.8 is inside 55/45 in all four so far.
- confirm1 (`candidates/make_confirm1.py`): E and Q4 at 0.7 / 0.8 / 1.0 of the step as handle
  cards (Yongma turn handle 1.113 / 1.112 / 1.110, low-speed acceleration held), 2048 games
  (10v10 1024) with 256 mixed, queued ahead of probe4 where the worker had not yet started it.

## 15:15 author's decision: vision stays equal

- Author (2026-11-16, via the coordinator): **both ships keep equal vision** (unequal vision
  makes scaling the game for human play harder). The vision cards (probe2's store+vision line,
  probe3's Q1, Q2, Q3, Q5, Jujak vision x0.77 in Q3) are information only, not options. Q4-style
  cards only: store, shot cost, Jujak shield, turn. If no vision-free card reaches the band, say
  so rather than bring vision back.
- probe4's jobs not yet started (dm2, dm10, el5, el10, FL5) moved to queue/*/held (mostly
  vision cards; confirm1 covers Q4). dm5 and FL10 had already started theirs.

## 15:25 probe3 complete (all seven)

Q4 rows (score; kill share change from E), 512 games (dm2, el5 1024):

| probe3 | dm2 | dm5 | dm10 | el5 | el10 | FL5 | FL10 |
|---|---|---|---|---|---|---|---|
| E (two repeats) | 44 / 48 | 51 / 53 | 50 / 45 | 45 / 42 | 47 / 49 | 60 / 54 | 71 / 69 |
| **Q4x0.8 (raw)** | **50** | **52** | **45** | **49** | **52** | **47** | **52** |
| Q4x1.0 | 49 | 53 | 43 | 49 | 52 | 42 | 47 |
| Q4x1.2 | 51 | 51 | 37 | 46 | 45 | 39 | 40 |

**The raw Q4x0.8 card is inside 55/45 in all seven mode-sizes** at 512-1024 games (dm10 at the
edge, where E itself reads 45-50). The vision cards are info only (author). Next: confirm1
(the handle version, 2048 games) and grid3 (vision-free neighbourhood) are queued.
- 15:45 G in handle space (`candidates/g-handles-spec.json` -> `g-handles.json`, solved on the
  box): measuring the tested Q4x0.8 handle card gives exactly seven handle changes (Yongma
  sustained turn x1.1119, max shield x0.55, ceiling x1.5216, climb time x1.5221, shots per store
  x2.3823; Jujak sustained turn x0.95, max shield x1.1832); re-solving those reproduces the
  tested card within 0.07% in every field, so `apply.py --candidates g-handles.json G` writes a
  consistent ship file. probe4 dropped from dm10 (waiting) and FL10 (after its current chunk,
  `/workspace/drop-job.sh`): mostly vision cards now; confirm1 runs next there.

## 16:05 confirm1 (handle cards, 2048 games; 10v10 1024): Q4x0.7 is G

Yongma team's score (SE 1.0-1.6):

| card | dm2 | dm5 | dm10 | el5 | el10 | FL5 | FL10 |
|---|---|---|---|---|---|---|---|
| E | 47.5 | 50.7 | 48.1 | 45.3 | 46.9 | 56.4 | 65.1 |
| **Q4x0.7h (G)** | **50.4** | **51.4** | **45.9** | **48.3** | **49.9** | **45.7** | **53.7** |
| Q4x0.8h | 49.7 | 51.7 | 45.2 | 48.4 | 46.1 | 44.6 | 54.7 |
| Q4x1.0h | 49.9 | 48.9 | 38.7 | 47.4 | 48.2 | 40.3 | 48.2 |

- Q4x0.7h is inside 55/45 in all seven (the edges: dm10 45.9 ± 1.5, FL5 45.7 ± 1.1, FL10
  53.7 ± 1.5); x0.8 has FL5 just outside. The handle version holds the raw probe's result
  (low-speed acceleration held, so the range rule holds).
- Aspects and kills a minute as E's in every mode (e.g. dm5 E v T mutual / offset / flank / tail
  4.0 / 15.6 / 61.7 / 18.7 against E's 4.2 / 14.9 / 61.6 / 19.3; kills/min 8.34 against 8.15).
- The residual pattern: dm10 and FL5 at the low edge, FL10 at the high edge. grid3 (queued)
  maps store / shot cost / Jujak shield around Q4x0.8, including closer-shield cards.
