# Yongma lone-duel lead turn (2026-11-13): progress

Branch `eval/yongma-lead` (from `origin/eval/morning` 566f0e9a), worktree
`/home/vizia/avesnova/bnb-yongma-lead`. Hard stop 11:00 KST. GPU: the laptop RTX 4070, one
job at a time as `systemd-run --user` units with MemoryMax=8G (`queue.sh`, which pauses
while a `python ... -m bnb play` process runs). Power profile `performance` (read at each
launch: 09:00, 09:11, 09:24).

**Question (the author's).** Does a lead turn given only to the Yongma, and only in a lone
duel, make the Yongma better?

## The gate (code)

- `Doctrine.lead_turn_in_crowd` (falling logistic, open by default): the lead turn's weight
  times the logistic of the sense's crowd count (the soft count of the other known enemies
  within `sense_crowd_in_gun_range`, r9m2: 3 gun ranges; no nose-angle filter or ally
  weight in r9m2). At r9m2's own `sense_in_crowd` (0.5, 0.1) the lead turn flies only with
  no other known enemy within 3 gun ranges. The crowd count moved above the lead turn in
  `scripted_combat.py`; the sense reads the same value.
- `Doctrine.lead_turn_in_others` (falling, open by default): the same on the count of the
  other known enemies at any range (at 1/2, the lead turn only against the last known
  enemy). Added for the stricter variant below.
- Tests: `test_the_lead_turn_crowd_gates_fly_it_only_in_a_lone_duel`; quick tier of
  `test_scripted.py`, `test_scripted_doctrine.py`, `test_scripted_search.py` passes; the
  reader page `docs/explanation/scripted-agent.md` names both gates.

## The runs

`make.py` builds the candidates: r9m2 (`maneuvers-2/champion2.json`) with the Yongma's
doctrine given morning item 3's lead turn (two-circle, full stick, crossing gate 0.5 rad,
time gate t s with scale min(0.1, t/2)) and `lead_turn_in_crowd` (0.5, 0.1); the Jujak flies
plain r9m2. Cards: balance-d2, frozen (the ship files), armour-tank (as `r9m2-maps/make.py`
builds them). r9m2 in the same run is the control. `maneuvers-2/sizes.py`:

- main (09:00-09:11, 512 games a cell, 10v10 256): t = 0.2 / 0.3 / 0.5 s; self-play energy v
  turn at 1/2/5/10, the Yongma mirror at 1v1, and against r9m2 at 1v1 and 5v5 in four
  lineups. `sizes.json`, `sizes.table.md`, `log.txt`.
- confirm (09:11-09:24, 1024 games, 10v10 512): t = 0.2 / 0.3 s, the same cells.
  `confirm.json`, `confirm.table.md`, `confirm-log.txt`.
- last (09:24-09:27, **cut** at the 2v2 by the coordinator: the author was playtesting and
  the job took the GPU; only its 1v1 completed): t = 0.2 / 0.3 s with `lead_turn_in_others` (0.5, 0.1) as well, 512 games,
  against r9m2 also at 10v10. `last.json`, `last-log.txt`.

**Reading the cells.** The Jujak is plain r9m2 in every game, so the self-play energy v
turn and the energy v turn against r9m2 are the same game (the variant's Yongma against
r9m2's Jujak); `summary.py` pools them. At 1v1 the "mixed" lineup against r9m2 is the
Yongma mirror (the variant's Yongma against r9m2's); turn v energy and the 1v1 mixed
reversed (the Jujak mirror) have plain r9m2 on both sides. Strength in crowds is the 5v5
(and 10v10) mixed / mixed reversed against r9m2. The run-to-run spread is about twice the
binomial SE (main and confirm controls differ by up to 0.03 at 10v10), so single cells of
±0.02 are noise; the pooled pattern across cards and runs is what counts.

## Results

Pooled Yongma energy v turn score (self-play and against r9m2 together; the change from the
same run's r9m2). 1v1 on frozen is about 0.01 for every doctrine (the Yongma cannot win the
frozen card's lone duel) and is left out.

| Card | t | 1v1 main / confirm / last | 2v2 main / confirm | 5v5 main / confirm | 10v10 main / confirm |
|---|---|---|---|---|---|
| D2 | r9m2 | 0.485 / 0.491 / 0.493 | 0.453 / 0.454 | 0.567 / 0.591 | 0.588 / 0.556 |
| D2 | 0.2 s | +0.016 / +0.022 / +0.023 | -0.026 / -0.006 | +0.015 / -0.013 | -0.070 / +0.007 |
| D2 | 0.3 s | +0.032 / +0.018 / +0.014 | -0.025 / -0.030 | +0.019 / -0.014 | -0.051 / -0.023 |
| D2 | 0.5 s | +0.021 | -0.018 | -0.000 | -0.074 |
| armour-tank | r9m2 | 0.508 / 0.510 / 0.503 | 0.397 / 0.414 | 0.569 / 0.572 | 0.525 / 0.481 |
| armour-tank | 0.2 s | +0.028 / +0.015 / +0.020 | -0.010 / +0.008 | -0.006 / -0.003 | -0.055 / -0.004 |
| armour-tank | 0.3 s | +0.020 / +0.023 / +0.018 | +0.023 / +0.025 | -0.005 / -0.017 | -0.057 / +0.003 |
| armour-tank | 0.5 s | +0.040 | +0.026 | -0.006 | -0.066 |
| frozen | r9m2 | 0.011 / 0.007 / 0.008 | 0.419 / 0.438 | 0.320 / 0.351 | 0.256 / 0.214 |
| frozen | 0.2 s | +0.000 / +0.003 / -0.000 | +0.004 / -0.004 | +0.024 / -0.022 | -0.029 / +0.004 |
| frozen | 0.3 s | +0.000 / +0.003 / +0.001 | +0.045 / -0.018 | -0.017 / -0.038 | -0.051 / -0.008 |
| frozen | 0.5 s | -0.003 | +0.039 | +0.009 | -0.049 |

(The "last" column is the stricter `lead_turn_in_others` variant; at 1v1 there is no other
enemy, so it is the same doctrine and a third replicate.)

The Yongma mirror at 1v1: the variant's Yongma against r9m2's (score; r9m2 against itself
in the same run is the control, 0.479-0.526), and both sides with the variant (mutual
head-on kill share, life, merges a kill):

| Card | t | head-to-head main / confirm / last | mirror (both variant) main; confirm |
|---|---|---|---|
| D2 | r9m2 | 0.516 / 0.526 / 0.498 | 0.62, 190 s, 2.02; 0.63, 186 s, 2.02 |
| D2 | 0.2 s | 0.520 / 0.472 / 0.497 | 0.59, 191 s, 2.00; 0.62, 188 s, 1.97 |
| D2 | 0.3 s | 0.504 / 0.485 / 0.506 | 0.62, 190 s, 2.04; 0.61, 191 s, 2.04 |
| D2 | 0.5 s | 0.472 | 0.64, 200 s, 2.00 |
| frozen | r9m2 | 0.512 / 0.507 / 0.493 | 0.62, 292 s, 3.00; 0.61, 275 s, 3.01 |
| frozen | 0.2 s | 0.490 / 0.489 / 0.468 | 0.56, 291 s, 3.02; 0.60, 292 s, 3.12 |
| frozen | 0.3 s | 0.458 / 0.493 / 0.497 | 0.60, 284 s, 3.02; 0.60, 302 s, 3.06 |
| frozen | 0.5 s | 0.447 | 0.55, 318 s, 2.79 |
| armour-tank | r9m2 | 0.479 / 0.510 / 0.479 | 0.34, 221 s, 2.17; 0.35, 214 s, 2.14 |
| armour-tank | 0.2 s | 0.499 / 0.478 / 0.470 | 0.36, 215 s, 2.08; 0.33, 222 s, 2.07 |
| armour-tank | 0.3 s | 0.492 / 0.468 / 0.482 | 0.33, 233 s, 2.14; 0.34, 231 s, 2.10 |
| armour-tank | 0.5 s | 0.476 | 0.33, 219 s, 2.01 |

Strength in crowds: the variant against r9m2 at 5v5, mixed / mixed reversed pooled, as Elo
(SE ±10 main, ±7 confirm). Controls (r9m2 against itself): main +7 / +1 / -24, confirm +1 /
+11 / +4 (D2 / frozen / armour-tank).

| t | D2 main / confirm | frozen main / confirm | armour-tank main / confirm |
|---|---|---|---|
| 0.2 s | +21 / -18 | -3 / -15 | -1 / -7 |
| 0.3 s | +0 / -12 | -1 / +6 | +0 / -1 |
| 0.5 s | -8 | -1 | -2 |

The full four-lineup cells are in `sizes.table.md` and `confirm.table.md`, the pooled ones
in `summary.md` (`summary.py sizes.json confirm.json last.json`).

## Findings

- **The lone-duel lead turn gives the Yongma a small, repeated gain in the energy v turn
  1v1**: +0.015 to +0.03 (about +10 to +20 Elo) on D2 and armour-tank in all three runs,
  at every t. On frozen the Yongma loses the duel 99:1 whatever it does. This is smaller
  than morning item 3's tilt with both types turning (D2 0.505 to 0.54-0.57): part of that
  tilt was the Jujak's own lead turn costing the Jujak.
- **Nothing beyond 1v1.** At 2v2 / 5v5 / 10v10 the changes are -0.04 to +0.045 with no
  consistent sign across cards and runs. The main run's 10v10 drop (-0.03 to -0.07 on every
  card, 256 games) did not repeat at 512 games in the confirm (-0.023 to +0.007); the main
  run's 10v10 controls were high.
- **It loses the Yongma mirror.** Against a plain r9m2 Yongma, the lead-turning Yongma scores
  0.447-0.506 (the controls 0.479-0.526): pooled over the runs and cards, about -0.025
  (-17 Elo) at 0.2 and 0.3 s, worse at 0.5 s (0.447-0.476). With both sides lead-turning,
  the mirror's shape is unchanged: mutual head-on kill share within 0.03 (a little lower at
  0.5 s on frozen), life and merges a kill within noise.
- **Crowds: no clear loss, but not clean at 0.2 s.** 5v5 mixed against r9m2 at 0.3 s is
  -12 to +6 (controls -24 to +11); at 0.2 s the confirm gives -18 / -15 / -7 against
  controls +1 / +11 / +4 (pooled about -15 ± 4 net), while the main run gave +21 / -3 / -1.
  The gate keeps the lead turn out of most crowd merges, but late-game lone duels inside a
  5v5 still fly it.
- **Best t: 0.3 s** (1v1 +0.014 to +0.032 in all six D2 and armour-tank cells, crowds
  neutral); 0.2 s gives the same 1v1 gain with the doubtful 5v5 confirm; 0.5 s costs the most
  in the Yongma mirror.

**Answer: no, not as it stands.** It helps the Yongma against the Jujak in the lone duel by
about +0.02 (+14 Elo), it is neutral in crowds at 0.3 s, but the same Yongma loses its own
mirror by about 0.025. As a balance knob it is weak (+0.02 where the r9m2 controls already
sit at 0.49-0.51 on D2 and armour-tank) and as strength it is a wash. Not recommended for
the stored doctrine; if a 1v1 tilt toward the Yongma is wanted, the gate could be narrowed
to cross-type duels (like `duel_extend_cross_in_others` / `sense_mirror_only`), which would
keep the 1v1 gain and remove the mirror loss; untested.

## Runs completed and cut

- main (512 games): complete.
- confirm (1024 games, 0.2 and 0.3 s): complete.
- last (the stricter last-enemy gate): **cut at 09:27** at the start of its 2v2 by the
  coordinator; the author was playtesting and got 5 fps. `queue.sh`'s play check matched
  only `python ... -m bnb play` and missed the `.venv/bin/bnb play` entry point. Fixed in
  `queue.sh` (pattern `(^|/)(python[0-9.]*( .*)? -m bnb|bnb) play`, checked against both
  forms and against its own shell's command text). Only its 1v1 (the same doctrine as the
  lone gate at 1v1) is in `last.json`. No further GPU jobs were started.
