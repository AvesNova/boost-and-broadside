# Author review 2 (2026-11-24)

Everything since the first review (`author-review.md`, gone through on the evening of
2026-11-23) that needs your check: decisions, the judgement calls made for you, and what was
done. Written from `core` at 45415027; updated on the evening of 2026-11-24 (section 2
rewritten for the day's flight and design work, S1's status, the order below). Numbers are
copied from the docs named beside them; none are new.

## How to use it

- Each section has three parts:
  - **Decisions for you** (S1, F1, L1, ...): answer them one by one.
  - **Judgement calls made for you**: one line each. Object to any; silence keeps it.
  - **Done, for information**: brief.
- Already decided (not asked again): everything in `core-decisions.md`, including the names
  (Old, BFM-default, BFM-design, BFM-CMA1, BFM-CMA2) and BFM-CMA2 as the provisional yardstick
  and BC teacher.
- Source shorthand: "screening" = `scripted-screening.md`, "divergence" =
  `scripted-divergence.md`, "yo-yo review" = `lead-turn-yo-yo-review.md`, "notes" =
  `morning-notes.md`, "worklog" = `core-worklog.md`. Paths without a folder are in
  `docs/rewrite/` or `lean/`.

## The decisions that matter most, in order

1. **F1** Fly the simplified P-51D (tonight's model and stick levels). Blocks: F3, L1, the
   clock measurement.
2. **S1** Confirm BFM-CMA2 as yardstick and teacher after flying it, with **F3** (a short
   re-check under the new stick levels). Blocks: the clock measurement, the BC dry run.
3. **S2** Lead turn and yo-yo: delete or keep. Blocks: BFM-CMA2's final vector.
4. **L1** Re-measure value supports, next-state ranges and hit shares on today's game.
   Blocks: the BC dry run.
5. **L2** The teacher term's schedule and its 0.45 gate. Blocks: reading the RL dry run.
6. **V3** One number to explain before quoting it (collect 2.5x faster at parity). Blocks:
   the dry runs' speed headline.
7. **S4** Freeze the 16 rows flat in skill and behaviour. Blocks: the persona arm, the next
   search.
8. **V1** Confirm the compile budget. Blocks: the tripwire as a hard check.
9. **S7** "Spending early wins": a balance question or the game's character? Blocks: nothing
   now; it shapes what BC learns.

---

## 1. Scripted agent

### Decisions for you

**S1. BFM-CMA2 as the yardstick and BC teacher (confirm after flying)**

- **What.** Provisional since about 12:30, pending your flying and the caveats below. It is
  the code default since 12:28 (BFM-default kept as `BFM_DEFAULT_2026_11_24`). Its numbers
  below predate tonight's stick levels (see F3).
- **Evidence** (screening, "CMA-ES round 2"; same code, fresh store, 9,216 games each):
  - Skill: BFM-CMA2 0.649, BFM-CMA1 0.623, BFM-default 0.529 (each +- 0.005).
  - Head to head against BFM-default: 0.625 (CMA1 0.591).
  - Held-out personas: 0.662 against 0.618 (CMA1) and 0.529 (default). Ahead of CMA1
    against all 8.
  - Plateau: its neighbourhood scores 0.651, paired +0.039 +- 0.002 over CMA1's.
  - Above BFM-default in all 18 conditions; above CMA1 in 14. The other 4 are within 0.013
    (error about 0.02 each).
- **Caveats.**
  - *It banks less store.* E2 was built so the teacher banks and cashes store, as you did
    to win a two-circle fight ("Scripted BFM" in core-decisions). The search goes the other
    way. Over the skill games: store 0.236 against 0.296 for default, climb share 0.071
    against 0.128 (screening, "Style").
  - *It flies further from today's behaviour.* JS divergence from default: 0.131 nats
    (CMA1 0.066, BFM-design 0.102), mostly in the power head: more dive, less climb.
  - *Two rows drifted.* yo_yo_time 0.19 (most likely drift on a row that is now flat; not
    isolated) and lead_turn_scale 0.043.
  - *The search did not converge.* Its step grew from 0.15 to 0.26. The plateau evidence is
    the neighbourhood, not convergence.
- **Options.** (a) BFM-CMA2. (b) BFM-CMA1 (closer to today's behaviour, 0.026 weaker).
  (c) BFM-default. (d) BFM-CMA2 with lead_turn_scale and yo_yo_time set to 0, re-confirmed
  in one run (round 1's confirmation took 4.8 min of GPU).
- **Recommendation.** (d) if S2 deletes or turns off those features, else (a). Fly it first.
  The flying list is in `scripted-bfm-design.md` ("Flying list"): the two-circle fight you won
  (does it bank and cash?), one-circle, neutral passes, being chased, leaving and coming
  back, head-on.
- **Blocks.** The clock measurement (scripted against scripted), the notes refresh, the BC
  dry run.

**S2. The lead turn and the yo-yo: delete or keep**

- **What.** The screening found both lowering skill at every setting above 0. The review
  found both built wrong in the same way: inside gun range they took the nose off the gun.
  The high yo-yo also had no aspect condition, so it lagged at the beam. Both are fixed: the
  gun owns the nose in range, and the high yo-yo lags only with him in my rear quarter.
- **After the fix** (yo-yo review, "Re-measured sweep", +- 0.015 a point, BFM-default 0.551
  in the same store):
  - Yo-yo: flat within noise (0.544 at the box top of 0.5).
  - Lead turn: at scale 1.0 the drop is now 0.08 (was 0.18). The rest is in sides spawns:
    0.439 against 0.564 there, 0.510 against 0.538 in mixed.
  - Neither gains anything, before or after the fix, and nor did a lead turn rewritten as a
    timed turn.
- **Cost.** The lead turn is 26 operations per call (notes, "E2b operations").
- **Options.**
  - (a) Delete the lead turn; keep the fixed yo-yo pursuit half as a persona trait (the
    reviewer's recommendation).
  - (b) Delete both.
  - (c) Keep both, off in the yardstick.
  - (d) Rebuild the lead turn together with the pre-merge offset by plan (not built).
- **Recommendation.** (a); (d) goes to the later list. BFM-CMA2 holds lead_turn_scale 0.043,
  so (a) changes it slightly (see S1 (d)).
- **Blocks.** BFM-CMA2's final vector; the persona box.

**S3. The yo-yo's closure signal saturates**

- **What.** The wanted closure is about 0.8 v_c at gun range against a scale of 0.1 v_c, so
  the yo-yo signal y sits near -1 almost everywhere (mean -0.97 over the low yo-yo's ticks
  in brawls; yo-yo review Y3). y also sets the yardstick's speed target and stick cap: the
  energy half of the yo-yo, which every parameter set flies.
- **Options.** Fix it now (this changes the yardstick, so a new search or confirmation is
  needed); fix it with the next search; leave it.
- **Recommendation.** Fix it with the next search, not before the dry runs (the reviewer).
  closing_time and nose_tail were tuned around the saturated signal, so fixing it means
  searching again.
- **Blocks.** Nothing now.

**S4. Freeze the rows that change neither skill nor behaviour**

- **What.** 22 of the 34 sampled rows show no skill effect (screening). Measured by behaviour
  as well (divergence), 16 are flat in both: turn, yo_yo_scale, power_scale, turning_floor,
  spend_range, relax_angle, spend_need, scissors_range, boundary_margin, switch_gain,
  defend_range, speed_floor, cash_angle, and the three overshoot rows.
- **Not in the 16.** engage_range changes behaviour clearly but not skill. level_three,
  merge_speed, weakest_weight, spend_share and speed_band change it a little.
- **Options.** (a) Freeze the 16: fixed at the default, no longer sampled by personas or
  searches. (b) Delete them as rows. (c) Keep sampling them.
- **Recommendation.** (a) (divergence). Keep engage_range for personas: variety at no skill
  cost. Freeze power_scale with temperature in mind: the two interact. Freezing keeps every
  agent's id. Deleting rows changes every id, because an id is the full vector with its
  row names (notes, V).
- **Blocks.** The persona arm; the next search's dimensions.

**S5. The overshoot rule almost never fires**

- **What.** It acts on 5 of 98,014 stored ship-states. Its track-crossing gate (under 1 rad)
  passes on 0.005% of states (divergence). Shaw gives no number for that gate (notes, E2b).
  The rule is the reversal in the design's flying list ("overshoot it fast and close and it
  reverses").
- **Options.** Widen the gate and re-measure; remove the rule; leave it.
- **Recommendation.** Try one widened gate in the S6 ablation pass. Remove the rule if it
  still does nothing. Check it when you fly the "being chased" case.
- **Blocks.** Nothing.

**S6. Trim the scripted call (ablate whole flat mechanisms)**

- **What.** The call is 433 operations for both teams after the fix (yo-yo review). It was
  208 before E2, the design estimated about 270, and the sign-off was 188. Compile cost
  (scoreboard, "Compile time"): E2 added 332 traced operations to act and +3.9 s cold,
  within noise. So it is not a regression at the budget.
- **Candidates** (every row flat; screening, "Recommendations"): the overshoot rule, the
  store-spending gates, the weakest-target choice, the plan switch on gain, the scissors
  slow-down, the two-circle cash, the break relaxation. Each needs one ablation set (the
  yardstick without it, against the panel) before deletion.
- **Options.** Run the ablation pass now; after the dry runs; never.
- **Recommendation.** After the dry runs, before the first full BC run. Flat mechanisms
  change the teacher little, and the compile cost is small.
- **Blocks.** Nothing before the dry runs.

**S7. "Spending early wins": balance or character?**

- **What.** The same signal keeps appearing:
  - In 1v1 equal starts the Old agent, which dives its store away at once, beat the BFM
    agent almost every time. BFM against BFM, the side with more energy wins nearly every
    game (notes, E2b).
  - Both CMA-ES rounds moved toward less climb and less store (S1).
  - Before BFM, scripted brawls flew slow (mean 0.70 corner speeds) on a nearly empty store
    (mean 0.05), because the store is regained only by climbing, which costs the speed they
    lack (notes, S finding).
  - The climb-and-dive round trip loses 4% of what it banks at corner speed, 12% at the best
    sustained speed (`scripted-bfm-design.md` Q7, kept as intended).
- **Against it:** you won a two-circle fight by banking and cashing, so banking can pay for
  a human.
- **Options.** Accept it as the game's character; a small balance study later (store size,
  exchange rate, round-trip loss); change a knob now.
- **Recommendation.** No change before the dry runs; a study on the later list.
- **Blocks.** Nothing now. It shapes what BC learns.

### Judgement calls made for you

E2 design questions, answered provisionally (`scripted-bfm-design.md`, "Open questions"):
- The best sustained speed is a speed reference beside corner and stall speed.
- Two-circle is flown at full stick with a speed target; the sustained level is used only to
  bank speed above 1.195 v_c.
- One-circle or two-circle by energy advantage, with a threshold per persona (not a fixed
  preference).
- **A climb is not quiet:** the disengaged regain climb restarts the shield wait.
- A memory field for the committed sense; the drawn head-on side stays.
- Temperature does not blunt the gun; it blunts decisions only.
- The round-trip energy loss is kept (physics as intended).
- Under fog his nose rate and store are from his last sighting (no dead reckoning).

E2 build (notes, "E2b"):
- One temperature per persona (it was one per gate). It divides the decision gates and the
  power law, not the gun.
- Dive only with a purpose (offensive, turning, defensive), never disengaged or at the merge.
  The climb fades over the store's top band, as the dive does over the bottom.
- Spending needs a turn worth the store (`spend_need`, 0.1 rad) and its own range row.
- Quadrant gates are log gates with a fixed scale; decisiveness varies through temperature.
- The low yo-yo raises the speed target only against a turning target. Shaw's AOT 30-60 deg
  condition on the high yo-yo is not built.
- The two-circle share is the measured flow, not the memory's plan. Two-circle banks at the
  sustained level until his advantage reaches 90 deg. One-circle slows to the floor inside
  one sustained radius.
- **Overshoot:** reverse only if I am no faster than him (Shaw p. 76; the brief had it
  backwards). Thresholds 0.67 R_st, crossing above the sustained rate, track-crossing under
  1 rad (no Shaw number). The reversal is flown level.
- **Plan switch:** a pass gaining under 10 deg (Shaw p. 107) switches the plan at the next
  commit. In mirror fights this alternates every 4 to 5 s; Shaw's "one or two turns" would
  need a counter.
- Hysteresis: release only 0.2 rad forward of the 3/9 line; a dead-astern tie (within 0.1 rad)
  takes the drawn side.
- The committed sense yields to the boundary steer as soon as the steer is active.
- The competence bound was lowered from 0.98 to 0.97 of long-lived ships firing: BFM
  defenders survive longer without a firing solution.
- BFM-default's values came from ablations against Old (pursuit point = the gun's lead point
  moved along his turn; yo-yo 0; lead turn 0; break whenever defensive within 3 gun ranges).
  Superseded by the CMA rounds.

Lead-turn and yo-yo fix (yo-yo review):
- The high yo-yo's rear-quarter gate (centre pi/3, scale 0.1) is a module constant, not a
  table row, so no agent's id changes. It could become a fixed row when the panel is next
  rebuilt.

Search set-up (screening):
- **Skill measure:** mean score against a panel of 8 (BFM-default, BFM-design, 6 seeded box
  personas) over 18 conditions (arena 3,000 / 4,200 / 6,000 m; 3v3 / 4v4 / 5v5; mixed or
  sides spawn), equal weight per cell. Fog and fields at the defaults, 5,400-tick clock.
- Round 2's nose_tail floor is 0.25 R_st (56 m). The game has no collisions, so Shaw's 500 ft
  debris minimum does not apply; a bullet still needs only 0.027 s at that range.

Power rule and memory (step S, notes):
- Dive below 1.8 stall speeds, or below 0.95 corner speeds in a hard turn. Climb above 1.2
  corner speeds, within 0.21 gun ranges, or on half the ticks when far (over 3 gun ranges)
  with the store under 0.3. Untuned; sharp speed gates.
- A slow ship never climbs, and dives only with store to spend; otherwise it coasts.
- A power floor of 0.02 on coast, dive and climb (replacing a 0.04 per gate that compounded).
- "Far from every enemy" means the kept target's distance. With no known enemy (fog) the
  ship counts as far.
- Memory is required everywhere (act returns action and memory). Collect keeps one memory per
  ship, serving both the BC teacher and the league opponent.
- **Persona box:** independent uniform draws (log-uniform centres, temperature e^+-0.4).
  Spreads above 1 are refused. The competence test is at spread 1.
- The lead in a field uses proper velocities; the gun's shorter reach inside a field is not
  modelled.

### Done, for information

- E2 built natively; reviewed against Shaw (`shaw-bfm-notes.md`). Not built from Shaw:
  jinks, the high yo-yo's AOT condition, keeping two-circle after a big bite, merge offset
  and speed by plan, no banking close to a higher-energy opponent.
- Screening: Morris, 4,000 Sobol configurations, a surrogate (R^2 0.862 against a noise
  ceiling of 0.904), local spreads, sweeps. Rankings hold across arena, team size and spawn
  (noise-corrected correlations 0.97 to 1.0), so no retuning per map is indicated. 1.56M
  games stored.
- CMA-ES rounds 1 and 2 (about 33 and 34 min of GPU).
- Divergence: 98,014 stored ship-states; behaviour is one continuum, no families. Axis 1 is
  the power balance (fight_speed), axis 2 the aim (lead_gain).
- The lead-turn and yo-yo review and its fix (BFM-default unchanged by the fix).

---

## 2. Game, flight and aircraft

Rewritten on the evening of 2026-11-24. Since the first version, the flight model and the ship
design system were reworked with you during the day (`core-decisions.md`: "Stick levels", "Ship
design", "Design profiles", "Simpler flight and design"). Merged at df4b8138.

### Decisions for you

**F1. Fly the simplified P-51D**

- **What.** Tonight's model, being merged now:
  - Design inputs: stall speed, g limit, top level speed, best climb, span efficiency e 0.85,
    separation drag 0.3 at exponent 8 (a gentle rise near max lift, not a wall). Only the
    zero-lift drag and the power are solved.
  - Readouts: best-climb speed, best sustained turn, corner, gap. There is no gap target and
    no hidden sustained-rate target any more.
  - Flaps removed. Thrust lapse tried and removed.
  - Stick levels gradual 1/3, middle 2/3, full 1. Keys: A or D middle, Shift full, Ctrl
    gradual.
- **Measured** (tuning.performance and implied; before = the documents row at 0ec71cd8):

  | | before | now |
  |---|---|---|
  | top speed, best climb | 375 mph, 3,600 ft/min | same |
  | best-climb speed | 165.0 mph (input) | 167.75 mph (readout) |
  | corner speed and rate | 286.1 mph, 33.55 deg/s | same |
  | best sustained turn | 18.99 deg/s at 181.6 mph | 17.85 deg/s at 172.6 mph |
  | tightest sustained speed, gap | 154.4 mph, 0.850 | 152.9 mph, 0.886 |
  | e, propeller efficiency, C_D0 | 0.965, 0.859, 0.0158 | 0.850, 0.875, 0.0160 |
  | stick gradual / middle / full | 0.44 / 0.88 / 1 | 1/3 / 2/3 / 1 |
  | speed where middle reaches the g limit | 305 mph | 350 mph |

- **The trade-off to judge by flying.** With no separation the trial flew 19.45 deg/s at 166
  mph; separation 0.3 at exponent 8 takes it to 17.85. Your reference chart (forum
  reconstruction) shows about 19.7 deg/s over 165-190 mph. Lower separation (a slider) moves
  it back up; the doghouse-shape investigation explains why a propeller aircraft's best
  sustained turn otherwise sits on the lift line.
- **Ctrl and W.** Ctrl + W closes the browser tab and a page cannot block it, so a gradual
  turn while diving with W closes the tab. The sandbox notes say to dive with Up instead;
  another modifier for gradual is the alternative.
- **Blocks.** F3, L1 (reads the final row), the clock measurement.

**F2. The 8 g limit is above the structure's 6.6 g at this mass**

- **What.** A plausibility warning fires at the default: the forum rule (64,000 lb-g at
  9,760 lb) gives 6.6 g. You set 8 g as a game choice.
- **Options.** (a) Keep 8 g and the warning. (b) Lower it towards 6.6 g.
- **Recommendation.** (a): it is a game choice, and the warning only reports it.
- **Blocks.** Nothing.

**F3. The scripted agents under the new stick levels**

- **What.** BFM-CMA2 (the provisional yardstick) and the other BFM vectors were tuned when
  "sustained" was derived through the best sustained point. Their turn levels and the speed
  that caps them (`middle_stick_speed`) move with the thirds.
- **Options.** (a) One short confirmation run after the merge: BFM-CMA2 against BFM-default
  and the personas under the new levels (round 1's took 4.8 min of GPU). (b) Re-run CMA-ES.
  (c) Nothing until the clock measurement.
- **Recommendation.** (a). (b) only if the ranking changes.
- **Blocks.** S1's confirmation, the clock measurement.

**F4. The corner rate under the soft corner**

- **What.** The soft minimum of lift against the g limit (exponent 16) keeps full stick a few
  percent below the hard limit near the corner (8% at the old model: 32.2 against 35.0
  deg/s). It will be re-measured on the new row.
- **Options.** (a) Accept. (b) A harder soft minimum (less smooth).
- **Recommendation.** (a).
- **Blocks.** Nothing.

### Judgement calls made for you

P-51D:
- Wing 21.67 m^2 (NACA), C_Lmax 1.6 (AAF power-on turn value), AR 5.83.
- **Separation 0.3 at exponent 8**: about +30% lift-dependent drag at max lift, rising from
  about 0.8 of max lift. "A little, physically realistic" was your brief; the number is mine.
  It is a slider.
- Thrust softness 2.83, the value that gave the flight test's 165 mph best-climb speed before
  that speed became a readout.
- Drag rise: doubling at M 0.77 with exponent 20 (fitted by eye to NACA ACR 4K02); invisible
  at game speeds.
- The drag check's proposals are superseded. C_D0 and the propeller efficiency are now solved
  from the top speed and climb, and e is your 0.85. Still open in `p51d-drag-check.md`, not
  acted on: sea-level climb power 1,660-1,780 hp (1,730 best supported), and a
  speed-dependent propeller efficiency.
- The guns, shield and nose values are Vizia's (`tuning.GAME_FIELDS`).

Game rules (G2, notes):
- **Spawn edge rule:** within 1,000 m of the boundary, an outward spawn heading is mirrored
  in the tangent. Both spawn rules use it.
- "Mixed" keeps v2's 0.85 R disc. "Sides" is a half disc of 0.80 R per team, its flat edge
  0.05 R (210 m) from the dividing line, noses within 15 deg of the axis.
- **Training spawn speed** runs from the stall speed to the top level speed. The one-eighth
  standard spawn is a per-environment coin (one eighth in expectation, not exact).
- Clock tie-break compares total shield in points; a lead within 1e-5 counts as equal.
- Pending action: a 42-way index (42 = none), read by the model as an embedding.
- Evaluation forces the training spawn off.
- Feature scales: speed and velocity symlog over 200 m/s, nose rate symlog over 1 rad/s,
  shield over 100.
- The sandbox flies the real game, fog and fields included (`--set fog=False --set
  field_count=0` for pure flight). Its opponent is the scripted agent by default.

### Done, for information

- G1, G2: the core game is the only game; P-51D the only row; clock tie-break, spawn rules,
  training spawn, one-tick action delay.
- Full stick is exactly the envelope's top line (no blending tax); only the true corner is
  soft.
- Ship design (option C): one `design_row` gives the card and the tool. The card row is
  derived at load from the design inputs and cached. Plausibility warnings, implied physics
  and solved coefficients are shown in the tool.
- Design profiles were tried (documents, performance points), then reduced to one table
  tonight. The table makes an input swap a few lines.
- Doghouse-shape investigation (`doghouse-shape-investigation.md`): the old separation wall
  made the Ps contours run along the lift line. With a parabolic polar, a propeller cannot
  produce a gap. A jet-like thrust lapse only opens a gap with a draggy, low-e wing (measured
  tonight, then removed).
- Web tool:
  - matplotlib charts drawn by the server;
  - a legend with line samples;
  - lines only where the game flies them;
  - flown checks of the critical points;
  - three unit systems;
  - presets (save, load, delete; they store the solved row);
  - a warm-started solve (0.1-0.4 s);
  - a progress line, fail-fast with the unmet inputs named, and cancellation (sliders,
    reset, Ctrl-C).
- The committed doghouse PNGs predate tonight's model: re-render after the merge.

---

## 3. Training and league

### Decisions for you

**L1. Re-measure the measured ranges before the BC dry run**

- **What.** Three sets of numbers were measured on 2026-11-23, before A1 changed the P-51D
  (flaps, solved terms) and before E2 and BFM-CMA2 changed the teacher:
  - the value supports (`learn/config.py`: damage +-7, kill_death +-8, outcome +-12);
  - the next-state ranges;
  - the hit-class shares (94.57 / 4.44 / 0.99%).
  `model/NOTES.md` says to measure again "when the flight numbers or the lineup change".
- **Options.** (a) One scripted collection with the chosen yardstick on today's game, before
  the BC dry run. (b) Rely on the range warnings during the dry run.
- **Recommendation.** (a). It is cheap, and (b) only reports clamping after the fact.
- **Blocks.** The BC dry run.

**L2. The teacher term's schedule and its gate**

- **What.** RL's teacher term is 0.5 x 0.95^k at RL update k. It switches off for good at a
  win rate of at least 0.45 against the scripted agent, or once the weight falls below 0.005
  (update 90). Not measured (notes, L1).
- **The gate in the code.** RL starts a fresh league (`lean/train.py` `new_stage`). Its
  scripted tally reads 0 until a game ends. A BC clone close to its teacher may reach 0.45
  within the first few updates and switch the term off almost at once. So the term acts only
  while the clone is clearly weaker than the teacher. Is that what you want?
- **Literature** (`literature-recipe-1.md`):
  - Kickstarting: "it is important to reduce this weight quickly"; a constant weight
    plateaus at the teacher.
  - VPT: 0.2 decayed by 0.9995 per iteration.
  - Wolczyk: 0.5 decayed by 0.99998 per step.
  - None tests a gate like this one.
- **Options.** (a) Keep as built. (b) Ignore the gate for a minimum number of updates.
  (c) Drop the gate and rely on the decay. (d) A different decay.
- **Recommendation.** (a) for the dry runs. Read `scripted_win_rate` and `teacher_weight` in
  the RL dry run's log, then decide.
- **Blocks.** Reading the RL dry run, not its start.

### Judgement calls made for you

Learner (L1, notes):
- Fused Adam on every device; a non-finite step is skipped without a host read.
- The warm-up counts attempted Adam steps. RL restarts its optimiser and warm-up at the
  hand-off, as v2 did.
- **BC hands its EMA to RL.** The uniform average includes the stage's early weights, so it
  is kept and saved but is not the default.
- BC: coefficient 1.0 on the teacher cross-entropy, no entropy bonus. During BC the league
  holds only the scripted agent.
- **The entropy bonus 0.003 stays on during the teacher phase** (VPT dropped it there).
- **BC uses RL's 32 minibatches;** v2's BC ran 256 (8x the Adam steps per update). A setting
  can be added if BC proves step-limited. BC is 96 updates, about 100M ticks.
- The teacher is stored as 11 marginals per ship (float16); the joint is built in the loss.
  `bc_kl` is logged beside `bc`.
- A level's advantage share is Cov(A_level, A) / Var(A). The shares sum to 1 and can be
  negative.
- KL monitor: a warning on any epoch's mean above 0.02. In the 3-update GPU check it fired on
  every update after the first (0.09 to 0.34): real early-BC movement (entropy 3.74 to 2.62),
  not a bug.
- `python -m lean.eval` defaults to the EMA weights, size 4, 64 games per match.

League (L2 and P, notes):
- **Capacity rule:** a draw that lands on a full member is redrawn once over the members with
  room; the rest fill free rows in member order. This is exact PFSP when no member is full.
- Capacity is ceil(1.25 L / members present), the scripted agent included. Shapes change
  only between segments.
- The averages join with the first snapshot (before it every league game is against the
  scripted agent). Their slots take the new averages after every update.
- **Snapshot retention:** the newest 8 plus the one nearest each age 2^k back, **capped at
  12**. On v2's schedule that is at most 11 to 500M steps, and the oldest (5M) is always
  kept as an anchor.
- Snapshot schedule: v2's exactly, in RL environment steps. A snapshot is the live weights
  after the update that crosses the step.
- A game against a dropped snapshot finishes in its retiring slot but is not tallied.
- 4 retiring slots. With none free, the games of the retiring slot with the fewest games are
  re-dealt and counted.
- Checkpoints hold the games in progress and the random states, so a resume continues the
  same games. Snapshots are written once and referenced (P): about 140 MB plus 28 MB per
  snapshot at parity, about 7.7 GB per 500M steps (arithmetic in the worklog).
- `--resume` takes settings and plan from the checkpoint; only `--updates` can change.
- Routing is one compiled call per network slot on its own rows. The vmap form was faster but
  recompiled at every new K and W (10.5 against 12.8 ms a tick at parity, 8 slots).

Model (M, notes):
- Rotary: 22 of 32 pairs (68.75%), just under the decided 70-75%; reaching 72% needs an 11th
  band or a third heading harmonic. Band turn 55.62 deg (golden section).
- The rotary reach is a setting (16,800 m), so a larger map changes one number.
- A recurrence "starts from zero" after a reset, after a masked tick, or at tick 0. A
  never-seen enemy coming into view starts that way too.
- The shield target is the change minus the recharge the game gave (exactly the damage
  absorbed).
- **Next-state ranges** are linear columns just outside the measured 0.1st and 99.9th
  percentiles. The shield is symlog from -90 to +0.3.
- Hit shares are measured over every bullet that ended, ghosts included.
- The pending action is an embedding of 43 rows (padding 42) inside the ship's first
  projection.

### Done, for information

- L1: BC stage, then RL with the teacher term; EMA and uniform averages; Adam settings;
  warm-up; KL monitor; advantage scaling; test stand-ins removed.
- L2: capacity routing, retiring slots, v2's snapshot schedule and retention, checkpoints
  every 10M steps, resume. P: snapshots written once.
- M: rotary, initialisations, next-state head and pending-action input as decided.
  2,155,565 parameters.
- Parity row (2026-11-24 08:21): loop 6,559 steps/s, peak 5.76 / 6.40 GiB, first update 380 s
  cold. The cache now lives in ~/.cache/bnb-inductor, so it survives a reboot.
- Review round 1 bug B1 (retiring slots stopped after one tick) is fixed, with a test.
- Tabled (your call, 2026-11-24): the cheese agents.

---

## 4. Evaluation and tooling

### Decisions for you

**V1. Confirm the compile budget**

- **What** (proposed in step K; scoreboard, "Compile time"):
  - The tripwire's miniature (`compile_check.py`) fails above 260 s cold. Baseline: 221.3 s,
    4 graphs, 5,721 traced operations.
  - It also fails on traced operations more than 10% over the baseline per function, and on
    any new graph, recompile or graph break.
  - A cold time 25% over the baseline only warns.
  - Time is compared only on the baseline's GPU and power profile.
- **Latest runs.** 224.7 s, quiet (worklog 11:21). 430 s at a load average of 40 (yo-yo
  review). The full first update at parity: 380 s cold.
- **Options.** Confirm; a tighter or looser budget.
- **Recommendation.** Confirm. Check the time part only at the careful measurements (the
  measuring policy of 2026-11-24); operations and graphs are checked at every hot-path merge.
- **Blocks.** The tripwire as a hard check.

**V2. Game versions and agent ids hash the source bytes**

- **What.** The game version hashes the bytes of lean/game, lean/scripted and contract.py.
  A network's id adds the bytes of lean/model and eval/runner.py (`eval/store.py`). Any
  edit there starts new conditions or ids, and old games stay in the file but are not mixed
  in. That includes a comment, a docstring, or making BFM-CMA2 the default.
- **Consequence.** The 1.56M screening games and the CMA-ES stores stop counting for new work
  at the next such edit. The style counters already gave networks new ids.
- **Options.** (a) Keep it strict. (b) Hash the code with comments and docstrings stripped.
  (c) A version constant bumped by hand on behavioural changes.
- **Recommendation.** (a) for now (safe; references are cheap to replay). (b) is a small
  later improvement. That is my view; no source weighs it.
- **Blocks.** Nothing.

**V3. Two numbers to explain before quoting them**

- **Collect 2.5x faster.** At parity, collect went from the 2026-11-23 core-game row to
  49,796 steps/s, loop +17%. The cause was not isolated (scoreboard, 2026-11-24 row). A jump
  that size can hide skipped work.
- **(Corrected 2026-11-24: the 0.4656 in the drain check was BFM-default against a variant with
  fight_speed 1.3, not against itself; the evaluation notes said "itself" by mistake. No seat
  asymmetry is indicated; the screening's self-play score was 0.517 over 576 games.)**
- **Recommendation.** Isolate the collect jump before the dry runs. It is a cheap check.
- **Blocks.** The dry runs' speed and score headlines.

### Judgement calls made for you

Runner and store (V, review fixes, notes):
- A runner call holds at most one network per seat. A network against a scripted agent
  always sits in seat A; jobs with more networks are split into calls.
- **The crossover now alternates sides** (the network was always team 0) and draws a
  position offset as training does (it was zero). Its games go into the store.
- **A scripted agent's id is its full vector with the row names,** so adding a row gives
  every agent a new id.
- A network's id hashes its checkpoint file plus the inference code. The analysis-only
  `game/tuning.py` is left out of the game version (the card rows are literals in
  `card.py`).
- Standard error from the score shrunk by half a game toward 0.5 (draws count half).
  Allocation is in lots of two (sides level) to the largest error.
- Bradley-Terry uses a Gaussian prior of 1000 Elo about the anchor (the yardstick at 1000),
  so unbeaten agents stay finite.
- Networks play in bf16 autocast on a GPU (training's precision).
- Fog off compiles a new graph; team sizes, arena, spawn and clock do not. A short queue uses
  16 games per slot (at least 256).
- The shared scripted gate tables no longer read the config's numbers, so persona values
  never recompile.

Style counters (`eval/NOTES.md`):
- **A turning fight** is the ship and its nearest living enemy both turning at least 0.25
  sustained rates. Two-circle means the same turn sense (the scripted agent's flow, with a
  hard floor); the rest is one-circle.
- The other counters: mean store fraction; speed in corner speeds; real shots per
  ship-minute; hits and range at hits (in gun ranges); climb and dive shares (power applied).
  Means are weighted by ship-minutes.
- They cost about 1-2% of scripted throughput (one clean pair).

Web tool (W):
- One FastAPI app: a Ships tab (sliders, live doghouse) and a plain Sandbox tab. The Tk
  sandbox is deleted.
- The tool loads Plotly from cdn.plot.ly, so its charts need a network connection.

### Done, for information

- V: the runner (rolling slots, unbiased drain), the SQLite store, adaptive allocation,
  Bradley-Terry Elo, CLI. Throughput: 810 scripted games/s; network against scripted 122k
  environment ticks/s.
- K: the "doubling" of compile time was two sizes compared. The real growth was a recompile
  in L2, now fixed (237 s at 256 environments, 4 graphs). Regional compilation does not trace
  in PyTorch 2.15 with dynamic shapes. A per-slot CUDA graph gave no gain at parity and was
  reverted.
- Review round 1 fixes: retiring slots, allocation with no games, the inference path in
  network ids, atomic checkpoints, the eval precision helper, `--start` needs `--stage rl`,
  one spawn path.
- Notes hygiene (D): one short NOTES.md per package, README and CONTRACT current,
  LAPTOP_PROMPT deleted.
- Carried open: the batched crossover's GPU gain over v1's walk is not shown; unequal team
  sizes with a network are untested in a full run.

---

## 5. Code health

### Decisions for you

**C1. The process-global ship card**

- **What.** tuning mutates the card module in place, so the web tool needs a lock and a
  child process per session (review round 1). Passing the card as data is a medium change.
- **Options.** Redesign now; after the dry runs; keep.
- **Recommendation.** After the dry runs (it touches the hot path's constants).

**C2. Line length** (carried from review 1, item 10.2)

- **What.** lean is written at 120 columns against the project's 100. `ruff check lean`
  gives 532 findings, 529 of them line length (`lean-review-2026-11-24.md`).
- **Options.** Reformat to 100; give lean/ its own 120 setting.
- **Recommendation.** Its own setting: one line, no churn. My view; no source.

**C3. Where `game/tuning.py` lives** (carried, 10.1)

- **What.** It is now the solver behind the card rows and the entry point for the web tool
  and the charts. It is excluded from the game version.
- **Options.** Keep it in lean/game; move it to an analysis folder.
- **Recommendation.** Keep it ("one source of truth for physics", decided 2026-11-24).

**C4. `core-brief.md`** (carried, 11.3; backlog F1)

- **What.** The plan document no longer matches core-decisions.
- **Options.** Rewrite it; mark it superseded by core-decisions.
- **Recommendation.** Mark it superseded (one line). core-decisions and the backlog already
  hold the plan.

**C5. Review 1's leftovers**

- **What.** Two items were not done (worklog 05:55):
  - one tick written three times (training's act keeps its direct policy call);
  - wide signatures (act returns an 11-tuple, advance takes 16 parameters, slot_actions 11).
- **Recommendation.** Leave them to the optional second review after the dry runs.

### Judgement calls made for you

- The KL-warning, measurement and compile rules above are recorded in core-decisions ("How
  much to measure").
- Analysis stores and scripts live in `docs/rewrite/sections/` and
  `~/avesnova/boost-and-broadside/data/scripted-screening/`.

### Done, for information

- `pytest lean -m "not slow"` passes at every merge recorded in the worklog.
- The compile tripwire stays within budget on a quiet machine (224.7 s).

---

## What is next, after your decisions

1. S2's change to BFM-CMA2 (or your choice), then F3's re-check under the new stick levels.
2. **Clock measurement (X).** Scripted against scripted wipe times in 4v4 elimination; set
   the clock so about 80% of games end before it. Context: the skill games averaged 1,252
   to 1,344 ticks against today's 5,400-tick clock (screening, "Style").
3. L1: re-measure the supports, next-state ranges and hit shares.
4. V3: explain the collect speed-up and the self-play score.
5. Notes refresh. Known stale lines:
   - `scripted/NOTES.md` says "the yardstick is still the weaker of the two agents" (Old is
     gone);
   - morning-notes' older entries use the old names;
   - the doghouse PNGs (after tonight's merge).
6. BC dry run.
7. RL dry run. Then decide L2 (the teacher gate) and whether the KL warning should become a
   stop (if it fires routinely).
8. Optional second independent review (R2), including C5.

## Later experiments (tabled or arms)

- Persona population arm. The divergence's rows to vary: fight_speed, lead_gain, in_range,
  closing_time, nose_tail, plan_threshold. The style knobs at little skill cost:
  engage_range, three_nine_line, temperature.
- Lead turn rebuilt with the pre-merge offset by plan. Shaw's other unbuilt items (S2, done
  list above).
- The yo-yo closure fix (S3) and the mechanism ablations (S6), with the next search.
- Energy-balance study (S7): store size, exchange rate, round-trip loss.
- Active subspaces, MAP-Elites, ship-space and joint searches.
- More aircraft: Fw 190 A-8, Bf 109 G-6, Spitfire IX, and a relative turning check.
- Cheese agents in the league. Retreat or extend (E4).
- Model arms: grid cells, per-level advantage normalisation, 512-tick sequences, a
  game-token outcome head.
- Weight averaging as an arm (uniform against EMA); a learning-rate cooldown once curves show
  the need.
- An ONNX export spike before choosing the game client.
