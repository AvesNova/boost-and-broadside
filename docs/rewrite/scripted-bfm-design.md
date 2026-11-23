# Scripted BFM: review of lean/scripted and a design

Step E2, part 1 (2026-11-23 night). A review of the scripted opponent as it is on `core`
(273e85e1) and a design for basic fighter manoeuvres (BFM) built into it. No code is
changed here. The author checks the design before it is built (core-backlog E2).

**Status (2026-11-24): built** on branch worktree-agent-aff8a23dae8e7f1b6 (E2 part 2), in the
order of B8, with Shaw's notes folded in (docs/rewrite/shaw-bfm-notes.md: the two-circle
energy plan, the mid-fight plan switch, the overshoot conditions, the lead turn's geometry,
the graded defence). What it is now: lean/scripted/NOTES.md. Departures from this design,
each a line in docs/rewrite/morning-notes.md ("E2b"): the pursuit point is the gun's lead
point moved tau along his turn (not his position), and the yardstick's yo-yo lag/lead and
lead turn are 0 and its defence breaks whenever defensive (personas draw the design's values)
because, measured against the old agent, each of them lost about 0.1 to 0.2 of the score;
the dive only with a purpose; a `last_pass` memory field for the plan switch; a
`sustained_stick_speed` reference. Operations 439 per call for both teams (estimate 270).
Measured: new against old 0.413 +- 0.019 (640 games, 4v4); the brawl's store 0.05 -> 0.23;
in 1v1 the old agent, spending its store at once, wins nearly every equal start.

### Flying list (sandbox, 2x; `--opponent scripted`)

- Two-circle, the fight you won: pass him with an offset and turn his way. Expect it to turn
  your way too when it has more energy (store plus speed), ease to the sustained level and
  climb (its store rises) while you are not gaining, then go full stick and dive when you
  get within about 90 deg of its tail. Bank more than it does and cash it: you should still
  win the angles.
- One-circle: pass and turn away from it. With less energy it turns nose to nose; inside one
  turn radius it slows toward 1.3 stall speeds (climb) and its scissors are reversals.
- Equal energy, repeated neutral passes: it switches between one- and two-circle at each
  pass that gained it less than 10 deg (it alternates; tell me if that looks erratic).
- Being chased: sit on its six inside about 3 gun ranges. It breaks into you at full stick
  and corner speed (dives with store) and holds that turn; overshoot it fast and close and it
  reverses (level); overshoot slowly or while it is the faster and it keeps turning.
- Chasing it out of range: it flies its gun's lead point; against your turn it dives toward
  corner speed (low yo-yo), against straight flight it coasts and lets the engine close.
- Leaving and coming back: fly away beyond 3 gun ranges. It does not dive to chase; above
  corner speed it climbs (store fills, the stick capped to the sustained level), so it
  returns with store.
- Head-on: it takes the head-on shot (no lead turn in the yardstick); a persona with lead
  turn starts turning toward your track about theta / rate before the pass.
- 2v2 is not flyable: the sandbox is 1v1 (team-mate and target-switch rules are covered by
  tests and the brawl numbers).

Sources: `lean/scripted/opponent.py` (line numbers below), `lean/scripted/NOTES.md`,
`lean/game/core.py`, `lean/game/tuning.py`, docs/rewrite/bfm-doctrine.md (the doctrine
notes, keys P-1289, P-1210, P-821, AAS, P-825 as there), and the Navy P-1210 yo-yo pages
fetched for this note (pp. 33 to 35, navyflightmanuals.tpub.com/P-1210). Shaw's *Fighter
Combat* could not be read (lending copy only, as in the doctrine notes); nothing here is
taken from it.

## The P-51D's numbers that matter

Game units (2x real time), point-mass P-51D, mid store, coasting unless stated. Computed with
`tuning.performance` and `tuning._evaluate` (scratch scripts, not kept).

| quantity | value | note |
|---|---|---|
| stall speed v_stall | 93 m/s | the wing holds 1 g |
| corner speed v_c | 263 m/s | best instantaneous rate 1.19 rad/s (68 deg/s), radius 222 m |
| best sustained turn | 0.650 rad/s (37.3 deg/s) at v_st = 151 m/s (0.575 v_c), radius R_st = 233 m | also the minimum sustained radius, at the same speed |
| time to turn 90 deg at the best sustained rate, t90 | 2.42 s | |
| full stick, coasting | settles at v_st | so the best sustained rate is flown with full stick, not the "sustained" level |
| sustained stick (0.70), coasting | settles at 181 m/s, 33.3 deg/s | it reaches the g limit only above v_c / sqrt(0.70) = 1.195 v_c |
| gun range | 637 m (2.7 R_st) | |
| full store | 34,900 m^2/s^2 | worth +110 m/s at corner speed |
| energy rate, full stick at corner | -7,500 m^2/s^2 per s | a full store holds corner speed for 4.7 s: about 140 deg more turn than the sustained rate gives in that time |
| dive at full stick | 107 m/s: 0.52 rad/s against 0.34 coasting (radius 204 m against 319 m); 151 m/s: 0.71 against 0.65 | the dive frees the 1 g on the wing; it matters most when slow |
| climb, straight | at corner speed loses 4% of the energy it banks; at v_st 12%; the dive gains about 1% | holding 2 g on the wing costs induced drag; banking is cheapest when fast |
| full stick and climb, below 1.0 v_c or above it at full stick | climbs nothing | the turn has the lift first; to bank speed in a turn, ease to the sustained level (no turn lost above 1.195 v_c) |

Two consequences shape the design. The store is worth a lot in a turning fight (140 deg of
turn for a full store), and the brawl's mean store of 0.05 means that the yardstick almost
never has it. And the radius differences are small (204 to 238 m below corner speed) while
the rate differences are large (0.65 to 1.19 rad/s), so in this game the one-circle against
two-circle choice is mostly a choice of speed and of when to spend the store.

## Part A: review

### A1. Cohesion

The stick is one coherent process: target (`engage`, 249-321), one aim point (the gun lead
point, 389-398), one wanted direction (the aim point blended with the centre steer, 402-416),
one error, three nested level gates and a side (446-462). The power is five rules stacked on
the same tick (470-479): dive when slow, or when below 0.95 v_c with P(at least the sustained
stick); else climb when fast, or close, or (half the ticks) far with a low store; combined by
noisy-OR. Fire is an independent product (483-484).

There is no model of the situation. The ship always flies lead pursuit at the gun lead point
(the stick's aim is `aim_x, aim_y`, 412-416, at every range up to the 4.78 gun range cap,
389). Whether a fight becomes one-circle or two-circle emerges from "turn the short way"
(452). The power rule does not know whether the ship is fighting, and the "dive below 0.95
v_c in a hard turn" rule (DIVE_IN_TURN, 470) drains the store whenever one exists: at the
brawl's 184 m/s (0.70 v_c) a dive empties a full store in 4.8 s, and refilling waits for "far
from every enemy" (474), which a 4v4 brawl rarely gives. That is the empty store in the
NOTES table.

### A2. Coupling between parameters

| coupling | where | effect |
|---|---|---|
| The stick gates feed the power rule | 470: `gate[DIVE_IN_TURN] * turn_above_one` | moving the turn or level-two centre changes how often the ship dives |
| One lead gain for pursuit and gunnery | 392 (flight time x `lead_gain`), used by both the stick (412) and the gun (398, 435) | a persona at lead gain 0.8 also aims its gun at the wrong point |
| Temperature divides all eleven sharpnesses | 169-173 | one draw softens stick, power and gun together; the gun's accuracy is a skill, the others decisiveness |
| Nested stick levels are products | 447-450 | P(full) = turn x level_two x level_three; each floor (0.02) compounds |
| Speed gates in two references | DIVE_SLOW in stall speeds (1.8), DIVE_IN_TURN and CLIMB_FAST in corner speeds (0.95, 1.2) | their order depends on the aircraft's v_stall / v_c (0.354 for the P-51D); at spread 1 a persona can have climb_fast at 0.80 v_c below dive_in_turn at 1.41 v_c, and dive wins by precedence (470, 475) |
| Keep factor and weak-target weight act on one score | 285-289 | the keep factor compares weighted scores, not distances, once the weight is not 0 |
| The regain share is a per-tick coin | 474 | "far and low" gives a 0.49 / 0.49 climb or coast label: dithering by design |
| Boundary margin, ramp low, ramp high | 403-408 | three numbers for one decision (when to turn back) |

### A3. Extensibility

A new gate touches seven places: `GATES` (26-38), the index unpacking (39-51), four parallel
positional tuples (centre, scale, rising, floor: 71-80), the quantity stack (424-439), and the
code that combines it. A new non-gate parameter touches five: the config field and its width
(88-120), `PERSONA_FIELDS` (123-130), the hand-indexed draw layout (`signed[..., 2 * gates +
3]`, 167-181), and both branches of the yardstick-or-persona read (363-370). Positional
parallel tuples and computed column offsets are where mistakes would hide.

### A4. Tunability under per-persona sampling

Each gate centre is drawn alone, factor 0.67 to 1.49. The stick gates are safe (their centres
are 7x and 5x apart). The power gates are not independent: their ordering matters and is in
mixed units (A2), and DIVE_SLOW alone at 1.49x (0.95 v_c) makes the ship dive whenever below
corner speed. Lead gain alone moves gun accuracy. The competence test (32 x 1,800 ticks at
spread 1: no ship past the boundary ramp, 49 in 50 fire) passes, but it checks staying in
bounds and firing, not flying a sensible energy fight.

### A5. Units

Decided: each parameter a multiple of one named reference of the ship's own performance.

| parameter | default | reference today | as decided? |
|---|---|---|---|
| turn, level two, level three centres | 0.0106, 0.0775, 0.386 | stall angle of attack (`/ c["stall"]`, 419) | no: a pointing error is a geometric angle; in radians 0.0029, 0.021, 0.104 (0.16, 1.2, 6.0 deg) |
| dive_slow | 1.8 | stall speeds | yes |
| dive_in_turn, climb_fast | 0.95, 1.2 | corner speeds | yes |
| climb_close | 0.21 | gun ranges | no: "too close" (overshoot) is a manoeuvre distance; 0.57 R_st |
| regain_far | 3.0 | gun ranges | yes (a threat distance: outside his weapons) |
| regain_low | 0.3 | store fraction | yes |
| aligned | 1.01 | the target's angular size | yes (an angle ratio) |
| in_range | 1.04 | gun ranges | yes |
| gate scales, floors, side floor, power floor, regain share | | log units, probabilities | dimensionless |
| head_on_band | 0.1 | radians | yes |
| target_keep_factor, weakest_weight | 0.8, 0 | ratios | yes |
| lead_cap_in_gun_range, lead_gain | 4.78, 1.0 | gun ranges, ratio | yes |
| boundary_margin_in_turn_radius | 2.28 | `speed^2 / max_acceleration` (403) | no: that is the g-limited radius at the current speed, half the real radius at 0.7 v_c (108 m against about 220 m); the margin shrinks to 246 m in a slow brawl |
| boundary ramp low, high | 0.0554, 0.52 | fractions of the margin | yes |
| target size clamp | 1, 45 deg | radians | yes |
| ally_clearance_in_hit_radii | 2.0 | the ally's hit radius | a game dimension, not one of the named references; it is the right one for a line of fire |

No parameter is a time or a rate. The game does not provide R_st, the best sustained rate,
v_st or t90: `_derived_characteristics` (core.py 209-240) derives cruise, stall, corner and
launch speeds and the gun range only.

### A6. Cost

208 compute operations per call for both teams (CPU, eager, regions rule), attributed by
source line (a dispatch-mode counter, scratch):

| part | lines | ops |
|---|---|---|
| target choice | 270-291 | 18 |
| target gather, distance | 296-303, 310 | 10 |
| ally in the line of fire | 304-307, 319 | 17 |
| fog view, map velocity | core 1325-1329, 226-227 | 10 |
| lead (with its signed angle) | 389-398 | 18 |
| boundary steer (with the stick's signed angle) | 402-416 | 45 |
| eleven gates in one logistic | 419-442 | 25 |
| stick | 446-461 | 19 |
| power | 470-478 | 27 |
| fire | 483-484 | 7 |
| sampling, memory | 506-509, 547 | 12 |

### Verdict

The stick path is one clean pipeline and the single stacked logistic is cheap (25 ops for
eleven gates). The power is a stack of rules with no notion of the fight, partly keyed to the
stick's gates, and it is what keeps the store empty. There is no situation model, so BFM
cannot be added as a rule here or there without stacking more of the same. Parameters are
coupled in four places (stick into power, one lead gain for pursuit and gun, one temperature
for skill and decisiveness, mixed speed units), three are not in the decided units, and
adding anything goes through positional parallel tuples. Keep the target, lead, gun,
line-of-fire, boundary and sampling parts; replace the power rule; add an assessment and a
tactic layer between the target and the stick; move every parameter into one named table.

## Part B: design

### B1. Structure

One pass per call, every stage tensors of shape (E, N), no data-dependent Python branch:

1. **Target** (kept): `engage`, extended to gather the target's heading, nose rate and store
   (the stack at 296-299 grows from 4 to 8 columns).
2. **Assess**: a `Situation` of dimensionless quantities (B2).
3. **Tactic**: five soft weights that sum to 1, and a committed turn sense in memory (B3).
4. **Commands**: the weights mix four scalars, never two directions: the pursuit time τ, the
   speed target v*, the stick cap, the turn sense (B4).
5. **Control laws**: stick from the pursuit point, power from v / v*, fire from the gun lead
   point (B5).
6. **Sample** (kept): one uniform per head.

Parameters live in one table of named rows (name, default, reference, sampling law, half
width). The yardstick reads floats from it, a persona reads (E, N) tensors under the same
names, so the two branches at 363-370 and the column arithmetic at 167-181 go. Gates become
named (centre, scale) pairs; the stacked logistic stays, built from the table.

### B2. Situation assessment

References, added to the game's derived characteristics (core.py `_derived_characteristics`,
by bisection with the game's own `_setting` and `_rates`, as `tuning.performance` already does
on a grid): v_st, the best sustained rate ω_st, R_st = v_st / ω_st, t90 = (π/2) / ω_st, and
A_store, the turn a full store buys at corner speed over the sustained rate
((ω_c - ω_st) x full store / |energy rate at corner, full stick|; 2.5 rad for the P-51D).

| quantity | definition | unit |
|---|---|---|
| range | r / R_st and r / gun range | manoeuvre, shooting |
| λ, angle off my nose to him | the error to his position | radians |
| α, aspect: angle off his tail to me | from his heading and the offset (in 2D, π minus his angle off nose to me) | radians |
| closure κ | -dr/dt from the relative velocity | corner speeds |
| line-of-sight rate | cross(offset, relative velocity) / r^2 | ω_st |
| turn sense, each ship | tanh(nose rate / (floor x ω_st)) | -1 right .. +1 left |
| flow f | my sense x his sense: +1 same rotation (two-circle), -1 opposite (one-circle), 0 one of us straight | P-1289 A-3 |
| energy advantage Δe | ((v^2 - v_t^2) / 2 + store - his store) / full store | store fractions |
| store | own store fraction | |

Under fog his heading, nose rate and store are his last sighting (enemy_view); no dead
reckoning (decided).

### B3. Tactic selection

Two logistic gates on cos λ and cos α, centred on the 3/9 line (doctrine's dividing line:
P-1289 5-5, 6-1; P-1210 p. 23), give four quadrants, and one gate on range gives engagement:

| tactic | weight | doctrine |
|---|---|---|
| disengaged | 1 - e (beyond the engage range, or no target) | climb to regain energy: P-1289 12-3 (extension), FM 1-15 16a |
| merge: he is ahead, I am ahead of him | e · front(λ) · (1 - rear(α)) | lead turn, set the flow: P-1289 7-3, 7-5; P-1210 p. 49, 55 |
| offensive: he is ahead, I am behind his 3/9 line | e · front(λ) · rear(α) | pursuit curves and yo-yos: P-1210 pp. 33-35; control zone P-1289 5-3 |
| turning: both behind each other's 3/9 line (after the pass) | e · (1 - front(λ)) · rear(α) | 1C or 2C by the flow: P-1289 5-6; P-1210 p. 52 |
| defensive: he is behind me, I am ahead of him | e · (1 - front(λ)) · (1 - rear(α)) | break into him, spend energy to deny the shot: P-1289 6-1, 7-2 |

The weights are smooth and need no hysteresis themselves, because they mix scalars. The one
discrete choice, which way round to turn when he is behind the 3/9 line, is committed in
memory by rule (decided: hysteresis by concrete rules; doctrine §6: change the turn only on a
listed trigger):

- **Commit**: when he moves aft of my 3/9 line (|λ| > π/2) and no sense is held, set
  `sense` (E, N) to the game plan's sense:
  - plan two-circle if Δe > the plan threshold (energy fight for the side with more energy:
    P-1210 p. 9; P-1289 7-2 "slow ... force a 1C fight"), else one-circle;
  - two-circle: his rotation sense, or toward his side if he is straight (across his tail);
  - one-circle: opposite to his rotation sense, or away from his side (reverse at the pass);
  - an exact tie (no sense, no side) takes the match's drawn side (decision 5.5).
- **Hold** while |λ| > π/2: the stick turns in the held sense, whatever the short way says.
  This replaces the head-on band (451-452) and removes the reversal near 180 deg.
- **Release** when he comes forward of my 3/9 line, the target changes, or on an overshoot:
  he crosses my extended six inside the overshoot range with a line-of-sight rate above the
  overshoot rate (P-1289 6-4, criteria 1 and 2; "if ever in doubt, do not reverse"). After a
  release the short way applies, which on an overshoot is the reversal.

### B4. Commands per tactic

The pursuit point is his position predicted τ ahead along his turn (second order: offset +
v_t τ + ½ a_t τ^2, a_t his lateral acceleration from his nose rate); τ > 0 lead, 0 pure, τ < 0
lag. v* mixes in log space; the stick cap is "full" or "sustained".

| tactic | τ | v* | stick cap | sense |
|---|---|---|---|---|
| disengaged | 0 (target, or centre) | merge speed (1.0 v_c) | sustained when above 1.195 v_c | short way |
| merge | two-circle plan: + min(time to the pass, lead-turn time); one-circle: 0 | merge speed | full | committed at the pass |
| offensive | the gun lead time when the shot gate is on; else - y x yo-yo time (lag when closing too fast, lead when falling behind) | fight speed x (v_c / v_st)^(-y), raised to v_c when the shot is near | sustained during a high yo-yo (y > 0) above v_st | short way (he is ahead) |
| turning | 0 | fight speed; v_c when the shot is near | full | committed |
| defensive | 0 (into him) | v_c within his defend range; else fight speed | full | committed |

- **Yo-yo signal** y = tanh((κ - κ*) / yo-yo scale), with κ* = (r - nose-tail distance) /
  (closing time x v_c): the closure that reaches the control zone in the closing time. y > 0
  is too much closure: a high yo-yo, lag and climb (bank speed, closure and radius fall, no
  energy lost: P-1210 p. 34, "exchanges airspeed for altitude", "prevents an in-close ...
  overshoot"). y < 0 is falling behind out of range: a low yo-yo, lead and dive (speed,
  closure and, slow, turn rate: P-1210 p. 33, "decrease range by increasing closure", "nose
  in front of the bandit (lead pursuit)"; stop it when angles grow: the y term turns over by
  itself as closure builds). The manuals alternate high and low yo-yos (p. 35); here that is
  y changing sign, not a mode switch.
- **Shot near**: the turn still needed (|λ| at the gun lead point) within spend share x
  store x A_store and in gun range: spend the store only for a purpose (P-1289 7-2: take a
  shot, deny a shot, take turning room). This is the author's two-circle win: bank speed
  between passes, cash it when it buys the shot.
- **v* floor**: v* ≥ speed floor x v_stall (P-1210 p. 52: slow enough to turn, not so slow
  it "affords no ability to turn").

What does not carry over from 3D, and how it is adapted: the yo-yos' out-of-plane geometry
(cutting across the circle in the vertical) has no analogue; what remains is the energy half
(the store as altitude) and the pursuit half (lag or lead), and in 2D the dive also frees the
1 g on the wing, the counterpart of the nose-low turn that gravity helps. The lead turn is
reduced to its timing. Scissors, the flats and the roller are not separate tactics: inside
each other's turn radius the committed sense with the overshoot release gives the reversals.

### B5. Control laws

- **Stick** (kept): error to the wanted direction (pursuit point blended with the boundary
  steer as now), three nested level gates in radians, the side by the committed sense or
  the short way; the cap folds the full level into the sustained one. No level for the rate
  fight beyond "full": full stick settles at the best sustained turn.
- **Power** (replaced): x = ln(v / v*). P(climb) = σ((x - band) / scale), P(dive) = σ((-x -
  band) / scale) x the store's edge fade (kept, 471), coast otherwise, floor 0.02 each. Three
  parameters, one meaning each; nothing reads the stick gates. This replaces DIVE_SLOW,
  DIVE_IN_TURN, CLIMB_FAST, CLIMB_CLOSE, REGAIN_FAR, REGAIN_LOW and the regain share.
- **Fire** (kept): aligned x in range at the gun lead point, ready, no ally in the line. The
  lead gain becomes the gun's aim skill only; the pursuit point is separate.
- **Boundary** (kept, one fix): the margin in R_st x max(1, (v / v_c)^2), the turn radius
  law (flat below corner, v^2 above), not v^2 / max acceleration.

### B6. Parameters

Every new parameter, each meaning one thing and competent alone across its box (log ±0.4 is a
factor 0.67 to 1.49):

| parameter | default | reference | box | source |
|---|---|---|---|---|
| engage range | 3.0 | gun ranges | log ±0.4 | today's regain_far; outside his weapons (P-1289 6-3) |
| 3/9 line | π/2 | radians | ±0.2 | doctrine's dividing line |
| quadrant gate scale | 0.15 | radians | log ±0.4 | none (sharpness) |
| turning floor | 0.25 | ω_st | log ±0.4 | "taken as zero below a small threshold" (doctrine §3) |
| plan threshold | 0.0 | store fractions | ±0.25 | P-1210 p. 9; P-1289 7-2 |
| lead-turn time | 1.0 | t90 | log ±0.4 | P-1289 7-5, P-825 p. 139 (no number in doctrine) |
| overshoot range | 0.67 | R_st | log ±0.3 | P-1289 5-5: 2,000 ft inside a 3,000 ft bubble |
| overshoot rate | 1.0 | ω_st | log ±0.3 | P-1289 6-4: "a high line-of-sight rate" |
| nose-tail distance | 0.67 | R_st | log ±0.4 | P-1289 5-3 control zone near edge; P-1210 p. 35 "at least 1,000 ft" |
| closing time | 1.0 | t90 | log ±0.4 | none |
| yo-yo scale | 0.1 | v_c (closure) | log ±0.4 | none |
| yo-yo time | 0.5 | t90 | log ±0.4 | none |
| merge speed | 1.0 | v_c | log ±0.1 | P-1289 7-3 (2C at corner); P-821 10-44 |
| fight speed | 1.0 | v_st | log ±0.2 | P-821 9-12 (Ps = 0 peak); P-1210 p. 52 |
| speed floor | 1.3 | v_stall | log ±0.1 | P-1210 p. 52 |
| spend share | 1.0 | of store x A_store | log ±0.4 | P-1289 7-2 |
| defend range | 1.0 | gun ranges | log ±0.3 | P-1289 6-3 (inside his weapons) |
| speed band, power scale | 0.05, 0.03 | ln units | ±0.03, log ±0.4 | none |

Kept: the stick gates (in radians), gun gates, size clamp, line-of-fire clearance, lead cap,
aim skill (today's lead gain), keep factor, weak-target weight, boundary margin (2.2 R_st,
today's 504 m at corner speed) and ramp, floors. Clamps (the v* floor, the edge fade, the
1.195 v_c cap point from the card's sustained level) keep every corner of the box flyable.
Temperature stays per persona and divides the decision sharpnesses; whether it should also
blunt the gun is an open question below.

### B7. Cost

Estimated against today's 208 (both teams, one call):

| change | ops |
|---|---|
| target gather 4 to 8 columns | +2 |
| assessment (λ, α as cosines, closure, LOS rate, senses, flow, Δe) | +25 |
| tactic weights (three gates in the stack, products) | +12 |
| pursuit point (second-order prediction, τ mix) | +12 |
| committed sense (commit, hold, release, plan) | +12 |
| v* mix and power law, against today's power | -8 |
| stick cap, boundary radius fix | +5 |
| total | about 270 (+30%), to be measured |

### B8. Build order

Each step is small, has its own tests (CPU quick tier on the touched files) and a stated
expectation against the step before.

1. **Game references** (core): v_st, ω_st, R_st, t90, A_store in `state.constants`. Test:
   equal to `tuning.performance` within its 0.05 m/s grid. Behaviour: unchanged.
2. **Parameter table** (refactor): named rows, personas from the table, stick gates in
   radians (today's values x the stall angle). Expect the same actions within rounding at
   spread 0; the same draws re-laid at spread > 0 (not bit-equal); ops unchanged.
3. **Boundary radius fix**. Expect: the competence test passes; earlier turn-back when slow.
4. **Situation** (computed, tested, not yet used): hand-built pairs give the right quadrant,
   flow sign, closure, Δe.
5. **Power by speed target**, v* from disengaged (merge speed) and engaged (fight speed,
   corner when the shot is near) only. Expect: the brawl's mean store well above 0.05, the
   share below 0.75 v_c down; measured against the NOTES table.
6. **Tactic weights and the pursuit point** (lag, pure, lead, yo-yos, defensive).
7. **Committed sense**, plan and overshoot release; the head-on band goes, the drawn side
   stays as the tie-break.
8. **Lead turn** at the merge.
9. **Personas**: widths for the new rows, competence at spread 1; new against old; ops.

### B9. Tests and measurements

- **Hand-built situations, one per tactic** (`marginals`, as `test_power_rule` does today):
  disengaged at 1.0 v_c and 1.3 v_c (coast, then climb with the sustained cap); a tail chase
  inside the control zone closing at 0.3 v_c (lag, climb) and one out of range opening (lead,
  dive); a head-on with lateral offset 2 R_st, Δe > 0 and Δe < 0 (turn toward his side before
  the pass, against reverse after it); him at my six in gun range nose on (full stick into
  him, dive with store); both turning left (f = +1) and opposite (f = -1); an in-close
  overshoot at high line-of-sight rate (release, reverse) and a slow one (hold).
- **Two-circle 1v1** from a fixed set of 2C starts (offset head-on passes): angles gained per
  revolution, store over time, who fires first; old against new and new against new.
- **Scripted brawls**: the NOTES table (speed, store, dive/climb/coast shares, kills per
  minute, game length, beyond the arena) for old, new yardstick and new at spread 1.
- **New against old**: 4v4 elimination in the core game, sides swapped, adaptive to a target
  standard error (no fixed count).
- **Operations**: game_ops.py, both teams and one team.
- **For the author to fly** (sandbox, 2x): the two-circle fight he won (does it bank and
  cash?), a one-circle fight (does it slow to the radius fight and dive when slow?), a head-on
  with offset (lead turn, which flow it sets), being chased (does it break and spend), leaving
  and coming back (does it return with a full store), and a 2v2.

## Open questions for the author

1. **Fight speed in v_st.** The decided references name corner and stall speed for speeds.
   The best-sustained speed comes from the same turn as R_st and t90; I propose it as a
   speed reference too (0.575 v_c for the P-51D, other ratios on other aircraft).
2. **"Two-circle at the sustained stick"** (backlog E2): in the point-mass P-51D the best
   sustained rate is full stick (it settles at v_st); the sustained level settles at 181 m/s
   and 33.3 deg/s. The design flies the rate fight at full stick with the speed target, and
   uses the sustained level only to bank speed above 1.195 v_c.
3. **The plan rule for equal aircraft**: two-circle with more energy, one-circle with less
   (threshold per persona). Or a fixed preference per persona?
4. **Climbing restarts the shield wait** (a climb is not quiet, core.py 949): the disengaged
   regain climb delays shield recharge. Accept the trade-off, or count a straight climb as
   quiet?
5. **A third memory field** (`sense`): replaces the head-on band; the drawn side stays.
6. **Temperature and the gun**: should temperature blunt the gun gates (skill), or only the
   stick, power and tactic gates (decisiveness)?
7. **The round trip costs energy** (the climb loses 4% of what it banks at corner speed, 12%
   at v_st): the design banks speed when fast. Physics as intended?
8. **Fog**: his nose rate and store are from his last sighting; the flow can be stale for a
   hidden target. Acceptable without dead reckoning (dropped)?

## Judgement calls

- Tactics mix scalars (τ, v*, the cap), never directions, so a soft weight cannot produce
  the average of two turns; the only hard decision (the sense) is committed by rule.
- Yo-yos are a signed closure signal, not two modes; the alternation is the sign changing.
- The power rule is a speed tracker; "spend only for a purpose" sets when v* is raised to
  corner speed.
- The lead gain moves to the gun only, so a persona's pursuit and its aim vary separately.
- Lead turn, scissors and Lufbery are not separate tactics; the merge's lead time and the
  committed sense with the overshoot release cover them. A stalemate breaker is left out
  (no extend or retreat, decision 5.7).
- Threshold numbers from T-45 doctrine are scaled by the bubble (its 3,000 ft turn radius)
  to R_st; numbers marked "none" have no doctrine source and are proposals to measure.
