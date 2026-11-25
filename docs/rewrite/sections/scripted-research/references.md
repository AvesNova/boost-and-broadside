# Scripted dogfight AI: references and what they imply (2026-11-26)

Research only: no code, no GPU. The question is whether the scripted opponent
(`lean/scripted/`, branch `scripted-fight-plan`) is built the right way, given four open
problems: repeated neutral head-on passes, the store draining in turning fights, the plan
(one- or two-circle) really being set by both ships' turn senses, and a growing rule count.

Each source is labelled **Says** (what the source states, read in the primary text unless
marked) and **Inference** (my reading for Boost & Broadside). Items I could not check in a
primary text are marked *not verified*.

## 1. Summary

- Burgin's AML went through the same three steps we did. First a **trial-maneuver search**:
  a handful of candidate maneuvers is predicted a short time ahead against the opponent's
  extrapolated path, scored by a weighted sum of yes/no situation questions, and the best is
  flown (1975). Then **IF-THEN rules**, because the search was too slow for 1980s
  minicomputers. Then, because the rules were **predictable**, a few **BFMs picked by random
  numbers** were added on top (1988). The 1988 report's open problems list our energy-drain
  problem almost word for word, and its neutral co-altitude runs stay neutral or
  "essentially stalemated".
- The game-theoretic line (Austin 1990, Park et al. 2016) keeps the search. It uses a small
  maneuver set for **both** aircraft (7 x 7 = 49 pairs in Park), a 1-2 s prediction, a score
  built from angles, range error and a speed term, and a **min-max (security) choice**. With
  the same algorithm on both sides in a symmetric head-on start, Park et al.'s ICAS paper
  reports **100% neutral**: a draw.
- McGrew et al. (2010) is the source of the widely reused score: angular term
  ½[(1 − AA/180°) + (1 − ATA/180°)] times a range term exp(−|R − R_d| / (180° k)) (the
  form as quoted by later papers; primary not verified). It is a learned value plus a
  rollout, so not a scripted agent.
- The DARPA AlphaDogfight adversaries from APL were tiers: Zombie (straight and level),
  Rosie (time-based deviations, "simulates one-circle flow"), BUD FSM (a small state machine
  over PID set points, "novice pilot") and an RL agent. ALPHA (Psibernetix) is a BVR missile
  controller, not a gun dogfighter, so it says little about our problem.
- DCS World's AI notes (developer newsletter) make the one-circle/two-circle choice **per
  aircraft type**, set turning room at the merge by the opponent's type, aspect and energy,
  and switch style by merge angle and relative airspeed.
- **Recommendation (inference):** keep the outer rules (target choice, fog memory, boundary,
  gun gate, merge miss offset). Replace the fight core (tactic weights, plan, sense
  commitment, speed target) with a **small candidate library scored a few seconds ahead
  against a few opponent replies**, all in parallel as an extra tensor axis, with a softmax
  for the BC distribution. Build it as a second opponent beside the yardstick and keep the
  yardstick until the new one beats it. Sections 5 and 6 list what this gives up.

## 2. Sources

### 2.1 Burgin et al., Adaptive Maneuvering Logic (AML), NASA, 1975-1988

Links: NASA CR-2582, *An Adaptive Maneuvering Logic Computer Program for the Simulation of
One-on-One Air-to-Air Combat*, Vol. I (Burgin, Fogel, Phelps, 1975),
<https://ntrs.nasa.gov/citations/19750022744>. NASA CR-4160, *Rule-Based Air Combat
Simulation* (Burgin, Sidor, 1988), <https://ntrs.nasa.gov/citations/19890018022>.

**Says (1975, trial maneuvers):**
- (a) At each decision point (an input interval; 0.5 to 2 s worked on Langley's DMS) the
  program extrapolates the opponent's path, predicts its own path for "a situation-dependent
  number of trial maneuvers", scores each predicted end situation and flies the best. A
  maneuver is (bank angle, load factor, thrust) in a "maneuver plane" through the velocity
  vector. The standard set has four trials: continue the present maneuver, max-g turns in the
  two neighbouring planes, and straight flight. Added when they apply: a max-g turn in the
  plane closest to the opponent, an intercept turn, and a defensive turn perpendicular to it
  when the value falls below a threshold.
- (b) The value is a scalar product of yes/no answers to 15 questions and input weights.
  The questions cover: is he in front of me, am I behind him, can I see him, can he not see
  me; my weapon envelope and his; the cone in front and behind; range rate (closing at a rate
  that crosses the weapon range in 2 to 10 decision intervals); line-of-sight angle under
  60°; line-of-sight rate inside a band (e.g. −10 to −30 °/s at 90° off, so the target
  reaches the nose in 3 to 9 s); and specific energy, five intervals ahead, within limits of
  the opponent's ("slightly greater"; conserving energy is "undesirable" against a much lower
  energy opponent). Weights "are usually all set to one"; "no systematic investigation has
  yet been made to optimize these weight factors". A strongly negative question
  (weight −13) keeps it off the ground.
- Prediction: own path as a circular arc at the trial's normal acceleration, with thrust and
  drag ignored; the opponent by second-order extrapolation of three past positions.
- Randomness: decision time, prediction time, range, line-of-sight angle, first-trial load
  factor and the opponent's energy can each be perturbed uniformly by ±20%.
- Results: against an F-4 pilot with 150 combat missions, 9 wins, 4 losses, 2 stalemates; in
  a second series 10, 3, 1 (winner = longer offensive time with advantage).

**Says (1988, rule-based):**
- The trial-maneuver search "exceed[ed] the capacity of VAX 11/780-class mini-computers for
  real-time", so the rule-based AML replaced it with production rules plus guidance laws.
  Decision interval for close-in combat: "typically between 0.5 and 1.5 seconds".
- (a, c) Steering is lead, pure or lag pursuit, chosen by region in the (line-of-sight angle,
  angle-off) plane (Fig. 3-2), with an evasive region. Load factor: max g above corner
  speed; "when the current airspeed is near or below corner velocity, the sustained load
  factor is commanded to avoid losing further energy"; in the forward sector, the intercept
  load factor if it is lower. Throttle is afterburner except idle to avoid an overshoot.
- (c) Nose-to-nose vs nose-to-tail: the logic produced "either a nose-to-nose conversion, or
  a nose-to-tail conversion, depending on the initial conditions". An "opposite turn" BFM
  (turn away from the opponent) was added. It helped only with the right speed. The authors
  suggest considering "the differential turn rate between the two aircraft at the time of
  the decision", which "might be a good way" to separate offensive from defensive aircraft.
- (d) Stalemates: neutral starts "remain essentially neutral"; forward-quarter passes
  "rapidly develop into well-defined scissors" and the situation "remains essentially
  stalemated". A "pull-up overshoot" BFM (wings-level 95% max-g pull-up for 5 s, then back
  to the lead/lag logic) needed a speed advantage and was shown "breaking the stalemate of
  the scissors". Predictability: pilots could "anticipate AMLB's future maneuvers after a
  few sessions", so BFMs were chosen "by the means of a random number" whenever a BFM and
  the base logic both applied.
- "In all cases, both AMLA and AMLB command initially high load factors which result in a
  rapid loss of airspeed." Open problems (Section 7) include "How can we prevent, early
  enough, the AML driven aircraft's energy to deteriorate to a very low value?" and how to
  bring in differential-game results.
- Evaluation: "Well over 100 different initial conditions must be exercised"; small changes
  in initial angles gave widely different outcomes.
- Performance index used for plots: PI = 50(1 − λ/180) + 50(1 − ε/180), λ the deviation
  (line-of-sight) angle, ε the angle-off; PI ≥ 80 is a necessary condition for a gun
  position.

**(e) Cost:** the trial-maneuver step is a handful of closed-form arc predictions and a
weighted sum per candidate. Too slow for a 1985 minicomputer at 50 ms frames; trivial now.
It maps to tensors directly: candidates on an extra axis, arcs in closed form, questions as
smooth indicator functions.

**Inference:** AML is the closest precedent to our agent, and its history argues against a
pure rule set. The reason it left the search (compute) does not apply on a GPU, and the
problems its rules then ran into (predictability, scissors stalemates, energy bleed) are
ours. Its rule-based load-factor law (max g above corner, sustained at or below) is a
one-line answer to our store drain that we do not use: in the turning tactic we chase
1.29 sustained speeds with the dive (ghosts notes, Section 3).

### 2.2 Austin et al., game-theoretic automated maneuvering (1990)

Link: F. Austin, G. Carbone, M. Falco, H. Hinz, M. Lewis, "Game Theory for Automated
Maneuvering During Air-to-Air Combat", *J. Guidance, Control, and Dynamics* 13(6),
1143-1149, 1990, doi:10.2514/3.20590. AUTOMAN record: <https://ntrs.nasa.gov/citations/19920031742>.
(Full text not read; abstract and citing papers only.)

**Says (abstract, via catalogue records):** a game-matrix approach for one-on-one
helicopter combat over hilly terrain. Scores built from predicted orientation, range,
velocity and terrain clearance are compared across maneuver combinations of **both**
aircraft. Installed on NASA Ames' simulator as the adversary for human pilots, and said to
generate "some of the tactics employed by experienced pilots". Burgin (1988) calls it "a
very similar trial maneuver technique".

**Inference:** this is the step from AML to a two-player matrix: score my candidate against
each of his. *Not verified:* the candidate count, horizon and decision rule; Park et al.
(below) build on it.

### 2.3 McGrew et al., ADP for air combat (2010)

Link: J. S. McGrew, J. P. How, B. Williams, N. Roy, "Air-Combat Strategy Using Approximate
Dynamic Programming", *JGCD* 33(5), 1641-1654, 2010, doi:10.2514/1.46815; author manuscript
at <https://dspace.mit.edu/handle/1721.1/67298>. Thesis: <https://dspace.mit.edu/handle/1721.1/44927>.
(The DSpace PDF refused automated fetches; I read the abstract and papers that quote it.)

**Says (abstract):** level flight, fixed velocity, one-on-one; approximate dynamic
programming with "feature development, reward shaping and trajectory sampling", and an
"online policy extraction method based on rollouts"; flight tests on micro-UAS in MIT's
RAVEN.

**Says (as quoted by Kurniawan et al., arXiv:2005.02057, and others):** the "McGrew score"
is S = A_M x R_M, with A_M = ½[(1 − AA/180°) + (1 − ATA/180°)] and R_M = exp(−|R − R_d| /
(180° k)), where R_d is the desired range and k the width of the range peak. A position of
advantage is quoted as range within [d_min, d_max] and AA, ATA under 60°
(arXiv:2112.01328). *Not verified in the primary:* the action set, time step and the
opponent model.

**Inference:** two lessons. (1) The score is a product, so a good angle at a bad range
counts for little. That is the right shape for guns. (2) McGrew needed a learned value
function on top of the score because a short-horizon greedy score is myopic. Our cheap
version of "seeing further" is a longer horizon with two-segment candidates (Section 5).
With our nose cone, the AA term should be replaced by the game's own `aspect_factor` (a
head-on position is worth nothing to the gun), not left at ½ as McGrew's form gives.

### 2.4 Park, Lee, Tahk, Yoo (2016), differential-game BFM with a scoring-function matrix

Links: H. Park, B.-Y. Lee, M.-J. Tahk, D.-W. Yoo, "Differential Game Based Air Combat
Maneuver Generation Using Scoring Function Matrix", *IJASS* 17(2), 204-213, 2016,
<http://past.ijass.org/On_line/admin/files/8)(204~213)15-100.pdf>
(<https://koreascience.kr/article/JAKO201623562836559.page?lang=en>). Companion: B.-Y. Lee
et al., ICAS 2016, <https://www.icas.org/icas_archive/ICAS2016/data/papers/2016_0096_paper.pdf>.

**Says:**
- (a) Hierarchical: a top level decides the pilot behaviour (pursue or evade); the lower level
  generates the maneuver and decides fire by the gun range. Each aircraft has 7 maneuvers
  (level, pull up, push down, left up, left down, right up, right down). All 49 pairs are
  predicted by integrating the equations of motion for 1-2 s (ICAS: the algorithm runs every
  0.6 s and predicts 0.6 s). The choice is min-max: the "security strategy" that avoids the
  worst case whatever the enemy picks. The ICAS paper adds a "counter-greedy" variant. It
  detects from the previous matrix whether the enemy is greedy and, if so, picks the best
  answer to his greedy move.
- (b) Score: orientation S_Ori = (2π − (BA + AA))/(2π)-style term (0 to 1, 1 on his tail),
  velocity error from the corner speed on the V-n diagram, and a range error score kept
  between gun range and a safety distance. The total is a product of the three (ICAS Eq. 4).
  A barrier function keeps altitude above a floor.
- (d) Symmetric head-on, the same security algorithm on both sides: "the combat result is
  draw", 100% neutral possession. Security vs greedy: 3.16% advantage to security.
  Counter-greedy vs greedy: 46.3%; counter-greedy vs security: 58.1%.
- (e) Compute: 49 short integrations every 0.6 s; "moving horizon" to keep it real-time.

**Inference:** this is the clearest template for a batched version: K x M candidate pairs,
a short prediction, a product score, min over replies. It also gives the warning that
matters most for us: **identical deterministic search agents in a symmetric start draw**.
A search alone will not end self-play stalemates. Asymmetry has to come from energy, the
draw, personas or the merge geometry. Min-max (security) is conservative. The gain of
counter-greedy shows that the aggregation over replies is a real design parameter (a
persona knob, Section 5).

### 2.5 Psibernetix ALPHA, genetic fuzzy trees (2016)

Link: N. Ernest, D. Carroll, C. Schumacher, M. Clark, K. Cohen, G. Lee, "Genetic Fuzzy based
Artificial Intelligence for Unmanned Combat Aerial Vehicle Control in Simulated Air Combat
Missions", *J. Def. Manag.* 6:144, 2016, doi:10.4172/2167-0374.1000144,
<https://scholar.uc.edu/downloads/bc386t26n>.

**Says:** a "Genetic Fuzzy Tree": a cascade of small fuzzy inference systems (branches for
high-level tactics, firing, evasion, situational awareness, defensiveness), all trained
together by a genetic algorithm. Known optimal sub-solutions and heuristics are plugged in
directly. Over 150 inputs. The mission is **purely beyond visual range** (2 blue vs 4 red,
long-, medium- and short-range missiles). Runs at 154 Hz on one 3.2 GHz core with the
simulator (over 1,100 Hz counting only its own computation).

**Inference:** not a guns dogfight, so no answer on turn direction or head-on passes. The
transferable idea is structural: decompose into small smooth scoring units with few inputs
each, and tune the numbers by a population search. That is close to what our logistic gates
and CMA screening already do. It argues for keeping parameters few and smooth, not for
fuzzy logic itself.

### 2.6 DARPA AlphaDogfight Trials (2019-2020): the non-RL adversaries

Links: C. R. DeMay, E. L. White, W. D. Dunham, J. A. Pino, "AlphaDogfight Trials: Bringing
Autonomy to Air Combat", *Johns Hopkins APL Technical Digest* 36(2), 2022,
<https://jhuapl.edu/sites/default/files/2024-09/36-02-DeMay.pdf>. A. P. Pope et al.,
"Hierarchical Reinforcement Learning for Air Combat at DARPA's AlphaDogfight Trials", IEEE
TAI 4(6), 2023, <https://arxiv.org/abs/2105.00990>.

**Says (APL):**
- (a) Tiers of APL adversaries: "Zombie" (straight and level, fixed speed: "cruise
  missile"); "Rosie" (time-based minor altitude and speed deviations: "one-circle flow");
  "BUD FSM" (recognises the state of the engagement and runs a scripted response: "novice
  pilot"); "AlphaMav0" (RL by self-play: "expert pilot").
- The basic agents are a three-loop PID: an outer loop computes a desired speed and heading
  change "in a few simple profiles" from "a handful of cases of relative distance and angle",
  each with "a fixed command that is manually tuned". For example, at 10+ miles and under 60°
  off, maximum speed straight at the target. The scripted agents add "a small state machine
  that selects an engagement strategy", e.g. a "flare" (rapid slowdown to force an overshoot)
  when an opponent closes from behind.

**Says (Lockheed, Pope et al.):** training used about 30 opponents plus self-play, including
APL's scripted agents, "Bud-FSM" and a random-maneuver agent "Randy". Expert knowledge went
into reward shaping, not hand-coded maneuvers. The shaping terms are logistic functions of
the normalised track angle, adverse angle, range and closure, with a gun-snap reward by
range band and a too-close penalty. The final agent placed 2nd of 8 and beat an F-16
Weapons Instructor Course graduate.

**Inference:** the people who built the reference adversaries kept the scripted tier
deliberately small (a state machine over set points) and labelled it "novice". Nobody in
ADT relied on a large scripted BFM rule set for strength. For a 1000-Elo anchor and BC prior,
"novice to decent, varied" is the target. A tier of opponents (chaser, yardstick, stronger
search agent) is an accepted way to span skill.

### 2.7 Open-source dogfight code

- **LAG / CloseAirCombat** (JSBSim, 1v1 and 2v2), <https://github.com/liuqh16/LAG>. **Says
  (code):** `envs/JSBSim/model/baseline.py` has `PursueAgent` (delta heading set to the
  target's bearing, fed to a pretrained recurrent low-level controller) and `ManeuverAgent`
  (fixed heading changes every 7 s). The `PostureReward` is orientation(AO, TA) x range(R)
  (`reward_functions/posture_reward.py`). **Inference:** the open baselines are pursuit
  plus a learned autopilot, much weaker than our yardstick. Their scoring repeats the
  McGrew product shape.
- **BVRGym** (JSBSim, BVR), <https://github.com/xcwoid/BVRGym>, paper
  <https://arxiv.org/abs/2403.17533>. **Says:** the adversary is a behaviour tree that puts
  missile evasion first and offence second. **Inference:** BVR, not relevant to the turning
  fight.
- **DCS World AI** (not open source, but its developer notes are public),
  <https://www.digitalcombatsimulator.com/en/news/newsletters/d48111b832efd4f57df12df8f46e7ca1>.
  **Says:** "each aircraft type will favor either one-circle (nose to nose), two-circle (nose
  to tail) fights or a combination". The AI sets turning room at the merge from the
  opponent's type, aspect angle and energy: a one-circle jet minimises turning room, a
  two-circle jet makes an offset for a lead turn. It may switch style by merge angle and
  relative airspeed. Skill level sets how much g it pulls and how closely it manages speed.
- **Falcon BMS, IL-2, FlightGear, OpenEaagles:** I found no public design notes on their
  close-in maneuver logic (forum threads only). Nothing cited.

### 2.8 Game-AI guidance

- **Reynolds, "Steering Behaviors for Autonomous Characters"** (GDC 1999),
  <https://www.red3d.com/cwr/steer/gdc99/> (PDF copy read). **Says:** pursuit seeks the
  target's position predicted T ahead, with T = D·c (larger when far) and reduced to zero
  when "aligned with, and in front of" the quarry. **Offset pursuit** steers "a path which
  passes near, but not directly into a moving target" ("fly-by", "strafing run") by aiming at
  a point offset by a radius R from the predicted position. These laws are "lightweight and
  nonoptimal", and evasion in nature is "intentionally nonoptimal in order to be
  unpredictable". **Inference:** our merge miss offset is Reynolds' offset pursuit at a 40 m
  radius. The missing pre-merge offset is the same law at about one turn radius.
- **Utility AI** (Dill and Mark, "Improving AI Decision Modeling Through Utility Theory",
  GDC 2010, <https://gdcvault.com/play/1012841/Improving-AI-Decision-Modeling-Through>).
  **Says (session listing and recap):** options are scored by weighted considerations
  through response curves, and chosen with weighted randoms. Weights differ per character.
  **Inference:** a candidate-scoring opponent is a utility system whose considerations are
  evaluated on predicted states, not current ones. Personas become weight vectors.
- **Rolling-horizon planning for a 2D space battle** (J. Liu, D. Pérez-Liébana, S. M. Lucas,
  "Rolling Horizon Coevolutionary Planning for Two-Player Video Games", CIG 2016,
  <https://arxiv.org/abs/1607.01730>). **Says:** two ships (rotate, thrust); a win is being
  behind the opponent, facing his back, within range. Simultaneous moves. All controllers
  share a heuristic built on a position-direction dot product, with 10 ms per tick. Open-loop
  MCTS, a one-step lookahead and two rolling-horizon evolutionary planners are compared; the
  coevolutionary planner does best. Shorter macro-actions did better. Strong searchers
  paired against each other score about 50 each. **Inference:** short-horizon search works
  for a 2D turning duel at game frame budgets, and evenly matched searchers stalemate here
  too.
- **Theory:** the turning duel of equal ships is Isaacs' game of two cars; A. W. Merz, "The
  Game of Two Identical Cars", *JOTA* 9, 324-343, 1972, doi:10.1007/BF00932932, solves its
  barrier (which relative states allow capture). **Inference:** with identical cars and no
  energy difference, much of the state space is outside the capture region. A neutral
  stalemate between equal, competent ships is expected, not a bug to be designed away.

## 3. What the sources agree on (Says, cross-cutting)

1. **Structure.** Three families: rules plus guidance laws (AML-RB, APL BUD FSM, BVRGym BT);
   candidate maneuvers predicted 0.6-2 s ahead and scored (AML 1975, Austin, Park); learned
   values or policies (McGrew, ADT entrants). The search versions were given up only for
   compute (AML 1988).
2. **Decision interval 0.5-2 s**, not every frame (AML, Park), with an override for safety
   (dive recovery each frame in AML).
3. **Score shape.** Every scored approach multiplies or adds an angular term (ATA/AA, or
   bearing and angle-off), a range term peaked inside gun range, and a speed or energy term
   (AML question 15, Park's corner-speed error). Gun-specific: Pope et al.'s gun snap and
   too-close penalty.
4. **Energy.** Rule logic bleeds speed early (AML 1988, "in all cases"). Energy appears
   either as a score term (AML, Park) or as a load-factor law by speed vs corner (AML-RB).
5. **Stalemates.** Neutral, symmetric starts between equal logic stay neutral (AML 1988, Park
   ICAS, Liu et al.). The breakers in the sources are randomised BFM choice (AML), an energy
   or speed advantage turned into a vertical or offset move (AML pull-up overshoot, DCS
   turning room), and a counter-strategy against a predictable opponent (Park counter-greedy).
6. **One- vs two-circle.** No source picks it from energy alone. DCS ties it to aircraft type
   and merge geometry and airspeed; AML 1988 proposes the differential turn rate; the
   matrix methods do not name it because it falls out of the joint choice of both turn
   directions.

## 4. What this implies for our four problems (Inference)

- **Head-on passes.** Our passes are nose to nose (4 m median lateral separation against a
  241 m turn radius, ghosts notes Section 3) because the stick aims at the lead point up to
  the pass. Every source that handles the merge separates the aim of the gun from the aim of
  the flight path before the merge: Reynolds' offset pursuit, DCS's turning room, Shaw's
  offset. A scorer with a horizon past the pass would find the offset by itself only if the
  horizon covers the turn after it, about 2.7 s at corner rate (180° at 67.2 °/s) to 4.5 s
  sustained (180° at 39.9 °/s). The sources' 0.6-2 s would not see it. Between two copies of
  the yardstick some neutral passes will remain whatever we do (Park, Merz). The aim is that
  the passes are not 4 m nose-to-nose and that the loop is not flown at 0.6 corner speeds
  with an empty store.
- **Store drain.** The direct fix from AML-RB is a load-factor and speed law: full stick
  above corner, sustained level at or below it, and no dive just to hold a speed target
  above sustained in a turn. The scored version is an energy term (own energy vs his, with a
  floor penalty) evaluated at the end of the horizon, so a candidate that empties the store
  for no angle loses.
- **Who commits first.** In a matrix method the plan is not a separate rule. My turn sense is
  scored against each of his candidate senses; min-max picks the sense that is safe whatever
  he does, and an expectation picks the one best on average. One-circle vs two-circle is then
  the outcome of the 2 x 2 (my sense x his sense) block of the matrix. Our "plan + lean +
  switch + stall passes" rules are a hand-made approximation of that block. The catch: the
  payoff of one- vs two-circle is decided over one or two full turns (4-9 s), longer than a
  cheap horizon. A short-horizon scorer will lean on the energy and turn-rate terms to
  represent it. A small analytic term could help: the time for each ship to bring its nose to
  bear on the other under each sense pair, from the quasi-steady turn rates the game already
  computes (`lean.game.core.steady_rates`).
- **Rule count.** The sources that lasted (AML, Park, DCS) all have a small number of
  behaviour primitives and put the judgement in a score or a few situation regions. AML 1988
  added BFMs one at a time as special cases, each "appropriate under a narrow set of
  circumstances", the same pattern as our lead turn, yo-yos and reversal that were added and
  then removed after the reach checks.

## 5. Options compared (Inference)

### A. Keep the rule set, add the three proposed fixes (pre-merge offset, keep energy, full stick after the pass)

- Gives: the smallest change, the existing tests and tuning (BFM-CMA2), the Shaw names and the
  `explain` reasons, a known compile size (act 1,704 traced operations; 475 for both teams'
  scripted call).
- Gives up: the plan stays a heuristic over a two-player choice. Each new geometry adds rows,
  gates and tests. The sources (AML 1988) suggest the stalemates and the predictability
  return.
- Use when: the goal is a usable yardstick soon and the stalemate measure (passes per pair,
  store by pass) improves enough.

### B. Simplify the rules toward AML-RB

- One steering law chosen by region in (angle off, aspect) with lead/pure/lag; the load
  factor by speed vs corner (full above, sustained at or below); throttle by overshoot risk;
  the merge offset. Drop the plan memory and choose the sense at the pass from a 2 x 2 payoff
  of quasi-steady times to bear.
- Gives: fewer parameters, the energy fix by construction, still cheap and legible.
- Gives up: some of the current tuned behaviour; still myopic, still predictable (AML's own
  criticism); variety only from the drawn parameters.

### C. Replace the fight core with a candidate library scored a few seconds ahead

Keep target choice, fog memory and roam, the boundary steer, the gun gate (aspect factor
worth, never through a team-mate) and the merge miss offset as they are. Replace tactic
weights, plan, sense commitment, speed target and stick-level gates with this:

1. **Candidates (mine), K about 15-25.** One-segment: stick in {full L, middle L, straight,
   middle R, full R} x power {coast, dive, climb} (15). Two-segment for the merge: fly the
   offset point (a lateral offset of about one sustained radius, left or right) for t1, then
   full stick toward him, with t1 in {0.5, 1.0} s (4). Optionally "keep the last choice" as
   an explicit candidate for hysteresis.
2. **Replies (his), M = 3.** Continue his current turn rate, full left, full right, at his
   current speed trend. Under fog, from his last-seen state (as `engage` does now).
3. **Prediction in closed form, not tick by tick.** A constant-rate arc per segment (AML did
   this), with turn rate, d speed/dt and d store/dt taken from the game's quasi-steady model
   (`steady_rates`) at the candidate's stick and power. Sample the predicted relative state
   at 3 times (e.g. 1/3, 2/3, end of the horizon), not only the end. No unrolled
   `flight_tick` loop: unrolled, H ticks multiply the traced operations by H; closed form adds
   a K x M axis to the same operations. Fields: use the index at the ship. The prediction is
   rough inside fields, the same limit the current lead point has.
4. **Score per (candidate, reply, sample time), weights w:**
   - gun value = alignment of my nose to the lead point (smooth, e.g. a logistic in angle) x
     in-range factor x the game's `aspect_factor` at the predicted aspect (a head-on shot is
     worth 0, as in the game);
   - threat = the same from his side (subtracted);
   - position = McGrew-style angular term on ATA and the angle off his tail x range term
     exp(−|R − R_d| / width), R_d about 0.3-0.6 gun range (210-420 m), width about 0.3 gun
     range;
   - energy = (my speed² /2 + store) − (his), normalised by the store size, plus a penalty
     below a store floor;
   - speed = Park's term, distance from corner speed in log space;
   - collision = penalty when the predicted closest approach is under a few hit radii;
   - boundary = penalty for predicted distance beyond the arena margin.
   Combine over sample times with increasing weights.
5. **Aggregate over replies:** λ·min + (1 − λ)·mean (λ = 1 is Park's security strategy).
6. **Choose:** softmax over K with temperature τ, plus a commitment bonus b for the
   candidate chosen last time. That gives hysteresis without a branch and avoids left-right
   dithering at symmetric states. Decision interval: rescore every tick and rely on b, or
   rescore every D ticks (D = 6-15, 0.2-0.5 s) and hold.
7. **BC distribution:** the softmax over candidates, marginalised onto the stick and power
   heads (each candidate's first-tick action), is the teacher's distribution directly. The
   gun head stays as now.

**Parameters it needs:** K and the candidate table; horizon T (2.5-4.5 s, set by the time to
turn 180° at corner and sustained rates; read from `state.constants` so it transfers across
ships); sample times; t1 values and the offset radius (in sustained radii); M and the reply
table; λ; weights for the seven terms (about 7); R_d and the range width (in gun ranges);
store floor; τ; commitment bonus b; decision interval D. About 15-20 numbers, against 49
rows in `PARAMETERS` now. Personas: λ (aggression), the energy weight (energy vs angles
fighter), T, τ.

**How it handles the four problems:**
- head-on: the two-segment candidates and a gun value of 0 head-on make an offset pass
  better than a nose-to-nose one, if T covers the turn after the pass;
- store: the energy term and floor make a candidate that drains the store for no angle lose;
- commit first: the min/mean over his three replies scores my sense against his. One- or
  two-circle falls out of that joint choice, not a separate plan rule;
- rules: the fight core becomes one scored choice; the parameters are weights in reference
  units.

**What it gives up:**
- *Legibility.* `explain` would show a score table instead of named tactics. Still
  inspectable (per-term contributions per candidate), but not Shaw's vocabulary.
- *Doctrine.* No named lead turn or yo-yo. They appear only if the score and the horizon
  reward them. A short horizon is myopic and favours pure pursuit, which is McGrew's reason
  for a learned value.
- *BC variety.* An argmax would be close to deterministic, worse than the current logistic
  mixture. A softmax with τ and drawn personas is required, and the variety still has to be
  measured (action entropy by situation, not only Elo).
- *Compile and memory.* The scripted call gets a K x M axis: E x N x K x M per intermediate.
  Training (4v4, N = 8) at E = 4,096 with K = 20, M = 3 is 2 M elements, fine. 32v32
  (N = 64) at E = 4,096 is 15.7 M elements, about 63 MB per float32 intermediate. With tens
  of intermediates that is several GB, so the 32v32 evaluation would need smaller E or a
  chunk over K. Traced operations should stay of the same order as now if the prediction is
  closed form. To be measured with `scoreboard/compile_check.py` and `game_ops.py`.
- *Tuning.* Weights interact. The CMA screening machinery can tune them, but BFM-CMA2's
  tuning does not carry over, and a new yardstick moves the 1000-Elo anchor.
- *Prediction error.* Constant-rate arcs ignore the nose lag and field refraction. The score
  can be wrong in fields and near the stall. That is acceptable for a prior, but it is a new
  failure mode.
- *Symmetric stalemates remain.* Park and Liu et al. show equal searchers draw. The search
  improves the quality of the fight, not the decisiveness of mirror matches.

## 6. Recommendation (Inference)

1. **Do not keep growing the rule set.** AML's history (rules, then random BFMs as patches,
   with the energy problem still open) is the strongest evidence in the sources, and it
   matches our worklog.
2. **Two cheap rule fixes now, independent of the decision** (both from the sources, both
   small): the AML-RB load-factor and speed law in the turning tactic (no dive to hold a
   speed above sustained), and the pre-merge offset as Reynolds' offset pursuit at about one
   sustained radius. They are proposals 1 and 2 of the ghosts notes. The sources back them,
   and they test whether geometry and energy, not the plan logic, are the main cause of the
   stalemate.
3. **Build option C as a second opponent** (`scripted` stays the yardstick), at first only
   for the engaged fight core, with the parameter set above. Judge it on: head to head
   against the yardstick (seats alternated), the loops measures (passes per pair, store by
   pass number, lateral separation at the pass), kills per minute, action entropy per
   situation for BC, the 32v32 memory, traced operations and cold compile time. Make it the
   yardstick only if it wins on strength and BC variety. If it does, the 1000-Elo anchor is
   re-based once.
4. **Treat some mirror-match stalemates as expected.** Measure stalemates against a mixed
   panel and the personas, not only yardstick vs yardstick. Break symmetry where the game
   allows (drawn personas, the drawn side, energy differences at spawn), as AML used random
   choice.
5. **Decisions for the author:** whether a second scripted opponent is worth building (C)
   or only the two fixes (A + 2); whether the plan logic stays if the two fixes resolve the
   stalemate measure; whether a named-doctrine `explain` is a requirement for the BC
   teacher.

## 7. Not verified

- McGrew et al. 2010: the exact score constants, the action set, the time step and the
  opponent model (primary PDF not fetched; formula as quoted by arXiv:2005.02057).
- Austin et al. 1990: candidate set, horizon and decision rule (abstract only).
- Park et al. 2016: the exact orientation formula's normalisation (PDF text extraction was
  garbled in that equation; the ICAS companion gives the product form).
- Falcon BMS, IL-2, FlightGear and OpenEaagles close-in AI: no public design notes found.

## 8. Links

- Burgin, Fogel, Phelps 1975, NASA CR-2582: <https://ntrs.nasa.gov/citations/19750022744>
- Burgin, Sidor 1988, NASA CR-4160: <https://ntrs.nasa.gov/citations/19890018022>
- Austin et al. 1990, JGCD 13(6), doi:10.2514/3.20590; AUTOMAN: <https://ntrs.nasa.gov/citations/19920031742>
- McGrew et al. 2010, JGCD 33(5), doi:10.2514/1.46815: <https://dspace.mit.edu/handle/1721.1/67298>
- Kurniawan et al. (quotes the McGrew score): <https://arxiv.org/abs/2005.02057>
- Park, Lee, Tahk, Yoo 2016, IJASS 17(2): <http://past.ijass.org/On_line/admin/files/8)(204~213)15-100.pdf>
- Lee et al., ICAS 2016: <https://www.icas.org/icas_archive/ICAS2016/data/papers/2016_0096_paper.pdf>
- Ernest et al. 2016 (ALPHA): <https://scholar.uc.edu/downloads/bc386t26n>
- DeMay et al. 2022 (APL, ADT): <https://jhuapl.edu/sites/default/files/2024-09/36-02-DeMay.pdf>
- Pope et al. 2023 (Lockheed, ADT): <https://arxiv.org/abs/2105.00990>
- LAG / CloseAirCombat: <https://github.com/liuqh16/LAG>
- BVRGym: <https://github.com/xcwoid/BVRGym>, <https://arxiv.org/abs/2403.17533>
- DCS World AI newsletter: <https://www.digitalcombatsimulator.com/en/news/newsletters/d48111b832efd4f57df12df8f46e7ca1>
- Reynolds 1999: <https://www.red3d.com/cwr/steer/gdc99/>
- Dill and Mark, GDC 2010: <https://gdcvault.com/play/1012841/Improving-AI-Decision-Modeling-Through>
- Liu, Pérez-Liébana, Lucas 2016: <https://arxiv.org/abs/1607.01730>
- Merz 1972: doi:10.1007/BF00932932
