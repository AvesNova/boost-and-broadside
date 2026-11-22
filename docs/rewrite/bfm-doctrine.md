# BFM doctrine for the scripted opponent

Research notes, 2026-11-22. They collect what published fighter doctrine says about the six questions below, so the scripted agent can fly basic fighter manoeuvres (BFM) the way the manuals teach them. Every passage quoted here was fetched and read; nothing is filled in from memory. Where my own reasoning or arithmetic goes beyond a source, it is marked **(derived)**. The "for the scripted agent" lists translate doctrine into deterministic rules for a 2D game. Their thresholds are proposals until the author decides.

Boost & Broadside is flat (top-down, no altitude), so the doctrine's vertical moves (pitch, slice, nose-high, nose-low, "the egg") have no direct equivalent. The manuals use altitude as an energy bank that can be traded for speed or turn. The game's energy store may play part of that role. That mapping is **(derived)** and is not doctrine.

## Sources

Fetched and read:

| Key | Source | Link | Notes |
|---|---|---|---|
| **P-1289** | Naval Air Training Command, *Flight Training Instruction: Basic Fighter Maneuvering (1v1), Section Engaged Maneuvering (2v1), T-45 Strike*, CNATRA P-1289 (Rev. 01-16), 2016 | https://web.archive.org/web/20201112042542/https://www.cnatra.navy.mil/local/docs/pat-pubs/P-1289.pdf | Full PDF from the Wayback Machine (the live CNATRA link is dead). Chapters 4–7, 12 and the Appendix A glossary were read. |
| **P-1210** | Navy *Air Combat Maneuvering, Flight Procedures* (T-45A), "(10-98) Original", P-1210 | https://navyflightmanuals.tpub.com/P-1210/index.htm | HTML transcription on tpub.com. I read pages 1–55, 75–77 and the glossary. I did not verify a CNATRA number on the page itself. |
| **P-821** | Department of the Navy, *Flight Training Instruction, Advanced Naval Flight Officer* (T-2C, VT-86), P-821 | https://navyflightmanuals.tpub.com/P-821/index.htm | HTML transcription. I read Chapter 9 (BFM theory, 9-1 to 9-26) and Chapter 10 BFM (10-29 to 10-55). Year not shown. |
| **AAS** | John R. Boyd, *Aerial Attack Study*, Report 50-10-6C, 1964 declassified version (re-created by Mark A. Hart, rev. 4 Jan 2016, page numbers match 1964) | https://code7700.com/pdfs/john_boyd/aerial-attack-study-1964_john_boyd.pdf | I read Part II chapters 1–2 (pp. 41–111). |
| **FM 1-15** | War Department, *Army Air Force Field Manual FM 1-15, Tactics and Technique of Air Fighting*, 10 April 1942 | https://ibiblio.org/hyperwar/USA/ref/FM/PDFs/FM1-15.PDF | WWII doctrine. I read paragraphs 16, 84, 91, 107–111. |
| **NACA 3130** | Clousing, Gadeberg, Kauffman, *Calculated and Measured Turning Performance of a Navy F2A-3 Airplane as Affected by the Use of Flaps*, NACA ACR 3130, Sept 1943 | https://ntrs.nasa.gov/api/citations/19930092640/downloads/19930092640.pdf | WWII flight test. I read the summary, introduction and conclusions (OCR text). |
| **Wetmore** | J. W. Wetmore, *Study of Turning Performance of a Fighter-Type Airplane Particularly as Affected by Flaps and Increased Supercharging*, NACA ACR, June 1942 | https://ntrs.nasa.gov/api/citations/20090016589/downloads/20090016589.pdf | I read the summary only (OCR is poor). |
| **VT-turn** | Virginia Tech AOE 3104 notes, "16. Turning Flight" (hosted under /lutze/, no date shown) | https://archive.aoe.vt.edu/lutze/AOE3104/turningflight.pdf | Equations read from rendered pages 2, 3, 9 and 10. |
| **P-825** | Navy *Intercept Procedures Textbook* (Strike Fighter Intercept, T-39N, VT-86), P-825 | https://navyflightmanuals.tpub.com/P-825/Leading-Turns-Accounting-For-Drift-In-Turns-145.htm | Only the pages on leading turns and displacement turns (pp. 58, 71, 135, 139). This is radar-intercept geometry, not BFM. |
| **WP-BFM** | Wikipedia, "Basic fighter maneuvers" (raw wikitext fetched 2026-11-22) | https://en.wikipedia.org/wiki/Basic_fighter_maneuvers | Secondary source. Used only where noted, and one error is flagged below. |
| **WP-EM** | Wikipedia, "Energy–maneuverability theory" | https://en.wikipedia.org/wiki/Energy%E2%80%93maneuverability_theory | Secondary source. Used for the P_s formula. |

Not fetched:

- **Robert Shaw, *Fighter Combat: Tactics and Maneuvering*** (1985). The archive.org copies (`fightercombattac00shaw`, `fightercombattac0000shaw`) are access-restricted lending items, and the full text returned HTTP 401. Nothing here is taken from Shaw.
- **AETC TTP 11-1** (USAF IFF fundamentals): the download returned an HTML error page.
- **USAF F-100 "Air Combat Maneuvering Phase" (Boyd collection, usmcu.edu)**: returned "Access Denied".

### Conflicts between the sources

These matter for the rules below:

1. **Wing loading.** P-821 (9-9) says the aircraft with higher wing loading has "a larger turn radius and a faster turn rate". P-1210 (p. 7) says "a larger turn radius and a slower turn rate". VT-turn Eq. 13 (turn rate ∝ √(1/(W/S)) at fixed C_L and n) agrees with P-1210, so P-821's "faster" is an error.
2. **Corner speed and sustained rate.**
   - P-821 (10-43) says "the best sustained turn rate will be obtained at an aircraft's corner speed".
   - P-1289 (4-4) gives corner 410 KIAS (best instantaneous) and best sustained 240–330 KIAS, which is a different speed.
   - P-1210 (p. 8) says corner gives "the maximum turn rate and the minimum turn radius" and adds "the Vn diagrams show only instantaneous turn performance".
   - WP-BFM defines corner speed as the minimum speed for the "maximum sustainable g-force load", which contradicts all three manuals ("max allowable g", P-821 9-10; "lift limit meets the load limit", P-1289 A-2).
   - **The manuals' definition is used here:** corner speed is where the lift limit meets the load limit.
3. **Which way "same direction" means.** P-821 (10-41) says 1C flow happens when "both aircraft turn in the same direction". P-1289 (A-3) says 1C is "one is turning left, the other is turning right" and 2C is "both in a left-hand turn". P-821 means the same geographic side of the pass; P-1289 means the same rotation sense. The P-1289 wording is the one a program can use (see §3).

---

## 1. One-circle against two-circle fights

### Passages

- **Definitions (P-1289, Appendix A, A-3, "Flow (1C/2C)"):** "If, after the merge, both aircraft turn across each other's tail, the flow is said to be "2C" because each aircraft is still on its own distinct turn circle. Both aircraft will be turning the same direction, i.e., both in a left-hand turn; note that in 2C flow, the two aircraft are fighting nose-to-tail. If, at the merge, one aircraft reverses towards the other aircraft, the flow is now said to be "1C" because both aircraft are now on the same turn circle but in opposite directions, i.e., one is turning left, the other is turning right. This is described as a nose-to-nose fight."
- **Who wins which (P-1289, 5-6):** "2C flow is a rate fight and the angular advantage usually goes to the aircraft with the higher turn rate at its corner speed. … During 1C flow, the fighter with the smaller turn radius will have the advantage. … Since turn rate is of little importance during 1C flow, it is often called a radius fight."
- **Sustained rate decides 2C (P-1210, p. 52):** "Unless time to kill is exceptionally short, the aircraft with the greatest sustained turn rate will win two-circle fights." Also, "When engaged in one-circle flow, you need to collapse your turn radius … This means transitioning to an airspeed that is both slower than his and controllable in terms of g available. 110 KIAS may well be slower than his airspeed, but it affords no ability to turn your jet."
- **Which aircraft picks which mind-set (P-1210, p. 9, "Game plans/mind-set"):** "A fighter who enjoys a superior weapons system and/or superior performance generally employs an energy-management type fight. … He is less willing to accept excursions from his corner airspeed and strives to keep separation from the bandit … At the other end of the spectrum is the aggressive position fight … greater willingness to trade airspeed for position … Against an adversary with better performance and/or superior weapons, this is the fight of choice. The greater the disparity in capabilities, the more aggressive the mind-set."
- **Energy state at the merge (P-1289, 7-2, "Controlling Merges"):** "The bandit may have an energy surplus over your own, allowing him to use the vertical; or, perhaps it would behoove you, in your current energy state (i.e., slow and on the deck), to force a 1C fight."
- **Thrust-to-weight (P-821, 9-9):** "Crews flying aircraft with thrust-to-weight ratios greater than one, will use different tactics than crews flying aircraft with ratios less than one." Wing loading: see the conflict note above.
- **Similar aircraft (P-821, 10-51):** "Since you are fighting against the same type of aircraft, you can engage in either a rate or a radius fight. Employ your game plan and drive the fight."
- **Merge speeds per game plan (P-1289, 7-3):** for 1C, "ensure you have enough energy to counter a pure nose-high maneuver, but not so much airspeed that when you set 1C flow, you will be unable to perform a lift-limit pull, and thus flush out in front of the bandit"; for 2C, "it is best to be at corner airspeed." The T-45 numbers (4-4) are nose-high game plan 350–420 KIAS, nose-low 270–350, level 370–410.
- **Game plans (P-821, 10-44 matrix):** for one circle, a turn-radius fight, "MAX KNOTS, CLOSE ABOARD, PITCH"; for two circle, a turn-rate fight, "280 [corner], CLOSE ABOARD, SLICE".
- **Forgiveness (P-1210, p. 55; same in P-821, 10-52):** "In two-circle flow, you keep a higher airspeed and since the fights normally take longer to develop, they tend to be more forgiving. … One-circle flow is much less forgiving … you probably won't see a gradual degradation of the fight. You will more likely go from a neutral or offensive position to a guns defense within the blink of an eye."
- **How the choice is made at the pass (P-1210, p. 49 and p. 55):** "One-circle flow occurs when one aircraft reverses at the merge". Also, "Reverse at the pass if necessary to create one-circle flow." And: "An uncooperative bandit may not allow you one-circle flow by reversing his own direction of turn following your initial reversal. You may be able to reverse your own turn once more, but by this time, the range between the jets is sufficient to negate the effects of turn radius. In other words, two circle conditions exist despite the appearance of one-circle flow. If you cannot work your game plan now, react to the bandit by establishing a competitive turn rate, and look for an opportunity to redefine later." For the energy fighter: "Hit the merge with the maximum knots possible and influence flow by initially turning across the bandit's tail."
- **The bandit always has a vote (P-1289, 12-4):** "After the check turn, set the desired flow. The bandit can always turn to change the flow. Proactive BFM will, however, force the bandit to turn in the direction that will set the flow the fighters desire. Reactive BFM allows the bandit to control the merge, making him less predictable."
- **Never a neutral pass (P-821, 10-51):** "To begin with, never accept a neutral pass. Always try to achieve an advantage in angles or energy at the pass." Also, "The crew that goes into a merge without a game plan will end up reacting to the bandit instead of forcing their will upon the bandit."
- **The perfect 180° pass (P-1210, p. 49):** "Assuming that neither jet is willing to give away any position advantage at the merge … you will pass with 500 ft … and 180 degrees off the tail." P-821 (10-50) says to check the instruments only "when the bandit is committed in their turn and is giving you a planform view … but not when the bandit is tail on or his direction of turn is still in doubt."
- **Pre-merge offset (AAS, pp. 106 and 110–111, nose-quarter attack):** the attacker dives for speed and turns "to offset point in the horizontal plane", which puts him "out to the side of your opponent on an anti-parallel course". He may "start the maneuver before the defender reaches his line-abreast position". The counter is: "Do not allow your opponent to offset. If he attempts to do so, turn toward the direction of the offset." If both fly it equally well: "If your opponent emulates your zoom-dive technique, you will be stalemated."

### Answer

- **1C (nose-to-nose).** The two aircraft fly one shared circle in opposite rotation senses. The smaller turn radius wins, and turn rate matters little.
- **2C (nose-to-tail).** Each aircraft flies its own circle, both in the same rotation sense. The higher turn rate wins, and over any long fight that means **sustained** rate.

The doctrine assigns them as follows:

- **2C / energy-management fight:** for the aircraft with superior performance (higher sustained rate, more thrust, more energy, better weapons). It merges near corner speed, keeps separation and avoids large speed excursions.
- **1C / aggressive position fight:** for the aircraft with inferior performance or less energy, or the one with the smaller minimum radius. It merges fast enough to be able to slow down and turn tight, but not so fast that it cannot reach its turn limit. It then trades speed for position. The bigger the performance gap, the more aggressive it should be.
- **Similar aircraft** can fight either. Recognising the flow first and flying the matching game plan gives the edge (P-1210, p. 50).

At the merge, either aircraft can create 1C by reversing (turning toward the other after the pass). 2C needs both to turn across the tails, and the opponent can counter-reverse at any time. A late re-reversal does not restore a radius fight, because by then the range has grown and the fight is a rate fight again.

In a perfect 180° pass, nobody "chooses" alone. Each side tries to arrive with an advantage: lateral offset, the right speed for its game plan, and an early turn (§2). The 2C fighter commits across the tail; the 1C fighter reverses. The manuals do not say "let the opponent commit first". They say never accept a neutral pass and drive the fight. They do say to watch whether the opponent's turn direction is "still in doubt" and to react once he is committed (P-821 10-50; P-1289 7-5: "If the bandit commits nose-low, wait until LOS accelerates, then roll …").

### For the scripted agent

1. **Game plan from a fixed comparison, computed once per engagement** (re-evaluated only on a merge):
   - Prefer **2C** if own sustained turn rate (and energy store, see §4) exceeds the opponent's by more than a margin.
   - Prefer **1C** if own minimum turn radius is smaller by more than a margin, or if own energy is clearly lower.
   - With similar aircraft, take a fixed default: the ship archetype's design intent, e.g. the turn fighter → 1C and the energy fighter → 2C **(derived)**.
2. **Merge speed target by plan:**
   - 2C: arrive at corner speed.
   - 1C: arrive above minimum-radius speed by the margin needed to still reach the lift limit after the pass. In the game that means speed in hand to bleed, never so fast that the ship cannot pull its tightest turn.
3. **Pass execution:**
   - 2C: turn across the opponent's tail, in the same rotation sense he will be in after the pass.
   - 1C: reverse at the pass (turn toward the opponent's side).
4. **One counter-reversal at most:** if I reversed for 1C and he counter-reversed, I may reverse once more only while range is under about one own turn radius **(derived threshold)**. Otherwise accept 2C and fly best sustained rate until the next merge.
5. **Deny his offset:** if before the pass the opponent's lateral offset from my flight path is growing, turn toward the offset side (AAS).
6. **No randomness:** the plan comes from the stat comparison; ties break by a fixed rule (e.g. ship-class priority).

---

## 2. The lead turn

### Passages

- **Blind lead turn (P-1210, p. 12; P-1289, 3-2, training rules):** "No blind lead turns. A blind lead turn is when your nose is out in front of the bandit's flight path, and you can't see the bandit." This defines a lead turn indirectly: a turn that puts your nose ahead of the bandit's flight path.
- **The defender gets lead-turned when he commits (P-1210, p. 43):** "A savvy bandit will anticipate your nose low move and lead turn you. You must honor this by either pulling for the overshoot … or redefining the fight … If you continue with the unload, you will most assuredly die."
- **Early turn into a merge (P-1289, 7-5, low-to-high merge):** "execute an early turn prior to the merge while ensuring the 500 ft bubble is maintained, and you can keep the bandit in sight (i.e., no blind lead turns). Time your turn to have your nose on the horizon when the bandit passes through the horizon. If the bandit attempts to counter the early turn by pitching into the vertical prior to your early turn, look to extend momentarily … to deny the bandit exclusive-use turning room. Place your lift vector in lead, post-merge, to force 1C flow." The counter for the other side is: "If … the bandit executes an early turn, counter by performing a lift limit pull as soon as you recognize the early turn. This will collapse weapons separation."
- **Turning room (P-1289, 5-2 and 7-3):** "Turning room is any separation that exists between two aircraft. Exclusive-use turning room is any turning room that is available to only one aircraft." Also, "Turning room belongs to whoever takes it. If you assess turning room exists, tighten down to take it out which will yield an angular advantage for you at the merge." And: "Check turn – as LOS rates take off, max perform across the bandit's tail to create a flight path crossing in the CZ. Check turns will take out any turning room from the bandit."
- **Lateral separation (P-821, 10-43):** "The rule of thumb is to take out all the lat sep possible at a neutral merge. … If your opponent allows greater lat sep at the merge, you will gain valuable angles on your opponent, which may result in a positional advantage. A smart bogey will take out the lateral separation resulting in a neutral merge."
- **Early turn in the scissors (P-821, 10-35 and 10-38):** "utilizing early/lead turns to force your opponent to flush out in front". In the roller: "get your nose down before the bandit gets theirs up (a.k.a. Early Turn)". P-1289 (5-16, flats): "Attempt to work in phase by performing early turns prior to flight path crossing."
- **Why turn before the line-of-sight passes (P-825, p. 139, intercept "leading the turn"):** "If the fighter were to wait to achieve the exact goal TA before turning … the bogey would continue to drift past the goal while the fighter was in the turn. This drift is due to the fighter's turn radius … The tighter the turn rate, the less effect the turn will have … the turn … must be made prior to reaching the goal TA. The further the turn, the more lead required. Similarly, more lead is required for shorter slant ranges because the bogey is drifting faster." Rule of thumb: "1° of drift for every 20° of turn". (Radar intercept, not BFM, but the same geometry.)
- **Speed set before the merge (P-821, 10-50):** "Airspeed etc. should be set as you approach the merge." P-1289 (7-3) says 2C "is best to be at corner airspeed", and P-1289 (A-2) says corner airspeed "provides the best instantaneous turn rate, at the highest bleed rates".

### Answer

A lead turn (the manuals more often say early turn) is starting your turn **before** the opponent passes your wing line. Your nose comes ahead of his flight path, using the turning room (lateral separation) between you, so that at the pass you have already turned some degrees he has not.

- **What it gains:** angles at the merge, and it removes the turning room the opponent could have used.
- **When to use it:** whenever turning room exists before the pass. Head-on with lateral offset, in scissors at each flight-path crossing, and against an opponent who commits early or deep into a predictable path (P-1210 p. 43, p. 55).
- **Limits:** keep sight (no blind lead turns), keep the safe pass, and do not flush out in front.
- **The counter:** the side being lead-turned honours it by tightening to its limit at once, which collapses the separation.

Why max turn performance at the merge: the manuals want the speed set before the merge, at corner for a rate fight, because corner gives the best instantaneous rate, and the first seconds after the pass decide who gains angles. They do not use the phrase "maximum turn performance at the merge". The closest are P-1289's corner speed for 2C and the 1C requirement to be able to pull to the lift limit after the pass.

### For the scripted agent

1. **Compute turning room before every pass:** lateral offset `d` = perpendicular distance from the opponent to my velocity line, and the line-of-sight (LOS) rate.
2. **Start the lead turn** when `d` is positive, the opponent is ahead of my 3/9 line, and his along-track distance to my wing line is under about own turn radius × `k` **(derived; `k` from the intercept rule that the turn starts before the drift arrives, larger for faster LOS rate)**. Turn toward his flight path, in the rotation sense of the game plan in §1.
3. **No blind lead turn:** if the agent has a sight cone, never start a lead turn that would put the opponent outside it before the pass.
4. **Being lead-turned:** if the opponent's nose crosses ahead of my flight path before the pass, go to the lift-limit (max-rate) turn at once, toward him.
5. **Arrive at plan speed:** on closure, with time-to-pass below a fixed horizon, adjust thrust or boost so that speed at the pass equals the plan speed from §1.

---

## 3. Recognising the fight

### Passages

- **The cues (P-1289, 5-7 and 5-8):** "Range … Change in range will indicate closure or separation. Target Aspect (TA): the angular difference between the bandit's nose and your aircraft. A bandit that is pointing directly at your aircraft has zero target aspect. … Angle off Nose (AON) … Angle off Tail (AOT) … the best way to assess the angular problem." Track crossing angle (A-7) is the "Angular difference in velocity vectors at any instant." WP-BFM notes that TCA "is often estimated by Heading Crossing Angle (HCA)".
- **Flow by rotation sense (P-1289 A-3, quoted in §1):** 2C means both turning the same way; 1C means one left, one right. Also (P-821, 10-43): "1 and 2 circle flow can occur in any plane … All that matters in determining the flow is how the two aircraft turn in reference to each other."
- **The bubble (P-1289, 5-2):** "An aircraft cannot turn inside its own bubble; this means that if you are inside another aircraft's bubble you are, at least momentarily, immune from attack from that aircraft. … by far the most important BFM concept". The T-45 template bubble is "3,000 ft turn radius". Control zone (5-3) is "2,000 ft to 4,000 ft behind a maneuvering aircraft, 20 degrees wide at the front and 40 degrees wide on the back side".
- **Flats against 1C (P-1289, 5-13):** "The flats is not a 1C fight. In the flats you are inside each other's bubble (the bandit's target aspect and position on the canopy remain relatively constant). In a 1C fight, you are on the outside of each other's bubble which allows for the nose-to-nose type of fight." Flats: "a slow-speed, high-AOA radius fight … attempting to minimize their downrange travel". Roller (5-18; A-6): "series of horizontal and vertical overshoots with both aircraft in each other's bubble".
- **1C decays into 2C with range (P-1210, p. 55):** after a counter-reversal, "the range between the jets is sufficient to negate the effects of turn radius … two circle conditions exist despite the appearance of one-circle flow."
- **In-plane against out-of-plane (P-1289, 7-3):** "In-plane (<45 degrees)" and "Out-of-plane (>45 degrees)". This does not apply in 2D.
- **Misaligned turn circles (P-1289, 5-6):** in 2C, "each aircraft's bubble is displaced over a different geographic point … Given the same exact turn performance, the geometry of MATC will enable an attacker's nose to eventually come to bear on the defender through pure geometry".
- **Defensive axioms (P-1289, 6-1):** "Attacker moving forward on canopy = Keep pulling. TA decreasing = Increase pull to hold attacker's nose off. If the bandit is <60º AOT and unable to perform steps 2 and 3, redefine."
- **Overshoots (P-1289, 5-5):** "Flight Path Overshoot – Flying through the adversary's flight path." "In-close Overshoot – A flight path overshoot that occurs with high AOT inside the near portion of the defender's CZ, within 2,000 ft". "3/9 Line Overshoot – … passing from behind the adversary to in front".
- **Reversal criteria (P-1289, 6-4):** "1. The bandit must have an ICFPOS (< 2,000 ft) with greater than 60 degrees TA. 2. The bandit must pass the extended six o'clock with a high line-of-sight rate. 3. You must be able to visualize reversing inside the bandit's turn radius. … If ever in doubt, do not reverse." Also (6-4): "if we observe a flight path overshoot where the attacker is in the CZ or aft and we attempt to reverse, we merely help his lag problem".
- **Scissors reversal timing:**
  - AAS (p. 56): "A good 'rule-of-thumb' is: rapid turn overshoot, early reversal; slow turn overshoot, late reversal."
  - P-1210 (p. 23): "when you are on or aft of the bandit's 3/9 line and your nose is on him, you should start your reversal. … If no nose-to-tail exists, reverse as you cross the bandit's flight path."
  - P-1289 (5-15): "To work in phase, turn aft of the bandit's 3/9 line prior to flight path crossing. If you notice a track crossing rate … and assess you are slower than the bandit, you should turn a little early to deny lateral room".
  - Defensive (6-7): "To work out of phase, reverse after the bandit crosses your extended six."
- **Who is winning a scissors (P-821, 10-38):** "if the bogey is in the forward part of your canopy most of the time, you are offensive and vice versa." The flats (P-1210 p. 22): "If you reduce your forward vector sooner than the bandit, you will gain the advantage." In the roller, a fighter that is neutral is slightly ahead at the bottom and slightly behind at the top (P-1289 5-21), so "relative position of the aircraft alone does not determine actual advantage".
- **Lufbery (P-1289 A-3):** "Horizontal or slightly oblique, stalemate-type engagement where both aircraft are across the circle from each other, turning in the same direction at a low energy state".
- **Arc (P-1289 A-2; P-1210 glossary):** "Arcing Turn (Arc): A turn executed at less than the optimum rate of turn, or an extension maneuver executed other than in a straight line."

### Answer

The manuals give cues rather than one classifier: range and closure, TA, AON, AOT, TCA, LOS rate, and each aircraft's turn direction. The usable rules are:

- **Flow is the relative turn sense after a pass.** Same sense means 2C (nose-to-tail); opposite sense means 1C (nose-to-nose).
- **Inside each other's bubble** (range under about one turn radius, TA and canopy position nearly constant, repeated flight-path crossings), it is a **scissors**. In 3D that is the flats; the roller needs the vertical. Outside the bubble, it is a 1C or 2C fight.
- **A 1C that has opened** past the point where radius matters is effectively 2C.
- **Both turning in the same sense across the circle** at low energy with no progress is a **Lufbery** (stalemate).
- **Who is ahead:** in 2C, the side whose nose comes to bear first. In a scissors, the side with less downrange travel, which keeps the other in its forward canopy.
- **Overshoots:** classified by where the attacker crosses the defender's flight path (inside the CZ near edge, about 2,000 ft for the T-45, with more than 60° TA, is "in-close") and whether he passes the defender's 3/9 line.

### For the scripted agent

All of the following are deterministic, computed each tick, with explicit hysteresis. Turn radius `R` is the agent's current lift-limit radius.

1. **Geometry:** range `r`, closure `−ṙ`, AOT (opponent's position off my tail), AON, TA (my position off his nose), HCA (heading difference), LOS rate, and the turn sign `s = sign(yaw rate)` for each ship, taken as zero when |yaw rate| is below a small threshold.
2. **Flow label:**
   - `s_me == s_him ≠ 0` → 2C.
   - `s_me == −s_him ≠ 0` → 1C.
   - Otherwise: unknown (keep the previous label).
   - Change the label only after the new value has held for `T_hold` ticks **(hysteresis by rule)**.
3. **Scissors:** if `r < R_me + R_him` **(derived: inside each other's bubble)**, |HCA| is large, and flight paths cross at least twice within a window, label `scissors`. Leave only when `r` exceeds that sum by a margin for `T_hold`.
4. **1C decays:** in 1C with `r > 2·R_me` **(derived threshold for "range negates radius")**, treat as 2C for decision purposes.
5. **Lufbery:** 2C with AOT for both sides in a band around 180° (across the circle), low speed (below best-sustained band) and no AOT trend for `T_stall` → label `stalemate`, which triggers the §5 disengage logic.
6. **Advantage measure** = (180° − own AOT on him) − (180° − his AOT on me). The rate of change of this measure says who is winning the turn. In scissors, compare along-track speed toward the downrange direction.
7. **Overshoot classification on the opponent's flight-path crossing:**
   - In-close if crossing distance < `x·R_me` and TA > 60° (P-1289 uses 2,000 ft, inside the 3,000 ft bubble).
   - 3/9-line overshoot if he passes ahead of my wing line.
   - Reverse only on in-close or 3/9 overshoots with high LOS rate; otherwise keep the turn ("if ever in doubt, do not reverse").

---

## 4. Rate against radius

### Passages and formulas

- **Level-turn kinematics (VT-turn, Eqs. 6–13):**
  - Load factor: `n = L/W = 1/cos φ`; `sin φ = √(n²−1)/n`; `tan φ = √(n²−1)`.
  - Radius (Eq. 9): `R = V²/(g √(n²−1))`.
  - Speed for a given n at a given C_L (Eq. 10): `V = √(nW / (½ρ S C_L))`.
  - Radius without V (Eq. 11): `R = [1/(½ ρ g C_L)] · (W/S) · n/√(n²−1)`.
  - Rate (Eq. 12): `ψ̇ = g √(n²−1) / V`.
  - Rate without V (Eq. 13): `ψ̇ = g √(½ρ S C_L / W) · √((n²−1)/n)`.
  - For both small radius and high rate: "Large C_L, low altitude (high density), high load factor, low wing loading".
- **Lift limit, the V-n diagram and corner speed (VT-turn, Eqs. 14–16 and p. 6):** the aerodynamic load-factor limit is `n_max,aero = V²/V_stall²`. "The flight condition on the V-n diagram at which this occurs [aero limit = structural limit] is called the corner point. … At the corner point we can obtain the max instantaneous turn rate and the minimum turn radius. It will be sustainable only if we have the thrust available to balance the drag at that load factor and speed."
- **Sustained turn (VT-turn, Eqs. 22–31)**, with drag polar `C_D = C_D0 + K C_L²`:
  - Thrust = drag gives (Eq. 25) `n² = (T q S − C_D0 q² S²)/(K W²)`.
  - Unconstrained maximum sustained rate (Eq. 28): `q = (W/S) √(K/C_D0)`, "the level-flight minimum drag speed".
  - If that violates C_Lmax, the speed for maximum sustained rate is (Eq. 30) `V² = (T/(½ρS)) / (C_D0 + K C_Lmax²)`.
  - If it violates n_max (Eq. 31): `C_D0 S² q² − T S q + K n_max² W² = 0`.
  - "In general, the conditions at maximum C_L generally yield the highest turn rate if the unconstrained result, Eq. (28) is not valid."
- **Specific excess power (WP-EM):** `P_s = V (T − D)/W`. P-821 (9-12): "the Ps = 0 curve can be used to determine the aircraft's best sustainable turn rate. … Where it peaks is where we find our best-sustained turn rate". E-M diagram contents (P-821 9-14): "1. Aircraft structural limitations. 2. Turn performance data (turn rate data and turn radius curves). 3. Specific power data in the form of Ps curves."
- **Lift-limited against load-limited (P-821, 9-10):** "Below this speed [corner], if you attempt to pull more 'G', the aircraft will enter buffet and stall at its aerodynamic limit. This results in an increase in the turn radius and a decrease in the turn rate. … if the aircraft is maneuvered above the corner speed, the max allowable 'G' becomes the limiting factor. The excess airspeed … will result in a turn radius increase and a turn rate decrease." Instantaneous against sustained (9-10): "Instantaneous turn rate: the max turn rate at any given airspeed (energy depleting). Sustained turn rate: the max turn rate available while sustaining energy."
- **Measured jet numbers. The speeds are three different bands (P-1289, 4-3 and 4-4, T-45, 10,000 ft):**

  | Speed | T-45 value |
  |---|---|
  | Corner airspeed (best instantaneous turn, highest bleed) | 410 KIAS |
  | Corner bleed rate | 8 knots/s |
  | Best sustained turn rate (P_s = 0) | 240–330 KIAS, "0.5° higher at 240 KIAS compared to 330 KIAS" |
  | Minimum radius airspeed band | 140–180 KIAS (all altitudes) |

  Above corner: "load limit pull, 7.33Gs". Below corner: "lift limit pull, 19-21 units (rumble of buffet)". "Compromise Performance" is "Load Limit Pull – Ease pull by 1-2Gs. Lift Limit Pull – Target 17 units (light buffet)". P-1289 (5-15): in the flats, "With minimum lateral separation and high fuselage alignment, flying slower than the minimum radius band may provide better results … With an increase in lateral separation however, we need to increase airspeed into the minimum radius band". P-1210 (p. 52): "your greatest turn rate will be achieved at 21 units at airspeed of approximately 410 KIAS. But you're not going to be able to maintain this pull for long."
- **T-2C (P-821, 9-14):** corner 280 KIAS; maximum instantaneous rate 21°/s; maximum sustained rate 19°/s; minimum turn radius at corner speed 1,500 ft; bleed at corner 63 ft/s.
- **Thrust at high angle of attack:**
  - AAS (pp. 55–56, scissors): "Maximum power should be used in this maneuver, because its force vector opposes the one-G gravity and is directed toward the inside of the turning circle. … since power provides a lower stall speed, it allows the defender to tighten his turn radius at a given airspeed as well as reduce airspeed to the lowest possible point."
  - NACA 3130 conclusion 2: "The effect of thrust on the maximum lift coefficient should be considered in predicting the turning performance of an airplane, especially in the low-speed region."
- **WWII flight test (NACA 3130, F2A-3, conclusions 3–16):**
  - "The radius of curvature for a given flap deflection is a nearly constant minimum over a wide range of speed."
  - At 13,000 ft: "For each flap deflection there is a speed at which the radius of curvature is a minimum, and for this turn the airplane is being flown at the maximum lift coefficient." "The minimum radius of curvature and the speed at which it occurs decrease as the flap deflection is increased." "At any airspeed above 112 miles per hour the radius of curvature is a minimum with flaps up".
  - At 27,000 ft: the minimum-radius turn is "not … flown at the maximum lift coefficient", because thrust limits it.
  - Summary: "for horizontal turns, the use of the standard airplane partial-span split flaps does not appear desirable for this airplane. For turns involving a loss of altitude, the turning radius is decreased by the use of the flaps."
- **Wetmore (summary):** "near the low-speed end of the speed range, the turning performance, as defined by steady level turns at a given speed, would be improved to some extent by any of the flaps … Fowler flaps and NACA slotted flaps appeared somewhat superior to split … Turning performance over the lower half of the speed range would probably not be materially improved … by increased supercharging … unless the propeller were redesigned to absorb the added power". It also notes there is "no simple and entirely adequate criterion" for turning performance.
- **Gravity (P-1289, 4-5):** radial g = load factor ± gravity component. Nose-high, gravity costs turn; over the top, it adds. This does not apply in a flat game.

### Answer

Rate and radius peak at different speeds because they depend on speed differently (`R ∝ V²/√(n²−1)`, `ψ̇ ∝ √(n²−1)/V`) and because the available n is capped twice:

- **Below corner (lift-limited):** `n = V²/V_stall²`. Rate rises with speed. Radius at C_Lmax follows Eq. 11, `R = (2(W/S)/(ρ g C_Lmax))·n/√(n²−1)`, so it is **almost flat**. **(Derived from Eq. 11:** n/√(n²−1) = 1.34 at n = 1.5, 1.15 at 2, 1.06 at 3, 1.01 at 7.3.) This matches the F2A-3 "nearly constant minimum over a wide range of speed".
- **Above corner (load-limited):** n is fixed. Radius grows as V² and rate falls as 1/V.
- **Corner speed:** the best *instantaneous* rate and, in the ideal model, the smallest instantaneous radius. It also bleeds energy fastest (T-45: 8 kt/s).
- **Sustained turns (P_s = 0):** thrust must equal the drag of the turn. The best sustained rate sits near the minimum-drag speed or at the C_Lmax/thrust intersection (Eqs. 28–31), well below corner (T-45: 240–330 against 410 KIAS).
- **Minimum radius in practice:** a low-speed band (T-45: 140–180 KIAS). The radius curve is flat there, thrust raises the usable C_L at low speed (NACA 3130, AAS), and slower flight cuts downrange travel. Real aircraft therefore show three speeds: radius lowest (slowest), sustained rate best (middle), instantaneous rate best (corner, fastest).
- **Flaps** raise C_Lmax. They shrink the radius at low speed or in descending turns, but cost radius above a crossover speed in level turns (F2A-3: 112 mph).
- **The E-M diagram** (called a doghouse plot by pilots, though no fetched source uses that word) plots turn rate against speed with: the lift-limit curve rising to the corner, the load-limit curve falling from it, the structural or speed limit, radius lines, and P_s contours. The P_s = 0 contour is the sustained-turn boundary; everything above it bleeds energy.

### For the scripted agent

1. **Derive three reference speeds from the ship's own physics** (offline, from the flight model), never hand-tune them:
   - `V_corner`: lift limit meets turn-rate or g cap.
   - `V_sustain`: argmax of turn rate on the P_s = 0 (or energy-neutral) curve.
   - `V_minR`: argmin of turn radius at maximum pull.
2. **Pull modes, as in the manuals:**
   - **Max:** the lift limit below `V_corner`, the cap above it.
   - **Compromise:** a fixed fraction short of the limit (P-1289: about 17 of 21 units, or the cap minus 1–2 g).
   - **Unload:** minimum pull, for acceleration.
3. **Speed targets by fight:**
   - 2C: hold `V_sustain`. Above it, pull max to bleed down; below it, compromise pull.
   - 1C or scissors: go to `V_minR` (or slower when inside his bubble with little lateral separation, P-1289 5-15).
   - Pursuit with a shot coming: allow corner-speed max pulls.
4. **Spend energy only for a purpose (P-1289, 7-2):** "take a shot, deny a shot, take or create exclusive-use turning room, deny exclusive-use turning room". Otherwise fly the energy-neutral pull.
5. **The game's energy store is the altitude analogue (derived):** spend it the way the manuals spend altitude (slice, nose-low) to hold `V_sustain` in a rate fight. Refill it during unloads and extensions.

---

## 5. Disengaging (extending)

### Passages

- **Definitions (P-1289, 12-3 and A-2):** "An extension is a short-term maneuver used to gain energy, distance, or separation with the intent to re-engage. … To perform an extension, unload … As a rule of thumb, a three-second unload will yield a twenty-knot airspeed increase. A short ease from a pull may be enough … However, easing a pull for too long is less effective than an extension because it will increase your turn radius and decrease your turn performance with little energy addition." Bug out: "To disengage from ACM in order to exit safely from the fight".
- **When (P-1289, 5-1 and 7-1):** offensive goal "if time to kill is up or you are losing the advantage, separate prior to becoming neutralized". High-aspect objective 6 is "Separate or bug prior to becoming defensive." P-1210 (p. 33): "Without a mission imperative, however, look to bug out first. Your odds of achieving a successful one are a whole lot better."
- **Bug criteria and mechanics (P-1289, 6-1 to 6-3):** "you must 'earn the right to bug.' … 1. 150°-180° pass – setting up a neutral merge and bugging towards the bandit's 6 o'clock will maximize the amount of degrees the bandit has to turn to arrive in a valid WEZ. 2. 500 ft pass". Then: unload, "Check turn 20°-30° away from the bandit's nose, then unload again. Check turn no more than twice". "Assess bug – if you look over your shoulder and the bandit is inside 1.0 nm with less than 90 degrees of turn remaining, you will not escape. If greater than 1.0 nm, or with more than 90 degrees of turn remaining, you should be outside of his weapons envelope. If you assess the bug is not valid, roll to place your lift vector on and perform a max performance break turn back into the bandit."
- **Rules of two (P-1289, 6-3):** "if your opening velocity plus your distance from the bandit equals the sum total of 2.0 or higher, the bug is successful and would kinematically defeat a missile launch. For example, if your airspeed is 400 KIAS and the bandit's is 300 KIAS, then you must be at least 1.0 nm away". The table uses (own − bandit airspeed)/100 kt + range in nm. For example, 390/300 at 1.1 nm gives 2.0 and succeeds; 390/300 at 0.8 nm gives 1.7 and fails.
- **Section bugout (P-1210, p. 75):** "achieve at least a 150-degree TCA between the engaged fighter and the bandit".
- **Neutral pass first (P-1210, p. 42):** "You are trying to force as neutral a pass as you can possibly get. That means maximum angles off the tail, minimum lateral separation. If you can achieve this merge with sufficient knots (you should have no lower than 300 KIAS), disengagement will likely be successful." Also, "unload for knots/extension. If you must check turn to keep sight, do it at 13 or 14 units and get right back on your unload. Do not arc!"
- **When unloading kills you (P-1210, p. 43):** "A savvy bandit will anticipate your nose low move and lead turn you. You must honor this … If you continue with the unload, you will most assuredly die."
- **Slow fights are hard to leave:**
  - P-1210 (p. 38): "disengaging at 130-150 KIAS is extraordinarily difficult, no matter how neutral the pass." Its remedy is the "butterfly": "a series of extensions/pitchbacks to separate your turning circle from the bandit's … You must initiate the pitchback before he brings his nose through more than 45 degrees or so of his turn … as the flat widens, it will begin to take on the characteristics of a two-circle fight".
  - Lufbery (p. 43): "avoid performing a one-move disengagement since the lateral separation and AOT are usually not great enough … begin a series of unloads and pullbacks to gain airspeed and nose-to-tail separation … Maintain your AOB to disguise your extension".
  - Rolling scissors (p. 27): "The only appropriate time to disengage is from the top of the roller."
- **Boyd (AAS):**
  - After a stalemated scissors, the defender "must employ afterburner, relax G and dive 180° away … immediately after the attacker has made his last observation". This "places the attacker in an overhead attack with a negative delta Mach" (p. 56).
  - Against a nose-quarter attacker who must turn 180°: "push over and dive for separation as your opponent moves past you on an anti-parallel course. Since your opponent must perform a 180° change of direction, this will provide you extreme longitudinal separation outside missile range" (p. 111).
  - Why continuing a turn away is fatal (p. 58): "the defender simply cannot refuse to reverse and continue his defensive turn. If he does … The defender will turn away and place his angular velocity cone in front of the attacker's flight path. As a consequence, the attacker will simply play his opponent's turn and maneuver toward the six-o'clock position."
- **WWII (FM 1-15, ¶16a):** "The unit possessing equipment with superior climb and speed can, within limits, choose the time and place for initiating the fight." ¶107: against non-fighters, pursuit has "ability to fight or withdraw at will due to superior performance." ¶84b: "care must be exercised to avoid overrunning the opponent, as this will result in placing him in a favorable position to fire on the attacker".
- **Defensive lost sight (P-1289, 6-4):** "stay in the direction that you were turning when you lost sight … reversing your turn typically helps the bandit gain a weapons solution."

### Answer

Separation works when four conditions hold:

1. It starts from a **near-180° pass**, with the opponent having to turn 150–180° before his weapons bear.
2. **Minimum lateral separation** at that pass, so he has no turning room.
3. **Speed in hand**, or an energy edge.
4. The opening speed plus range beats his weapon envelope. The Navy's kinematic check is the "rule of two": opening speed in hundreds of knots plus range in nm ≥ 2.0. The T-45 test is that he is beyond 1.0 nm *or* has more than 90° of turn to go.

It is fatal when started at the wrong moment:

- from a turning fight where he is inside about 1 nm with under 90° to go;
- from a slow fight (flats or Lufbery), where one move does not open enough AOT or range;
- when the opponent can lead-turn the unload;
- when the defender turns *away* while the attacker is still inside or near his turn circle. That rotates the defender's vulnerable cone toward the attacker, and the attacker simply follows his turn to the six.

If the check fails mid-bug, the doctrine reverses at once into a maximum break back toward him.

### For the scripted agent

1. **Arm a disengage only at a pass:** TCA ≥ 150°, his nose not on me, lateral separation below a small threshold (or I can remove it), and own speed ≥ the plan's floor.
2. **Separate (deterministic sequence):** unload (minimum pull, full thrust/boost), heading toward his six (away from his nose). At most two small check turns (20–30°) for sight; no arcing turns.
3. **Abort rule (evaluated every tick after the first 1–2 s):** if `r < R_weapon_him` **and** his degrees-to-go to point at me < 90°, abort. Break back toward him at max pull.
4. **Success rule (rule of two, scaled to the game) (derived):** `(v_me − v_him)·τ + r ≥ k·R_weapon_him`. Here τ and k come from the game's weapon range and time of flight, not from the T-45 numbers. Once it holds for `T_hold`, the separation is complete and the agent may re-engage on its own terms.
5. **Forbidden disengages:** from scissors or stalemate at low speed in one move. Use the butterfly: extend, then pitch back before he has turned more than about 45°, and repeat until the pass is fast and wide.
6. **The energy fighter (2C plan) may extend repeatedly to rebuild speed.** The turn fighter (1C plan) disengages only when losing.

---

## 6. The cost of indecision

### Passages

- **Arcing (P-1289 A-2; P-1210 glossary):** "Arcing Turn (Arc): A turn executed at less than the optimum rate of turn, or an extension maneuver executed other than in a straight line." P-1210 (p. 42): "Do not arc!" P-1289 (12-3): "an arcing engaged fighter is giving the bandit an opportunity to achieve an offensive position and/or a WEZ."
- **Half-measures (P-1289, 12-3):** "easing a pull for too long is less effective than an extension because it will increase your turn radius and decrease your turn performance with little energy addition."
- **Reversals (P-1289, 6-4 and 6-5):** "If ever in doubt, do not reverse." Reversing after a control-zone overshoot "merely help[s] his lag problem and aid[s] in our own demise." After lost sight: "reversing your turn typically helps the bandit gain a weapons solution."
- **Repeated reversals at the merge (P-1210, p. 55):** after the bandit's counter-reversal, "You may be able to reverse your own turn once more, but by this time, the range … is sufficient to negate the effects of turn radius … react to the bandit by establishing a competitive turn rate".
- **Turning level, reacting (P-1289, 7-8):** "Fighter turning level is a reactionary gameplan … It provides options to counter the bandit but does not typically yield an offensive advantage. Against an aggressive bandit, angles can still be lost in the long term because you are behind every move the bandit makes." P-821 (10-54): "turning level across the horizon is not a good game plan because you are not maximizing your jet for any particular fight."
- **No plan means reacting (P-821, 10-51):** "The crew that goes into a merge without a game plan will end up reacting to the bandit instead of forcing their will upon the bandit." P-1289 (7-5): "strive to think one merge ahead to develop a proactive fighting stance, instead of simply reacting".
- **Scissors deadline (AAS, p. 56):** "When employing the scissors maneuver, a pilot should attempt to secure an advantage as quickly as possible – certainly by the second turn reversal. If not, he will lose airspeed rapidly, therefore maneuverability, and will be prevented from ever acquiring an advantage."
- **Unload after being lead-turned (P-1210, p. 43):** "If you continue with the unload, you will most assuredly die." This is the opposite failure: holding a decision after it has been countered.
- **First error (P-1210 p. 55; P-821 10-52):** "With similar aircraft, the fight normally will go to that fighter who makes the first error."
- **Sustained rate (P-1210, p. 52):** "When you recognize that the flow is two-circle, any airspeed below 300, or any pull less than 17 units (when above 300 KIAS) will significantly reduce your turn rate".

### Answer

The manuals do not use the word "racetrack", and no fetched source quantifies the loss from one reversal. What they do say is consistent:

- A reversal or an eased pull spends seconds turning at less than the best rate, or turning the wrong way. Each such second is degrees the opponent gains for free. In 2C he keeps turning at full rate while you do not ("arcing").
- A reversal that does not meet the overshoot criteria helps the attacker.
- A second re-reversal at the merge only produces a rate fight from a worse position.
- Reacting move by move ("turning level", no game plan) loses angles in the long run.
- The cure is a game plan committed to before the merge, a fixed set of reversal criteria, and a deadline: win the scissors by the second reversal or redefine.

**(Derived)** In 2C, a pilot who flies `a` seconds at rate `ω_low` instead of `ω_max` gives away `(ω_max − ω_low)·a` degrees. A full reverse-and-return gives away roughly twice the heading change spent going the wrong way. This is the geometry behind the manuals' advice, not a quoted rule.

### For the scripted agent

1. **Commit by rule, change by rule:**
   - The game plan (§1) is fixed at the merge.
   - Turn direction changes only on a listed trigger: an in-close or 3/9 overshoot (§3.7), the scissors reversal timing (§3: reverse when he is on or behind my 3/9 line with my nose on him, or at flight-path crossing if there is no nose-to-tail; rapid overshoot means early, slow means late), or the §5 abort.
   - After any reversal, lock the turn direction for `T_lock` ticks or until the next flight-path crossing, whichever comes first.
2. **No arcing:** in a turning fight the pull is always one of {max, compromise, unload}, chosen by the §4 speed rules. No in-between "ease" lasting longer than a fixed few ticks.
3. **Merge re-reversal limit:** one counter to a counter-reversal at most (§1.4). After that, fly 2C at `V_sustain`.
4. **Scissors deadline:** if the advantage measure (§3.6) is not improving by the second flight-path crossing, redefine. Go to 2C at speed if own sustained rate is better; otherwise use the butterfly or disengage (§5).
5. **Stalemate breaker:** in a Lufbery or a neutral scissors for `T_stall`, run the §5 extension sequence instead of hunting with reversals.

---

## What I could not settle

1. **Shaw's text.** *Fighter Combat* could not be fetched (archive.org lending only). Its treatment of the lead turn, one-circle and two-circle choice for dissimilar aircraft, and the "angles against energy fighter" distinction is not in these notes. WP-BFM cites Shaw p. 105 for "angles fighters", but I did not verify that against the book.
2. **A standalone definition of "lead turn".** None of the fetched manuals defines it directly. They define the "blind lead turn" and use "early turn" and "lead turn" in context, and the definition above is assembled from those. A threshold for *when* to start a BFM lead turn (range against turn radius against lateral separation) is not given; only the radar-intercept rule of thumb exists (P-825).
3. **"Let the opponent commit first" in a perfect 180° pass.** No fetched source states this as doctrine. The manuals say "never accept a neutral pass", drive the fight, and react after the bandit's direction is no longer "in doubt". Whether the scripted agent should wait has to be decided by testing.
4. **Numeric thresholds in 2D.** The T-45 numbers (3,000 ft bubble, 2,000–4,000 ft CZ, 1.0 nm / 90° bug rule, rule of two) are aircraft- and weapon-specific. Every scaled threshold above (k·R, 2·R for "range negates radius", T_hold, T_lock) is **derived** and needs measurement in the game.
5. **The "racetrack" against a true circle.** No fetched source uses the term or quantifies the angle loss from a reversal or from a non-circular turn. §6 rests on arcing, reversal criteria and the AAS deadline, plus my arithmetic.
6. **Why the T-45 minimum-radius band (140–180 KIAS) lies so far below corner (410 KIAS).** The ideal model (Eq. 11) puts the minimum instantaneous radius at corner, with a nearly flat curve below it. P-1289 does not explain the band. Low-speed thrust effects (NACA 3130, AAS) and the difference between sustained and instantaneous radius are plausible reasons, but I did not find a source that states the cause for this aircraft.
7. **The vertical.** Most game-plan detail (pitch against slice, the egg, the roller, nose-high/low merges) depends on altitude and gravity. How much of it the game's energy store can stand in for is a design question, not a doctrine one.
8. **WWII BFM specifics.** Only FM 1-15 (general WWII doctrine, 1942) and two NACA turning studies were read. No WWII pilot-training text on one-circle or two-circle choice or the lead turn was fetched. The 1C/2C vocabulary in these sources comes from later Navy training manuals.
9. **Source wording conflicts** (wing loading, sustained rate at corner, the meaning of "same direction", the WP-BFM corner-speed definition) are listed under Sources. The rules above follow the reading that matches the physics (VT-turn) and P-1289.
