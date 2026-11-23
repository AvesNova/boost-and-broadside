# Shaw's BFM core, distilled for the scripted BFM design

Robert L. Shaw, *Fighter Combat: Tactics and Maneuvering* (1985), the author's own copy. Read 2026-11-23 for
docs/rewrite/scripted-bfm-design.md (E2). Pages are given as printed page / PDF page (printed = PDF - 19). Quotes are
short and exact from the text layer; everything else is paraphrase. Figures are referenced by number; the rendered
images and a figure list are outside the repo (data/references/shaw/figures/, README.md there).

Read: Ch 1 Gun Employment and Guns Defense (pp. 15-31 / PDF 34-50); Ch 2 BFM (pp. 62-96 / PDF 81-115; lag rolls,
rolling scissors and defensive spiral skimmed); Ch 3 guns-only angles and energy fights, comparison, defensive
manoeuvring (pp. 99-113, 130-138 / PDF 118-132, 149-157); Ch 4 intro and the dissimilar angles fight (pp. 139-150 /
PDF 158-169), plus a skim of its energy fight (pp. 151-160 / PDF 170-179); Appendix instantaneous turn, energy,
acceleration, sustained turn (pp. 387-395, 405-411 / PDF 406-414, 424-430). Key plan-view figures were read as images.

Shaw's terms: AOT angle off the target's tail; AON angle off the nose; TCA track-crossing angle, "the angular
difference in velocity vectors at any instant" (p. 75 / PDF 94); flight-path separation, "the perpendicular distance
from the attacker to the extended flight path of the defender" (p. 76 / PDF 95). He never says one-circle or
two-circle: nose-to-nose = one circle, nose-to-tail = two circle (Fig 2-11, p. 78 / PDF 97).

## 0. The one thing that changes when Shaw is read against our game

Shaw's nose-to-nose (1C) argument rests on "turn radius is minimized ... by pulling max-G at or below corner speed"
(p. 101 / PDF 120) and on slower speed meaning a tighter radius: "Since sustained turn radius is usually more
sensitive to speed reduction than is sustained turn rate ..., nose-to-nose turns generally provide the greatest angular
gain per knot of speed loss" (p. 100 / PDF 119). His Fig A-3 (p. 392 / PDF 411) already shows instantaneous radius
nearly flat between just above stall and corner speed: "little change can be expected in RT at any speed between Vc
and slightly greater than Vs" (pp. 391-392 / PDF 410-411). The game has exactly this shape (204-238 m below corner,
rate 0.65 to 1.19 rad/s), and in the game the best sustained rate and the minimum sustained radius are at the same
speed (v_st), whereas Shaw puts minimum sustained radius well below best sustained rate speed, "generally 1.4 to 1.5
times power-on stall speed for jets" (p. 409 / PDF 428).

Consequences for us:

- Between two equal ships at full stick (both settle at v_st, equal radius) a 1C fight is close to neutral: Fig 2-14
  case 1 and Fig 2-12 show that equal radius gives no gain. The levers left in 1C are (a) speed when the two are
  closer than a turn radius (Fig 2-13 case 1, "relative speed is the primary factor", p. 79 / PDF 98), (b) the
  opponent's radius growing as v^2 above corner speed, and (c) the dive's freed 1 g at low speed (204 m at 107 m/s
  diving against 233 m at v_st coasting, design table), which is about 12% (Shaw calls aircraft dissimilar above
  "about 10 percent", p. 139 / PDF 158).
- In 2C, rate decides ("primarily turn rate that produces advantage in nose-to-tail maneuvers", p. 80 / PDF 99), and
  in the game rate is bought with speed toward corner, i.e. with the store (about 140 deg per full store).
- So the store is worth much more in 2C than in 1C. That is a cleaner reason for the design's plan rule than Shaw's
  own: the side with the energy goes 2C to cash it as rate; the side without goes 1C, where rate "has only a minor
  effect" (p. 79 / PDF 98) and his store buys little.

## 1. Nose-to-nose (1C) against nose-to-tail (2C)

Rule (Ch 2, pp. 77-81 / PDF 96-100, Figs 2-11 to 2-14):
- Nose-to-nose: smaller radius wins; turn rate "has only a minor effect" (pp. 78-79 / PDF 97-98, Fig 2-12 case 2: the
  bigger-radius, faster-rate bogey gains almost nothing).
- The weight of radius against speed depends on the maximum separation, reached at parallel headings (Fig 2-13,
  pp. 79-80 / PDF 98-99):
  - separation below the larger turn radius: "relative speed is the primary factor" (the faster ship "just flies out in
    front");
  - between the larger radius and the larger diameter: both matter, "radius is the dominant factor";
  - beyond the larger diameter: little gain at all, "Relative speeds contribute essentially nothing".
  - After a head-on pass, "the maximum separation that will result from a subsequent immediate nose-to-nose turn is
    about one turn diameter" (p. 101 / PDF 120), so a 1C fight after a close pass sits in the radius-dominant band.
- Nose-to-tail: "primarily turn rate that produces advantage"; equal rate with smaller radius stays neutral
  (Fig 2-14, p. 80 / PDF 99). A radius advantage helps in 2C only through lead turns when separation at the pass grows.
- 2C keeps the fighters farther apart (sight harder, escape easier for the opponent, p. 81 / PDF 100); a
  faster-turning 2C opponent finishes his turn at longer range and gets a "period of acceleration" before his lead turn
  (Fig 3-5, p. 108 / PDF 127).
- Energy cost: for the same load factor, "nose-to-nose dynamics result in its gaining a given angular advantage in about
  half the time" of nose-to-tail (p. 109 / PDF 128). Hence Shaw's angles tactics are built on the 1C turn (p. 100 /
  PDF 119).
- Who chooses what:
  - Similar aircraft, angles tactician: nose-to-nose (p. 100 / PDF 119).
  - Similar aircraft, energy tactician: a level sustained nose-to-nose series for sight (p. 106 / PDF 125), but "a
    nose-to-tail turn series is actually superior ... for the purposes of bleeding the bogey's energy" when sight is no
    problem (p. 108 / PDF 127).
  - Low wing loading (turn fighter): nose-to-nose, lead turns, flat scissors; keep the fight horizontal (p. 142 /
    PDF 161). Its sustained-rate edge in 2C "would be very slow and is definitely inferior to the nose-to-nose
    technique" (p. 143 / PDF 162).
  - High T/W (energy fighter): "Nose-to-tail geometry is usually preferable", lag pursuit, vertical/oblique (p. 142 /
    PDF 161).
  - Defender with a speed (energy) deficit: "The nose-to-nose turn is the defender's primary tool for bleeding the
    energy of a faster opponent" (p. 135 / PDF 154, Fig 3-13).
- Large angular advantage at a pass: keep the turn going nose-to-tail rather than reversing. Similar: "take advantage of
  the gift and continue ... in the nose-to-tail direction" (p. 102 / PDF 121). Dissimilar: "once the angles fighter has
  gained about 60° to 70° angular advantage, it is probably better for it to continue nose-to-tail" (p. 146 / PDF 165).
- Bogey's choice after the pass in the dissimilar case: if the bogey goes straight or turns away (making it 1C), keep
  the turn; "Should the bogey turn toward the attack, however, a turn reversal is called for" (p. 143 / PDF 162). The
  angles fighter re-establishes 1C after each neutral pass by reversing (Fig 3-1).

Game: see section 0. The plan rule by energy is supported; its strength in the game is the store's value in 2C, not a
radius gap. Add: keep 2C (do not reverse) when my angular advantage at the pass is already large (Shaw: 60-70 deg,
dissimilar case).

## 2. Angles fight and energy fight

Definitions (p. 99 / PDF 118): angles first seeks "a position advantage (angles), even at the expense of relative
energy"; energy seeks "an energy advantage ... while not yielding a decisive position advantage", then converts it
"usually without surrendering the entire energy margin".

Angles fight, guns only (pp. 100-104 / PDF 119-123, Figs 3-1 to 3-3):
- Engage at corner speed; above it, zoom to shed the excess without losing total energy (p. 101 / PDF 120). Prop
  fighters (best climb below corner) should "accelerate to corner speed, or as close to it as possible" (p. 102 /
  PDF 121).
- Before the pass, turn away to build lateral separation, then reverse into a lead turn (Fig 3-1, time 2).
- After a neutral pass, reverse to nose-to-nose and pull max G at corner, nose-low "trading altitude for angles, rather
  than bleeding excessive airspeed" (p. 101 / PDF 120), then relax G to regain altitude and conserve speed (p. 102 /
  PDF 121).
- Limits: stay within "about one-quarter of a turn radius or one-half of an effective guns range, whichever is less, of
  the bogey's altitude", and never below the speed needed to go purely vertical. With these, expect "between 20° and
  30° on a similar opponent on the first turn" (p. 102 / PDF 121). "The angles tactician cannot afford to get too
  greedy."
- If the bogey does not reverse after a gained pass: press in the nose-to-tail direction "using alternate low and high
  yo-yos ... to make repeated low-to-high gun passes ... small angular gains on each pass" (p. 103 / PDF 122).

Energy fight, guns only (pp. 104-111 / PDF 123-130, Figs 3-4, 3-5):
- Not satisfied with a neutral start; lead turn in a level plane, not bleeding below "best sustained-turn-rate speed or
  vertical-maneuvering speed, whichever is greater" (p. 106 / PDF 125).
- In the nose-to-nose series, the energy fighter keeps "the slowest possible speed consistent with vertical-maneuvering
  potential" (p. 107 / PDF 126) and lets the bogey gain, monitoring the gain per pass: "Large bogey gains (20° to 30°)
  indicate a substantial speed advantage for the energy fighter. Smaller angular gains (10° or less), however, could
  mean only minimal speed differential" (p. 107 / PDF 126).
- Concede no more than about 90 deg: zoom when the bogey has "between 60° and 90° of angular advantage" (p. 107 /
  PDF 126); "The bogey cannot be allowed to gain much more than 90°" (p. 110 / PDF 129); "As long as the bogey's
  angular advantage is limited to approximately 90° ... separation and termination of the engagement are possible
  throughout" (p. 112 / PDF 131).
- In nose-to-tail: speed "about the speed for maximum sustained turn rate"; "Load factor, however, should be held to
  only about two-thirds of the maximum sustained-G capability at this speed, and the remaining PS should be used for
  climbing" (p. 110 / PDF 129). Increase G and go level or nose-low if the bogey turns very hard or will not follow.
- Energy fighter with a 90 deg advantage of its own: lag pursuit in a nose-to-tail turn at best sustained rate to bleed
  the bogey (pp. 110-111 / PDF 129-130).
- Lufbery: a diving spiral near corner speed can hold a similar bogey in a 2C turn "as long as altitude allows" (p. 109
  / PDF 128); the stalemated nose-to-tail turn.

Choice and switching (pp. 111-113 / PDF 130-132):
- Angles: aggressive, quick, riskier ("betting the farm" once committed to the vertical, p. 112 / PDF 131). Energy:
  patient, safer. "Unless a pilot has extensive training in energy methods, the cautious use of angles tactics is
  probably preferable"; but "take what the opponent will allow": a tentative opponent gets angles, an aggressive one
  should be met by the "finesse" of energy tactics (p. 112 / PDF 131).
- Angles to energy: if the opponent "matches angle-for-angle through one or two turns", convert "to a nose-to-tail turn
  at the next pass and relaxing the G to the minimum value required to keep sight", accelerating toward best
  sustained-rate speed (or climbing in a low-G turn above it), "adjusting G and climb/dive angle so that the enemy gains
  about 90° by the next pass", then trade the excess speed for height (p. 112 / PDF 131). If very slow, dive in the turn
  first (p. 113 / PDF 132).
- Energy to angles: if the opponent "refuses to take the angular advantage offered", start angles "normally by starting a
  low yo-yo from across the circle" (p. 113 / PDF 132).
- "Once a trend is established in one-versus-one ACM, it is seldom reversed without a serious mistake on the part of the
  winning pilot" (p. 130 / PDF 149); losing an advantage you had is the time to bug out.

Game: the store is the altitude. "Zoom to shed speed above corner" = climb above v_c (the design's merge speed with
the sustained cap). The energy fighter's 2/3-G-and-climb is "ease the stick and bank while turning"; the game's
sustained level (0.70) is close to Shaw's two-thirds if the stick level maps to load factor (to check). The "zoom
once he has 60-90 deg" has no position analogue in 2D (no vertical separation); its energy half is "stop conceding,
cash the store at corner as rate". The tactic switches are rules the design does not have yet (section 11).

## 3. Corner speed, instantaneous and sustained turn

- Corner speed: intersection of lift limit and structural limit; "a fighter attains maximum instantaneous turn
  performance" there (p. 390 / PDF 409, Fig A-1). Above it rate falls and radius grows; below it radius is nearly flat
  down to just above Vs, then rises steeply (Fig A-3, pp. 391-392 / PDF 410-411). RT ~ V^2/n, TR ~ n/V (p. 390 /
  PDF 409).
- Minimum radius averaged over a short turn: "commencing the turn at about corner speed, pulling maximum available G,
  and allowing speed to decay as the turn is completed" (p. 146 / PDF 165).
- Sustained: best sustained rate "slightly below that for maximum sustained G"; minimum sustained radius at "1.4 to 1.5
  times power-on stall speed for jets" (p. 409 / PDF 428), about 1.5 for jets "(usually somewhat slower for prop
  fighters)" (p. 137 / PDF 156).
- Speed control: "Pulling too great a load factor will cause speed to bleed off below the optimum value, resulting in
  reduced sustained turn. This lost speed can be regained only by relaxing G ... or by diving" (p. 409 / PDF 428).
- Power-limited fighters "unable to maintain corner speed at maximum G" turn best in nose-low spirals (p. 87 /
  PDF 106); "If speed ever increases to above corner velocity in the spiral, deceleration is in order" (p. 96 /
  PDF 115).
- Unloading (0 G) gives the fastest acceleration, most at low speed (p. 405 / PDF 424, Fig A-16).
- Low-wing-loaded fighter: "best sustained-turn-rate speed, rather than corner velocity, is normally the best engagement
  airspeed" and "should maneuver only as hard as necessary" (p. 143 / PDF 162).

Game: a speed tracker with v* at v_st for the turning fight and v_c only to buy a shot or a lead turn is what Shaw's
speed-control passages say. "Below the optimum: relax or dive" is the design's power law (dive when v < v*). The dive
in a hard turn below corner (today's DIVE_IN_TURN) is Shaw's nose-low spiral; Shaw keeps it bounded (altitude limits,
section 2), which is the "spend only for a purpose" rule.

## 4. Pursuit: lead, pure, lag

Definitions by velocity vector ahead of, at, or behind the target (p. 62 / PDF 81, Fig 1-7).
- Lead: "primarily to increase closure on the target by use of geometry" (p. 63 / PDF 82); best lead holds the target
  stationary against the horizon (proportional navigation). It lets even a slower attacker close on a target arcing
  toward him (Fig 2-1), but needs ever tighter turning and "results in increasing AOT" (p. 64 / PDF 83).
- Pure: closes "unless the target has a significant speed advantage and AOT is very small"; less closure than lead but
  AOT grows more slowly (p. 65 / PDF 84).
- Lag: slows or stops closure while holding or reducing AOT; lets a faster fighter stay in the rear hemisphere (Fig 2-3,
  pp. 65-66 / PDF 84-85). Relaxing the turn into lag is also "a good opportunity to increase his energy by
  accelerating". Stabilised lag "is generally a temporary state of affairs"; it needs a speed advantage and the same
  turn rate.
- Counters: hot-side lag by tightening the turn (nose-low); cold-side lag by a reversal, which against a gun-armed
  attacker "usually results in at least a snapshot opportunity for the attacker" (p. 67 / PDF 86).
- Closure management: "High closure is desirable at long range ... But as firing range is approached, the rate of closure
  should be reduced"; "The attack should be planned so that the firing position (preferably in the target's rear
  quarter) is achieved just as desired firing range is reached"; "The opponent must be beaten first, and then shot"
  (p. 18 / PDF 37). Continuous tracking needs the shooter slower than the target; "Some speed advantage is usually
  preferable, which inevitably results in closure" (p. 18 / PDF 37).
- Break off: "whenever he is unable to maintain proper lead ... Further turning in the target's plane of maneuver usually
  results in excessive loss of speed and often leads to an overshoot. Instead, the shooter can reposition" (p. 19 /
  PDF 38).
- Against a speed-advantaged runner: "Turning during the run-out (arcing) allows even a slower fighter to close the range
  by flying across the circle" (p. 23 / PDF 42).

Game: τ > 0 lead to close at range, τ < 0 lag to hold position and bank, gun lead when the shot gate is on: confirmed.
Shaw adds that lead should be taken early so the target flies into the aim point (section 9), and that the switch from
lead to lag is driven by closure near the firing range, which is what the design's κ* does.

## 5. High and low yo-yo

High yo-yo (pp. 71-73 / PDF 90-92, Fig 2-6):
- Purpose: prevent an overshoot and reduce AOT by going out of plane instead of pulling more G, "thereby allowing the
  attacker to retain greater energy".
- Trigger: "best suited to conditions of moderate AOT (about 30° to 60°), when the attacker is more nearly co-speed with
  the defender"; the attacker is in pure pursuit "with rapidly increasing AOT and closure".
- Stop: roll back toward the bogey "As the closure slows to nearly zero"; choose lead, pure or lag by the nose-tail
  separation and the range wanted.
- Errors: starting too late (too nose-high, range opens, defender extends); greed (lead option from the top, too
  nose-low, vertical overshoot). Without a turn-performance advantage, "the attacker should not expect an immediate
  lethal firing position" (p. 73 / PDF 92).
- At low AOT (below about 30 deg) with high overtake at close range, "simply relaxing the turn may not slow the closure
  fast enough to prevent overshooting" (p. 67 / PDF 86); speed reduction, a hard turn away, or an out-of-plane move
  are the options, the last usually best (pp. 67-68 / PDF 86-87, Figs 2-4, 2-5).

Low yo-yo (pp. 73-74 / PDF 92-93, Fig 2-7):
- Trigger: "a fairly long-range (i.e., probably hot-side) lag-pursuit situation, where the attacker does not have the
  turn capability to pull his nose to the target quickly for a shot, or where doing so would cause excessive speed
  loss".
- Method: nose low inside the turn for excess lead and speed, then fly a straight intercept so that "gun-firing range and
  lead are reached simultaneously".
- Limits: the larger the lead, the larger the AOT at intercept; "Generally, it is more prudent to close the range a
  little at a time in several steps"; a low yo-yo can be followed by a high yo-yo; "In general, two small yo-yos (high or
  low) are safer than one big one."

Game: climb = the yo-yo's energy half (bank speed, closure falls, energy kept); easing the stick = its pursuit half
(lag). Full stick and climb banks nothing below 1.0 v_c in the game, so the high yo-yo must ease the stick to bank:
the design's "sustained cap during a high yo-yo" is required, not optional. The low yo-yo = dive + lead. Shaw adds two
conditions the design's closure signal lacks: AOT about 30-60 deg for the high yo-yo (below that, a plain lag is not
enough), and small steps (a short yo-yo time).

## 6. Lead turn at the merge

- Definition: an "early" turn started before passing the opponent in a forward-hemisphere approach; "It does not
  necessarily connote lead pursuit, and in fact it may be a lag maneuver" (p. 74 / PDF 93, Fig 2-8).
- Rewards and risks: the earlier, the greater the reward, but too early and the attacker passes in front of the
  defender's nose (danger against a gun). Earlier turn means closer range and smaller TCA at the overshoot, both
  raising risk; a slower or tighter-turning attacker can start sooner (pp. 75-76 / PDF 94-95).
- Separation (Fig 2-9, p. 76 / PDF 95): attacker radius half the separation gives about 180 deg; radius equal to the
  separation about 90 deg. "the potential angular advantage of a lead turn against a nonmaneuvering opponent is
  proportional to the ratio of flight-path separation to attacker turn radius when the turn is commenced." Better
  turn rate lets the turn start closer, leaving the opponent less time to counter (turning toward to cut the
  separation).
- Hence tight-turning fighters benefit most from separation, and "less-maneuverable fighters [should] reduce
  flight-path separation to a minimum by attempting to pass as closely as possible" (pp. 76-77 / PDF 95-96). The
  energy fighter does this by turning toward the bogey before the pass (p. 157 / PDF 176, Fig 4-7).
- Without separation (Fig 2-10, p. 77 / PDF 96): possible but it gives the opponent separation; too early and he uses
  it. "fighters with inferior turn performance (larger radius) or greater speed generally should not attempt this
  maneuver"; "the lead turn should be delayed until the opponent's reaction time does not allow him to counter before
  the pass has occurred."
- 2C plan: "the pilot of the better-turning fighter should try to gain some flight-path separation with the bogey, in
  the plane of the intended turn, before the pass ... by making a small turn away from the bogey" (pp. 80-81 /
  PDF 99-100). 1C plan: "minimize in-plane flight-path separation at the pass" (p. 78 / PDF 97).
- Dissimilar 2C mid-game (Fig 4-3, pp. 145-146 / PDF 164-165): turn hard to point at the centre of the bogey's turn,
  accelerate unloaded toward corner speed, then a minimum-radius turn at corner at the early-turn point, ideally
  passing "almost directly over, under, or slightly behind the bogey".

Game: the geometry of Fig 2-9 gives a usable rule. A turn of radius R from a reciprocal track offset d reaches the
other's track after a heading change θ with R(1 - cos θ) = d, so θ = arccos(1 - d/R): d = R gives 90 deg, d = 2R gives
180 deg, Shaw's two cases. (This formula is my geometry, consistent with his figure; he gives only the proportionality.)
With R about R_st, the lead time θ/ω is t90 when d = R_st and 2 t90 when d = 2 R_st. Timing must still put the
attacker behind the defender (nose-tail separation > 0), which needs the defender's own travel; that is a measurement.

## 7. Overshoots and reversals

- "Technically, any time the attacker crosses behind the target an overshoot has occurred; but this is usually not
  dangerous unless the target is moving at a slower speed or has a tighter turn radius" (p. 67 / PDF 86).
- "the greatest danger exists for the attacker when overshoots result at close range and low TCAs against a slower
  bogey with a tighter turn radius" (p. 76 / PDF 95). A long-range overshoot at about 90 deg TCA between equal
  fighters is of "little danger"; a defender reversing then "places the attacker on the inside of his turn in lead
  pursuit and subjects himself to a guns pass" (p. 75 / PDF 94).
- Defender's reversal: a high-deflection gun attempt "will usually lead to an overshoot with high closure and little
  nose-tail separation—the ideal set-up for the defender to perform a nose-to-nose reversal", "level or slightly
  nose-low in an attempt to force a flat scissors", good because the defender is "likely to be at slower speed and
  inside the attacker's turn radius" (p. 134 / PDF 153). Against lag at large advantage, reverse "as the attacker
  crosses six" and turn nose-to-nose "at or below corner speed" (p. 134 / PDF 153).
- Reversals after angular-defensive turns: "If a reversal is executed it should be level or slightly nose-low. A
  nose-high reversal often results in a zooming contest or a rolling scissors" (p. 132 / PDF 151).
- Attacker's prevention: start the high yo-yo or lag roll early (section 5); "If the attacker had recognized the
  situation earlier he could have ... pulled up vertically to minimize his overshoot" (p. 93 / PDF 112).

Game: the design's release (crossing my extended six inside a range with a high LOS rate) matches "close range, high
closure, little nose-tail separation". Shaw adds two conditions: low TCA, and that the reverser is slower (or tighter)
than the overshooter. "Level or slightly nose-low" = no climb during the reversal; a dive is allowed.

## 8. Flat scissors

- "a series of nose-to-nose turns and overshoots performed by two fighters essentially in the same maneuver plane"
  (p. 82 / PDF 101, Fig 2-16). Phases: nose-to-nose turn, reversal, lead turn (p. 83 / PDF 102).
- Nose-to-nose phase: the slower or tighter-turning fighter wins; as the fighters are drawn together, "speed usually
  remains the determining factor"; "To gain advantage during this phase, a fighter should decelerate as quickly as
  possible" (p. 83 / PDF 102).
- Reversal: as fast as possible (each fraction of a second is straight flight); timing sets TCA and nose-tail separation
  at the overshoot. Early reversal: lower TCA and AOT later, closer range; "the earliest possible reversals lead to the
  earliest advantage for a gun-equipped fighter", up to the point where nose-tail separation reaches zero (passing ahead
  of a gun-armed opponent is not recommended) (pp. 83-84 / PDF 102-103, Fig 2-17).
- Ends when the winner gets "in phase" and no longer overshoots (p. 85 / PDF 104).
- Who should avoid it: "Less maneuverable, high-speed fighters should avoid this situation like the plague"; avoid by
  keeping speed for vertical manoeuvring and refusing a co-planar nose-to-nose turn with a slower, better-turning
  opponent (p. 85 / PDF 104).
- Exit (Fig 2-18, pp. 85-86 / PDF 104-105): do not reverse at the overshoot; continue nose-to-tail, extend, and come
  back to meet the attacker near 180 deg TCA.

Game: inside each other's radius the committed sense plus overshoot release produce the scissors, as the design says.
Shaw adds the speed rule: in a scissors v* should go down (to the floor), not stay at v_st; in the game the dive at
low speed (204 m at 107 m/s) is the only way to get a tighter radius than the opponent at v_st, so the scissors is a
place where the store is spent on radius. And the rule for the faster or energy-poor ship: do not enter (stay 2C).

## 9. Gun geometry

- Envelope: effective max range "about 3,000 ft" for modern guns; minimum range "500 ft might be a reasonable minimum"
  (p. 15 / PDF 34). Min range "much greater in the target's forward hemisphere because of higher closure"; tracking only
  in the rear and beam regions, snapshots elsewhere (Fig 1-4, p. 24 / PDF 43). Head-on: "very narrow (if it exists at
  all)" (p. 29 / PDF 48).
- Tracking needs the pipper steady past the sight's settling time; optimum tracking region "about 30° to 60° off the tail
  with a LCOS, 0° to 30° for fixed sights" (p. 16 / PDF 35); "Closer to the stern position may be better with fixed
  sights because of reduced deflection" (p. 25 / PDF 44).
- Snapshot: the pipper passes through the aim point. Low-G snapshot: put the pipper well ahead on his path and fly
  straight while he flies through it (short firing time). High-G snapshot: "almost a tracking shot", more lead first
  (pp. 20-21 / PDF 39-40). Guns-only fights realistically aim for the snapshot envelope: "it is usually more practical to
  maneuver for the snapshot envelope" (p. 99 / PDF 118).
- Excess lead: a common error is "not allowing sufficient excess lead"; "It is much more effective to allow ... the
  target to fly up to the pipper as firing range is approached"; shooter G "should be stable or constantly increasing"
  (p. 19 / PDF 38). Rear-quarter attack on a turning target: AOT grows, stabilises, falls near min range; peak required
  G comes soon after AOT starts falling.
- Bursts of about one second until the solution is sure (pp. 17, 22-23 / PDF 36, 41-42).
- At every forward-quarter pass "consideration should be given to attempting a short gun burst" (p. 112 / PDF 131);
  passing with a good bite "may offer the attacker a forward-quarter gun snapshot opportunity, which he should take"
  (p. 103 / PDF 122).

Game: the fire rule (aligned at the gun lead point, in range) is a computed-lead snapshot/tracking rule; fine. Shaw
suggests the pursuit point should lead the gun lead point slightly before range so the target flies in (no number).
Shaw's min/max ratio is 1/6; the game's equivalent would be a min-range or collision concern, not modelled.

## 10. Defensive manoeuvres

Guns defence, in order (p. 31 / PDF 50): deny any shot (extension), deny a good shot (break), make a poor shot hard
(out-of-plane, jinks).
- Extension: if faster and seen early, turn to put him dead astern and run straight; turning while running lets him
  cut across (p. 23 / PDF 42).
- Break: "a maximum-performance 'break' turn toward the attacker to rotate him into the forward hemisphere" (p. 25 /
  PDF 44). It raises his lead requirement and tracking G. But once he holds range and lead, a continued in-plane turn
  gives him a steady target: switch to an out-of-plane manoeuvre (p. 26 / PDF 45, Fig 1-5).
- Delaying the break to the last instant: "Although this method can be effective, it cannot be recommended" (p. 28 /
  PDF 47).
- Jinks: turn hard "long enough to change his flight path significantly, but not long enough to allow the shooter to
  correct his aim and track"; each jink "in a plane at least 90° from the direction of the previous jink"; not
  predictable; continue until he reaches min range or you open beyond max range (p. 28 / PDF 47). If he has closure and
  cannot be outrun, increase his closure (power back, drag) to hurry him through (pp. 28-29 / PDF 47-48). Against a
  head-on shooter "one or two jinks are usually sufficient" (p. 29 / PDF 48). In the energy fight, "One quick, hard,
  out-of-plane pull, followed by a 180° roll and another pull" spoils a forward-hemisphere snapshot (p. 110 / PDF 129).
- How hard (Ch 3, pp. 131-134 / PDF 150-153):
  - "The first goal should be to regain neutrality without being shot"; going straight from defence to offence leaves
    the pilot "energy defensive" (pp. 131-132 / PDF 150-151).
  - "A break turn is wasteful when a hard turn will suffice" (p. 132 / PDF 151).
  - Bogey ahead with an angular advantage but outside range or without lead: an in-plane turn "hard enough to pull the
    bogey to the nose by the time maximum effective firing range is reached" (p. 132 / PDF 151); if too much advantage,
    a turn "just hard enough to stabilize the angular situation".
  - Against lag: pull "just hard enough to prevent the bogey from reaching the blind spot at six o'clock" (p. 132 /
    PDF 151).
  - Against an out-of-plane (yo-yo) move: stop the defensive turn, unloaded nose-low extension, then a hard turn back
    "in the closest direction" (p. 133 / PDF 152).
  - Large advantage (more than 90 deg): weapons defence first; "the initial break turn is usually continued until the
    bogey's advantage is reduced to about 90°, then it can be relaxed to stabilize the situation while reducing energy
    bleed" (p. 134 / PDF 153).
  - "the angular disparity should be corrected first, since this is normally the more dangerous of the two" (p. 135 /
    PDF 154).
- Energy disadvantage (pp. 135-137 / PDF 154-156, Fig 3-13): level sustained nose-to-nose turns at the pass,
  reversing at each pass and taking what the bogey allows; when the gain per pass stops growing the two are co-speed.
  If the bogey keeps nose-to-tail, accelerate (unloaded) and only turn enough for a head-on next pass. Against a higher,
  possibly slower bogey, slow to minimum-sustained-radius speed (about 1.5 Vs) so that "a similar bogey cannot
  outperform him in nose-to-nose turns regardless of the opponent's speed" (p. 137 / PDF 156). Do not go vertical with
  a higher-energy attacker (p. 134 / PDF 153).
- Spending energy in defence: the break is max performance; "As long as the defender has awareness, speed, and altitude
  for maneuvering, he can make the task of an attacking gunfighter almost impossible" (p. 31 / PDF 50). The Lufbery
  (diving spiral near corner speed) holds a similar attacker off in 2C "as long as altitude allows" (p. 109 / PDF 128).

Game: the design's defensive (full stick into him, v* = v_c within the defend range, dive with store) is Shaw's
break plus the Lufbery. Shaw adds: relax from the break once he is back to about 90 deg (in the design this is the 3/9
gate handing over to turning); a hard rather than a break turn when he is outside range or has no lead; jinks as short
reversals when he is about to fire, not as a held turn; and no climbing (banking) into a higher-energy attacker.

## 11. Other points a 1v1 controller needs

- Assessing angles: "The magnitude of this advantage is simply the difference in the turns required by each fighter to
  point at the other", easiest to judge when one of them points at the other (p. 131 / PDF 150). In the design's terms:
  advantage = |his AON to me| - |my AON to him| = (π - α) - |λ|, a scalar the Situation can carry.
- Assessing energy: the higher one has more unless known slower; below corner the faster turns faster; "Nose-to-nose
  turns provide perhaps the best measure of relative speed" (p. 131 / PDF 150). The scripted agent reads speeds and
  (last-seen) store directly, so this is easier for us.
- Zoom contests: "The first fighter to begin a zoom normally will peak first" and is then slow and vulnerable (p. 158 /
  PDF 177); "A zooming contest is not wise ... if a pilot is unsure of his relative energy state" (p. 131 / PDF 150).
  In the game: banking (slowing) close to an opponent with more energy hands him the rate fight.
- Disengagement: Shaw recommends bugging out when a held advantage is lost (p. 130 / PDF 149) and extensions after
  defeating an attack. The design has no extend or retreat (decision 5.7); noted, not proposed.
- Tactical plan: "Victory depends on having a tactical plan prior to the engagement and executing that plan
  aggressively" (p. 138 / PDF 157), yet recognising and making tactical transitions mid-fight are "signs of great experience" (p. 113 / PDF 132).

## Against the design (scripted-bfm-design.md, B2-B6)

### Tactics and rules

| design item | Shaw | verdict and suggestion |
|---|---|---|
| Quadrants on the 3/9 line (B3) | 90 deg is his working limit everywhere: break until the bogey's advantage is "about 90°" (p. 134 / PDF 153); the energy fighter concedes at most about 90 deg (pp. 110, 112 / PDF 129, 131); against lag reduce him to "about 90° or less as he crosses six" (p. 134 / PDF 153) | Confirms π/2 as the dividing angle. |
| Disengaged: climb to regain energy | energy is "a very precious and hard-to-get commodity", grab it before the pass (p. 101 / PDF 120) | Confirms. |
| Merge: lead turn, set the flow | Fig 2-8 to 2-10, 3-1; 2C planner opens lateral separation by a small turn away; 1C or less manoeuvrable planner minimises separation, passes close (pp. 78, 80-81, 76-77 / PDF 97, 99-100, 95-96) | Confirms; add the pre-merge offset by plan (turn slightly away for 2C, toward for 1C). |
| Merge τ: lead time for 2C, 0 for 1C | lead angle grows with separation/radius (p. 76 / PDF 95); no-separation lead turn only by the better turner, delayed until he cannot react (p. 77 / PDF 96) | Replace the fixed lead time by θ = arccos(1 - d/R) and time θ/ω (section 6); no lead turn when d is small unless I am slower. |
| Offensive: pursuit by closure, yo-yos | high yo-yo at AOT 30-60 deg, co-speed, stop near zero closure; low yo-yo from long-range lag; small steps (pp. 71-74 / PDF 90-93) | Confirms the closure signal; add an AOT condition and keep steps small. |
| Turning: 1C or 2C by the flow | section 1 | Confirms. |
| Defensive: full stick into him, v_c, spend | break toward him (p. 25 / PDF 44); relax at about 90 deg (p. 134 / PDF 153); hard not break when outside range or no lead (p. 132 / PDF 151); Lufbery (p. 109 / PDF 128) | Confirms; add the relax and the hard-turn grade. |
| Commit: Δe > threshold gives 2C, else 1C | energy side prefers 2C (pp. 108, 142 / PDF 127, 161); energy-deficit side 1C (p. 135 / PDF 154); in the game the store buys rate (2C) far more than radius (1C), section 0 | Confirms, with the game's reason. |
| Commit: 2C his sense, 1C opposite | defined by Fig 2-11; "checks the bogey's direction of turn" at the pass (pp. 106, 135 / PDF 125, 154) | Confirms. Add: if my advantage at the pass is already large, keep the current turn nose-to-tail whatever the plan (pp. 102, 146 / PDF 121, 165). |
| Hold while he is aft of 3/9 | "once a trend is established ... seldom reversed" (p. 130 / PDF 149); reversals only on a reason | Confirms. |
| Release on overshoot (range, LOS rate) | danger for the overshooter is close range, low TCA, slower or tighter defender (pp. 67, 76 / PDF 86, 95); a 90 deg TCA overshoot at long range between equals is not a reason to reverse (p. 75 / PDF 94) | Confirms; add TCA below a bound and v_me ≤ v_him to the release. Reverse level (no climb), dive allowed. |
| Yo-yos as one signed closure signal | Shaw alternates them and says "two small yo-yos ... are safer than one big one" (p. 73 / PDF 92) | Confirms the signal; keep the yo-yo time short. |
| Shot near: spend the store only for a purpose | nose-low turn at corner to trade altitude for angles in the first 1C turn (p. 101 / PDF 120), bounded (p. 102 / PDF 121); "maneuver only as hard as necessary" (p. 143 / PDF 162) | Confirms; Shaw also spends for the first post-merge turn and for the lead turn (corner at the early-turn point, p. 146 / PDF 165). |
| v* floor | min sustained radius about 1.4-1.5 Vs (jets), slower for props (pp. 137, 409 / PDF 156, 428); Shaw's real floor is "vertical-maneuvering speed" (no 2D analogue) | Keep a floor in stall speeds. |
| Lead turn, scissors, Lufbery not separate tactics | scissors: decelerate in the nose-to-nose phase, reverse early (pp. 83-84 / PDF 102-103) | Agree they need not be modes; add "v* to the floor (with dive) when inside one radius in 1C". |
| No stalemate breaker (decision 5.7) | Shaw's breaker is the angles-to-energy switch after one or two even turns (p. 112 / PDF 131), not an extension | Suggest it as a rule (below); it stays inside the fight. |

Rules Shaw adds that the design lacks (proposals; each is one condition and one command change):

1. Angular advantage scalar A = (π - α) - |λ| in the Situation (p. 131 / PDF 150).
2. Energy-plan concession cap: in a 2C turn with the energy plan, ease the stick and bank (Shaw's two-thirds G and climb,
   p. 110 / PDF 129) while his advantage is below a cap; at the cap (Shaw: 60-90 deg, pp. 107, 110 / PDF 126, 129) stop
   banking and go full stick, cashing the store. Shaw: about 90 deg.
3. Angles-to-energy switch: if the 1C fight shows no gain over one or two passes (Shaw: "matches angle-for-angle through
   one or two turns", p. 112 / PDF 131), switch the plan to 2C at the next pass. Converse: if he will not take the
   offered angles in 2C within one turn, switch to angles via a low yo-yo (p. 113 / PDF 132). Needs a pass counter or
   an angle-gain memory (one more memory field).
4. Keep 2C after a big bite: at the pass, if A ≥ a bound (Shaw: 60-70 deg, dissimilar case, p. 146 / PDF 165), keep
   the current sense.
5. Defensive grade: full stick (break) while A < -π/2 and he is in or near range; a hard (sustained-level) turn when he is
   outside range or has no lead (p. 132 / PDF 151); relax at about -π/2 (p. 134 / PDF 153).
6. Jink: when he is in range with lead (his aim gate on me), reverse the stick for a short time, then reverse again; not
   periodic (p. 28 / PDF 47). The design has no jink; the committed sense blocks it. Suggest a short jink override in
   the defensive tactic, duration a fraction of t90 (Shaw gives no time).
7. Scissors speed: in 1C inside one turn radius, v* to the floor with dive allowed (pp. 79, 83 / PDF 98, 102).
8. Do not bank (climb) close to a higher-energy opponent (zoom contests, pp. 131, 158 / PDF 150, 177); the disengaged
   regain climb is fine.

### Table B6, row by row

| parameter | design | Shaw | suggestion |
|---|---|---|---|
| engage range | 3.0 gun ranges | no number for engagement; guns max range about 3,000 ft (p. 15 / PDF 34); extension aims to stay outside it (p. 23 / PDF 42) | keep 3.0; Shaw gives no number |
| 3/9 line | π/2 | 90 deg limits throughout (pp. 107, 110, 112, 134 / PDF 126, 129, 131, 153) | keep π/2 (confirmed) |
| quadrant gate scale | 0.15 rad | none | keep; no source |
| turning floor | 0.25 ω_st | none | keep; no source |
| plan threshold | 0.0 store fractions | no number; similar aircraft, angles "probably preferable" unless trained in energy (p. 112 / PDF 131); energy is hard to judge (p. 131 / PDF 150) | keep 0 or a small positive value (prefer 1C when even); Shaw gives no number. Add the switch rule (3) rather than tune this |
| lead-turn time | 1.0 t90 | advantage ∝ separation/radius; R = d/2 gives 180 deg, R = d gives 90 deg (p. 76 / PDF 95) | replace by θ = arccos(1 - min(d/R, 2)) and time θ/ω_st (equals 1.0 t90 at d = R_st, so the default is right for that offset); 0 when d is small and I am not slower (p. 77 / PDF 96) |
| overshoot range | 0.67 R_st | no number; danger at "close range" with low TCA against a slower, tighter bogey (p. 76 / PDF 95) | keep 0.67 R_st; add TCA and speed conditions; Shaw gives no number |
| overshoot rate | 1.0 ω_st | no number ("high closure and little nose-tail separation", p. 134 / PDF 153) | keep; add TCA < a bound (Shaw: 90 deg TCA at long range is not dangerous, p. 75 / PDF 94; no threshold given) |
| nose-tail distance | 0.67 R_st | no nose-tail number for guns; gun min about 500 ft against max about 3,000 ft (p. 15 / PDF 34), i.e. about 1/6 of max range | keep; 0.67 R_st = 0.25 gun range, inside Shaw's min-to-max band |
| closing time | 1.0 t90 | none; "high closure at long range, reduced near firing range", firing position "just as desired firing range is reached" (p. 18 / PDF 37) | keep; no number |
| yo-yo scale | 0.1 v_c | none; high yo-yo stops "as the closure slows to nearly zero" (p. 71 / PDF 90) | keep; consider gating the high side on AOT 30-60 deg (p. 71 / PDF 90) |
| yo-yo time | 0.5 t90 | none; "two small yo-yos ... are safer than one big one" (p. 73 / PDF 92) | keep small (0.5 t90 or less); no number |
| merge speed | 1.0 v_c | angles fighter engages at corner, zooms off the excess (pp. 101-102 / PDF 120-121); energy fighter not below max(v_st, vertical speed) (p. 106 / PDF 125); low-wing-loaded fighter: best sustained-rate speed (p. 143 / PDF 162) | keep 1.0 v_c with a full store for the angles/1C side; Shaw would allow the 2C side to merge slower (down to v_st) if that keeps the store; consider merge speed by plan |
| fight speed | 1.0 v_st | energy fighter in 2C at max sustained-rate speed (p. 110 / PDF 129); turn fighter at best sustained-rate speed (p. 143 / PDF 162) | keep 1.0 v_st for 2C (confirmed); for 1C inside one radius go lower (rule 7) |
| speed floor | 1.3 v_stall | min sustained radius 1.4-1.5 Vs for jets, "somewhat slower for prop fighters" (pp. 137, 409 / PDF 156, 428) | keep 1.3 (consistent with "slower for props"); the game's dive at 107 m/s = 1.15 v_stall gives the smallest radius, so the floor is a guard, not an optimum |
| spend share | 1.0 of store x A_store | spend for a purpose, bounded: angles fighter's altitude loss limited to a quarter radius or half guns range (p. 102 / PDF 121); "maneuver only as hard as necessary" (p. 143 / PDF 162) | Shaw supports a share below 1 but gives no number in energy terms; keep 1.0 for the shot, add a smaller share for the first post-merge 1C turn if built |
| defend range | 1.0 gun ranges | break before he reaches the tracking area; a late break "cannot be recommended" (p. 28 / PDF 47); hard turn to have him on the nose by max range (p. 132 / PDF 151) | keep 1.0 or slightly above for the break; use a sustained-level turn outside it (rule 5); Shaw gives no number |
| speed band, power scale | 0.05, 0.03 ln | none | keep; no source |

Not in B6 but numbers Shaw gives that may be useful: first-turn gain 20-30 deg for a good angles fighter (p. 102 /
PDF 121); per-pass gain 20-30 deg means a big speed gap, 10 deg or less a small one (p. 107 / PDF 126); dissimilar
above about 10% (p. 139 / PDF 158); fixed-sight tracking 0-30 deg off the tail (p. 16 / PDF 35); nose-to-nose reaches
a given angle in about half the time of nose-to-tail (p. 109 / PDF 128).

### Where Shaw does not carry over

- Everything vertical: zoom height as position, overhead attacks (altitude advantage about one minimum radius, p. 150 /
  PDF 169), rolling scissors, defensive spiral, barrel rolls, lag rolls. Only their energy half maps to the store.
- Sight and blind spots (a large share of his 1C/2C choice and of the lag discussion): the scripted agent has fog with
  last sightings, not a canopy.
- Altitude-differential limits ("a quarter turn radius") are geometric in Shaw; the store has no position, so they have
  no direct translation. An energy-difference bound would be a new, unsourced number.
