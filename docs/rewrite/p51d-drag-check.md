# P-51D: checking C_D0 and propeller efficiency

Analysis, 2026-11-24 (branch `core`). Nothing in the code changed. The game's P-51D row uses
C_D0 0.0150 and propeller efficiency 0.82. Both were fitted together to two points from flight
test 44-15342: 375 mph at sea level on 1,630 hp, and 3,600 ft/min. This note re-derives the
pair with the right engine power at each point and with an efficiency that varies with speed.
It then compares the result with NACA's propeller-off dives and with Loftin, and says what the
game should use.

Speeds are real (1x) and true airspeed. "hp" means brake (shaft) horsepower. "Thrust hp" means
thrust x speed. Every chart reading here is my own, and the uncertainty is given where it
matters.

## Short answer

- **C_D0 0.0155**, range 0.0148-0.0163 (Oswald e 0.8).
- **Propeller efficiency at top speed 0.85**, range 0.82-0.885.
- **In the climb (165 mph): 0.78**, range 0.75-0.81.
- The robust quantity is the ratio. Top speed fixes C_D0 / efficiency at 0.0183 (e 0.8). Each
  estimate below differs only in how it splits that ratio.
- The game's 0.0150 / 0.821 is right only if the climb really used the report's 1,780 hp.
  The fit puts 1,630 hp into the climb, about 9% too little. A constant efficiency overstates
  the propeller's climb efficiency by about the same 9%, and the two errors cancel. The
  middle estimate, a climb power of about 1,730 hp, gives 0.0155 / 0.85.
- In the game, the change would be: efficiency 0.85 taken as a prior, not fitted; C_D0
  re-solved (0.0156); and the thrust softness (`thrust_softmin_exponent`) solved, 8 -> 3.7.
  Top speed and climb stay on target.
  - The best-climb speed moves from 144 to 160 mph (the test flew 165).
  - The low-speed acceleration falls from 3.69 to 3.39 m/s^2.
  - The best sustained turn barely moves (18.99 -> 19.05 deg/s).
  - Shaft power and static thrust do not change.
- This is a proposal, not a decision.

## Sources used

From `~/avesnova/boost-and-broadside/data/aircraft-references/p51d/` unless noted:
- **44-15342**: AAF memo report TSCEP5E-1908, 15 Jun 1945 (A).
  - Fig. 2, standard weight vs altitude (p.12).
  - Fig. 3, level flight data.
  - Fig. 4, power required in level flight, clean (p.14).
  - Fig. 5, WER climb (p.15).
  - Fig. 8, drag curves (p.18).
  - Appendix B (pp.21-22).
  - The report's web page tables (http://www.wwiiaircraftperformance.org/mustang/p51d-15342.html,
    read 2026-11-24): the level flight tables (Sec. IV.B.2-4) and the WER climb table.
- **The V-1650-7 low-blower engine chart**: Packard "Power Plant Curve #27", 12-3-43 (B).
- **NACA ACR 4K02**: Nissen, Gadeberg and Hamilton, 1945 (A).
  - Text: PDF pp.5, 7, 9-10, 14.
  - Figs. 12 and 14 (PDF pp.22-23).
- **Loftin, NASA SP-468** (`../comparative/`) (D): p.130 and Appendix C, p.509.
- **NACA MR L5G09a**, via `aircraft-references.md`: the P-47D's measured propeller efficiency.

## 1. The engine power at the two test points

From the report's tables (all at 67 in Hg, 3,000 rpm):
- **Level flight:**
  - sea level: 375 mph on 1,630 hp;
  - 5,000 ft: 396 mph on 1,667 hp;
  - 10,000 ft: 417 mph on 1,700 hp.
- **WER climb:**
  - sea level: 3,600 ft/min at 165 mph IAS, 1,780 hp;
  - 4,800 ft: 3,600 ft/min at 165 mph IAS, 1,730 hp;
  - 5,000 ft: 3,575 ft/min on 1,720 hp (66.5 in).
  - Full throttle in the climb is at 4,800 ft. In level flight, with ram, it is at 10,000 ft.

The climb's sea-level 1,780 hp is doubtful:
- In level flight, power at fixed boost rises with altitude, from 1,630 to 1,700 hp. That is the
  normal trend: lower exhaust back-pressure. The climb table runs the other way, from 1,780 hp
  at sea level to 1,730 hp at 4,800 ft.
- On the engine chart, the full-throttle line passes 67 in at about 1,720 hp (my reading). The
  climb's 1,730 hp at full throttle agrees with that. The sea-level 1,780 hp is above the
  engine's own full-throttle figure.
- The chart's correction note says: subtract 1% for each 6 C of intake air above standard. Ram
  heating is about 14 K at 375 mph and about 3 K at 165 mph (V^2 / 2 c_p). That gives the climb
  about 2% more power than level flight, about 1,660 hp, not 9%.
- So the climb power at sea level is somewhere in 1,660-1,780 hp. The value with the most
  support is about 1,730 hp: the climb table's own figure at 4,800 ft, where the rate of climb
  is still 3,600 ft/min. The pair of level and climb points at 4,800-5,000 ft (section 3) uses
  only the report's numbers at one altitude, and it gives the same answer.

The climb speed is 165 mph IAS, corrected for instrument error. At sea level that is 165 mph
true (73.8 m/s), with a 14 deg climb angle.

## 2. The propeller model

Momentum theory with a constant profile efficiency (the actuator disc):
- thrust T = 2 rho A v (V + v), where v is the induced velocity at the disc;
- shaft power P = T (V + v) / eta_p.

So the efficiency is eta = eta_p x V / (V + v). At V = 0 it gives the static thrust, with a
figure of merit equal to eta_p. The model is standard (Glauert, McCormick). Its one free
parameter, eta_p, carries the blade profile and the compressibility losses.

The P-51D's propeller works far from static at both test points:
- disc 3.40 m;
- propeller speed 1,437 rpm (gear 0.479 on the engine chart, partly legible);
- climb (165 mph): advance ratio J 0.91, helical tip Mach 0.78;
- top speed (375 mph): J 2.06, helical tip Mach 0.90;
- power coefficient C_P 0.159 at 1,630 hp.

The model gives the climb about 8.5% less efficiency than top speed at the same shaft power
(0.751 / 0.820). The real ratio could differ a little: tip Mach 0.90 costs some efficiency at
top speed, and blade angle effects cost some in the climb. Neither is in the folder's sources.

The climb is solved properly here: thrust = drag at load factor cos(gamma) + W sin(gamma), at
the tested 165 mph. The game's own fit uses Ps at 1 g at its own best speed. With 1,630 hp and
a constant efficiency the two methods give 0.01502 / 0.8205 and 0.01504 / 0.821, so the
method does not matter.

## 3. Re-fits at sea level (9,760 lb, S 21.67 m^2, AR 5.83, e 0.8)

**Constant efficiency** (the game's model):
- climb on 1,630 hp (the game's fit): C_D0 0.01504, eta 0.821, best climb at 142 mph;
- climb on 1,780 hp: C_D0 0.0136, eta 0.746, best climb at 134 mph.
- The 1,780 hp case puts the efficiency at top speed below the P-47D's measured 0.78-0.80 at
  J 1.2. A constant efficiency with the right power is the worst of the three.

**Momentum model**, by the climb's shaft power (C_D0; eta at top speed; eta in the climb):
- 1,660 hp: 0.0163; 0.885; 0.811
- 1,690 hp: 0.0159; 0.868; 0.795
- 1,730 hp: 0.0155; 0.846; 0.775
- 1,780 hp: 0.0150; 0.820; 0.751

In every case:
- the model's best-climb speed is 176-178 mph, and its rate at 165 mph is within 0.5% of the
  best;
- the constant-efficiency model's best-climb speed is 130-142 mph.
- The test was flown at 165 mph. Test schedules are normally flown near the best speed, so the
  speed-dependent model fits that point better.

**The pair at 4,800-5,000 ft** (396 mph on 1,667 hp; 3,600 ft/min at 177 mph true on 1,730 hp;
weights from the climb's fuel used):
- momentum model: C_D0 0.0156, eta 0.848 at top speed, 0.781 in the climb;
- constant efficiency: 0.0142, 0.775.

**The ratio C_D0 / eta is pinned by the top speed:** 0.0183-0.0184 in every row above.

**The climb table's higher rows are not used.** Paired with the 10,000 ft level point, they
need:
- 10,000-15,000 ft: an efficiency at top speed of 0.86-0.87;
- 19,000-30,000 ft, high blower: 0.94-0.98.

Those are not physical. Either the climb table's power accounting or its rates at altitude
disagree with the level tables. I could not find out which.

## 4. More test points: the power-required chart (Fig. 4)

Fig. 4 plots brake horsepower at standard weight against true speed, at 1,500, 5,000, 10,000,
15,000, 20,000, 25,000, 30,000 and 35,000 ft. The standard weights run from 9,700 down to
9,530 lb, as labelled on the curves.

How it was read:
- I digitised the faired curves where they cross fixed powers from 550 to 1,700 hp. That gave
  127 readings.
- The axes were calibrated from the printed tick labels, to about +-1 mph and +-5 hp.
- The two clean curves of Fig. 8 ("drag curves", 5,000 and 25,000 ft density altitude) lie on
  the Fig. 4 curves. Fig. 8 adds only the store increments, and it says density altitude. I
  used ISA density at the labelled altitudes.

What it gives:
- **C_D0 / eta (e 0.8), at powers of 1,000 hp and above, by altitude:**
  - 1,500 ft 0.0179; 5,000 ft 0.0177; 10,000 ft 0.0181;
  - 15,000 ft 0.0184; 20,000 ft 0.0188; 25,000 ft 0.0189; 30,000 ft 0.0192.
  - The rise with altitude fits two things: more tip-Mach loss (the helical tip Mach reaches
    about 1.0 near 25,000 ft at 440 mph) and a lower Reynolds number. Neither matters at the
    game's sea level.
- **The level tables give a slightly higher ratio:**
  - sea level: 0.0185 at WER, 0.0187 at military, 0.0184 at normal power;
  - 10,000 ft: 0.0185-0.0191.
  - It is the same at 1,075 and 1,630 hp. So the efficiency at high speed hardly depends on
    power, as momentum theory says: ideal efficiency is about 0.99 throughout.
- **Fig. 4 and Fig. 3 disagree.** At the WER points, Fig. 4's curves sit 5-7 mph faster than
  Fig. 3 at the same power:
  - 1,500 ft: 386 mph at 1,630 hp, against about 381 interpolated from the table;
  - 5,000 ft: 403 mph at 1,667 hp, against 396;
  - 10,000 ft: 424 mph at 1,700 hp, against 417.
  - That is 3-4% less power at a given speed. I found no reason for it in the report. One
    possibility: Fig. 3 was corrected to the cooling-flap positions for a standard day (the
    report's web page). Taken at face value, Fig. 4 moves C_D0 about 3% lower. That sets the
    0.0148 end of the range.
- **Fig. 4 does not give the induced factor.** A parabolic polar fits all 127 readings badly:
  the rms error is 3.2% in power, and e x eta comes out at 1.07, which is not physical. The
  cause is the low-power ends of the curves. For example, 1,500 ft at 228 mph needs only about
  418 hp, which would need an efficiency of about 0.96 with e 0.8.
  - Those points are cruise settings, and their power is read from engine charts at part
    throttle and low rpm. I treat them as the least reliable.
  - The points at 1,000 hp and above, up to 10,000 ft, give e x eta = 0.73 (relative standard
    deviation about 10%). That is e about 0.86 at eta 0.85: weak, but consistent with the
    NACA value below.
- **So the chart that `aircraft-references.md` calls "the only chart for fitting drag
  directly" gives C_D0 / eta, not C_D0.** Splitting the ratio still needs the climb, or a
  propeller prior.

## 5. Independent values

**NACA ACR 4K02, the P-51B with the propeller off** (towed up, dived to M 0.755):
- Fig. 12, flat part of each dive (M 0.37-0.65, lift coefficient about 0.05-0.2):
  - C_D about 0.0213 on flight 109 (least dust);
  - 0.0218 on flight 110;
  - 0.0233 on flight 108 (most dust).
- Fig. 14 (flight 109; my readings of the faired curve):
  - 0.0213 at C_L 0.05, 0.0281 at 0.35, 0.0384 at 0.55, 0.0546 at 0.75, 0.0615 at 0.82;
  - a parabola fitted to these gives C_D0 0.0197-0.0206 and k 0.060-0.062;
  - that is **e 0.88-0.91** at the P-51B's AR of 5.815.
  - Someone has pencilled "e = .9" on the scan. That annotation is not NACA's; my fit agrees
    with it.
- Test conditions (text):
  - PDF p.5: the carburettor scoop was sealed 3 ft from the lip, the bomb racks were removed,
    and the surface was "sanded with fine sandpaper, shellacked, and waxed";
  - PDF p.7: the radiator-scoop flap was locked flush;
  - PDF pp.9-10: dust from the lake bed on take-off raised the minimum drag;
  - PDF p.14: the flight values are slightly above the tunnel's.

**Why the propeller-off 0.0205 sits above the powered 0.0155.** The gap is about 0.005. At
375 mph at sea level, 0.001 of C_D is 84 lbf, or 84 thrust hp. So the gap is about 420 lbf.
- **Meredith effect.** In the dives the engine was not running: the radiator duct had cold
  flow, so it was pure internal drag. In powered flight, the coolant's heat gives the duct
  thrust back. Loftin (p.130) says the system "was designed with the objective of obtaining a
  net thrust from the cooling air". No source in the folder measures that thrust.
- **Exhaust thrust.** The Merlin's ejector stacks give thrust in powered flight and none in
  the dives. The fit puts it into the effective C_D0. Not quantified in the folder.
- **Reynolds number.** The dives ran at 28,000 ft and M 0.4-0.6. That is a Reynolds number of
  0.35-0.52 of the sea-level top speed's (my arithmetic; ISA, mean chord 1.92 m). The
  flat-plate rule (Schlichting) puts turbulent skin friction 11-18% higher in the dives. With
  friction most of C_D0, that is up to about 0.002. Fig. 12 stays flat over a 1.5x Reynolds
  range within each dive, so the real effect is probably smaller.
- **Surface and fittings, in both directions.**
  - Dust adds up to 0.002 (flight 108 against flight 109).
  - The NACA airplane was smoother than a service airframe (sanded and waxed) and had no
    racks.
  - 44-15342 was flown "clean" with two wing bomb racks.
- **Propeller and slipstream.** The fit puts the slipstream's extra drag and the blade-root
  losses into the propeller efficiency. That is why the efficiency is "propulsive", not the
  propeller's alone.
- **Summary.** Reynolds number and the dust account for at most about half the gap. The rest,
  roughly 0.0025-0.004 (200-340 lbf at 375 mph), must be net thrust from the radiator and the
  exhaust. That is plausible, but it is not checked against a source.

**Loftin, 0.0163** (SP-468 p.130; method Appendix C, p.509):
- He fitted 437 mph at 25,000 ft on 1,490 hp, at 10,100 lb.
- His efficiency was "between 0.70 and 0.85, depending on the aircraft"; the P-51D's own value
  is not stated.
- Back-calculated (my arithmetic, e 0.70-0.80), his 0.0163 needs an efficiency of 0.76-0.77
  on 1,490 hp, or 0.80-0.815 on the 1,410 hp the test measured at 25,000 ft.
- His ratio C_D0 / eta is about 0.020-0.021. That sits above the test's 0.0189 at 25,000 ft:
  his weight and power are not the test's.
- His 0.0163 is a fit like ours, with a lower efficiency. It does not show that the airframe is
  dirtier.

**P-47D, the only measured propeller efficiency in the survey:** 0.78-0.80 at J 1.19-1.27
(Hamilton Standard 6507A-2, NACA MR L5G09a).
- A P-51D at J 1.2 is at about 220 mph.
- The momentum model gives 0.79 there on 1,780 hp, about 0.81 on 1,730 hp and 0.86 on
  1,660 hp.
- Another propeller at another power coefficient, so this only says that the 1,730-1,780 hp
  end is plausible. It does not choose between them.

## 6. What the game should use

**Recommendation:**
- efficiency at top speed 0.85 (0.82-0.885), entered as a prior, not fitted;
- C_D0 0.0155 (0.0148-0.0163);
- e kept at 0.8 for the point mass.

Where the range comes from:
- **Low end** (0.0148 / 0.82): the report's 1,780 hp climb, plus Fig. 4's lower power
  required.
- **High end** (0.0163 / 0.885): a 1,660 hp climb (level power plus the ram correction).
- **e** moves C_D0 by only +-0.0001 between 0.75 and 0.9. e matters to the game in another way
  (see "Open").

**How to get it in the game.** Under the game's single-power, constant-thrust-power point mass,
the two targets fix C_D0 at 0.0150 for any efficiency. Only a thrust that falls at low speed
can carry a higher C_D0. The game already has that term: the soft minimum of thrust against
the static thrust.

I checked with the game's own code (`tuning.fit_point_mass`, `game_row` and `performance`,
with `POINT_MASS_TERMS` and `P51D_TERMS` patched in a scratch script):
- **Efficiency 0.85, static thrust as now** (figure of merit 0.75, 26.4 kN):
  - solving C_D0 and the thrust exponent for 375 mph and 3,600 ft/min gives C_D0 0.01558 and
    exponent 3.71 (now 8);
  - the full game row (terms on, flaps) then solves to C_D0 0.01553, efficiency 0.846 and
    e 0.822 (equivalents).
- **Moving the figure of merit instead**, with the exponent kept at 8, needs FM 0.38: a
  16.9 kN static thrust, not physical. So the softness is the lever, not the static thrust.
- **Exponent 2** (v2's value) with FM 0.75 is too soft: it needs an efficiency of 0.93-0.98.
  Exponent 2 does match momentum theory within 3% over 90-380 mph, but only with the larger
  momentum static thrust (FM = eta_p, about 30 kN).

**What changes in the game row**, current -> proposed (real units, sea level):
- top speed 375.0 and best climb 3,600 ft/min: unchanged (targets);
- best-climb speed: 144 -> 160 mph (test: 165);
- thrust power at top speed: +3.2% (efficiency equivalent 0.820 -> 0.846); shaft power
  1,630 hp unchanged;
- static thrust: 26.4 kN, unchanged;
- thrust softness: 8 -> 3.7;
- zero-lift drag: C_D0 equivalent 0.01503 -> 0.01553;
- best sustained turn: 18.99 -> 19.05 deg/s, at 158 -> 159 mph, radius 214 m;
- tightest sustained turn: 202 m at 135 -> 136 mph;
- tightest / best-rate speed ratio: 0.852 -> 0.856 (still inside the 10% margin);
- settled full-stick turn: 17.11 -> 16.98 deg/s;
- low-speed acceleration (at the stall speed): 3.69 -> 3.39 m/s^2. v2's estimate was 2.73.

Why the sustained turn hardly moves: the climb target already pins the excess power near
160 mph, where the sustained turn lives. So the drag split barely reaches the turn.

The separation drag (0.1) needs no change: the best sustained turn's speed moved 1 mph. The
stall, the corner and the corner rate are lift-limited and do not move.

**If the report's 1,780 hp is taken as right**, the current row is already right: 0.0150 /
0.82. Only the best-climb speed (144 against 165) and the low-speed thrust would be off, and
the thrust exponent alone fixes those.

## Open

- **Which climb power?** 1,780 hp (the table) or about 1,730 hp (the table at full throttle,
  my choice). The report's text, which I have not seen, may explain it.
- **The Fig. 4 / Fig. 3 offset** (3-4% in power) is unexplained.
- **e.** NACA's propeller-off polar gives e 0.88-0.91. The point mass assumes 0.8.
  - With e 0.9 (constant efficiency), the point mass's best sustained turn, the game row's
    target, rises from 18.99 to 19.72 deg/s.
  - C_D0 hardly changes: 0.0149 instead of 0.0150.
  - This moves the game more than anything above.

## Reproducing

Scripts are in the session scratchpad (not durable):
`/tmp/claude-1000/-home-vizia-avesnova-boost-and-broadside-v2/37fb8c6c-e7e2-4827-af78-10c9ab60c021/scratchpad/drag/`.
- `data.py`: the readings.
- `fit.py`, `run1.py`-`run5.py`: the fits above.
- `game_variants.py`, `game_shape.py`, `game_row_check.py`: runs with the game's code. Run them
  from the v2 root with `PYTHONPATH=.:src:<scratchpad>` and `BNB_CARD_CACHE` pointed at a
  scratch directory.
- The game-code harness reproduces the current row: C_D0 0.01503, efficiency 0.820, e 0.819,
  climb at 144 mph, sustained 18.99 deg/s at 158 mph, ratio 0.852, settled 17.11 deg/s,
  acceleration 3.69 m/s^2.
