# WWII fighter reference data for the flight model

Gathered 2026-11-23, to build famous WWII fighters as ships with the airframe route
(`lean/game/tuning.py`, `Airframe` and `airframe_row`) and check them against published figures.
The P-51D card and its comparison are in `p51d-model.md`.

**Model inputs.** Each airframe needs:
- mass, wing area S, aspect ratio AR and Oswald e;
- CD0 and clean CLmax;
- shaft power at one rating, propeller efficiency and propeller diameter;
- g limit, stall angle and service ceiling.

The model is quasi-steady at ISA sea level.

**Checks.** Top speed, best climb, sustained and instantaneous turn, corner speed and stall speed.

**Downloads.** All files are under `~/avesnova/boost-and-broadside/data/aircraft-references/<aircraft>/`, written below as `refs/<aircraft>/`.
- Each folder has a `SOURCES.txt` mapping every file to its source URL.
- The scans have no text layer. The figures were read from rendered pages.
- spitfireperformance.com is now a parked domain, and the wwiiaircraftperformance.org P-47 page returns 404. Both sets of files came through web.archive.org.

**Reliability grades:**

| Grade | Meaning |
|---|---|
| A | primary flight test (AAF Wright Field, US Navy Patuxent/TAIC, A&AEE, RAE, AFDU, Rechlin, firm test reports, NII VVS) |
| B | official manual, data sheet or manufacturer calculation (Datenblatt, Aircraft Data Card, Detail Specification, NAA calculation) |
| C | NACA/RAE analysis, or a test of a sister mark carried over |
| D | reputable book (Loftin NASA SP-468, Hoerner, Horikoshi via Nozawa) |
| E | enthusiast reconstruction or simulator data |
| F | forum |

"Derived" marks arithmetic done here, not a source, at ISA sea level unless stated. "Read off chart" means approximate.

**Altitude.** Almost every speed figure is at altitude; sea-level figures are the ones to check the
model with, and the table rows say where they apply. Stall speeds are indicated airspeed unless
stated. Several reports warn that the airspeed system reads wrongly near the stall, which is the
main hazard when deriving CLmax.

## How the inputs can be specified, in general

- **CD0.** Measured CD0 exists for one aircraft only: the Fw 190 A-8/D-9, from a Focke-Wulf drag table. Hoerner analysed the Bf 109 G, and NACA measured the P-51B with the propeller off in dives. Loftin's values (P-51D 0.0163, F6F-3 0.0211) are back-calculated from top speed and power, with an assumed propulsive efficiency of 0.70-0.85 and e of 0.70-0.75 (SP-468 Appendix C). They are fits like ours, not measurements.
- **CD0 depends on the drag state.** Values that include cooling drag differ from "airframe" values by 20-30% (Fw 190: 0.0265 at high speed, 0.0340 in the climb with the cooling flaps open). A fit to top speed and climb absorbs this, together with exhaust thrust and the Meredith effect, into CD0 and propeller efficiency.
- **Oswald e.** Directly available only for the Fw 190 (induced-drag factor K, giving e 0.84 at high speed and 0.78 in the climb) and from Hoerner for the Bf 109 G (about 0.93, optimistic). Everything else is assumed.
- **Clean CLmax.** Measured with corrected airspeed by NACA for the Spitfire VA, F6F-3 and F4U-4. For the others it comes from stall speeds on uncorrected airspeed indicators, which give implausibly high values (CL 1.7-2.1) for the A6M and the Spitfire XIV. The NACA F6F-3 report separates power-off (1.35-1.45), power-on 1 g (2.1-2.2, slipstream and thrust included) and accelerated-turn (about 1.4) values. **The accelerated value is the one a turn model should use.**
- **Turn checks.**
  - Absolute sustained-turn measurements: one primary set (NII VVS 360° times for the Bf 109 G-2/G-4).
  - Instantaneous turn: one calculated radius (AAF, P-51B at 4 g) and one NACA turn time history (Spitfire VA).
  - Everything else is relative: TAIC 17 "gains one turn in 3.5", AFDU "easily out-turns", and similar.
  - **No true energy-manoeuvrability diagram from the period was found for any of these aircraft.** The only E-M charts found are simulator-derived (grade E).
- **g limits.** No g limit was found in a primary document for any aircraft. The P-51's 8 g is a forum figure tied to an 8,000 lb design weight. A degraded Bf 109 G-6 data sheet reads 6.7.

## P-51D Mustang

| Quantity | Value | Condition | Source | Grade |
|---|---|---|---|---|
| Gross weight, as tested | 9,760 lb (4,427 kg) | full ammunition, 184 gal wing + 25 gal fuselage fuel; "clean" includes a bomb rack on each wing | AAF memo report TSCEP5E-1908, P-51D-15 44-15342, 15 Jun 1945 ([page](http://www.wwiiaircraftperformance.org/mustang/p51d-15342.html)) | A |
| Gross weight, calculation | 9,611 lb take-off, 180 gal; wing loading 41.2 lb/ft² | | NAA report NA-46-130 ([page](http://www.wwiiaircraftperformance.org/mustang/p-51d-na-46-130.html)) | B |
| Gross / empty weight | 10,100 / 7,125 lb | "typical maximum gross" | Loftin, NASA SP-468, Table III, p.484 | D |
| Wing area | 233.2 ft² (21.67 m²) | P-51B-1 43-12111 | NACA ACR 4K02 (Nissen, Gadeberg, Hamilton, 1945), p.3 | A |
| Wing area | 233 ft² | P-51D | Loftin Table III | D |
| Span, AR | 37.03 ft, AR 5.815 | P-51B | NACA ACR 4K02 | A |
| Span, AR | 37.0 ft, AR 5.86 | P-51D | Loftin Table III | D |
| CD0 | 0.0163 (f 3.80 ft²) | back-calculated from 437 mph at 25,000 ft and 1,490 hp; assumed η and e (0.70-0.75) | Loftin Table III and Appendix C | D (a fit) |
| (L/D)max | 14.6 | gives e ≈ 0.755 with Loftin's CD0 and AR (derived) | Loftin | D |
| CD, propeller removed, low CL, below drag divergence | about 0.021-0.022 at CL 0.1-0.2, M 0.35-0.65 (read off chart) | P-51B dives from 25,000-30,000 ft, dusty surface; radiator flow without engine heat, so no Meredith thrust | NACA ACR 4K02, Figs. 12-13 | A |
| CD vs CL slope, same dives | ≈0.035 at CL 0.45 (read off chart) | high Mach, partly transition movement | NACA ACR 4K02 Fig. 13 | A |
| Clean stall | about 88 mph IAS at 7,400 lb, rising to about 100.5 mph IAS at 9,800 lb, flaps 0°, gear up (read off chart) | power condition not stated; the chart warns these are "approximately 10 mph higher IAS than previous P-51 airplanes due to a change in the airspeed system" | Pilot's Flight Operating Instructions AN 01-60JE-1, 5 Apr 1944, Fig. 24, p.25 | B |
| Clean stall | 100 mph at 10,100 lb | | Loftin | D |
| Stall, calculation | 95.4 mph at 9,071 lb | landing condition (flaps and gear presumably down) | NA-46-130 | B |
| CLmax implied | 1.50 at 101.5 mph and 9,250 lb, S 21.83 (the October fit); about 1.62 at 100 mph and 9,760 lb, S 21.67; Loftin's figures give about 1.70 | the spread comes from weight basis and IAS position error | derived | - |
| CLmax used in an AAF turn calculation | ≈1.60, back-solved from the 883 ft radius below | "estimated maximum power-on lift coefficient" | derived from the AAF table | - |
| Engine | V-1650-7. WER 67 in Hg / 3,000 rpm; military 61 in / 3,000; normal 46 in / 2,700 | | 44-15342 report | A |
| Power | 1,490 hp | rating not stated | Loftin | D |
| Power vs altitude | V-1650-7 low- and high-blower power charts | | `refs/p51d/V-1650-7-Low-Blower-power-chart.jpg`; others listed on the site | B |
| Propeller | Hamilton Standard four-blade, Dwg K6523A-24 (11 ft 2 in per the October fit) | | 44-15342 report | A |
| Top speed, SL | 375 mph WER; 363-364 military; 323 normal | 9,760 lb | 44-15342 | A |
| Top speed, SL, calculation | 368 WEP / 355 military | 9,611 lb | NA-46-130 | B |
| Top speed vs altitude | WER 375 / 396 / 417 at 0 / 5,000 / 10,000 ft; 442 at 26,000 ft (high blower); 429 at 30,000; 407 at 35,000 mph TAS | 9,760 lb | 44-15342 | A |
| Climb vs altitude | WER 3,600 ft/min from SL to 4,800 ft; 2,925 at 10,000; 2,275 at 15,000; 3,200 at 19,000 (high blower); 1,700 at 30,000; service ceiling 41,600 ft | 9,760 lb | 44-15342 | A |
| Climb, calculation | 3,410 WEP / 3,030 military ft/min at SL | 9,611 lb | NA-46-130 | B |
| Power required (BHP vs TAS) | clean and with stores, at 5,000 and 25,000 ft density altitude | | 44-15342 Fig. 8 "Drag curves", Fig. 4 "Power required" | A |
| Turn radius, calculated | 883 ft at 4 g, 10,000 ft density altitude | P-51B at 9,000 lb, at the estimated maximum power-on CL. "Not necessarily constant altitude turns" | AAF Materiel Command, "Performance Data on Fighter Aircraft", 26 Jul 1944 | B/C |
| Sustained turn | no test value found. Relative: a Zeke 52 gained an advantage on the P-51D "in less than one 360" at 10,000 and 25,000 ft | | TAIC 38 (Eglin, Apr 1945) | A (relative) |
| Sustained turn, simulator | about 15°/s at 150 mph IAS, WEP, 500 ft ("Warbirds v1.11") | | J. Hong, combatsim.com 1998 | E |
| Dive limit | 505 mph IAS; pull-out chart for 4 g | | AN 01-60JE-1, Fig. 23 (limitations placard) and Fig. 26 | B |
| Dive, measured | Mach 0.85 | 44-14134 compressibility dives | `refs/p51d/p-51d-dive-27-feb-45.pdf` | A |
| g limit | +8 / -4 g at an 8,000 lb design gross weight, 12 g ultimate; about 6.4-6.6 g at 9,760-10,000 lb if n·W is held | | forum posts only (ww2aircraft.net; DCS forum quoting a manual) | F |

Charts (all `refs/p51d/`):
- `P-51D_15342_Level.jpg`, `P-51D_15342_Climb-Fig5.jpg`: speed and climb vs altitude (A).
- `P-51D_15342_Drag_Curves.jpg`, `P-51D_15342_Power-Required.jpg`: power-required curves at two altitudes (A). These are the best P-51 material for fitting CD0 and e together, given a propeller efficiency.
- `na-46-130-chart.jpg`: NAA calculated altitude performance (B).
- `P-51D-manual-5april44.pdf`: AN 01-60JE-1 (B). Stall-speed chart p.25 (Fig. 24/25); dive chart p.26; limitations Fig. 23.
- `NACA_ACR_4K02_P-51B_drag_flight_vs_tunnel.pdf`: propeller-off drag vs Mach and CL (A).
- `../comparative/Performance_Data_on_Fighter_Aircraft.pdf`: AAF 1944 table of climb, dive, 4 g turn radius and roll for the P-38J, P-47, P-51B and P-80A (B/C).
- `../comparative/Power_required_available_P-51A_P-51B_P-47B.pdf` (A/B; not read in detail).
- `../comparative/combatsim_hong1998_energy2-p51.jpg`: E-M chart (doghouse) from a simulator (E). Useful only as a picture of the form.

Conflicts:
- **Sea-level top speed.** 375 mph tested against 368 calculated, at the same WER and different weights. The test aircraft carried racks.
- **Wing area.** NACA's 233.2 ft² (21.67 m²) against the card's 21.83 m² (235 ft²). The 0.7% difference moves CLmax and CD0 by the same fraction.
- **CLmax and stall speed.** The manual's IAS stall is known to read about 10 mph high against earlier P-51s, so stall-derived CLmax ranges from 1.50 to 1.70 with the weight basis. The AAF's own turn estimate implies about 1.6.
- **CD0.**
  - The fitted 0.0149 and Loftin's 0.0163 sit well below the measured propeller-off 0.021-0.022. This is expected: the propeller-off airframe has no Meredith thrust from the radiator and no slipstream accounting, and its surface was dusty.
  - It means the fitted CD0 is an "effective" value tied to the fitted propeller efficiency (0.82), not the airframe's drag.

What was not found: a measured clean CLmax with calibrated airspeed for the P-51D; the stall angle; a primary g limit; any measured turn radius, turn rate or time for 360°.

**Completeness: 8/10.** The best-documented airframe for speed, climb and power-required (A), and the only one with a power-required chart for a drag fit. CLmax is good to about ±8%. Turn and g limit are unchecked.

## Supermarine Spitfire XIV (and IX for comparison)

| Quantity | Value | Condition | Source | Grade |
|---|---|---|---|---|
| XIV tare / service load | 6,578 / 8,488 lb (maximum permissible 8,500) | RB.141 | A&AEE weighing report (spitfireperformance.com/rb141weights.html, archived) | A |
| XIV test weights | JF.319 8,400 lb (level speeds corrected to 7,980 lb); AFDU RB.179 8,400; CFE 8,500 | | A&AEE JF.319, 27 Oct 1943; AFDU; CFE | A |
| Wing area, span | 242 ft² gross (22.48 m²), 36 ft 10 in, AR 5.61 (derived) | full-span "universal" wing | Aircraft Data Card, XIV, 28.9.44 | B |
| AR | 5.65 (span 36.92 ft) | Spitfire I | ARC R&M 2535 (Mair), Table 1 | C |
| CD0 | none measured. ≈0.019 derived from 359-360 mph at SL, +18 lb, ≈1,840 bhp, η 0.82 and e 0.8 assumed, exhaust thrust ignored, so ≈0.020-0.021 airframe | | derived | - |
| Wind-tunnel CD | ≈0.018-0.019 at CL 0-0.1, M 0.3-0.5, R 1.1×10⁶ | Spitfire I model, no cooling flow; a lower bound | R&M 2535 Fig. 5 | C |
| CLmax (Spitfire VA) | power off 1.18 (gun ports covered) / 1.15 (open) at 90-91 mph indicated; power on (+3¾ lb) 1.63-1.68 at 76-77 mph | 6,184 lb, calibrated system | NACA WR L-353 (Vensel & Phillips, 1942), p.7 | A for VA, C for XIV/IX |
| CLmax in accelerated flight | 1.21-1.22; about 1.10 averaged through a stalled turn | VA, flaps up | NACA WR L-334 p.9, L-353 p.8 | A (VA) |
| XIV stall | 87 mph ASI clean; 75 mph flaps and gear down | 8,490 lb, ASI uncorrected; position error is large here (CAS about 93-97 mph), so do not fit CLmax to 87 | A&AEE RB.141 handling | A (with caution) |
| Griffon 65 power (2,750 rpm, read off chart) | +18 lb: ≈1,840 bhp at SL, ≈2,000 at MS FTH (~12,500 ft), ≈1,720 at ~25,000 ft FS; +21: ≈2,040 at SL; +25: ≈2,320 at SL | | Rolls-Royce Fig. 4 (ENG.8722) | B |
| Propeller | Rotol five-blade, 10 ft 5 in, reduction 0.51 | JF.319 | A&AEE | A |
| XIV top speed SL | 359 mph +18 (Ministry of Supply 20.7.44); ≈360 (CFE, 8,500 lb); 363 (JF.319, 7,980 lb, RG5SM engine); +25: 389 (MoS) / ≈396 (RAE TN Aero 1501, RB.176) | | as stated | A/B |
| XIV top speed vs altitude | CFE +18 at 8,500 lb: ≈416 at ~12,000 ft, ≈447 at ~26,000 ft (read off chart); ADC 417 at 12,000 and 448 at 26,000 ft | | CFE Report 78 / AFDS chart; ADC | A/B |
| XIV climb | JF.319, +18, 8,400 lb: 5,040 ft/min at 2,100 ft; 3,600 ft/min from 10,000 to 21,800 ft; 20,000 ft in 5.1 min. CFE: ≈4,700 ft/min SL to ~9,000 ft (read off chart) | | A&AEE JF.319 Table I; CFE chart | A |
| XIV ceiling | 44,000 ft service (JF.319 estimate); 43,500 ft (ADC, maximum weight) | | | A/B |
| XIV turn | no numbers. AFDU RB.179: turning circles of the XIV and IX "identical"; XIV "easily out-turns" the Bf 109 G and Tempest V and turns inside the Fw 190 A | | AFDU | A (relative) |
| XIV dive | 470 mph ASI reached, satisfactory | RB.141 | A&AEE | A |
| LF IX weights | tare 5,749; maximum 7,450 lb (ADC). Tests: BS.543 7,485 lb (speeds at 7,110 lb); JL.165 7,400 lb | | ADC 28.10.43; A&AEE | A/B |
| Merlin 66 power (3,000 rpm, read off chart) | +18: ≈1,570 bhp at SL; +25: ≈1,975 at SL | | Rolls-Royce Fig. 3 (ENG.8720) | B |
| LF IX top speed SL | 336 mph +18 (BS.543); 340.8 (RAE, Oct 1944); 330 (RR, JL.165); +25: 354 at SL (RR), 360 at 2,000 ft (A&AEE JL.165) | | as stated | A |
| LF IX top speed altitude | BS.543 384 at 10,800 ft (MS); 407 at 22,000 ft (FS) | | A&AEE | A |
| LF IX climb | BS.543 +18: 4,620 ft/min at SL, 4,700 at 7,000 ft; +25: 5,080 ft/min at SL (A&AEE JL.165), 5,740 (Rolls-Royce JL.165) | | | A |
| IX CD0 | ≈0.020 derived (BS.543, +18, η 0.82, e 0.8); ≈0.022 from JL.165 at +25 | | derived | - |
| IX instantaneous turn (VA) | rapid 180° turn entered at 174 mph IAS, peak about 3.5-4 g at the stall; about 180° in 8-9 s (about 20-22°/s); speed fell to about 158 mph (read off chart) | 6,184 lb | NACA WR L-334 Fig. 13 | A (VA) |
| g limit | design factors 9.0 (CP forward) and 6.0 (CP aft) from the F.7/30 specification, quoted from Morgan & Shacklady | not stated whether proof or ultimate | via PPRuNe | D via F |
| Dive limit IX | 450 mph IAS | Pilot's Notes as quoted | forum | F |

Charts (`refs/spitfire/`):
- XIV: `spitfire14-afds-speedchart.jpg`, `spit14speedchart.jpg`, `spit14climbchart.jpg` (CFE, A); `JF319-level-speeds.jpg`, `JF319_Report_P3792.pdf` (A); `Speeds_Spitfire_XIV.jpg` (B); `spit14+25lbs.jpg` (RAE TN 1501, A).
- IX: `bs543speed.jpg`, `bs543climb.jpg`, `jl165speed.jpg`, `jl165climb.jpg` (A); `Spitfire_IX_RAE_Level_Speed.jpg` (A).
- Data cards: `Spitfire_XIV_ADC.jpg`, `Spitfire_IX_LF_ADC.jpg` (B).
- Engines: `griffonhp_b.jpg`, `merlin66hpchart.jpg` (B).
- NACA: `NACA-WR-L-334_Spitfire_VA_flying_qualities.pdf` (turn time histories Figs. 13-20), `NACA-WR-L-353_Spitfire_VA_stalling.pdf` (CLmax table) (A).
- Tunnel: `ARC_RM_2535_Mair_HS_tunnel_Spitfire_Spiteful_Mustang.pdf` (C).
- Not inspected: `spit14at21.jpg` (unidentified hand-drawn chart), `merlin66_18_25.jpg`.

Conflicts:
- **XIV sea-level speed.** JF.319 (363 mph) was a prototype with a lower MS gear, a polished finish, and speeds corrected to 95% weight. Production aircraft run 359-360. The MS-gear peak sits at 5,050 ft on JF.319 but about 12,000 ft on the Griffon 65.
- **LF IX at +18.** Sea level is 330 / 336 / 341 mph on different airframes.
- **LF IX at +25.** Vickers measured faster than A&AEE on JL.165, and A&AEE noted it. Airframe-to-airframe spread is about ±1.5%.
- **Climb.** Climb depends on the radiator flaps: about 415 ft/min in the FS gear. Firms measured with the flaps shut, A&AEE with them open.
- **CLmax.** The XIV's raw ASI stall implies CL ≈1.8, against NACA's 1.15-1.18 power off for the VA. The NACA value is the reliable one.

Not found: measured CD0 or polar for a full-scale Spitfire; e; propeller efficiency; stall angle; Pilot's Notes limits (AP 1565); any sustained-turn number for the IX or XIV.

**Completeness: XIV 6/10, LF IX 6.5/10.** Speed, climb, power and weights are A/B with several airframes. CLmax is A-grade but from the VA. CD0, e and propeller efficiency must be fitted. Turn checks are relative only. The IX gains half a point for the NACA VA instantaneous-turn trace of the same wing.

## Messerschmitt Bf 109 G-6 and K-4

Ratings:
- **1.30 ata / 2,600 rpm:** climb and combat power (30 min).
- **1.42 ata / 2,800 rpm:** take-off and emergency power, barred until about spring 1944.
- **MW 50:** about 1.70 ata (G-6/AS) and 1.80/1.98 ata (K-4, DB 605 DB/DC).

**Do not mix 1.30- and 1.42-ata figures.**

| Quantity | Value | Condition | Source | Grade |
|---|---|---|---|---|
| G-6 empty / equipped / take-off | 2,268 / 2,679 / 3,196 kg | fighter, internal fuel 296 kg | GL/C-E2 data sheet 13.8.44 (`me109g6-glce2-13aug44.jpg`) | B |
| G-6 test weight | ≈7,488 lb (3,397 kg), 43.6 lb/ft² | G-6/U-2 TP814, gondola cannon | AFDU tactical trials 1944 | A |
| Wing area | 16.05 m² (span 9.92 m, AR 6.13, self-consistent); also 16.2 (GL/C-E2 20.8.44), 16.16 (NII VVS), 172 ft² (Hoerner) | | as stated | B |
| CD0 / flat plate | f 5.8 ft² (0.539 m²); CD0 0.0337 on 172 ft² | 1944 G at 610 km/h, 22,000 ft, 1,200 hp, 6,700 lb, η 0.85, ≈140 lb exhaust thrust; includes ≈6% compressibility and the radiators | Hoerner, *Fluid-Dynamic Drag*, pp.14-3 to 14-7 (`Hoerner-Me_109.pdf`) | D (an analysis by a former Messerschmitt aerodynamicist) |
| CD0 at low speed | ≈0.031-0.032 (Hoerner less compressibility); ≈0.027 cross-check from 510 km/h at SL, 1,310 PS, η 0.82 | | derived | - |
| Oswald e | ≈0.93 (effective AR 5.8 against geometric 6.1, +2% planform) | at CL 0.21 | Hoerner p.14-4 | D, optimistic |
| Drag breakdown | wing 37.5%, fuselage 13.7%, tail 6.9%, engine and radiators 23.3%, appendages 11.4%, induced 7.2% | at top speed | Hoerner p.14-7 | D |
| Clean CLmax | not measured. ≈1.2 inferred from a 112 mph IAS clean stall | G-2 trop; weight not stated; the report says "treat with reserve" | No. 209 Group trial | A (weak) |
| Slat opening | 230-240 km/h IAS required; 180-220 on some aircraft | | Messerschmitt VB 109 10 E 42 | A |
| DB 605 A power | 1,310 PS at SL / 1,250 at 5.8 km (1.30 ata); 1,475 / 1,355 at 5.7 km (1.42 ata); 1,075 continuous | | GL/C-E2; Messbrief WNr 16647 | B |
| Propeller | VDM 9-12087, three-blade, 3.0 m | | GL/C-E2 | B |
| G-6 top speed | 510 km/h SL / 630 at 6.6 km (1.30 ata, 3,196 kg); 530 / 640 (1.42) | calculation | GL/C-E2 13.8.44 | B |
| G-6/AS with MW 50 | ≈558 km/h SL, ≈690 at ~7.3 km (read off chart) | 1.7 ata | Daimler-Benz test 7.6.44 (`me109-gj-fx.jpg`) | A |
| G-1 measured | 314 mph SL, 391 mph at 21,654 ft (average of 90 aircraft, 6,724 lb, 1.30 ata) | | Messerschmitt 14.1.44; VB 109 19 L 42 | A |
| Climb | 17.0 m/s at SL, 6 km in 6.0 min, ceiling 11.2 km (3,200 kg, 1.30) | calculation | GL/C-E2 | B |
| Climb, measured | G-1 average 3,427 ft/min SL (1.30); G-2 NII VVS 16.8 m/s at 3,235 kg; Finnish G-2 20.6 m/s (300 km/h IAS climb, radiators more closed) | | Messerschmitt; NII VVS 1944; Finnish MT-215 | A |
| **Sustained turn, 360°** | G-2 3,235 kg: 22.6 s left / 22.8 s right. G-2 3,023 kg: 20.0 / 21.5 s. G-4 3,027 kg: 20.5 / 21.0 s | 1,000 m, 1.30 ata, speed and bank not stated | NII VVS "German aircraft" 1944, p.23 (`109G_NIIVVS_1944.pdf`, hosted by kurfurst.org) | A |
| Combat-turn climb | 1,100-1,150 m gained from 1,000 m | G-2/G-4 | NII VVS p.22 | A |
| Turn, relative | Spitfire IX "easily out-turns" it; Tempest V equal below 350 mph IAS (AFDU). "Radius of turn is very poor" (USAAF EB-102) | | | A (relative) |
| Dive limit | 750 km/h IAS to 3 km; 700 at 5 km; 575 at 7 km; 450 at 9 km | | GL technical instruction 28.8.42 | B |
| Dive measured | 737 km/h IAS at 4.5 km; Mach 0.805 at 7 km | 2,900 kg | Messerschmitt 109 05 E 43 | A |
| g limit | Lastvielfaches 6.7 at 3.32 t (degraded scan; limit or ultimate not stated) | | GL/C-E2 13.8.44 | B (uncertain reading) |
| K-4 design weight | 3,400 kg | | Messerschmitt Oberammergau "Leistungen 8-109 K4 und K6", 5026/17, 11.12.44 | B |
| K-4 power (DB 605 DC) | 2,000 PS at SL, 1,800 at 4.9 km (1.98 ata, C3 + MW 50, 2,800 rpm); 1,370 PS at 2,600 rpm short-time | | DB 605 DC engine card, Dec 1944 | B |
| K-4 top speed (calculation, read off chart) | ≈600-607 km/h at SL, ≈718-721 at ~6 km (1.98 ata); ≈596 at SL (1.80 ata); ≈535 at SL (1.45 ata) | 3,400 kg, no Mach losses | 5026/18, /27 | B |
| K-4 top speed, measured | ≈538 km/h at 3 km to ≈630-638 at ~8.5 km (read off chart) | 1.35 ata only; the engine failed at 1.98 ata | Aspera VDM 4427, 4.1.45 | A |
| K-4 climb (calculation) | ≈24-24.7 m/s at SL (1.98 ata); ≈22 (1.80 ata) | 3,400 kg | 5026/19, /28 | B |
| K-4 CD0 | ≈0.024 cross-check (596 km/h, ≈1,850 PS, η 0.82) | | derived | - |

Charts (`refs/bf109/`):
- G-6 performance: `me109g6-combat-emergency.jpg` (speed vs altitude at 1.30/1.42 ata and engine powers), `me109-gj-fx.jpg` (MW 50), `me109g6-glce2-13aug44.jpg` and `glce2-109g6.jpg` (data sheets).
- Other test material: `VB-109-19-L-42.pdf` and `me109g-14026-page6/8-1200.jpg` (G-1 measured); `Hoerner-Me_109.pdf` (drag); `109G_NIIVVS_1944.pdf` (turn times); `Me_109_9228_dive-Mach_Nr.jpg`, `tig28aug42.jpg` (dive).
- K-4: `K4_Mtt_Projektbuero_5026-*.jpg` (speed and climb calculations), `Me_109K-4_DB605DB_Climb.jpg`, `Aspera_VDM_4427.pdf`, `DB605DC_limits_dec44_Motorenkarte.jpg`.
- Saved but not read in detail: `me-109g2-finn.pdf`, `me109g2-6-pg26.jpg`.

Conflicts:
- **Wing area** (16.05-16.2 m²): differing treatment of the centre section.
- **Sea-level speed at 1.30 ata** (495-515 km/h): airframe condition, worth ±8 km/h each for the gear doors, finish, tropical filter and gondolas.
- **Finnish and Soviet climb figures run higher than German ones:** climb speed, radiator setting and reduction method.
- **Hoerner's CD0 (≈0.031 at low speed) against the sea-level cross-check (≈0.027):** Hoerner includes exhaust thrust and works at altitude.
- **K-4:** almost all figures are calculations for an ideal series aircraft. The only measurement is at reduced boost.

Not found: a measured clean CLmax or a German ca max; the stall angle; a Messerschmitt turn calculation (Wendezeit); speed and bank for the NII VVS turns; a primary g-limit (Festigkeit) document; any K-4 handling or turn data; the RAE/A&AEE G-6 trials.

**Completeness: G-6 7/10, K-4 4/10.**
- **G-6:** speed, climb, power, weights and dive limits are A/B, with a drag analysis and the only primary 360° times in this set. CLmax is weak, and e is optimistic.
- **K-4:** calculations only, at its headline boost. Airframe values must be inherited from the G.

## Focke-Wulf Fw 190 A-8 and D-9

| Quantity | Value | Condition | Source | Grade |
|---|---|---|---|---|
| Wing area, span, AR | 18.3 m², 10.5 m, AR 6.0 (15.6 m² outside the fuselage) | | Focke-Wulf "Widerstandsdaten von Flugzeugen", Dec 1944 (`fw190_drag_data.jpg`); GL/C-E2 | B |
| Wing area, British | 203 ft² gross (18.86 m²); span 34.5 ft | A-3 | US Special Observer report 234, 5 Aug 1942 | A/B |
| **A-8 drag polar** | cw·F = 0.485 + 1.16 ca² m² (high speed); 0.623 + 1.24 ca² (climb, cooling flaps open). CD0 0.0265 / 0.0340; e 0.84 / 0.78 (derived from K) | includes cooling (15%), guns (7.2%), intake, radio | Focke-Wulf drag table | B |
| **D-9 drag polar** | cw0·F 0.444 (high speed) / 0.596 (climb); CD0 0.0243 / 0.0326; same K, so e 0.84 / 0.78 | intake 4.3%, cooling 8.8% | same | B |
| ca max | 1.58 | configuration not stated; matches the landing speed of 165 km/h at 3.8 t with flaps 60°, so probably flaps down | same; GL/C-E2 | B |
| Clean CL at stall | ≈1.2-1.34 derived from an idle stall at Va 190-200 km/h, ≈4.25 t, gear and flaps up (indicated, uncorrected) | A-8 733705 | Flugbericht 733705 Nr. 1 | A (derived) |
| Clean stall, US | ≈118 mph IAS power off | A-5(?), weight not given | AAF Eng-47-1658-D | A |
| A-8 weights | take-off 4,300 kg (Fw Flugleistungen 25.10.44); 4,395 (drag table); 4,365 (GL/C-E2); equipped empty 3.2 t | | as stated | B |
| A-8 test weight | 4,224 kg | 733705 report 2 | | A |
| BMW 801 D-2 power | 1,800 PS at 2,700 rpm, 1.42 ata, SL (take-off/emergency); ≈1,560 PS at 2,400 rpm, 1.32 ata (combat). Erhöhte Notleistung 1.58/1.65 ata given only as +22/+25 km/h | | GL/C-E2; BMW 801 handbook p.15; Fw TD 284 | B |
| Propeller | VDM three-blade, 3.30 m; static thrust 1,836 kg | | GL/C-E2; drag table | B |
| A-8 top speed | measured 537 km/h SL, 617 at 6.1 km (take-off power, 4,224 kg); calculated 548 (578 with ENL) at SL, 644 at 6.3 km (4,300 kg) | | 733705 report 2; Leistungsdaten 1.10.44 | A / B |
| A-8 climb | ≈13.4-14.4 m/s at SL at 2,700 rpm (Fw sheets); 9.6 m/s at combat power; 14.0 with ENL (Leistungsdaten) | 4,300 kg | Fw Flugleistungen 13.11.43, GL/C-E2; Leistungsdaten 1.10.44 | B |
| A-3/A-4 captured | RAE A-3: 372 mph at 17,000 ft (8,580 lb, 1.34 ata), 2,810 ft/min at 3,250 ft. US Navy A-4: 334 mph at 200 ft, 410 at 25,000 ft (8,690 lb) | | RAE 21.12.43; PTR-1107 | A |
| A-8 dive | measured Mach 0.79, 912 km/h TAS (Wk 0022, from 10.1 km). Limits 750 / 690 / 580 km/h by altitude band | | Fw memo 24.9.43; US Tactical and Technical Trends No. 23 | A / C |
| D-9 take-off weight | 4,270 kg (drag table, Fw climb chart); 4,350 (Rechlin, with ETC 504) | | | B / A |
| Jumo 213 A power (read off chart) | ≈2,100 PS SL with MW 50; ≈1,780 Start/Not; ≈1,600 Steig/Kampf; exhaust thrust ≈140/115/95 kg at SL | | Jumo Bl. 2610, 9.10.44 (`Jumo_213A_power_graph.jpg`) | B |
| D-9 propeller | Heine wooden three-blade, 3.50 m, t/D 11.5%; static thrust 2,072 kg (2,227 with MW 50) | | Rechlin 210006; drag table | A/B |
| D-9 top speed | Rechlin 540-550 km/h SL, 645-655 at 6.6 km (3,250 rpm, 4,350 kg, ETC); 210001: 341 mph SL, 413 at 20,177 ft; with MW 50 606 km/h SL (210002, sealed and polished); calculated 576 (612 MW 50) SL | | Rechlin 2.3.45; Fw Flugberichte; Leistungsdaten | A / B |
| D-9 climb | Rechlin 17.0 m/s at SL (4,300 kg, 3,250 rpm), climb speed 280 km/h; Fw chart ≈21 m/s with MW 50 (read off chart) | | | A / B |
| Ceilings | A-8 9.95 km (10.6 with ENL); D-9 11.1 km measured (V53) | | | A/B |
| Turn | no figures. "Turn is poor" (AAF); F4U-1 and F6F-3 "far superior" in turning circles (Navy PTR-1107); "radius of turn is poor" for the D-9 (AAF FE-121) | | | A (relative) |
| g limit | not found in a primary source | | | - |

Charts (`refs/fw190/`, URLs in its `SOURCES.txt`):
- Drag: `fw190_drag_data.jpg`, the drag polar for the A-8, A-9, D-9, D-12 and Ta 152. **The best single drag document in this set.**
- A-8 performance: `fw190-a8-25oct44.jpg`, `fw190-a8-level-speed-13nov43.jpg`, `fw190-a8-climb-13nov43.jpg`, `leistungsdaten-1-10-44.jpg`, `fw190-a8-glce2.jpg`.
- A-8 tests: `Fw_190_A8_733705_Rep1.pdf`, `Fw_190_A8_733705_Rep2.pdf`.
- D-9: `fw190d9-climbchart-flugmechanik-24-3-45.jpg`, `Fw_190_D-9_210006_flight_performance.jpg`, `Fw_190_D-9_2100*`, `Fw_190_V53_*`, `wright-field-fw190d-9.pdf`.
- Engines: `Jumo_213A_power_graph.jpg`, `bmw801-handbook-pg15.jpg`.
- Captured aircraft: `fw190-a3-rae-21dec43.pdf`, `ptr-1107.pdf`, `fw-190-rep234.pdf`.
- Dive: `Fw_190_Dive.pdf`.

Conflicts:
- **Wing area.** 18.3 m² against 18.86 m² British gross. Use 18.3 with AR 6.0, the basis of the drag polar.
- **CD0 against the P-51.** 0.0265 against 0.0149 looks extreme, but the Focke-Wulf figure is a full drag state including cooling, guns and intake. With it, 548 km/h at SL needs ≈1,510 PS of thrust power, against 1,800 PS shaft: η ≈0.84 (derived). For the D-9, η is ≈0.77 (derived).
- **ca max.** 1.58 is probably flaps down; the clean value is about 1.2-1.34.
- **Measured against calculated speed.** A-8 733705 measured 2-4% below the calculation. Focke-Wulf attributes the D-9 shortfall to early engines 60-100 PS weak, unsealed cowl gaps (~15 km/h), and the ETC rack and wheel doors (8-10 km/h).

Not found: a quantitative turn figure; a Focke-Wulf turn comparison; a primary g limit; a measured propeller efficiency; a D-9 stall speed; the BMW 801 ENL power in PS; an NACA tunnel test.

**Completeness: A-8 7.5/10, D-9 7/10.** The only aircraft with a manufacturer drag polar giving CD0 and e directly, in two drag states. Power curves (with exhaust thrust for the D-9) and speed and climb are A/B. CLmax is approximate. Turn and g limit are missing.

## Mitsubishi A6M2 Model 21 and A6M5 Model 52

| Quantity | Value | Condition | Source | Grade |
|---|---|---|---|---|
| A6M2 weight as tested | 5,555 lb (2,520 kg) | "full military load", Akutan Zero, Navy San Diego | AAF memo 23 Oct 1942 (`a6m2-oct2342.pdf`); Informational Intelligence Summary 85, Dec 1942 (`intelsum85-dec42.pdf`) | A |
| A6M2 wing area, span | 22.44 m² (241.5 ft²), 12.0 m, AR 6.42 (derived) | | Japanese figures via ja.wikipedia (Nozawa) | D |
| A6M2 engine | Sakae 12: take-off 940 PS at 2,700 rpm; 950 PS at 4,200 m. US estimate 900 hp at 16,000 ft | | ja.wikipedia (Nozawa); IIS85 | D / B |
| A6M2 top speed (Navy calibrated) | 270 mph SL; 305 at 10,000; 326 at 16,000 ft (critical altitude); 306 at 30,000 | 5,555 lb | IIS85 p.5 | A |
| A6M2 top speed (Navy preliminary) | 277 SL; 335 at 16,000 ft | not reduced to standard | Oct 1942 memo | A |
| A6M2 climb | 2,750 ft/min at SL; 2,380 at 15,000 ft; ceiling about 38,500 ft | Navy calibrated | IIS85 | A |
| **A6M2 stall** | gear up: 78 mph power off / 74 mph power on; gear and flaps down: 69 / 61 mph (indicated) | 5,555 lb | IIS85 p.6 | A |
| A6M2 CLmax implied | ≈1.48 power off / ≈1.64 power on (IAS taken as EAS) | | derived | - |
| Other A6M2 stall data | 62-70 mph (23rd FG, Kunming); about 65 mph (Wright Field). These imply CL 1.85-2.07: airspeed position error, discard | | `p5016.pdf`, `ENG-47-1673-A.pdf` | A (airspeed unreliable) |
| A6M2 CD0 | ≈0.020-0.022 from 326 mph at 16,000 ft, ≈937 hp, η 0.80-0.85, e 0.8 | | derived | - |
| A6M2 handling | above 200 mph IAS increasingly hard to manoeuvre; at 300 mph IAS "practically impossible to make it roll"; above 300 IAS "virtually impossible to reverse a turn" | | 23rd FG report; IIS85 | A |
| A6M2 turn, relative | "no comparison" with the F4F-4's turning circle; superior to the P-38F in slow turns at all altitudes; turns "a little shorter" than the P-36A | | IIS85; 23rd FG | A (relative) |
| A6M2 g limit | 7 × 1.8 safety factor (Navy requirement) per Horikoshi, seen only in forum transcriptions; a forum alternative is 6 × 2.0 | | | D claimed / F as seen |
| A6M2 dive limit | 629.7 km/h (340 kt) | | ja.wikipedia | D |
| A6M5 weights | 6,094 lb as flown (TAIC 17, Patuxent); 5,860 lb (TAIC 38, Eglin); empty / normal 4,236 / 6,026 lb (TAIC 102D) | | TAIC 17 = PTR-1111, Nov 1944; TAIC 38, Apr 1945; TAIC manual 102D, Mar 1945 | A / B |
| A6M5 wing area, span | 230 ft² (21.37 m²), 36.1 ft (11.0 m), AR 5.68 (derived); Japanese 21.30 m² | | TAIC 102D; ONI table | B |
| A6M5 engine | test aircraft labelled Sakae 31A, no water injection. 102D: take-off 1,120 hp at SL; military 1,080 hp at 9,300 ft; Japanese Sakae 21: 1,130 PS take-off | | 102D; ja.wikipedia | B / D |
| A6M5 propeller | three-blade, 10 ft (3.05 m) | | 102D | B |
| A6M5 top speed | measured 335 mph at 18,000 ft (Patuxent, 6,094 lb); ≈290 SL and ≈335 at 19,500 ft (Eglin chart, rough airframe, 5 in Hg short); 102D estimate 295 SL / 351-358 at 22-23,000 ft, noting "only 340 mph has been obtained in flight tests" | | TAIC 17, 38, 102D | A / B |
| A6M5 power sweep | 301.9 mph at 5,030 ft (2,700 rpm, 37 in) and 279.0 / 256.3 / 219.6 / 174.7 mph at lower settings | | ATSC memo TSFER-1973, Zeke 52 EB-2, 1946 | A |
| A6M5 climb | 102D 3,140 ft/min at SL (estimate); EB-2 "3,?00" ft/min (digit illegible); Patuxent: about F6F-5 +600 ft/min to 9,000 ft, equal to the F4U-1D to 10,000 ft | | | A / B |
| **A6M5 turn, relative** | at 10,000 ft it gains one turn in 3.5 on the F6F-5 and the F4U-1D, falling to about parity at 30,000 ft; one turn in 8 on the FM-2. Gains on the P-51D within one 360 at 10,000 and 25,000 ft | | TAIC 17; TAIC 38 | A (relative) |
| A6M5 handling | controls tighten about 160 kt; above 200 kt "objectionably heavy"; rolls equal to the F6F/F4U below 200 kt, worse above | | TAIC 17 | A |
| A6M5 dive | red line 325 mph IAS in test; 360 kt Japanese limit | | TAIC 38; ja.wikipedia | A / D |

Charts (`refs/a6m/`):
- `zeke52-taic38.pdf` p.9-10: speed and manifold pressure vs altitude; time to climb.
- `Zeke-52-TAIC-102D.pdf` p.1: speed, climb and range.
- `japfighters-comp.jpg`: ONI comparative table.
- Pilots' reports and tests: `ptr-1111.pdf` (TAIC 17), `intelsum85-dec42.pdf`, `a6m2-oct2342.pdf`, `p5016.pdf`, `ENG-47-1673-A.pdf`, `Zeke_52_Pilots_Comments.pdf`.
- No turn charts or drag polars exist in these files.

Conflicts:
- **Top speeds.** Measured figures run below Japanese and TAIC-estimate figures by 15-25 mph. The test aircraft were short of manifold pressure, rough, used US fuel or were not reduced to standard. Japanese figures are D grade and could not be checked against the source books.
- **Stall and CLmax.** Only the Navy San Diego stalls give plausible values. The others carry large airspeed errors.
- **g limit.** Unverified.
- **Engine.** The Sakae 31A in the US tests and the Sakae 21 in other tables are unreconciled.

Not found: measured CD0 or e (the Akutan Zero was in the Langley full-scale tunnel in March 1943, but no report was written, per NASA's *Cave of the Winds*); any absolute turn figure; primary Japanese test reports; the A6M2 propeller diameter; a primary g limit.

**Completeness: 5/10.**
- Weights, A6M5 geometry, and speed and climb are A/B but degraded.
- CLmax rests on one Navy stall set.
- CD0 and e must be fitted. The A6M2 wing area is D only.
- Turn checks are relative. They are, though, the most useful relative turn data in the set: TAIC 17's turn ratios against three US fighters, and the control-force onset that caps instantaneous turn above about 200 kt.

## Optional extras

### Grumman F6F-3 / F6F-5 Hellcat (well documented; recommended)

| Quantity | Value | Condition | Source | Grade |
|---|---|---|---|---|
| Wing area, span, AR | 334 ft², 42.833 ft, AR 5.5 | | NACA MR L5B13 (flying qualities, Part I), p.2 | A |
| Propeller | Hamilton Standard three-blade, 13 ft 1 in | | NACA MR L5B13 | A |
| Weights | NACA as flown 10,600-11,500 lb; normal fighter 11,364 lb; Navy tests 11,364-12,680 lb | | NACA; Navy reports | A |
| **CLmax, clean, power off** | 1.35-1.45 | ~11,200 lb, corrected airspeed, normal-force coefficient | NACA MR L5B13b (Part III, stalling), p.3 | A |
| CLmax, clean, power on (2,550 rpm, 43 in) | ≈2.1-2.2 (includes slipstream and thrust) | | same | A |
| **CLmax, accelerated turn** | ≈1.4 | wind-up turns, climbing power | same, p.4 | A |
| CLmax vs Mach | ≈1.35 at M 0.35; ≈1.0 at M 0.5; ≈0.7 at M 0.65 (read off chart) | ~25,000 ft | NACA TN 1044, Fig. 4 | A/C |
| Stall | 93.5 power off / 91.5 power on mph (11,364-11,573 lb); 100.5 / 98.5 mph (12,680 lb) | clean | NAS Patuxent final reports (02982, 42874) | A |
| CD0 | 0.0211 (f 7.05 ft²), back-calculated | 2,000 hp, 375 mph at 17,300 ft | Loftin Table III | D |
| Power | 1,650 bhp military (high blower); combat ≈2,000 bhp at take-off | | Navy reports | A |
| Top speed | F6F-3 375-382 mph at about 22,300-24,550 ft; specification 312 mph at SL; F6F-5 391 mph at 23,100 ft | | Navy tests; Detail Specification (B) | A/B |
| Climb | Navy tests 2,500-3,120 ft/min (normal/military); combat 3,090-3,470 (low blower); specification 3,650 at SL | | | A/B |
| Turn | relative: Zeke 52 gains one turn in 3.5 at 10,000 ft (TAIC 17); F6F-3 out-turns the Fw 190 A-4 (PTR-1107) | | | A (relative) |

Files: `refs/f6f/` (the NACA MR L5B13 and L5B13b reports, NACA TN 1044, and Navy reports with speed and climb charts).

Conflicts:
- **Aspect ratio.** Loftin's AR of 5.34 does not follow from his own span and area. Use 5.5.
- **Power-off CLmax.** The Navy minimum speeds imply about 1.5. NACA's corrected 1.35-1.45 is preferred.

Not found: g limit, dive limit, measured CD0 or e, measured turn rate.

**Completeness: 7/10.** The best-measured CLmax set in this survey, split into power-off, power-on and accelerated values.

### Vought F4U-1 / -4 Corsair

- **Geometry.** NACA RM A7C05 (F4U-4) gives span 40.98 ft and wing area 276.3 ft² (AR 6.08). The usual figure is 314 ft², not confirmed here from a primary page. Pick one reference area and fit CD0 on it.
- **Stalls (A).**
  - NACA F4U-4, 12,100 lb, approximate: about 105 mph glide and about 95 mph power on.
  - Navy F4U-4 80765: 89.4 kt CAS power off and 84.5 kt power on, at 12,500 lb.
  - Navy F4U-1: 101.5 / 97.5 mph at 11,194 lb.
- **Speed and climb (A).**
  - F4U-1 17930: 350 mph military / 365 mph WEP at SL; climb 3,040 / 3,210 ft/min.
  - F4U-4: 303 kt at SL and 3,760 ft/min (military).
- **Turn.** Relative only (TAIC 17).
- **Not found.** g limit and dive limit.

Files: `refs/f4u/`.

**Completeness: 6/10.**

### Republic P-47D Thunderbolt

- **Geometry and weight.** Wing area 300 ft² (NACA RM L8A06). The span was not confirmed from a primary source.
- **Speed and climb vs altitude (A), at 13,234 lb, 56 in Hg with water injection, ≈2,210-2,335 bhp.** AAF Eng-47-1652-A:
  - top speed 333 mph at SL, 435 mph at 31,000 ft;
  - climb 2,870 ft/min at SL.
- **Propeller efficiency, measured in flight:** 0.78-0.80 at J 1.19-1.27 (Hamilton Standard 6507A-2), from NACA MR L5G09a. **This is the only measured propeller efficiency in this survey.** It is a useful prior for every other fit.
- **Stall.** The stall speeds exist only on the time histories in NACA RM L8A06, which are unreadable.
- **AAF 1944 table:** 990 ft radius at 4 g, 10,000 ft, 13,000 lb (calculated).
- **Not found.** CLmax, e, CD0, g limit, measured turn.

Files: `refs/p47d/`.

**Completeness: 6/10.**

### La-7 and Yak-3

- **Not built.** No NII VVS test reports were found online in usable form, and wwiiaircraftperformance.org has no Soviet section.
- **La-7 secondary figures** (airpages.ru, citing Shavrov and Kosminkov; grade D): 3,310 kg; 579/613 km/h at SL; a 19 s turn at an unstated altitude.
- **Use.** Enough for a sanity check, not for a model.

## Turn-performance references across the set

| Figure | Aircraft | Type | Source | Grade |
|---|---|---|---|---|
| 360° in 20.0-22.8 s at 1,000 m | Bf 109 G-2/G-4, 3,023-3,235 kg, 1.30 ata | sustained, measured; speed not stated | NII VVS 1944 | A |
| 180° in ≈8-9 s from 174 mph IAS, peak 3.5-4 g | Spitfire VA, 6,184 lb | instantaneous, time history | NACA WR L-334 Fig. 13 | A |
| CL ≈1.4 in wind-up turns | F6F-3 | accelerated CLmax | NACA MR L5B13b | A |
| Radius 883 ft at 4 g, 10,000 ft | P-51B, 9,000 lb (P-47 990 ft at 13,000 lb; P-38J 838 ft) | calculated at the estimated maximum CL | AAF Materiel Command, 26 Jul 1944 | B/C |
| Zeke 52 gains 1 turn in 3.5 on the F6F-5 / F4U-1D; 1 in 8 on the FM-2 | A6M5, F6F-5, F4U-1D, FM-2 | relative, sustained, 10,000 ft | TAIC 17 | A (relative) |
| Zeke 52 gains on the P-51D within one 360 | A6M5, P-51D | relative | TAIC 38 | A (relative) |
| XIV = IX; XIV out-turns the Bf 109 G and turns inside the Fw 190 A | Spitfire, Bf 109, Fw 190 | relative | AFDU | A (relative) |
| P-51D ≈15°/s at 150 mph IAS; E-M charts for the P-51D, Bf 109 G-6 and P-38L | | simulator (Warbirds v1.11) | Hong, combatsim.com 1998 (`refs/comparative/combatsim_hong1998_*.jpg`) | E |

Two checks are possible for the turn model:
- **Absolute:** the Bf 109 G's 20-23 s 360° at 1,000 m.
- **Relative orderings:**
  - Zeke 52 > FM-2 > F6F-5 ≈ F4U-1D at 10,000 ft, with the turn-time ratios about 1.14 and 1.4 against the Zeke (derived from "one in 8" and "one in 3.5");
  - Spitfire IX/XIV > Bf 109 G;
  - Spitfire > Fw 190 A;
  - F6F/F4U > Fw 190 A.

## Ranking

| Rank | Aircraft | Score | Why |
|---|---|---|---|
| 1 | P-51D | 8 | Complete A-grade speed, climb and power-required set at a stated weight and rating. The only drag-fit chart. CLmax good to ±8%. No turn or g limit |
| 2 | Fw 190 A-8 | 7.5 | Manufacturer drag polar gives CD0 and e directly. Strong power and speed data. CLmax approximate; no turn |
| 3 | F6F-3 (extra) | 7 | NACA-measured CLmax in three forms. Strong Navy speed and climb. CD0 must be fitted |
| 3 | Bf 109 G-6 | 7 | Only primary absolute turn times; Hoerner drag analysis; good speed, climb and power at a clear rating. CLmax weak |
| 3 | Fw 190 D-9 | 7 | Same drag table; power with exhaust thrust; Rechlin speed and climb. No stall or turn |
| 6 | Spitfire LF IX | 6.5 | Many A-grade airframes at +18 and +25; NACA CLmax and turn trace from the VA. CD0 fitted |
| 7 | Spitfire XIV | 6 | As the IX, without a same-wing instantaneous-turn trace of its own |
| 7 | F4U-1/-4, P-47D (extras) | 6 | Good speed and climb. The P-47D has the only measured propeller efficiency |
| 9 | A6M2 / A6M5 | 5 | Speeds and climbs degraded; geometry partly D. The best relative turn data |
| 10 | Bf 109 K-4 | 4 | Calculations only at its headline boost |

Best reference charts:
1. `refs/p51d/P-51D_15342_Level.jpg`, `P-51D_15342_Climb-Fig5.jpg` and `P-51D_15342_Drag_Curves.jpg` (AAF 44-15342): the model's own test, with a power-required chart for the drag fit.
2. `refs/fw190/fw190_drag_data.jpg` (Focke-Wulf, Dec 1944): CD0 and e for the A-8 and D-9 in two drag states.
3. `refs/f6f/NACA_MR_L5B13b_F6F-3_flying_qualities_III_stalling_19930092602.pdf`: measured CLmax, power off, power on and accelerated.
4. `refs/bf109/109G_NIIVVS_1944.pdf` p.22-23: the only primary 360° turn times.
5. `refs/spitfire/NACA-WR-L-353_Spitfire_VA_stalling.pdf` and `NACA-WR-L-334_Spitfire_VA_flying_qualities.pdf`: CLmax and an instantaneous-turn time history.
6. `refs/a6m/ptr-1111.pdf` (TAIC 17): relative sustained-turn ratios across four types.
7. `refs/p47d/NACA_P-47D_HS6507A-2_propeller_climb_highspeed_19930093653.pdf`: a measured propeller efficiency (0.78-0.80) to use as a prior for the fits.
8. `refs/bf109/Hoerner-Me_109.pdf`: a full drag breakdown of a WWII fighter.

Gaps common to all:
- No period energy-manoeuvrability diagram exists in what was found.
- No primary g limit was found for any type.
- The stall angle of attack was not found for any type.
- Measured CD0 exists only for the Fw 190 (manufacturer) and the propeller-off P-51B.
- For the other types, CD0 and propeller efficiency can only be fitted together, as was done for the P-51D. Their split is not identifiable from top speed and climb alone without a propeller-efficiency prior (the P-47D's 0.78-0.80 is the one measured value).
