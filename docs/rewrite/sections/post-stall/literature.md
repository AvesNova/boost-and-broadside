# Post-stall regime: literature check

Time-boxed web check (2026-11-26). Tags: **[S]** = from a source I could read, **[S?]** = secondhand (forum quoting a report; unverified), **[E]** = my estimate or general-knowledge value, not verified in this check.
Gaps: the primary NACA PDFs (L-334 Spitfire VA, WR L-566 XP-51, TN 3361 NACA 0012 0-180 deg) were scanned images or blocked (403), so no per-degree tables were extracted. Nothing found from DCS/IL-2/War Thunder developers about over-rotation limits.

## Numbers

| # | Quantity | Value | Tag | Source |
|---|---|---|---|---|
| 1 | Spitfire Mk.VB stall AoA (IL-2 museum spec) | 18.8 deg clean, 16.0 deg landing config | [S] | https://il2sturmovik.com/museum/fighters/spitfire-mkvb/ |
| 1 | Fw 190 A-3 / P-40 / Yak-1 stall AoA | 15.5 / 14 / ~18 deg | [S?] | https://forum.il2sturmovik.com/topic/22155-naca-data-indicates-that-the-bos-yak-clmax-is-too-high/page/2 (forum, uncited) |
| 1 | Spitfire VA behaviour past CLmax | Buffet before stall is a good warning; little stick travel to reach stall onset; large aft stick gives no violent motion; nose drops at onset in slow or steep turns; well behaved even beyond maximum lift | [S] | NACA WR L-334 (Phillips, Vensel 1942), https://ntrs.nasa.gov/citations/19930092582 (read via search summary only) |
| 1 | Transient AoA reached in snap pulls (18/20/25/30 deg) | No measured WWII figure found | gap | - |
| 1 | Transient AoA, my estimate | Dynamic overshoot: wing holds attached flow above static stall for a fraction of a second at high pitch rate; nose reaches roughly stall+5 to +10 deg before lift collapses | [E] | - |
| 2 | CLmax by aircraft (Mach 0.30) | Fw 190 / F4U (23016-09) ~1.40; P-51 (66-2-215) ~1.17; P-51 flight test 1.48 at M 0.175, 1.1-1.32 at M 0.25-0.3; Fw 190 full scale tunnel 1.3 clean | [S?] | NACA TN 1044 as quoted at https://forum.il2sturmovik.com/topic/22155-naca-data-indicates-that-the-bos-yak-clmax-is-too-high/page/2 |
| 2 | Post-stall lift, flat plate / airfoils | After stall (15-20 deg) CL dips, rises again between 20 and 25 deg to a peak of ~1.2 at 45 deg, then falls roughly linearly to 0 at 90 deg. Airfoil shape matters little once separated | [S] | https://aerospaceweb.org/question/airfoils/q0150b.shtml (Sandia 0-180 deg data) |
| 2 | Same, interpretation | Past stall the CL does not collapse toward zero: it plateaus near 0.7-1.0x of the pre-stall peak up to ~45 deg. For thick cambered sections (23018) the drop right after stall is gradual (trailing-edge separation); for laminar P-51 sections it is sharper (leading-edge) | [E] | - |
| 3 | Post-stall drag | CD ~0 at 0 deg, bell-shaped, peaks ~2.0 at 90 deg (flat plate 1.98) | [S] | aerospaceweb link above; TU Delft table https://repository.tudelft.nl/file/File_a008cc18-2fee-46ee-a8a2-4c9e0e08f962 |
| 3 | Flat-plate model | CD ~ 2 sin^2(alpha) gives 0.19 at 15 deg, 0.35 at 20, 0.54 at 25, 0.8 at 30, 1.0 at 45; CL ~ sin(2 alpha) overestimates (1.0 at 45) vs measured ~1.2 | [E] (formula), CL peak [S] | - |
| 3 | Real wing CD at stall | Clean fighter CD0 ~0.02-0.03 plus induced ~0.1-0.15 at CLmax 1.3-1.4, so CD at stall ~0.15-0.2 | [E] | - |
| 4 | Pitch rate in a snap | No WWII measured figure found. Steady-turn pitch rate at corner (about 5-6 g, 150-170 kt TAS) ~ 20-25 deg/s; elevator-limited snap peaks higher, ~40-60 deg/s for a heavy-ish fighter | [E] | - |
| 5 | IL-2 GB | Spitfire: buffet early, pulling on without easing forward "may stall and spin even at high speed"; stall made more severe in a 2017 update | [S] | https://il2sturmovik.com/news/317/dnevnik-razrabotchika-161/ , https://il2sturmovik.com/news/319/dev-blog-162/ (summaries only) |
| 5 | DCS | P-51D / Spitfire: only forum anecdotes (P-51 considered too stall-prone; Spitfire criticised for not spinning); no developer note on over-rotation limit | [S?] | https://forum.dcs.world/topic/101221-p-51d-dogfight-stalls-and-spins |
| 5 | War Thunder | Not found | gap | - |

## Recommendation (all values [E], anchored on the sourced flat-plate shape and Spitfire qualitative behaviour)

1. Deepest commanded AoA: 1.5x stall angle (about 22-23 deg for a 15 deg stall), hard cap 2x (30 deg). Beyond 2x the sourced data show lift near flat-plate level anyway, so nothing is gained.
2. Lift fraction at that angle: about 0.8 of CLmax at 1.5x (gradual loss, Spitfire-like), falling toward ~0.65 at 2x. Use smooth taper from 1.0 at stall, not a step.
3. Drag at that angle: about 3x the drag at stall at 1.5x stall angle (flat-plate sin^2 scaling plus separation), about 5x at 2x.
4. Nose swing time: 0.25-0.4 s to reach the deepest angle (about 40-60 deg/s), hold 0.3-0.5 s, release and recover within 0.5-1 s total (matching the brief); recovery pitch-down is natural because the AoA decays once the stick is released.
5. Validate by feel: expect a speed bleed of roughly 10-20 percent per snap, and tune to that rather than to any single table, since no measured WWII transient-AoA data was located. Best next primary sources: NACA WR L-566 (XP-51), TN 3361 tables, NACA TN 1044.
