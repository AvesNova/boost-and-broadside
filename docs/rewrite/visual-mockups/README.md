# Visual mock-ups

Status: **brainstorm material, nothing decided.** Flat vector mock-ups made during the
2026-10-27/28 gameplay and visuals brainstorm (see `../gameplay-brainstorm.md`). Every image
is generated from the source next to it.

## Files

| Image | Source | What it shows |
|---|---|---|
| `battlefield.png` | `battlefield.py` | First layout: stadium arena, bases, turrets, fields, three roles, fog (blue view). |
| `ships.png` … `ships6.png` | `ships.py` … `ships6.py` | Six rounds of 16 ship silhouettes, each at 24 px and 12 px in both team colours; from set 3 on, also 12 px at other headings. |
| `fields.png` | `fields.html` | Field and fog options A–J: opaque looks, lifted looks (ally inside), fog treatments, sight shadows. |
| `test.png` | `test.html` + `scene.js` | Visibility test scene: all 16 cases of the `perception.py` rules, labelled, with hidden enemies outlined. |
| `test-topo.png` | `test-topo.html` + `scene.js` | The visibility test scene drawn with the topographic contours (linear bands, bold core edge, high-index arena edge). |
| `test-lemon.png` | `test-lemon.html` + `scene.js` | The topo test scene with lemon and ice fields (both ways round), green / magenta outlined ships, shield / energy rings and trails. |
| `test-purple.png` | `test-purple.html` + `scene.js` | The topo test scene with purple and ice fields (both ways round), green / vermilion ships, fog at 60%. |
| `colourblind-purple.png` | `colourblind.py` (on the `test-purple` scene) | The purple / ice, green / vermilion scene simulated for six colour-vision deficiencies plus achromatopsia (Machado et al. 2009), with palette swatches and CIE76 ΔE between key colours. |
| `colourblind-greenpink.png` | `greenpink.html` + `greenpink.py` | Three green / pink team pairs on the lemon / ice scene under five colour-vision deficiencies, with ally-vs-enemy ΔE. |
| `greenpink2.png` | `greenpink2.html` + `scene.js` | Twelve green + pink variations: six as ship colours on light fields, six as the two field colours with other ship pairs. |
| `mintpink.png` | `mintpink.html` + `scene.js` | Twelve ship colour pairs on mint thin air and pink thick air, fog at 60%. |
| `fastslow.png` | `fastslow.html` + `scene.js` | Twelve new themes where fast air is light and saturated and slow air dark and desaturated, with ship pairs that double as UI colours (UI strip per panel). |
| `fogdark.png` | `fogdark.html` + `scene.js` | The lemon / ice scene with the fog darkened to six brightness levels (90% to 40%), saturation kept. |
| `thincolours.png` | `thincolours.html` + `scene.js` | Six thin-air colours (lavender, purple, periwinkle, lemon, peach, mint) with ice thick air, fog at 60%. |
| `styles.png` | `styles.html` + `scene.js` + `styles.js` | The same test scene in nine visual styles, as a player would see it. |
| `index.png` | `index.html` + `scene.js` | Six ways to show each field's refractive index. |
| `patterns.png` | `patterns.html` | Sixteen pattern families across the four index levels, under fog and at battle zoom. |
| `contours.png` | `contours.html` + `scene.js` | Contour-ring variants, and true index contours on the test scene. |
| `shields.png` | `shields.html` | Twelve shield displays (ring at the hitbox) and eight energy displays, swept full to empty, with regeneration or boosting and battle zoom. |
| `shields2.png` | `shields2.html` | Centred arcs: shield with 1, 2 and 3 centres, energy as one nose-centred arc (inside, outside, tinted, long), plus a shield × energy matrix. |
| `shields3.png` | `shields3.html` | Twelve more centred-arc combinations: halves (shield nose, energy tail), shares, interleaved 3 + 3 and 6 + 6, 4, 6 and 12 centres. |
| `exposed.png` | `exposed.html` | Twelve ways to show a ship with no shield left (next hit kills), on ally and enemy, at battle zoom and in a crowd. Base display: shield 240° on the nose, energy 120° on the tail. |
| `trails.png` | `trails.html` | Twelve flight-trail styles on the same 2.5 s manoeuvre, each with a battle-zoom inset. |
| `trails2.png` | `trails2.html` | Trails round 2: thin taper, inner-wingtip vortex, Mach diamonds, line-to-dots, ripples; and nine ally / enemy colour pairs over the topo fields. |
| `palettes.png` | `palettes.html` | Sixteen colour palettes (teams, shield, energy, fields, background) on a small scene, plus eight trail-colour variants. |
| `mapcolours.png` | `mapcolours.html` | Twenty-four background and field colour pairs (light, mono, earthy, dark) on the small scene, teams fixed, enemies outlined. |
| `themes.png` | `themes.html` | Twenty-four whole-look themes: ship colours, outlines, energy, trail tint, map and fields chosen together, on the small scene. |
| `whites.png` | `whites.html` | Twenty-four whole-look themes on near-white skies with green / magenta or blue / vermilion teams. |
| `whites2.png` | `whites2.html` | Twenty-four themes with every ship outlined and both field types in a real colour, green / magenta or blue / vermilion teams on near-white skies. |
| `thinair.png` | `thinair.html` | Ice, lemon, mint and peach thin air, each paired with darker thick-air colours, for both team pairs; outlined ships, white sky. |
| `pairs4.png` | `pairs4.html` | Every ordered pairing of ice, lemon, mint and peach as thin and thick air, for both team pairs. |
| `map-vis.png` | `map-vis.html` + `scene.js` | The larger example map with all 16 visibility cases placed on it; 20 m of edge band per level (40 m at the extremes, as in the current code); mint / pink air, navy / amber ships, fog at 60%. |
| `colour-tool.html` | `colour-tool.src.html` + `scene.js` + `styles.js`, bundled by `build-tool.py` | **Interactive.** Self-contained colour playground on the example map with every visibility case: ship, field, map and fog colours and sliders, presets, copyable settings. |
| `map-contours.png` | `map-contours.html` | The first topographic map of the refractive index (wider, 30 m per level bands). |

## Rules the scenes follow

- **Visibility** follows `src/boost_and_broadside/env/perception.py`, with capsule fields: team-shared
  vision within range; a field's core (outer radius minus half the band) blocks any sight line that
  comes within its radius, unless both ends are inside that same core; a successful shot and a spawn
  reveal the ship; own bullets always visible, enemy bullets only where an ally could see them.
- **Index composition** follows `compose_refractive_index` (union rule in log space). Levels are
  exponents of √2 (n = 0.5, 0.71, 1.41, 2; ambient 1); speed in a field scales as 1/n.
- **`map-contours.html`** uses a linear index ramp with band width proportional to level (30 m per
  level), so contours are evenly spaced (10 m, one per third of a level). Contours sit at whole thirds
  of a level, so each field's innermost contour lands exactly on its core edge, drawn bold. The arena
  edge is a high-index field with the same slope, then a damage zone. Note: the physics today uses a
  smootherstep band, which this does not match; a linear ramp has kinks at the band ends.

## Current state

See `../gameplay-brainstorm.md` for what the author has chosen so far and what is still open. In
short: near-white sky; topographic field contours with the innermost line on the core edge; fog
darkened to 60% with saturation kept; every ship outlined; shield 240° on the nose and energy 120° on
the tail in one ring; fading trail with a faint inner-wingtip vortex; field and ship colours still
being chosen (mint / pink air is the current favourite).

## Regenerating

From this directory: `python3 battlefield.py`, `python3 ships.py` (and `ships2.py` … `ships6.py`)
write the SVGs. The HTML pages render in any browser; `node shotfull.js "$PWD/map-contours.html"
"$PWD/map-contours.png" 1900` screenshots one with the pre-installed Playwright.
