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
| `map-contours.png` | `map-contours.html` | **Current favourite.** Example map drawn as a topographic map of the refractive index. |

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

## Leanings from the brainstorm (not decided)

- Shield and energy: one ring at the hitbox, shield 240° centred on the nose, energy 120° centred on
  the tail, each shrinking toward its centre (`shields3.png`, row 3).
- Trail: fading line plus a faint vortex on the inside wingtip during hard turns (`trails2.png`, 3).
- Ships: the Komet / Cranked-arrow family (sharp nose, notched or swept back) reads best at small size.
- Fields: topographic contours of the true index (`map-contours.png`); thin air teal, thick air violet.
- Fog: terrain always known (fields, bases, turrets drawn), units hidden; sight shadows from field cores.
- Open: whether field cores may overlap (crossing core edges), and a faint "zero" contour at the
  outer edge of each band.

## Regenerating

From this directory: `python3 battlefield.py`, `python3 ships.py` (and `ships2.py` … `ships6.py`)
write the SVGs. The HTML pages render in any browser; `node shotfull.js "$PWD/map-contours.html"
"$PWD/map-contours.png" 1900` screenshots one with the pre-installed Playwright.
