# Section 10 (play shell), milestone 1: notes

Branch `section/play`. The spec (`play-spec.md`, with the theme editor added to section 6
at the author's request) is implemented in `bnb.play` and `bnb.controllers.human`. The
whole suite passes (`uv run --no-sync pytest`); the GL tests carry the `gpu` marker and
skip without an OpenGL 3.3 (EGL) context.

## Design

### Layout

| Module | Job |
|---|---|
| `play/snapshot.py` | `Snapshot` (ships, bullets, fields, zones, match) as numpy; `snapshot(rules, types, state, observation)`; `core_radius` |
| `play/interpolate.py` | `interpolate(previous, current, alpha)`, `angle_lerp` |
| `play/loop.py` | `MatchLoop`: the realtime sim, chasers plus one pilot, the accumulator, pause, step, time scale |
| `play/index_map.py` | `IndexBuilder`, `build`: ln n on a grid from `medium`, tile by tile |
| `play/trails.py` | `Trails`: the per-ship sample store, in game time |
| `play/camera.py` | `Camera`: centre, metres per pixel, fit, pan, zoom, follow; the shaders' camera uniforms |
| `play/themes.py`, `play/themes/*.toml` | `Colour` (8-bit sRGB plus alpha, hex), `Theme`, load, validate, save |
| `play/theme_editor.py` | the ImGui theme editor |
| `play/app.py` | `Options` (tyro), `Session` (one match, independent of any window), the offscreen runner |
| `play/window.py` | the GLFW window: input to the session and to ImGui |
| `play/benchmark.py` | `python -m bnb.play.benchmark` |
| `play/png.py` | a PNG writer (no image library) |
| `play/render/` | `Renderer` and one module per pass, each with its `.glsl` files; `programs.py` loads them (`#include "common.glsl"`) |
| `controllers/human.py` | `INPUTS`, `human_action(inputs) -> Action` |

### The loop

One thread, a fixed timestep, as proposed. `MatchLoop.advance(elapsed)` adds
`elapsed * time_scale` to the accumulator, steps the sim while it holds a tick, and the
frame draws at `alpha = accumulator / tick`. Each tick: the chaser for both teams (one call
per team view), the pilot's row overwritten by `human_action`, `Realtime.step`, then a
snapshot. A backlog beyond `CATCH_UP_TICKS` (4) per unit of time scale is dropped, so a
stall does not snowball. Paused, `single_step` runs one tick and draws it at alpha 1. The
time scales are 1, 2, 4, 8 (`TIME_SCALES`).

`Session.update` then: builds index tiles of a new map within `INDEX_BUDGET` (4 ms), takes
the interpolated frame, records the trails and moves a following camera. `Session.draw`
renders and then runs one ImGui frame (HUD, theme editor). No threads were needed: the
measurements below fit the frame budget on one thread.

The process sets one torch thread (`--threads`, default 1). Measured on this machine under
the other agents' load: `sighted` on a 128² grid took 24 ms with 1 thread and 1842 ms with
the default 16; a B = 1 tick gains nothing from more threads.

### The snapshot

`snapshot` copies everything it reads (`.numpy().copy()`), so it is pure. Ships carry
position, heading, flight path (heading minus angle of attack), angle of attack, nose rate,
shield and energy as fractions of the type's maxima, alive, team, type, fired, spawned,
respawn wait (s), hit radius and vision range (the game's `vision_range`), and
`visible (2, N)` from each team's observation, put back in sim order with `view_order`.
Bullets are the (N K) slots with active, moves left, owner and team. Fields add
`core_radius`. Zones carry role, meter (progress), owner (the team a spawn or defense zone
belongs to, -1 neutral) and attacker (whose capture the meter counts). The match part:
arena radius, mode, tick, clock, episode length, score and score target, front and front
threshold, `reset` (tick 0), and two game constants the drawing scales by
(`log_index_per_level`, `stall_angle`), so rendering imports nothing from the game.

### Interpolation

Linear positions, shield and energy; headings and flight paths by the shorter way; a ship
that spawned (or was dead on the earlier tick, or the episode reset) snaps. A bullet slot
is fresh when it was idle or its moves left did not go down; a fresh bullet draws from its
owner's earlier position, which is where `fire` launches it.

### The passes and how each look element is drawn

Everything draws into a 4x multisampled renderbuffer, resolved onto the target (the
window's framebuffer or an offscreen texture); ImGui draws on the target after the resolve.
Uniforms per frame: `camera` (centre, clip units per metre), `pixel` (metres per pixel),
`viewport`.

1. **Sky** (`sky.frag.glsl`, one full-screen triangle): the sky inside the arena disc;
   beyond it the outside colour hatched in the boundary colour (the thick air's, per the
   brainstorm) at 35% opacity, 3 px lines every 12 px, anchored to the world so they pan
   with the map; a 1.5 px edge line in the thick-air core colour.
2. **Fields** (`fields.*.glsl`): one instanced quad per field over its outer disc
   (radius plus half the band). A fragment inside several outer discs is drawn only by the
   first of them (the fragment loops over the earlier fields' discs), so overlaps draw once.
   It samples the index texture with a 16-tap Catmull-Rom filter (exact at texel centres),
   divides by the contour step (a third of a level) to `steps`, and:
   - **fill:** terraces `floor(|steps| + 0.02)`, mixing the sky toward the thin or thick
     air colour by `terrace / 6 * fill_strength` (the colour tool's stepped fill);
   - **contours:** a 1 px line where `|steps| + 0.02` crosses a whole number (the 0.02 keeps
     a line off a core's plateau), antialiased with `fwidth`, faded where lines come closer
     than about 3 px;
   - **the core edge:** drawn separately through the arcs program as a 2.4 px circle at
     `core_radius`, in the thin or thick core colour.
3. **Zones** (arcs): a disc tinted with the owner's colour (10% on spawn zones, 20% on
   defense zones; neutral grey on the neutral zone), a 2 px outline, and on each defense
   zone the meter: a 5 px arc 7 px outside the disc, in the attacker's colour, growing
   clockwise from the top over a faint full track.
4. **Trails** (`trails.vert.glsl`): no vertex buffers. The trail store (positions, age,
   heading; the turn fraction; per ship team, shown, ring radius, wingtip) goes up as three
   small float textures, and vertex `v` is a corner of segment `v / 6`. Width tapers 2.5 to
   0.5 px and alpha fades with age over `TRAIL_SECONDS` (1.5 s, proposal); samples every
   1/60 s of game time. The vortex is a second draw of the same store, offset to the
   inside wingtip (the side the angle of attack turns toward) and shown where
   |angle of attack| / stall angle exceeds the theme's `vortex_threshold`, ramping up over
   a quarter of the threshold.
5. **Bullets** (instanced quads, capsule distance in the fragment shader): 10 px by 2.5 px
   streaks behind the bullet along its direction, in the shooter's team colour. In a team
   view an enemy bullet samples the fog mask and is discarded where the team cannot see.
6. **Ships** (instanced meshes): the colour tool's Komet for fighters, a broader Komet
   (wider shoulders and wings) for heavies, in units of the ring radius. Each mesh holds
   the outline (the polygon grown along capped miters by `max(1.3 px, 3.5% of the ring
   radius)`) and then the fill, so every ship is outlined and overlaps paint per ship. The
   ring radius is the hit radius, or 14 px when that is smaller on screen; the silhouette
   and the ring scale together.
7. **Rings** (arcs): per ship a faint track (15%) and a solid arc for the shield, 240
   degrees centred on the nose, and the same for the energy, 120 degrees on the tail; each
   solid arc's half span is the full one times the fraction, so it shrinks toward its
   centre. Round ends, 0.1 rad gaps (the colour tool's). An empty shield or energy draws no
   solid arc.
8. **Fog** (team views): first the mask, then a full-screen quad with blending
   `DST_COLOR, ZERO` that multiplies the scene by `mix(fog, white, seen)`; `seen` is a
   four-tap tent filter of the half-resolution mask. The fog colour #999999 is the 60%
   brightness of the brainstorm, saturation kept.
9. **HUD** (ImGui foreground draw list, the vector default font): top middle the score (in
   team colours, with the target) or the front, then the clock; top left the view, theme
   and time scale; bottom middle the flown ship and its respawn timer; F1 the key list.

The circles, arcs, zone discs and rings all go through one program (`arcs.*.glsl`, an
instance table of centre, radius, width in metres and in pixels, fill, line and arc).

### The fog mask

A fragment shader (`sight.frag.glsl`) at `MASK_SCALE` (0.5) of the target's resolution,
each frame, in screen space, with the frame's interpolated ship positions. For each living
observer of the team in range (`d² <= range²`), a core blocks the line when exactly one
end is strictly inside it, or neither is and the perpendicular test of `sighted` holds
(`across² < r² |d|²`, `0 < along < |d|²`). This is `sighted`'s rule, written per core
instead of with bitmasks. Observers and cores are small float textures.

### Themes

`Theme` has every colour the passes use plus `fill_strength` and `vortex_threshold`;
pairs (team, shield, trail) hold team 0's then team 1's. Files list every value, and
`parse` names any missing, unknown or malformed one. Shipped: `mint-pink` (default),
`lemon-ice`, `purple-ice`, from the colour tool's presets; contour and core colours are
the tool's ink (the air colour mixed 45% toward #1C2030, contours at 75% alpha), trails the
team colour mixed 30% toward the sky at 70% alpha.

The editor: one `color_edit3` per colour (`color_edit4` with an alpha bar for the
contours, trails and vortex) with `display_hex`, so the inline field is the hex code; the
swatch opens ImGui's picker. A test clicks the sky field, selects it, types `#204060` and
checks the theme. "Save as theme" writes `<name>.toml` into the themes directory and
refreshes the T cycle; "Revert" restores the file's values.

## Decisions and deviations

1. **Library and wheel check.** moderngl 5.13.0, imgui-bundle 1.93.0 and glfw 2.11.0 all
   publish cp314 (or py3-none) wheels for Linux x86_64, macOS arm64 and Windows, and
   install next to torch 2.15.0 with numpy 2.5 (`uv add`, 5 packages, no conflicts). No
   fallback to wgpu-py was needed. An EGL standalone context gives OpenGL 3.3 core on the
   RTX 4070 Laptop (driver 595.99.02).
2. **ImGui through moderngl.** imgui-bundle's Python OpenGL backends need PyOpenGL. Rather
   than a second GL wrapper on one context, `render/imgui_renderer.py` (78 lines) draws
   ImGui's draw data with moderngl, including the 1.92 texture protocol (the atlas grows as
   text needs glyphs). The window's input to ImGui is in `window.py` (mouse, wheel, the keys
   text fields need, characters). Each `Session` owns its ImGui context and makes it
   current (ImGui's `CreateContext` does not when another exists).
3. **`snapshot` takes the ship card:** `snapshot(rules, types, state, observation)`. The
   spec's signature has no card, but the shield and energy fractions and the hit radius
   need its per-type values, and milestone 3 edits the card live.
4. **The fog mask is a shader** (allowed with a test). Measured with the game's `sighted`
   on the CPU, eager, one thread: a 128² grid at 5v5 took 24 ms, 256² 64 ms; at 50v50 with
   40 fields 128² took 456 ms and 256² 2.9 s. That does not fit a frame even every few
   ticks. The shader computes it each frame at half resolution (cost below), and
   `test_fog_mask_agrees_with_sighted` compares it with `sighted` at every texel away from
   the mask's edges, at 5v5 and at 12v12 with 30 fields.
5. **Index texture resolution: 2 m per texel at map scale 1, bicubic,** rather than about
   1 m. The texel grows with the map scale (6.3 m on the 8200 m maps), so every map's
   texture is the same size, 2624² R32F (27.5 MB). See the measurements: the bicubic
   filter keeps the contours within a small fraction of a metre at 2 m. Tiles that touch no
   field's outer disc are skipped (exactly 0 there), and each tile passes `medium` only the
   fields that touch it, padded with a field far inside out; the sums and the product are
   unchanged, and a test compares the grid with `medium` at sampled texels.
6. **The core radius is computed in `bnb.play.snapshot.core_radius`,** the same expression
   `visibility` builds inline. `test_core_radius_is_where_the_sight_test_blocks` holds them
   together (an enemy just inside the radius is hidden, just outside is seen). Request 1.
7. **Map changes** are detected by comparing the fields with the built map, not only by
   the snapshot's `reset` flag: at x8 a frame runs several ticks and can skip the reset
   tick. The `reset` flag still stops interpolation into a new episode.
8. **`--fog` is opt-in,** literally. Without it the chasers see everything, and a team's
   view darkens what its ships cannot see but draws every ship (the observation shows
   them all). Question 1.
9. **Options beyond the spec:** the sizes (`--ships-per-team`, `--fields`,
   `--arena-radius`, "or the sizes"), `--fly/--no-fly`, `--width`, `--height`,
   `--vsync/--no-vsync`, `--compile/--no-compile` (a quick start), `--threads`. Views are
   named `spectator`, `team0`, `team1`.
10. **Keys beyond the spec's flight keys:** V view, T theme, E editor, F follow, P pause,
    N step, `=`/`-` (and the keypad's) time scale, Tab next pilot, Home fit, F1 keys, Esc
    quit. Mouse drag pans, the wheel zooms about the cursor (0.05 m per pixel at the
    closest, three times the fitted zoom at the farthest).
11. **Theme values beyond the spec's list:** outside colour, boundary hatching, neutral
    zone, HUD text, contours and core lines per air type, `fill_strength`,
    `vortex_threshold`.
12. **The heavy silhouette** is mine, a broader member of the Komet family (the brainstorm
    names the family, not the per-type shapes). The fighter is the colour tool's Komet.
13. **Saved themes go to the package's `themes/` directory** (`src/bnb/play/themes/` in a
    checkout). Question 2.
14. **Docs outside the listed files:** `mkdocs.yml` (nav entries for the guide and the
    reference; the strict build fails on a page left out of the nav) and
    `docs/hooks/generated.py` (the blocks `play-options`, `play-keys`, `play-themes`, and
    the status `milestone 1 built`, which the hook did not accept). The screenshots are in
    `docs/images/play/`.
15. **The benchmark** is `bnb/play/benchmark.py` (`bench/` is outside my boundaries); it
    imports `bench.lock` inside `main`, so it runs from the repository root.

Items marked (proposal) in the spec, implemented as written: one thread with a fixed
timestep; trails of the last 1.5 s; the lemon/ice and purple/ice themes; the keys W, S,
A, D, Shift, Ctrl and Space.

## Measurements

**Provisional: taken under load.** Several section agents were running at the same time
(load average 17 to 22 during the run), so these numbers are contended; the coordinator
takes the final ones on a quiet machine. A second complete run after merging v2
(load rising from 3 to 14 during it) confirmed the script end to end.

All on this machine: RTX 4070 Laptop GPU (8 GB), 16 CPU threads, Python 3.14.2, torch 2.15.0, one
torch thread. `python -m bnb.play.benchmark --seconds 10`, benchmark lock held, at
1920 x 1080, offscreen (EGL), 4x MSAA, no frame cap, `context.finish()` at the end of
each frame (what a swap without vsync waits for). The sim is compiled, every game has
fog of war, nobody flies (all chasers), seed 0. Code: commit e6452fb9 plus the index
batch size of ee89371e (`TILES_PER_CALL` 4), which the map-change columns use.
50v50 is 50 ships per team, 40 fields, arena radius 8222 m (team deathmatch).

### Frame time (ms) and the sim's rate

The frame time is display time per frame, sim ticks included (a frame runs zero, one or
occasionally two ticks). Ticks per second: ticks run over the wall time of the timed run;
real time is 30. The map change rebuilds the index texture a few tiles per frame
(`INDEX_BUDGET`, 4 ms, with `TILES_PER_CALL` 4); its columns are the frames it took and
their p99 and max.

| Game | View | Frames | p50 | p90 | p99 | max | Ticks per s | Map change: frames, p99, max |
|---|---|---|---|---|---|---|---|---|
| deathmatch | team0 | 17232 | 0.55 | 0.60 | 2.08 | 6.44 | 29.99 | 8, 6.29, 6.35 |
| deathmatch | spectator | 17668 | 0.54 | 0.62 | 2.12 | 5.87 | 29.99 | 9, 6.66, 6.66 |
| deathmatch_wide | team0 | 12833 | 0.69 | 0.86 | 2.51 | 9.03 | 29.99 | 7, 6.85, 6.95 |
| deathmatch_wide | spectator | 17934 | 0.53 | 0.58 | 2.00 | 5.92 | 29.99 | 7, 6.97, 7.03 |
| frontline | team0 | 14130 | 0.67 | 0.76 | 2.33 | 7.08 | 29.99 | 19, 9.44, 9.55 |
| frontline | spectator | 14749 | 0.61 | 0.72 | 2.38 | 7.25 | 29.99 | 15, 11.73, 11.93 |
| 50v50 | team0 | 6298 | 0.97 | 3.67 | 11.11 | 24.65 | 29.98 | 101, 12.43, 12.56 |
| 50v50 | spectator | 9061 | 0.72 | 2.23 | 6.22 | 10.61 | 29.98 | 73, 8.78, 9.13 |

Every p99 is under 16.7 ms and the sim keeps real time in every cell. The 50v50 team view
has one frame of 24.65 ms in 6298. A first run an hour earlier (the GPU busier, the trail
geometry still built on the CPU) gave a 50v50 team-view p99 of 18.0 ms: the trails' CPU
geometry took 5.7 ms a frame at 100 ships, which moved them into the vertex shader (now
0.2 ms of Python, the texture uploads).

### Per pass: GPU / CPU (ms), mean of 240 frames

GPU from timer queries; CPU is the pass's own Python (table building and uploads). A
frame's sim work is in the next table.

| Game | View | fog mask | sky | fields | zones | trails | bullets | ships | rings | fog | resolve | imgui |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deathmatch | team0 | 0.030 / 0.071 | 0.024 / 0.019 | 0.011 / 0.056 | 0.001 / 0.011 | 0.012 / 0.072 | 0.001 / 0.015 | 0.002 / 0.050 | 0.001 / 0.051 | 0.041 / 0.012 | 0.033 / 0.021 | 0.004 / 0.039 |
| deathmatch | spectator | - | 0.067 / 0.031 | 0.027 / 0.063 | 0.001 / 0.012 | 0.030 / 0.080 | 0.001 / 0.018 | 0.002 / 0.056 | 0.002 / 0.053 | - | 0.080 / 0.023 | 0.010 / 0.043 |
| deathmatch_wide | team0 | 0.039 / 0.063 | 0.039 / 0.017 | 0.015 / 0.050 | 0.001 / 0.010 | 0.016 / 0.065 | 0.001 / 0.014 | 0.002 / 0.045 | 0.001 / 0.046 | 0.069 / 0.011 | 0.052 / 0.024 | 0.006 / 0.035 |
| deathmatch_wide | spectator | - | 0.064 / 0.031 | 0.024 / 0.062 | 0.001 / 0.012 | 0.029 / 0.078 | 0.001 / 0.018 | 0.002 / 0.055 | 0.001 / 0.053 | - | 0.078 / 0.023 | 0.010 / 0.042 |
| frontline | team0 | 0.051 / 0.078 | 0.045 / 0.020 | 0.019 / 0.060 | 0.010 / 0.046 | 0.018 / 0.078 | 0.001 / 0.006 | 0.002 / 0.056 | 0.001 / 0.055 | 0.080 / 0.013 | 0.056 / 0.024 | 0.007 / 0.043 |
| frontline | spectator | - | 0.074 / 0.029 | 0.029 / 0.058 | 0.010 / 0.041 | 0.030 / 0.073 | 0.001 / 0.007 | 0.002 / 0.051 | 0.001 / 0.049 | - | 0.088 / 0.021 | 0.011 / 0.040 |
| 50v50 | team0 | 0.167 / 0.133 | 0.030 / 0.034 | 0.082 / 0.129 | 0.001 / 0.012 | 0.044 / 0.209 | 0.001 / 0.058 | 0.003 / 0.094 | 0.005 / 0.124 | 0.055 / 0.014 | 0.170 / 0.066 | 0.005 / 0.099 |
| 50v50 | spectator | - | 0.025 / 0.050 | 0.066 / 0.155 | 0.001 / 0.013 | 0.060 / 0.266 | 0.001 / 0.114 | 0.003 / 0.146 | 0.006 / 0.102 | - | 0.200 / 0.048 | 0.004 / 0.089 |

### Per stage, CPU (ms), mean over the profiled frames

`controllers` (both teams' chasers and the pilot), `sim` (`Realtime.step`, compiled) and
`snapshot` are per tick; `advance` (all the ticks of a frame, with their stages) and
`interpolate` are per frame.

| Game | View | controllers | sim | snapshot | advance | interpolate |
|---|---|---|---|---|---|---|
| deathmatch | team0 | 0.396 | 0.315 | 0.337 | 0.525 | 0.067 |
| deathmatch | spectator | 0.430 | 0.319 | 0.355 | 0.553 | 0.070 |
| deathmatch_wide | team0 | 0.351 | 0.280 | 0.299 | 0.466 | 0.059 |
| deathmatch_wide | spectator | 0.409 | 0.368 | 0.352 | 0.566 | 0.069 |
| frontline | team0 | 0.432 | 0.400 | 0.372 | 0.604 | 0.075 |
| frontline | spectator | 0.401 | 0.335 | 0.340 | 0.539 | 0.066 |
| 50v50 | team0 | 0.789 | 1.403 | 0.785 | 1.492 | 0.110 |
| 50v50 | spectator | 0.709 | 1.340 | 0.871 | 1.453 | 0.177 |

A tick's whole cost (controllers, sim, snapshot) is about 1.1 ms at 5v5 and 3 ms at
50v50, so a frame that runs a tick stays far inside the budget on one thread.

### The fog mask

960 x 540 (half of 1920 x 1080), recomputed every frame on the GPU: 0.03 to 0.05 ms at 5v5
and 0.17 ms at 50v50 with 40 fields, plus 0.04 to 0.08 ms for the fog pass that applies it.
The CPU alternative, the game's `sighted` on a grid (eager, one thread, before the
decision): 24 ms (128²) and 64 ms (256²) at 5v5; 456 ms (128²) and 2.9 s (256²) at 50v50.

### The index texture

2624 x 2624 R32F (27.5 MB) on every map: 2 m per texel at map scale 1 (the 2600 m arenas),
6.3 m on the 8200 m arenas. Built whole at startup (one thread): 0.21 s for a 5v5 map and
0.56 s for the 50v50 map; a map change spreads the same work over frames (above). The
bicubic sample's error against `medium`, at 200 000 random points in the field bands,
as the shift of a contour (the error in ln n over |grad ln n|, where the gradient is at
least a fifth of its maximum):

| Map | Texel at scale 1 | Actual texel | Build | Max error (contour steps) | Shift p99 | Shift max |
|---|---|---|---|---|---|---|
| deathmatch | 1 m | 1.00 m | 0.61 s | 0.0001 | 0.000 m | 0.002 m |
| deathmatch | **2 m** | 2.00 m | 0.21 s | 0.0010 | 0.003 m | 0.014 m |
| deathmatch | 4 m | 4.00 m | 0.09 s | 0.0081 | 0.037 m | 0.120 m |
| 50v50 | 1 m | 3.16 m | 2.90 s | 0.0052 | 0.014 m | 0.048 m |
| 50v50 | **2 m** | 6.32 m | 0.56 s | 0.0451 | 0.191 m | 0.545 m |
| 50v50 | 4 m | 12.65 m | 0.13 s | 0.3834 | 1.599 m | 3.517 m |

At 2 m a 5v5 contour sits within 1.4 cm of its true place, under a pixel at the closest
zoom (0.05 m per pixel). On the 8200 m maps the band is still 40 m wide while the texel
grows, so the worst contour shift is 0.55 m: 2 pixels at 0.25 m per pixel, 11 at the
closest zoom. A per-band or per-field texture would fix it if it shows; the texture
would no longer be one size.

## Screenshots

In `docs/images/play/` (1600 x 900, offscreen, embedded in the guide and the explanation
page):

- `frontline-spectator.png`, `frontline-team0.png`: a Frontline match, spectator and team
  0's fog (zones, meters, sight shadows behind cores);
- `dogfight.png`: a close-up of two ships, rings, trails and a wingtip vortex;
- `contours.png`: the contours and the bold core edge at 0.25 m per pixel;
- `theme-editor.png`: the theme editor and the key list, lemon and ice;
- `50v50-team0.png`: 50 against 50 with 40 fields, purple and ice, team 0's fog.

## What milestone 2 should build on

- **`Session` is the seam.** It owns the loop, renderer, camera, view, theme and panels,
  independent of the window, and runs offscreen for tests. A developer panel is one more
  `draw` call in the ImGui frame after the HUD, like `ThemeEditor`.
- **Already there:** the team-view toggle (`Session.view`), take control of any ship
  (`MatchLoop.fly(index)`), the speed dial's state (`paused`, `single_step`,
  `time_scale`, `TIME_SCALES`), theme switching and editing.
- **Replays:** `MatchLoop.tick(actions)` is the one place a tick happens with explicit
  actions; recording is a clone of `realtime.state` every k ticks plus the actions, and
  playback feeds `tick`. The renderer only needs snapshots, so a replay player can feed
  it without a live sim.
- **Pickers (mode, map, lineup):** a new `MatchLoop` per choice. `MatchLoop(compile=False)`
  runs now; the decided "uncompiled while the new graph compiles in the background" needs
  `Realtime` to swap in its compiled tick when ready (request 3).
- **What made it harder:** imgui-bundle's Python backends need PyOpenGL (hence our
  renderer and input wiring); only the keys text fields need are mapped to ImGui, so
  panels with keyboard navigation need more of the table; an ImGui window takes its size
  on its second frame, which offscreen tests must allow for.

## The code style draft

v2 now has a code style draft (`docs/contributing/code-style.md`, merged from v2 after
this work). Where the play shell departs from it, by the spec or by necessity:

- **The game state is read outside `bnb.game`,** in `bnb.play.snapshot.snapshot` only, as
  the decisions require ("reads state through a pure `snapshot(state)`").
- **Global seeding in tests:** `Realtime` draws from torch's global generator, so the
  determinism, snapshot and render tests seed it with `torch.manual_seed` (request 2 would
  remove this). The tests that set one torch thread restore the old count.
- **`.numpy()` host copies** in the snapshot are the point of it: the shell runs at B = 1
  on the CPU, not in a training hot path.

## Requests and open questions

1. **Request (game):** a public `cores(fields)` (or `core_radius`) in
   `bnb.game.perception.sight`, used by `visibility`, so the play shell draws the core edge
   from the game's own function.
2. **Request (runtime):** `Realtime` takes a `torch.Generator` for its spread and
   uniforms. Today it draws from the global generator, so the shell's determinism rests on
   the entry point's `torch.manual_seed` and on nothing else drawing in between; replays
   in milestone 2 want a per-match generator.
3. **Request (runtime):** `Realtime` returns or keeps the tick's `Events` (died, killer,
   damage), for a kill feed and the "shards when the shield breaks" idea, and supports
   swapping in a background-compiled tick (milestone 2's pickers).
4. **Question:** should `--fog` default on? A team's view of an unfogged match shows every
   ship under the darkening, which is accurate but may surprise.
5. **Question:** saved themes go into the package's `themes/` (the repository in a
   checkout). An installed client would want a user directory; which?
6. **Open:** the boundary hatching at 35% and the 14 px minimum ring radius were set by
   eye against the mock-ups; the colour tool has no value for either.

## Size

- `src/bnb/play/`: 2466 lines of Python in 28 files (the largest `app.py`, 231), 441 lines
  of GLSL in 19 files, 66 lines of theme TOML; `src/bnb/controllers/human.py`: 36 lines.
- Tests: 755 lines in 9 files, `tests/sections/test_play_*.py` (38 tests, about 10 s on
  one thread, of which 9 need an OpenGL context).
- Docs: the explanation page and the guide (new prose), the reference page, three
  generated blocks and a status in the hook, and edits to installation, running the tests,
  the training loop, the roadmap legend, the guides index, the README.
- Dependencies: moderngl, imgui-bundle, glfw.
