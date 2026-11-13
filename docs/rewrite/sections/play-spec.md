# Play shell spec, milestone 1: the renderer (section 10)

Status: spec, 2026-11-01. Decisions: `play.md`. Look: `../gameplay-brainstorm.md`
(Visuals) and `../visual-mockups/` (start with `map-vis.png`, `shields3.png`,
`trails2.png`, `fogdark.png`, `ships3.png`, and the interactive `colour-tool.html`).
**(proposal)** marks a choice the spec makes, implemented as written and listed for the
author.

Milestone 1 is a window that plays a scripted match at the display rate, in the
brainstorm's look, with spectator and per-team views, theme switching, and one ship flown
from the keyboard. The developer panels, replays, live ship cards and trained agents are
later milestones; leave room for them, and build none of them now.

## 1. Packages and dependencies

- **`bnb/play/`** (new):
  - `snapshot.py`: the pure snapshot;
  - `loop.py`: the match loop;
  - `interpolate.py`;
  - `render/`: the passes and their shaders, as `.glsl` files next to the code;
  - `themes.py` and `themes/*.toml`;
  - `app.py`: window, input, camera;
  - `__main__.py`.
- **`bnb/controllers/human.py`:** the keyboard controller, a pure function of an input
  array (decisions: "a human agent is a pure function of an input array the shell fills
  from the keyboard each frame").
- **Libraries:** moderngl, imgui-bundle (its GLFW backend for the window and input), and
  numpy, added to the project's dependencies.
  - First check that wheels exist for the project's Python (3.14) and torch.
  - If either library has none, use the fallback in `play.md` (wgpu-py), and say so in
    the notes.
- **Layering:** `play` may import `runtime`, `controllers`, `game.api` and the contracts.
  Rendering never imports torch: the snapshot hands it numpy arrays.

## 2. The match loop

- **The sim:** `bnb.runtime.realtime.Realtime`, compiled, at 30 Hz.
  - Controllers: the chaser for every ship, except a ship the human flies.
  - Map, mode and fleet come from a preset (`deathmatch`, `deathmatch_wide`,
    `frontline`) or the sizes.
- **One thread, a fixed timestep (proposal):**
  - each frame, add the elapsed time to an accumulator;
  - step the sim while the accumulator holds at least one tick;
  - then render at `alpha = accumulator / tick`, interpolating between the last two
    snapshots.

  Add threads only if a measurement shows the frame budget needs them, and report it.
- **Determinism:** the loop's sim with the same seed and the same per-tick actions gives
  the same states as `Realtime` alone. The renderer never touches the sim's randomness.
- **Speed control:** a time scale (pause, ×1, and ×2 to ×8) and single-stepping while
  paused, as keys. Fast-forward runs more ticks per frame.

## 3. The snapshot

- **`snapshot(rules, state, observation) → Snapshot`:** pure, numpy arrays, one match. It
  holds what drawing needs:
  - per ship: position, heading, flight-path direction, angle of attack, nose rate, the
    shield and energy fractions of its type's maxima, alive, team, type, fired this tick,
    respawn wait, spawned this tick;
  - active bullets: position and direction, with the shooter's team;
  - fields: centre, radius, band width, level;
  - zones: centre, radius, role, meter, owner;
  - the arena radius, the mode, the scores or front, the clock;
  - per team, which ships it sees, from that team's observation (`visible`).
- **A reset** (or a map change) is flagged, so the static layers rebuild.

## 4. Interpolation

- Positions are linear between the previous and current snapshot. Headings take the
  shortest angle. Shield and energy are linear.
- **No interpolation across a teleport:** a ship that spawned, respawned or reset draws at
  its new state; bullets that were fired draw from their launch point.
- The rendered state lags the sim by up to one tick (33 ms).

## 5. Rendering (OpenGL 3.3 core)

Draw order, each a pass:
1. **Sky and arena:** the near-white sky inside the arena disc. The boundary damage zone
   beyond the arena radius is drawn per the brainstorm (hatched, in the thick-air colour).
2. **Fields, a topographic map of the true index:**
   - **The index texture:** computed with the game's own `medium(fields, x, y)` on a grid
     over the arena, once per map, and uploaded as a float texture. Fields are static
     within an episode. Never reimplement the profile.
   - **The fragment shader:**
     - fill by sign: thin air (ln n < 0) and thick air (ln n > 0) in the theme's
       colours, stronger with |level|;
     - contour lines at every third of a level, antialiased with screen-space
       derivatives.
   - **The band edges:** the core edge drawn as the one bold line and the outer edge as a
     very thin one in the same colour, exactly where the game's sight test changes (the
     radii `bnb.game.perception.sight` uses), drawn from the field geometry.
   - Resolution: fine enough that contours are smooth at the closest zoom **(measure;
     about 1 m per texel at 5v5 maps)**.
3. **Zones** (Frontline): their discs, role colours and meter arcs.
4. **Trails:**
   - per ship, a ring buffer of its last 1.5 s of interpolated positions **(proposal)**,
     drawn as a tapering strip that fades along its length;
   - in hard turns (the angle of attack beyond a theme threshold, as a fraction of the
     stall angle), a faint vortex line at the inside wingtip.
5. **Bullets:** short streaks along their direction.
6. **Ships:**
   - instanced silhouettes from the Komet family (`ships3.png`), filled with the team
     colour, every ship outlined;
   - a minimum on-screen size, so ships stay visible zoomed out.
7. **Shield and energy:** one ring at the hitbox:
   - shield on a 240° arc centred on the nose, energy on a 120° arc centred on the tail;
   - each shrinks toward its centre as it empties (`shields3.png`, row 3);
   - no shield arc when the shield is empty.
8. **Fog** (team views only):
   - a visibility mask of where the team can see, from the game's own `sighted` evaluated
     on a grid of target points against the team's living ships and the fields' cores
     and bands;
   - non-visible areas darkened to 60% brightness, saturation kept (`fogdark.png`);
   - terrain (arena, fields, zones) stays drawn under fog.
   - **Its cost:** measure the grid that fits the frame budget. Updating the mask every
     few ticks is allowed. A display-only reimplementation in a shader is allowed only
     if a test shows it agrees with `sighted` on sampled points, except within one grid
     cell of a boundary.
9. **HUD** (the player layer):
   - the clock;
   - the scores (team deathmatch) or the front (Frontline);
   - the respawn timer of the flown ship;
   - the current view, theme and time scale.

   Text through ImGui's foreground draw list is fine for milestone 1.

Antialiasing: MSAA, or distance-field edges in the shaders. Ships, rings and contours
must look clean at both a full-arena zoom and a dogfight zoom.

## 6. Views, camera and themes

- **Views, cycled with a key:**
  - **spectator:** fog off, everything drawn, including every bullet;
  - **team 0** and **team 1:** that team's fog. Only the ships it sees are drawn, and
    only its own bullets plus enemy bullets inside its visible area.
- **The camera:**
  - spectator: fit the arena, pan by mouse drag, zoom with the wheel;
  - follow the flown ship on a toggle.
- **Themes:** data files (`themes/*.toml`): sky, thin and thick air, contour, fog
  darkening, the two team colours, outline, shield, energy and trail. Cycled with a key.
  - The default: mint and pink air with navy and amber ships (`map-vis.png`).
  - Also ship the brainstorm's two other favourites (lemon and ice air with green and
    magenta ships; purple and ice air with green and vermilion ships) **(proposal)**.
  - **The theme editor** (author request, 2026-11-01), an ImGui panel toggled with a key:
    - it lists every colour of the current theme: sky, thin and thick air, contours, the
      bold core line, the fog darkening, the two team colours, outline, shield, energy and
      trail;
    - each colour has ImGui's picker (`color_edit3`, or `color_edit4` where alpha
      matters), so a colour can be dragged with the mouse or typed as a hex code
      (`#1F3A93`); hex entry is visible and works, not only the RGB fields;
    - changes apply on the next frame, with no restart: the colours are shader uniforms;
    - "Save as theme" writes the current colours to a new `themes/*.toml` (a name field),
      and "Revert" restores the theme's file values. Saved themes join the theme cycle.
    - Tests: a hex string round-trips through the theme's colour type; an edited theme
      saves and reloads equal; an offscreen render after an edit shows the new colour at a
      known pixel.
    - Docs: the editor is in the "Playing and spectating" guide.

## 7. The keyboard pilot

- **One ship, team 0's first by default,** flown by the human controller. Another key
  hands it to the next ship. The released ship returns to the chaser.
- **Keys (proposal):**
  - power: W boost, S reverse, neither coast;
  - stick: A and D steer at the middle level, with Shift for full and Ctrl for the
    lightest;
  - shoot: Space.
- A key reference is drawn in the HUD on F1.

## 8. Entry point

- `uv run --no-sync python -m bnb.play [--preset frontline] [--fog] [--view spectator]
  [--theme mint-pink] [--seed N]`.
- `--offscreen DIR --frames N` renders N frames to PNG files without a window (an
  EGL/standalone context), for tests and review.
- Leave `bnb.cli` alone: other sections are editing it. A `bnb play` subcommand comes
  later.

## 9. Acceptance

1. **The look:** offscreen renders of a Frontline and a team deathmatch scene, in each
   view and theme, read as the mock-ups describe. Inspect your PNGs, and put
   representative ones in `docs/` for the author.
2. **Frame rate,** at 1920×1080 with vsync off, the benchmark lock held:
   - the frame time p99 is at most 16.7 ms while the sim keeps 30 Hz real time, at 5v5
     (all three presets) and at 50v50 with 40 fields;
   - the report breaks it down: sim tick, controllers, snapshot, uploads, each pass, and
     the fog mask.
3. **Tests** (GL tests carry the `gpu` marker):
   - the snapshot is pure and complete;
   - interpolation is linear, takes the shortest heading, and snaps on teleports;
   - the index texture equals `medium` at sampled points;
   - the fog mask agrees with `sighted`;
   - the keyboard mapping covers every power, stick level and shoot;
   - loop determinism (section 2);
   - an offscreen smoke render with no GL errors and pixel checks: a ship pixel has its
     team colour, a fogged pixel is darker than the same place unfogged, and a hidden
     enemy draws nothing in its enemy's view;
   - themes load and validate.
4. **Docs:**
   - replace the play shell's in-progress page with an explanation of the renderer, and
     a guide, "Playing and spectating" (keys, views, themes, offscreen);
   - set its status in `docs/roadmap.yml` to milestone 1 built;
   - follow `docs/contributing/writing-docs.md`.
