# Section 10: the play shell (renderer and UI)

Status: decisions settled for milestone 1, 2026-11-01. Spec: `play-spec.md`.

## Already decided

- **The play layer** (`bnb/play/`, scaffold section 4): window, input, render snapshots,
  replay, spectator. It sits above `runtime` and adds no game rules. It reads state
  through a pure `snapshot(state)`.
- **Realtime play** is the same game `step` at B = 1 on the CPU, compiled
  (`bnb.runtime.realtime`; about 0.15 ms per tick at 10 ships).
- **Client-portable** (scaffold rule 9): the client sim is CPU torch, agents run in ONNX
  Runtime, and rendering reads snapshots on the GPU through OpenGL or wgpu, independent
  of torch.
- **Replays** record state keyframes plus actions (decisions, Presentation), so seeking is
  a jump to a keyframe and a short re-simulation.
- **The look** comes from the gameplay brainstorm (`../gameplay-brainstorm.md`, Visuals;
  mock-ups in `../visual-mockups/`, copied from branch
  `claude/game-rules-mechanics-szzz5h`):
  - a flat, minimalist look on a near-white sky;
  - fields drawn as a topographic map of the true refractive index: evenly spaced
    contours, with the innermost contour, drawn bold, on the core edge, and a thin line
    on the band's outer edge, the two radii where sight changes;
  - fog drawn by darkening to 60% brightness, saturation kept;
  - every ship outlined;
  - one ring at the hitbox: shield 240° centred on the nose, energy 120° centred on the
    tail, each shrinking toward its centre;
  - a fading trail, plus a faint vortex at the inside wingtip in hard turns;
  - normal colour vision first, a colour-blind option later.
- **Open in the brainstorm,** so they are themes here, not fixed: field colours (favourite:
  mint thin air, pink thick air) and ship colours (with mint and pink air, navy/amber and
  cobalt/gold read best).

## Decided (author, 2026-11-01)

- **The stack:** a Python desktop app, OpenGL through moderngl, and Dear ImGui
  (imgui-bundle) for the developer panels. If a library lacks wheels for the project's
  Python, the fallback is wgpu-py, with ImGui on its backend.
- **Two layers over one renderer:**
  - **the player view:** the game drawn in shaders, plus the in-game HUD;
  - **the developer view:** ImGui panels on top, toggled with a key.
- **Speed:**
  - the sim stays at 30 Hz, compiled;
  - the renderer runs at the display rate (60 fps or more) and interpolates between
    ticks.

  Measured 2026-11-01: an uncompiled B = 1 tick costs 25–30 ms at 10 ships on the loaded
  CPU, so compilation stays. Live ship-card edits need no recompile (the card is
  tensors). Structural changes (mode, field or zone count, fleet size) do: the view runs
  the uncompiled tick slowly while the new graph compiles in the background.
- **Milestones:**
  1. **The renderer:** 60 fps with interpolation and the brainstorm's look, a scripted
     match, spectator and per-team views, and one ship flown from the keyboard.
  2. **The developer view:**
     - team-view toggle and colour schemes;
     - take control of any ship;
     - mode, map and lineup pickers;
     - the speed dial (pause, step, fast-forward);
     - replays with a time scrubber.
  3. **After the ship card and the learner merge:**
     - live ship-card editing (handles or raw parameters, applied without recompiling);
     - loading trained agents per side.
- **Later, by agreement:** capsule fields and a new mode, after the ship card merges
  (they touch the medium and the sight test).
- **Bullets in team views (author, 2026-11-02):** the human UI draws every bullet in
  every view, fogged or not. Agents never see bullets (the observation has none), so
  this is an advantage humans have. A ship that fires is visible while it fires, as the
  game's sight rule already makes it. This reverses the milestone 1 fix round's masking
  of enemy bullets outside the visible area.
- **Libraries:** PyOpenGL is fine if it helps (author). imgui-bundle's own GLFW and
  OpenGL backends need it, so milestone 2 can replace the custom ImGui renderer and key
  table with them.
- **Bullets under fog are drawn at full brightness (author, 2026-11-02):** the bullets
  pass draws after the fog pass, so fog never darkens a bullet.
- **On-screen controls, not hotkeys (author, 2026-11-02):** the interface follows strategy
  games such as Age of Empires II and Civilization V. The most important information is
  visible at once without covering the battlefield, and clicking an object, in the game or
  on an info panel, opens a card with more about it. Every action has an on-screen control;
  keys are optional shortcuts, shown in tooltips. The layout for milestone 2's spec:
  - **top bar** (always shown): score, clock, the mode's objective (Frontline's meter,
    zone states), and the speed controls as buttons (pause, step, fast-forward);
  - **left card:** each team's fleet: type, shield and energy, alive, and the controller
    (human, scripted or agent); a row selects its ship;
  - **right card:** the selection, picked in a card or on the battlefield (a CPU hit test
    on the snapshot): a ship's live state, controller and a "take control" button; a
    field's index and core; a zone's progress and attackers;
  - **bottom strip** in replays: the scrubber with event markers (kills, captures);
  - **minimap** (lower right): the whole arena with fields, zones and ships as team
    dots, the camera's view as a rectangle; click or drag to move the camera. In a team
    view it shows that team's fog, as the battlefield does;
  - **menus** for mode, map, lineup and per-side agents, and a pause menu;
  - the cards collapse to slim tabs. While the player flies a ship the in-game HUD stays
    minimal, drawn in the game's style, and the cards open on demand.

  The panels are Dear ImGui, themed to the brainstorm's flat, light look.
