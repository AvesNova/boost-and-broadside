# Play shell spec, milestone 2: the on-screen interface (section 10)

Status: spec, 2026-11-03. Decisions: `play.md` (above all "On-screen controls, not
hotkeys"). Milestone 1: `play-spec.md` and `play-notes.md` (its "What milestone 2 should
build on" and its requests). **(proposal)** marks a choice the spec makes, implemented as
written and listed for the author.

Milestone 2 turns the renderer into something a person can use without reading a key
list: the information is on screen, every action has a control, and a match can be set up,
watched, flown, paused, replayed and scrubbed from the interface. It also plays trained
agents, so the MVP's RL agent can be watched and played against.

## 1. Principles

- **Everything on screen.** Every action has a button, menu item or click target. Keys are
  optional shortcuts, shown in the control's tooltip. The F1 key reference stays as a
  reference, not as the only way in.
- **The battlefield stays visible.** Cards sit at the edges, are semi-opaque, and collapse
  to slim tabs. Nothing covers the centre of the screen except a menu the player opened.
- **Click to inspect.** Clicking a ship, field or zone, on the battlefield, the minimap or
  a card row, selects it and opens the right card.
- **One look.** The panels are Dear ImGui, themed to the brainstorm's flat, light look:
  the theme's sky and team colours, flat fills, thin outlines, no gradients or heavy
  shadows. Panel colours come from the theme files, so the theme editor edits them too.
- **The sim never waits for the interface.** The panels read the snapshot and the loop's
  state; they never step the sim or touch its randomness.

## 2. The layout

- **The top bar** (always shown, one row):
  - the score, and the clock (match time and the time limit);
  - the mode's objective: Frontline's meter and each zone's state; in team deathmatch the
    kill score;
  - the speed controls: pause/play, step (while paused), and ×1 / ×2 / ×4 / ×8;
  - the view selector (spectator, team 0, team 1) and the menu button.
- **The left card: the fleets.** One section per team: a row per ship with its type, a
  shield and an energy bar, alive or respawning (with the wait), and its controller
  (human, scripted, or the agent's name). Clicking a row selects its ship. A team's
  controller can be changed here (section 5).
- **The right card: the selection.**
  - **A ship:** live state (shield, energy, speed, angle of attack, power and stick, the
    last shot), team, type, controller, kills and deaths; a "Take control" button (a
    "Release" button on the flown ship); "Follow" to put the camera on it.
  - **A field:** its index and core radius.
  - **A zone:** its owner, its progress, and the ships contesting it.
  - Nothing selected: the card collapses.
- **The bottom strip** (replays only, section 6): the scrubber with event markers.
- **The minimap** (lower right): the whole arena with fields (their cores), zones and ships
  as team dots, and the camera's view as a rectangle. Click or drag to move the camera. In
  a team view it shows that team's fog and only what that team sees, as the battlefield
  does. It is drawn by the renderer into its own viewport, not by ImGui
  **(proposal)**.
- **The kill feed** (upper right, under the top bar): the last few kills, killer and
  victim in team colours, fading after a few seconds **(proposal)**.
- **Menus** (from the menu button, or Escape):
  - **New match:** mode, map, fleet size and lineup (each ship's type), and each side's
    controller (section 5); "Start";
  - **Replays:** open a saved replay (section 6);
  - **Settings:** theme (with the theme editor), fog, interface scale. Fog is on by
    default (author, 2026-11-03), so team views show what the team sees; the spectator
    view still shows everything. `--fog` becomes `--no-fog`;
  - **Pause menu** when a menu opens during a live match: resume, restart, new match, quit.
- **Collapsing:** each card has a collapse control and collapses to a tab on its edge; the
  state is remembered for the session.
- **While the player flies a ship:** the in-game HUD stays minimal and in the game's
  style (shield, energy, power and stick, near the ship or at the bottom centre); the
  cards stay collapsed unless opened.
- **Scale:** the layout works from 1280×720 to 2560×1440; an interface scale setting
  multiplies ImGui's font and sizes **(proposal)**.

## 3. Selection and hit testing

- **A CPU hit test on the snapshot,** in world coordinates: the nearest ship within its
  drawn radius plus a few pixels, else the zone, else the field, under the cursor. In a
  team view, only what that team sees can be selected.
- **Selection survives** death and respawn (by ship index), and is cleared by a new match.
- **The selected object** is outlined on the battlefield and on the minimap.

## 4. The camera

- Spectator: fit the arena on start; pan by drag, zoom with the wheel; the minimap moves
  it; "Follow" on a selected ship.
- Flying: follows the flown ship by default, as in milestone 1.

## 5. Controllers per side, and trained agents

- **Each side's controller** is chosen in the New Match menu and changed live from the
  fleet card: scripted, a trained agent, or (per ship) the human.
- **Trained agents:** a policy file (`bnb.runs.load_policy`), picked from a list of the
  local runs' policies (`runs/*/policy/` and checkpoints) or a file dialog. Its game
  config and ship card must match the match's, or the shell says why it cannot load it.
  - It runs with torch on the CPU at B = 1, its carry kept per side and reset at each
    ship's episode boundary, as in training. A model that cannot keep 30 Hz on the CPU
    may run on the GPU when one is present. ONNX Runtime, the client's eventual path, is
    not part of this milestone (author, 2026-11-03).
  - Sampling: the policy's distribution, with a temperature control on the fleet card
    (default 1) **(proposal)**.
- **Taking control** of a ship of an agent's side replaces that ship's action only; the
  agent keeps controlling the rest of its team and keeps its carry.

## 6. Replays

- **Recording:** every live match is recorded: the setup (config, ship card, seed,
  controllers), a state keyframe every k ticks (`Realtime`'s state, cloned), and every
  tick's actions. Saved on match end or quit to `replays/<date>-<mode>-<name>/`, keeping
  the newest 50 (author, 2026-11-03).
- **Playback** re-simulates from the nearest keyframe, so seeking is a jump plus at most
  k ticks. Determinism needs `Realtime` to take a per-match generator (milestone 1
  request 2).
- **The scrubber:** play/pause, speed, drag to seek, step a tick; markers for kills,
  captures and the match's end, from the recorded events; clicking a marker seeks a few
  seconds before it.
- **In a replay,** every view, selection and card works; "Take control" forks a new live
  match from the current tick; the replay is untouched (author, 2026-11-03).

## 7. Runtime requests carried from milestone 1

- **`Realtime` takes a `torch.Generator`** for its draws (request 2).
- **`Realtime` keeps the tick's `Events`** for the kill feed and the event markers
  (request 3).
- **The pickers** build a new match; a structural change (mode, map, fleet) runs the
  uncompiled tick while the new graph compiles in the background, then swaps it in
  (request 3).
- **From the cleanup pass:** `PRESETS` stays in `bnb.game.modes.rules`, read through
  `bnb.game.api` (settled 2026-11-03: presets are game configuration); `Realtime(compiled=)`;
  `snapshot.py` reads the state only through `bnb.game.api` (the opaque-state rule);
  format `bnb/play/` with ruff and add its missing docstrings, so the format check and the
  docstring count cover it.
- **The custom ImGui renderer and key table** may be replaced by imgui-bundle's own GLFW
  and OpenGL backends (PyOpenGL is allowed, `play.md`).

## 8. Acceptance

1. **Usable without keys:** a scripted test drives the offscreen session through the
   interface only (clicks on known controls, through ImGui's test hooks or synthetic
   input): start a Frontline match, select a ship on the battlefield, take control,
   release, pause, step, fast-forward, change the view, open the minimap's far corner,
   start a new team deathmatch match with an agent on one side, save and open its
   replay, seek to a kill marker.
2. **Screenshots for the author:** offscreen renders of the interface in each view and
   theme, live and in a replay, at 1920×1080 and 1280×720, in `docs/`.
3. **Frame rate:** milestone 1's target still holds with every card open and the minimap
   drawn: frame p99 ≤ 16.7 ms at 5v5 (all three presets) and at 50v50, sim at 30 Hz;
   the report adds the interface's own cost.
4. **Tests** (GL tests carry the `gpu` marker; quick unless slow):
   - hit testing picks the right object, and respects the team view;
   - the minimap's camera mapping round-trips;
   - a replay re-simulates bitwise to the recorded states, and a seek equals playing to
     the same tick;
   - a trained-agent side (a tiny saved policy) plays a match, and a mismatched policy
     file is refused with its reason;
   - taking control of an agent's ship leaves the agent's other ships' actions unchanged;
   - the kill feed and the event markers match the match's `Events`;
   - collapsing and the interface scale leave the battlefield's pixels uncovered where
     the layout says.
5. **Docs:** the "Playing and spectating" guide rewritten around the interface (keys as
   shortcuts only), the play explanation page updated, `docs/roadmap.yml` at milestone 2
   built.

## 9. Not in this milestone

- Live ship-card editing (milestone 3).
- ONNX export and ONNX Runtime agents.
- A colour-blind theme; sound; networked play.
