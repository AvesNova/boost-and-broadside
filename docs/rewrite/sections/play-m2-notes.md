# Section 10 (play shell), milestone 2: notes

Branch `section/play-m2`. The spec is `play-m2-spec.md`; the brief `play-m2-brief.md`.
Milestone 1's notes are `play-notes.md`.

## Progress

- [x] 1. Runtime requests (spec 7): `Realtime(compiled=)`, a per-match generator, the
  tick's events kept, keyframes, the background compile, the lineup.
- [x] 1. Cleanup requests: `snapshot.py` reads the state only through `bnb.game.api`
  (new `api.display`); `bnb/play/` formatted with ruff and out of the format exclusions;
  every public object has a docstring and `bnb.play` is out of the docstring exemptions.
- [x] 2.1 Layout skeleton, ImGui theme, collapsing, interface scale.
- [x] 2.2 Top bar and speed controls; view selector.
- [x] 2.3 Selection, hit test, selection card, fleets card.
- [x] 2.4 Minimap.
- [x] 2.5 Menus and New match, with the background compile.
- [x] 2.6 Trained agents per side.
- [x] 2.7 Replays, scrubber, kill feed.
- [x] 2.8 Flying HUD.
- [x] Screenshots in `docs/images/play/`.
- [x] Docs: the guide, the explanation page, the reference, the roadmap.
- [ ] Measurements (frame time with every card open) — see below.
- [ ] Full suite under the lock.

(Updated as the work goes; a fresh agent continues from the first unchecked item.)

## Design

### Layout of the code

| Module | Job |
|---|---|
| `runtime/realtime.py` | `Realtime(config, compiled, fogged, generator, lineup, background)`: the match's own generator (draws outside the compiled tick), `events` (the last tick's `Events`), `keyframe()` / `restore()`, the compiled tick shared per `Rules`, the background compile (`compiling`, `wait_compiled`, `uses_compiled`), `step(actions, eager=)` |
| `play/loop.py` | `SideSetup`, `MatchSetup` (game, lineup, fog, seed, sides), `MatchEvent` (kill, capture, end), `tick_events`, `MatchLoop` (sides, pilot, kills and deaths, `on_tick`) |
| `play/sides.py` | `Scripted`, `Agent` (Yemong or relational at B = 1, its carry, Gumbel-max at a temperature from its own generator), `load_agent` / `AgentError` / `mismatch`, `policy_choices`, `label_of`, `describe`, `build_sides` |
| `play/replay.py` | `Recorder`, `save` / `replays` / `prune` (newest 50), `Replay.load`, `ReplayLoop` (`seek`, `markers`) |
| `play/selection.py` | `Selection`, `hit_test`, `shown_ships` |
| `play/app.py` | `Options`, `Session`: the match or replay, camera, view, selection, follow, feed, recorder, and every command the controls call |
| `play/window.py` | GLFW to ImGui's input state, nothing else |
| `play/interface/` | `interface.Interface` (one ImGui frame: cards, menus, battlefield input, shortcuts), `layout.Layout` (every card's rectangle), `style` (ImGui theme from the theme file, the scale, Roboto), `controls` (named controls, icons, bars), `cards` (window, header, tab), `top_bar`, `fleets`, `inspector` (the selection card), `minimap`, `kill_feed`, `scrubber`, `flying`, `menus`, `shortcuts`, `text` |
| `play/render/minimap.py` | `MinimapPass`: the minimap's own scene, texture and fog, from the battlefield's passes |
| `play/render/overlay.py` | `outline_table`: the selection's ring |
| `game/api.py` | `display(state)` and `DISPLAYED` (added; see deviations) |

### Input

All input goes through ImGui's input state. The window (`window.py`) only forwards GLFW's
events; the session reads the rest inside its ImGui frame: the cards take what they are
over, and `Interface._battlefield` takes the rest (a click without a drag runs the hit
test, a drag pans, the wheel zooms), then `_keyboard` sets the pilot's inputs from the
flight keys and calls the shortcuts' session methods. So an offscreen test drives the
whole interface with synthetic ImGui events, and every control records its rectangle by
name (`Controls`), which the acceptance test clicks.

### Layout and collapsing

`Layout` computes every rectangle from the screen size, the scale and which cards are
open; each card's window is placed on its rectangle (the fleets and selection cards grow
with their content up to it). A collapsed card leaves a tab on its edge. The open set
lives in `Interface.open` for the session. Taking control collapses the fleets and
selection cards; a card the player opens stays open.

### The theme for ImGui

`Theme` gained five colours: `panel` (the cards' fill, semi-opaque), `panel_line`,
`control`, `control_active`, `selection`; every shipped theme lists them, and the theme
editor shows them like the others. `style.apply_style` sets every ImGui colour and size
from them and the scale (font 15 px times the scale; sizes times the scale). Fonts are
Roboto Regular and Bold, shipped with imgui-bundle.

### Live and replayed matches

A live match is a `MatchLoop`; a replay is a `ReplayLoop` (a `MatchLoop` whose actions
come from the recording). Both call `Session._on_tick` after every tick, which feeds the
kill feed and the recorder. The session saves a recording when its match ends (and starts
the next one), when the player starts another match or opens a replay, on "Save replay",
and on close. A fork from a replay (Take control) hands the replay's `Realtime` to a new
live `MatchLoop`, with the recorded sides' agents reloaded where they load.

### Agents

`Agent.act` runs `model.step` (Yemong, with its carry) or `model(view)` (relational) on the
team's view and samples `policy_logits / temperature` by Gumbel-max from the agent's own
generator; temperature 0 is the argmax. Yemong draws position offsets from the default
generator; the agent runs those draws inside `torch.random.fork_rng`, seeded from its own
generator, so torch's global generator is never touched and an agent is reproducible
from the match's seed. Loading builds the model inside `fork_rng` for the same reason.

## Spec items

### 1. Principles

- **Everything on screen:** every action is a button, menu item, combo or click target;
  `interface/shortcuts.py` maps the optional keys to the same session methods, and each
  control's tooltip names its key. F1 (or Settings, "Show the key list") shows `KEYS`.
- **The battlefield stays visible:** the cards sit at the edges, semi-opaque (`panel`
  #FFFFFFE0), and collapse to tabs; only a menu covers the centre. Tested (8.4 below).
- **Click to inspect:** the battlefield (hit test), the minimap (moves the camera; see
  deviations), the fleets card's rows and the top bar's zone dots select.
- **One look:** the cards are ImGui themed from the theme file (above).
- **The sim never waits for the interface:** the cards read the snapshot and the loop's
  counters; the controls call session methods between frames' ticks; nothing in the
  interface draws randomness.

### 2. The layout

- **Top bar:** menu button, the view selector (team buttons in team colours), the score
  (team deathmatch: kills in team colours and the target) or Frontline's front (a tug of
  war of `front_threshold` cells a side) and zone dots (owner's colour, spawn zones
  lighter, a defense zone's meter as an arc in the attacker's colour), the clock and the
  limit, and pause/play, step (enabled while paused), ×1 ×2 ×4 ×8. In a replay a
  "REPLAY" tag.
- **Fleets card:** per team a swatch, the name, the controller combo (and an agent's
  temperature); per ship its name ("Heavy 2", in team colour), shield and energy bars,
  alive or the wait, and the controller ("you", "scripted", or the policy's directory).
  A row selects its ship.
- **Selection card:** ship (team, type, controller, kills / deaths, shield, energy, speed
  from the last two ticks, angle of attack, power and stick of its last action, "firing",
  the last shot's age; Take control / Release, Follow / Following; out of a fogged team
  view's sight it says so instead of the live state); field (index n and the speed
  factor, radius, core, band); zone (role, owner, progress and attacker, the ships
  contesting it, in a fogged team view only those the team sees). "Clear selection".
- **Bottom strip:** the scrubber, in replays (section 6).
- **Minimap:** see 2.4 in the progress list; drawn by the renderer (the proposal).
- **Kill feed:** upper right, under the top bar, left of an open selection card; the last
  five kills in team colours ("killer → victim", or "the boundary → victim"), each fading
  in its last second of six (proposal: six seconds of game time).
- **Menus:** main (Resume, Restart, New match, Replays, Save replay, Settings, Quit); New
  match; Replays; Settings (theme and Edit, fog in new matches, interface scale, the key
  list). Opening the menu pauses a live match; closing resumes it. `--fog` is now
  `--fog/--no-fog`, on by default.
- **Collapsing:** a chevron in each card's header, a tab on the edge; remembered for the
  session.
- **While flying:** the fleets and selection cards collapse; the flying HUD (below).
- **Scale:** 0.75, 1, 1.25, 1.5, 2 (proposal: font and every size multiplied).

### 3. Selection and hit testing

`selection.hit_test` as specified (ship within its drawn ring radius plus 4 px, else zone,
else field; team views see only their own and visible ships). The selection is by sim
index (survives respawn; a dead or unseen selected ship is not outlined), cleared by a new
match. The outline is a ring (a faint wide halo in the sky colour under a 1.6 px line in
`selection`) on the battlefield and the minimap.

### 4. The camera

Spectator: fitted at start; drag pans (after 3 px of movement), the wheel zooms about the
cursor; the minimap moves it; Follow keeps it on a ship. Flying follows the flown ship.
Home fits the arena.

### 5. Controllers per side, and trained agents

Done as specified, with these choices: the list holds `runs/*/policy/`, each run's
`checkpoints/*` and `league/*` (newest run first); a combo also takes a typed policy
directory (the file dialog; see deviations). The match must equal the policy record's
`game` (field by field, floats to 1e-6) and `ship_card`, and use the default lineup; the
refusal names the fields. A record without its game is refused. Temperature on the fleet
card, 0 to 2 (proposal: default 1). Taking control replaces the flown ship's action only;
the agent's carry and generator move on as without it (tested).

### 6. Replays

Recording, saving (on the match's end, on starting another match or a replay, on Save
replay, on quit; `replays/<YYYYmmdd-HHMMSS>-<mode>-<side0>-vs-<side1>/`, newest 50 kept),
playback by re-simulation from keyframes every 90 ticks, the scrubber with markers, and
the fork as specified. Playback repeats each tick compiled or not as recorded; a recorded
compiled tick plays eager until the replay's compiled tick is ready (a few seconds after
opening a replay of a new kind of match), which can differ in the last float bits until
the next keyframe (the states are restored at each keyframe).

### 7. Runtime requests

All done: the generator, the events, `compiled=`, the background compile and swap, the
lineup (`Rules.team_types`), keyframes. `PRESETS` read through `bnb.game.api` (as it was).
`snapshot.py` reads the state only through `bnb.game.api.display` and the display helpers.
`bnb/play/` is formatted and out of both exclusions. The custom ImGui renderer and key
table stay (no PyOpenGL needed); the key table is now only GLFW to ImGui.
