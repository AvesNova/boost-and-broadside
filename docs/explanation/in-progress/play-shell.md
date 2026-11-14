---
covers: [src/bnb/play/, src/bnb/controllers/human.py]
reviewed: 2026-11-03 09:45
---

# The play shell

<!-- generated: section-status -->

The play shell (`bnb.play`) is a desktop window that plays one match at the display rate:
the realtime sim at its fixed tick underneath, the game drawn in shaders on top, and a
layer of Dear ImGui for the interface: the cards, the menus and the minimap's frame. It adds no game rules. It reaches the game only through
`bnb.game.api`: it reads a tick through one pure function, `bnb.play.snapshot.snapshot`,
and draws the medium, the cores and the zones with the game's own functions. [Playing and spectating](../../guides/playing.md) covers the
interface, views, agents, replays, themes and offscreen renders.

![Team 0's view of a Frontline match](../../images/play/frontline-team0.png)

## The match loop

`bnb.play.loop.MatchLoop` holds a `bnb.runtime.realtime.Realtime` match, compiled, at its
fixed tick, set up by a `MatchSetup` (the game, the lineup, fog, the seed, and each
side's controller). Each side is played by its controller (`bnb.play.sides`): the
scripted agent (`bnb.controllers.scripted.Scripted`, the default), the chaser
(`bnb.controllers.scripted.Chaser`), or a trained agent. At most one ship, the
pilot, is flown from the keyboard instead; its action replaces that ship's only.

The loop runs on one thread with a fixed timestep:

1. Each frame adds the elapsed display time, times the time scale, to an accumulator.
2. The sim steps while the accumulator holds a whole tick.
3. The frame draws at `alpha = accumulator / tick` between the last two snapshots.

So the drawn state lags the sim by at most one tick. Pausing stops the accumulator, and
`MatchLoop.single_step` advances exactly one tick. A backlog beyond `CATCH_UP_TICKS` ticks
per unit of time scale is dropped, so a stall does not snowball.

The match draws its randomness from its own `torch.Generator`, seeded by the setup, outside
the compiled tick; the loop draws none. Given the same seed and the same per-tick actions,
it produces the same states as `Realtime` alone
(`tests/sections/test_play_loop.py::test_the_loop_gives_the_same_states_as_realtime_alone`).

## The snapshot and interpolation

`bnb.play.snapshot.Snapshot` is one tick of one match as numpy arrays: the ships, bullets,
fields and zones, the arena, the clock, the score or the front, and each team's
visibility from its own observation's `visible` column. `snapshot` copies every value, so
a snapshot does not change when the sim steps on. Its `fogged` flag says whether the
match has fog of war.

`bnb.play.interpolate.interpolate` gives the drawn state between two snapshots:

- positions, shields and energies are linear;
- headings take the shorter way round;
- a ship that spawned or respawned draws at its new state, with no interpolation across
  the teleport (an episode reset spawns every ship and clears every bullet);
- a bullet fired on the later tick draws from its launch point, the shooter's position at
  the earlier tick, where the game fires it from.

Rendering reads only snapshots and never imports torch
(`tests/sections/test_play_layers.py`). This keeps the renderer independent of the sim,
as the client design requires: the client sim is CPU torch, the agents run in ONNX
Runtime, and rendering is OpenGL.

## The passes

The renderer (`bnb.play.render.renderer.Renderer`) draws into a multisampled framebuffer
and resolves it onto the target; then the minimap into its own texture, and the ImGui
layer over everything. The passes, in order:

| Pass | Draws |
|---|---|
| Sky | The sky inside the arena disc; beyond it the boundary damage zone, hatched in the theme's boundary colour, and a line on the edge. |
| Fields | A topographic map of the true refractive index: terraces filled by sign (thin air below ambient, thick air above), stronger with depth, and a contour at every third of a level. The core edge is drawn bold and the band's outer edge thin, in the same colour. |
| Zones | Frontline's zones, tinted with their owner's colour, with each defense zone's capture meter as an arc in the attacker's colour. |
| Trails | Each ship's recent path, a strip that tapers and fades with age, and in hard turns a faint vortex line from the inside wingtip. A dead ship's trail fades out where it ended. |
| Ships | Silhouettes from the Komet family, filled with the team colour, every ship outlined. A ship type without a silhouette of its own draws a plain dart. |
| Rings | One ring at the hitbox: the shield on an arc centred on the nose, the energy on an arc centred on the tail, each shrinking toward its centre as it empties. |
| Fog | In a team's view, everything the team cannot see is multiplied by the theme's fog colour: darker, with its saturation kept. |
| Bullets | Short streaks along each bullet's direction, in the shooter's team colour, drawn above the ships and the fog, so fog never darkens a bullet. |
| Selection | A ring around the selected ship, field or zone, above everything. |

Ships, rings, bullets and trails keep a minimum size on screen, so they stay visible at the
full-arena zoom. Lines are antialiased in their shaders, and the ships by the multisampled
framebuffer.

### The index texture

The fields pass draws from a float texture of ln n on a grid over the map,
built by `bnb.play.index_map` with the game's own medium (`bnb.game.api.log_index`). Fields
are static within an episode, so the grid is built once per map. The grid is a square
over the arena and every field's outer edge (`bnb.play.index_map.grid_for`). Its texel is
`TEXEL_METRES` at map scale 1 and grows with the map, so maps whose fields stay inside the
arena all have a texture of one size, and a field that reaches past the arena makes the
texture larger. Only tiles that touch a field's outer edge are evaluated, each with only the
fields that touch it; beyond every outer edge ln n is exactly 0. A new map is built a
few tiles per frame (`IndexBuilder.run` with a time budget), so a map change does not stall
a frame.

The shader samples the texture with a bicubic (Catmull-Rom) filter through the texel
centres, which keeps the contours smooth at the closest zoom. The contours are lines of
the sampled ln n, antialiased with screen-space derivatives, and fade where they would be
closer together than a few pixels.

The two band edges are drawn from the field geometry, where the game's sight test changes:
the bold line at `bnb.game.api.core_radius`, inside which a point is hidden from outside
the field, and the thin line at `bnb.game.api.outer_radius`, beyond which a point is hidden
from the core
(`tests/sections/test_game_api_display.py::test_core_radius_is_where_the_sight_test_blocks`
and `::test_outer_radius_is_where_a_core_stops_seeing`).

### The fog mask

In a team's view the renderer first draws a visibility mask in screen space, at
`MASK_SCALE` of the target's resolution: a fragment shader applies the sight rule of
`bnb.game.perception.sight.sighted` to the world point under each texel, against the team's
living ships and the fields' cores and bands. A test holds the mask to `sighted` at every texel away
from the mask's edges
(`tests/sections/test_play_render.py::test_fog_mask_agrees_with_sighted`). The game's
own function on the CPU was too slow for a frame budget at large fleets; the shader runs
the same rule on the GPU.

Every bullet is drawn in every view, above the fog at full brightness; agents never see
bullets, so this is an advantage human players have. Ships come from the observation instead: a team's view draws only the
ships its observation shows (a ship that fires is visible while it fires, by the game's
sight rule). Each trail sample records which teams saw
the ship then, so a team's view draws only the parts of an enemy's path it saw; an enemy
revealed for a moment by firing does not show where it came from. Terrain (the arena, the
fields and the zones) stays drawn under the fog.

In a match without fog of war every ship is visible to both teams, so a team's
view has no fog: it draws everything and darkens nothing.

## The interface

The interface (`bnb.play.interface`) follows strategy games such as Age of Empires II and
Civilization V: the most important information is on screen without covering the
battlefield, and clicking an object opens a card about it. Every action has a control;
the keys are shortcuts, named in the controls' tooltips.

`bnb.play.interface.layout.Layout` places every card for a screen size and an interface
scale: the top bar, the fleets card on the left (wide enough for the longest ship name,
`bnb.play.interface.fleets.width`), the selection card and the minimap on the right, the
kill feed under the top bar, the scrubber along the bottom in a replay, and the flying
HUD at the bottom while a ship is flown. A card is drawn inside its rectangle, or
as a slim tab on its edge when collapsed, so everything outside the rectangles is
battlefield; a test renders with and without the interface and checks those pixels are
unchanged. The cards are Dear ImGui windows themed from the theme file: the panel,
line and control colours are theme values like the battlefield's, so the theme editor
edits both. The cards are translucent, and every control has a 1-pixel outline in the
line colour, so a control stays visible where the fog greys the card behind it.

All input reaches the session through ImGui's input state, which the window fills from
GLFW (and a test fills with synthetic clicks). What no card takes belongs to the
battlefield: a click runs the hit test (`bnb.play.selection.hit_test`, on the CPU on the
drawn snapshot: the nearest ship within its drawn ring, else the zone, else the field
under the point; in a team view only what the team sees), a drag pans, the wheel zooms.
Every control records its rectangle by name (`bnb.play.interface.controls.Controls`), so
a test drives the whole interface by clicking named controls.

The minimap is drawn by the renderer at its own square size with the battlefield's own
passes (sky, fields, zones, the team's fog from its own mask) and ships as dots, into a
texture the minimap card shows; the camera's view is a rectangle over it.

## Trained agents

A side can be a trained agent: a policy file (`bnb.runs.checkpoint.load_policy`) of a
Yemong or relational model, run with torch at B = 1 on the GPU when one is present and on the CPU otherwise. The policy file records
the game it was trained on and its ship card; the shell loads it only when both equal the
match's (and the lineup is the game's default), and otherwise names what differs. The
agent keeps its carry from tick to tick, handing the model the view's reset flag as
training does, and samples its policy at a temperature with Gumbel-max from its own
generator; its model's own draws (the position offset) are keyed by a decision key from a
second generator of its own. Flying one of its ships replaces that ship's action only: the agent decides for
all its ships as usual and keeps its carry.

## Replays

Every live match is recorded (`bnb.play.replay.Recorder`): the setup, a keyframe every
`KEYFRAME_TICKS` ticks (the state, every team's latest view and the generator's state),
every tick's actions, whether each tick ran compiled, and the match's events. Playback
(`ReplayLoop`) restores the nearest keyframe and repeats the recorded actions, each tick
compiled or not as it ran, so it reproduces the match (the tests hold integers exactly and
floats to float32's tolerance: `tests/sections/test_play_replay.py`) and a seek costs at most `KEYFRAME_TICKS` ticks.
The scrubber's markers and the kill feed come from the same events (`MatchEvent`: kills
from the tick's `Events`, captures from the front's moves, the end from `done`).

Taking control in a replay continues the replay's own match from the shown tick as a new
live match; the saved replay is not changed.

## Compiling in the background

The window's thread never compiles, traces or measures: Python's interpreter lock is
shared by the process's threads, so a compile on any thread of the window's process
starves the frame loop (a cold `torch.compile` of the tick froze the window for 8-18 s,
and even a compile from warm caches for 2-4 s). Slow work runs in child processes, which
have their own lock, and the window only waits for their results:

- **The tick.** A new kind of match (another mode, map size or fleet) needs its tick built.
  `bnb.runtime.realtime.library` exports it and packages it with AOTInductor in a child
  process (`bnb.runtime.built`, about 30 s), keeps it in the user's cache directory
  (`bnb.cache`, keyed by the game's code, torch's version and the static rules), and loads
  it in milliseconds. `Realtime(background=True)` does this on a thread that waits for the
  child, while the match runs the uncompiled tick (about 5 ms a step), then swaps the built
  tick in (about 0.4 ms). The built tick agrees with the uncompiled one to float32 fusion
  rounding, and replays repeat each tick as it ran. A built tick is shared by every match of
  the same static rules, in this process and every later one.
- **The scripted agent's step.** Uncompiled, a scripted agent's decision is a few hundred
  small torch operations: about 4.3 ms a side on one CPU thread, flat over the match, so two
  scripted sides held a frame with a tick to 12-15 ms. It is built the same way
  (`bnb.runtime.scripted_step`, about 15 s): from the shapes of the agent's first step,
  with the ship characteristics an input (a tuned card plays on the same library), keyed by
  the code, the parameters and the shapes. Built, it takes about 0.1 ms a side. Its random
  draws are keyed by the decision key it is given, drawn from the agent's own generator,
  so built and uncompiled give the same actions. Until it is in, the agent steps
  uncompiled.
- **The scripted agent's characteristics** are measured once per card, in a child process
  (`bnb.game.handling.characteristics.measure_aside`), and kept in the same cache. A side
  set to a scripted agent whose card is not measured yet (`Session.set_side`, or a match
  started with one) plays on as it was, the chaser at a match's start, and the agent
  takes over when the measurement is in.
- **A GPU agent's CUDA graph** is captured on a thread before the agent plays
  (`Agent.prepare`); the side plays on as it was meanwhile.

A state the tick returns never shares a tensor between two of its fields
(`perceive` copies the ships' alive flags and nose rates into the match's "previous"
fields): a compiled tick traced on unshared inputs would otherwise fail its guards on the
match's own state and recompile on the window's thread.

## Live tuning

The tuning panel (`bnb.play.tuning_panel`, F2) changes a live match's numbers while it
plays, through `bnb.play.tuning.Tuner` and `MatchLoop.retune`:

- **Rule settings** (`bnb.game.api.tuning`: the arena and clock, the bullet drag, the
  shared flight constants, spawns and each mode's rules) apply from the next tick, without
  a recompile: the compiled tick takes their values as tensors (`bnb.game.tuning.live_of`) and
  is keyed on the rules with them blanked (`bnb.game.tuning.static_of`), so every value runs on
  the same compiled tick. The state carries over (`bnb.game.api.retune`: Frontline's zone
  ring is rebuilt for the new radius and ring; shields and stores are capped). The map and
  the spawns follow the rules from the next episode. Sizes (ships, fields, lineup, the
  tick) take a new match.
- **Handles**, per ship type, go through the card, also an input of the compiled tick. The
  panel shows a type's handles at once when its card and the rules its handles depend on
  (the flight constants and the bullet drag, `bnb.play.tuning.SET_BY`) are its ship
  file's: each file's card was solved exact to its handles. Otherwise (a flight constant
  changed, a preset, the time scale) the type's handles are measured before they show.
  What each card flies is measured lazily, one type at a time on a measuring worker: the
  type the panel shows first, then the match's types; no other type is flown.
- A changed handle applies the instant estimate (`bnb.game.api.Estimate`, about 50 ms, on a
  worker process) on the next frames; once the handles settle, the exact solve of the
  changed types runs on a second worker from that estimate, one type at a time, swapping
  each better card in and reporting the handles it flies; the panel marks a handle flown
  more than 0.5% off its set value with a red dot, the flown value in its tooltip. A
  combat handle of an exact card whose flight is measured skips both: the flight does not
  read the combat fields, so the solve's answer is that card with its combat fields in
  closed form (`combat_card`), applied at once. A flight constant re-measures the cards'
  handles (the cards are kept); the bullet drag changes only the gun range, in closed form (each type keeps its bullet
  life and drag scale; a gun range set afterwards solves them under the new drag).
- **The workers are processes**, not threads: the solve and the estimate are thousands of
  small tensor operations, and on a thread each waits for the interpreter lock while the
  match runs. Each worker flies the compiled flight tick (`bnb.runtime.flights`), compiled
  on its first flight of each batch size (one type's measurement, one type's solve
  iteration) and flight constants: about a minute the first time on a machine, then
  seconds from the on-disk cache; the solving worker compiles when the tuner starts, ahead
  of the first solve. Compiled, one type's measurement takes about 1 s for a standard type
  and 7-13 s for a real-scale one (its climb lasts about an hour of game time); the game's
  own tick is about ten times slower.
- **Bullet slots.** A play-shell match allocates a slot per tick of the longest bullet life
  a card may have (`bnb.game.rules.LONGEST_BULLET_LIFE`, 2 s;
  `bnb.game.tuning.most_bullet_slots`, 60 at 30 Hz), so no fire rate or gun range runs out
  of them; the compiled 5v5 tick cost the same 0.7-1.0 ms at 6, 15 or 45 slots.
  Elsewhere the rules hold what the card needs (`bnb.game.rules.bullet_slots_for`). Fire rates are
  whole shots per second a whole number of ticks apart.
- The scripted agent's characteristics are re-measured for each exact card. A trained
  agent plays a card or lineup it was not trained with, with a warning.
- A new match or a restart keeps the tuning (`MatchSetup.tuning`) when it plays the same
  ship set; a match of another set starts untuned. A tuned match is not recorded: a replay
  re-simulates with the numbers it started from.

**The time scale** (the panel's *Time scale*, `Tuner.time_scale`) makes the whole game k
times faster with the map's lengths fixed, by the units of every quantity
(`bnb.game.handling.similarity`): speeds x k, accelerations and gravity x k^2, rates and
fire rates x k, times / k, energies x k^2 and specific powers x k^3. Energies scale with
gravity, so a store keeps its height (energy / g): a time-scaled game is an exact
similarity of the original, up to the tick, which stays at 30 Hz, and the values held to
whole ticks or Hz (bullet lives, clocks, shield delays, fire rates), which snap; the
panel's message lists what snapped. The rule settings change through
`api.tuning.tuned(..., bounded=False)` (a scaled value may lie past a slider's end); the
card is transformed in closed form, with the shared values no setting holds (the
exchanges, the bullet spread, the shield recharge) scaled through the card's per-type
scales; then the handles are re-measured. **Make this speed 1×** takes the playback speed
(×¼ to ×8, `bnb.play.loop.TIME_SCALES`) as k and sets playback back to ×1, so the match
looks the same and plays at real time.

**Ship types** live in tracked files, one per type, in ship sets (`bnb.game.ships`; see
[The ship card](../game/ship-card.md#ship-sets)): the handles the type flies and the card
the game reads, in the order of `order.toml`. A match plays its config's set
(`GameConfig.ship_set`), whose default card is read from them. The panel's **Save ships**
writes the match's set, and the tuned flight constants and bullet drag with it (into the
set's `rules.toml`: the rules its cards fly under); **Add** copies the chosen type under a
new name and **Remove** deletes one (after a confirmation popup; the last type stays),
each starting a new match with the lineup remapped.

**Maps** (`bnb.play.maps`) are the games the shell starts by name: the game presets
(`bnb.game.api.PRESETS`, read-only) and the saved maps, one JSON file each in the user's
configuration directory or in the project (`bnb.play.maps.PROJECT`, beside the ship sets in
`src/bnb/game/`, tracked in git). A saved
map is a whole game: a `GameConfig`, a lineup by type name (beside the config, as
`bnb.game.api.LINEUPS` keeps the presets', since a policy record treats a config field it lacks as
another game) and a `Tuning` (the rule settings, the changed cards by type key, the time
scale), the tuned arena radius, clocks and kills to win folded into the config. They
replace the tuning panel's former rule presets: a map made from a tuned match carries the
same numbers. The menu's Maps page (`bnb.play.interface.map_editor`) makes, edits, saves,
renames and deletes them; `bnb play --preset big-sky` starts the saved map `big-sky`.

**New match** offers one map per game preset's map (`bnb.play.interface.menus.maps`: a
preset's name without its mode's word, so Standard, Wide, Real scale and Arena), then the
saved maps (each in its own mode, with its tuning), and a dropdown per lineup slot listing
the map's ship set's types by name, five to a row (`bnb.play.interface.lineups`).

## Themes

A theme is a TOML file holding every colour the renderer uses, as hex codes, plus two
numbers (`bnb.play.themes.Theme`). The shipped themes, in `src/bnb/play/themes/`, are
read-only; the user's live in a per-user configuration directory
(`bnb.play.themes.Library`). The colours reach the shaders as uniforms each frame, so a
change shows on the next frame. The theme editor (`bnb.play.theme_editor.ThemeEditor`)
edits them live and saves new theme files to the user's directory.

## The keyboard pilot

`bnb.controllers.human.human_action` maps an input array, in `bnb.controllers.human.INPUTS`
order, to one ship's action. It is a pure function: the interface fills the array from the
keyboard each frame, and the tests check that every power, stick level and shoot is
reachable. A pilot flies through the same action contract as any other controller.
