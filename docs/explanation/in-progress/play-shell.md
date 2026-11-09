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
scripted `bnb.controllers.scripted.Chaser`, or a trained agent. At most one ship, the
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
| Fields | A topographic map of the true refractive index: terraces filled by sign (thin air below ambient, thick air above), stronger with depth, and a contour at every third of a level. The core edge is drawn bold. |
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

The bold line on each core edge is drawn from the field geometry, at
`bnb.game.api.core_radius`, the radius at which the game's sight test blocks a sight line
(`tests/sections/test_game_api_display.py::test_core_radius_is_where_the_sight_test_blocks`).

### The fog mask

In a team's view the renderer first draws a visibility mask in screen space, at
`MASK_SCALE` of the target's resolution: a fragment shader applies the sight rule of
`bnb.game.perception.sight.sighted` to the world point under each texel, against the team's
living ships and the field cores. A test holds the mask to `sighted` at every texel away
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
scale: the top bar, the fleets card on the left, the selection card and the minimap on the
right, the kill feed under the top bar, the scrubber along the bottom in a replay, and the
flying HUD at the bottom while a ship is flown. A card is drawn inside its rectangle, or
as a slim tab on its edge when collapsed, so everything outside the rectangles is
battlefield; a test renders with and without the interface and checks those pixels are
unchanged. The cards are Dear ImGui windows themed from the theme file: the panel,
line and control colours are theme values like the battlefield's, so the theme editor
edits both.

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
generator. Flying one of its ships replaces that ship's action only: the agent decides for
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

A new kind of match (another mode, map size or fleet) needs its tick compiled, which
takes several seconds. `Realtime(background=True)` compiles it on a thread while the
match runs the uncompiled tick, then swaps it in; compiled ticks are shared by every match
of the same rules in the process, so a restart or a replay of the same game starts
compiled.

## Live tuning

The tuning panel (`bnb.play.tuning_panel`, F2) changes a live match's numbers while it
plays, through `bnb.play.tuning.Tuner` and `MatchLoop.retune`:

- **Rule settings** (`bnb.game.api.tuning`: the arena and clock, the bullet life, the
  shared flight constants, spawns and each mode's rules) apply from the next tick, without
  a recompile: the compiled tick takes their values as tensors (`bnb.game.tuning.live_of`) and
  is keyed on the rules with them blanked (`bnb.game.tuning.static_of`), so every value runs on
  the same compiled tick. The state carries over (`bnb.game.api.retune`: Frontline's zone
  ring is rebuilt for the new radius and ring; shields and stores are capped). The map and
  the spawns follow the rules from the next episode. Sizes (ships, fields, lineup, the
  tick) take a new match.
- **Handles**, per ship type, go through the card, also an input of the compiled tick. A
  changed handle applies the instant estimate (`bnb.game.api.Estimate`, about 50 ms, on a
  worker process) on the next frames; once the handles settle, the exact solve runs on a
  second worker from that estimate, swapping each better card in and reporting the handles
  it flies, which the panel shows beside the set values. A flight constant re-measures the
  card's handles (the card is kept); the bullet life changes only the gun range, in closed
  form.
- **The workers are processes**, not threads: the solve and the estimate are thousands of
  small tensor operations, and on a thread each waits for the interpreter lock while the
  match runs. A warm-up process compiles the flights the solver flies
  (`bnb.runtime.flights`) into the on-disk cache when the panel first opens; the workers
  switch to the compiled flights once it is done (a measurement about 1 s instead of
  12.5 s, a solve about 4 s instead of 40, on the development CPU).
- **Bullet slots.** A play-shell match allocates a slot per tick of the longest bullet life
  a setting allows (`bnb.game.tuning.most_bullet_slots`, 45 at 30 Hz), so no fire rate or bullet
  life runs out of them; the compiled 5v5 tick costs the same 0.7-1.0 ms at 6, 15 or 45.
  Elsewhere the rules hold what the card needs (`bnb.game.rules.bullet_slots_for`). Fire rates are
  whole shots per second a whole number of ticks apart.
- The scripted agent's characteristics are re-measured for each exact card. A trained
  agent plays a card or lineup it was not trained with, with a warning.
- A new match or a restart keeps the tuning (`MatchSetup.tuning`). A tuned match is not
  recorded: a replay re-simulates with the numbers it started from.

**Ship types** live in tracked files, one per type (`src/bnb/game/ships/`, `bnb.game.ships`):
the handles the type flies and the card the game reads, in the order of `order.toml`. The
default card is read from them. The panel's **Save ships** writes them; **Add** copies the
chosen type under a new name and **Remove** deletes one, each starting a new match with the
lineup remapped. Rule presets (`bnb.play.tuning.Presets`) save the rule settings and the
whole card to the user's configuration directory.

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
