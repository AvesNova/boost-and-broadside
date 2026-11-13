---
covers: [src/bnb/play/app.py, src/bnb/play/camera.py, src/bnb/play/render/grid.py, src/bnb/play/render/contours.py, src/bnb/play/render/bullets.py, src/bnb/play/window.py, src/bnb/play/interface/, src/bnb/play/sides.py, src/bnb/play/replay.py, src/bnb/play/themes.py, src/bnb/play/themes/, src/bnb/play/theme_editor.py, src/bnb/play/overlays.py, src/bnb/play/predictions.py, src/bnb/play/render/predictions.py, src/bnb/play/tuning.py, src/bnb/play/tuning_panel.py, src/bnb/play/maps.py, src/bnb/game/ships/]
---

# Playing and spectating

The play shell opens a window on a match: watch it, fly any ship, pause and step it, play
it against a trained agent, and replay it afterwards. Everything has a control on screen;
the keys are optional shortcuts, and each control's tooltip names its key. How it works is
in [The play shell](../explanation/in-progress/play-shell.md).

![A Frontline match, with every card open](../images/play/interface-frontline-1920.png)

## Start

<!-- not-runnable: opens a window -->
```bash
uv run bnb play
uv run bnb play --preset frontline --view team0 --seed 3
uv run bnb play --preset real_scale_frontline
uv run bnb play --preset arena
uv run bnb play --preset clover
uv run bnb play --agent1 runs/<run>/policy
```

The first match comes from the options (a preset, `real_scale_deathmatch` by default, and
the sizes that override it); every later one from the menu's **New match**. A new kind of
match starts at once and never stalls the window: the first time a game (or a changed
version of the code) is played, its tick and the scripted agent's step are built in
separate processes (about 30 s and 15 s) while the match runs them uncompiled, slower
(about 14 ms a tick instead of 1.5, so a frame that holds a tick can miss the
display's); both are kept in the user's cache directory, so every later match of that game starts
compiled. The options:

<!-- generated: play-options -->

## The screen

- **The top bar** (always shown): the menu button, the view selector, the
  **Overlays** button (see [Prediction overlays](#prediction-overlays)), **True size**
  (see [Ship size](#ship-size)), **Grid** and **Contours** (see [The camera and the grid](#the-camera-and-the-grid)) on the left; the
  score, or Frontline's front and zones (captioned "Front" and "Zones", each zone numbered
  in its owner's colour, with a tooltip saying what it shows), and the clock in the middle; the speed controls
  on the right: pause or play, one tick while paused, and ×1, ×2, ×4, ×8. Left of the
  speed controls, the frame rate: frames per second and the frame time, smoothed over about
  half a second, and the longest frame of the last second (`max`), so a stall shows. F4,
  **Settings** → **Frame rate** and `--no-fps` hide it (offscreen renders never draw it).
- **The fleets card** (left): each team's ships, with type, shield, energy, alive or the
  respawn wait, and who controls it. Click a row to select the ship. Each team's
  controller is chosen here (see below).
- **The selection card** (right): what you selected. A ship shows its live state (shield,
  energy, speed (at ×1 under a [time scale](#tuning-the-game)), angle of attack, power and stick, the last shot), kills and deaths, and
  the buttons **Take control** and **Follow**; a field its refractive index and core; a
  zone its owner, progress and the ships contesting it.
- **The minimap** (lower right): the whole arena, with the camera's view as a rectangle.
  Click or drag on it to move the camera.
- **The kill feed** (upper right): the last few kills, in team colours.

Each card's chevron collapses it to a slim tab on its edge; the tab opens it again.
Clicking a ship, field or zone on the battlefield, on the minimap or in a card selects it;
dragging the battlefield pans and the wheel zooms.

![The interface at 1280 × 720](../images/play/interface-deathmatch-1280.png)

## Views

The view selector (or V) chooses:

- **Spectator:** no fog; every ship and every bullet is drawn.
- **Team 0** and **Team 1:** what that team sees. Fog of war is on by default
  (`--no-fog` turns it off): what the team cannot see is darker, its colours kept, and
  only the ships the team sees are drawn, with the parts of their trails it saw. The
  minimap shows the same. Every bullet is drawn in every view, above the fog at full
  brightness; agents never see bullets. In a team view only what the team sees can be
  selected.

Without fog of war every ship is visible to both teams (see
[Perception and fog](../explanation/game/perception.md)), so a team's view looks like the
spectator's.

A bullet draws as a tracer: a round head in its team's colour where it is, between ticks
too (it flies straight within a tick), and a short trail behind it in the team's trail
colour. The trail leaves the head as wide as the head (5 px) and tapers and fades to a
thin tail (0.5 px) as a ship's trail does. It spans the last 0.04 s of the
bullet's flight as the screen shows it (times the playback speed), never back past where
it was fired, so a burst reads as continuous streaks at any time scale or playback speed.
A bullet has no size in the game, so the tracer is a few pixels wide at every zoom,
**True size** included.

**Settings** → **Bullets** changes the look while you play: **Tracer** (the above) or
**Trail only** (the fading trail, no head), the head's size (its radius, tracer only), the
trail's width at the head and at its tail (screen pixels) and its length (seconds). They
are part of the theme (see Settings and themes): they save with it. The options override
the theme's for the session: `--bullet-style trail`, `--bullet-head-size`,
`--bullet-trail-head-width`, `--bullet-trail-tail-width`, `--bullet-trail-seconds`.

## Ship size

A ship is drawn as a silhouette inside a ring at its hitbox, whose radius is half the
type's wingspan (the ship card's `hit_radius`). Zoomed out, the ring is never smaller than
14 pixels in radius, so every ship stays visible. **True size** in the top bar (or Z) draws
the ships at their size in metres at any zoom instead: the ring at the hitbox and the
silhouette grown to reach it from wingtip to wingtip, so the drawn span is the real one.
Trails, ghosts and the selection's outline follow it. Zoomed out, a true-size ship is a
few pixels across; clicking still selects it from as far as the default size reaches. The
setting is the theme's `true_size` (the option is `--true-size`).

## The camera and the grid

Where no ship, field or zone is near, three things show how the view moves.

**The look-ahead.** A camera that follows a ship (the flown one, or one you **Follow**)
leads it: the middle of the view sits ahead of the ship along its velocity, about where it
will be 4 s on, never more than 35% of the way from the middle to the nearer edge of the
window, so you see what you are flying into. The lead eases with a critically damped
spring (about 2 s to settle, no overshoot, the same at any frame rate), so a turn or a
change of speed slides the ship across the view gently. It is held in pixels, so a zoom
leaves the ship where it is on screen, and the lead then eases to the new zoom's (4 s of
travel is more pixels zoomed in). **Settings** → **Look ahead** chooses **Always** (the
default), **Flying** (only the ship you fly) or **Off** (the ship stays in the middle);
it is the theme's `look_ahead` (the option is `--look-ahead`). Panning, the minimap or **Home** stop following, as before.

**The dot grid.** Faint dots fixed to the map, inside the arena, under everything else
(they scroll with the world, so speed reads from them). Three patterns: **Square**,
**Triangle** (each dot with six equidistant neighbours; the default) and **Hex** (the
corners of a honeycomb). As you zoom, the spacing steps by the pattern's factor (×2 for
square and triangle, ×3 for hex) and the finer dots fade out across a step, so the density
on screen stays about the same, as in a map app. **Grid** in the top bar (or G) goes to the
next pattern, **Settings** → **Dot grid** picks one, and `--grid off|square|triangle|hex`
overrides the theme's pattern for the session. The pattern is the theme's `grid_pattern`; the dots' colour and opacity are the theme's `grid` (alpha: how faint).

**The contours.** Faint lines at equal distances from the map's shapes, as a topographic
map draws heights: around the fields, parallel to the map edge, around the zones, inside
each shape and outside it (the edge's lines continue beyond the arena). Where two shapes'
lines meet they join in one rounded line instead of a sharp corner (a smooth union of the
distances, rounded over about 8% of the arena radius), so close fields read as one hill.
The spacing steps by ×2 with the zoom, as the grid's does, and every fifth line is a little
stronger. The lines draw over the fields' fill and under the zones, trails, ships and
bullets. **Contours** in the top bar (or O) goes to the next set of shapes, **Settings** →
**Contours** picks one: **Off** (the default), **Fields**, **Edge**, **Fields + edge**,
**All** (fields, edge and zones) or **Zones**; `--contours
off|fields|edge|fields+edge|fields+edge+zones|zones` overrides the
theme's `contour_shapes` for the session. They can be on with
the dot grid. Their colour and opacity are the theme's `contours`.

## Flying a ship

Select a ship and press **Take control**. Its side keeps controlling every other ship. The
camera follows it, zoomed in from the whole-arena view, every card stays as it was (open or
collapsed), and a small HUD at the bottom shows the shield, energy, power and stick:

- W and S: boost and reverse (neither: coast);
- A and D: turn left and right, Shift for the full turn, Ctrl for the lightest;
- Space: shoot.

The shield recharges only while you fly quiet: coasting, the stick straight or at the
lightest turn (Ctrl), and not shooting. Any other key, like a hit, restarts the delay
before it: the shield comes back only after the whole delay of quiet flight
([Physics](../explanation/game/physics.md)).

**Release** (in the HUD or the selection card) hands the ship back.

![Flying a ship](../images/play/interface-flying-1920.png)

## New match and trained agents

The menu (the top bar's menu button, or Escape) pauses a live match and offers **Resume**,
**Restart**, **New match**, **Maps**, **Replays**, **Save replay**, **Tune the game**,
**Settings** and **Quit**.

**New match** chooses the mode, the map (**Standard**, **Wide**, **Real scale**: the
[real-scale variant](../explanation/game/real-scale.md), where a metre is a metre and the 27
researched types are late-war fighters from the P-51D to the Me 262 and a few invented
designs (with any type a player added after them), on a
map three times the standard one; or **Arena**: a 2v2 dogfight of a P-51D and a Spitfire
XIV a team on a 1950 m map with no fields, a ship shot down back in the fight on the next
tick, the `arena` preset in [Modes](../explanation/game/modes.md#clocks-and-presets); then
your [saved maps](#making-and-saving-maps), each in its own mode).
Choosing a map takes its fleet size, fields and lineup (and a saved map's mode and tuning); then the ships a side, the fields
(0 to 64), the lineup (a dropdown per
slot of the map's ship types, the same for both teams; **Default** alternates the P-51D and
the Spitfire XIV, or a set's first two types where it has not both; the map starts at **Real
scale**), fog, the seed, and each side's controller: **Scripted agent r9m2** (the default:
the scripted agent, `bnb.controllers.scripted.Scripted`:
it plays the mode's objectives, remembers each enemy it saw and dead reckons it while
hidden; r9m2 flies r9 with the maneuver doctrine elements and the team-gated duel extension,
`src/bnb/play/doctrines/r9m2.json`, from maneuvers-2's `champion2.json`), **Scripted agent
r9** (the stored doctrine of training, the BC teacher and the ratings' anchor,
`scripted_doctrines.json`;
`--doctrine0 r9` / `--doctrine1 r9`; a doctrine applies to the ship types it holds one for,
the Yongma and the Jujak), **Chaser** (`--scripted0 scripted`: it turns toward the nearest visible enemy and
fires, and reads no zones) or a trained agent. The scripted agent and the chaser need their
ship card's characteristics, measured once per card in a separate process (about 15 s for
the standard set and about 2 minutes for the 27 real-scale types on a laptop CPU) and kept
in the user's cache directory; until they are measured the side plays on as it was (in a
new match its ships coast straight and hold their fire) and a notice says it will take
over. An agent that runs on the GPU captures its step
aside the same way before it plays. The agent list is every policy file of the local runs:
`runs/*/policy/`, the training checkpoints and the league's snapshots (`--runs` sets the
directory). A policy plays only the game it was trained on, with the same ship card and the
default lineup; otherwise the shell says what differs and keeps the match as it was. A
policy trained on a mixed batch records the batch's first game (frontline), so the shell
plays it there; [`bnb figures video`](figures.md#videos) plays it in any mode.

An agent runs on the GPU when there is one (the CPU otherwise) at one decision per tick, with its memory kept from tick to tick
as in training. Its card row has a temperature: 1 samples the policy as trained, lower is
more deterministic, 0 always takes its most likely action. A team's controller can also
be changed in the fleets card during a match.

![New match](../images/play/interface-new-match-1280.png)

## Making and saving maps

A map is a whole game: the mode, the ship set, the ships a side, the fields, the arena
radius, the match length, the respawn wait, team deathmatch's spawn groups and kills to
win, the lineup, and optionally a tuning (rule settings such as gravity, changed ship
cards, the time scale). The menu's **Maps** page makes and edits them:

- **Start from** chooses what the form starts as: **This match** (its game and lineup, and
  its tuning, the tuning panel's changes included), a game preset, or a saved map.
- The form changes any of the game's numbers. The match length and the respawn wait are
  seconds at ×1, held to whole ticks (1/30 s); left at the mode's own, they follow the
  mode. Spawn groups are typed as `2, 2, 1` (empty: uniform spawns) and must sum to the
  ships a side. A tuning that came with the form is named in one line; **Drop** removes it.
- **Name** (lowercase words joined by hyphens, as `big-sky`; never a game preset's) and
  **Saved**: **Yours** (one JSON file per map in your configuration directory,
  `maps/<name>.json` beside the themes) or **In the project** (`bnb.play.maps.PROJECT`:
  `maps/` beside the ship sets in `src/bnb/game/`, tracked in git; the place for a map to
  keep with the code, to review in a diff or, later, to train on).
- **Save** writes the form under the name. When the form started from a saved map and the
  name is still its own, Save replaces it (edits it; it moves when **Saved** changed);
  otherwise it makes a new map and refuses a name already saved. **Rename** gives the chosen
  saved map the typed name, in its place. **Delete** removes it, after a popup asks.
  **Play** starts a match of the form, saved or not.

The tuning panel's **Save as a map** opens the page on this match. A saved map shows on
**New match** after the game presets' maps, in its own mode, and starts from the command
line by name:

<!-- not-runnable: opens a window -->
```bash
uv run bnb play --preset big-sky
```

A saved map's tuning plays instead of the current match's (a game preset's map keeps the
current match's tuning, as before). The file holds the config's fields, the lineup by type
name, and the tuning where there is one: the rule settings and the time scale's factor at
×1, and the ship cards it changes by type key (only the fields that differ from the type's
file), so a map stays valid when a type is added to the set. Training does not read saved
maps yet.

## Prediction overlays

The **Overlays** button opens a submenu that draws, on the battlefield and while the match
plays, what an agent thinks beside what is true. Each overlay is on or off on its own, and
the submenu lists only those the sides being played can provide (`bnb.play.overlays`):

| Overlay | From | Draws |
|---|---|---|
| True bullet paths | the game, every match | each live bullet's flight ahead through the fields, by the game's own transport (`bnb.game.api.bullet_paths`, the learner's re-simulation of a shot) |
| Predicted shot path | a Yemong agent | the selected ship's next shot: the shot head's mean at each age from its launch point, a one-sigma circle at each age |
| Next-state ghosts | a Yemong agent | each ship's predicted pose one tick on, a translucent silhouette with its one-sigma circle |
| Belief ghosts | a Yemong agent | each hidden enemy's believed pose, with one- and two-sigma circles |
| Dead-reckoning ghosts | the scripted agent | each remembered enemy it cannot see, at its recalled position, with a line back to its last sighting; its opacity is the agent's weight on it (the memory states no sigma) |
| Density | a Yemong agent | the density head's expected ships per hex cell |
| Enemy action | a Yemong agent | in the selection card, for a selected enemy: the power, stick and shoot marginals, the action it took outlined |
| Value and outcome | a Yemong agent | in the selection card, for a selected own ship: the critic's value per value level, the outcome head's P(win), P(tie), P(loss) and P(unresolved) (a critic with one), and the next shot's P(enemy hit) |

![A fight: the true bullet paths, the selected ship's predicted shot and its critic](../images/play/overlays-fight-1600.png)

The chaser against the chaser offers the bullet paths only; a relational agent
provides nothing; a replay runs no controllers, so it offers the bullet paths only.

**Whose.** Each side's agent has its own predictions. An overlay draws the side picked
under **Whose**, or by default the selected ship's side when it provides that overlay,
else the other side (with nothing selected, the view's team first). Each entry names the
team it draws.

**Show the truth** adds the truth: a ring at each ghost's true position joined to the
ghost, and the density's **True** and **Error** maps (predicted minus true: over in the
theme's prediction colour, under in its truth colour). **Density** picks the predicting
side's enemy or ally cells. The truth draws in every view, so in a team view it shows
ships the team cannot see.

The predictions are those of the side's latest decision, made from the observation of the
tick before the one drawn, and decoded by the decoder layer as the figures decode them
(`bnb.models.yemong.decode`, see [Figures](figures.md)); the truth beside a ghost is that
tick's (for the next state, the tick it forecasts). While one of its overlays is on, a
Yemong agent decides with `Yemong.inspect`, the same decision with every head's output
from the same trunk pass, so the overlays do not change its play; an agent on a GPU then
steps eagerly instead of from its CUDA graph. Off, the overlays cost nothing.

With a Yemong agent and the scripted agent on the CPU (one torch thread, the tick
uncompiled), every overlay on adds about 5 ms a frame at ×1 (the predictions are made once
a tick: about 4 ms for the agent's heads and 4 ms for the bullet paths). Without overlays a
Yemong agent's decision takes about 2.2 ms a tick on a laptop GPU (its CUDA graph) and
6.6 ms on one CPU thread; the scripted agent's, built, 0.2 ms.

`--overlays`, `--overlay-side` and `--truth` turn them on from the start:

<!-- not-runnable: opens a window, and needs a policy file -->
```bash
uv run bnb play --agent0 runs/<run>/policy --scripted1 scripted_agent \
    --overlays bullet_paths belief density --truth --view team0
```

![Belief ghosts against the truth, and the density](../images/play/overlays-belief-density-1600.png)

## Replays

Every live match is recorded and saved when it ends, or when you quit, to
`replays/<date>-<mode>-<sides>/` (`--replays` sets the directory); the newest 50 are
kept. **Save replay** in the menu saves the match so far. **Replays** lists them; open
one, or start the shell on one with `--replay replays/<name>`.

A replay plays back by re-simulating the recorded actions, so it is exactly the match.
The bottom strip has a tick back, play or pause, a tick on, and the timeline: drag it to
seek, or click a marker (kills in the killer's colour, captures, the match's end) to watch
the seconds before it. Every view, selection and card works in a replay. **Take control**
in a replay starts a live match from that moment, with you flying the ship; the replay
itself is untouched.

![A replay with its markers](../images/play/interface-replay-1920.png)

## Settings and themes

**Settings** chooses the theme, and edits the look the theme holds (below): the dot grid,
the contours, the look-ahead, the bullets' look. It also sets fog in new matches, the
interface scale (0.75 to 2; the
layout works from 1280 × 720 to 2560 × 1440) and shows the key list. The shipped themes
are in `src/bnb/play/themes/`; your own live in your configuration directory
(`bnb.play.themes.USER`; on Linux, ~/.config/boost-and-broadside/themes).

A theme is the whole look: its colours and the look settings (ships at true size, the dot
grid's pattern, the contours' shapes, the look-ahead, the bullets' look). Settings and the
top bar's **Grid**, **Contours** and true-size buttons edit the current theme's values
live; **Save as theme** (in the theme editor) writes all of them with the colours, so
they come back when the theme is chosen again (`--theme NAME`). Match settings (the preset,
fog, agents, seed, tuning) and the interface scale are not part of a theme. A look option
(`--grid`, `--contours`, `--true-size`, `--look-ahead`, `--bullet-*`) overrides the
loaded theme's value for that session only, never the file; choosing another theme in
the session uses that theme's own values. A theme file may omit any look setting: it
takes its default (ships at the minimum size, triangle grid, no contours, always look
ahead, the tracer), so themes saved before the look settings existed load and look the
same.

<!-- generated: play-themes -->

**Edit** (or E) opens the theme editor, with every colour of the current theme, the
cards' own included:

- Each colour shows its hex code. Click it and type a new one (`#1F3A93`, or `#RRGGBBAA`
  for the colours whose alpha matters); the swatch next to it opens a picker.
- A change draws on the next frame.
- **Save as theme** writes the colours to your themes as `<name>.toml`, under the name in
  the name field (lowercase words joined by hyphens). The shipped themes are read-only,
  so a shipped name is refused, and replacing one of your own themes asks first.
- **Revert** restores the colours and looks from the theme's file.

To make a theme by hand, copy a file from `src/bnb/play/themes/` into your themes and
change its values; a file must list every colour and number (the look settings may be
left out), and `bnb.play.themes.load` names any
value that is missing, unknown or malformed.

## Tuning the game

**Tune the game** in the menu (or F2) opens the tuning panel over a live match, at the
right below the top bar; drag or resize it like any window, and scroll it when it holds more
than fits. Every change takes effect while the match plays; pause first if you want to
compare.

- **Ships**: choose a ship type at the top. Each handle (cruise speed, sustained turn rate,
  climb time, half-lift drag, gun range, shield, ...) has a slider and, beside it, a number
  box showing 4 significant digits: click the box, type a value and press Enter. Hover a
  row for its full name and unit. The ships change within a frame or two, from an instant
  estimate of the card that meets the new values; when you stop, the exact card is solved
  in the background and swapped in, seconds later. A red dot after a box marks a handle the
  ships fly more than 0.5% off its set value; hover the dot for what they fly. The fire
  rate is a choice of the rates a ship can hold (whole shots per second, a whole number of
  ticks apart). The bleed rate is shown, not set: the half-lift drag (the drag of a
  half-stick turn over a full one) sets the drag's shape.
- The sliders show as soon as the panel opens: each type's handles come from its ship
  file, where its card was solved to them. What the ships actually fly is measured in the
  background, the chosen type first, then the types in the match, so a type's red dots
  appear a few seconds after you choose it (tens of seconds for a real-scale type); until
  then the dot's tooltip says "not measured yet". Once a type is measured, a change to a
  combat handle (hit damage, fire rate, gun range, shots per store, shield, ...) is exact
  at once: it does not change how the ship flies.
- **Save ships** writes every type to its file in the match's ship set (`src/bnb/game/ships/`
  for the standard set, `src/bnb/game/ships/real-scale/` for the real-scale one), tracked
  in git: the handles and the card, and the tuned flight constants and bullet drag into
  the set's `rules.toml`, as the cards fly under them. **Add a copy as this name** writes a
  copy of the chosen type under the name in the field to its own file in the set at once
  (the panel names the file); the other types' files are left as they are, and their
  unsaved changes play on in the new match until **Save ships**. It waits only for the
  chosen type's exact card, and asks for **Save ships** first when the flight constants or
  bullet drag were changed. **Remove** deletes the chosen type after asking. Each starts a
  new match; the last type cannot be removed.
- **Rules**, by group: the arena and the clocks, the bullet drag, the flight constants every
  type shares, spawns, and the current mode's rules (Frontline's zone radius and capture
  time, team deathmatch's kills to win). Each has a slider and a number box, and applies on
  the next tick; those marked with * are drawn at an episode's start. Hover a row for
  the game's value.
- Changing a flight constant re-measures the handles (the ships keep their card, so what
  they fly changes), each type when it is chosen or plays; a type's handles cannot be set
  until it is measured, a few seconds (tens of seconds for a real-scale type). **Save
  ships** needs every type's handles: it first measures the ones not yet measured, and
  says so; press again once the status line says exact.
- **Time scale**: **Apply** makes the whole game the given factor faster than its design
  (×1) with the map's lengths fixed (speeds x k, gravity and accelerations x k^2, rates and fire rates x k,
  times / k, energies x k^2, so a store keeps its height). Fire rates snap to whole Hz that
  divide 30, and bullet lives, clocks and shield delays to whole ticks; the message says
  what snapped. **Make this speed 1×** takes the playback speed (×¼ to ×8 in the top bar)
  as the factor and sets playback back to ×1: pick the speed at which the game feels
  right, then make it the game's own. The factor is always against ×1: **Apply** 2 gives ×2
  whatever the game runs at now (2 then 2 is still ×2), and 1 goes back to the design's
  own speed. **Make this speed 1×** is the one that compounds: playback ×2 of a ×0.5 game
  makes it ×1.
- Under a time scale every value in the panel stays the design's, at ×1: the handles,
  what the ships fly (measured at ×1), the rules and the fire-rate choices, and the
  selection card's speed. An edit sets the ×1 value, and **Save ships** writes the same
  files as at ×1; the match plays the design k times faster. A line at the top of the
  panel names it ("Values at x1; the game runs xk"). Nothing is re-measured when the
  factor changes. A new match, a restart and **Add** keep the factor (a saved map plays its own); **Game's
  numbers** keeps it too.
- The clocks are the exception: they count the seconds played, what a player sees pass at
  ×1 playback. The top bar's clock and its match length, the fleets card's and the flying
  card's respawn wait, the selection card's "respawns in" and "last shot ... ago", and the
  replay scrubber are all ticks times the tick (1/30 s, which a time scale never changes).
  So under ×k a match the design makes 600 s long shows "/ 10:00" at ×1 and "/ 5:00" at ×2,
  a 10 s respawn waits 5 s, and the clock runs at the wall clock's pace. The panel's status
  line says so, and each rule's tooltip gives the value it plays as (the match length in
  the clock's seconds). The playback speeds (×¼ to ×8) are a different thing: they run more
  or fewer ticks per second of the display without changing the game, so every clock runs
  that many times faster or slower than the wall clock (at ×8 the clock gains 8 s a second;
  the kill feed's lines, 6 s of play, last 0.75 s). Paused, every clock stops.
- **Maps**: **Save as a map** opens the menu's Maps page on this match, its tuning
  included (the design's values, at ×1), to save under a name (see
  [Making and saving maps](#making-and-saving-maps)); **Game's numbers** goes back to the
  ship files' and the rules' own, at the same time scale.

A new match or a restart keeps the tuning, unless it plays another ship set. Ship counts, fields and the lineup are set in
**New match**. A trained agent given a ship card or lineup it was not trained with plays,
with a warning. A tuned match is not recorded as a replay. The first time the panel opens
on a machine (and after a flight constant changes), the measurement's and the solver's
flights compile first (about a minute); after that they come from the cache.

## Keys

The keys are shortcuts for the controls above; F1 shows this list in the window. They do
nothing while you type into a field. The window takes the keys without the desktop's
input method (IBus, for example), so a field gets the letters you type as they are.

<!-- generated: play-keys -->

## Offscreen renders

`--offscreen DIR --frames N` renders N frames without a window, through an EGL context,
and writes them to `DIR/frame_0000.png` onwards, with any `--overlays` drawn. The match runs on a fixed clock, so a
seed gives the same frames each time.

<!-- not-runnable: needs an OpenGL 3.3 context (a GPU driver with EGL) -->
```bash
uv run bnb play --offscreen /tmp/frames --frames 120 --preset frontline --view team1
```

## Frame time

`python -m bnb.play.benchmark` measures the frame time offscreen at each game and view,
with every card open, with the [benchmark lock](benchmarking.md) held, and breaks it down
by pass and by stage (`--no-interface` measures the battlefield alone; `--contours
SHAPES` draws the contours, timed as their own pass).

<!-- not-runnable: needs the GPU and the benchmark lock -->
```bash
uv run python -m bnb.play.benchmark --games deathmatch frontline --seconds 10
```
