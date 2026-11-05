---
covers: [src/bnb/play/app.py, src/bnb/play/window.py, src/bnb/play/interface/, src/bnb/play/sides.py, src/bnb/play/replay.py, src/bnb/play/themes.py, src/bnb/play/themes/, src/bnb/play/theme_editor.py]
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
uv run --no-sync python -m bnb.play
uv run --no-sync python -m bnb.play --preset frontline --view team0 --seed 3
uv run --no-sync python -m bnb.play --agent1 runs/<run>/policy
```

The first match comes from the options (a preset, and the sizes that override it); every
later one from the menu's **New match**. A new kind of match starts at once: its tick
compiles in the background while the match runs the uncompiled tick, a little slower, for
the first several seconds. The options:

<!-- generated: play-options -->

## The screen

- **The top bar** (always shown): the menu button and the view selector on the left; the
  score, or Frontline's front and zones (captioned "Front" and "Zones", each zone numbered
  in its owner's colour, with a tooltip saying what it shows), and the clock in the middle; the speed controls
  on the right: pause or play, one tick while paused, and ×1, ×2, ×4, ×8.
- **The fleets card** (left): each team's ships, with type, shield, energy, alive or the
  respawn wait, and who controls it. Click a row to select the ship. Each team's
  controller is chosen here (see below).
- **The selection card** (right): what you selected. A ship shows its live state (shield,
  energy, speed, angle of attack, power and stick, the last shot), kills and deaths, and
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

## Flying a ship

Select a ship and press **Take control**. Its side keeps controlling every other ship. The
camera follows it, zoomed in from the whole-arena view, the cards collapse (their tabs open them), and a small HUD at the
bottom shows the shield, energy, power and stick:

- W and S: boost and reverse (neither: coast);
- A and D: turn left and right, Shift for the full turn, Ctrl for the lightest;
- Space: shoot.

**Release** (in the HUD or the selection card) hands the ship back.

![Flying a ship](../images/play/interface-flying-1920.png)

## New match and trained agents

The menu (the top bar's menu button, or Escape) pauses a live match and offers **Resume**,
**Restart**, **New match**, **Replays**, **Save replay**, **Settings** and **Quit**.

**New match** chooses the mode, the map (standard or wide), the ships a side, the
fields, the lineup (each slot's ship type, the same for both teams), fog, the seed, and
each side's controller: **Scripted** (the chaser) or a trained agent. The agent list is
every policy file of the local runs: `runs/*/policy/`, the training checkpoints and the
league's snapshots (`--runs` sets the directory). A policy plays only the game it was
trained on, with the same ship card and the default lineup; otherwise the shell says
what differs and keeps the match as it was. A policy trained on a mixed batch records the
batch's first game (frontline), so the shell plays it there;
[`bnb figures video`](figures.md#videos) plays it in any mode.

An agent runs on the GPU when there is one (the CPU otherwise) at one decision per tick, with its memory kept from tick to tick
as in training. Its card row has a temperature: 1 samples the policy as trained, lower is
more deterministic, 0 always takes its most likely action. A team's controller can also
be changed in the fleets card during a match.

![New match](../images/play/interface-new-match-1280.png)

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

**Settings** chooses the theme, fog in new matches, the interface scale (0.75 to 2; the
layout works from 1280 × 720 to 2560 × 1440) and shows the key list. The shipped themes
are in `src/bnb/play/themes/`; your own live in your configuration directory
(`bnb.play.themes.USER`; on Linux, ~/.config/boost-and-broadside/themes).

<!-- generated: play-themes -->

**Edit** (or E) opens the theme editor, with every colour of the current theme, the
cards' own included:

- Each colour shows its hex code. Click it and type a new one (`#1F3A93`, or `#RRGGBBAA`
  for the colours whose alpha matters); the swatch next to it opens a picker.
- A change draws on the next frame.
- **Save as theme** writes the colours to your themes as `<name>.toml`, under the name in
  the name field (lowercase words joined by hyphens). The shipped themes are read-only,
  so a shipped name is refused, and replacing one of your own themes asks first.
- **Revert** restores the colours from the theme's file.

To make a theme by hand, copy a file from `src/bnb/play/themes/` into your themes and
change its values; a file must list every value, and `bnb.play.themes.load` names any
value that is missing, unknown or malformed.

## Keys

The keys are shortcuts for the controls above; F1 shows this list in the window. They do
nothing while you type into a field.

<!-- generated: play-keys -->

## Offscreen renders

`--offscreen DIR --frames N` renders N frames without a window, through an EGL context,
and writes them to `DIR/frame_0000.png` onwards. The match runs on a fixed clock, so a
seed gives the same frames each time.

<!-- not-runnable: needs an OpenGL 3.3 context (a GPU driver with EGL) -->
```bash
uv run --no-sync python -m bnb.play --offscreen /tmp/frames --frames 120 --preset frontline --view team1
```

## Frame time

`python -m bnb.play.benchmark` measures the frame time offscreen at each game and view,
with every card open, with the [benchmark lock](benchmarking.md) held, and breaks it down
by pass and by stage (`--no-interface` measures the battlefield alone).

<!-- not-runnable: needs the GPU and the benchmark lock -->
```bash
uv run --no-sync python -m bnb.play.benchmark --games deathmatch frontline --seconds 10
```
