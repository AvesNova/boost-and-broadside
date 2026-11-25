# lean/view

Watching lean matches in v2's look: offscreen MP4 replays and the live window.

## NEEDS REVIEW: v2's play shell copied as-is

`lean/view/play/` is v2's `src/bnb/play` copied as it is, bloat and all, through an adaption
layer (author 2026-11-26; review and trim later). The project's no-legacy and naming rules are
waived for `lean/view/play/` until that review. It still imports `bnb.*` (core's `src/bnb`, v2's
game) for its rules objects, presets, maps and panels. Edits to the copy are few and marked
`lean:` in comments:

- the fog's sight test is lean's sight ellipse: the observers' fourth value is the heading
  (v2: unused) and the eccentricity a uniform (`render/fog.py`, `render/sight.frag.glsl`,
  `render/renderer.py`, `render/minimap.py`); at e = 0 it is v2's circle.
- the `NEEDS REVIEW` line in `play/__init__.py`.

## The adaption layer (lean code)

| Module | What it is |
|---|---|
| `snapshot.py` | `snapshot(config, state, ...)`: one lean environment as numpy (ships, bullets in v2's slot layout, fields, the tick's deaths, kills and collisions, the result). The only place, with `index_map` and `match`, that reads lean's `State`. `to_v2(frame)` fills v2's `play.snapshot` dataclasses (no zones, no respawns, boundary fixed). |
| `index_map.py` | ln n on a grid from `lean.game.core.medium` (for the video; the live window uses v2's own builder, the same formula, built a few tiles per frame). |
| `match.py` | `MatchSettings` (sizes, seed, fields, tracking, vision range and eccentricity), `Match` (scripted yardstick against itself, batch 1, CPU), `record`. |
| `marks.py` | Lean's extras over v2's battlefield: sight ellipse outlines, tracking circles, death crosses, collision rings (v2's arcs and lines programs). |
| `interface.py` | The video's HUD in v2's ImGui style: top bar (view, ships left, clock, title), kill feed, minimap card, collision labels, the result card. |
| `video.py` | Offscreen MP4 + PNG stills: EGL context, v2's `Renderer`, `marks`, `interface`, ffmpeg. |
| `live.py` | The live window: v2's `Window` and `Session` (`LeanSession`) on a `LeanLoop`; the lean game runs in a child process and the window plays its ticks back like a replay. `--offscreen DIR --frames N` renders the session's frames to PNG (a check without a display). |

Commands (from the repository root, `PYTHONPATH=.:src`, `uv run --no-sync`):

    python -m lean.view.video --ships-per-team 4 --seeds 0 --out DIR [--no-fields]
        [--view spectator|0|1] [--vision-eccentricity E] [--vision-range M]
        [--tracking on|off] [--draw-vision] [--theme NAME] [--stills 0.25 0.5 1.0]
    python -m lean.view.live --ships-per-team 4 --seed 0 [--fields 0] [--view team0]
        [--vision-eccentricity E] [--draw-vision] [--theme NAME]
    pytest lean/view -q

Dependencies: the `view` extra (moderngl, imgui-bundle, glfw, pillow; also core dependencies
today) and the ffmpeg command line tool.

## What works through the adapter

The video: v2's sky, hatched boundary, fields (topographic fill, contours, band edges), the
distance contours (the theme's default), Komet silhouettes with shield and energy rings, trails
and wingtip vortices, tracers, fog per team (lean's ellipse), the minimap, the three themes.
Lean's extras: sight ellipses and tracking circles (`--draw-vision`), death and collision marks
with the collision damage.

The live window: all of v2's session that does not drive v2's game: views, theme switch and
theme editor, true size, grid, contours, follow and zoom (v2's camera), the fleets card, the
selection/inspector card, the kill feed, the minimap, pause, step, x1/4 to x8, the scrubber
with kill markers and seek, the menus' New match (sizes and seed).

## Turned off (each needs more than an adapter)

| Feature | Why, and what adapting it would take |
|---|---|
| Prediction overlays (all, including True bullet paths) | They read v2's realtime game and Yemong's heads (`bnb.models.yemong.decode`). Lean's policy has next-state logits (6 bins per ship), hit logits and value logits: next-state ghosts and the value card could be decoded from them by running the lean network for a side in the game process and sending its outputs with each snapshot (about half a day); shot path, belief, density, enemy action and dead reckoning have no lean head. True bullet paths need lean's `bullet_motion` run forward in the game process (small). |
| Tuning panel ("Ships") | Retunes v2's realtime rules and card live; lean's card is derived by `lean.game.tuning`: a new panel over lean's design inputs, or the web tool's. Large. |
| Map editor / map panel | Edits v2's maps and modes (zones, shrink, spawns); lean has one fixed map and random fields per deal: adapting means a lean map format and deal from it. Medium. |
| Flying a ship | Needs the lean game stepped in step with the window with the human's action (the game now runs ahead in a child process): a command channel to the game process. Medium. |
| Changing sides (agent / Imugi / chaser) | v2's controllers; lean has the yardstick, the chaser and networks (`net:` checkpoints): a lean side setup sent to the game process. Small to medium; policy against scripted is the natural first step. |
| v2 replays (record, open) | v2 replays re-simulate v2's game; the live window keeps every lean tick in memory instead. Saving lean frames would be small. |

Zones: lean has none (no frontline, no respawn waves, no shrink): the zone pass draws nothing,
the top bar shows "ships left" and "edge fixed".

## Numbers

Offscreen 1920x1080 on the laptop's 4070 (EGL), CPU game: a 4v4 match of about 1,050 ticks
simulates in 4-5 s and renders and encodes in about 10 s more.

## Open

- The review of `lean/view/play/` (trim to what lean uses, the naming and no-legacy rules).
- The prediction overlays on lean's model (above).
