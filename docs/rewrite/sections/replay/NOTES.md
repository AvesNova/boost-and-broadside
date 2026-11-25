# Replay videos

`replay.py` plays scripted-vs-scripted 4v4 games of the core game on the CPU and writes one MP4
per game (1080x1080, 30 fps, one frame per tick, H.264 yuv420p, faststart; the last frame is held
3 s with the result). Nothing in `lean/` is changed: the game is `lean.game.deal` / `tick` at batch
size 1, `GameConfig()` (the 4,200 m map with its fields, fog on, the 1,800-tick clock, the P-51D
card, the standard spawn as in evaluation; training's randomised spawn is off), and every ship is
`ScriptedConfig()` (the default yardstick), both teams in one `decide` call as in evaluation.
The seed seeds one `torch.Generator` used for the deal, the scripted memory, the action draws and
the tick.

## Re-run

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:src uv run --no-sync python \
        docs/rewrite/sections/replay/replay.py --seeds 0 1 2 \
        --out ../data/replays-2026-11-25 [--still 150 450 ...]

`--still` also saves those ticks as PNGs. `--vision-range M` sets every type's vision range to M m
for that render only (in-process card, nothing on disk changes; files `scripted4v4_visionM_seedN.mp4`).
`--vision-eccentricity E` sets every type's sight-ellipse eccentricity (lean/game/NOTES.md "Fog":
the vision range to the side, range / (1 - E) ahead, range / (1 + E) behind) for that render only
and draws each living ship's sight outline faintly in its team colour (files `..._ellipseE_seedN.mp4`);
`--draw-vision` draws the outlines without changing the card (circles at the default e = 0).
`--ships-per-team N` plays N v N instead of 4v4 (the standard spawn; files `scriptedNvN_...`).
`--no-fields` plays that render with `field_count = 0` (no fields on the map, so sight is
range only, plus the reveal on firing or being hit; files `scripted4v4_nofields_..._seedN.mp4`).
Needs matplotlib's Pillow (the `tools` extra) and
/usr/bin/ffmpeg. About 5-7 s to simulate and 20-35 s in all per game (laptop CPU, 4 threads).

## What is drawn

- Ships: triangles in team colour (blue team 0 = slots 0-3, red team 1 = slots 4-7), a 1.5 s
  trail; filled when the enemy team sees the ship this tick, hollow when it is hidden from it
  (fog). Dead ships vanish and leave an X for 1.5 s (all X's on the result frame).
- Under each ship: shield bar (white) and store bar (cyan), fractions of full; beside it the
  slot, the label and the speed in corner speeds (proper speed / corner speed).
- Label: the dominant tactic weight of `explain` for that tick's decision: SEEK (disengaged:
  no target known, or out of the engaged range), MERGE (he is ahead, I am not behind him),
  OFF (offensive: he is ahead and I am behind his 3/9 line; the chase speed), TURN (he is behind
  my 3/9 line, I am behind his) with the committed plan 1c / 2c (sign of `last_pass`; none
  before the first commit), BREAK (defensive). "A/B" when the top weight is below 0.6 (B the
  runner-up). A faint line joins a ship to its target unless it is seeking.
- Real bullets as dots (ghost bullets are not drawn); fields as discs (warm: higher index,
  slower; cool: lower); the arena edge (yellow) and the end of the damage ramp (red); a 500 m
  grid. The camera follows the living ships (at least 2.2 km wide); the minimap shows the whole
  arena and the camera's box.
