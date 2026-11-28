# lean/sandbox

The flight sandbox's world, without drawing. You fly it in the browser (`lean/tool/NOTES.md`).

`World` (`lean/sandbox/world.py`) is the game at batch size 1 on the CPU: the player (slot 0)
and an optional opponent (slot 1), stepped by `lean.game.core.tick`, so bullets, hits,
shields, the boundary, collisions, the deflector circle, the nose armour and the fog behave as
in the game (the opponent sees the player by the game's sight rule, the card's sight egg with
the tracking circle, or on a tick the player fires, is hit or collides). The opponent is the scripted opponent
(`lean/scripted/opponent.py`, the yardstick; no memory) or the chaser
(`lean/scripted/chaser.py`, `World(opponent="chaser")`). `HitRange` (same file) is the tool's
Hit range tab: a shooter and a pinned target on the real step (`lean/tool/NOTES.md`).

The tool ticks `World` at 30 Hz in a child process (`lean/tool/session.py`) at the time scale
and with the ship tuned in the Ships tab (`lean.game.tuning.flying`); the browser draws it.
The web tool always flies against the scripted opponent; the chaser is `World(opponent="chaser")`
in Python only.

## Controls (in the browser)

| Key | Action |
| --- | --- |
| Left / Right (or A / D) | turn left / right at Max |
| + Shift | Snap (Shift wins when left Alt is held too) |
| + left Alt | Sustain (the page keeps Alt from opening the browser's menu bar) |
| Up (or W) | dive: power 1, store into speed |
| Down (or S) | climb: power 2, speed into store |
| neither | coast |
| Space | fire |
| C | opponent on / off |
| R | reset |
| + / - or mouse wheel | zoom |

## On screen

- The ship (blue) with its velocity vector (green, half a second long); the gap between nose
  and vector is the angle of attack. The trail of the last 5 s, a grid (the finest of 25 m to
  2500 m at least 40 px apart, every fourth line brighter), the arena edge (yellow) and the
  end of the damage ramp (red), and the field bands.
- The opponent (red); your bullets are yellow.
- Readout: speed (m/s over the map), turn rate and radius of the flight path, lateral g in the
  game's g, angle of attack against the stall angle, store, shield, deaths and kills.

You cannot die: a lethal hit or the boundary refills the shield and counts a death. A killed
opponent comes back 2 s later, 1500 m away, facing the player.

## Open

- Not in the browser sandbox: `GameConfig` overrides, a pause key, a speed and store strip chart,
  an off-screen opponent marker.
