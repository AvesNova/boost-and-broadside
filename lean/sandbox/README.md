# Flight sandbox

A quick, rough sandbox for flying one ship by keyboard in a top-down view. It flies the game
(`GameConfig()` with no clock: `lean.game.core.step` and so `flight_tick`), so a setting
changed with `--set` shows up here without changes. One player ship and an optional chaser
(`lean/scripted/chaser.py`) run through the full game step at batch size 1 on the CPU, so
bullets, hits, shields, the boundary and the fog all behave as in the game (the chaser sees
the player within 1,000 m, or on a tick the player fires or is hit).

Drawn with Tkinter from the standard library: nothing to install.

## Run

```bash
PYTHONPATH=.:src uv run --no-sync python -m lean.sandbox
PYTHONPATH=.:src uv run --no-sync python -m lean.sandbox --no-chaser --set field_count=0
PYTHONPATH=.:src uv run --no-sync python -m lean.sandbox --time-scale 1
```

The ship is the P-51D (`--ship` picks a card type). `--time-scale K` flies at K times real
time (default 2, the game's): lengths stay real, speeds and turn rates are x K, accelerations
and gravity x K^2, power per kg x K^3; the card is rebuilt from its real-time values by
`lean.game.tuning.time_scale` for this process only (training never uses it). At K = 1 the
readout's speeds and turn rates are real ones. Guns scale too; a bullet's life is capped at
what the 11 bullet slots hold at the fire interval (at 1.5x: 11 ticks, not 15), and the fire
interval is whole ticks (at 3x: one shot a tick, 30 a second, not 45). The `time` line of the
readout shows K.

`--set NAME=VALUE` overrides any `GameConfig` field (repeatable; the value is read as a
Python literal when it parses). Every non-default field is listed on the `physics` line of
the readout, for example `--set fog=False`.

## Controls

| Key | Action |
| --- | --- |
| Left / Right (or A / D) | turn left / right at the full stick level |
| + Shift | sustained level |
| + Ctrl | fine level |
| Up (or W) | dive: power 1, store into speed |
| Down (or S) | climb: power 2, speed into store |
| neither | coast |
| Space | fire |
| Tab | next ship type of the card, kept in flight (one type today: the P-51D) |
| C | chaser on / off |
| P | pause |
| + / - or mouse wheel | zoom |
| R | reset |
| Esc | quit |

Physics runs at the game's 30 Hz tick; the view redraws as fast as Tk allows and
interpolates the ships between ticks.

## On screen

- The ship (blue) with its velocity vector (green, half a second long); the gap between nose and
  vector is the angle of attack. A faint circle is the 12 m hit radius.
- The trail of the last 5 s, a grid (the finest of 25 m to 2500 m that is at least 40 px apart,
  every fourth line brighter), the arena edge (yellow, where boundary damage starts) and the end
  of the damage ramp (red, dashed), and the field bands (ten fields drawn at each reset).
- The chaser (red), with an edge marker and its distance when off screen. Your bullets
  are yellow and the chaser's orange.
- Readout: speed (m/s over the map), turn rate of the flight path (deg/s), turn radius,
  lateral acceleration in the game's g (39.24 m/s^2), angle of attack against the stall
  angle, store (% of max, with a bar), shield, ship type, stick level and index, power.
- Strip chart (bottom right): speed and store over the last 10 s.

You cannot die: a lethal hit or the boundary refills the shield and counts a death. A
killed chaser comes back 2 s later, 1500 m away.

## Testing without the keyboard

`--headless --demo --seconds 5` hides the window, feeds a scripted input cycle and quits after
5 s, printing the tick and frame counts. A display is still needed (Tk), but no window shows.
