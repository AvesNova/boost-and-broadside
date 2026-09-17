# Frontline playtest

Run `.venv/bin/bnb play` for 5v5 Frontline. See [the current rules](environment.md)
and [scripted strategy](frontline-scripted-strategy.md).

Five 330 px zones sit on a 1200 px ring. Their cyclic roles are neutral → Team 0
spawn → Team 0 defense → Team 1 defense → Team 1 spawn. Capture takes eight seconds
at a one-ship lead; a lead of n progresses at H(n), the harmonic number. Equal counts
pause capture. Simultaneous captures net to zero and reset both meters.

All zones are opaque. Ships sharing a zone can see each other. Own spawn protects
ships; shields recover anywhere after four seconds without damage. Initial ships and
instant respawns start at 15 shields, 20 energy and 30 proper speed. Defense zones,
enemy spawn and field interfaces cause no passive damage. The outer boundary remains
a hazard. The match ends at front ±3 or five minutes.

Controls: WASD, Shift to turn sharply, Space to fire; Tab cycles ships and spectator;
C follows the selected ship; F fits the battlefield; R/Home resets the camera;
mouse wheel zooms; middle/right drag pans; +/- changes simulation speed.

Evaluate whether zone occlusion creates useful approaches, retreat earns recovery,
spawn protection prevents repeated spawn kills, and fights convert into captures.

Batched playtest:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python benchmarks/frontline_scripted_suite.py \
  --device cpu --games 8 --team-size 5 --max-ticks 9000 --output /tmp/frontline.json
```
