# lean/tool

A small local tool for the author: ship tuning charts and the flight sandbox in the browser.
It computes nothing itself: the server calls the game's own code (`lean.game.tuning`,
`lean.game.core`, `lean.sandbox.world.World`); the page only draws.

```bash
uv sync --extra tools        # fastapi, uvicorn, websockets 
PYTHONPATH=.:src uv run --no-sync python -m lean.tool [--port 8000]
```

Open http://127.0.0.1:8000/ (the page loads Plotly.js from cdn.plot.ly, so it needs a network
connection for the charts).

- **Ships tab.** A preset (the card's types), a time scale (1x real time, 1.5x, 2x the game,
  3x) and chart units (game or real time: speeds, turn rates and Ps divided by the scale,
  radii unchanged). Seven sliders move the design's handles (cruise speed, stall speed, g
  limit, settled full-stick turn rate, zero-lift drag, stall angle, dive/zoom trade); each
  change, 250 ms after the last, goes to the server, which solves the card row through
  `tuning.handles_row` (low-speed acceleration left free: the preset's static thrust stays)
  and returns `tuning.envelope` for it and for the unchanged preset (dotted reference).
  Charts: the doghouse (turn rate against speed with Ps shading, the stick levels, dive and
  climb, corner, best sustained) and the sustained turn (rate and radius against speed).
  Readouts: corner speed and rate, best sustained rate and its speed, tightest sustained
  radius and its speed, the gap between those speeds, top speed, best climb, stall speed.
  The same `tuning.envelope` draws the PNGs in `docs/rewrite/sections/turns/doghouse.py`.
- **Sandbox tab.** Flies the design tuned in the Ships tab at its time scale against the
  scripted opponent; controls in `lean/sandbox/NOTES.md`. Each tab visit starts a new
  session: the server starts a child process that ticks the real `World` at 30 Hz and sends a
  frame per tick over `/api/sandbox`; the browser interpolates between frames.

The card is process-global (`tuning.flying`), which is why the sandbox runs in a child
process and why envelope requests are serialised. Tests: `lean/tool/tests`.
