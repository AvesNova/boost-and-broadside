# View: v2's look for lean matches (branch `view`)

## 2026-11-26 07:50: milestones A and B

Tried: first a cleaned-up port of v2's renderer; on the author's change of plan (copy as-is,
adapter only) switched to copying `src/bnb/play` into `lean/view/play` verbatim (NEEDS REVIEW)
with an adaption layer (`lean/view/snapshot.py` `to_v2`, `match.py`, `index_map.py`,
`marks.py`, `interface.py`, `video.py`, `live.py`).

Worked:
- Milestone A: offscreen MP4s with stills, EGL on the 4070, about 15-20 s per match end to
  end. In `data/replays-2026-11-26-view/`: 1v1 no fields, 4v4 fields (spectator and team 0),
  4v4 team 0 with eccentricity 0.5 and vision outlines. Stills compared with
  `docs/images/play/*.png` on branch v2: same sky, hatched boundary, field fills and band
  lines, Komet silhouettes with shield/energy rings, trails, tracers, fog, minimap card, top
  bar style. Differences: the distance contours (v2's current default, absent from the older
  docs shots) are on; the video's top bar is a simpler one than the live window's.
- Milestone B: the live window is v2's own session on a lean match (game in a child process,
  played back like a replay): views, themes, follow, zoom, pause, step, speeds, scrubber,
  kill feed, fleets and inspector cards. Checked with the offscreen session runner
  (`--offscreen`) and a 25 s window launch (no errors; not looked at on screen).
- The fog uses lean's sight ellipse (a small marked edit to the copied shader).

Didn't: the prediction overlays, the tuning panel, the map editor, flying, changing sides and
v2 replays are turned off in the lean session (lean/view/NOTES.md lists each with what adapting
it would take). Policy against scripted in the window is not done.

Next: the author looks at the videos and the window; then the prediction overlays on lean's
model outputs (next-state and value first) and a policy side, if wanted.
