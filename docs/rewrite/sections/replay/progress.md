# Replay videos: progress

## 2026-11-25

- Tried: looked for existing replay or video tooling (lean/eval, lean/tool, lean/sandbox,
  docs/rewrite/sections). None writes video; the browser tool draws the sandbox only. The
  scripted agent's `explain` gives its tactic weights and plan per ship, used for the labels.
- Worked: `replay.py` (CPU only, PIL drawing at 2x then reduced, piped to ffmpeg). Seeds 0, 1, 2
  rendered to data/replays-2026-11-25/ (3.1-6.4 MB, 44-63 s each).
- Didn't: labels overlap when ships bunch within ~100 m; not fixed.
- Next: the author watches; adjust what is drawn on request.
