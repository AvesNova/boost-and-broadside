# Brief: play milestone 2, fix round

Milestone 2 of the play shell is merged (`play-m2-spec.md`, `play-m2-notes.md`). This
round fixes four things the author and coordinator found.

1. **Agents run on the GPU whenever one is present** (author, 2026-11-03), and on the CPU
   otherwise (`src/bnb/play/sides.py`). The sim stays on the CPU, so each tick moves the
   observation to the agent's device and the action back. Measure one agent tick and two
   agents per tick on each device, at 5v5 with the default Yemong. The notes measured
   about 15 ms for two agents on one CPU thread, at the edge of the 16.7 ms frame. Add a
   test that the device follows CUDA's availability; it skips cleanly without CUDA.
2. **The objective in the top bar must be readable.** Frontline's meter and zone states
   are now bare squares and dots. Label them: a short caption (for example "Front" and
   "Zones"), each zone's letter or number, the owner's team colour, and a tooltip saying
   what each shows. Check the result in an offscreen screenshot at 1920 and at 1280.
3. **The camera follows the flown ship by default** (spec 4). In
   `docs/images/play/interface-flying-1920.png` the view is the whole arena. Fix it and
   add a test: after "Take control", the camera is centred on the flown ship within a few
   pixels.
4. **The kill-feed mutation check** (moved outside its rectangle) never ran; run it.

Regenerate the screenshots that change (`docs/images/play/interface-*`) and look at each.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-play-fix`, branch `fix/play-m2` (from v2). Work
  only here. Scratch in `/tmp/bnb-scratch-play-fix`.
- **Do not read** any other directory under `/home/vizia/avesnova`.
- **Python:** `uv sync`, then always `uv run --no-sync ...`.
- **GPU:** GL and CUDA runs hold the benchmark lock. Use the wrapper
  `/tmp/bnb-scratch-fast-yemong/locked.py <command...>`. Another agent shares it, so keep
  your runs short.
- **You cannot see a window:** render offscreen and read the PNG files.

## Tests and boundaries

- **May change:** `src/bnb/play/**`, its tests, the play docs and screenshots, and
  `play-m2-notes.md` (add a "Fix round" section).
- **The tiers:** the `gpu` marker for GL or CUDA, `slow` for a test over 10 s.
- **Done:** every play test passes under the lock (`uv run --no-sync pytest -q
  tests/sections/test_play*`), and the docs tests pass. The coordinator runs the rest.
- **Commits:** to `fix/play-m2`, pushed (`git push -u origin fix/play-m2`). Author Julian
  Kizanis <julianzia@gmail.com>; end every message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message:** what changed, the timings, and the new screenshots' paths, in half a
  page.
