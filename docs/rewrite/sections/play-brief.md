# Brief: section 10, the play shell, milestone 1 (the renderer)

You are building **the renderer** for Boost and Broadside v2: a desktop window that plays
a match at the display rate, interpolating a 30 Hz simulation, in the look the author
chose in a design brainstorm. The goals are **speed, beauty and maintainability
together**: 60 fps or more, a clean flat look, and code the next person can extend into
the developer view.

## Where you work

- **Your worktree:** `/home/vizia/avesnova/bnb-play`, branch `section/play`. Work only
  here.
- **Scratch:** use `/tmp/bnb-scratch-play` for temporary files.
- **Do not read or modify** any other directory under `/home/vizia/avesnova`. In
  particular, stay out of the other worktrees, the v1 code, and the `physics-race*`
  repositories.
- **Python:** set up with `uv sync` in your worktree, then always run through
  `uv run --no-sync ...` (never plain `python`).
- **You cannot see a window.** Check your work by rendering offscreen to PNG files and
  reading them (your Read tool shows images). Compare them with the mock-ups.

## Read first

1. `docs/rewrite/sections/play.md`: the decisions.
2. `docs/rewrite/sections/play-spec.md`: **the spec you implement.**
3. `docs/rewrite/gameplay-brainstorm.md` (the Visuals sections) and
   `docs/rewrite/visual-mockups/`:
   - read `README.md`, then look at `map-vis.png`, `shields3.png`, `trails2.png`,
     `fogdark.png`, `ships3.png`, `test-lemon.png`, `whites2.png` and `map-contours.png`;
   - `colour-tool.html` holds the palette values in its source.

   The brainstorm describes some things v2 does not have yet: capsule fields, bases,
   turrets and roles. Draw v2's game as it is: disc fields, a disc arena, Frontline
   zones.
4. The user docs in `docs/` (`docs/explanation/`, the glossary, `docs/contributing/`), and
   `docs/rewrite/scaffold.md` sections 3 and 4.
5. `src/bnb/`, especially:
   - `runtime/realtime.py`;
   - `game/api.py`, `game/state.py`, `game/physics/medium.py`,
     `game/perception/sight.py`;
   - `contracts/`;
   - `controllers/`.

## What to do

1. **Implement the spec.**
2. **Tests:**
   - **Everything must pass:** `uv run --no-sync pytest`, nothing skipped except what
     needs hardware you lack.
   - **Never change** anything under `tests/reference/`.
   - **Add your own** under `tests/sections/test_play*.py`. A test that needs a GL context
     or the GPU carries the `gpu` marker.
3. **Speed:**
   - Measure what the spec asks for.
   - Any GPU timing holds the benchmark lock (`bench.lock.bench_lock()` in your script).
     Never wrap a lock-taking command in plain `flock`, and never run GPU tests while
     your own benchmark holds the GPU.
   - Other agents share this machine's GPU and CPU: keep your GPU runs short, and
     re-measure if a number looks contended.
4. **Docs.** The docs in `docs/` and `README.md` are checked by the suite.
   - `uv run --no-sync pytest` fails if a backticked name, a repository path, a `bnb`
     command option, a runnable example, or `mkdocs build --strict` breaks. Fix those
     as part of your change.
   - Never copy code-defined values into prose.
   - Replace the play shell's page under `docs/explanation/in-progress/` with prose, and
     set its status in `docs/roadmap.yml`.
   - Follow `docs/contributing/writing-docs.md`.
5. **Commit and push often:**
   - commit to `section/play`, then `git push -u origin section/play`;
   - author Julian Kizanis <julianzia@gmail.com>;
   - end every commit message with
     `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Boundaries

- **You may change:**
  - `src/bnb/play/**` (new);
  - `src/bnb/controllers/human.py` (new), and `controllers/__init__.py` to export it;
  - `pyproject.toml` and `uv.lock`, for the dependencies;
  - `tests/sections/test_play*.py`;
  - the user docs (`docs/` outside `docs/rewrite/`, and `README.md`);
  - your notes file.
- **You may not change:** `src/bnb/game/**`, `runtime/**`, `learn/**`, `models/**`,
  `contracts/**` or `cli.py`. Other agents are editing several of them. If you need
  something from them, take the most literal workaround inside `bnb.play` and write it
  up as a request.
- **Reuse the game's functions** (`medium`, `sighted`, the state and observation). Never
  reimplement game rules, except a display-only approximation that a test checks against
  the game.
- **If the spec looks wrong:** take the most literal reading, keep going, and write it up
  as a request in your notes. Items marked (proposal): implement them as written and list
  them.
- **Names:** full words. **No legacy:** no shims or dead code.

## Hand back

Write `docs/rewrite/sections/play-notes.md` with:
- **Design:** the loop, the passes, the snapshot, how each look element is drawn.
- **Decisions and deviations,** each with its reason. Include the library and wheel check.
- **Measurements:** frame time per pass and per preset, the fog mask's cost and
  resolution, the index texture's resolution.
- **Screenshots:** paths of the representative PNGs you put in `docs/`.
- **What milestone 2 should build on,** and anything that made the developer view harder.
- **Requests and open questions.**
- **Size.**

Your final message should summarize the same, in a page or less.
