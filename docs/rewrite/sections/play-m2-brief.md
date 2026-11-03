# Brief: section 10, the play shell, milestone 2 (the on-screen interface)

You are building **the on-screen interface** of the Boost and Broadside v2 play shell, on
top of milestone 1's renderer. The goal is a game a person can set up, watch, fly, pause,
replay and play against a trained agent from the screen alone, in the style of strategy
games like Age of Empires II and Civilization V. It must look clean and stay fast.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-play-m2`, branch `section/play-m2` (from v2).
  Work only here. Scratch in `/tmp/bnb-scratch-play-m2`.
- **Do not read or modify** any other directory under `/home/vizia/avesnova`: no v1, no
  `physics-race*`, no other worktree.
- **Python:** `uv sync`, then always `uv run --no-sync ...`.
- **You cannot see a window.** Check your work by rendering offscreen to PNG files and
  reading them (your Read tool shows images). Look at every screen you build.
- **GPU:** GL tests and timings hold the benchmark lock (`bench.lock.bench_lock()`; never
  plain `flock`). The wrapper `/tmp/bnb-scratch-fast-yemong/locked.py` runs a command
  under it: `uv run --no-sync python /tmp/bnb-scratch-fast-yemong/locked.py <command...>`.
  Another agent (the league) holds it for long runs; keep yours short, and do most of your
  work on the CPU and offscreen.
- **Keep your context small:** send long output to files and read their tails.

## Read first

1. `docs/rewrite/sections/play.md`: the decisions, above all "On-screen controls, not
   hotkeys".
2. `docs/rewrite/sections/play-m2-spec.md`: **the spec you implement.**
3. `docs/rewrite/sections/play-notes.md`: milestone 1's design, "What milestone 2 should
   build on", and its requests.
4. `docs/rewrite/visual-mockups/README.md` and the mock-ups it names: the look the panels
   must match.
5. `src/bnb/play/`, `src/bnb/runtime/realtime.py`, `src/bnb/runs/` (policy files),
   `src/bnb/game/api.py`.
6. `docs/contributing/code-style.md`, `docs/contributing/writing-docs.md`, and
   `docs/rewrite/scaffold.md` sections 3 and 4.

## What to do

1. **First, the runtime requests (spec 7)** and the cleanup pass's requests for
   `bnb/play/`: format it with ruff, add its docstrings, and drop it from the format-check
   and docstring exclusions. Commit each separately.
2. **Then the interface,** in this order, one commit or more each, pushed:
   1. the layout's skeleton, the theme for ImGui, collapsing and the interface scale;
   2. the top bar and speed controls; the view selector;
   3. selection, the hit test, the right card, the fleets card;
   4. the minimap;
   5. menus and New Match, with the background compile;
   6. trained agents per side;
   7. replays, the scrubber and the kill feed;
   8. the flying HUD.
3. **Show the author:** after step 2.3 and at the end, put screenshots in `docs/` and list
   them in your notes.

## Tests

- **Fixed:** everything under `tests/reference/`.
- **Spec section 8** is the acceptance list: a test for each item.
- **The tiers:** the `gpu` marker for GL or CUDA (skips cleanly without them); `slow` for
  a test over 10 s. The quick tier stays around 2.5 minutes in all. The suite's guards
  fail a non-`gpu` test that initialises CUDA, and a test that changes torch's global CPU
  generator (use the `global_generator` fixture).
- **Everything passes:** the full suite under the lock, once at the end.

## Boundaries

- **You may change:** `src/bnb/play/**`, `src/bnb/controllers/human.py`,
  `src/bnb/runtime/realtime.py`, `pyproject.toml` (dependencies and the play exclusions), tests outside
  `tests/reference/`, the user docs, your notes.
- **Do not change** the game's rules, the models, the learner, the league
  (`src/bnb/league/`, being built now), or `src/bnb/runtime/rollout.py`.
- **If the spec looks wrong:** take the most literal reading, keep going, and write it up
  as a request. Items marked (proposal) are implemented as written and listed.
- **Names:** full words. **No legacy:** no shims.

## Commits

Commit to `section/play-m2` and push (`git push -u origin section/play-m2`). Author
Julian Kizanis <julianzia@gmail.com>; end every commit message with
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Hand back

Write `docs/rewrite/sections/play-m2-notes.md` with:
- the design;
- each spec item, with what was done;
- the screenshots;
- the measurements;
- deviations, requests and open questions;
- a Progress section kept current, so a fresh agent can take over.

This is a long task. Stop at a clean commit if you have been working long. Your final
message summarizes it in a page or less.
