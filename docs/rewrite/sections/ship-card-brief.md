# Brief: section 1b, the ship card

You are implementing **the ship card** for Boost and Broadside v2: designer handles, a
solver that finds the physics parameters meeting them by flying the game's own code, the
per-type parameters that needs, two small physics changes, and a reference-chart tool.
The goals are **accuracy, independence of the handles, and maintainability**: code the
next person can read and change.

## Where you work

- **Your worktree:** `/home/vizia/avesnova/bnb-ship-card`, branch `section/ship-card`.
  Work only here.
- **Scratch:** use `/tmp/bnb-scratch-ship-card` for temporary files.
- **Do not read or modify** any other directory under `/home/vizia/avesnova`. In
  particular, stay out of the other worktrees, the v1 code, and the `physics-race*`
  repositories.
- **Python:** set up with `uv sync` in your worktree, then always run through
  `uv run --no-sync ...` (never plain `python`).

## Read first

1. `docs/rewrite/sections/ship-card.md`: the design behind this section.
2. `docs/rewrite/sections/ship-card-spec.md`: **the spec you implement**.
3. `docs/rewrite/ship-card.md`: the older card notes, for the P-51D similarity set.
4. `docs/rewrite/sections/game-spec.md` (sections 1.1, 3, 6 and 7) and `game-notes.md`:
   the physics you parameterise.
5. `docs/rewrite/contracts.md` (the current-schema section) and `src/bnb/contracts/`.
6. `docs/rewrite/scaffold.md`: section 3 (design rules) and section 4 (layout and
   dependency rule).
7. `src/bnb/game/`, especially `ship_types.py`, `rules.py`, `physics/flight.py`,
   `physics/bullets.py`, `physics/combat.py`, `perception/`, and `tests/`.

## What to do

1. **Implement the spec.**
2. **Tests:**
   - **Everything must pass:** `uv run --no-sync pytest`, nothing skipped except what
     needs hardware you lack.
   - **Fixed:** everything under `tests/reference/`, adapters included: the game must not
     change at the default card. Never change those files or the goldens.
   - **Add your own** under `tests/sections/` for every acceptance item. Give a test that
     needs the GPU the `gpu` marker.
3. **Speed:**
   - Measure what the spec asks for (the bleed exponent's cost at the bench cells).
   - Any GPU timing must hold the benchmark lock (`bench.lock.bench_lock()` in your
     script). Never wrap a lock-taking command in plain `flock`, and never run GPU tests
     while your own benchmark holds the GPU.
   - Keep it proportionate. The solver is offline and runs on the CPU.
4. **Commit and push often:**
   - commit to `section/ship-card`, then `git push -u origin section/ship-card`;
   - author Julian Kizanis <julianzia@gmail.com>;
   - end every commit message with
     `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Boundaries

- **You may change:**
  - `src/bnb/game/**`;
  - `src/bnb/cli.py`, for the `ship-card` command;
  - `src/bnb/contracts/**`, only where the per-type card forces it;
  - `pyproject.toml`, only for the optional `tools` extra;
  - `tests/sections/**`, and your notes file.

  List every change outside `bnb.game` in your notes.
- **You may not change:** `bench/`, `tests/reference/`, or the docs (apart from your
  notes file).
- **Physics:** only the two changes the spec names (the bleed exponent; the cruise fix,
  imposed by the solver). The default card's numbers stay as they are. If the low-speed
  thrust cap breaks the cruise fix by more than 2%, report it; do not change the
  formula.
- **If the spec or a contract looks wrong:** take the most literal reading, keep going,
  and write it up as a request in your notes. Items marked (proposal): implement them as
  written and list them.
- **Names:** full words.
- **No legacy:** no shims or dead code.

## Hand back

Write `docs/rewrite/sections/ship-card-notes.md` with:
- **Design.**
- **Decisions and deviations,** each with its reason.
- **Measurements:** the default card's handles and what the solver makes of them under
  the cruise fix; the round-trip, independence and tick-rate results; the condition
  numbers; the cruise variation across the store; the bleed exponent's cost.
- **Changes outside `bnb.game`.**
- **Requests and open questions.**
- **Size.**

Your final message should summarize the same, in a page or less.
