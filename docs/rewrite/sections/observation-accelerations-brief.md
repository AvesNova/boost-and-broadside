# Brief: accelerations in the observation, and policies that record their observation layout

**Decided by the author (2026-11-08, `round2-plan.md` O1, O2, O10):** the visible ships'
accelerations become observation columns (world frame, like velocity), and a policy file
records the observation columns it was trained with, so adding columns no longer locks out
old policies. The model keeps velocity-frame codes for the inputs (O2) and keeps filling
hidden rows from the belief. Arm 1 (`2026-11-08-bc-arm-C-accel`, trained with the
accelerations computed in the model) must keep playing the same.

Read first: `docs/rewrite/sections/round2-plan.md` (5.1, 8), `acceleration-inputs-brief.md`
(what arm 1 built), the observation contract (`bnb.contracts.obs`: `SHIP_COLUMNS`,
`SHIP_HIDDEN_ZERO`, `SCHEMA_VERSION`), `bnb.game.perception.observe`, `bnb.game.state`,
`bnb.runs.checkpoint` (`schemas()`, the schema check in loading, `load_policy`),
`bnb.models.yemong.inputs` (`accelerations`, `Motion`, `Carry.motion`), `bnb.models.mlp`
(reads the ship row by width).

## Part 1: policies record their observation layout (O10, first)

- A policy file (and a training checkpoint's policy part) records the observation's column
  names: ship columns, and any other column lists the contract has (globals, fields), next to
  the build record.
- Loading: a policy whose record has its columns loads when every column it lists exists in
  the current contract under the same name; a missing one fails loudly, naming it. A model that
  reads columns by name (Yemong reads `SHIP[name]` from the code's own contract) needs nothing
  more; a model that reads rows by width (the toy MLP) gets its recorded columns selected by
  name, in its recorded order (an adapter at load, not a change to the model).
- Policies written before this change have no column record: they get the column list of the
  contract as it is at v2 today (frozen as data in the code, one constant, with a comment).
- The observation schema version stays for real breaks; adding columns does not bump it.
  The rule, in the contract's docstring and the runs spec: a column's meaning never changes
  under the same name.
- Tests: an old-style policy (no record) loads and plays bit for bit as today; a policy with
  a record of fewer columns than the contract loads (Yemong and the MLP); a record naming a
  column that does not exist fails with its name; the round trip writes the record.

## Part 2: acceleration columns (O1)

- New hidden-block columns (zero for a ship the team cannot see), world frame:
  `acceleration_x`, `acceleration_y` (m/s^2, (v_t - v_{t-1}) / dt) and `nose_acceleration`
  (rad/s^2, (omega_t - omega_{t-1}) / dt), plus `acceleration_valid` (1 or 0).
- Valid only when the ship is alive at both ticks, was not spawned or respawned at t, it is
  not the first tick of a game, and **the viewing team saw it at both t-1 and t** (so "just
  spotted" is invalid in the observation itself, and nothing about t-1 leaks). Zero otherwise.
- The game state carries what this needs (each ship's velocity and nose rate at t-1, and
  each team's sight of it at t-1), reset correctly at game resets and respawns. Keep the sim's
  speed: measure a rollout's steps per second before and after (CPU is fine for the ratio)
  and say it.
- Yemong with `acceleration_inputs` on: visible rows read the new columns and turn them into
  the velocity frame at t for the codes (as today); hidden rows keep the belief's own change
  (`Carry.motion` shrinks to what hidden rows need, or say why not). The validity rule must be
  the same as arm 1's (the model's "just spotted" check becomes the observation's flag).
- **Equivalence:** on recorded play, the inputs from the new path equal arm 1's model-computed
  inputs within float32 rounding on every row (visible and hidden), and arm 1's policy
  (`/home/vizia/avesnova/runs-v2/2026-11-08-bc-arm-C-accel/checkpoints/` latest, read only, or
  any build with the switch on) steps to the same actions on a recorded segment. If they
  differ, find out why before going on.
- Replays and anything that stores observation rows: widen as needed; say what changed in
  memory per environment step.
- Tests: the columns on scripted motion (straight push, steady turn), each invalid case,
  just spotted, the equivalence above, `unroll` equals a loop of `step`, the off path
  unchanged (C-fixed's recorded outputs in `tests/sections/golden/yemong_arms_steps.pt`, to rounding),
  the MLP with the wider observation (through Part 1), the quick tier
  (`pytest -m "not slow and not gpu"`), ruff, docs (the observation contract page, `yemong.md`,
  the runs spec).

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-observation-accel`, branch
  `feat/observation-accelerations` (from v2). Scratch `/tmp/bnb-scratch-observation-accel`.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`, at most 8 threads; a BC run owns the
  GPU. Never take the bench lock; never touch the queue file, units or run directories (read
  only). No training runs.
- **Memory:** anything long runs under `systemd-run --user --scope -p MemoryMax=8G
  -p MemorySwapMax=0`, streaming to disk.
- **Git:** never use `git stash` (the repository has old shared stashes).
- **Python:** `uv sync --all-extras`, then `uv run --no-sync ...`.
- **Progress notes:** `/tmp/bnb-scratch-observation-accel/progress.md`, `## HH:MM` from
  `date +%H:%M`, 1-3 bullets, ETAs as clock times. Start with a plan note.
- If something here is wrong, missing or ambiguous, say so in a note and in the final
  message; make a sensible choice and keep going.

## Done

- Commit and push to `feat/observation-accelerations` in logical commits. Author Julian
  Kizanis <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message: the record format and the load rule, the new columns and the state they
  need, the sim speed before and after, the equivalence result, the tests, anything open, the
  time.
