# Brief: a soft scripted agent, tuned by population round robin

The scripted agent is the BC teacher, the rating anchor (1000) and the ladder. As built, it
is a poor teacher (`docs/rewrite/sections/scripted-softness-notes.md`):

- **It is near-deterministic.** Shoot has probability exactly 1 on 99% of steps; power on
  60–74%; stick on 51–62%.
- **Its ramps are step functions.** Shoot-by-distance, reverse, aim, turn, level two and
  boost-slow are 1–24% of their input's spread wide, and all end at exactly 0 and 1.
- **It cannot survive noise.** A 1% uniform floor costs 0.10 of its 0.5; a 5% floor leaves
  0.06. Its clones score 0.18–0.27 against it.

v1's agent ended every ramp inside about 0.04–0.95, and its ramps were wide. The author
wants that principle back: gradual ramps, no sharp decision edges, an agent that plays well
across a range of temperatures. **"Good enough" is the goal, not optimal:** a solid
opponent that is also a good teacher.

## 1. The agent: logistic ramps with a temperature and a floor

- **Every probability ramp is a logistic** in its input, with a centre and a scale (the
  searched numbers), and a direction (fixed, from today's ramp): p(x) = ε + (1 − 2ε) σ(±(x −
  centre) / (scale · T)).
  - Ramps that are weights, not probabilities (e.g. the boundary steer), keep their form;
    say which ones in the docs.
- **ε = 0.04 per binary decision**, fixed (v1's floor). It is not searched, so the search
  cannot sharpen it away.
  - Where a factor's value is chosen without a ramp (e.g. a turn direction taken from a
    sign), give it the same treatment: a logistic in the quantity behind the sign, or an
    ε-floored choice.
  - The aim: no action the agent can take sits at probability exactly 0 or 1 at T = 1.
- **The temperature T** is a property of the agent (a field, or a per-row value in the
  parameter table, so one batch can hold several T). T = 1 is the agent; larger T flattens
  every ramp.
- **Smoothness:** each scale's lower bound is a fixed fraction of its input's observed
  spread (from the audit; ≥ 0.1 × std is a starting point). No step-like ramp is reachable.
- **Report the joint entropy per living ship-step at T = 1, 2 and 4.** There is no target
  beyond the floor and the bounds.

## 2. The search: a population round robin in one batch

Rework `bench/scripted_search.py`; it currently keeps one best and plays it against the
chaser and the old parameters.

- **Population:** P candidates (e.g. 32; one parameter-table row per view, so one compiled
  rollout per generation).
- **Matches:** candidates play each other, round robin or random pairs, enough games per
  pair for a usable fit. Also, as fixed reference players:
  - today's scripted agent (`ScriptedParameters()` defaults, raw);
  - the chaser.
- **Temperatures:** each seat's T is drawn per game from a set such as {1, 1.5, 2, 3}, so
  the ranking rewards strength across temperatures.
- **Presets:** all three: frontline, deathmatch, deathmatch_wide. The clones fail most in
  deathmatch_wide.
- **Ranking:** a Bradley–Terry fit over every game of the generation (`bnb.league.ratings`).
- **Next generation:** keep the top K; fill the rest with perturbations of them, and
  recombination if it helps. Shrink the step as generations improve.
- **Stop when the top's rating against the reference players plateaus.** Good enough is
  the goal.
- **Log:** every candidate's parameters, ratings and per-preset scores as JSON lines, as
  now. The run must be resumable.

## 3. Wire it in

- **New defaults:** the winning parameters become `ScriptedParameters()`'s defaults.
- **The BC teacher is the agent at T = 1.** `TeacherConfig` defaults to T = 1 and floor 0.
  - Keep a floor only if the KL needs one numerically. If so, say so and use the smallest
    that works.
  - BC's score stop (parity against the scripted agent in the lanes) is then a target the
    teacher itself meets.
- **The ladder becomes temperatures.** The rungs move from "scripted with probability p,
  else random" to the scripted agent at temperature T. Name them clearly
  (e.g. `rung_T2`), from T = 1 up to a T where the agent is close to random, with random
  kept at the bottom.
  - Update the league, lanes, head-to-head and docs.
  - Note in the docs that this resets ladder ratings: the rungs are new players.
- **Head-to-head's tempered player (`scripted@T=...`)** uses the agent's own temperature
  from now on, not the post-hoc tempering of a distribution.
- **Acceptance tests** (`tests/sections/test_scripted_acceptance.py`): rebaseline them to
  the new agent, keeping their intent.

## Acceptance, to report

All against 128 games per preset:

- The new agent at T = 1 against today's raw agent, and against today's BC teacher
  (`scripted@floor=0.01`, which scores 0.398 against today's raw agent). Good enough
  means it beats today's teacher.
- The new agent at T = 1.5, 2, 3 and 4 against itself at T = 1. This is the
  temperature curve.
- The new agent at T = 1 with a 1% uniform floor, against itself without. The floor should
  cost far less than today's 0.10.
- The softness audit (the earlier agent's `audit.py`, in
  `/tmp/bnb-scratch-scripted-softness`) rerun on the new agent: shares at exactly 0 and 1
  per factor, entropy, and ramp widths against their input spread.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-scripted-redesign`, branch `feat/scripted-redesign`
  (from v2). Work only there. Scratch in `/tmp/bnb-scratch-scripted-redesign`. Do not read
  other directories under `/home/vizia/avesnova`, except the earlier scratch named above.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **GPU:** hold the bench lock for every GPU command:
  `uv run --no-sync python ~/.local/share/bnb/locked.py <command...>`.
  - A BC training run holds the lock until about 11:30 local time. Do parts 1 and 3 and
    their CPU tests first; the lock simply waits until the GPU is free.
- **W&B:** none.
- **Progress notes:** append brief milestone notes to
  `/tmp/bnb-scratch-scripted-redesign/progress.md`, with a `## HH:MM` heading (from `date`)
  and 1–3 bullets. Write one when a part lands (with its commit), when a generation summary
  is worth showing (the top's rating and scores, the population's spread), when something
  fails, and when you change plan. Start with a plan note.
- **Tests:** CPU tests for what you change, plus the quick tier; ruff clean; docs updated.
  No GPU test suite and no dummy training run: the next BC run is the integration test.

## Done

- Commit and push to `feat/scripted-redesign` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, one page:**
  - the agent's form;
  - the search's run (generations, what improved);
  - the acceptance tables;
  - anything left open.
