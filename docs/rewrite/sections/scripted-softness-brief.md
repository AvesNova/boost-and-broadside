# Brief: how soft is the scripted agent, and what is BC's ceiling against it

The new BC clone (`/home/vizia/avesnova/runs-v2/2026-11-05-mixed-sunny-ember/policy`, 50M
steps) scores 0.190 ± 0.019 against the raw scripted agent (frontline 0.23, deathmatch 0.25,
deathmatch_wide 0.09). The old clone (`.../2026-11-04-mixed-wandering-quill/policy`, 100M)
scores 0.250. BC's stop targets parity (0.5).

A rung that is 95% scripted and 5% random scores 0.065 against scripted. The clone imitates a
softened teacher (`bnb.learn.bc.TeacherOpponent`: tempered marginals, then a uniform floor),
so parity may not be reachable however well it imitates. That is one hypothesis, not the
answer: training all three modes at once instead of one is another. Measure; don't argue
for either.

## 1. Head-to-head players for the teacher and for tempered scripted agents

- `bnb head-to-head` gains a player kind for the scripted agent playing a transformed
  distribution: temperature T and floor f, as `TeacherOpponent` does (e.g.
  `scripted@T=1,floor=0.01`; pick a clean syntax and document it).
- Its draws are the same code path as the teacher's.
- Tests: T = 1 and floor 0 play exactly as `scripted`; the parsing is covered.

## 2. Measurements against the raw scripted agent

128 games per preset (the default three presets). Each with its score ± SE and the split
by preset:

- BC's actual teacher: the defaults `TeacherConfig()`.
- T ∈ {0.5, 1, 2, 4} at floor 0.
- Floor ∈ {0.01, 0.05} at T = 1.
- `rung_0.95` and `rung_0.99`, for comparison.
- Also both clones against BC's actual teacher, so the clone-to-teacher gap is known.

Read each result as "the strength a perfect imitator of that distribution would have".

## 3. A softness audit of the scripted agent

Over a few thousand ticks of scripted self-play per preset, for each action factor (power,
stick, shoot) report:

- The distribution of the agent's probability for its chosen value. What share of
  decisions are at exactly 0 or 1, above 0.99, or in between.
- The mean entropy per factor, in nats per living ship step.
- For each ramp in `ScriptedParameters`: its width relative to the spread of its input
  quantity, as observed in the same games, and its end values. Flag ramps that act as step
  functions (narrow, ending at exactly 0 and 1).

For comparison, v1's scripted agent (coordinator's numbers; don't look for v1) ended its
ramps at about 0.04–0.95, never at 0 or 1:
- boost 0.68 → 0.04;
- reverse 0.5 → 0.05;
- turn 0.068 → 0.95;
- sharp turn 0.05 → 0.95;
- shoot by angle and by distance 0.95 → 0.05.

Its ramps were wide, e.g. shoot distance 200–500 units. Every binary decision kept at
least about 4–5% on each side.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-scripted-softness`, branch
  `diag/scripted-softness` (from v2). Work only there. Scratch in
  `/tmp/bnb-scratch-scripted-softness`. Do not read other directories under
  `/home/vizia/avesnova`, except the two policy directories above.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **GPU:** you are the only GPU job. Hold the bench lock for every GPU command:
  `uv run --no-sync python ~/.local/share/bnb/locked.py <command...>`.
- **W&B:** none.
- **Progress notes:** append brief milestone notes to
  `/tmp/bnb-scratch-scripted-softness/progress.md`. Use a `## HH:MM` heading (take the
  time from `date`), then 1–3 bullets: what landed, what was measured (the numbers), what
  failed. Write one when each part finishes. No play-by-play.

## Done

- Part 1 with CPU tests, ruff clean, the CLI docs updated. No GPU test suite.
- Parts 2 and 3 as tables in `docs/rewrite/sections/scripted-softness-notes.md`.
- Commit and push to `diag/scripted-softness`. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, one page:**
  - the tables;
  - what they say about BC's reachable score against raw scripted;
  - which ramps are step-like.

  Don't propose a redesign; the author and coordinator will do that.
