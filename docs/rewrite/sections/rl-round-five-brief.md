# Brief: RL changes for the next run: league clean cut, no ladder, DPPO, clip-higher

**Decided by the author (2026-11-06; `decisions.md`, "Round four"):** the changes below go
into the RL learner and league before the next RL experiments (short fine-tunes from
calm-fern's 73M policy, then a full run). CPU work only.

## 1. League turnover: a clean cut instead of drains

Read `docs/rewrite/sections/league-sampling-sim.md` (rounds 1 and 2,
`bench/league_sampling.py`, `bench/league_sampling_2.py`) and `docs/explanation/league.md`.

- **The cut:** every `cut_every` updates (config; default 2), one slot (`cut_slots`,
  default 1, round robin) gets a freshly drawn entry: its stacked weights are replaced in
  place at the update boundary. No drain, no incoming set (the extra stacked set goes).
  Games in progress on those rows continue; the opponent's results for those games are
  not credited to anyone (they still train the learner: the learner's ticks are on-policy
  data).
- **The opponent's carry at a cut** (option, `carry_at_cut`: `"reset"` or `"keep"`,
  default `"reset"`): reset the opponent's model carry on the cut rows, or keep the old
  occupant's Griffin state (only when the incoming entry has the same build; else reset).
  The belief part of the carry (observations, last sightings) is always kept.
- **The ladder out of the RL training pool:** a config switch (`ladder_in_pool`, default
  false for PPO; BC is unaffected). The rating lanes still play the scripted agent and
  whatever ladder they use.
- Keep p(1 - p) and the floor. Keep the gauge reading the one fit; add the guard round 1
  recommends: a test that a near-sweep (28 of 29 against a ladder entry) rates finite and
  within a few hundred points of its maximum, and an alert when an entry's rating is more
  than 1,500 from live's or its error is infinite after it has played.
- Log: cuts per update, cut rows, games uncredited, the opponent mix per update.
- Tests (CPU): the cut swaps the right slot at the right update and nothing else; keep vs
  reset; same-build check; ladder out; resume after a cut is exact.

## 2. The memory hand-over test (author: "double check with a test")

CPU only, no training: on a captured self-play game (or a short CPU rollout) with two
calm-fern policies of the same build (for example `players-calm-fern/step_73400320-live`
and `step_88080384-live`, also `step_7340032-live` for a far pair; read only under
`/home/vizia/avesnova/runs-v2/players-calm-fern/`), switch the opponent from A to B
mid-game three ways: (a) B with its own state warmed from the game's start (the truth),
(b) B starting from A's state, (c) B from a reset state. Report B's action-distribution KL
from (a) over the following ticks (by seconds after the switch), for a near and a far pair,
both modes. Say whether (b) beats (c), and how fast each converges.

## 3. DPPO and clip-higher (options in the PPO policy term)

Read the summary of Qi et al., "Rethinking the Trust Region in LLM Reinforcement Learning"
(arXiv 2602.04879, ICML 2026) in `docs/rewrite/sections/dppo-notes.md` (the coordinator
writes it from the paper's read-out; if missing, read the paper's HTML).

- `LearnerConfig.trust_region`: `"clip"` (today, bit for bit), `"clip_higher"` (asymmetric
  ratio clip, `clip_low` 0.2 and `clip_high` 0.28), `"dppo_tv"` (exact total variation of
  the joint 42-action distribution at the state, old (rollout) vs current), `"dppo_binary_tv"`
  (|mu(a) - pi(a)| of the sampled joint action). `dppo_delta` (default 0.15).
- The DPPO mask, per sample: zero the gradient when D > delta and the update moves further
  away (A > 0 and r > 1, or A < 0 and r < 1); otherwise the unclipped r * A.
- The joint distribution over 42 actions is the product of the three heads' distributions;
  the rollout must store what the exact form needs (the old per-head probabilities, or the
  joint) at collection time; say what it costs in memory.
- Log the mask rate (overall, by head entropy bucket if cheap) and the divergence quantiles.
- Tests: `clip` equals today; the mask's four cases; exact TV against a hand computation;
  binary TV is a lower bound of exact TV.

## 4. Small fix

The per-block representation series (`representation/*`) are tapped at the temporal
sublayer; with `yemong.order = temporal_first` that is the block's first sublayer. Tap the
block's output in both orders (spatial_first stays as today: its temporal sublayer is last).

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-rl-five`, branch `feat/rl-round-five` (from v2).
  Work only there. Scratch in `/tmp/bnb-scratch-rl-five`. Don't read other directories
  under `/home/vizia/avesnova`, except the read-only policies named above.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`; BC arms own the GPU. Never take the
  bench lock. No training runs.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **Progress notes:** `/tmp/bnb-scratch-rl-five/progress.md`, `## HH:MM` from `date +%H:%M`,
  1-3 bullets; ETAs as local clock times. Start with a plan note. Do part 2 (the test) early
  and report its result in a note as soon as it is in.
- **Tests:** CPU tests for what you add, the quick tier (`pytest -m "not slow and not gpu"`),
  the CPU resume tests (you touch the league), ruff, docs (`league.md`, `learner.md`).
- If something here is wrong, missing or ambiguous, say so in your notes and final message;
  make a sensible choice and keep going.

## Done

- Commit and push to `feat/rl-round-five` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, half a page:** each part, the hand-over test's result, the exact `--set`
  lines for each option, anything open.
