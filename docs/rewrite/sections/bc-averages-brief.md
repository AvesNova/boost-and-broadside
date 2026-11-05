# Brief: averaged agents in behaviour cloning

The averaged agents are the product, in BC as in RL (decisions, "Behaviour cloning, round
three", 2026-11-05/06): a **uniform average** and an **exponential moving average** of the
live weights. RL's league keeps both (`bnb.league.pool.average_into`,
`LeagueConfig.exponential_decay`, and the uniform one starts after
`LeagueConfig.uniform_from`, 10M environment steps). BC keeps neither: its lanes rate only
the live clone against the scripted agent (`bnb.runtime.lanes.BC_MATCHUPS`), its stop rule
reads the live clone's score (`bnb.learn.bc.StopRule`), and it saves the live clone as the
run's policy.

Evidence it matters: the last BC run's live clone scored 0.272 against the scripted agent;
a uniform average of its retained checkpoints (66M-151M, built offline) scored 0.350.

## What to build

1. **BC keeps both averages,** folded in after every update, with the same functions and
   rules as the league (reuse, don't copy): the uniform average from the warm-up's end
   (10M environment steps; share the setting with the league's, or give BC its own field
   with the same default, and say which), the EMA throughout. Both are saved in BC's
   training checkpoints and restored on resume.
2. **Both are rated in BC's lanes** against the scripted agent, alongside the live clone
   (the lanes' matchups and shares; keep the lane count, split it sensibly, and say how).
   Their ratings appear as `rating/average/<mode>` (and a name for the uniform average's
   series) like RL's.
3. **The stop rule reads an average.** Parity against the scripted agent, over the same
   window, read on the better-rated of the two averages (the live clone's score stays
   logged). Say what you chose for ties and for the first updates, before the uniform
   average exists.
4. **The saved policy is an average.** The run's `policy/` holds the average the stop rule
   read, with its lane ratings (so a warm start seeds the RL ratings from it, as now);
   also write the other average and the live clone beside it (for example `policy/`,
   `policy-uniform/`, `policy-exponential/`, `policy-live/`; choose clear names and
   document them).
5. **Checkpoint retention** must not delete what a later offline uniform average would
   need: with the uniform average kept in the learner this matters less; say whether the
   retention rule still needs a change and why.
6. Docs (`docs/explanation/behaviour-cloning.md`, `docs/guides/behaviour-cloning.md`, the
   lanes and league pages where they describe BC) and CPU tests for each part.

## The machine

An RL run owns the GPU for hours. Work on the CPU (`CUDA_VISIBLE_DEVICES=""`), never take
the bench lock, no training runs. The next BC run is the integration test.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-bc-averages`, branch `feat/bc-averages` (from
  v2). Work only there. Scratch in `/tmp/bnb-scratch-bc-averages`. Don't read other
  directories under `/home/vizia/avesnova`.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **Progress notes:** append brief milestone notes to
  `/tmp/bnb-scratch-bc-averages/progress.md`, `## HH:MM` heading (from `date`) and 1-3
  bullets: a part landed (with its commit), something failed, a change of plan. Start with
  a plan note.
- **Tests:** CPU tests for what you add, plus the quick tier; ruff clean; docs updated.
- If something here is wrong, missing or ambiguous, say so in your notes and final message;
  make a sensible choice and keep going.

## Done

- Commit and push to `feat/bc-averages` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, half a page:** what was built, each choice you made where this brief
  left one open, and anything left open.
