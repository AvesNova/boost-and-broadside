# Brief: diagnostics (section 9), round 2

Round 1 built the collector, the sinks (JSONL, live W&B), the series the first BC and RL
runs need, all 21 trigger rules, the skipped-step action and the dashboards. It is merged
into v2. Its notes, `docs/rewrite/sections/diagnostics-notes.md`, list every spec item,
the coverage table, the deviations, and what is not built yet. Read them first, then
`diagnostics-brief.md` (the rules of the task, all still in force) and
`diagnostics-spec.md`.

## What to do

Build what round 1 left, in its priority order ("Not built yet" in the notes):
1. **The cadence groups:** `grad/`, `repr/`, the actor/critic split. They run as their own
   compiled function every 10 updates; measure them against the 5% amortised soft cap.
2. **The auxiliary heads' baselines and gains** (audit groups 7–11), and the outcome
   ledger (group 6).
3. **`input/`** (C1).
4. **Fog, physics and frontline analytics** (groups 13–14).
5. **The reward event share and the zero-sum residual.**
6. **Side results, lifespan and events.** That includes the seat record that the
   side-balance rule needs, so the rule stops being idle.
7. **Fog slicing** of the per-pool series.
8. **The replay of one environment every N updates (H1),** in the format
   `bnb.play.replay` reads.
9. **`critic/explained_variance_fit`.**

Then update the coverage table, and re-measure the per-update cost at the 4096-environment
cell if it fits in the time, or say why not.

## The GPU is training

The BC run holds the GPU for about 6 hours from the start of your task. **Do not run GPU
work until the coordinator tells you the GPU is free.**
- Work on the CPU: tests that need no CUDA, run with `CUDA_VISIBLE_DEVICES=""`.
- Write each GPU test as you go, and mark it `gpu`.
- When you are told the GPU is free, run the GPU tests, the cost measurement, and the full
  suite under the lock, once at the end.

If you finish the CPU work before then, stop at a clean commit and say so in your final
message. A fresh agent or the coordinator will run the GPU part.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-diagnostics-2`, branch `section/diagnostics-2`
  (from v2). Work only here. Scratch in `/tmp/bnb-scratch-diagnostics-2`.
- **Everything else** is as in `diagnostics-brief.md`: what to read, the boundaries, the
  test tiers, hand breaks, no network in tests, full words, no legacy.

## Commits and hand back

- Commit to `section/diagnostics-2` and push. Author Julian Kizanis <julianzia@gmail.com>;
  end every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Update `diagnostics-notes.md`: the items, the coverage table, the costs, and a Progress
  section.
- Your final message summarizes it in a page or less.
