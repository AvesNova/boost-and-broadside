# Teacher 6 progress (queue item 6, branch `teacher-6`)

The scripted teacher only where it is used, and a compact teacher record (training-path
audit findings 5 and 6). Side task: bisect core's cold compile (164 s against the 150 s budget).
GPU jobs are `bnb-6-*` units under the GPU lock, one at a time; logs and scratch worktrees in
`data/teacher6/` (not committed).

## 2026-11-28

- 08:35 bisect started: cold compile_check at 522e25f3 (heads-8c's baseline), dc0d5480 (8b
  merged, before 8c) and 857c9613 (8b + 8c), each with its own empty inductor cache. ETA 08:48
  (worst 08:55). Meanwhile: code reading and the design.
- 08:43 bisect done (performance profile, no other bnb unit; load 0.9-3.5 at the starts, the
  previous job's compile workers). Cold compile_check, per graph backend seconds:

  | tree | act ops / s | minibatch_loss ops / s | advance s | slot_actions s | cold total |
  |---|---|---|---|---|---|
  | 522e25f3 (8a + 8c, no 8b) | 4,613 / 37.1 | 1,627 / 24.2 | 20.0 | 9.8 | 151.5 s |
  | dc0d5480 (8a + 8b, no 8c) | 6,239 / 46.6 | 861 / 10.6 | 19.6 | 9.0 | 139.2 s |
  | 857c9613 (8b + 8c merged) | 6,325 / 47.0 | 1,627 / 23.9 | 20.6 | 11.0 | 163.9 s |

  Both merges add about the same: 8b's look-ahead in the scripted agent +1,626 traced
  operations in act, +9.5 s of act's backend; 8c's geometry heads +766 in the loss, +13.6 s
  of its backend. The guess (8b the cause) is half of it. 522e25f3 measured 151.5 s today
  against its 141.6 s at 05:05: run-to-run spread on this laptop is about 10 s, so the 150 s
  budget sits inside the noise of either tree alone.
- 08:45 first change in (5ec863cc): the scripted call out of act into its own graph
  (`scripted_actions`), league environments always, self-play only with the teacher; compact
  teacher record; nothing copied at weight 0. Quick tier lean/learn passes (106); compile
  check: 6 graphs, no break, no recompile (both scripted calls one graph), act 5,788 -> 555
  operations + scripted_actions 5,258 (the same scripted work, now its own graph), cold
  164.8 s (core's baseline 164.3): the compile is not cut, the scripted agent is traced once
  either way. Over the budget as core is; budget not raised.
- 08:49 smoke queue started (`teacher-6/trainer.py`, lean.train at 2,048 environments, 3
  updates each, its own cold cache): core BC, branch BC, core RL teacher off
  (teacher_cutoff=1.0: off from the first weight), branch RL teacher on, branch RL teacher
  off. ETA 09:35 (worst 09:50).
- 09:34 smoke runs done (performance profile, one bnb unit each, load 2.7-6.9 at the starts:
  the previous job's compile workers). 3 updates each, update 0 compiling; no recompile (the
  static graphs would raise). Collect is a 128-tick segment, mean of the four of updates 1 and 2:

  | | store (pinned + device records) | teacher record | collect segment | update alone, 1 / 2 | steps/s, updates 1 / 2 | peak allocated |
  |---|---|---|---|---|---|---|
  | BC, core 01d31655 | 3,345.5 MiB | 1,008.1 MiB | 2.71 s | 106.14 / 107.48 s | 8,971 / 8,848 | 1.16 GiB |
  | BC, branch | 2,601.5 MiB | 264.0 MiB | 2.69 s | 107.89 / 107.86 s | 8,843 / 8,833 | 1.16 GiB |
  | RL teacher off, core | 3,345.5 MiB | 1,008.1 MiB (copied at weight 0) | 3.07 s | 109.40 / 107.76 s | 8,619 / 8,729 | 1.23 GiB |
  | RL teacher on, branch | 2,601.5 MiB | 264.0 MiB | 2.98 s | 107.57 / 108.06 s | 8,779 / 8,734 | 1.28 GiB |
  | RL teacher off, branch | 2,337.4 MiB (after the drop) | 0 | 2.71 s | 107.86 / 109.27 s | 8,839 / 8,727 | 1.28 GiB |

  Teacher off: the scripted agent on the league half only saves 0.36 s a segment, 2.8 ms a
  tick (the audit's estimate: about half of 5.5 ms), 1.4 s an update (1.3% of it); the
  update's run-to-run spread (about 1.5 s) hides it in steps/s. The record: -744 MiB of
  pinned store with the teacher on, -1,008 MiB once off. BC speed unchanged within noise (both
  scripted calls run). The BC losses of the two BC runs (same seed) agree to the shown digits
  at every update (loss 4.959 / 4.958, 4.464 / 4.465, 3.945 / 3.945; bc 3.687, 3.300, 2.999 /
  3.000). Cold first collect 96.8 s (core) / 105.3 s (branch) in BC, 124.9 / 126.9 / 125.5 s
  in RL. Peak allocated in RL 1.23 -> 1.28 GiB (+0.05; not chased).
- 09:35-09:48 checks: the compact record's BC cross-entropy against the old whole joint
  record on the same tick (16 environments, 3 seeds, 48 own rows each): relative difference
  1.2e-7 / 0 / 0, joint entries within 3e-8. Quick tier lean/learn, model, eval, scripted 212
  passed; slow lean/learn 6 passed; ruff clean. Compile check (baseline reset for the new
  graph, budget kept at 150 s): 6 graphs, no break, no recompile, cold 164.7 s (load 1.2 at
  the start). Over the budget as core is.
- Next (for the author): the cold compile. Item 6 moves the scripted agent's 5,258 operations
  into a graph of their own but does not shrink them; cutting the compile means a smaller
  scripted graph (8b's look-ahead) or loss graph (8c), or a budget decision.
