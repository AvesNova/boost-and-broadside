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
