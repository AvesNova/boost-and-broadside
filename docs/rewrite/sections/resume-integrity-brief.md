# Brief: the resume test checks integrity, not rounding

The author's rule (contracts, "Resume: integrity, not bits"): a resumed run must continue
the same run within rounding; bit equality is not required. `tests/runs/test_resume.py`
compares the GPU cases' floats to 1e-4 after training continues. The hygiene round found
that a one-ulp change to every weight at the resume also fails at 1e-4: Adam turns
rounding noise in near-zero gradients into full steps within one update. So a resume
whose autotuner picks another kernel configuration could fail a test meant to allow it.

## What to do

Split the GPU resume check in two:
1. **The restore is exact.** In the resumed process, right after loading and before any
   compute, the whole training state equals what the first process held when it saved:
   parameters, optimizer state, every generator (the learner's named streams and the
   default CPU and CUDA generators), the rollout's carry, the league's state, the step
   and update counters, and the schedule's position. Compare bitwise: no arithmetic has
   happened, so rounding cannot enter. Dump the state in the first process at the save,
   then compare it with the resumed one. Enumerate the state from the checkpoint's own
   list, so a new field is covered automatically.
2. **Training continues as the same run.** After the resume, compare the straight run
   and the resumed one statistically, not elementwise: the update's losses, entropy, KL
   and the pool's reward rates within a relative tolerance that rounding drift passes and
   a different run fails. Integers and categoricals (counters, the opponents drawn,
   results) stay exact over the first update after the resume. Choose the tolerance from
   measurements: (a) a one-ulp weight change at the resume must pass; (b) a different
   seed for the minibatch order must fail.

Keep the CPU cases bitwise end to end.

**Break by hand:** leave one generator out of the checkpoint, then the optimizer state,
then the league's state. Each must fail part 1, with a message naming the field. Revert.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-resume`, branch `fix/resume-integrity` (from
  v2). Work only here. Scratch in `/tmp/bnb-scratch-resume`.
- **Do not read** any other directory under `/home/vizia/avesnova`.
- **Python:** `uv sync`, then always `uv run --no-sync ...`.
- **GPU:** every GPU run holds the benchmark lock (the wrapper
  `/tmp/bnb-scratch-fast-yemong/locked.py <command...>`). Another agent shares it.
- **May change:** `tests/runs/test_resume.py`, `tests/runs/resume_worker.py`,
  `tests/runs/runs_cases.py`, and the runs page's resume paragraph
  (`docs/explanation/runs.md`). Production code only if the state cannot be dumped
  without it, and then only by adding a read-only accessor.

## Done

- `tests/runs` passes under the lock, and the docs tests pass.
- Commit to `fix/resume-integrity` and push (`git push -u origin fix/resume-integrity`).
  Author Julian Kizanis <julianzia@gmail.com>; end every message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Your final message, in a few lines: what each part compares, the tolerance and how you
  chose it, and the hand breaks.
