# One-head A/B on gpubox2: progress

Brief tried / worked / didn't / next notes, newest last (KST).

- 00:50 Reconstructed the champion from its run.json records (champion/). BC segments on
  fadd6065 / 6736d1d4 / e84ee16a / 09db261c; RL segments on 6af08e09 / 82e12ff9 / f45ada61.
  src/ of 09db261c, e84ee16a, 6af08e09, 82e12ff9, f45ada61 is identical; fadd6065 and
  6736d1d4 (BC 0-43M) lack only the physics-hidden-fade switch (off) and tournament reuse.
  The arm runs everything on 6af08e09 + these scripts. Next: box setup.
- 12:45 Done (the author ended the A/B after BC). gpubox2 (interruptible) verdict: 14 stops
  in 8.5 h, 27% up, ~4% of uninterrupted throughput: not worth it. gpubox3 (4090, on-demand):
  BC to 120.6M in 3.5 h after a 73-min cold compile. Verdict: one head is worse at BC (KL
  +0.089, agreement -0.053 at 120M, consistent from 30M on). Didn't: RL, tournament (cancelled).
