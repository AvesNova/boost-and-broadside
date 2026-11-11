# One-head A/B on gpubox2: progress

Brief tried / worked / didn't / next notes, newest last (KST).

- 00:50 Reconstructed the champion from its run.json records (champion/). BC segments on
  fadd6065 / 6736d1d4 / e84ee16a / 09db261c; RL segments on 6af08e09 / 82e12ff9 / f45ada61.
  src/ of 09db261c, e84ee16a, 6af08e09, 82e12ff9, f45ada61 is identical; fadd6065 and
  6736d1d4 (BC 0-43M) lack only the physics-hidden-fade switch (off) and tournament reuse.
  The arm runs everything on 6af08e09 + these scripts. Next: box setup.
