# Symlog preview: progress

- 11:48 tried: symlog_preview.py (target_histograms.py's collection, keeping every raw target
  as float32 .npy in data/symlog-2026-11-25/, chunks per 300 ticks). worked: CPU smoke 32 x 900,
  analyse runs; event knees from the code: hit 21 points -> 0.575 x e^-3 = 0.0286, kill 2.83 x
  e^-3 = 0.141. next: full 2,048 x 3,600 collection on the GPU (bnb-symlog, profile
  performance), then analyse + charts.
- 11:49 worked: full collection 2,048 x 3,600 in 48 s (EXIT 0), 35.4M value rows, 42.7M
  next-state rows, all raw. analyse 25 s on the CPU.
- 12:05 didn't: the data rule as first written finds no knee for damage, kill_death and the
  shield (the density falls as a power law from the float floor up; no flat region), and the
  shield's crossing lands in float residue. tried: a flat-core check (the decade below the
  knee averages a slope above -0.25) with fallbacks (event knee for values, knee 1 for the
  shield). worked: along / across / heading / speed have clean cores. next: charts, page.
- 12:20 worked: 9 target PNGs + summary.png checked by eye; symlog-preview.md. next: commit.
