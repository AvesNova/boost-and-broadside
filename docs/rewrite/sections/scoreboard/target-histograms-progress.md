# Target histograms: progress

- 11:17 tried: target_histograms.py beside target_ranges.py (same collection; fine
  histograms, stats, occupancy on the current 70-bin supports, symlog knees for the critic).
  worked: CPU smoke 32 x 600. next: the full 2,048 x 3,600 collection on the GPU
  (bnb-histograms), power profile performance.
- 11:21 worked: full collection 2,048 x 3,600 in 53 s (performance profile), EXIT 0;
  stats.json + histograms.npz in target-histograms/data; 12 PNGs; target-histograms.md.
  Finding: damage/kill_death symlog spreads mass far more evenly (smaller knee, more even);
  outcome is bimodal beyond +-5.8 and linear is its best. didn't: symlog for next-state
  columns (not asked). next: report.
