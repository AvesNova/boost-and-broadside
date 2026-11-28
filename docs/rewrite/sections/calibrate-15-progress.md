# Calibrate 15: progress

Queue item 15: re-measure the learner's supports and scales under the core rules of
2026-11-28 (items 12 and 13). Branch `calibrate-15`. Data and table: `calibrate-15/`.

- 18:56 started. Plan: one collection script (`calibrate-15/calibrate.py`: value returns,
  next-state, geometry targets, hit outcomes, scaled features) at 4v4, 1v1 and 16v16 on the
  GPU, then the analysis, the code, tests, compile check, smoke. ETA 20:25 (worst 21:20).
- 19:05 worked: 4v4 (1,024 x 8,000 ticks, 5,439 episodes, 327 s) and 1v1 (2,048 x 8,000,
  8,758 episodes, 311 s) collected on the GPU (performance, load ~1); 16v16 (256 x 8,000)
  running, ETA 19:13. Value coefficient 0.25 committed; target_ranges.py and
  target_histograms.py fixed to the current scripted_actions (CPU smoke 8 x 2,400 both run).
  First read: hit prior unchanged (4v4 0.9863 / 0.0109 / 0.0028); map edge badly balanced (6
  of 16 bins, largest 38%); field edge's largest bin 21%; feature scales fine. ETA 20:25
  (worst 21:15).
