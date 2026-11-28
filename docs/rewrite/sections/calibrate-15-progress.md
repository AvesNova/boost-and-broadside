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
- 19:25 worked: 16v16 collected (256 x 8,000, 1,278 episodes, 429 s). Applied (a3b7dc4c):
  value damage +-5.4 -> 5.6, kill_death +-7.5 -> 8.2; next-state ranges by the 99.9% rule
  (along +-0.53, across +-0.60, heading +-0.039, speed +-3.6, energy +-460 knee 0.18, shield
  -100..0); field edge -1,500..6,200 knee 200; map edge 32 bins -3,900..300 knee 1,000; hit
  prior 0.9863 / 0.0109 / 0.0028; feature scales unchanged. Table: calibrate-15/table.md.
  Running: quick tier (bnb-15-quick). Next: slow learn, compile check, smoke. ETA 20:20
  (worst 21:00).
