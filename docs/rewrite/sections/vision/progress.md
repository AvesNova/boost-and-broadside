# Vision range measurement: progress

- 11:40 worktree ../v2-vision (branch vision-measure from core) with the uncommitted 686 m gun card (BULLET_SLOTS 12, gun_range 686) applied for the run only; game readout gun range 697 m. Script vision_measure.py written; CPU smoke (32 x 2,400) ran and analysed. Power profile: performance. Next: GPU run 1,024 x 3,600 as unit bnb-vision.
- 11:43 GPU run done (1,024 x 3,600 in 41 s, 3,394 games, EXIT 0). Analysis + charts done. Found: 59% of turning-fight contact losses are within 1,000 m (fields); TURN 2c p95 1,238 m matches 4R at the flown 313 m radius; engage gate (3 gun ranges = 2,091 m) truncates TURN labels. Next: page, commit.
