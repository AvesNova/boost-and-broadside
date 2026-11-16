# Option G on v23 and its shield trims: deathmatch without fog

The balance check of 2026-11-16 evening, on the RTX 5090 box: `run-fog.sh candidates.json trims.json`,
2048 games per entry and condition, `--variants 2 --environments 4096`. Deathmatch is played
without fog at every size (the author's decision of 2026-11-16); elimination and Frontline keep fog.
Lines: /workspace/out/g-v23-fog/<condition>.jsonl.

Each cell gives Yongma's (side A) kill share ± se and the score. Bands (kill share for a 45–55% win
rate): DM 5v5 .494–.507, DM 10v10 .497–.504, elimination 5v5 .498–.545, elimination 10v10
.492–.526, Frontline 5v5 .491–.513, Frontline 10v10 .493–.508, DM 2v2 .481–.521, 1v1 none.
The deathmatch bands were fitted with fog on, so they are approximate here. Without fog, 2v2 has
about 7 kills a game; 1v1 still has about 0.02, so its share is noise.

| Candidate | deathmatch-5v5 | deathmatch-10v10 | elimination-5v5 | elimination-10v10 | frontline-5v5 | frontline-10v10 | deathmatch-2v2 | deathmatch-1v1 | in band | worst miss |
|---|---|---|---|---|---|---|---|---|---|---|
| v23 | 0.302±0.001 (score 0.001) | 0.316±0.001 (score 0.001) | 0.113±0.004 (score 0.038) | 0.190±0.003 (score 0.048) | 0.331±0.002 (score 0.035) | 0.341±0.002 (score 0.038) | 0.234±0.003 (score 0.028, 9.2 k/g) | 0.108±0.038 (score 0.605, 0.0 k/g) | 0 | 0.385 |
| G-v23 | 0.527±0.001 (score 0.434) | 0.528±0.001 (score 0.618) | 0.588±0.007 (score 0.525) | 0.571±0.005 (score 0.594) | 0.507±0.002 (score 0.553) | 0.536±0.001 (score 0.814) | 0.482±0.003 (score 0.137, 7.0 k/g) | 0.350±0.074 (score 0.195, 0.0 k/g) | 2 | 0.045 |
| Y0.50 | 0.518±0.001 (score 0.364) | 0.519±0.001 (score 0.513) | 0.554±0.007 (score 0.493) | 0.541±0.005 (score 0.545) | 0.497±0.002 (score 0.494) | 0.523±0.001 (score 0.765) | 0.472±0.003 (score 0.126, 7.0 k/g) | 0.267±0.066 (score 0.185, 0.0 k/g) | 1 | 0.015 |
| Y0.46 | 0.506±0.001 (score 0.315) | 0.506±0.001 (score 0.414) | 0.550±0.007 (score 0.481) | 0.505±0.005 (score 0.467) | 0.485±0.002 (score 0.445) | 0.508±0.001 (score 0.684) | 0.464±0.003 (score 0.124, 7.4 k/g) | 0.283±0.061 (score 0.184, 0.0 k/g) | 3 | 0.017 |
| J1.25 | 0.521±0.001 (score 0.388) | 0.522±0.001 (score 0.554) | 0.570±0.007 (score 0.509) | 0.548±0.005 (score 0.542) | 0.495±0.002 (score 0.510) | 0.526±0.001 (score 0.767) | 0.478±0.003 (score 0.139, 7.0 k/g) | 0.282±0.072 (score 0.176, 0.0 k/g) | 1 | 0.025 |
| Y0.50-J1.22 | 0.512±0.001 (score 0.345) | 0.515±0.001 (score 0.472) | 0.557±0.007 (score 0.497) | 0.527±0.005 (score 0.513) | 0.487±0.002 (score 0.460) | 0.514±0.001 (score 0.715) | 0.463±0.003 (score 0.109, 7.1 k/g) | 0.262±0.068 (score 0.169, 0.0 k/g) | 0 | 0.018 |

"in band" counts the seven banded conditions. "worst miss" is the largest distance outside a band.

Closest: Y0.46 (Yongma max_shield factor 0.46, Jujak's G factor 1.1586). It is in band on 3
conditions (DM 5v5, elimination 10v10, Frontline 10v10). Its other misses are DM 10v10 +0.002,
Frontline 5v5 -0.006, elimination 5v5 +0.005 and DM 2v2 -0.017.
