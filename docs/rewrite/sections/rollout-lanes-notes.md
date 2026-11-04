# Rollout lanes, lookahead and v1's batch: notes

Branch `feat/rollout-lanes`; the brief is `rollout-lanes-brief.md`, with the author's changes
of 2026-11-04/05 relayed by the coordinator (the layout chosen by measurement; snapshot
opponents in the lanes, spaced by rating). Measured on the development laptop (RTX 4070
Laptop GPU, 8 GB, 46 GB host memory), the mixed preset, PPO warm-started from the clone
`2026-11-04-mixed-wandering-quill`, bf16, under the bench lock, nothing else on the GPU.

## The layout (part 1)

1,048,576 trained ticks per update (128-tick segments), a 128-tick lookahead on every lane,
1,024 rating lanes of the first mix (live and averaged agent against the scripted agent and
the ladder chain). Rollout: per update, the shards, the lookahead and the lanes; the learner
time is the update's (1 to 4 epochs, as the KL stop allows; it is the same work at every
layout).

| Layout | Played ticks per game | Rollout s / update | GPU reserved peak | Host RSS peak | Learner s / update |
|---|---|---|---|---|---|
| 1024 x 8 | 1,152 | 22.7 (chunk 32), 22.4 (chunk 8) | 5.74 GB (32), 5.40 GB (8) | 18.8-19.6 GB | 69-212 |
| 2048 x 4 | 640 | 21.8 | 6.82 GB (32) | 20.2 GB | 70-214 |
| 4096 x 2 | 384 | out of memory (32); 24.0 (chunk 8) | 7.55 GB of a 7.65 GB budget (8) | 19.9 GB | 70-213 |
| 8192 x 1 | 256 | out of memory, chunk 8 too | - | - | - |

- The rollout's time is linear in the games played: 1,024 games took 1.91 s per 128 ticks
  and 2,048 took 3.76 s (the training lineup alone), so a wider layout plays more lookahead
  ticks for the same trained ticks and is not faster.
- The trajectory already streams to pinned host memory in chunks (the rollout's ring of two
  captured chunks, copied on a side stream as the next one plays; staging allocated on the
  caller's thread). The chunk is now `TrainConfig.graph_ticks` (default 8, a memory knob a
  resume may change): at 1024 x 8, 8-tick chunks cost nothing (22.44 against 22.70 s) and
  save 0.35 GB. What binds the wide layouts is not the trajectory but what scales with the
  number of games: the game state, the carries and the captured graphs' private pools
  (1.0 GB at 4096 games, 2.0 GB at 8192), with the learner's `prepare` on top.
- **Chosen: 1024 x 8** (BC and PPO): the cheapest rollout of those that fit with room for
  the lanes, at the brief's v1-shaped batch. 2048 x 4 is 4% faster in rollout (under 1%
  of an update) but leaves 1 GB less for lanes. On a larger GPU 2048 x 4 is the better
  layout (twice the independent games per update); it is `--environments 2048 --shards 4`.
- The policy-gradient signal-to-noise comparison was not run: the RL investigation's probe
  measures a token standard error, which assumes independent tokens and so cannot see the
  difference between layouts; a fair comparison needs the variance of the update's mean
  push over repeated collections, about an hour of GPU per layout.

## The rating lanes (part 4)

Episode lengths: with the clone, most lane games run to the 300 s limit (frontline and
deathmatch live against scripted: mean 286-300 s), so a lane finishes about 1,152 / 9,000 =
0.13 games per update: the brief's worst case is the real case. Lane games per update per
moving matchup at 1024 lanes, first mix (about 170 lanes each): 15-23.

Cost, measured at 1024 x 8 with chunk 8 (rollout s per update, GPU reserved peak), with a
mix of ten network weight sets in the lanes (live, average, two newest snapshots, six
snapshot slots, since replaced): 0 lanes 17.4 s (5.26 GB); 1,024 lanes 30.8 s (5.51 GB);
2,048 lanes 41.4 s (5.77 GB). The intercept is about 3 s and the slope about 10.6 s per 1,024
lanes of mostly two-network games: the cost is the network rows, about 5.5 s per 1,024
network rows per update (the first mix, one network row per lane, cost 5.3 s per 1,024
lanes), not the number of weight sets. With a learner of 70-210 s per update, 1,024 lanes
are 2-6% of an update at the default mix (about one network row per lane on average).

The rating error, from a simulated climb (a live policy rising from 1000 to 2000 over 100
updates, every game at its expected score and at the 300 s limit, the worst case and the
measured one): with a rating-spaced snapshot chain, the live policy's anchored standard
error at the end was 70 / 49 / 35 points at 1,024 / 2,048 / 4,096 lanes; the chain's links,
not the live window, dominate it. Against the scripted agent alone a 2000-rated policy
with 300 window games has an error of about 180.

**The default (author, 2026-11-05):** 1,024 lanes; the lanes' network opponents are the
league's own slot entries (drawn by `p (1 − p)` against the live policy, so near-even), with
live and average against each, against each other, a small anchor share on the scripted
agent, and the ladder chain (`RL_MATCHUPS`). The dedicated snapshot slots were dropped as
more machinery than the goal (train to 1500) needs.

## TODO

- **Tune the rating lanes when the ratings need it** (high Elo): the lane count, the
  matchups' split and the anchoring. The numbers to start from: about 5.5 s per 1,024
  network rows per update; the simulated climb's anchored error of 70 / 49 / 35 points at
  1,024 / 2,048 / 4,096 lanes with a rating-spaced snapshot chain (six slots, a new
  snapshot 200 points above the newest; the simulation fed `LaneRatings` the expected
  results of every lane game, 0.128 games per lane per update); a direct
  rating against the scripted agent at 2000 is nearly blind (error about 180 from 300
  games).

## Open

- The memory probe still checks and refuses; it does not pick the micro-batch
  (`portable-batch.md`, decided but not built).
- The lanes' network players run one weight set after another in the act stage (six by
  default: live, average, four league slots); batching them would save only the per-set
  intercept (about 3 s per update at ten sets), as the cost is the rows. The act stage's
  compile grows with the sets (about 16 minutes at ten).
- The ladder's lower rungs gather 1-3 lane games per update each, most of them draws at the
  time limit; their ratings need many updates.
