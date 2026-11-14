# Before the arms: to-do

Work to finish before renting the 5090s for the pairwise arms, whatever arms are chosen. Started 2026-11-14. Decided items are marked; candidate arms are in [ideas.md](ideas.md).

## In progress

- [ ] **Seed isolation** (branch `feat/seed-isolation`). Initial weights keyed by parameter path; maps by (seed, environment, episode); actions by (seed, environment, tick, ship, part) through the policy's cumulative distribution; the scripted agent, seats, stagger and view offsets on their own keyed streams. Still to do after it lands: a steps-per-second check on the GPU.
- [ ] **Cheese agents** (branch `feat/cheese-agents`). Simple scripted agents that exploit one behaviour: zone rusher, turtle, evader, deathball, jouster, kiter. Available as league members and as an exploit check in evaluation.
- [ ] **Rating allocator and shared pool** (branch `feat/rating-allocator`).
  - Fix the bug where most games go to saturated pairs.
  - Allocate by expected information, as a design distribution rather than greedy, with a limit on distinct players per batch.
  - Loose SE floor for the ladder, tight SEs for decision checkpoints and contrasts.
  - One pool for every arm on the shared volume, safe for concurrent writers.
  - Loop check: goodness of fit and a likelihood-ratio test against mElo.
  - A probe of tournament batch layout. The GPU run waits for the laptop GPU.

## First, before the game freeze

- [ ] **Clock ends (58).** Choose each mode's free knob so most games end by the objective and some end on the clock, with progress deciding the winner:
  - deathmatch: a fun time limit, then tickets to fit it;
  - frontline: the time limit;
  - elimination: the boundary's shrink rate.

  Tune on a fixed reference matchup.

## Decided, to implement

- [ ] **Snapshot schedule (decided 2026-11-14).** RL snapshots at 1, 2, 3, 4, 6, 8, 10, 12, 16 and 20M, then every 5M. `policy_snapshot_every` takes one number today, so it needs a list plus a tail.
- [ ] **Deathmatch outcome: win/loss/tie by the kill difference (`kill_lead`), decided 2026-11-14.** Like a sports result, regardless of margin. Training showed no difference from kill share.
- [ ] **100% fog in training, BC and the game (decided 2026-11-14).** Full information only for debugging. The scripted agent already reads only visible enemies (`scripted_combat.py`, `scripted_memory.py`).
- [ ] **Lineups (decided 2026-11-14).** Each slot is a coin flip between Yongma and Jujak, the same for both teams.
- [ ] **Mode mix in sixteenths: 6/16, 5/16, 5/16.** The 6 goes to the most starved mode (to discuss). Today's mix is frontline 8/16, deathmatch 5/16, elimination 3/16.
- [ ] **League snapshot schedule.** About 100–200 Elo between league members, read off the RL curve. Re-derive it after the game freeze.
- [ ] **Rating prior.** Replace the ±1000 prior around Baekho (to discuss: a random walk along each run, or the median of earlier runs at matched steps).
- [ ] **BC fixed at 100M** for every arm. The plateau stop only logs.
- [ ] **One committed settings file per arm.** No hand-typed overrides.

## To write down before the first run

- [ ] **The decision rule.**
  - Endpoint Elo (mean of the last 2–3 snapshots) in one shared pool, confirmed by a direct head-to-head of the final checkpoints.
  - Extend both arms when the gap is within noise and the slopes differ.
  - How modes are weighted, or "no regression in any mode".
  - The SE target, from the 2 × 2 seed measurement.
- [ ] **The run budget and the arm list,** costed by class: RL-only forks from one BC, or full BC+RL.
- [ ] **The evaluation suite used to pick winners.** Zero-shot play on other map sizes and team sizes, and play while outnumbered (to discuss).
- [ ] **The figures for the paper,** including Elo over steps for every arm from the shared pool.

## Box setup

- [ ] A setup script that creates the volume layout (runs, caches, ratings, repository and virtual environment) and sets `CACHE`, `RUNS` and the uv and Hugging Face cache paths in the shell profile.
- [ ] Supervisor services that resume runs after a reboot.
- [ ] A periodic copy of the volume to the laptop.
- [ ] First jobs on the box: 2 arms × 2 seeds, which measures the noise of the difference between arms; and the steps-per-second check of the seed streams.

## Cheap checks (no arm needed)

- [ ] **Seat symmetry (47).** A probe that predicts the viewer's team from the trunk's features should score no better than chance. A mirror match, a checkpoint against itself scored by seat, should come out 50/50.
- [ ] **Old log-probabilities (11).** The KL between the rollout's recorded probabilities and the learner's recomputed ones.
- [ ] **DPPO divergence.** The distribution of the total variation between old and new policies under today's clipping, to see what `dppo_delta` = 0.15 would mask before running the DPPO arm.

## To discuss

- [ ] **A new name for the scripted agent.** Speech-to-text can't pick up "Baekho".
- [ ] **The evaluation suite's conditions and how they combine.** Generalisation decides; choose the datum points.

- [ ] **Freezing the game.** Deathmatch on win/loss; clock and ticket tuning so about 25% of games end on the clock; frontline's minimum team size; bullet-drag units (75); pending flight and balance work. One game-version bump, before the first arm.
- [ ] **Removing fields from the game and masking their slots.** Check whether it exists; benched ships are already masked.

## Early arms (decided)

- **Variable map sizes against fixed.** It decides whether variable map sizes stay in the game and the paper. If the agent trains well on variable maps, they stay. If not, drop the requirement and re-tune the colour codes and rotary frequencies for the fixed sizes.
- **Variable ship and field counts against fixed.**
