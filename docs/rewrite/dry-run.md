# The dry run: the baseline's curves on every condition

The first baseline run (R0 at seed 0, wave 0 in [run-order.md](run-order.md)) is rated on every evaluation condition, not only at 5v5. That shows how, for example, 10v10 evolves while the agent trains at 5v5, which informs the final branch choices (the author's decision, 2026-11-16, in [arms-todo.md](arms-todo.md)). The launch itself follows [sections/launch/runbook.md](sections/launch/runbook.md); its settings files are being replaced by one settings file per run (arms-todo), so this page covers the curves only. Build notes are in [sections/dry-run-curves/progress.md](sections/dry-run-curves/progress.md).

## How: rating the snapshots after training

Every snapshot the chain keeps (BC and RL, `policies/step_<n>/`, at 1, 2, 3, 4, 6, 8, 10, 12, 16 and 20M steps, then every 5M) is rated in a rating pool of its own, one Bradley–Terry fit per condition with Imugi at 1000 (`bnb pool`, `bnb.league.rating_pool`):

- **Conditions:** the suite's ten head-to-head conditions (5v5 in all three modes, 1v1 and 2v2 in deathmatch and elimination, 10v10 in all three modes), each its mode's preset at that team size on the preset's own map, exactly as `bnb evaluate` plays it (`bnb.cli.named_game`).
- **Lineups:** each game's lineup drawn slot by slot, Yongma or Jujak with equal chance, both teams flying it, as the suite draws them (the pool's `--lineup-types`).
- **Players:** the exponential average (the averaged agent we ship) of every scheduled snapshot of both stages, about 26 each to 100M, and Imugi. `--kinds exponential_average live` adds the live weights as their own series (twice the games).
- **Break-even (optional):** `bnb crossover` on a subset of the snapshots, the ratio of Imugi's ships to ours at even odds per mode.
- **The figure:** one PNG of small multiples, a panel per condition, Elo against the chain's steps. BC and RL share one axis, with RL after BC's last snapshot, and a dashed line marks the hand-off. With break-even, a fourth row has one panel per mode.

**Why not extra rating lanes during training.** The lanes are rows of the training rollout and share its tensors, which are fixed at 5 ships a side when the graph is captured. A 10v10 lane would need every training environment widened to 10 slots, and break-even to 20, which would slow all of training by roughly 2–4×. 1v1 and 2v2 lanes would need benching in the lanes (the A2 team-size code). Rating afterwards costs training nothing, uses the tools the arms' ladder uses, and can be rerun or extended from its records.

## Cost (provisional, until the dry run measures it)

Each batch is appended to the pool's records as it ends, with its seconds, and each line prints the budget's end as a clock time. So the real cost is read off the run, and a stopped rating resumes from its records.

- **Per snapshot, all ten conditions:** about 3–7 minutes on the 5090. This scales the laptop's 5v5 ladder (140–370 games per checkpoint and mode to about ±25–45 Elo, 0.36 s a game) by the design pilot's seconds per game in each condition (the ten conditions together take 3.3× the three 5v5 ones). The 3.8× laptop-to-5090 rollout ratio converts it ([sections/arm-costs/numbers.md](sections/arm-costs/numbers.md)).
- **The dry run's 52 snapshots (100M BC + 100M RL):** about 2.5–6 hours on the 5090 for the ten conditions at ±45 Elo, plus about 10–20 minutes compiling the ten engines. The three 5v5 conditions alone take about 30% of that. Training takes about 2.7 hours, so this is the largest part of the dry run.
- **Break-even:** about 5 minutes per snapshot for the three modes (two 256-game batches a mode at a 0.1 log-ratio target, plus three engine compiles). Eight snapshots (10, 20, 50 and 100M of each stage) take about 40 minutes.
- **Cheaper if needed:** `--target-stderr 60` takes about 55% of the games. The conditions may also run one at a time (5v5 first), stopped and resumed at will.

## Commands

On the box after the chain has finished (the GPU idle), from the repository, with the runbook's environment (`source docs/rewrite/sections/launch/env.sh`):

<!-- not-runnable: needs CUDA and the dry run's policies -->
```bash
RUNS=/workspace/runs; CHAIN=mainline-s0; OUT=/workspace/ratings/dry-run
POOL=$OUT/pool
CONDITIONS="frontline_5v5 deathmatch_5v5 elimination_5v5 frontline_10v10 deathmatch_10v10 \
  elimination_10v10 deathmatch_1v1 deathmatch_2v2 elimination_1v1 elimination_2v2"

# the pool: the suite's drawn lineups; then both stages' scheduled snapshots
bnb pool create $POOL --lineup-types Yongma Jujak
bnb pool add $POOL --runs $CHAIN-bc $CHAIN-rl --runs-directory $RUNS

# every condition to about +-45 Elo per snapshot, every snapshot in one session
# (rerun the same line to continue: the stored games count, a fresh seed each time)
bnb pool rate $POOL --presets $CONDITIONS --target-stderr 45 --budget 20000 \
  --session-players 64 --runs-directory $RUNS 2>&1 | tee -a $OUT/rate.log

# optional: break-even at 10, 20, 50 and 100M of each stage (the first file at or after)
for RUN in $CHAIN-bc $CHAIN-rl; do
  for MILLIONS in 10 20 50 100; do
    STEP=$(ls $RUNS/$RUN/policies | sed 's/step_//' | sort -n \
      | awk -v t=$((MILLIONS * 1000000)) '$1 >= t {print; exit}')
    [ -z "$STEP" ] && STEP=$(ls $RUNS/$RUN/policies | sed 's/step_//' | sort -n | tail -1)
    bnb crossover $RUNS/$RUN/policies/step_$STEP/exponential_average \
      --target-stderr 0.1 --blocks 8 --block-environments 32 \
      --out $OUT/break-even/$RUN/step_$STEP 2>&1 | tee -a $OUT/break-even.log
  done
done

# the curves (CSV) and the figure (needs the tools extra: matplotlib)
bnb pool curves $POOL --presets $CONDITIONS --out $OUT/curves.csv \
  --figure $OUT/curves.png --break-even $OUT/break-even
bnb pool table $POOL --presets $CONDITIONS > $OUT/table.txt   # leaderboards and fit checks
```

- **Labels:** every series is named by its stage and weights kind ("BC exponential average", "RL exponential average"). Live weights appear only with `--kinds ... live`, as their own series.
- **Bounds:** a snapshot whose games give no finite rating (it won or lost everything) is drawn as an open marker at its bound. That happens mostly with early BC snapshots in the small conditions.
- **Re-drawing:** `curves` reads the pool's records only, so the figure can be redrawn anywhere the pool directory is copied, the laptop included (no GPU).
