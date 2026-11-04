# Brief: rating lanes, lookahead, chained advantages and v1's batch

The first BC and RL runs showed that v2 trained at a quarter of v1's batch, that its
advantages are cut at every shard, and that its live ratings rest on 0–6 games per window
on an uncalibrated ladder. Fix all of that before the next long runs (author, 2026-11-04).
Evidence: `docs/rewrite/sections/rl-degradation-brief.md` and its branch's notes, and
`/tmp/bnb-scratch-rl-degradation/progress.md`.

## 1. The batch: 1,024 environments × 8 shards × 128 ticks

- Make this the default geometry for both PPO and BC (`TrainConfig.environments`,
  `shards`). v1 used 3,840 × 128 per update; v2 trains three modes, so twice v1's.
- Segments live in host memory where the device cannot hold them; the memory probe picks
  the micro-batch (`docs/rewrite/sections/portable-batch.md`). Confirm it fits on this 8 GB
  GPU and report the peak and the update time.

## 2. Advantages chained across shards

- Shards are consecutive segments of the same games, but each is now its own GAE problem
  with a critic bootstrap at its end (`bnb.learn.ppo` `shard_returns`, `prepare`). Compute
  the returns over the whole update's ticks, so a shard's end bootstraps only where the
  update ends (or the lookahead's end, below). The per-shard cut was right when shards were
  a memory knob; they are not any more.
- A test: chained returns over N shards equal the returns of one segment of N × length
  ticks.

## 3. Lookahead ticks

- After the last shard, play K = 128 more ticks (a `TrainConfig` field) on **every lane,
  training and rating alike, in the same tensors**. They are never trained on: they supply
  the returns and advantages' tail (rewards, values, dones) and the labels that need the
  future (the shot labels; the bullet lives 12 ticks).
- The next update continues from the end of the lookahead; nothing is replayed. The
  lookahead's ticks are not stored for the learner beyond what the targets need.
- Checkpoints and resume keep working (the carry is the state after the lookahead), and the
  resume-integrity test covers a run with lookahead.

## 4. Rating lanes

- **Extra environments in the same rollout**, in the same tensors, that play rated
  matchups and are never stored for the learner or counted in its losses. The rollout's
  shapes are fixed at capture: the lanes' assignment of slots is static, the weights in a
  slot may change (as the league's do).
- **Matchups:** any pair of loaded policies. At least, per mode: the live policy against
  the raw scripted agent; the league's averaged agent against it (the RL target is the
  averaged agent's rating); and the ladder's rungs against it, which calibrate the ladder.
- **Size:** measure first. Rollout time against the total number of environments (1,024 to
  ~8,192), and each mode's episode length in ticks. The worst case (every game to the
  300 s cap, 9,000 ticks; linear cost) gives, with six matchups, about ±25 Elo per matchup
  per 10 updates at 1,024 lanes (+7% time) and ±18 at 2,048 (+13%). Choose from the
  measured curve where the error stops paying for the time, report the curve and the
  choice.
- **Ratings from the lanes:** the live and averaged ratings, per mode, come from lane games
  against the scripted agent (1000) and the calibrated ladder.
  - The ladder bug: `bnb.league.gauge` rates rung p at 1000·p, but rung_0.95 scores 0.065
    against scripted. Rate the rungs from their lane games instead.
  - A warm start's live rating starts from its parent's, not 0, and no rating is logged
    before a matchup has games.
- **BC uses the lanes too:** its score stop reads the live policy's lane score against the
  raw scripted agent (not the softened teacher), plain score, target parity (0.5). The BC
  budget defaults to 50M steps, v1's time to parity: a run that reaches the budget stops
  with a checkpoint, and `--resume` with a raised budget continues it.

## 5. Entropy per action factor

- Replace the single entropy coefficient with one per action factor (power, stick, shoot),
  each defaulting to today's value, so behaviour is unchanged until someone sets them.
  Tuning comes later, from the new clone; do not tune here.

## 6. Dashboards

- Split the crowded panels by game mode: one `episode/end` panel per mode with a line per
  opponent kind, one rating panel per mode with the live and averaged ratings. Add the
  lanes' series (games, scores and ratings per matchup). The dead-panel test covers them.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-rollout-lanes`, branch `feat/rollout-lanes`
  (from v2). Work only there. Scratch and runs in `/tmp/bnb-scratch-rollout-lanes`.
  Do not read other directories under `/home/vizia/avesnova`, apart from the evidence
  paths above.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **GPU:** you are the only GPU job. Hold the benchmark lock for every GPU command:
  `uv run --no-sync python ~/.local/share/bnb/locked.py <command...>`.
- **W&B:** none; offline only in tests. Credentials are in `~/.netrc`; never print them.

## Done

- Each part with tests; the quick tier and the slow tests you touched pass; the GPU tests
  in the files you touched pass under the lock; `ruff check` and `ruff format --check` are
  clean; the docs that describe each behaviour are updated.
- A short GPU run at the new defaults (mixed preset, PPO from the clone
  `/home/vizia/avesnova/runs-v2/2026-11-04-mixed-wandering-quill/policy`, a few updates):
  its update time, memory peak, lane games per update per matchup, and the lane ratings.
- Commit to `feat/rollout-lanes` and push, in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end every message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message, one page: what each part does, the cost curve and the lane count chosen,
  the short run's numbers, and anything left open.
