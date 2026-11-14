# Rating allocator and shared pool: progress

Branch `feat/rating-allocator` from `integrate/next-runs` (997a700e).

## 2026-11-14 19:10 to 20:30: root cause and the design

- **Tried:** reproduced the 672-of-1152 allocation in a three-player simulation (scripted,
  two arms at +1500 and +1450 Elo, 2 blocks of 96): the old `allocate_blocks` spent 960 of
  1152 games on one anchor pair and 96 on the arms' pair.
- **Root cause:** the fit's prior was one virtual game against the anchor. A virtual game's
  information is p (1 - p) at the estimate, so after a sweep the estimate grows as the log
  of the games and n p (1 - p) stays at about one game's: the strong player's standard
  error never falls (175 Elo after 96 or 960 games). The greedy trace allocator, which
  values the next block at the current estimate, saw the largest variance there and kept
  choosing that pair; the arms' difference, which the games could resolve, was not a target.
- **Worked:** `bnb.league.design`: a Gaussian prior (1000 Elo about the anchor), games
  planned at the less informative end of a one-SE interval (at most 200 Elo), a continuous
  A-optimal design rounded one block at a time under players-per-batch and lanes, targets
  per player plus decision players and contrasts, and out-of-reach / hopeless rules. Same
  simulation with the arms' contrast at 20 Elo: 480 of 576 games to the arms, then stops.
- **Didn't:** a one-standard-error margin without a cap made every unplayed pair look
  saturated (the prior's 1000 Elo); a floors-then-greedy rounding left a ladder's first batch
  with no anchor games (the greedy trace is myopic); fixed: cap and sequential rounding.
- **Next:** the shared pool, the loop check, the CLI, the probe, docs, the refit of the
  live-elo records.

## 2026-11-14 20:30 to 22:00: the shared pool, the loop check, the probe

- **Worked:** `bnb.league.rating_pool` and `bnb pool create/add/import/rate/table/curves`.
  One file per writer process, merged on read, batch ids from uuid4 (imports: from the
  line's content, so a re-import adds nothing). Sessions plan on the posterior's marginal
  over at most `--session-players` playable players and re-read the pool before each batch.
- **Worked:** `bnb.league.goodness_of_fit`: deviance of the played pairs, and Bradley-Terry
  against mElo-2 by likelihood ratio. A planted rock-paper-scissors triple is found at
  p < 1e-6; a transitive field passes.
- **Didn't:** Pearson residuals: a 2600-Elo policy drawing one of 192 frontline games
  against the scripted agent gave z = +7.4 on one game. Replaced by deviance residuals
  (+2.5 for the same pair). The mElo fit's "cyclic share" on near-unpenalised vectors was
  0.99 on the frontline pool with a non-significant test: dropped from the report.
- **Refit of the live-elo records** (377 batches of bc-v19, bc-v20, rl-v20, rl-dm, rl-kl,
  rl-mix-kl and the four win/loss re-ratings, imported into a scratch pool): against each
  run's last live table, the median |difference| is 1 to 10 Elo (0.03 to 0.3 of the old
  standard error) for every kill-share curve. Larger: bc-v19's elimination and frontline
  snapshots (-53 and -125 Elo, 0.5 and 0.9 SE: they lose every game to the scripted agent,
  so the prior sets them), and the win/loss deathmatch re-ratings (+30 to +100 Elo,
  0.3 to 1.0 SE: each re-rating had a few hundred games, now pooled with the others').
  Goodness of fit: deathmatch (kill share) deviance 10.7 on 40 df; elimination 80.8 on 89
  (p 0.72), loops p 0.82; frontline 138.9 on 108 (p 0.024), loops p 0.16; the worst
  frontline pairs are the scripted agent against strong policies (a draw or loss in 192
  where the model predicts none: the logistic tail is thin there, not a cycle).
- **Probe:** `bench/rating_layout_probe.py`, smoke-tested on the CPU (1x2x16 and 2x2x16,
  a 5-second game). Pending on the GPU (free after about 01:10), bound 7 layouts x 4
  batches; it prints its end time per batch:

      systemd-run --user --unit=rating-layout-probe -p MemoryMax=8G \
        uv run --no-sync python bench/rating_layout_probe.py \
        players=scripted,/home/vizia/avesnova/live-elo/rl-v20/runs/rl-v20/policies/step_75497472/exponential_average,/home/vizia/avesnova/live-elo/rl-mix-kl/runs/rl-mix-kl/policies/step_75497472/exponential_average,/home/vizia/avesnova/live-elo/rl-v20/runs/rl-v20/policies/step_30408704/exponential_average \
        preset=real_scale_frontline layouts=2x96x32,4x96x32,8x96x32,4x128x32,8x128x32,2x96x64,8x128x64 \
        batches=3 device=cuda lineup=Yongma,Yongma,Jujak,Jujak,Jujak \
        out=docs/rewrite/sections/rating-allocator/probe-frontline.jsonl

- **Next:** the GPU probe; then choose `--blocks`, `--block-environments` and the segment
  length for `bnb pool rate` (now 2 x 96, as live_elo.sh); the open questions in the report.
