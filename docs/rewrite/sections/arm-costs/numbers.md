# Arm costs: the extracted numbers (2026-11-15)

`extract.py` reads the run records in `/home/vizia/avesnova/box-5090-backup/runs/` (each run's `run.json` and log) and the laptop live-Elo logs in `/home/vizia/avesnova/live-elo/*/live_elo.log`:

<!-- not-runnable: needs the run records on the laptop -->
```bash
uv run --no-sync python docs/rewrite/sections/arm-costs/extract.py \
  /home/vizia/avesnova/box-5090-backup/runs /home/vizia/avesnova/live-elo
```

- **GPU:** from each record's memory probe (32 GB is the rented 5090, host `0af31102a785`; 8 GB is the laptop's 4070, host `vizia`). rl-v20-seed1 ran on the laptop.
- **Steady:** the mean of updates 3 onward, periodic saves included; every update is 1,048,576 steps (2048 environments × 512).
- **Segment 1:** the run's first segment in `run.json`, from start to end time, so compile and saves are included.
- **Ladder:** per rated checkpoint, the games over the three 5v5 modes and the summed batch seconds; the standard error each mode stopped at was about 25–35 Elo. The wall minutes include each checkpoint's start-up and compile.

## Output

| run | GPU | updates logged | first update s | steady rollout s | steady learn s | steady SPS | segment 1 steps | segment 1 wall h | SPS incl. compile |
|---|---|---|---|---|---|---|---|---|---|
| bc-v19 | RTX 5090 (32 GB) | 26 | 409 | 3.17 | 45.01 | 21,763 | 27262976 | 0.45 | 16,892 |
| bc-v20 | RTX 5090 (32 GB) | 81 | 401 | 3.08 | 44.85 | 21,877 | 84934656 | 1.18 | 20,032 |
| bc-v21 | RTX 5090 (32 GB) | 4 | 211 | 3.08 | 44.77 | 21,917 | - | - | running |
| rl-v20 | RTX 5090 (32 GB) | 166 | 755 | 6.32 | 39.05 | 23,114 | 149946368 | 1.98 | 21,066 |
| rl-dm | RTX 5090 (32 GB) | 72 | 329 | 5.91 | 38.40 | 23,665 | 75497472 | 0.97 | 21,707 |
| rl-kl | RTX 5090 (32 GB) | 72 | 167 | 5.93 | 38.31 | 23,704 | 75497472 | 0.92 | 22,795 |
| rl-mix-kl | RTX 5090 (32 GB) | 72 | 181 | 6.05 | 38.32 | 23,632 | 75497472 | 0.93 | 22,638 |
| rl-v20-seed1 | laptop RTX 4070 (8 GB) | 143 | 1431 | 23.15 | 262.82 | 3,667 | 149946368 | 11.68 | 3,566 |

| pool | checkpoints | games per checkpoint (3 modes) | batch seconds per checkpoint | wall minutes per checkpoint (median) |
|---|---|---|---|---|
| bc-v19 | 3 | 1920 | 433 | 8 |
| bc-v20 | 9 | 1813 | 456 | 8 |
| rl-dm | 8 | 1032 | 372 | 5 |
| rl-kl | 9 | 939 | 364 | 6 |
| rl-mix-kl | 6 | 1280 | 472 | 7 |
| rl-v20 | 19 | 1182 | 383 | 7 |

## Read-offs

- **BC, 5090:** about 21,900 steps per second steady; 20,000 including compile (bc-v20, 84.9M in 1.18 h). 100M ≈ 1.4 h.
- **RL, 5090:** 23,100–23,700 steady (rl-v20 at commit 5b85b6c2 learns in about 38–39 s per update; the later commits about 38.3 s); 21,100–22,800 including compile. 100M ≈ 1.3 h; rl-v20's 150M took 1.98 h.
- **Compile:** the first update takes 170–330 s with a warm cache and 400–755 s cold (BC 400 s, RL 755 s on rl-v20).
- **Laptop 4070 against 5090 (rl-v20-seed1 against rl-v20, same configuration but micro-batch rows 96 against 480 and a host store):** rollout 23.2 s against 6.1 s (3.8×), learn 263 s against 38–39 s (6.8×), wall 11.7 h against 2.0 h (5.9×). The rollout (policy inference plus the game) is the closest proxy for evaluation games; 3.8× converts laptop engine-hours to 5090 hours below. It's a proxy, not a measured evaluation ratio.
- **Ladder, laptop:** 370–470 batch seconds per checkpoint for 940–1,280 games at 5v5 (RL pools); 5–8 wall minutes.
- **Mode mix barely moves the speed:** deathmatch-only rl-dm and rl-kl run at the same steady rate as the all-mode runs (23,700 against 23,600).
