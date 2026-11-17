# Box scripts

Scripts rescued from deleted branches, for setting up and running training on a rented
multi-GPU Linux box (vast.ai): supervisor services that resume after a reboot, health
heartbeat, config check, and copying results back to the laptop or the Hugging Face hub.

They are unchanged from the source commits and **untested against the current code**. They
still refer to the old locations (`/workspace/boost-and-broadside`,
`docs/rewrite/sections/gpubox2/`, `docs/rewrite/sections/gpu-box/`, run names of 2026-11-09 to
2026-11-12, laptop host aliases `gpubox2`, `gpubox3`); adjust them before use.

Sources: branch `run/onehead-champion` (tip ebef01f2), directory `docs/rewrite/sections/gpubox2/`;
branch `perf/gpu-box` (origin only, tip 3497f549), directory `docs/rewrite/sections/gpu-box/`.
The last commit touching each file is given.

| Script | What it does | Source (commit) |
|---|---|---|
| `dry_run.sh` | the v2 dry run (BC 100M then RL 100M) under the GPU lock, every update's policy files kept | written 2026-11-16 |
| `release_run.sh` | the release run: dry run 3's settings with `--preset release_jujak_elimination` (Jujak-only clear elimination, the hemisphere spawn, the belief off, the opponents' pools at 5v5 / 4v5 / 3v5 with the learner smaller), BC 100M then RL to 200M | written 2026-11-17 |
| `eval_queue.sh` | the dry run's evaluation queue after the chain: averaging sweep and champion (`eval_champion.py`), suite, map scales, 5v5 curves, break-even, team sizes, media, charts; resumable, under the GPU lock | written 2026-11-17, steps tested small on the laptop CPU (see dry-run.md) |
| `fetch_eval.sh` | laptop side: copies the queue's outputs, the champion's policy and the chain's records to `review/dry-run/` | written 2026-11-17 |
| `setup.sh` | sets up or restores a box on its persistent volume: repository, locked venv, Python, uv/inductor/Triton/hub/bnb caches, runs/out/queue/logs, `env.sh` sourced by every shell, the heartbeat as a supervisor program; idempotent (re-run after a recycle) | written 2026-11-16, tested on the 5090 box |
| `supervisor.conf` | supervisor programs `heartbeat` and `onehead` (the pipeline driver), started at boot and restarted, so the box resumes alone | run/onehead-champion gpubox2/supervisor.conf (ef75c278) |
| `heartbeat.sh` | logs a UTC line a minute with GPU utilisation, memory, power; a "boot" line at each start, so gaps are downtime | run/onehead-champion gpubox2/heartbeat.sh (9f11922e) |
| `downtime.py` | reads the heartbeat log and prints interruptions and up/down totals | run/onehead-champion gpubox2/downtime.py (893d89ba) |
| `pipeline.sh` | resumable multi-stage driver: stages skipped when finished, resumed from the newest checkpoint, stale `running` records fixed after a stop, caches on the persistent volume | run/onehead-champion gpubox2/pipeline.sh (04563a5f) |
| `check_config.sh` | compares a run's resolved `--print-config` with a recorded champion config; fails on unexpected differences | run/onehead-champion gpubox2/check_config.sh (75f19d0a) |
| `fetch_champion.py` | downloads policies from the public hub repo with SHA-256 checks (standard library only) | run/onehead-champion gpubox2/fetch_champion.py (ef75c278) |
| `pull.sh` | laptop side: periodically rsyncs the box's runs (not checkpoints), heartbeat and supervisor log, trying each host, retried; run as a systemd user unit | run/onehead-champion gpubox2/pull.sh (dd8f7323) |
| `reach.sh` | laptop side: a minute-by-minute log of whether the box's SSH ports accept connections | run/onehead-champion gpubox2/reach.sh (944e951c) |
| `sync.sh` | box side: copies run records to git and uploads run directories to the hub; `--loop N`; guards disk by pruning the compile cache | perf/gpu-box gpu-box/sync.sh (f12acf0c) |
| `killpat.sh` | SIGTERM processes matching a regex, skipping itself and its ancestors | perf/gpu-box gpu-box/killpat.sh (4405b680) |
| `hub_fetch.py` | downloads a run's policies and records from the hub | perf/gpu-box gpu-box/hub_fetch.py (af5a16ae) |
| `hub_verify.py` | verifies local files byte for byte against the hub (optionally uploads missing ones) and writes a manifest | perf/gpu-box gpu-box/hub_verify.py (3497f549) |
| `hub_repair.py` | checks policy/checkpoint files against their recorded SHA-256 and the hub copy; re-uploads where the hub differs | perf/gpu-box gpu-box/hub_repair.py (3732c072) |

No environment/volume setup script existed on either branch; setup was done by hand and
described in `docs/rewrite/sections/gpu-box-log.md` and `gpubox2/log.md` on those branches
(deleted with them; recoverable from the commits above until garbage collection).
