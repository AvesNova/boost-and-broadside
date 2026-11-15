# Compile memory: results (interim, 2026-11-16 08:40)

Host RAM of a cold `torch.compile` of the full training preset (`real_scale_all_modes`,
the command in `measure.sh`), measured as the process tree's PSS (`tree_memory.py`, every
2 s; PSS counts each shared page once, so it is what the tree holds; RSS sums count the
forked compile workers' copy-on-write pages once per worker and read ~2.5x too high).

## Box measurement: c370b243 (origin/feat/imugi-boundary), 256 environments

RTX 3070 8 GB, 24 CPUs, 62 GB RAM; default compile workers (24); fresh caches.
Stopped at 27.7 min (the box's GPU lock was needed by the sweeps), in the learner's
compile. A full run without the sampler (same commit, 22:04-22:52 UTC) finished: first
update rollout 1055 s (the rollout stages' compile), learn 1758 s (the learner's
compile), 10 008 cubins, 4.3 GB of cache; total wall time 49 min.

| phase (seconds since start) | root PSS | workers PSS (24) | ptxas PSS | tree PSS |
|---|---|---|---|---|
| build, before compile (0-240) | 3.6 GB | - | - | 3.6 GB |
| rollout compile (245-1237) | 4.0 -> 7.9 GB | up to 3.4 GB | up to 1.1 GB (18 at once) | peak 10.4 GB (783 s) |
| between (capture, rollout; 1237-1435) | 7.9 GB | 0.7 GB (pool idle) | - | 8.4 GB |
| learner compile (1505-1662, cut) | 7.9 -> 8.4 GB, rising | ~2.0 GB | - | 10.4 GB at the cut |

Data: `data/head-c370b243-256-partial.csv`.

- The main process holds most of it and grows through the whole compile (+4.3 GB in the
  rollout compile, still rising in the learner's when cut): Dynamo's and AOT's graphs and
  Inductor's IR of very large graphs. By 27 min the cache held 3034 generated sources;
  the finished run compiled 10 008 cubins (Inductor's reductions compile several configs).
- The compile workers are ~140 MB PSS each at the peak (24 workers: 3.4 GB); ptxas up to
  1.1 GB. Capping the workers caps only this part: at 4 workers about -2.8 GB.
- The cache grew to 2.2 GB by the cut (4.3 GB at the end of a full run).

## Why the laptop OOMs at 16G (kernel log, 06:25)

The run's cgroup at the kill: anon 13.2 GB + shmem 3.8 GB. The main process anon 7.9 GB
(as on the box) + shmem 1.7 GB; 16 workers + pool parent; 5 ptxas. The laptop's /tmp is
a tmpfs (23 GB; 15 GB of it /tmp/torchinductor_vizia), so the default Inductor cache is
RAM charged to the run's cgroup and cannot be reclaimed with the swap full (8 GB, full).
On a disk-backed cache directory those ~2-4 GB are page cache and reclaimable.

## Open (not measured yet)

- The full peak of the learner's compile (the cut was in it).
- The bisect (862116a9, 1eee6860) and the worker-count sweep: compile runs on the box
  need the GPU while the sweeps hold it; the measurement runs are waiting on the
  permission to run them alongside the sweeps without the GPU lock.
