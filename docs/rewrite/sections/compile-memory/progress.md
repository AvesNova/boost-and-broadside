# Compile memory: progress

## 2026-11-16 07:05 step 0: setup and the laptop OOMs read
- Box: RTX 3070 8 GB, 24 CPUs, 62 GB RAM (`ssh -p 12149`), clone /workspace/compile-mem
  at c370b243 (origin/feat/imugi-boundary), worktrees for 862116a9 and 1eee6860.
  `measure.sh` runs the full preset's command on fresh caches (inductor, Triton, bnb)
  under the GPU lock; `tree_memory.py` samples the process tree's RSS and PSS every 2 s.
- Read the laptop's 06:25 OOM (MemoryMax 16G) from the kernel log: the cgroup held
  anon 13.2 GB + shmem 3.8 GB. The main process alone: anon 7.9 GB, shmem 1.7 GB; 16
  compile workers (16 CPUs) + their pool parent, 5 ptxas. The laptop's /tmp is a tmpfs
  (23 GB, 16 GB used, 15 GB of it /tmp/torchinductor_vizia), so the default Inductor
  cache is RAM charged to the run's cgroup, and the swap (8 GB) was full.
- Next: the box measurements (current tree first).
