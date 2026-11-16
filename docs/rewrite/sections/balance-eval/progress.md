# Balance eval: progress (2026-11-16)

Branch `feat/balance-eval` from `origin/feat/imugi-boundary` (not merged).

## 17:05 KST: the switches built, CPU checks pass

- Tried/built: `bench/balance_sweep.py` (conditions, `BALANCE` settings: no fog, no fields,
  seeded spawns, whole episodes; sweep candidates and any-card pairs; kill share pooled with
  kills and deaths a minute per side; check and speed subcommands) and `bench/balance_worker.py`
  (the balance-final worker on it, 32 variants per card table by default). `bench.game_loop`:
  static fog switch (the sight test skipped: `perceive(fog=False)`), seeded spawns
  (`seeded_uniforms`, by job game index), first-match tallies (`match_kills`,
  `match_deaths`, `match_ticks`), a pluggable controller factory, `Loop.set_table`, a scored
  batch stopping once every first match is over.
- Worked: CPU eager check on every condition tried: fogged 0/B, no fields present, every
  living row visible; spawns repeat under a seed, differ under another, follow the game index
  across batch positions. Quick tests (53) and ruff pass.
- Didn't: torch's CPU generator keeps only 32 bits of a seed (seed << 32 + game gave the same
  spawns for every seed); the spawns use numpy's seed sequence of (seed, game) instead.
- Next: the box is off (author); the GPU check on the laptop through laptop_job.sh.
