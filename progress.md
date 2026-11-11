# Position-aliasing probe: progress

- 22:50 KST. Tree: cca3c1fc (feat/player-temperature, the commit the temperature sweep ran:
  batch 0 of deathmatch started ~17:58 KST, game_version 2; cca3c1fc only adds a docstring
  line to 6513532f). Policies fetched from julianzia/boost-and-broadside-runs; weight hashes
  match the sweep's identities.
- Tried: Rollout on CUDA (captures a CUDA graph, so a Python-side seat cannot run) -> CPU.
- Smoke (8 envs, 3 s, champion 121.6M): sanity KL(driver||B) ~1e-9; R8192 KL ~2.4e-5 while
  controls and colour aliases are 0.25-0.5. First sign: fully rotary-aliased.
- Next: full runs, 4 policies deathmatch; champion+null frontline; arena-input 12500 and fog
  variants for the champion.
- 22:58 Wrapped-set smoke (8 envs, 2 s, champion): every far shift F against its near twin W
  (F less a whole number of 8192 m per axis): KL(W||F) 1-3e-5 while KL(F||B) 0.16-0.37. The
  far enemy is read as its near twin mod 8192 m: the policy sees an 8192 m torus.
- 22:56 Launched 9 runs (64 envs x 40 s, stride 15 ticks, CPU, 4 at a time) + 2 wrapped
  runs (32 envs x 30 s). Wave 1 ETA 23:45 (worst 23:55); wave 2 ~00:35 (worst 00:50).
