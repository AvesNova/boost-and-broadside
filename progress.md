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
