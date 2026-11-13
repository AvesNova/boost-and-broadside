# Yongma and Jujak under r9m2: progress

Branch `tune/ships-r9m2` (from `origin/eval/r9m2-maps` 22764d67, with `bench/game_loop.py`'s
type-row lineups from `origin/eval/morning`). The task: re-balance the original two ships,
Yongma (energy fighter) and Jujak (turn fighter), under the scripted doctrine r9m2, keeping
Jujak the better turner (sustained and instantaneous) and Yongma the faster, tougher,
harder-hitting, better-diving ship; and a three-ship set with a middle ship. Hard stop 13:00
KST. GPU: gpubox7 (RTX 5060 Ti 16 GB, shared with a ratings job).

## Method

- `make.py <round>`: cards as factors on the frozen ship files (the Yongma slot on
  `yongma.toml`, the Jujak slot on `jujak.toml`), flown by r9m2 (`maneuvers-2/champion2.json`)
  with the Jujak slot's duel-extension gate at 0.7 (`g0.7`) or 1.0 (`g1.0`).
- maneuvers-2's `sizes.py`, unchanged: self-play, 300 s games, fog on, the density rule's arena;
  512 games a cell while searching (10v10 256, 20v20 128), 1024 to confirm. Lineups: energy v
  turn (Yongma's score, win = more kills, draws half) at each size, both 1v1 mirrors.
- `flight.py`: each card's sustained turn (full stick, settled), instantaneous peak turn,
  cruise, top and corner speed.
- `threeship.py`: the three-ship set in one card table (type-row lineups).
- `remote.sh`: a run in tmux on gpubox7.

Kill mix m / o / f / t: mutual head-on, one-sided head-on, flank, tail.
