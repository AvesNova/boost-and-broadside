# r9m2 on the play maps: progress

Branch `eval/r9m2-maps` (from `origin/tune/maneuvers-2` d253044e, merged with
`origin/integrate/next-runs` 102f3255; no conflicts). The task: how the best scripted doctrine
r9m2 (`maneuvers-2/champion2.json`) plays on the ship candidates the author playtests, against
r9 on the same maps. Hard stop 07:50 KST. GPU: the laptop RTX 4070 (alone), power profile
`performance`.

## Method

- `make.py`: each map's Yongma and Jujak card fields as absolute card values on the bench's
  base card (the real-scale rows; the map's unlisted fields are the ship files', as in the
  play shell), flown by r9 and by r9m2 (r9 with every doctrine field and scripted parameter of
  `champion2.json`; checked equal to `player_from(champion2.json)`).
- maneuvers-2's `sizes.py`, unchanged: self-play, 1024 games a cell (10v10 512), 300 s games,
  fog on, the density rule's arena. Lineups: A v B (ship A in the Yongma slot) at 1v1 / 2v2 /
  5v5 / 10v10, the A v A and B v B mirrors at 1v1. r9 and r9m2 run in the same compiled
  loops (one per size), so the r9 numbers here are a same-run baseline.
- maneuvers-2's `table.py` for the post-merge choice entropy.

## 05:40 KST: run 1 (`../maneuvers-2/r9m2-maps/`): six maps x {r9, r9m2}, 1024 games

Running.
