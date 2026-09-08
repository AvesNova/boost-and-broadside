# Frontline Gate-1 playtest

Run the playable prototype with:

```bash
uv run bnb play
```

If `uv` cannot access its cache, the equivalent local entry point is:

```bash
.venv/bin/bnb play
```

The five physical zones stay fixed while their roles rotate with the unwrapped front.
Their cyclic order is neutral → Team 0 spawn → Team 0 defense → Team 1 defense → Team 1
spawn → neutral, so the two capturable damaging defenses are adjacent. Team 0 advances
the front toward `+5`; Team 1 toward `-5`. A defense capture takes six uncontested
seconds under the provisional flat-rate rule. Simultaneous completion changes the front
by zero and resets both meters.

Ships die and immediately reappear in their current spawn at 25 health. Friendly spawn
heals 12 health/second, enemy spawn deals 8 damage/second, and either defense deals 2
environmental damage/second. The circular soft boundary starts at radius 2600 and its
damage increases with distance outside it. A match ends at a net front lead of five or
after five minutes; timeout uses the sign of the front and zero is a draw. All values are
provisional Gate-1 tuning, not settled balance.

Controls:

- `W`/`S`: boost/reverse; `A`/`D`: turn; Shift: sharp turn; Space: shoot.
- Tab: cycle allied ships, then spectator mode. No ring means all ships are scripted.
- C: follow/release the selected ship.
- F: fit the playable region; R/Home: reset to the full toroidal world.
- Mouse wheel: zoom; middle/right drag: pan.
- `+`/`]` and `-`/`[`: raise/lower game speed through 1x, 2x, 4x, and 8x.

Please judge attack/defend clarity, zone spacing and travel time, capture pacing,
defense damage, spawn healing, low-health respawn, simultaneous captures, whether combat
collapses into a point-centered scrum, and whether “win the fight, then commit” emerges.

Reproduce the batched scripted-v-scripted diagnostic on CUDA with:

```bash
.venv/bin/python benchmarks/frontline_scripted_suite.py \
  --device cuda --games 256 --output /tmp/frontline-scripted-suite-256.json
```
