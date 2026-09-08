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
Only defense zones can be captured. Team 0 advances the front toward `+5`; Team 1 toward
`-5`. A defense capture takes six uncontested seconds under the provisional flat-rate
rule. Simultaneous completion changes the front by zero and resets both meters.

Ships die and immediately reappear in their current spawn at 25 health. Friendly spawn
heals 12 health/second, enemy spawn deals 8 damage/second, and either defense deals 2
environmental damage/second. The circular soft boundary starts at radius 2600 and its
damage increases with distance outside it. A match ends at a net front lead of five or
after five minutes; timeout uses the sign of the front and zero is a draw. All values are
provisional Gate-1 tuning, not settled balance.

Controls:

- `W`/`S`: boost/reverse; `A`/`D`: turn; Shift: sharp turn; Space: shoot.
- Tab: select another allied ship. The white ring marks the controlled slot.
- C: follow/release the selected ship.
- F: fit the playable region; R/Home: reset to the full toroidal world.
- Mouse wheel: zoom; middle/right drag: pan.

Please judge attack/defend clarity, zone spacing and travel time, capture pacing,
defense damage, spawn healing, low-health respawn, simultaneous captures, whether combat
collapses into a point-centered scrum, and whether “win the fight, then commit” emerges.
