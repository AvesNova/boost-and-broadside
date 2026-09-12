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
spawn → neutral, so the two capturable damaging defenses are adjacent. Every zone has a
330 px radius. Team 0 advances the front toward `+3`; Team 1 toward `-3`. A defense
capture takes 8 seconds at a one-ship lead. Capture pressure is harmonic in the *net*
ship advantage inside the defense: a lead of `n` advances the meter at `H(n) = 1 + 1/2 +
… + 1/n`, so `2v1` and `4v3` both take 8 seconds, `2v0` takes 5.3, and `4v0` takes 3.8.
Each further ship of the lead is worth less than the last, which lets combat dominance
convert into territory without making one blob the whole game. A defending lead
stabilizes on the same curve. Equal counts pause the meter.
Simultaneous completion changes the front by zero and resets both meters.

Ships die and immediately reappear in their current spawn at 25 health. Friendly spawn
heals 12 health/second, enemy spawn deals 8 damage/second, and either defense deals 2
environmental damage/second. The circular soft boundary starts at radius 2600 and its
damage increases with distance outside it. A match ends at a net front lead of three or
after five minutes; timeout uses the sign of the front and zero is a draw. All values are
provisional Gate-1 tuning, not settled balance.

At episode start, every scripted ship independently draws an offensive (50%), defensive
(25%), or timid (25%) tendency. With neither defense under attack, offensive ships first
gather one-third of the way from their spawn to the enemy defense. The wave advances
once every living offensive ship has reached or passed that rally; a respawn naturally
causes another regroup. Defensive ships hold their own, and timid ships follow their
team's non-timid majority (a per-team episode coin breaks ties). Timid followers join an
offensive rally but do not count toward its readiness. Uncontested defensive ships orbit
just outside their damaging point, alternating direction to reduce bunching. If an enemy
comes within local engagement range they leave the patrol to fight. If enemies occupy
exactly one defense, both fleets converge on it; if enemies occupy both defenses, ships
return to their tendencies. Thus “contested” includes an undefended capture attempt,
not only a fight with both teams already inside the point. Nearby enemies take priority
over point orders.

Timid ships retreat below 30% health and remain in their spawn until fully healed; they
also fully heal after every respawn. Offensive and defensive ships normally fully heal
after respawning, but local fights or a contested defense interrupt that healing. A
ship's tendency survives death and is redrawn only for a new episode.

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
