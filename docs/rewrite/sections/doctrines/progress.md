# Doctrines: progress

Branch tune/doctrines (from integrate/next-runs). The task: Yongma (energy) and Jujak (turn)
as real ships, two scripted doctrines (one agent, a doctrine per ship type) tuned for
strength, and the game design (frontal armour, armour cone, shield recharge, the cards)
adjusted until winning play gives the wanted aspect mix; at most four rounds.

Targets (the check on the game design): head-on TTK 3-4 passes (card); most kills and
damage from the flank and tail; mutual head-on rare; energy v turn near 50%; mirrors
healthy (few timeouts, fights resolve).

## 2026-11-12 13:20-14:40 KST: the pieces

- Ships: `yongma.toml` and `jujak.toml` in the real-scale set (`ships.py` here: vizia's
  handles with the s = 1 deltas, solved; the gun range follows the set's 0.6 s bullet life
  at each one's cruise, so the set's longest bullet life stays 18 ticks). Not in any
  default lineup.
- Aspect classes: four (`ASPECTS`: mutual head-on < 10 degrees, offset head-on 10-45, flank
  45-135, tail > 135); Events schema 4; the game's numbers unchanged.
- Game version 5 (the author's addition): any input other than a quiet one restarts the
  shield's recharge wait at the delay, as a hit does.
- Doctrines: `bnb.controllers.scripted_doctrine` (a per-type table the agent reads per own
  ship; the old `energy_fight` fields moved out of `ScriptedParameters`), elements extend
  (with an energy/store re-engage rule), slash, bleed, lag, deny. Hand-set starting
  doctrines in `scripted_doctrines.json`.
- Bench: `bench/doctrine_arena.py` (`play`, `report`, `search`): 5v5 elimination, no
  fields, 7800 m; win rates with standard errors, aspect shares, passes per kill,
  first-merge kills, P(killer landed the pair's first hit), the game's first hit, shield
  recharges per kill, timeouts.
