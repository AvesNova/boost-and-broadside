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

## 2026-11-12 14:25 KST: round 1 baselines; paused (handoff)

Paused at the author's request (the new bullet mechanics go in first; the loop restarts on
them). Every unit stopped cleanly; nothing is running.

**Built** (all committed):

- `src/bnb/controllers/scripted_doctrine.py`: `Doctrine` (one type's doctrine), `table`,
  `per_ship`, `pack`/`unpack`/`from_row`, `named`/`from_named`, `stored`/`set_table`
  (`scripted_doctrines.json`, by ship set and type key). The combat layer
  (`scripted_combat.doctrinal`, `advantage_weight`) applies it; `Scripted(...,
  doctrines=table)` and `decide(..., doctrines=)` take it; None (the default everywhere)
  flies no doctrine, the agent exactly as before.
- Elements and parameters (weights in [0, 1], 0 = off; logistics are centre, scale):
  gate `strength` x `in_advantage` (logistic of own/target turn^`turn_exponent` x climb,
  cruise, gun range exponents); `fight_in_gun_range` (the fight's range: extend and bleed);
  **extend** `extend`, `extend_in_stall_angle` (nose off the target), re-engage rule
  `engage_in_energy_ratio` (own/target v^2/2 + store, seen target) and
  `engage_in_store_fraction` (store floor); **slash** `slash`, `slash_in_gun_range` (boost
  in, never reverse); **bleed** `bleed`, `bleed_in_corner_speed`, `bleed_in_stall_angle`
  (reverse toward corner speed in a turning fight); **lag** `lag`, `lag_in_gun_range` (aim
  behind the target's track, weight (1 + cos) / 2 of the off-tail angle); **deny** `deny`,
  `deny_in_stall_angle` (target's nose on me), `deny_in_gun_range`, `deny_angle` (turn off
  the line, to the side already off its nose).
- Hand-set starting doctrines in `scripted_doctrines.json`: Yongma extend 1.0 within 3 gun
  ranges, nose > 3 stall angles, re-engage at energy ratio 0.9, store floor 0.15, slash 0.8;
  Jujak bleed 0.7, lag 0.5 (0.3 gun ranges), deny 0.8 (target nose within 1 stall angle,
  within 3 gun ranges, 0.5 rad).
- The old `energy_fight` fields left `ScriptedParameters` (stored sets and populations
  trimmed; `--relative-tactics` became `--own-time`).
- Ships `yongma.toml`, `jujak.toml` (`ships.py` here regenerates them; `ARMOUR` there is the
  round's armour knob, vizia's values now). Aspect classes four (Events schema 4). Game v5:
  a non-quiet input restarts the recharge wait.

**Bench** (`bench/doctrine_arena.py`, CPU, eager; 5v5 `real_scale_elimination`, no fields,
7800 m; lineups energy v turn, turn mirror, energy mirror, mixed Y-J alternating):

    PYTHONPATH=src python -m bench.doctrine_arena play --record R.jsonl --label L \
        [--doctrines FILE] [--doctrines-b FILE] [--off-a] [--environments 256] [--threads 4]
    PYTHONPATH=src python -m bench.doctrine_arena report --record R.jsonl [...] --out R.json
    PYTHONPATH=src python -m bench.doctrine_arena search --type yongma|jujak --log S.jsonl \
        [--generations 6 --children 7 --games 40]
    PYTHONPATH=src python -m bench.doctrine_arena card   # TTK in passes, armour, recharge
    PYTHONPATH=src:. python docs/rewrite/sections/doctrines/version4_recharge.py play ...

`off.json` here is the no-doctrine file. Cost: 256 games of one lineup about 8 min on 4
threads with four jobs sharing the machine (load ~15); finished games leave the batch every
60 ticks. A search generation (8 candidates x 40 games x 3 lineups) is ~25-30 min.

**Round 1 measured** (vizia's armour: head-on fraction 0.2 at 42 degrees, recharge 12/s
after 6 s; v5 rule; card head-on TTK 2.35 passes Y on Y, 2.24 J on J, 1.91 Y on J, 2.72 J
on Y; 256 games a lineup; A's score ± SE; damage shares mutual / offset / flank / tail).
Full table: `round1/table-baseline.md`.

| Doctrines | Lineup | A scores | Timeouts | Damage m / o / f / t | Kills m / o / f / t | Merges a kill | First-merge kills | TTK median s |
|---|---|---|---|---|---|---|---|---|
| off | Yongma v Jujak | 0.46 ± 0.03 | 0.00 | 0.46 / 0.24 / 0.27 / 0.03 | 0.30 / 0.28 / 0.37 / 0.05 | 0.56 | 0.51 | 2.9 |
| off | Jujak mirror | 0.55 ± 0.03 | 0.04 | 0.53 / 0.23 / 0.22 / 0.02 | 0.40 / 0.28 / 0.29 / 0.03 | 0.70 | 0.59 | 3.0 |
| off | Yongma mirror | 0.52 ± 0.03 | 0.11 | 0.42 / 0.27 / 0.28 / 0.03 | 0.26 / 0.31 / 0.37 / 0.05 | 0.58 | 0.47 | 2.7 |
| off | mixed | 0.54 ± 0.03 | 0.03 | 0.46 / 0.24 / 0.27 / 0.02 | 0.33 / 0.27 / 0.35 / 0.05 | 0.60 | 0.51 | 2.8 |
| hand | Yongma v Jujak | 0.26 ± 0.03 | 0.54 | 0.05 / 0.14 / 0.10 / 0.71 | 0.03 / 0.12 / 0.12 / 0.74 | 0.19 | 0.12 | 1.0 |
| hand | Jujak mirror | 0.48 ± 0.03 | 0.30 | 0.01 / 0.06 / 0.35 / 0.58 | 0.00 / 0.01 / 0.29 / 0.69 | 0.42 | 0.01 | 6.6 |
| hand | Yongma mirror | 0.48 ± 0.03 | 0.05 | 0.36 / 0.27 / 0.24 / 0.14 | 0.22 / 0.31 / 0.27 / 0.19 | 0.53 | 0.45 | 3.4 |
| hand | mixed | 0.49 ± 0.03 | 0.57 | 0.18 / 0.16 / 0.24 / 0.42 | 0.06 / 0.13 / 0.25 / 0.55 | 0.31 | 0.16 | 2.2 |

- Without doctrines the fight is jousting: mutual head-on is the largest damage class
  (42-53%), tail 2-3%; the killer landed the pair's first hit in 90-94% of kills; the side
  landing the game's first hit wins about 0.6. Energy v turn is even (0.46 ± 0.03).
- The hand Jujak doctrine (lag + deny + bleed) turns it into tail fighting: Jujak's damage
  is 96% from the tail, mutual head-on almost gone, and it beats Yongma 0.74. The hand
  Yongma doctrine extends but rarely comes back to a winning attack: 54% of energy v turn
  games and 57% of mixed games time out at 180 s. The aspect target is already met by the
  turn doctrine; the balance and the timeouts are not.
- Recharges a kill: 0.00 in every row (no victim recharged between its first hit and its
  death) under v5. The v4 comparison (`version4_recharge.py`, units doctrine-r1-v4-*) was
  stopped before its first batch: **no v4-against-v5 number yet**.
- The two round-1 searches (yongma, jujak) were stopped inside generation 0: no search
  result. The mixed lineup's A score in a 24-game smoke run was 0.29 ± 0.09 with identical
  sides; at 256 games it is 0.54 ± 0.03 (no seat bias seen).

**Next** (after the bullet mechanics land): regenerate the two cards (`ships.py`) on the
new mechanics; rerun the off/hand baselines and the v4 comparison; run the searches (the
Yongma doctrine first: its extension needs a re-engage that pays, or the search will turn
it off); round 2's design knobs: head-on fraction ~0.13 with a ~10-degree half-strength
angle for 3-4 passes (card), the recharge delay/rate, and the author's bullet direction
(slower muzzle, more spread, smaller hit radius) as a candidate round.

**Parameters against recompiling** (the arena is eager on the CPU, so nothing recompiles
there; for the compiled training tick and scripted step):

- The doctrine table is a tensor ((T, WIDTH) or (R, T, WIDTH)): new values written in
  place need no rebuild; a new WIDTH (a field added) or type count does. `doctrines=None`
  against a table is a Python branch: switching between them is a different graph. Each
  logistic's `rising` is a Python bool (structural).
- `ScriptedParameters` as a per-view table (`unpack`) is tensors; as one set of Python
  floats, its values are constants of a compiled graph (a change recompiles).
- Card fields (`ShipTypes`: head-on fraction and angle scales, recharge rate and delay
  scales, damage, shield, hit radius, muzzle speed, bullet life) are tensors: no rebuild.
  `Rules` fields (`head_on_fraction`, `head_on_angle`, `shield_recharge_*`, `bullet_drag`,
  `bullet_spread`) are Python floats baked into the compiled tick; `longest_bullet_moves`
  and `bullet_slots` set static shapes (a type's longer bullet life raises them).
