# lean/scripted: notes

## What is implemented

- `opponent.py`
  - `ScriptedConfig`: the yardstick's gate centres, scales, floors, target rule, lead,
    stand-off, boundary margin and ramp, fallback ship characteristics, and the standard
    deviations of the persona population.
  - `personas(config, envs, slots, generator, device, spread)`: per-ship parameters as a
    dict of (E, N) or (E, N, 8) tensors. `spread = 0` uses no random draw and returns the
    yardstick for every ship.
  - `act(config, state, team, persona, uniforms, memory=None)` -> (E, N, 3) long.
    `uniforms` is (E, N, 3) float32 on [0, 1), one per head, from `draw_uniforms`.
  - `marginals(...)` -> the three independent distributions (stick 7, power 3, fire),
    which are the soft labels for behaviour cloning; `sample` draws from them.
  - `fresh_memory`: optional, see "Choices".
  - `engage`: the target choice and geometry, shared with the chaser.
- `chaser.py`: `chaser_act(config, state, team)`; deterministic, points at the nearest
  enemy with a stick level by error size, coasts, fires when within 0.05 rad and one gun
  range.
- `tests/test_scripted.py`: range and zero rows (opponent with and without memory,
  chaser, spread 0 and 1); marginals are distributions; spread 0 is one fixed setting and
  repeatable; a target on the left gets a left stick; the 5v5 match against the chaser.

## What was run and what was not

- `PYTHONPATH=. python -m pytest lean/scripted/tests -q`: 5 passed, 1 skipped (CPU).
  The tests use a random `lean.contract.State` built in the test file, not the game.
- **The match test was not run.** It skipped because `lean.game` could not be imported
  when this was written (`lean/game/__init__.py` imported a `core` module that did not
  exist yet). It is written against CONTRACT.md's `GameConfig()`, `reset`, `step`,
  `draw_noise` and has never executed, so expect to fix small things in it.
- Nothing ran on a GPU and nothing was compiled.
- Not checked against the real game: the stick side (`left_is_even`), the names in
  `state.constants`, the arena radius, whether `cooldown <= 0` means ready, and whether
  the bullet inherits the shooter's velocity (the lead assumes it does, as in v2).

## Operation count (measured, eager CPU, E = 256, N = 10)

Top-level `aten::` calls per `act` call, by the profiler:

| | all calls | without views and indexing (select, slice, unsqueeze, expand, unbind, flatten, to) |
|---|---|---|
| `act`, no memory | 203 | 166 |
| `act`, with memory (hysteresis and hold) | 217 | 176 |
| `chaser_act` | 74 | |

The audit's budget of 130 to 170 counts arithmetic; by that count the memoryless call is
inside it and the call with memory is 6 over. 54 of the calls are multiplies. Easy
further cuts if wanted: precompute the gate floor tensor and `ln(centre)` in the persona
(about 6), drop the per-call `torch.cat` for the stand-off by folding it into
`gate_centre` at `personas` time (about 5), cache `world_velocity` in the game state.

## Imugi behaviours kept

All in Imugi's own units and with the `real_scale_elimination` centres and scales:
nearest living enemy; first-order intercept with the drag law's time of flight, the
distance capped at 4.78 gun ranges, relative velocity (bullet inherits the shooter's);
P(turn) and the two level gates in stall angles (0.0106, 0.0775, 0.386) with the 0.02
floors and a 0.02 floor on the side; boost when slow (2.61 corner speeds, which at this
card means nearly always) or far with store (7.22), reverse inside 0.21 gun ranges,
floors 0.04; gun = aligned (1.01 target sizes, size clamped 1 to 45 degrees) times in
range (1.04), zero with no target, a cooling gun or too little store; boundary steer
inside 2.28 turn radii when flying outward with the 0.0554 to 0.52 ramp; logistics in
the log of the quantity with the logit clamp.

## Imugi behaviours dropped, and why

- Frontline layer, mode inference, keyed draws, type doctrines, manoeuvres: the audit's
  section 6.3 could not measure their worth in this mode.
- Dead reckoning and the last-seen memory: nothing is hidden without fog.
- The switch-rate commitments (target, turn side, goal, recovery): replaced by one
  comparison for the target; the committed turn side for errors near 180 degrees is
  gone, the side is the error's sign.
- The target weight in the gun gate (a fade for remembered enemies): always 1 here.
- "Hold straight when there is no target and the ship is near the centre": without a
  target the ship steers to the centre; in elimination no target means the game is over.
- The per-type power rule (the audit's optional 40 operations): not written.
- Temperature as a call argument: it is a persona field folded into `gate_sharpness`.
- The share-of-random-play ladder: not written (the caller can mix with uniform actions).

## Choices the audit left open

- **The eight gates are one tensor.** `persona["gate_centre"]` and `["gate_sharpness"]`
  are (E, N, 8) in `GATES` order, so all gates cost one log and one sigmoid.
- **Memory is an optional extra argument**, not part of the contract's signature:
  `act(..., memory=None)`. With `fresh_memory` the kept target gets its score multiplied
  by `target_keep_factor**2` (0.8: switch only to an enemy closer than 0.8 of the kept
  one's distance), and the action hold applies. Without it the opponent is memoryless
  and the hold is ignored. The caller owns the dict; `act` overwrites its entries. After
  `reset_done` a stale kept target is harmless (a dead or absent target scores infinity;
  a living one is only a 0.8 preference for one tick), so no reset hook was added.
  One memory dict serves one team; use two for two teams.
- **The yardstick** is spread 0, which has hold 1; whether it uses the memory
  (hysteresis) is the caller's choice and should be fixed before any measurement. The
  match test plays it memoryless.
- **Persona population** (options A, B, C, D, E of the audit): gate centres log-normal
  (sigma 0.35 x spread); temperature log-normal (0.3); stand-off half-normal (0.5 gun
  ranges) acting as a lower bound on the reverse gate's centre, so a kiter is a ship
  that brakes early (it does not turn away); `weakest_weight` half-normal, the target
  score being distance x (1 + weight x enemy store fraction); lead gain 1 +- 0.2;
  boundary margin log-normal; action hold uniform on 1..6 ticks with a phase. The
  widths are guesses, not tuned. Reaction delay, persistent aim noise, the common-target
  rule, extend-after-pass and the line-of-fire check (F) are not written.
- **Characteristics**: `characteristic()` reads `state.constants[name]` when present
  and the config's float otherwise. Names used: `gun_range`, `bullet_launch_speed`,
  `bullet_drag`, `stall_angle`, `corner_speed`, `max_acceleration`, `max_energy`,
  `hit_radius`, `bullet_energy_cost`, `arena_radius`. The fallbacks are Vizia's measured
  values, so with a game that provides none of the derived names, Yongma and Jujak are
  flown with Vizia's gun range, corner speed and launch speed (a few percent off). The
  target's angular size uses the shooter's `hit_radius` (all types have 12 m).
  `arena_radius` falls back to 3000 m, a placeholder: set it from `GameConfig`.
- **Velocity** is speed times the nose turned back by the angle of attack, with a
  refractive index of 1 (no fields).
- **Sampling** is inverse transform on each head with `uniform >= cumulative`.

## Measuring the exchange rate against v2's Imugi

Not done. The audit's route: write this opponent as a v2 controller (about 150 lines
against `bnb.contracts.Obs`, own rows first; `marginals` ports almost line for line,
with `Characteristics` supplying the ten values above and `obs.globals` the arena
radius), then in v2's elimination without shrink or fog, 5v5 Yongma and Jujak
alternating, seats alternated: (a) 512 games against Imugi with stored doctrines and
memory (the audit's 64-game figure for the equivalent cut-down Imugi is 0.55 +- 0.06),
(b) the same against v2's `Chaser` as a floor (Imugi scores 0.98), (c) v2's champion
against it when a GPU is free, which re-bases 1.49. The audit's `match.py` scratch
script is the harness.

## Check first on the GPU

- The match against the chaser, then the operation count under compilation.
- `torch.where(..., math.inf)` and the `argmin` over an all-infinite row (rows without
  a living enemy) give index 0; those rows are masked by `has_target`.
