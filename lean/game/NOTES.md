# lean/game notes

## What is implemented

- `lean/contract.py`: `State`, `StepOut`, `Obs`, `split_joint`, `join_action`, sizes.
  `SHIP_FEATURES = 27` (20 scalars, 4 flags, 3-type one-hot), `GLOBAL_FEATURES = 3`.
- `card.py`: the real-scale card (Yongma, Jujak, Vizia) and the shared rule constants,
  copied as literals from v2's `api.ship_types` / `api.rules_of` for
  `real_scale_elimination`. `lean` does not import `bnb`.
- `core.py`: `GameConfig`, `ship_constants` (gathered once per reset into
  `State.constants`), `medium`, `flight_tick` (from `lean_flight.py`), `bullet_motion`,
  `draw_noise`, `step`, `reset`, `reset_done`, `observe`.
- Step order: fire, flight, bullet motion, swept test (dense (E, N K, N)), first contact,
  damage (kinetic x cored closeness x aspect), boundary (constant radius), shields with
  the quiet rule, deaths as a mask, win check, clock, counters. No respawn, shrink, zones
  or fog. `step` never resets; the caller passes `StepOut.done` to `reset_done`.

## What was run (CPU, torch 2.15, Python 3.14)

- `tests/test_invariants.py`, first version: the two 300-tick finite/padding tests
  (3v5 and 20v20, each batched with smaller sizes) PASSED. The team-swap test's
  comparisons (done, died, mirrored result, damage, positions over 120 ticks) all held;
  it failed only on its last line, which demanded that some game finish in 120 ticks.
  That line was changed to demand that some damage was dealt. **The changed file was not
  re-run.** The run took 100 s, nearly all in the 20v20 case: too slow for "seconds";
  cut the environments or ticks.
- `tests/test_against_v2.py` (checks A and B): **written, never completed a run.** One
  attempt was killed by a 110 s timeout before printing anything (v2's import and eager
  stepping are slow on this CPU). **Not yet passing; unknown what differs.** Nothing in
  the lean step has been compared with v2 numerically.
- Not run: any GPU, `torch.compile`, an operation count, fields (every run had
  `fields=()`), the "sides" spawn in a test, check C and check D.

## Choices where the audit left it open (section 7.1)

- Q1 tripwire: not measured. The flight is `lean_flight.py`'s form; nothing was thinned.
- Q2 fields: the step has the medium in flight and bullets from the start;
  `GameConfig.fields` defaults to none. No four-field map is defined yet.
- Q3 end with no shrink: v2's health rule at the clock (5,400 ticks): the sum of shield
  fractions of living ships; more wins, equal is a draw (result 2).
- Q4 arena: default radius 4,200 m with the side rule. Boundary ramp 300 m (v2's 100 m
  times its map scale 3; assumed, not checked against v2).
- Q5 friendly fire: on; reported apart as `StepOut.friendly_damage`.
- Q6 lineup: a fixed cycle by rank within the team (`lineup`, default Yongma, Jujak),
  the same for both teams. Per-episode mirrored draws are not implemented.
- Q8: game version 20 real-scale constants as they stand today.
- Q9: the step emits (E, N) sums only (dealt to enemies, dealt to team-mates, taken); no
  (E, N, N) matrix. `damage_taken` is hit plus boundary damage received, before the
  shield's clamp.
- Q10: first contact only; ordering by two reductions (smallest tau, then the smaller
  miss among contacts within 1e-6 of it), no bit trick.
- Slots are team-contiguous (team 0 first); N is the largest total in the batch.
- "mixed" spawn: every ship uniform in the disc of 0.85 arena radii with a uniform
  heading, then the edge rule. "sides": v2's rule, with a random axis per environment.

## Known differences from v2

- The medium is not cached in the state: it is evaluated at tick start for ships and
  bullets (same points as v2's cache, so the same values, one more evaluation per tick).
- A ship that dies keeps its speed and nose rate (v2 sets the speed floor and zero).
- Killer: earliest tau, then lowest bullet index (v2: tau, miss distance, index).
- Arming distance and the classic closeness are dropped (unused by the real-scale card).
- `softmin` is the prototype's clamp form, not v2's guarded division.
- Result coding: 0 or 1 the winning team, 2 a draw, -1 not done (v2: +1, -1, 0).
- Stick index 1, 2 = fine, 3, 4 = sustained, 5, 6 = full, odd indices negative (v2's
  order). Power 0 coast, 1 boost, 2 reverse. Quiet = coast, stick index 0 to 2, no shot.
- "Fired last tick" in the observation is inferred from the bullet ring; a shot that
  hits on its first move is missed. `State` has no `fired` field.
- Team sizes of 0 are not handled (the game would end at once with that team wiped).

## Check first on the GPU

1. Run `tests/test_against_v2.py` to completion and read check A's table; the flight
   came from a prototype that matched v2, but this copy was renamed by hand.
2. Operation count per tick (target: flight under 1,000, the rest under 500).
3. Memory of the (E, N K, N) hit test at the evaluation sizes; about 30 temporaries.
4. `torch.compile` of `step` (the `SimpleNamespace` mutation in `_kick` and the Python
   branch on the number of fields are the likely snags).
5. `reset_done` draws every environment each call; confirm its cost and that generators
   on the device behave.
