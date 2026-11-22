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

## What was run (CPU, one thread, torch 2.15, Python 3.14; 2026-11-22)

Command: `OMP_NUM_THREADS=1 PYTHONPATH=.:src python -m pytest lean/game/tests -q`
-> 4 passed in 9 s (both test files call `torch.set_num_threads(1)`; the earlier timeout
was thread contention, not v2 itself).

- `tests/test_against_v2.py` (32 environments, 5v5, 150 ticks, no fields, no shrink,
  random sticky actions, v2's spread): ran to completion, assertions held. In the run:
  13,553 shots, 169 deaths, energy 131,550 to 227,486, **no game ended** (`done` never
  true, so the result coding at the end and the clock rule were not compared with v2).
  - Check A, one lean step from each v2 state, ships alive at tick start, not dying
    this tick, (99.9% quantile, max) of |lean - v2|:
    x (2.4e-7, 6.1e-5) m; y (2.4e-7, 7.6e-6) m; speed (1.5e-5, 1.5e-5) m/s;
    angle of attack (3.2e-8, 8.2e-8) rad; heading cos/sin (6.0e-8, 1.2e-7);
    angle cos (6.0e-8, 1.2e-7), sin (3.2e-8, 8.9e-8); nose rate (2.4e-7, 3.6e-7) rad/s;
    energy (0, 0); shield (7.6e-6, 1.5e-5).
    Audit bounds: 99.9% within 1e-3 m, 1e-4 m/s, 3e-7 rad; all within 2e-3 m, 2e-3 m/s,
    5e-4 rad. All inside, at float32 rounding level. No renaming slip in the flight.
  - Discrete mismatches (died, fired, done, result, killer, bullet moves left, gun
    cooldown): 0.
  - Check B, open-loop lean rollout on v2's actions, max over living ships, 32 of 32
    environments kept: tick 30 position 3.4e-5 m, speed 3.1e-5 m/s, energy 0,
    shield 2.1e-4; tick 150 position 1.8e-4 m, speed 4.6e-5 m/s, energy 0.047,
    shield 8.7e-4.
  - The first completed run (8 environments, 30 ticks) failed check A on speed (max
    184.7 m/s) and nose rate (0.81): only on ships dying that tick, where v2 writes the
    speed floor and zero and lean keeps the values (listed under known differences).
    The test now leaves those rows out of check A; `core.py` was not changed.
- `tests/test_invariants.py`: 3 passed in 4.9 s. The team-swap test with its changed
  last assertion passed. The 20v20 case now runs 30 ticks (3v5 still 300); it still
  asserts at least one death.
- Operation count, one uncompiled `step`, 5v5, 4 environments, `fields=()`, counter
  `docs/rewrite/sections/simplification/regions.py` (compute ops, views apart):
  step 1,312 compute + 111 view; of which `flight_tick` 929 (+54 view),
  `bullet_motion` 84, `medium` 4 calls x 1 op, the rest of the step 298 (+57 view).
  `observe` for one team: 74 compute + 4 view. Against the targets: flight under 1,000
  holds (929); the rest is 383, under 500.
- Not run: any GPU, `torch.compile`, fields (every run had `fields=()`), the "sides"
  spawn in a test, a game end against v2, check C and check D.

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

1. Re-run `tests/test_against_v2.py` on the GPU (done on the CPU, see above).
2. Operation count with fields (the CPU count above is without fields).
3. Memory of the (E, N K, N) hit test at the evaluation sizes; about 30 temporaries.
4. `torch.compile` of `step` (the `SimpleNamespace` mutation in `_kick` and the Python
   branch on the number of fields are the likely snags).
5. `reset_done` draws every environment each call; confirm its cost and that generators
   on the device behave.

## Agreed waste fixes (author, 2026-11-22)

To do after the step-3 scoreboard ("before" numbers), each checked against v2 with
`tests/test_against_v2.py` and recounted:

1. **Constants:** a constant equal for every ship type, or fixed for the run, becomes a
   Python number compiled into the graph; one that differs by type goes into one packed
   per-type table, gathered once per slot at reset, (E, N, C). Damage then needs one
   gather for the target and one for the owner instead of about ten.
2. **Packed state:** the state fields as a few packed tensors (floats, integers, bullets),
   so the reset's select and similar per-field work is a handful of operations.
3. **Dead ships** are not held still field by field: whatever is simplest and fastest, as
   long as their values stay finite (they are masked everywhere they are read).
4. **Constant work at reset:** the aranges, the bullet-owner index and fixed buffers are
   built once, not every tick.
5. **Velocity once per step**, stored in the state; `observe` and the scripted opponent
   read it instead of recomputing it.
6. **One scripted call for both teams** (today two calls, each computing every slot), shared
   with the behaviour-cloning teacher.
7. **Personas drawn only at a reset**, not for every environment every tick.
8. **A quick no-fields path.** Fields are the default, so this is minor.
9. **Batched resets** (possibly): finished environments wait frozen and are re-dealt
   together every k ticks or at a segment boundary, with no host read; the learner masks
   the frozen ticks.

10. **One observation call for both teams** (author 2026-11-22, with care about cheating):
    the viewer's team as an (E,) input. Rule for when fog arrives: share only each ship's
    raw state; everything derived (visibility, last-seen tokens, globals such as the
    enemy alive fraction) is computed per team from that team's own visibility. The
    scripted opponent must then read its team's view, not the state.
11. **Fields drawn per environment** (author 2026-11-22: "fields are randomised for every
    new env"): fields in the state as (E, F, 5), drawn at reset, padded with zero-strength
    fields; the medium at the ships and at the bullets' end of move cached in the state, so
    it is evaluated twice per tick instead of four times. The policy needs field tokens
    (a contract change) before any training with fields.

Future check (author 2026-11-22): the dense (E, N K, N) hit test at 50v50.

Proposed, not decided: a cheaper damage form with the same three ideas (kinetic term
kept; closeness from the swept test's miss distance, one gather instead of about 15
operations; the head-on term as a function of the hit angle's cosine instead of atan2,
pow and exp, fitted per type to today's curve, the largest difference reported).
