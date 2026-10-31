# Section 1: physics (and the perception code that reads its state)

Status: decisions in progress, 2026-10-31. The starting spec is the race spec
(`physics-race/SPEC.md`, branch `v3-general`) and its PyTorch arm, already in
`src/bnb/game`.

## Decided (author, 2026-10-31)

1. **Flight model:** the race spec as it is. This is today's proposed energy model with
   gravity (1 g held in level flight, exchange commands, full-stick sink). The battery
   story is not adopted for v1.
2. **Engine at a high store:** a flat region, then less power as the store fills. The
   exact curve is chosen for speed and differentiability (the race's smoothstep from
   the knee qualifies).
3. **Gravity:** option C, keep the race model (flight-model items 9 and 10 resolved that
   way).
4. **General principles:**
   - Make the sim smooth and differentiable wherever it costs no throughput.
   - Repeatability is nice to have, never at the expense of performance.
5. **Shield rule:**
   - Damage greater than the current shield in a tick breaks it (shield 0); overkill is
     ignored.
   - Any damage to a ship whose shield was already broken at tick start kills it.

   This equals the race spec text. Pending confirmation: boundary damage counts, and
   damage equal to the shield breaks it.
6. **Friendly fire is on:** a bullet is a bullet.
   - Proposed: a bullet never hits its own shooter.
   - Cost: the hit test pairs every bullet with every ship, about 2x the pairwise hit
     work.
   - The modes section's ally-damage terms become live.
7. **Spawns:** a random slot assignment.
   - Proposed: an evenly spread formation, randomly permuted over the spawning ships and
     randomly rotated each spawn, so ships spawning together do not overlap.
   - Nothing reveals the seat; the strict xfail becomes a real test.
8. **Fair maps by symmetry,** with the symmetry chosen per mode:
   - opposite spawns (deathmatch) use a 180-degree rotation about the centre;
   - Frontline (a ring, clockwise vs counter-clockwise teams) uses a reflection through
     a line across the centre, with the direction roles randomized per episode.

   The generator draws one side and applies the mode's symmetry. Fields come in mirrored
   pairs, with one on the symmetry axis when the count is odd. Physics v1 needs rotation
   only.
9. **Agent inputs:** the agent gets the spec and the goldens, not the float64 oracle
   ("we don't want to poison it"). The oracle and goldens are updated for every spec
   change here before the agent starts.

## Recommended, awaiting the author

- **Scope:** the physics agent also ports `game/perception` (the sight test and observe)
  to its new state layout, accepted against the same goldens. The observation contract
  does not change.
- **Constants:**
  - Ship-card parameters are tensors per ship type (one compile for any values; sweeps
    can vary them per environment).
  - Fixed numerics (tick, soft-min exponent, bands) are Python numbers.
  - Eager mode stays fully supported for compile-free tuning.

## Open: combat numbers (see the v1 vs race table in the session notes)

- (a) Head-on damage fraction: 0.3 (v1 training since run 731) or 0.1 (race spec).
- (b) Ammunition from spawn: about 13 shots (v1: "lethality from energy, not volume") or
  about 68 (race: cost tied to engine power).
- (c) Bullet drag: v1's quadratic drag, or the race's drag-free bullets.

## Carried from the race review

- Add tests, including a promoted sight-line check, eager vs compiled, and fresh draws
  in CUDA-graph replays.
- Remove process-global settings.
- Delete the stale tools.
- Remove the `Preset.flight` trap.
- State as dataclasses.
