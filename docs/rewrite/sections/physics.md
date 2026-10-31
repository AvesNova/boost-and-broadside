# Section 1: physics (and the perception code that reads its state)

Status: decisions nearly complete, 2026-10-31. The starting spec is the race spec
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

   This equals the race spec text. Boundary damage counts as damage, and damage equal to
   the shield breaks it (confirmed).
6. **Friendly fire is on:** a bullet is a bullet.
   - A bullet never hits its own shooter (confirmed).
   - Cost: the hit test pairs every bullet with every ship, about 2x the pairwise hit
     work.
   - The modes section's ally-damage terms become live.
7. **Spawns:** a random slot assignment.
   - An evenly spread formation, randomly permuted over the spawning ships and randomly
     rotated each spawn, so ships spawning together do not overlap (confirmed).
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
10. **Principles, refined:** repeatability, differentiability and throughput are traded by
    judgement, not by an absolute rule. Example: if repeatability costs 0.01%
    throughput, take it.
11. **Bullets have drag and interact with fields.** Drag acts on proper speed so it
    composes with the curved-space motion.
12. **Combat values will be tuned by feel later.** Defaults:
    - v1's training values where v1 has them: head-on fraction 0.3, about 13 shots from
      a spawn store, and a range of about 186 m with drag;
    - race values elsewhere.
13. **High-level handles drive the physics (the ship card, its own section after this
    one):**
    - designers set handles such as corner speed, and code solves the physics
      parameters;
    - the handles form a full coordinate system, so every solve pins all of them;
    - changing one handle leaves the others unchanged, and a whole new set is the same
      solve;
    - offline, float64;
    - successor of v1's `config/handling.py` (a log-linear solve plus a simulated
      correction), with the handle list from `ship-card.md` revised for the new model.
14. **Rewards and Yemong keep today's ideas and math, in new code.**
    - Yemong starts as the current v1 model, which is known to train.
    - The Yemong-review design (Griffin-only memory, the dual next-state head, damped
      rotary encodings for hidden tokens) is added later as an option for a head-to-head
      comparison.
    - The Yemong section must keep both buildable from config.

## Agreed: constants

- Ship-card parameters are tensors per ship type (one compile for any values; sweeps can
  vary them per environment).
- Fixed numerics (tick, soft-min exponent, bands) are Python numbers.
- Eager mode stays fully supported for compile-free tuning.

## Proposed: agent scope (awaiting the author)

One coding agent, launched with a written brief.

- **Workspace:** its own git worktree and branch (`section/physics`) and its own scratch
  folder.
- **Reads:** this file, the physics spec, the contracts, the plan, the current `src/bnb`
  and tests, and the goldens.
- **Does not see:** the float64 oracle or the v1 code.
- **Changes:**
  - `src/bnb/game/physics/`;
  - `src/bnb/game/perception/`: only what the state changes force (field access, the
    bullet representation). Sight rules and the observation contract stay fixed; the
    observation goldens check it. Option (a) of question 9; the perception section's own
    work (fog-off envs, the mode block, sight-test speed) stays separate;
  - new tests under `tests/sections/`.
- **Must not change:** contracts, runtime, the goldens or the spec docs. It files
  requests in its report instead.
- **Checks:** tests on the CPU, and the skeleton benchmark under the GPU lock.
- **Hands back:** commits and a report (decisions, benchmark changes, size, contract
  requests).

Then a blind review by a separate agent, then integration after the author has seen
both.

## Carried from the race review

- Add tests, including a promoted sight-line check, eager vs compiled, and fresh draws
  in CUDA-graph replays.
- Remove process-global settings.
- Delete the stale tools.
- Remove the `Preset.flight` trap.
- State as dataclasses.
