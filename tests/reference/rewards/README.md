# Rewards and Frontline equivalence fixtures

Goldens computed in float64 on the CPU from v1's own code at tag `v1-torch-final`. They
cover the damage ledger, the reward levels and Frontline's capture rules. The v2
compositions are each section's own tests (zero-sum, ledger conservation); these goldens
pin the v1 math they are built from.

- **Run:** `uv run --no-sync pytest tests/reference/rewards`. The tests skip until
  `rewards_arm.make_arm()` returns an implementation.
- **Fixed:** the tests, the helpers, the conftest and the goldens. Only `rewards_arm.py`
  is written by the sections.
- **Validated:** the tests pass on v1, and each of 20 deliberate mutants of it fails by a
  value comparison.
- **Tolerance:** float64, 1e-9 relative to each array's own scale.
- **No wrap:** v1 was toroidal; v2's arena is an open disc. Every case keeps its
  displacements far inside v1's world, so v1's wrap was the identity.

## Conventions

- **Sim order:** ships `(B, N)`, team-major; pairs `(B, N_attacker, N_target)`; zones
  `(B, 5)`. `team` is 0 or 1 per ship.
- **Applied vs raw:** `raw` is the hit damage before the shield. `combat_applied` and
  `boundary_applied` are what the shield actually lost.
- **The ledger** is each (attacker, target) pair's outstanding damage, ship attackers
  only. The boundary's share of a deficit is what the ledger leaves of it.
- **Frontline roles** are indexed 0 team-0 spawn, 1 team-0 defense, 2 neutral,
  3 team-1 defense, 4 team-1 spawn. Positive front favours team 0.

## Interface

### Ledger

- `accumulate_ledger(ledger, raw, combat_applied)`: the ledger plus each attacker's share
  of the applied damage, in proportion to raw impact.
- `attribute_recharge(ledger, recharge, deficit)`: returns
  `(attributed (B, N, N), unattributed (B, N), ledger)`. Recharge is split pro rata over
  the deficit: fraction `min(recharge / deficit, 1)`, 0 for a zero deficit.

### Levels

Each level returns its events, before weights.

- `enemy_team_share(value, team)`: each ship's value split over the opposing team's slots
  (dead included), divided by the recipient team's size.
- `outcome_events(team, result, done)`: `(B, N, 1)`, ±1 or 0 by team. `result` is +1 team
  0 won, −1 team 1 won, 0 a draw.
- `kill_death_events(team, combat_death, boundary_death, raw, ledger)`: `(B, N, 4)`
  death, kill, ally kill, unforced.
- `damage_events(team, combat_applied, boundary_applied, raw, recharge,
  recharge_attributed, unattributed, unit)`: `(B, N, 6)` taken, dealt, friendly fire,
  recharge, charge-back, unforced, in units of `unit` points.

### Frontline

- `roles_from_front(front)`: `(B, 5)`.
- `terminal_distances(front, threshold)`: `(offensive, defensive)` captures to the end.
- `harmonic(counts)`: `H_n`, with `H_0 = 0`.
- `zone_membership(ship_position, zone_position, zone_radius)`: `(B, N, Z)`, distance
  ≤ radius.
- `capture_step(team, alive, inside, roles, progress, attack_ledger, defense_ledger,
  front, tick_seconds, capture_seconds)`:
  - `inside` is already masked to living ships; the ledgers are `(B, Z, N)`;
  - returns a dict with `progress, direction, attack_ledger, defense_ledger, gain, loss,
    team0_captured, team1_captured, front, roles`.
- `zone_progress_events(team, inside, roles, previous_progress, progress,
  team0_captured, team1_captured)`: `(B, N, 2)` held, absent (the objective-progress
  attribution).
- `timeout_result(front)`: +1, −1 or 0 from the front's sign.
