# Rewards spec (section 3b)

Status: spec, 2026-11-01. Decisions: `rewards.md`. Fixtures: `tests/reference/rewards/`
(the `ledger_*` and `levels_*` cases; README there).

Rewards turn the game's `Events` into per-ship rewards for five levels. This is learning
side code: `bnb/learn/rewards.py`, called by the runtime each tick. v1's math is stated
here and checked by the fixtures. **(v2)** marks a decided change.

## 1. Levels

In this order, which is also the critic-head order:

| level | weight | events (before weight) |
|---|---|---|
| `damage` | 0.274 per 100 points | taken, dealt, friendly fire, recharge, charge-back, unforced |
| `kill_death` | 0.283 | death, kill, ally kill, unforced |
| `objective_progress` | 2.0 | the mode's `objective_progress` |
| `objective` | 2.0 | the mode's `objective` |
| `outcome` | 1.0 | the result, ±1 or 0 by team, on the final tick |

- A level's reward is its weight times the sum of its events.
- The weights live in a `RewardConfig` with these defaults.
- There are no payout ratios **(v2)**: every level is zero-sum by construction.
- The damage unit is a fixed 100 points for every ship type **(v2)**.

## 2. State: the ledger (v2: with a boundary row)

- **Shape:** `ledger (B, N + 1, N)`, indexed `[attacker, target]`. Rows `0…N−1` are ships;
  row `N` is the boundary.
- **What it holds:** each entry is damage the attacker did to the target that has not
  healed.
- **The invariant:** for every living ship, its column sums to its shield deficit,
  `S_max − S`, within float tolerance.
- **Ownership:** the runtime carries the ledger with the game state. Rewards are a pure
  function: `(ledger, events) → (rewards (B, N, 5), ledger')`.

## 3. One tick

Inputs: `Events` (sim order) and each ship's team. Write `raw = events.damage` and
`boundary = events.boundary_damage`.

1. **Applied damage.**
   - With `total_t = Σ_a raw[a, t] + boundary_t` and `lost = events.shield_lost`:
     - `combat_applied[a, t] = raw[a, t] · lost_t / total_t`;
     - `boundary_applied_t = boundary_t · lost_t / total_t`;
     - both are 0 where `total_t = 0`.
   - This is how a shield that breaks with overkill ignored splits what it actually lost.
2. **Accumulate:** `ledger[a, t] += combat_applied[a, t]`, and
   `ledger[N, t] += boundary_applied_t`.
3. **Kill/death** (v1, fixture `levels_kill_death`):
   - a combat death is `died_t ∧ Σ_a raw[a, t] > 0`; a boundary death is any other death;
   - the events are v1's death, kill, ally kill and unforced, computed from
     `raw[:, t]` and the **ship rows** of the ledger. Kill shares are half by the final
     tick's raw impact and half by outstanding damage. Ally blame pays the opposing team
     twice; boundary deaths and unheld shares pay the opposing team, split evenly over
     its slots (dead ones included).
4. **Recharge attribution** (v1, fixture `ledger_recharge`):
   - with `restored = events.shield_restored`, `deficit_t = Σ_rows ledger[:, t]` and
     `fraction_t = min(restored_t / deficit_t, 1)` (0 where the deficit is 0), each row
     attributed `ledger[r, t] · fraction_t`;
   - the ship rows are v1's `recharge_attributed`, and the boundary row's share plus any
     remainder `restored − Σ attributed` is v1's `unattributed`;
   - then `ledger −= attributed`, every row.
5. **Damage** (v1, fixture `levels_damage`): v1's six events from `combat_applied` summed
   per target, `boundary_applied`, `raw`, `restored`, the ship-row attribution and the
   unattributed part, in units of 100 points. Damage dealt to an enemy pays the shooter;
   dealt to an ally, it blames the shooter and pays the opposing team twice. Recharge
   refunds the target and charges back each attacker pro rata.
6. **Objective levels:** `events.objective_progress` and `events.objective`, as they come.
7. **Outcome:** each ship gets `+result` on team 0 and `−result` on team 1, on done
   environments: the living and the dead alike.
8. **Clear (v2):**
   - the columns of ships that died this tick are zeroed, after steps 3–5, because their
     death made that damage stick;
   - rows are never cleared by a death: a ship keeps its claims across its own death and
     respawn;
   - an environment that is done has its whole ledger zeroed after this tick's rewards.

## 4. Runtime and contracts (v2)

- **The rollout computes rewards** each tick from the tick's `Events` and its ledger. It
  then routes them to views as the skeleton does.
- **`Trajectory.reward` becomes `(L, R, S, 5)`,** one column per level. The skeleton PPO
  sums the levels until section 6; that is a mechanical change.
- **The level names** are a constant in `bnb.learn.rewards`, and Yemong's level
  configuration defaults to them, with `outcome` as its outcome level.
- **The skeleton reward stub** `bnb/game/modes/deathmatch.py` is deleted.

## 5. Acceptance

1. `tests/reference/rewards` (the `ledger_*` and `levels_*` cases) pass through
   `rewards_arm.py`.
2. **Zero-sum:** every level sums to zero over every environment on every tick, on real
   rollouts of every mode (sandbox, team deathmatch uniform and grouped, Frontline), with
   friendly fire, boundary deaths, respawns and resets. Tolerance: float32, relative to
   the tick's largest reward.
3. **Ledger conservation:** each living ship's ledger column sums to `S_max − S`, within
   1e-3 points, through the same rollouts.
4. **Scenario tests:**
   - **A duel healed:** A takes B to 10 of 100; B recharges fully. A's damage level nets
     exactly 0 and B's nets 0.
   - **A kill:** shares are paid by the final hits and the outstanding damage, with ally
     blame and a boundary death.
   - **A dead attacker's claims:** A damages B, A dies and respawns. B then dies, and A
     gets its outstanding share. B heals instead, and A is charged back.
   - **A respawned target:** after it respawns, damage from its previous life is neither
     charged back nor credited to a later kill.
5. **Speed:** report the reward step's cost per tick at the bench sizes. It is `O(N²)` per
   environment, like the damage matrix.
