---
covers: [src/bnb/learn/rewards.py, src/bnb/contracts/trajectory.py, src/bnb/runtime/rollout.py]
---

# Rewards

The game emits events, not rewards. `bnb.learn.rewards` turns each tick's
`bnb.contracts.Events` into per-ship rewards at several **levels**, on the learning side. The
math is v1's, checked by the fixtures in `tests/reference/rewards`; the spec is
`docs/rewrite/sections/rewards-spec.md`.

## Levels

The levels run from the least to the most directly tied to winning. Their order,
`bnb.contracts.REWARD_LEVELS`, is also the order of `Trajectory.reward`'s last axis and of
the critic's levels.

<!-- generated-table: reward-levels -->

A level's reward is its weight times the sum of its events. The weights are
`bnb.learn.rewards.RewardConfig`:

<!-- generated-table: reward-config -->

- **damage:** damage taken, dealt, friendly fire, recharge, charge-back and unforced
  (boundary) damage, in units of `damage_unit` points for every ship type.
- **kill_death:** death, kill, ally kill and unforced (boundary) death.
- **objective_progress** and **objective:** the mode's two objective events, as they come
  (see [Modes](modes.md)).
- **outcome:** the match result, +1 or -1 by team or 0 for a draw, to every ship of the
  team, living or dead, on the final tick.

**Every level is zero-sum** over each environment on every tick. A team's gain is the
other team's loss: damage to an ally blames the shooter and pays the opposing team twice,
and a boundary death or boundary damage pays the opposing team.

## The ledger

Kill credit and recharge depend on history, so the reward step carries state: the
**ledger** (`bnb.learn.rewards.Ledger`), each attacker's damage to each target that has not
healed.

- `ships (B, N, N)` is indexed `[attacker, target]`; `boundary (B, N)` is the boundary's
  share, a pseudo-attacker.
- **The invariant:** for every living ship, its column plus its boundary entry sums to its
  shield deficit, the maximum shield minus the shield.
- The runtime carries the ledger beside the game state; it is reward state, not game
  state.

## One tick

`bnb.learn.rewards.reward_step` is a pure function, `(ledger, events) -> (rewards, ledger')`,
with rewards `(B, N, len(REWARD_LEVELS))` in level order. The rollout calls it every tick, in the resolve
stage.

1. **Applied damage.** What each shield actually lost is split over its raw hits and its
   boundary damage, so a shield broken with overkill ignored splits only what it lost.
2. **Accumulate** the applied damage into the ledger.
3. **Kill and death.** A dead ship's kill is shared half by the final tick's raw hits and
   half by the outstanding damage in the ledger. Shares held by enemies are kills; shares
   held by allies blame them.
4. **Recharge.** A shield's recharge is attributed pro rata over its column of the ledger:
   the target is refunded and each attacker is charged back its share.
5. **Damage**, from the applied damage and the attribution.
6. **Objective** levels from the mode's events, and the **outcome** on done environments.
7. **Clear.** A dead ship's column is zeroed, because its death made that damage stick. A
   ship keeps its claims on others (its row) across its own death and respawn. A finished
   match clears the whole ledger.

The step is memory-bound on its `(B, N, N)` matrices, so the component functions accept
column totals the step already has, instead of summing a matrix again.

The tests check the levels are zero-sum and the ledger holds every deficit through real
play in every mode
(`tests/sections/test_rewards.py::test_levels_are_zero_sum_and_the_ledger_holds_every_deficit`).
