---
covers: [src/bnb/learn/rewards.py, src/bnb/contracts/trajectory.py, src/bnb/runtime/rollout.py]
reviewed: 2026-11-15
---

# Rewards

The game emits events, not rewards. `bnb.learn.rewards` turns each tick's
`bnb.contracts.Events` into per-ship rewards at several **levels**, each in two **parts**,
on the learning side. The
math is v1's, checked by the fixtures in `tests/reference/rewards`; the spec is
`docs/rewrite/sections/rewards-spec.md`.

## Levels

The levels run from the least to the most directly tied to winning, in the order
`bnb.contracts.REWARD_LEVELS`.

<!-- generated: reward-levels -->

A level's reward is its weight times the sum of its events. The weights are
`bnb.learn.rewards.RewardConfig`, v1's times 10, so the returns are of the order of 1 the
critic's bins and the return scale assume ([the learner](../learner.md)):

<!-- generated: reward-config -->

- **damage:** damage taken, dealt, friendly fire, recharge, charge-back and unforced
  (boundary) damage, in units of `damage_unit` points for every ship type.
- **kill_death:** death, kill, ally kill and unforced (boundary) death.
- **objective_progress** and **objective:** the mode's two objective events, as they come
  (see [Modes](modes.md)).
- **outcome:** the match's score, to every ship of the team, living or dead, on the final
  tick: the team's win weight less its loss weight (`Events.win_loss`). Elimination and
  Frontline score the result, +1 or -1 by team or 0 for a draw. Team deathmatch scores
  the result too by default (`RewardConfig.deathmatch_outcome = "kill_lead"`), by the
  sign of the final kill difference, as its default rating does. With `--set
  reward.deathmatch_outcome=kill_share` it scores its kill share instead: `2 share - 1`,
  with share the team's kills over both teams' kills (0.5 when neither scored), so +1 for
  every kill and -1 for none.

**Every level is zero-sum** over each environment on every tick. A team's gain is the
other team's loss: damage to an ally blames the shooter and pays the opposing team twice,
and a boundary death or boundary damage pays the opposing team.

## Offence and defence

Each level's reward is the sum of two parts (`bnb.contracts.REWARD_PARTS`), and each event
belongs to one of them, by whose ships it is about:

- **offence:** the enemy's ships. Damage dealt and its charge-back when the target heals,
  kills, the enemy team's own mistakes (its friendly fire and boundary losses, the unforced
  events), and in the objective and outcome levels the gains: the mode's credit for the
  team's progress and captures, and the team's win weight.
- **defence:** the ship's own team. Damage taken and healed, deaths, the blame for its own
  team's friendly fire and its refund when the ally heals, and the losses: the charge for
  the enemy's progress and captures, and the team's loss weight, negated.

Defence keeps its sign, so a cost is negative and a level is offence plus defence. Every
damage and kill/death event pairs with one of the other part on the other team, so a team's
offence on these levels is the other team's defence negated; the same holds for the outcome
(`tests/sections/test_split_critic.py::test_a_teams_offence_is_the_other_teams_defence`). The
objective levels split each ship's signed amount by its sign: a ship credited in one zone
and charged in another on the same tick keeps only the net.

`Trajectory.reward`'s last axis holds the parts, level by level, in the order of the critic's
value levels, `bnb.contracts.VALUE_LEVELS` (`damage_offence`, `damage_defence`, ...). The
critic learns a value for each part, and the advantage weighs them
([the learner](../learner.md#returns)).

<!-- generated: reward-events -->

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

`bnb.learn.rewards.reward_step_events` is a pure function, `(ledger, events) -> (rewards,
ledger', event rewards)`, with rewards `(B, N, len(VALUE_LEVELS))`, each level's parts in
order. The rollout calls it every tick, in the resolve stage. `bnb.learn.rewards.reward_step`
gives each level's reward, its parts summed (`bnb.learn.rewards.level_rewards`).

1. **Applied damage.** What each shield actually lost is split over its raw hits and its
   boundary damage, so a shield broken with overkill ignored splits only what it lost.
2. **Accumulate** the applied damage into the ledger.
3. **Kill and death.** A dead ship's kill is shared half by the final tick's raw hits and
   half by the outstanding damage in the ledger. Shares held by enemies are kills; shares
   held by allies blame them.
4. **Recharge.** A shield's recharge (only on a ship's quiet ticks, see
   [Physics](physics.md)) is attributed pro rata over its column of the ledger: the target
   is refunded, each enemy attacker is charged back its share, and each ally attacker's
   blame is refunded.
5. **Damage**, from the applied damage and the attribution.
6. **Objective** levels from the mode's events, and the **outcome** on done environments.
7. **Clear.** A dead ship's column is zeroed, because its death made that damage stick. A
   ship keeps its claims on others (its row) across its own death and respawn. A finished
   match clears the whole ledger.

The tests check the levels are zero-sum and the ledger holds every deficit through real
play in every mode
(`tests/sections/test_rewards.py::test_levels_are_zero_sum_and_the_ledger_holds_every_deficit`),
and that each level's parts sum to its reward
(`tests/sections/test_split_critic.py::test_each_levels_parts_sum_to_its_zero_sum_reward`).
