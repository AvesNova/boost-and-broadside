# Split offence/defence critic: progress

Branch `feat/split-critic`, from `feat/imugi-boundary` (2026-11-15). Decision: arms-todo.md,
"Split critic for every arm" and "Offensive premium levels" (2026-11-14).

## 2026-11-15: built

**Tried / worked**
- Each reward level's reward is now two parts, offence and defence
  (`bnb.contracts.REWARD_PARTS`, `VALUE_LEVELS`: `damage_offence`, `damage_defence`, ...).
  Every reward event has one part (`bnb.learn.rewards.REWARD_EVENTS`), by whose ships it is
  about: offence the enemy's (damage dealt and its charge-back, kills, the enemy's own
  friendly fire and boundary losses), defence the own team's (damage taken and healed,
  deaths, the own team's friendly-fire blame and its refund). v1's charge-back event is split
  into the enemies' charge-back (offence) and the allies' refund (defence). The objective
  levels split by sign (gain / loss); the outcome into the team's win weight and its loss
  weight negated. Defence keeps its sign, so a level is offence + defence, unchanged.
- `Trajectory.reward` is (L, R, S, 10). The critic has 10 value levels: 8 per-ship
  categorical outputs and the outcome head's two parts, `P(win)` and `-P(loss)` (no new
  outcome output; one outcome loss).
- Advantage: `sum_k w_k A_k` over the 10 parts (`returns.team_aggregate`), weights 1 except
  the damage and kill/death offence, which take `LearnerConfig.offence_premium`, a keypoint
  schedule over environment steps (`bnb.learn.schedules.Keypoints`, default 1 throughout).
  Critic targets and the return scale's sum are unweighted, so one BC serves every premium.
- Per-level discount/trace decay/team spirit stay one per reward level, applied to both parts.
- HL-Gauss critic targets (author, 2026-11-15): `critic.value_targets`, a Gaussian of 0.75 bin
  widths on the symlog bin coordinate, renormalised, clamped to the outer bins
  (`codes.hl_gauss`); `critic.two_hot` removed. The value read-out is still the expectation.
- Diagnostics per part: `critic/explained_variance/*`, `critic/return_mean/*`, the advantage
  series, `reward/rate/*`, `reward/level_share/*`, `episode/reward/*`; value losses and value
  KL per value head (8 parts + `outcome`); `schedule/offence_premium`;
  `reward/zero_sum_residual/*` stays per level.
- The single value arm (`summed_reward`) is gone from the learner (no unsplit path). Old
  policy files (unsplit or summed) still load and play; the learner refuses to train them.
- Overhead: +79,156 parameters (2,854,930 to 2,934,086, +2.8%, the own-ship head's value
  outputs 4x77 to 8x77). Update time not measured.

**Didn't / open**
- Objective-level sign split: a ship credited in one zone and charged in another on the same
  tick keeps only the net. Irrelevant to the advantage at 1:1 (decided); only the critics'
  split of those levels sees it.
- The return scale does not follow the premium, so 2:1 also enlarges the combat levels' step.

**Next**
- The shared BC, then the premium forks (1:1, 2:1, 2:1 annealed).
