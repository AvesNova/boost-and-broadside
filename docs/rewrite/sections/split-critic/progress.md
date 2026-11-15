# Split offence/defence critic: progress

**Resume note (2026-11-15 20:53, before shutdown):** done, nothing half-done. Split critic,
offence premium schedule, HL-Gauss value targets and the padded value support (2 bins per
side in new builds) are all committed and pushed on `feat/split-critic`, with
`origin/feat/imugi-boundary` merged in. Quick tier passes on every touched test file (859
passed in the last rerun). Next: the author's review (return scale vs premium, sign-split
objective levels, retired v1 two-hot fixture checks), then merge and the shared BC.

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
- Padded value support (author, 2026-11-15): `value_padding` bins past each end
  (`codes.PADDING_BINS` = 2 in a new build via `TrainConfig.yemong/relational/nearest`; the
  config default 0 is what older build records read). 77 + 4 = 81 bins; returns clamp to
  the outermost padding centre (`critic/support_clip`); the read-out covers every bin. The
  target's mean position is the return's within 1% of a bin up to +-limit; the read-out (in
  reward units over symlog bins) is about 0.2% larger in magnitude than the return (half the
  target's symlog variance), the price of HL-Gauss on symlog bins.
- v1's critic-loss fixture case (`learner_critic`, two-hot) and the two-hot check of
  `critic_two_hot` are no longer run; the section's independent-computation loss test
  covers the HL-Gauss loss.
- Diagnostics per part: `critic/explained_variance/*`, `critic/return_mean/*`, the advantage
  series, `reward/rate/*`, `reward/level_share/*`, `episode/reward/*`; value losses and value
  KL per value head (8 parts + `outcome`); `schedule/offence_premium`;
  `reward/zero_sum_residual/*` stays per level.
- The single value arm (`summed_reward`) is gone from the learner (no unsplit path). Old
  policy files (unsplit or summed) still load and play; the learner refuses to train them.
- Overhead: the own-ship head's value outputs go from 4x77 to 8x81 (split and padding);
  2,854,930 to 2,942,310 parameters (+87,380, +3.1%; the split alone +79,156). Update time
  not measured.

**Didn't / open**
- Objective-level sign split: a ship credited in one zone and charged in another on the same
  tick keeps only the net. Irrelevant to the advantage at 1:1 (decided); only the critics'
  split of those levels sees it.
- The return scale does not follow the premium, so 2:1 also enlarges the combat levels' step.

**Next**
- The shared BC, then the premium forks (1:1, 2:1, 2:1 annealed).

## 2026-11-16: the author's two fixes (branch `fix/critic-premium-scale`)

**Return scale follows the premium (done):** `returns.team_aggregate` weighs the returns'
sum (the return scale's input) with the advantage's part weights. A premium now changes
only the balance of offence and defence, not the step: where the premium levels' offence is
the only reward, the scaled advantage is the same at 1, 2 and 3.5
(`test_the_return_scale_follows_the_premium`). At 1:1 the weights are all 1, so nothing
changes. Critic targets stay each part's own.
