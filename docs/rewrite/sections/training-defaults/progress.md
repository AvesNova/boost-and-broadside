# Training defaults: progress

Branch `feat/training-defaults` from `integrate/next-runs` (e0840b71). The six decided
defaults of [arms-todo.md](../../arms-todo.md) ("Decided, to implement"). CPU only.

## 2026-11-15 07:03 to 07:15: deathmatch outcome `kill_lead`

- **Tried:** two settings chose deathmatch scoring: the learner's outcome
  (`RewardConfig.deathmatch_outcome`, set on `ModeRules.deathmatch_outcome`) and the
  ratings' fit (`deathmatch_score` of `bnb ratings`, `bnb head-to-head`, `bnb pool create`,
  `bnb.league.tournament`). Both defaulted to kill share.
- **Worked:** both default to the kill lead now: the learner's outcome `"kill_lead"`, the
  ratings' `"win_loss"` (the same rule: the sign of the final kill difference). Kill share
  stays as the setting for an ablation. `bnb evaluate` already scored by win, draw or loss.
- **Kept:** the ratings' value keeps its name `win_loss` rather than `kill_lead`: it is the
  name recorded with every batch (`scoring`) and the rule the other modes are rated by.
- **Fixed on the way:** a record batch that names no scoring was read under "the default";
  with the default changed it would have been misread. Such batches were all written while
  kill share was the default, so `tournament.UNNAMED_SCORE = "kill_share"` names that now
  (`unnamed_scoring`), apart from the default.
- **Next:** mirrored coin-flip lineups.
