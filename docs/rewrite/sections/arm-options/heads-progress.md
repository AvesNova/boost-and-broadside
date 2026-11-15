# Auxiliary-head switches (arm A13): progress

Branch `feat/head-switches`, from `feat/imugi-boundary`. Brief: every run's code is
identical, so A13 must launch from configuration alone; defaults unchanged (every head on).

## 2026-11-15

**Tried / built.**

- `YemongConfig` switches beside `geometry_head` and `delta_split` (the error head's):
  `next_state_head`, `enemy_action_head`, `density_head` (bool, default true) and
  `shot_head` (`"trajectory"` or `"none"`, `model.SHOT_HEADS`; a later `"now"` for A22 is
  one more value). `next_state_head` off needs `forecast_feedback` and `delta_split` off.
- A head that is off is not built: no parameters, its `Unrolled` / `Decision` field None.
  The enemy-action and next-state heads share one `HeadGroup` (`ship_heads`); its outputs
  are now the built ones (`Yemong.ship_head_names`), so the default build's parameter names
  and initial draws are unchanged.
- `Yemong.auxiliary_heads` names the heads a build has, by the learner's term names. The
  learner computes labels only for those, skips the shot re-simulation (and its spread
  draws and store records) without the shot heads, skips the absent heads' terms in the
  gradient decomposition and reports no loss series for them. Head statistics are split by
  head and skip the absent ones.
- `belief.advance` with no enemy-action logits stores uniform actions.
- Play overlays and figure sets of an absent head are left out (`figures.sets.drawable`).

**A13 settings lines** (on a new run's build; also in `docs/explanation/yemong.md`):

- all but next-state:
  `--set yemong.enemy_action_head=false --set yemong.density_head=false --set yemong.shot_head=none --set yemong.geometry_head=false`
- all:
  `--set yemong.next_state_head=false --set yemong.delta_split=false --set yemong.physics_hidden_decay=null --set yemong.forecast_feedback=false --set yemong.enemy_action_head=false --set yemong.density_head=false --set yemong.shot_head=none --set yemong.geometry_head=false`

**Parameter counts** (a new run's build, 5v5; measured with `Yemong(model_build(...))`):

| Build | Parameters | Heads |
|---|---|---|
| every head (new run) | 2,942,310 | next_state, correction, enemy_action, density, shot_trajectory, shot_outcome, geometry |
| A13 all but next-state | 2,521,480 | next_state, correction |
| A13 all | 2,236,470 | none |

**Open for the author.**

- "All but next-state" keeps the error head (`delta_split`), so the belief A14 measured
  keeps its whole source (physics plus error head). Removing it too is
  `--set yemong.delta_split=false --set yemong.physics_hidden_decay=null` on top.
- The error head's switch keeps its name, `delta_split`: it also changes which rows the
  physics head trains on, and renaming it would break every stored build record.
- Without the next-state head the extrapolation baseline's rates are still computed each
  step (cheap; they feed nothing).
