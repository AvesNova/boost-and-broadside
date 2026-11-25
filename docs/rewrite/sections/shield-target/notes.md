# Shield target: why most nonzero rows are tiny (2026-11-25)

Question: the next-state shield column is 0 on 96.2% of rows and the nonzero rows are mostly
far below a hit (21 points). Why?

Run: CPU, 512 environments x 3,600 ticks, scripted yardstick both teams,
`GameConfig(training_spawn=True)`, LearnConfig team sizes, seed 0 (throwaway scripts, not
committed, in the session scratchpad `shield/`: `trace.py`, `rows2.py`,
`factors.py`, `deaths.py`; analysis in `analyse.py`, `an2.py`). 10,718,444 valid rows. The
repro matches the GPU preview: 96.15% zero, 1.9% of nonzero positive and below 3e-6, median
nonzero magnitude 0.0032.

## The target

`lean/model/policy.py` next_state_target: shield(t+1) - shield(t) - StepOut.shield_restored.
The game (`lean/game/core.py` _shields) recharges only on an unhurt quiet tick, so the
target is minus the damage the shield took: negative-only by construction, recharge
deliberately subtracted. Matches -min(damage, shield) to 3.8e-6 on every row.

## What the nonzero rows are

| Share of nonzero rows | What | Share of shield-loss mass |
|---|---|---|
| 69.6% | real hits under 1% of hit_damage that would be >= 1% without the armoured nose | 0.58% |
| 1.2% | hits under 1% even without the nose (grazing: closeness smoothstep) | 0.02% |
| 25.1% | hits >= 1% of hit_damage | 99.36% |
| 0.3% | boundary damage only (smootherstep ramp) | 0.04% |
| 3.8% | no damage at all: float32 residue, (s + r) - s - r on recharging ticks | 0 |

The tiny hits are the head-on rule: P51D_VALUES head_on_fraction = 0.0 ("no damage dead
ahead", game v12 choice, docs/rewrite/p51d-inputs-sources.md), half damage at 45 deg,
sharpness 3. Per real hit (525,338 hits): median aspect factor 1.6e-4, median damage 0.0024
points; 82.6% of hits land with the bullet within ~25 deg of the target's nose
(hit_cos > 0.9), and the shooter sits in the target's front half for 94.6% of those (the
sign is right). Scripted hits do 8.2% of the damage they would do with no armoured nose.

Not a bug in the game: a real mechanic, with scripted agents mostly scoring in head-on passes.

## Bug fixed: float residue

The 3.8% residue rows (|value| <= 3e-6, both signs) broke the docstring's "0 on a tick without
damage". Fix: the shield column is 0 outright on a recharging tick
(`torch.where(shield_restored > 0, 0, next - state)`), exact since the game gives no recharge
on a hurt tick. After the fix: no positive rows, 96.29% zero. Test:
`test_next_state_shield_is_exactly_zero_on_a_recharging_tick`. Model side only, not the game.

## Damage reward and value targets

lean/learn/rewards.py tick_events takes hit damage straight from StepOut and the shield lost as
min(shield, damage): no residue, but the same tiny frontal hits pay the damage level (and the
damage value target) their tiny amounts. Consistent with the game; nothing to fix.

## Strange

- A ship at shield 0 dies to any damage > 0: 563 of 9,798 deaths (5.7%) came on a tick
  whose total damage was under 0.21 points, 137 under 0.001 points: frontal near-zero hits.
- 9,408 valid rows were hurt yet the target is 0: damage below the shield's float32 spacing
  (shield - d == shield). They still restart the recharge wait.
