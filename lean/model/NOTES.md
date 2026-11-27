# lean/model

The policy network: v2's model maths with the decided changes (author review, 2026-11-23).
Reasons: `docs/rewrite/model-audit.md`, `docs/rewrite/literature-model-choices.md`; decisions:
`docs/rewrite/core-decisions.md`; history: `docs/rewrite/core-worklog.md`.

## Entry points

- `Policy(ModelConfig())` (`policy.py`): `initial_carry`, `step(obs, carry, position_offset, reset)`
  (one tick, rollout) and `sequence(obs, carry, reset, ...)` (T ticks, the learner). One code
  path; `step` is T = 1. Inputs are `lean.contract`'s feature counts. Outputs are per ship:
  42 policy logits, 5 value levels x 63 bins; `sequence` also returns the next-state logits
  (6 x 63) and the hit logits (3).
- `agent.py`: how a network acts, shared by `lean.learn`'s rollout and `lean.eval`'s runner:
  `Network` (a weightless `Stepper` template per architecture plus weights; new weights are new
  data, so no recompile), `agent_step` (the one agent tick: precision, forward, the joint
  action's sample; returns an `AgentStep`) for training's live policy (`Policy.step`) and every
  `Network` (`network_step`), `sample_joint`, `seat_view`, `precision`, `reset_carry` (the
  evaluation runner's; training passes the step `reset`), `routed_step` and `merge_routed`
  (capacity routing: each network on its own W rows, then one merge).
- `temporal.py` (`Temporal`), `scan.py` (`linear_scan`: the recurrence in one pass, a custom
  operator with Triton kernels forward and backward on the GPU, a loop on the CPU, float32),
  `position.py` (position codes, `Rotary`),
  `value.py` (`symlog`, `symexp`, `buffered_centres`, `hl_gauss_targets`, `decode_value`,
  `value_bin_centres`).
- `next_state_target(config, state, next_state, step_out, tick_length)` makes the next-state
  head's targets of one game step (collect records them); `next_state_cross_entropy` is the loss.

## Shape

Width 128, 2 blocks (each one temporal sublayer, then two spatial), 2 heads of 64, MLPs 4 x
width, heads hidden 256, map memory width 64. Trunk tokens `[game, ship 0 .. N-1]`; fields are
keys and values only. Carry `(blocks, E, 1 + N, 4, width)` float32: per token the RG-LRU state
and the convolution's last three inputs. **2,152,738 parameters** (63 bins). The speed of the
whole loop is in `lean/learn/NOTES.md`.

## The pieces

- **Encoder:** per kind (ship, game, field) `gelu(norm(Linear))`, then a shared Linear and norm.
  A ship = `Obs.ship` + fog pair (`visible`, `since_seen`) + position code of `pos + offset` +
  the pending action as an embedding (row 42 = none, a zero row).
- **Spatial:** pre-norm attention of the trunk tokens over trunk tokens and fields in one
  softmax, an element-wise sigmoid gate after the values are rotated back, a gated MLP. No
  biases, no null key.
- **Temporal:** Griffin's recurrent block (4-tap causal convolution, RG-LRU) by
  `linear_scan` (core-decisions "The training-path fixes"), then a gated MLP. Resets and masks are folded into decay and update. A state
  starting from zero takes its first input at scale 1 (recurrentgemma's rule), which makes
  `step` and `sequence` agree without a flag in the carry.
- **Initialisation:** recurrentgemma's rule for all six residual blocks (input projections
  N(0, 1/fan_in), outputs N(0, s/fan_in) with s = 2/6, RG-LRU decay a^8 uniform in
  [0.9, 0.999]); heads' last layers orthogonal 0.01; the hit head's biases at the log of the
  measured shares.
- **Rotary** (`position.Rotary`): ten bands, periods 64 m x 2^k (the coarsest, 32.8 km, is the
  first above twice the 8.4 km map diameter, so it alone tells apart every separation), band k
  turned k x 55.62 degrees (an irrational share of 90), heading harmonics 1 and 2; 22 of each
  head's 32 pairs (68.75%). Rotary uses true positions. The finest band clears one tick's
  motion (two ships closing at 335 m/s move 22.4 m a tick; a 64 m band aliases below 44.7 m).
  Fields carry their centre and a heading toward the arena centre.
- **Position codes** (`ModelConfig.position_code`): "fourier" (default, 3 directions x 6 periods
  64 m to 20 km), "grid_cells" (an experiment arm), "v2_codes". The random position offset's range
  is the code's coarsest period (`draw_position_offset`).
- **Heads:** `last(gelu(norm(first x)))`. Group 1 (every tick): policy and value. Group 2
  (`sequence` only): next state and hit (missed, hit an enemy, hit a team-mate).

## Next-state head

63 bins per column, HL-Gauss cross-entropy (sigma 0.75 bin widths in bin space), nats per
column: ln 63 = 4.143 at a uniform prediction. Every categorical support (these columns and the
critic's levels, author 2026-11-25) has 63 bins, uniform in its bin space s(x) = sign(x) knee
ln(1 + |x| / knee) (`lean.model.value.symlog`; knee 0 = linear): 57 across the measured range,
the outermost centres at the measured ends, then `BUFFER_COLUMNS` = 3 more of the same width
beyond each end (`buffered_centres`). A symmetric range has 0 on the middle centre; there is no
zero class. HL-Gauss targets are the Gaussian over the bins in bin space; the decode is the
expectation over the centres mapped back to target units (`decode_value`). Ranges from 42.7M
rows (scripted yardstick against itself, 4v4, training game: spawn, fields, fog, the
1,800-tick clock, 2026-11-25; `docs/rewrite/sections/scoreboard/target_ranges.py`): each
symmetric range is +-the larger measured extreme (the table below); knees from the flat-core
data rule (`docs/rewrite/sections/scoreboard/symlog-preview.md`). The shield keeps -90 to 0 and
knee 1 (= `contract.symlog`) pending its bug check.

Bin widths in target units: innermost (the narrowest) and outermost (the widest, a buffer column).

| Column | Target | Measured range | Knee | Innermost, outermost bin | Outermost centres |
|---|---|---|---|---|---|
| along (m) | move minus world velocity x tick, in the nose frame | +-1.89 | 2.4e-3 | 6.1e-4, 0.92 | +-3.87 |
| across (m) | the same, across the nose | +-1.82 | 0.0244 | 3.9e-3, 0.45 | +-2.91 |
| heading (rad) | wrapped heading change minus nose rate x tick | +-0.100 | 7.7e-5 | 2.1e-5, 0.055 | +-0.216 |
| speed (m/s) | change of proper speed | +-5.39 | 0.156 | 0.0205, 1.04 | +-7.98 |
| energy (store) | change | +-976 | 21.8 | 3.08, 205 | +-1,481 |
| shield (points) | change minus the recharge given | -90 to 0 | 1 | 0.082, 9.34 | -115 to 0.27 |

The hit head's shares, from 42.5M bullets that ended (ghosts included): 93.74% missed, 5.47%
an enemy, 0.80% a team-mate (`hit_class_frequencies`).

## Tests

`tests/test_model.py`: `sequence` equals repeated steps (resets, masks, fields, rotated
values on and off); a reset forgets state; padding and permutation leave real ships unchanged;
fields are an unordered memory; attention reads only relative positions; the rotary band ladder
and the coarsest band's recovery of separations to 0.1 m; one set of weights at any N and F;
initialisation and the heads at their starting values; a state from zero takes its first input
unscaled; next-state targets, bins and loss; the position codes; HL-Gauss targets; every support's 63-bin layout (0 a centre of a
symmetric range), the symlog round trip and HL-Gauss decoding within a bin on every support; backward
reaches every parameter and no masked row; Adam fits a tiny batch; bf16 is close to float32;
compiled `step` matches eager; the model reads fog.

## Open

- The value supports, next-state ranges and hit shares are the game's of 2026-11-25; measure
  again when the flight numbers, the lineup, the clock or the scripted agent change
  (`docs/rewrite/sections/scoreboard/target_ranges.py`, about a minute on the laptop GPU). Due:
  since then the hit damage (21 to 8), the bullets (ghost-only, every ship a bullet a tick), the
  egg deflector, collisions, the post-stall levels and the look-ahead scripted agent changed.
- Grid cells (gain, phases, spacings) and the `v2_codes` period are builder picks.
- No parity test against v2's modules; no compiled or bf16 `sequence` test here (the learner's
  compiled tests cover the loss graph).
- Not here: zones, nearest-k attention, an enemy-action head, a game-token outcome head (an
  ablation arm only).
