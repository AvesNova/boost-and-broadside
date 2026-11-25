# lean/model

The policy network: v2's model maths with the decided changes (author review, 2026-11-23).
Reasons: `docs/rewrite/model-audit.md`, `docs/rewrite/literature-model-choices.md`; decisions:
`docs/rewrite/core-decisions.md`; history: `docs/rewrite/core-worklog.md`.

## Entry points

- `Policy(ModelConfig())` (`policy.py`): `initial_carry`, `step(obs, carry, position_offset, reset)`
  (one tick, rollout) and `sequence(obs, carry, reset, ...)` (T ticks, the learner). One code
  path; `step` is T = 1. Inputs are `lean.contract`'s feature counts. Outputs are per ship:
  42 policy logits, 5 value levels x 70 bins; `sequence` also returns the next-state logits
  (6 x 64) and the hit logits (3).
- `agent.py`: how a network acts, shared by `lean.learn`'s rollout and `lean.eval`'s runner:
  `Network` (a weightless `Stepper` template per architecture plus weights; new weights are new
  data, so no recompile), `agent_step` (the one agent tick: precision, forward, the joint
  action's sample; returns an `AgentStep`) for training's live policy (`Policy.step`) and every
  `Network` (`network_step`), `sample_joint`, `seat_view`, `precision`, `reset_carry`.
- `temporal.py` (`Temporal`, `doubling_scan`), `position.py` (position codes, `Rotary`),
  `value.py` (`symlog`, `symexp`, `buffered_centres`, `hl_gauss_targets`, `decode_value`,
  `value_bin_centres`).
- `next_state_target(config, state, next_state, step_out, tick_length)` makes the next-state
  head's targets of one game step (collect records them); `next_state_cross_entropy` is the loss.

## Shape

Width 128, 2 blocks (each one temporal sublayer, then two spatial), 2 heads of 64, MLPs 4 x
width, heads hidden 256, map memory width 64. Trunk tokens `[game, ship 0 .. N-1]`; fields are
keys and values only. Carry `(blocks, E, 1 + N, 4, width)` float32: per token the RG-LRU state
and the convolution's last three inputs. **2,172,527 parameters** (checked 2026-11-25, 70 bins).
Operations per eager `step` (E = 4, N = 8, F = 10): 531 compute + 304 view (2026-11-23); the
650 tripwire is exceeded, about 150 of them eager RMSNorm decompositions that fuse when compiled.
The speed of the whole loop is in `lean/learn/NOTES.md`.

## The pieces

- **Encoder:** per kind (ship, game, field) `gelu(norm(Linear))`, then a shared Linear and norm.
  A ship = `Obs.ship` + fog pair (`visible`, `since_seen`) + position code of `pos + offset` +
  the pending action as an embedding (row 42 = none, a zero row).
- **Spatial:** pre-norm attention of the trunk tokens over trunk tokens and fields in one
  softmax, an element-wise sigmoid gate after the values are rotated back, a gated MLP. No
  biases, no null key.
- **Temporal:** Griffin's recurrent block (4-tap causal convolution, RG-LRU) by
  `doubling_scan`, then a gated MLP. Resets and masks are folded into decay and update. A state
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

## Re-measured 2026-11-25

After the night's changes (P-51D at 2,200 hp, stick thirds, flaps gone, the trimmed BFM-CMA2
scripted agent, the 1,800-tick clock, the shared tick), one collection: 2,048 environments x
3,600 ticks on the GPU, both teams the scripted yardstick, `GameConfig(training_spawn=True)`,
4v4 (`LearnConfig`), 6,592 episodes ended; returns are discounted per level (Monte Carlo to the
episode's end) for every living ship's tick of the episodes that ended inside the run.

Next-state columns, 0.1st and 99.9th percentiles (42.7M rows; 2026-11-23: 5.55M rows):

| Column | 0.1% | 99.9% | Extremes | Range 11-23 | Range 11-25 |
|---|---|---|---|---|---|
| along (m) | -0.363 | 0.338 | -1.57, 1.89 | +-0.35 | +-0.37 |
| across (m) | -0.579 | 0.581 | -1.81, 1.82 | +-0.6 | +-0.6 |
| heading (rad) | -0.0422 | 0.0422 | -0.100, 0.099 | +-0.05 | +-0.043 |
| speed (m/s) | -2.65 | 2.78 | -5.39, 4.02 | +-3 | +-2.8 |
| energy (store) | -640 | 594 | -976, 976 | +-700 | +-640 |
| shield (points) | -14.6 | 0 | -61.5, 0 | -90 to 0.3 | -90 to 0 |

The old ranges on the new data: outside +-0.35 along 0.20%, +-0.6 across 0.16%, +-0.05 heading
0.10%, +-3 speed 0.11%, +-700 energy 0.09%. The shield: 3.62% of rows lose shield, 0.07% gain
(by at most 3e-6, rounding); none below -90. Its low end stays the four-hit crossfire; its high
end was 0.3 (about three bins above 0, the buffer by hand) and is now 0 with the buffer rule.

Value returns (35.5M rows), in return units:

| Level | 0.1% | 99.9% | Extremes | Support 11-23 (outermost centres) | Measured range 11-25 | Outermost centres 11-25 |
|---|---|---|---|---|---|---|
| damage | -2.08 | 2.12 | -3.91, 5.37 | +-7 | +-5.4 | +-5.91 |
| kill_death | -2.83 | 3.68 | -6.47, 7.50 | +-8 | +-7.5 | +-8.21 |
| outcome | -9.985 | 9.997 | -10, 10 | +-12 | +-10 (its bound) | +-10.95 |

The 2026-11-23 supports were the measured extremes (damage -4.9 to 5.9, kill_death -5.7 to 7.9,
4v4 to 8v8, two games and the learner's first rollouts) with a margin by hand; the new ones are
the measured extremes (symmetric) with the buffer columns as the margin. Objective levels stay
+-1 placeholders.

Hit shares (42.5M bullets; 2026-11-23: 5.52M): missed 94.57% -> 93.74%, an enemy 4.44% ->
5.47%, a team-mate 0.99% -> 0.80%.

Columns: 64 -> 70 per value level and per next-state column (64 across the measured range, 3
buffer columns at each end); 70 -> 63 on 2026-11-25 (57 + 3 + 3, symlog, ranges the extremes;
above). Parameters 2,172,527 -> 2,152,738.

## Tests

`tests/test_model.py` (31): `sequence` equals repeated steps (resets, masks, fields, rotated
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
  (`docs/rewrite/sections/scoreboard/target_ranges.py`, about a minute on the laptop GPU).
- Grid cells (gain, phases, spacings) and the `v2_codes` period are builder picks.
- No parity test against v2's modules; no compiled or bf16 `sequence` test here (the learner's
  compiled tests cover the loss graph).
- Not here: zones, nearest-k attention, an enemy-action head, a game-token outcome head (an
  ablation arm only).
