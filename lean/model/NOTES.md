# lean/model: notes

## What is (2026-11-23, step M)

- `policy.py`: `ModelConfig`, `NextStateColumn`, `Policy` (`initial_carry`, `step`,
  `sequence`, `next_state_target`), `Encoder` (`KindProjection` per kind), `Spatial`,
  `HeadGroup`, `Block`, `rotate`, `field_heading`, `draw_position_offset`, and the
  next-state functions `next_state_target`, `next_state_centres`,
  `next_state_distribution`, `next_state_cross_entropy`.
- `temporal.py`: `Temporal` (Griffin's recurrent block and a gated MLP), `doubling_scan`.
- `position.py`: the position codes (`FourierCode`, `GridCellCode`, `ColourCode`,
  `make_position_code`) and the rotary angles (`Rotary`).
- `value.py`: `value_bin_centres`, `hl_gauss_targets`, `decode_value`.

Shape: width 128, 2 blocks, 2 heads of 64, each block one temporal sublayer then two
spatial sublayers; map memory width 64; MLPs 4 x width; heads hidden 256. Trunk tokens
`[game, ship slot 0 .. N-1]`; fields are keys and values only. The carry is `(blocks, E,
1 + N, 4, width)` float32: per token row 0 the RG-LRU state, rows 1..3 the convolution's
last three inputs, oldest first. `step` and `sequence` share one code path (`step` is
T = 1). Inputs are lean.contract's feature counts (no overrides).

Parameters **2,155,565** (fourier; was 2,058,420 before step M: +96,889 for the
next-state head's 6 x 64 logits, +256 for the pending embedding's "none" row).
Ops per eager `step` (E = 4, N = 8, F = 10, no_grad, regions.py's Counter): **531 compute
+ 304 view** (was 515 + 298); `sequence` at T = 8: 587 + 343 (was 571 + 336). New code
on the path (not counted per region): the fields' heading (atan2), the temporal "starts
from zero" test in each block, the embedding in place of the one-hot.

## The pieces

- **Encoder** (v2): per kind (ship, game, field) `gelu(norm(Linear(inputs -> 2 width)))`,
  then one shared `Linear(2 width -> width)` and norm; fields then `Linear(width -> 64, no
  bias)` and a norm. Ship = `Obs.ship` + fog pair (visible, since_seen) + position code of
  `pos + offset`, plus the **pending action** as an `nn.Embedding(43, 2 width,
  padding_idx=42)` row added before the norm. Why an embedding: it is the one-hot's
  function (the row is that one-hot input's column of the Linear; "none" = 42 is a zero row
  never trained, as an all-zero one-hot) without building a 42-wide one-hot and 42 more
  input columns: one gather. Rows
  start uniform within 1/sqrt(fan-in incl. the 42), as the one-hot columns did.
  Game = `Obs.globals` + position code of `offset` (the arena centre, offset like any
  token); field = `Obs.field` + position code of `field_pos + offset`.
- **Spatial**: pre-norm attention of the M trunk tokens over the M tokens and the F fields
  in one softmax; element-wise gate `sigmoid(W x)` after the values are rotated back,
  before the output projection (Qiu et al. G1; W is the fourth block of
  `query_key_value_gate`); pre-norm gated MLP. No biases, no null key.
- **Temporal** (Griffin 2.3): `recurrent_input`, a 4-tap depthwise causal convolution, the
  RG-LRU (gates read the convolved signal), `state * gelu(gate_branch(normed))`, output;
  then a pre-norm gated MLP. Recurrence by `doubling_scan`; resets (`fresh` 0) and masks
  (`keep` 0) are folded into the decay and the update; the convolution's taps are cut by a
  reset or a masked tick inside their window.
- **The reset factor**: the input is scaled by sqrt(1 - a_t^2) only on a running state; a
  state starting from zero takes its first input at 1 (the released recurrentgemma code's
  rule at a segment's first position). "Starts from zero": a reset before the tick, the
  row masked on the tick before, or (at tick 0) a carry state that is exactly zero; the
  last makes `step` and `sequence` agree without a flag in the carry (a live state is
  never exactly zero across 128 channels).
- **Initialisation** (one rule for all six residual blocks, recurrentgemma's;
  docs/rewrite/literature-model-choices.md section 3): every input projection N(0,
  1/fan_in) (temporal branches and gates; spatial query/key/value/gate, map key/value, MLP
  gate and up rows); every output projection and MLP down projection N(0, s/fan_in), s =
  2/6; RG-LRU decay with a^8 uniform in [0.9, 0.999]; convolution N(0, 0.01/4), biases 0.
  Encoder: PyTorch defaults. Heads: first blocks orthogonal gain sqrt 2; last layers
  orthogonal 0.01 (policy, value levels, next state), biases 0; the hit head's biases
  log(0.9457, 0.0444, 0.0099), the measured shares (below).
- **Rotary** (`position.Rotary`): per band two perpendicular axes; band k turned k x 55.62
  degrees (90 x the golden section 0.618..., an irrational fraction of 90); periods 64 m x
  2^k up to the first above 16.8 km (`rotary_reach`, twice the 8.4 km diameter, so the
  coarsest band alone tells apart every separation on the map); heading harmonics 1 and
  2. 22 pairs = 44 of each head's 64 dimensions (68.75 %, the literature's 22 to 24).
  Values rotated by the key's angles and the sum rotated back by the query's. Rotary uses
  true positions (arena centre at the origin, no offset). Ships: position and heading.
  Game token: the arena centre, heading 0 (the identity). Fields: their centre, heading
  towards the arena centre (`field_heading`; rotating the map about the centre turns
  every field's heading with it).

  | band | period (m) | axes turned (deg, mod 90) |
  |---|---|---|
  | 0 | 64 | 0 |
  | 1 | 128 | 55.62 |
  | 2 | 256 | 21.25 |
  | 3 | 512 | 76.87 |
  | 4 | 1,024 | 42.49 |
  | 5 | 2,048 | 8.12 |
  | 6 | 4,096 | 63.74 |
  | 7 | 8,192 | 29.36 |
  | 8 | 16,384 | 84.98 |
  | 9 | 32,768 | 50.61 |

  Closest two bands' axes: 5.0 degrees apart (bands 0 and 8). Finest band against one
  tick's motion: two ships closing head-on at 335.4 m/s (top level speed; 336.6 m/s was
  the fastest measured) close 22.4 m a tick, so a band aliases below 44.7 m; 64 m clears
  it. Input position code: Fourier and grid cells coarsest 20 km, v2_codes 236 km, all
  above 8.4 km.
- **Position codes** (`ModelConfig.position_code`): "fourier" (default; hexagonal Fourier
  features, 3 directions x 6 periods 64 m to 20 km), "grid_cells" (an experiment arm;
  Blair et al. 2007 with Blair's gain, 3 x 3 phases per spacing), "v2_codes" (v2's
  3-colour code, 9 levels, period 236,196 m). The offset range is the chosen code's
  coarsest period.
- **Heads** (v2's `HeadGroup`): `last(gelu(norm(first x)))`, hidden 256; group 1 policy
  (42) and value (5 levels x 64 bins); group 2, run only by `sequence`: next state (6 x 64)
  and hit (3: missed, hit an enemy, hit a team-mate). Values per ship for every level.

## Next-state head

A categorical of 64 bins per column; the learner's loss is the HL-Gauss cross-entropy
(sigma 0.75 bin widths), nats per column (`next_state_cross_entropy`; ln 64 = 4.159 at a
uniform prediction; the targets' own entropy, the floor, is 1.20 nats per column).
`next_state_target(config, state, next_state, step_out, tick_length)` makes the targets
from the two states and the step's recharge; collect asks the policy
(`Policy.next_state_target`). Bins uniform from `low` to `high` (outermost centres) in
target units, or in symlog space.

Ranges from 5,554,329 rows (every living ship, alive after the tick, episode not ended):
scripted yardstick against itself, 4v4 elimination, training spawn on, fields on, fog on,
256 environments x 4,000 ticks (698 episodes), 2026-11-23. Linear ranges are symmetric,
just outside the 0.1st and 99.9th percentiles; the rarer tails fall in the outermost bins.

| column | target | bins | range | min | 0.1 % | 1 % | 50 % | 99 % | 99.9 % | max | outside |
|---|---|---|---|---|---|---|---|---|---|---|---|
| along (m) | move minus world velocity x tick, along the nose | linear | +-0.35 | -1.04 | -0.315 | -0.087 | 0.006 | 0.104 | 0.302 | 1.17 | 0.15 % |
| across (m) | the same, to the nose's left | linear | +-0.6 | -1.35 | -0.556 | -0.264 | 0.000 | 0.263 | 0.560 | 1.33 | 0.13 % |
| heading (rad) | wrapped heading change minus nose rate x tick | linear | +-0.05 | -0.112 | -0.043 | -0.016 | 0 | 0.016 | 0.043 | 0.113 | 0.12 % |
| speed (m/s) | plain change of proper speed | linear | +-3 | -4.79 | -2.57 | -1.49 | -0.03 | 1.61 | 2.80 | 4.16 | 0.10 % |
| energy (store) | plain change | linear | +-700 | -877 | -644 | -365 | 0 (71 % zero) | 332 | 517 | 866 | 0.02 % |
| shield (points) | change minus the recharge given (minus the damage absorbed) | symlog | -90 to 0.3 | -44.2 | -14.3 | -0.08 | 0 (97 % zero) | 0 | 0 | 3e-6 | 0 |

The shield's low end is four hits of about 21 in one tick (a crossfire of four ships, the
author's range; the largest measured was 44), its high end three bins above 0. The
recharge subtracted is `StepOut.shield_restored` (the schedule's recharge, given only on
an unhurt quiet tick), so the column is exactly minus what the shield absorbed.

Hit head: of 5,524,833 bullets that ended in the same collection (ghosts included),
94.57 % missed, 4.44 % hit an enemy, 0.99 % hit a team-mate (`hit_class_frequencies`).

## Open

- The ranges and shares are the core game's of 2026-11-23 (P-51D, store_loops 2); they
  need re-measuring when the flight numbers or the lineup change (measure script: one
  scripted collection with `next_state_target` per tick, as above).
- Grid cells (gain, phases, spacings) and the v2_codes period are my picks.
- No parity test against v2's modules; no compiled or bf16 `sequence` test here (the
  learner's compiled tests cover the loss graph).
- The ops tripwire (650 per step) is exceeded (531 + 304); about 150 compute ops are eager
  RMSNorm decompositions, fused when compiled.
- `lean/eval/matches.py` calls `step` without a position offset (zero offset).
- Not here: zones, an absent-zone token, nearest-k attention, an enemy-action head, a
  game-token outcome head (an ablation arm only).

## Tests (`tests/test_model.py`, 28)

Sequence equals repeated steps (resets at tick 0 and inside, masks, fields, a random
carry; rotate_values on and off); a reset forgets state and buffer; masked rows keep a
zero carry; ship padding and permutation; fields as an unordered memory; a spatial
sublayer reads only relative positions (ships and fields translated, headings kept); fields
head towards the arena centre; the band ladder (ratio, reach, per-tick limit, turns, share,
position code span); the coarsest band alone recovers every separation up to 8.4 km to
0.1 m; one set of weights at any N and F; the doubling scan; the temporal and spatial
initialisation; the gate near 0.5; heads near uniform and the hit head at the shares; a
state starting from zero takes its first input unscaled (reset, zero carry, after a
masked tick; a running state does not); next-state targets and bins on a hand-made pair;
the next-state loss in nats (ln 64 uniform, entropy floor); each position code; position
code symmetries; HL-Gauss targets; backward reaches every parameter and no masked row;
Adam fits a tiny batch; bf16 close to float32; compiled `step` matches eager; the model
reads fog.
