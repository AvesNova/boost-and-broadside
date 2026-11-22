# lean/model: notes

## What is implemented (v2 port, 2026-11-23)

- `policy.py`: `ModelConfig`, `Policy` (`initial_carry`, `step`, `sequence`), `Encoder`
  (`KindProjection` per kind), `Spatial`, `HeadGroup`, `Block`, `rotate`,
  `draw_position_offset`, `next_state_target`.
- `temporal.py`: `Temporal` (Griffin's recurrent block and a gated MLP), `doubling_scan`.
- `position.py`: the position codes (`FourierCode`, `GridCellCode`, `ColourCode`,
  `make_position_code`) and the rotary angles (`Rotary`).
- `value.py`: `value_bin_centres`, `hl_gauss_targets`, `decode_value` (unchanged).
- `tests/test_model.py`: 23 tests.

Shape: width 128, 2 blocks, 2 heads, each block one temporal sublayer then two spatial
sublayers (v2's `temporal_first`, the champion's order); map memory width 64; MLPs 4 x
width; heads hidden 256. Trunk tokens `[game, ship slot 0 .. N-1]`; fields are keys and
values only. The carry is `(blocks, E, 1 + N, 4, width)` float32: per token row 0 the
RG-LRU state, rows 1..3 the convolution's last three projected inputs, oldest first. One
tensor, so `initial_carry`, `step` and `sequence` keep their signatures; only the shape
changed. Parameters: **1,997,681** (fourier), 1,988,465 (grid_cells), 2,032,241 (v2_codes).

`step` and `sequence` share one code path (`_forward`, `step` is T = 1), including the
convolution's taps and the scan.

### The pieces, with their source

- **Encoder** (v2 `network.py` `Encoder`, `KindProjection`): per kind (ship, game, field)
  `gelu(norm(Linear(inputs -> 2 width)))`, then one shared `Linear(2 width -> width)` and
  norm. Fields then go through `Linear(width -> 64, no bias)` and a norm (v2's map memory).
  Inputs: ship = `Obs.ship` + position code of `pos + offset`; game = `Obs.globals` +
  position code of `offset` (the arena centre in the shifted frame, as v2's global token);
  field = `Obs.field` + position code of `field_pos + offset`.
- **Spatial** (v2 `spatial.py` `SpatialBlock`): pre-norm attention of the M trunk tokens
  over the M tokens and the F fields in one softmax; the fields with their own norm and
  `Linear(64 -> 2 width, no bias)` key/value; element-wise gate; output projection; pre-norm
  gated MLP `down(gelu(gate x) * up x)` at 4 x width. No biases (v2).
- **Gate** (Qiu et al. 2025, G1): `attended * sigmoid(W x)`, x the normed query token. W is
  the fourth block of the query/key/value projection (`query_key_value_gate`, width -> 4
  width, no bias), so it is initialised exactly as the query projection is, the released
  code's arrangement (Qwen3-Next, the official repo). Measured on unit-RMS inputs: mean
  0.500, std about 0.13. **No null key.**
- **Temporal** (v2 `temporal.py` `TemporalBlock`; Griffin section 2.3): `normed = norm(x)`;
  `recurrent_input` (width -> width, no bias); depthwise causal convolution of 4 taps;
  RG-LRU whose two gates read the convolved signal (one `Linear(width -> 2 width)`);
  `state * gelu(gate_branch(normed))`; `output` (no bias); residual; then a pre-norm gated
  MLP of 4 x width (v2 has it: `TemporalBlock.feedforward`). Branch width = width (v2; the
  paper's 4/3 is not taken).
- **Recurrence over time**: `doubling_scan` (recursive doubling, log2 T rounds; the form of
  `framework-spike/update_bench.py`). h_t = keep_t (a_t fresh_t h_(t-1) + u_t): reset
  (`fresh` 0) and the key mask (`keep` 0) are folded into the decay and the update, so the
  scan is plain.
- **Convolution and resets**: the tap at lag k for tick t is `x_(t-k)` times the product of
  `link_j = keep_j * fresh_(j+1)` over j = t-k .. t-1: it is cut by a reset inside the
  window or by the row being masked in between. The buffer leaving a segment is the same
  taps at a virtual tick T with no reset. This is what `step` does tick by tick (zero the
  buffer at a reset; zero it after a masked tick); tested by sequence-equals-steps with
  resets at tick 0 and inside the segment, and by "a reset forgets state and buffer".
  v2 cuts taps at starts only (it has no key mask in the temporal block) and ignores
  `starts[0]`; lean's `reset[0]` zeroes the carry, as before.
- **Rotary** (`position.Rotary`): v2's layout, two axes per band, each band's axes turned
  35 degrees from the coarser band's, periods 64 m x 3.405^k; heading harmonics 1 and 2;
  the same layout in both heads (one table for all heads). 5 bands: 64, 218, 742, 2,527,
  8,603 m. 12 pairs = 24 of each head's 64 dimensions (37.5 %). Values rotated by the key's
  angles and the attended sum rotated back by the query's (`rotate_values=True`). Ships
  rotate by their true position and heading; fields by their centre with heading 0 (v2's
  memory tokens); the game token by "centre" (see choice 1).
- **Position codes** (`position.py`, chosen by `ModelConfig.position_code`, one module in
  the graph): "fourier" (lean's hexagonal Fourier features, 3 directions x 6 periods from
  64 m to 20 km, 36 features); "grid_cells" (Blair et al. 2007's three-grating model with
  Blair's gain `exp(0.3 (u + 1.5)) - 1`, spacings = the same 6 periods, 2 x 2 spatial phases
  per spacing, 24 features); "v2_codes" (v2's 3-colour categorical code at sigma 0, 9
  levels, outer product of the x and y codes per level, 81 features, period 236,196 m as a
  new v2 run's, finest colour 12 m; 30 lines). The offset range is the chosen code's
  coarsest period (`ModelConfig.offset_span`).
- **Heads** (v2 `heads.py` `HeadGroup`): `last(gelu(norm(first x)))`, hidden 256, the first
  projections of a group one matrix; group 1 policy (42) and value (levels x 64), group 2
  the next-state head (only in `sequence`). Initialisation v2's spec 7: first blocks
  orthogonal gain sqrt 2, lasts orthogonal gain 0.01, biases zero. No final norm before the
  heads (v2 has none; the heads norm their hidden layer).
- **Critic bins**: 64 per level (`ModelConfig.bins`), linear; HL-Gauss with sigma 0.75 bin
  widths is the learner's (`LearnConfig.value_sigma_ratio`, unchanged). Four levels, as the
  learner has today.
- **Initialisation of the temporal sublayer** (paper text + recurrentgemma code, read
  2026-11-23 from `recurrentgemma/torch/modules.py` and `layers.py`): decay so a^c is
  uniform in [0.9, 0.999], c = 8 (paper); gates N(0, 1/width) (LeCun normal), biases zero;
  `recurrent_input` and `gate_branch` N(0, 1/width); convolution N(0, 0.01/4), bias zero;
  `output` N(0, s/width) and the MLP's down projection N(0, s/hidden) with s = 2 /
  residual layers (the code's `final_w_init_variance_scale`; 6 here, see choice 4); the
  MLP's up projection N(0, 1/width) (the code's `Einsum`, std sqrt(1/width)).

### Changes to `lean/learn` (call sites only)

- `collect.carry_axis`: the environment axis is 1 for any carry of 4 or more dimensions (the
  model's carry is now 5-D).
- `learn/tests/standins.py`: the stand-in model config names `offset_span` (what
  `draw_position_offset` reads now) in place of `position_periods`.
- `next_state_target` is unchanged (the same seven State fields exist).

## Choices the author should check

1. **Game token rotary: "centre" (v2), not lean's "zero".** The brief: keep "zero" unless
   v2 differs. v2 differs: its global token and absent zones sit at the map centre with
   heading 0 (`inputs.py`, `query_position`). In lean's true-position frame the arena
   centre is the origin, so this is the identity rotation (lean's old "identity" option).
   It makes ship-to-game-token scores depend on the ship's place relative to the arena
   centre (meaningful: the boundary is centred there) and leaks no offset (rotary uses true
   positions). "zero" stays as an option. The old translation test now runs with "zero" and
   a fixed position code (`test_attention_reads_only_relative_positions`).
2. **Rotary bands: two axes per band (v2), not lean's three.** The decision says "x, y and
   attitude in each head"; v2 uses x and y turned by 35 degrees per band. Coarsest band
   8,603 m, just above the 8,400 m diameter; the next (29.3 km) is 3.5 diameters and is
   dropped. Consequence: one band alone aliases a separation d with d - 8,603 m, so it is
   unambiguous only up to half its period (4.3 km, the radius); ship pairs further apart
   are told apart by the finer bands jointly. If that is not enough, add the 29.3 km band
   (14 pairs, still fits).
3. **Gate after the value back-rotation.** With `rotate_values`, the attended sum is rotated
   back to the query's frame first, then gated, then projected: the gate acts on the head's
   output in the frame the output projection reads.
4. **Depth scale s = 2 / 6.** recurrentgemma counts residual blocks (temporal mixing + MLP);
   here each temporal sublayer and each spatial sublayer is one such block: 2 x (1 + 2) = 6.
   Applied to the temporal output projection and the temporal MLP's down projection only.
   The spatial sublayers keep PyTorch's default init (as v2 and old lean), apart from the
   gate, so the two kinds of sublayer are initialised differently.
5. **Temporal MLP initialised the recurrentgemma way** (the MLP belongs to the temporal
   sublayer the brief put under Griffin's init). The spatial MLPs keep the default.
6. **sqrt(1 - a_t^2) is kept at a reset** (v2 and old lean; the recurrentgemma code uses 1 at
   a segment's first position). Not on the list, so v2's. Floor 1e-6 under 1 - a_t^2 (v2).
7. **Heads ported from v2** (hidden 256, norm, gelu; orthogonal init). Not on the list; the
   departure audit flagged lean's single linear heads as an unrecorded departure. The
   next-state head's last layer starts at orthogonal 0.01 rather than v2's zero: v2 zeroes
   it because its next-state logits sit on a baseline code, which lean's regression head
   does not have; a zero last layer would also leave the head's first layer without a
   gradient on the first step.
8. **Next-state head stays lean's regression** of seven scaled changes (recorded choice 11),
   not v2's symlog HL-Gauss delta code; no enemy-action head; outcome stays per-ship (no
   outcome head on the game token). These are learner-side decisions.
9. **Norm epsilon: torch's default** (the dtype's machine epsilon, v2's `layers.norm`), in
   place of lean's 1e-6.
10. **The model no longer holds value supports** (`value_low`, `value_high`, the
    `bin_centres` buffer): nothing read them; the learner owns the support.
11. **Grid cells**: Blair's exponential gain (smooth, non-negative) rather than a rectifier;
    2 x 2 phases per spacing (translations on a half-lattice grid of the unit cell);
    spacings equal to the Fourier periods. All three are my picks.
12. **v2_codes period**: a new v2 run's 236,196 m (36 m finest level). For a 4.2 km arena
    most of the coarse levels carry nothing but the offset; a shorter period would be a
    departure from v2.
13. **Fields' rotary heading**: heading 0 (identity in the heading pairs), as v2's memory
    tokens. "No heading" could also mean cos = sin = 0 there.
14. **Sizes**: v2's defaults (width 128, 2 blocks, 2 heads, 2 spatial per block, map 64) and
    the champion's order; nothing in the old NOTES recorded a reason to differ.

## What was not done

- No `torch.compile` of `sequence` or of a backward pass; no bf16 `sequence`.
- The ops tripwire (650) is exceeded: 509 compute + 294 view = 803 per `step` (below). The
  added structure is the decided one (conv, gate branch, temporal MLP, field memory, gate,
  head MLPs). About 150 of the compute ops are eager RMSNorm decompositions (25 norms x 6),
  which a compiled graph fuses.
- No parity test against v2's modules (transition tests): the maths was read, not compared
  numerically. Candidates: v2's `CausalConvolution.sequence` and `RGLRU` against
  `Temporal` with copied weights.
- Zones (frontline), the absent-zone token, nearest-k attention: not here.
- `lean/eval/matches.py` calls `step` without a position offset (zero offset): unchanged.

## Tests (`lean/model/tests/test_model.py`, 23)

Kept and extended: sequence equals repeated steps (resets at tick 0 and inside the segment,
masks, fields, a random starting carry with its buffer; rotate_values on and off); a reset
forgets state and buffer (outputs after a reset equal a fresh run from tick k); masked rows
keep a zero carry (state and every buffer row); ship padding and permutation (both
positionless options); fields as an unordered memory (permuting, padding with absent
fields, F = 0, carry shape independent of F, present fields are read); attention reads only
relative positions; one set of weights at N = 10, 40, 100 and F = 0, 3, 20; doubling scan
equals the loop (T = 1, 2, 3, 5, 8, 13, 64, float64, 1e-12); Griffin's initialisation
(a^c range, conv std 0.05, zero biases); the gate starts near 0.5; rotary bands stop a
little above the arena; each position code runs in the policy (also under bf16) and its
span is the offset range; position-code symmetries (a grid cell repeats over its lattice
and not over half of it; the colour code is a distribution per level; the Fourier code
repeats after 2 P along x); HL-Gauss targets (64 bins); backward reaches every parameter
and no masked ship row or absent field; the same with the real initialisation; Adam fits a
tiny batch; bf16 step close to float32; compiled `step` matches eager.

Sequence-equals-steps tolerance is 1e-4 (measured 1.4e-5 on logits up to 4.5: float32
rounding of the scan's other summation order).

## Report (2026-11-23)

Run on the laptop (power profile "performance", load about 1.4 from other jobs):

    OMP_NUM_THREADS=1 PYTHONPATH=.:src uv run --no-sync python -m pytest lean -q
    78 passed, 1 xfailed
    OMP_NUM_THREADS=1 PYTHONPATH=.:src uv run --no-sync python -m lean.train --smoke --device cpu
    completes; first value loss 4.157 = ln 64 (uniform over 64 bins)

A rollout of 6 ticks with `GameConfig(field_count=10)` and one PPO update ran (fields
stacked (6, 4, 10, 3) in the rollout).

| measure | value |
|---|---|
| parameters (fourier / grid_cells / v2_codes) | 1,997,681 / 1,988,465 / 2,032,241 (was 616,677) |
| ops per `step` forward, eager, no_grad (regions.py's Counter) | 509 compute + 294 view = 803 (was 289 + 150) |
| ops per `sequence`, T = 8 | 552 compute + 328 view |
| CPU `step`, E = 256, N = 10, F = 10, 1 thread | 144 ms median (137 min) |
| CPU `step`, same, 4 threads | 51 ms median (49 min) |
| GPU (RTX 4070 laptop) `step`, same, bf16 autocast, eager | 3.10 ms |
| GPU, `torch.compile(step)`, bf16 | 1.38 ms steady; first call 10.9 s |

Ops by region (eager `step`, compute; counted at E = 4, so the totals differ from the table by a
few ops): spatial 61 per sublayer (was 85 before rotating
the stacked query/key/value once and building the additive mask once), temporal 72 per
block, encoder 52, heads 23, position code 5 per call (3 calls), rotary 4 per call.
