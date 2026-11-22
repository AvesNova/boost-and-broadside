# lean/model: notes

## What is implemented

- `policy.py`: `ModelConfig`, `Policy` (`initial_carry`, `step`, `sequence`), the layers
  (`Encoder`, `Temporal`, `Spatial`, `Block`), `position_wave_vectors`,
  `draw_position_offset`, `next_state_target`.
- `value.py`: `value_bin_centres`, `hl_gauss_targets`, `decode_value`.
- `tests/test_model.py`: ten tests (below).

Shape of the model: width 128, 2 blocks, 2 heads; each block is one temporal sublayer
then two spatial sublayers. Trunk tokens are `[global, ship slot 0 .. N-1]`; the carry is
`(blocks, E, 1 + N, width)` float32. The trunk layers follow the parity-tested spike
(`docs/rewrite/sections/framework-spike/torch_bench.py`): same RG-LRU maths, same
`rotate`, same gated MLP, same pre-norm layout. Differences from the spike: the null key,
the wave vectors (three axes instead of x and y), heading rotary, the position code in
the token, the carry mask and reset, time-first layout `(T, E, ...)` as the contract
states (the spike was `(E, T, ...)`), four value levels, the next-state head.

`step` and `sequence` share one code path (`_forward`; `step` is `T = 1`), so they
cannot drift apart. `step` does not compute the next-state head.

Two keyword arguments beyond the contract, both optional: `position_offset` on `step`
and `sequence`, and `reset` on `step` ((E,) bool, zero the carry before this tick, the
single-step form of `sequence`'s `reset`).

## What was run

CPU, torch 2.15, eager, float32, from the repository root:

    PYTHONPATH=. <venv>/bin/python -m pytest lean/model/tests -q -p no:cacheprovider
    10 passed in 1.87s

- `sequence` equals repeated `step` calls (with resets, masked rows, a non-zero starting
  carry and an offset), for `rotate_values` off and on.
- Masked rows keep a zero carry.
- Real ships' logits, values and carry are unchanged by padding with masked rows and are
  permuted by a permutation of the slots, for both `positionless_rotary` settings.
- Ship outputs do not change when every ship is translated together and the token code is
  held fixed (so attention sees no absolute position). Tolerance 2e-3 on logits: the
  phases of the 64 m scale lose float32 precision at kilometre coordinates.
- One set of weights runs at N = 10, 40, 100.
- Decay initialisation: a^c within [0.9, 0.999].
- Default periods: six, from 64 m, coarsest above 16 km, ratio not an integer.
- HL-Gauss targets sum to 1 and decode to the value (within 1 % of a bin) when the value
  is three or more bins inside the support.

Measured with contract sizes (SHIP_FEATURES 27, GLOBAL_FEATURES 3):

- Parameters: **616,677**.
- Operations per `step` forward (a dispatch counter with the same rule as
  `docs/rewrite/sections/simplification/regions.py`: every aten call, metadata calls
  excluded): **289 compute + 150 view = 439**, eager, under `no_grad`. The tripwire is
  650. I used a copy of the counting rule, not `regions.py` itself. `sequence` at T = 8:
  348 compute + 203 view (the scan adds 5 compute per tick per block).
- CPU time per `step`, 4 threads, a box shared with other jobs, so indicative only:
  about 50 ms at E = 256, N = 16 and about 70 ms at E = 64, N = 100.

## What was not run

- No GPU, no `torch.compile`, no bf16 autocast, no backward pass, no training.
- No parity check against the spike's JAX twin for the changed parts.
- `draw_position_offset` and `next_state_target` are not tested.
- `next_state_target` reads field names from `lean.contract.State`; the scales in
  `ModelConfig.next_state_fields` are guesses that have not been compared with the game's
  per-tick changes.

## Choices the audit and brief left open

1. **Null key.** One learned vector (`Policy.null_token`), shared by every spatial
   sublayer, appended as the last key and value in each sublayer after passing through
   that sublayer's norm and key/value projection. It is never a query, has no residual
   stream and no carry. Reason: "a key that is always attendable" is all the audit asks
   for, and a key with no stream cannot accumulate state.
2. **Global token.** A full trunk token (query and key, with a carry), always attendable.
   It is the only path by which `obs.globals` reaches the ships.
3. **Tokens with no position (lesson 4).** Default `positionless_rotary = "zero"`: the
   global and null tokens get cos = sin = 0 in the rotary terms, which removes the rotated
   dimensions (42 of the 64 in each head) from every score that involves one of them, as
   query or as key. They interact through the 22 unrotated dimensions only, so they are
   position-free rather than "a ship at the arena origin heading east", and no absolute
   position leaks into attention. The task text asked for the identity rotation; that is
   `positionless_rotary = "identity"`, kept as an option and tested for the invariances,
   but with it the score between a ship and the global or null token depends on the
   ship's true absolute position and heading, which is the failure lesson 4 describes and
   also defeats the random offset. **This default is a deviation from the letter of the
   task; flip it if the identity form is wanted.**
4. **Rotary layout.** 18 position pairs (3 axes x 6 periods, the same wave vectors as the
   token code) plus 3 heading pairs (harmonics 1, 2, 4) = 21 pairs = 42 dimensions of
   each 64-wide head; the same rotation in both heads. Rotary uses the true positions
   and the token code uses offset positions: only differences enter a score, so the
   offset would cancel.
5. **Periods.** Geometric from 64 m to 20 km, ratio 3.155. The 20 km assumes an arena
   diameter under 10 km (the spike used radius 4 km). **The model does not read the game
   config; whoever builds the policy must check `max(position_periods) > 2 * diameter`.**
6. **Position offset.** Supplied by the caller, `(E, 2)` (or `(T, E, 2)` in `sequence`).
   `draw_position_offset` draws it uniformly over one coarsest period. It must be redrawn
   at each match start and stored with the rollout, or the learner's forward differs from
   the rollout's.
7. **Temporal block.** The RG-LRU inside a pre-norm residual, as in the spike, not
   Griffin's full recurrent block (two linear branches, a temporal convolution and a GeLU
   gate branch). Reason: reuse of the parity-tested form, fewer operations. The
   recurrence gates are computed from the normed token and the recurrence runs in float32.
8. **Carry rules.** `reset` zeroes the carry before the tick's update (all tokens of that
   environment, the global token included). A row with `key_mask` False has its state
   set to zero after the update, so a ship that dies loses its memory at once.
9. **Value support.** Uniform bin centres, one `(low, high)` per reward level
   (`value_low`, `value_high`, default -2 to 2 for every level, a placeholder). Reason:
   HL-Gauss is defined on uniform bins, and lesson 2 (means of a wide symlog categorical
   are tail-dominated) argues for narrow linear supports now that rewards are averaged per
   ship. The centres are in the buffer `Policy.bin_centres` `(LEVELS, BINS)`. The learner
   must set the supports at least three bins wider than the returns on each side.
10. **HL-Gauss.** The paper's definition: Gaussian of sigma 0.75 bin widths integrated
    over each bin, renormalised over the support; values clamped to the outermost centres.
    Written from the definition rather than lifted from v2's `scalar_encode`, which
    carries a moment-matching path this does not need.
11. **Next-state head.** Plain regression of the scaled one-tick change of seven State
    fields (x, y, heading_cos, heading_sin, speed, energy, shield), one linear layer.
    Reason: it is the simpler form (one output per scalar, no supports to choose, no
    periodic code), and the audit rates the head category (d) with no outside support, so
    the cheapest version is the right first arm. The loss (Huber or squared error) and the
    mask (rows alive in both ticks, no reset between) are the learner's.
12. **Head initialisation.** Policy head weights scaled by 0.01 and bias zero; value head
    all zero (a uniform first value distribution). Other layers keep PyTorch's default.
13. **`rotate_values`** (off): values are rotated by the key's terms and the attended sum
    rotated back by the query's. With `positionless_rotary = "zero"` the positionless
    tokens then pass nothing through the rotated value dimensions.
14. **Nearest-k attention** is not implemented (an arm in the audit's list).

## Check first on the GPU

1. `torch.compile` of `step` and of `sequence`: the scan is a Python loop over T, which
   the compiler unrolls; check compile time at the learner's T and that N changing
   between runs does not recompile per call.
2. bf16 autocast: the recurrence casts to float32 and the scaled-dot-product call takes
   bf16. CPU result below: autocast did NOT leave the phases float32 (it casts `@`);
   fixed. Still to do on the GPU: the same drift measurement under cuda autocast.
3. Speed and operation count compiled, against the 650 tripwire and the spike's figures.
4. Float32 phase precision: positions near the arena edge plus an offset of up to 10 km
   against the 64 m period; the translation test's 2e-3 tolerance is where to look.
5. The fused attention path with a boolean mask of shape `(B, 1, 1, M + 1)` and unequal
   query and key counts.

## Backward, learning, compile and bf16 checks (CPU, torch 2.15, OMP_NUM_THREADS=1)

Run after the above; three tests and one fix added. Suite now:

    PYTHONPATH=. <venv>/bin/python -m pytest lean/model/tests -q -p no:cacheprovider
    14 passed

**Backward** (`test_backward_reaches_every_parameter_and_no_masked_row`, T = 8, E = 4,
N = 10, one slot never live, resets at (3, 1) and (5, 3), random starting carry; loss =
policy cross-entropy + value cross-entropy over 4 x 77 bins + squared error on next_state,
live rows only, plus the mean square of the final carry). Every parameter gets a finite,
non-zero gradient; none is dead. Gradients of the loss with respect to `ship`, `pos`,
`heading` are exactly zero on masked rows and non-zero on live rows; the starting carry's
gradient is exactly zero for slots masked at tick 0. Replacing masked rows' ship, pos and
heading with large values leaves the loss and the final carry unchanged (atol 1e-5).
`test_gradient_without_perturbed_heads_or_loss_on_carry` repeats the dead-parameter check
with the real initialisation (value head zero) and the loss alone: also none dead. (With
the value head at zero the value loss sends no gradient into the trunk on the first step;
the trunk's gradient then comes from the policy and next-state losses. That is by design
of the zero init, not a dead weight.)

**Learning** (`test_adam_fits_a_tiny_fixed_batch`, T = 8, E = 2, N = 6, real init, Adam
lr 1e-3, 60 steps, policy cross-entropy to fixed random actions on live rows): 3.738 ->
0.083. The test asserts a halving.

**torch.compile, default options, eager-vs-compiled on live rows and carry.** Measured
with four compile processes running at once on a 4-core machine, so the times are upper
bounds; measured BEFORE the float32-phase fix below, except where stated.

| what | N | first call | recompiled | max abs error |
|---|---|---|---|---|
| `step` | 10 | 31.4 s | (first compile) | 1.7e-6 |
| `step` | 20 | 21.9 s | yes: unique_graphs 1 -> 2 | 1.7e-6 |
| `step` | 40 | < 0.01 s | no | 1.7e-6 (carry 3.1e-6) |
| `step` | 10 again, 30 | < 0.01 s | no | 2.2e-6 |
| `step`, `dynamic=True` | 10 | 40.6 s | (first compile) | 1.7e-6 |
| `step`, `dynamic=True` | 20, 40, 30 | < 0.01 s | no | 1.7e-6 (carry 3.1e-6) |
| `sequence` T = 4 | 10 | 31.9 s | (first compile) | 2.5e-6 |
| `sequence` T = 4 | 20 | 23.7 s | yes: 1 -> 2 | 2.6e-6 |
| `step`, rerun AFTER the fix | 10 | 5.7 s | (first compile) | 1.7e-6 |
| `step`, rerun AFTER the fix | 20 | 8.4 s | yes: 1 -> 2 | 1.7e-6 |
| `step`, rerun AFTER the fix | 40, 10, 30 | < 0.01 s | no | 2.2e-6 (carry 3.1e-6) |

The rerun had the machine nearly to itself (only the 4 s test suite alongside) and a warm
disk, so 6 to 8 s is the better estimate of a `step` compile; `dynamic=True` and
`sequence` were not rerun after the fix.

No compile errors. Recompile counts are `torch._dynamo.utils.counters["stats"]
["unique_graphs"]`. With default options the first new N costs one recompile, which
makes the ship dimension symbolic; every later N reuses that graph. `torch.compile(...,
dynamic=True)` makes it symbolic from the first compile (one graph in total, about 30 %
slower to compile here). `sequence` at N = 40 and at the learner's T, a changing T (the
loop over T is unrolled, so T is a constant of the graph and each new T will recompile),
E changing, and compiling a backward pass were not run. Steady-state `step` time compiled
was 1.5 to 3.4 ms per call at E = 4 (one thread, noisy, not a benchmark).

**bf16** (`torch.autocast("cpu", dtype=torch.bfloat16)`, `step`, E = 4, heads perturbed
to std 0.1 so logits have std about 1.1; differences on live rows against float32):

| | N = 10 before fix | N = 10 after | N = 40 before | N = 40 after |
|---|---|---|---|---|
| policy logits, max abs | 0.183 | 0.028 | 0.265 | 0.031 |
| value logits, max abs | 0.251 | 0.034 | 0.354 | 0.036 |
| policy probability, max abs | 0.015 | 0.0044 | 0.024 | 0.0029 |
| argmax agreement | 1.00 | 1.00 | 0.953 | 0.992 |
| carry after 1 tick, max abs | 0.104 | 0.0060 | 0.139 | 0.0072 |
| after 32 ticks: logits / carry | 0.182 / 0.366 | 0.029 / 0.027 | 0.181 / 0.481 | 0.032 / 0.030 |

It runs; outputs are bf16, the carry stays float32. The bug found: autocast casts the
`pos @ wave_vectors.T` matmuls to bf16, so both the position code and the rotary phases
(hundreds of radians at the 64 m period) were computed with 8 bits. The fix (the only
change to `policy.py`): the two phase computations sit in
`torch.autocast(device_type, enabled=False)` with `.float()` inputs. Must stay float32:
the phases (now pinned) and the recurrence (already pinned). The remaining 0.03 drift is
ordinary bf16 linear/attention error and does not grow over 32 ticks.
`test_bf16_autocast_step_stays_close_to_float32` holds this (thresholds 0.08 / 0.03).
torch warns once under bf16: "Mismatch dtype between input and weight ... Cannot dispatch
to fused implementation" from `rms_norm` (float32 weight, bf16 input); harmless, slower.
Not run: `sequence` or a backward pass under bf16; anything on a GPU.

## Temporal block against Griffin's recurrent block (not implemented)

Here: `x + RGLRU(norm(x))`, the gates and the recurrence input both taken from the normed
token, the state added to the residual stream with no output projection. Parameters per
temporal sublayer at width 128: two gate linears (2 x 16,512), decay 128, norm 128 =
33,280. Carry: one state, `(blocks, E, 1 + N, width)`.

Griffin's recurrent block (arXiv 2402.19427, section 2.2 / figure 2; v2's `TemporalBlock`
in `src/bnb/models/yemong/temporal.py` is that block at expansion 1 without biases):

    normed = norm(x)
    branch_a = gelu(Linear(normed))                    # gate branch
    branch_b = RGLRU(Conv1D_4(Linear(normed)))         # depthwise causal conv, 4 taps
    x = x + Linear(branch_a * branch_b)

What differs from ours:
1. An input projection before the recurrence (`recurrent_input`), so the recurrent state
   lives in its own basis, not the residual stream's.
2. A depthwise causal convolution over the last 4 ticks before the RG-LRU; the RG-LRU's
   gates see the convolved signal (ours see the normed token).
3. A multiplicative GeLU gate branch on the recurrence output (ours has none).
4. An output projection before the residual add (ours adds the state directly).
5. The paper widens the two branches to 4/3 of the width; v2 does not.
6. The paper follows the block with a gated MLP; v2 puts one (4 x width) in its temporal
   block. Ours has no MLP in the temporal sublayer; the spatial sublayers' MLPs follow.

Adopting it (v2's form, width 128, no branch expansion) would add per temporal sublayer:
three 128 x 128 linears without bias (49,152) and the convolution (4 x 128 weights + 128
bias = 640), 49,792 parameters, 99,584 over the two blocks (plus an MLP per temporal
sublayer if item 6 is taken too). Operations per tick and token: three more matmuls, a
4-tap weighted sum, a GeLU and a product. The carry gains the convolution's buffer of the
last 3 projected inputs, `(blocks, E, 1 + N, 3, width)`: four times today's carry, either
as a second tensor (which changes `initial_carry`, `step` and `sequence` returns: a
contract change) or packed into one `(blocks, E, 1 + N, 4, width)` tensor (shape changes,
signatures do not). Reset and the key mask must zero the buffer as well as the state, and
`sequence` needs the buffer handled across resets inside a segment (v2 does this). The
sequence-equals-steps and masked-carry tests would need to cover the buffer.
