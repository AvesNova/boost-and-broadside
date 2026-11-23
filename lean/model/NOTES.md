# lean/model

The policy network: v2's model maths with the decided changes (author review, 2026-11-23).
Reasons: `docs/rewrite/model-audit.md`, `docs/rewrite/literature-model-choices.md`; decisions:
`docs/rewrite/core-decisions.md`; history: `docs/rewrite/core-worklog.md`.

## Entry points

- `Policy(ModelConfig())` (`policy.py`): `initial_carry`, `step(obs, carry, position_offset, reset)`
  (one tick, rollout) and `sequence(obs, carry, reset, ...)` (T ticks, the learner). One code
  path; `step` is T = 1. Inputs are `lean.contract`'s feature counts. Outputs are per ship:
  42 policy logits, 5 value levels x 64 bins; `sequence` also returns the next-state logits
  (6 x 64) and the hit logits (3).
- `agent.py`: how a network acts, shared by `lean.learn`'s rollout and `lean.eval`'s runner:
  `Network` (a weightless `Stepper` template per architecture plus weights; new weights are new
  data, so no recompile), `network_step`, `sample_joint`, `seat_view`, `precision`, `reset_carry`.
- `temporal.py` (`Temporal`, `doubling_scan`), `position.py` (position codes, `Rotary`),
  `value.py` (`hl_gauss_targets`, `decode_value`, `value_bin_centres`).
- `next_state_target(config, state, next_state, step_out, tick_length)` makes the next-state
  head's targets of one game step (collect records them); `next_state_cross_entropy` is the loss.

## Shape

Width 128, 2 blocks (each one temporal sublayer, then two spatial), 2 heads of 64, MLPs 4 x
width, heads hidden 256, map memory width 64. Trunk tokens `[game, ship 0 .. N-1]`; fields are
keys and values only. Carry `(blocks, E, 1 + N, 4, width)` float32: per token the RG-LRU state
and the convolution's last three inputs. **2,155,565 parameters** (checked 2026-11-24).
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

64 bins per column, HL-Gauss cross-entropy (sigma 0.75 bin widths), nats per column: ln 64 =
4.159 at a uniform prediction; the targets' own entropy, the floor, is 1.20. Ranges from
5.55M rows (scripted yardstick against itself, 4v4, training spawn, fields, fog, 2026-11-23);
linear ranges sit just outside the 0.1st and 99.9th percentiles.

| Column | Target | Bins | Range | Outside |
|---|---|---|---|---|
| along, across (m) | move minus world velocity x tick, in the nose frame | linear | +-0.35, +-0.6 | 0.15%, 0.13% |
| heading (rad) | wrapped heading change minus nose rate x tick | linear | +-0.05 | 0.12% |
| speed (m/s) | change of proper speed | linear | +-3 | 0.10% |
| energy (store) | change | linear | +-700 | 0.02% |
| shield (points) | change minus the recharge given | symlog | -90 to 0.3 | 0 |

The hit head's shares, from 5.52M bullets that ended (ghosts included): 94.57% missed, 4.44% an
enemy, 0.99% a team-mate (`hit_class_frequencies`).

## Tests

`tests/test_model.py` (28): `sequence` equals repeated steps (resets, masks, fields, rotated
values on and off); a reset forgets state; padding and permutation leave real ships unchanged;
fields are an unordered memory; attention reads only relative positions; the rotary band ladder
and the coarsest band's recovery of separations to 0.1 m; one set of weights at any N and F;
initialisation and the heads at their starting values; a state from zero takes its first input
unscaled; next-state targets, bins and loss; the position codes; HL-Gauss targets; backward
reaches every parameter and no masked row; Adam fits a tiny batch; bf16 is close to float32;
compiled `step` matches eager; the model reads fog.

## Open

- The next-state ranges and the hit shares are the core game's of 2026-11-23 (P-51D, two-loop
  store); measure again when the flight numbers or the lineup change (one scripted collection
  with `next_state_target` per tick).
- Grid cells (gain, phases, spacings) and the `v2_codes` period are builder picks.
- No parity test against v2's modules; no compiled or bf16 `sequence` test here (the learner's
  compiled tests cover the loss graph).
- Not here: zones, nearest-k attention, an enemy-action head, a game-token outcome head (an
  ablation arm only).
