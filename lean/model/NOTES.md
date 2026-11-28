# lean/model

The policy network: v2's model maths with the decided changes (author review, 2026-11-23).
Reasons: `docs/rewrite/model-audit.md`, `docs/rewrite/literature-model-choices.md`; decisions:
`docs/rewrite/core-decisions.md`; history: `docs/rewrite/core-worklog.md`.

## Entry points

- `Policy(ModelConfig())` (`policy.py`): `initial_carry`, `step(obs, carry, position_offset, reset)`
  (one tick, rollout) and `sequence(obs, carry, reset, ...)` (T ticks, the learner). One code
  path; `step` is T = 1. Inputs are `lean.contract`'s feature counts. Outputs are per ship:
  42 policy logits, `value_levels` value levels x 63 bins (3: the learner's, lean.learn; 5 with
  the objective levels); `sequence` returns a `Sequence` (NamedTuple)
  that also holds the next-state logits (6 x 63), the hit logits (3), the map-edge logits
  (16) and, given a `PairQuery`, the pairwise head's logits (`geometry.py`).
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
- `geometry.py`: the relative-geometry heads (`PairHead`, `PairQuery`, `GeometryColumn`, the
  columns, the circular supports and their HL-Gauss targets); the learner picks the pairs and
  makes the targets (`lean.learn.geometry`).

## Shape

Width 128, 2 blocks (each one temporal sublayer, then two spatial), 2 heads of 64, MLPs 4 x
width, heads hidden 256, map memory width 64. Trunk tokens `[game, ship 0 .. N-1]`; fields are
keys and values only. Carry `(blocks, E, 1 + N, 4, width)` float32: per token the RG-LRU state
and the convolution's last three inputs. **2,301,906 parameters** with the geometry heads
(149,168 of them the geometry heads), 2,152,738 without (`geometry_heads=False`). The speed of
the whole loop is in `lean/learn/NOTES.md`.

## The pieces

- **Encoder:** per kind (ship, game, field) `gelu(norm(Linear))`, then a shared Linear and norm.
  The game token = `Obs.globals` (`contract.GLOBAL_COLUMNS`, 9 since 2026-11-28: the zone now
  and at its stage's end, the time to the stage's end, the alive fractions) + the position code
  of the map centre (the origin, offset like any token). A checkpoint from before 2026-11-28
  (3 globals) does not load: the game encoder's first Linear changed width.
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
  (`sequence` only): next state, hit (missed, hit an enemy, hit a team-mate) and the map edge.
  The pairwise head (`sequence` only, given pairs) below.

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

The hit head's shares (`hit_class_frequencies`), re-measured 2026-11-28 on the current game (hit
damage 8, the egg deflector), the scripted yardstick against itself, 4v4, training game, 48
environments x 4,200 ticks on the CPU (finished games re-dealt), 1.6M bullets that ended (ghosts
included): 98.65% missed, 1.10% an enemy, 0.25% a team-mate (was 93.74 / 5.47 / 0.80 from
2026-11-25, before the egg cut the hit width). The correctness review's shorter sample (32 x 600
ticks, mostly the approach, which has no hits) gave 0.70% / 0.15%. The enemy
class is rare: watch whether the hit head learns beyond the prior.

## Geometry heads (queue item 8c, author 2026-11-28)

Representation shaping: the current geometry as the team sees it, exact targets from the game's
own functions (`lean.game.pair_geometry`, `field_edge_distance`, `geometry_record`'s edge
column; `lean/game/tests/test_geometry.py`). Rows the team sees: its living ships and the
enemies visible this tick; a pair with a hidden or dead ship is masked (`lean.learn.geometry`).

- **Pairwise head** (`PairHead`): per ship two gated projections of its trunk token (norm,
  then value x sigmoid(gate) to 128), u (the ship's side, shared) and v (the other ship's
  side); fields their own other-side projection of the map memory. A pair is GELU(u_i + v_j)
  (`pair_combine` "product": u_i x v_j), then one Linear per target; gating only in the
  per-ship stage, so the pair stage is one add, one GELU and the last Linears. The learner
  samples K pairs per view and tick before the pair layers (cost scales with K).
- **Ship pairs** (first < second, the upper triangle): range, range rate, line-of-sight rate
  (symmetric) and the energy difference (antisymmetric: first minus second; the lower triangle
  would be its negative) on one order; the antenna train angle and the gun solution error in
  both orders (first on second, second on first). Aspect angle and heading crossing angle
  follow from the train angles in both orders (aspect = pi - the other's train angle; the
  crossing angle = own train angle - the other's + pi), so they are not predicted.
- **Ship-field pairs:** the signed distance to the field's edge (the middle of its band,
  negative inside).
- **Map edge:** the signed distance beyond the zone's current edge (`lean.game.boundary_distance`
  at the row's tick: it follows the shrinking radius), per ship on the auxiliary head group.
- The gun solution error: from the launch direction of a shot now (velocity + muzzle along the
  nose, over the map, no spread) to the lead point (the target at its world velocity after the
  bullet's drag-law time of flight, 3 fixed-point passes, capped at the bullet's life). A
  bullet the game's `bullet_motion` flies along it passes the target within 3 cm (open map,
  150-550 m; the test).

Supports: every column a categorical in nats, HL-Gauss sigma 0.75 bin widths. Linear and
symlog columns use `buffered_centres` (the measured range plus 3 buffer columns each end, as
every support here). Angles are a circle of bins over (-pi, pi], uniform in s = symlog(angle,
knee) (knee 0: uniform), the Gaussian wrapped onto the circle (`circular_hl_gauss_targets`:
each bin takes the mass at its nearest image; no seam at +-pi).

Ranges measured 2026-11-28 (scripted yardstick vs itself, 4v4 training game, 64 environments x
1,800 ticks, every 4th tick, both views; 0.70M visible pairs, 3.0M ship-field pairs, 0.30M
ships): each range the 99.9% interval rounded out; each knee about the finest scale that
matters. Bin widths in target units: innermost, outermost.

| Column | Bins | Range | Knee | Bins: innermost, outermost | Measured 0.1% / 50% / 99.9% (max) |
|---|---|---|---|---|---|
| range (m) | 32 | 0 to 6,700 | 100 | 18 m, 1.75 km | 28 / 956 / 6,686 (7,423) |
| range rate (m/s) | 32 | +-1,000 | 50 | 13, 471 | -723 / -80 / 568 (-1,088, 987) |
| line-of-sight rate (rad/s) | 32 | +-6 | 0.05 | 0.021, 6.1 | -5.8 / 0 / 5.8 (+-280 when touching) |
| energy difference (store) | 32 | +-33,000 (the full store, 32,979) | linear | 2,640 | -32,810 / 0 / 32,790 |
| train angle (rad) | 32 | circle | uniform | 0.196 (11.25 deg) | about uniform, 8% within 0.1 |
| gun error (rad) | 32 | circle | 0.02 | 6.9 mrad, 0.73 | 0.9% within 0.01, 8% within 0.1 |
| field edge (m) | 32 | -2,000 to 7,000 | 50 | 19 m, 5.8 km | -1,264 / 2,214 / 6,136 |
| map edge (m) | 16 | -4,200 to 300 | 100 | 66 m, 10 km | -4,132 / -2,274 / -427 |

Knees: range 100 m (collisions are tens of metres, the gun range 697 m); range rate 50 m/s; the
line-of-sight rate 0.05 rad/s (a sustained turn is 0.70 rad/s); the gun error 0.02 rad (a hit
needs about 8 mrad at 300 m: half the 4.75 m hit width); the field edge 50 m (the band is 160
m wide); the map edge 100 m (the boundary ramp is 300 m). 16 bins only for the map edge (the
scripted agent rarely nears it; coarse is enough). The bounded columns (range, energy
difference, map edge) keep the buffer columns beyond their hard bounds (unused bins, 3 of 32
or 16), for one support layout everywhere.


`tests/test_model.py`: `sequence` equals repeated steps (resets, masks, fields, rotated
values on and off); a reset forgets state; padding and permutation leave real ships unchanged;
fields are an unordered memory; attention reads only relative positions; the rotary band ladder
and the coarsest band's recovery of separations to 0.1 m; one set of weights at any N and F;
initialisation and the heads at their starting values; a state from zero takes its first input
unscaled; next-state targets, bins and loss; the position codes; HL-Gauss targets; every support's 63-bin layout (0 a centre of a
symmetric range), the symlog round trip and HL-Gauss decoding within a bin on every support; backward
reaches every parameter and no masked row; Adam fits a tiny batch; bf16 is close to float32;
compiled `step` matches eager; the model reads fog; the circular bins wrap with no seam
(both knees), the gun-error circle is finest near zero, every geometry support's size; the
pairwise head reads only its two ships and the order.

## Open

- The value supports, next-state ranges and hit shares are the game's of 2026-11-25; measure
  again when the flight numbers, the lineup, the clock or the scripted agent change
  (`docs/rewrite/sections/scoreboard/target_ranges.py`, about a minute on the laptop GPU). Due:
  since then the hit damage (21 to 8), the bullets (ghost-only, every ship a bullet a tick), the
  egg deflector, collisions, the post-stall levels and the look-ahead scripted agent changed.
- Grid cells (gain, phases, spacings) and the `v2_codes` period are builder picks.
- No parity test against v2's modules; no compiled or bf16 `sequence` test here (the learner's
  compiled tests cover the loss graph).
- The geometry ranges are the game's of 2026-11-28 (re-measure with the others).
- The pairwise head has a ship and a field other-side projection; zones would add theirs.
- The map centre is the origin in three places: the game token's position code (`code(offset)`)
  and its rotary (identity: position 0, heading 0), and the fields' rotary heading (toward the
  origin). The zone's centre is the origin today (item 13); a moving centre (staged zones)
  would move the game token's code and rotary to the zone's centre (from Obs, not from the
  scaled globals) or add the zone as its own key. The position codes themselves (Fourier, 64
  m to 20 km, random offset over the coarsest period) and the rotary reach (twice the arena's
  diameter, `ModelConfig.for_arena` on the radius at the deal) read no zone and need no change.
- A checkpoint saved before the geometry heads does not load into a ModelConfig with them on
  (missing keys): load it with `geometry_heads=False` (no shim kept).
- Not here: zones, nearest-k attention, an enemy-action head, a game-token outcome head (an
  ablation arm only).
