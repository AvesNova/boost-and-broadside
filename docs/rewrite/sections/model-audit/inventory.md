# Inventory: Yemong (v2) model and learner, champion configuration

Factual inventory from code only. Nothing here judges a choice or records why it was made.

**Scope.** Preset `real_scale_all_modes`, `TrainConfig.yemong` defaults, PPO with league slots, BC -> RL chain, as `docs/rewrite/sections/launch/launch.toml` selects. Paths are relative to `src/bnb/` unless they start with `docs/`.

**How the numbers were checked.** Every width, bin count and parameter count below was either read from the source line cited or printed by building the model on the CPU from `TrainConfig(game=PRESETS["real_scale_frontline"], lineup=(Yongma, Yongma, Jujak, Jujak, Jujak))` (torch 2.15, in a scratch environment outside the repository). Built model: 2,852,618 parameters (trunk blocks 1,675,008; encoder 283,008; all heads 894,602).

**Sizes in the champion configuration.** 5 ships per team (S = 5, N = 10 rows per view), 10 field slots (M = 10, all present), 5 zone slots (Z = 5), 2 teams, tick 1/30 s, 3 ship types on the card (`Yongma`, `Jujak`, `vizia`), arena radius 7800 m (frontline, elimination) or 4200 m (deathmatch). Query tokens per view Q = N + 1 = 11; attention keys per view Q + M + Z = 26.

**Cost column legend** (from `docs/rewrite/sections/simplification/records/`):
- **R** = `v2_ppo_2048_1024.txt`: compute ops in one rollout tick (2048 training + 1024 lane environments; the tick calls a Yemong controller 3 times: learner, opponent slots, lanes). "per call" divides by the calls the record lists. Whole tick: 27,019 ops; stage `_act` 21,352.
- **L** = `v2_learn_ppo_L16_L32.txt`, the segment-length-32 block (384 views): compute ops per phase. Phases: prepare 353,330 (once per update); label codes 2,338 (per micro-batch); forward and loss 4,796 (per micro-batch); backward 782 (per micro-batch); clip and optimizer 2,808 (per minibatch); heads' sums 10,916 (cadence only).
- **F** = `v2_cost_ppo_L32.txt`: FLOPs per view-tick (micro-batch of 12 views, L = 32): forward 60.6 M, forward + backward 213.6 M.
- The records do not state which build they traced; rows quote them as given. "not in records" means no line of those files names the region.

---

## 1. Inputs

### 1a. What is observed (raw observation contract)

| # | Name | What it computes | Where | On? | Cost | Depends on |
|---|---|---|---|---|---|---|
| 1.1 | Ship rows of a team view | (N, 29 columns) per view, own team's S rows first, then enemies. Columns: x, y, heading cos/sin, velocity x/y, proper_speed, angle_of_attack, nose_rate, energy, shield, shield_recharge_wait, gun_cooldown, log_refractive_index, log-index gradient x/y, acceleration x/y, nose_acceleration, acceleration_valid, alive, visible, fired, respawn_wait, ship_type, team_relative, time_since_seen, spawned, benched. The first 20 are zero for a ship the team cannot see. World coordinates, no rotation or mirroring. | `contracts/obs.py:21-66` | yes | R: game perceive 828 ops per tick | 2.1 |
| 1.2 | Columns observed but not read by the model | `proper_speed` and `time_since_seen` are in the observation and are not read by `inputs.compose` (the model uses its own belief age). | `models/yemong/inputs.py:360-555` | n/a | not in records | 1.1 |
| 1.3 | Field rows | (M, 6): x, y, radius, band_width, level, present. | `contracts/obs.py:70` | yes | not in records | 2.3 |
| 1.4 | Zone rows | (Z = 5, 11): x, y, radius, role (0-4), progress, direction, offensive_distance, defensive_distance, present, lock_own, lock_enemy. Z is the constant `ZONE_SLOTS = 5` in every mode; a mode without zones shows them absent. | `contracts/obs.py:79-91, 131` | yes | not in records | 2.4 |
| 1.5 | Global row | 15 columns: episode_progress, arena_radius, fogged, reset, boundary_fraction, boundary_time_to_zero, three mode flags, score_own, score_enemy, score_target, front, front_threshold, next_wave_own. | `contracts/obs.py:104-120` | yes | not in records | 2.2 |
| 1.6 | No bullet observation | The observation has no bullet part; the model reads no bullets. | `contracts/obs.py:171-173` | n/a | n/a | 5.9, 8.8 |
| 1.7 | Fog of war | Every training pool has `fog_share = 1.0` (all environments fogged). | `contracts/pools.py:49`; `runtime/train.py:83` | yes | n/a | 4.1 |

### 1b. Per-ship feature groups (the ship token, 607 inputs)

| # | Name | What it computes | Where | On? | Cost | Depends on |
|---|---|---|---|---|---|---|
| 1.8 | Ship state code as input | The 542-wide state code of the row's believed state (row 1.20), read as is (`position_inputs = "colour"`). | `models/yemong/inputs.py:438`; `state_code.py:416-426` | yes | R: input features and codes 3,111 per tick, 1,037 per call (whole token build) | 1.20, 4.1 |
| 1.9 | Team one-hot | `team_relative` as a 2-way one-hot. | `inputs.py:439` | yes | within 1.8's region | 1.1 |
| 1.10 | Ship-type one-hot | `ship_type` as a one-hot over the card's types (3). | `inputs.py:440` | yes | within 1.8's region | card |
| 1.11 | Flags | alive, visible, fired, benched as 4 raw 0/1 inputs. | `inputs.py:75, 441` | yes | within 1.8's region | 1.1 |
| 1.12 | Time since seen | symlog(belief age in seconds), 1 input. Age is the model's own count of ticks since the row was visible. | `inputs.py:431, 442` | yes | within 1.8's region | 4.1 |
| 1.13 | Respawn wait features | `respawn_wait` s as [w/16, sin(2 pi k w/16), cos(...)] for k = 1..4: 9 inputs (horizon 16 s, 4 harmonics). | `features.py:8-18`; `shots.py:176` | yes | within 1.8's region | 1.1 |
| 1.14 | Pending action | 42 inputs. Own rows: one-hot of the ship's previous joint action. Enemy rows: the model's stored forecast of the enemy's next joint action (probabilities, int16 fixed point /32767), or the one-hot of the null action (joint 0) when the ship has just spawned. | `inputs.py:412-417` | yes | within 1.8's region | 4.6, 6.1 |
| 1.15 | Refractive-index gradient | log-index gradient x, y divided by `GRADIENT_SCALE` = 1.875 x 2 ln 2 / 40 (about 0.065 per m); zero where not visible. 2 inputs. | `inputs.py:36, 419-420` | yes | within 1.8's region | 1.1 |
| 1.16 | Local presence | Per ship, ln(1 + sum of exp(-d^2 / (2 x 150^2))) over allies (not itself) and separately over enemies, the sum taken over rows with a valid belief. 2 inputs. Radius 150 m (`presence_radius`). Computed as an (N, N) pairwise table. | `features.py:21-39`; `inputs.py:421-426`; `model.py:255` | yes | within 1.8's region | 4.1 |
| 1.17 | Position of a row with no belief | A row with no valid belief is placed at the map centre (in the shifted frame) for its code and its rotary angles. | `inputs.py:405-409` | yes | n/a | 4.1, 1.26 |

### 1c. Encodings

| # | Name | What it computes | Where | On? | Cost | Depends on |
|---|---|---|---|---|---|---|
| 1.18 | Random position offset | Every position the model reads (ships, fields, zones, map centre) is shifted by a per-view offset drawn uniformly in [0, 236,196 m) per axis at each reset and held for the match. | `model.py:848-869, 402-404`; `inputs.py:173-194, 405` | yes | R: carry reset 210 per tick, 70 per call | 1.19 |
| 1.19 | Position colour code | Per axis a 3-colour circular code at 9 levels with periods 236,196 / 3^l m (l = 0..8; finest 36 m); the two axes' codes are combined as a 3 x 3 outer product per level: 9 groups of 9 = 81 values. Blurred toward uniform by the belief's position sigma (factor exp(-(2 pi sigma / p_l)^2 / 2) per level). | `codes.py:31-73, 178-190`; `state_code.py:74-75, 366-371`; `model.py:101` | yes | L: `circular_coordinates` 134 (7 calls), `circular_encode` 65 (5 calls) in forward and loss | 1.18 |
| 1.20 | Ship state code (542 wide, 23 softmax groups) | Concatenation of: position 9 x 9 (1.19); velocity 3 x 99 (1.21); heading 4 x 4 (1.22); nose rate 41 bins (1.23); five 21-bin scalars (1.24); gun cooldown 2 bins (1.25). Encodes means plus spreads (a belief's uncertainty) into probabilities per group. | `state_code.py:178-245, 357-406` | yes | L: `scalar_encode` 213 (3 calls), `match_moments` 177 (3 calls) in forward and loss | 1.19-1.25 |
| 1.21 | Velocity code | Velocity log-compressed radially (v x 100 ln(1 + speed/100) / speed, knee 100 m/s), projected on three axes at 0, 120, 240 degrees, each a 99-bin scalar code with spacing 5 (range +-245 compressed units; reach 1,059 m/s). The belief's 2 x 2 velocity covariance sets each axis's blur. | `codes.py:346-442`; `state_code.py:76-83, 144-146` | yes | within 1.20 | 1.20 |
| 1.22 | Heading code | 4-colour circular code at 4 levels, periods 2 pi / 4^l (l = 0..3): 16 values, blurred by the heading sigma. | `state_code.py:84-85, 375-377` | yes | within 1.20 | 1.20 |
| 1.23 | Nose-rate code | 41 bins over [-8, 8] rad/s; two-hot when sharp, a moment-matched sampled Gaussian when the belief has spread. | `state_code.py:86-87, 378-384`; `codes.py:286-324` | yes | within 1.20 | 1.20 |
| 1.24 | Scalar codes | 21 bins each for: angle of attack over +-0.541 rad (twice the widest stall angle); shield recharge wait over [0, 3] s; shield and energy as fractions of the type's maximum over [0, 1]; log refractive index over +-ln 2. Same sharp/blurred form as 1.23. | `state_code.py:88-91, 324-333, 387-394` | yes | within 1.20 | 1.20, card |
| 1.25 | Gun cooldown code | `cooldown_bins` = 1 + the most ticks between shots on the card = 2 bins, two-hot plus a stored residual distribution. | `state_code.py:172, 352-355, 395` | yes | within 1.20 | card |
| 1.26 | Rotary position bands | 8 bands; band k has period 339,625.1 / 3.405^k m (339,625; 99,743; 29,293; 8,603; 2,527; 742; 218; 64 m) and reads position along two orthogonal axes turned k x 35 degrees from world x/y. Angle = 2 pi (u mod p) / p with u the position's projection. 16 rotation pairs. Applied to attention queries and keys (row 3.3). | `model.py:94-100, 349-363`; `layers.py:18-46`; `runtime/train.py:220-223` | yes | within the spatial block's region | 1.18, 3.3 |
| 1.27 | Rotary heading harmonics | 3 further rotation pairs with angle = heading x {1, 2, 4}. Tokens without a heading (global, fields, zones) use heading 0. Total 19 pairs = 38 of each head's 64 dims rotated. | `runtime/train.py:224`; `model.py:338-347, 1054-1058`; `layers.py:49-56` | yes | within the spatial block's region | 3.3 |
| 1.28 | Arena code | ln(arena_radius / 7800) as an HL-Gauss code (sigma 0.75 bins): 21 real bins a quarter octave apart from 0.25x to 8x (1,950 m to 62,400 m) plus 2 padding bins per side = 25 inputs on the global token; values past the outermost padding centre are clamped. | `inputs.py:94-112, 474` | yes | within 1.8's region | 2.2 |
| 1.29 | Raw arena radius input | arena_radius / 2600 as one scalar on the global token (in addition to 1.28). | `inputs.py:37, 462` | yes | within 1.8's region | 2.2 |
| 1.30 | Timer code | Seconds as [t/16, 4 sin, 4 cos harmonics]: 9 inputs, used for the own team's next respawn wave (global), the two capture locks (zone) and the boundary's time to zero (global). | `inputs.py:83-91, 473, 476, 527-528` | yes (`frontline_timer_inputs`) | within 1.8's region | 2.2, 2.4 |
| 1.31 | symlog scalars | sign(x) ln(1 + |x|) applied to: time since seen, score target, zone offensive/defensive distances. | `critic.py:13`; `inputs.py:442, 467, 520-521` | yes | within 1.8's region | — |

---

## 2. Tokenisation and embeddings

| # | Name | What it computes | Where | On? | Cost | Depends on |
|---|---|---|---|---|---|---|
| 2.1 | Ship token | One per ship row (own and enemy, alive or dead, seen or not): 607 inputs = 542 state code + 2 team + 3 type + 4 flags + 1 time since seen + 9 respawn + 42 pending action + 2 gradient + 2 presence. A query token (it has a trunk state and outputs). | `inputs.py:142-152, 436-451` | yes | R: 1,037 per call (all tokens) | 1.8-1.17 |
| 2.2 | Global token | One per view, 136 inputs: 81 position code of the map centre; 8 scalars (episode progress, arena radius / 2600, fogged, own score / target, enemy score / target, symlog(target), front, front threshold); 3 mode flags (team deathmatch, frontline, elimination); 9 next-wave timer; 25 arena code; 1 boundary fraction; 9 boundary time to zero. A query token placed at the map centre with heading 0. | `inputs.py:115-128, 453-479` | yes | within 2.1's region | 1.5, 1.28-1.30 |
| 2.3 | Field token | One per field slot, 84 inputs: 81 position code of its centre, radius / 1000, band width / 40, level / 2. A memory token: a key and value of attention only, never a query, no trunk state, no outputs. | `inputs.py:38-40, 131, 481-498` | yes | within 2.1's region | 1.3 |
| 2.4 | Zone token | One per zone slot (5), 109 inputs: 81 position code, radius / 1000, 5-way role one-hot, progress, direction, symlog of the two distances, 2 x 9 capture-lock timers. A memory token. | `inputs.py:41, 133-139, 499-538` | yes | within 2.1's region | 1.4 |
| 2.5 | Absent-zone token | A zone slot with present = 0 is replaced after projection by one learned 64-wide vector (`absent_zone`, initialised N(0, 1)) placed at the map centre. In team deathmatch and elimination all 5 slots are this vector. | `network.py:58-63, 103-106`; `inputs.py:502-507` | yes | not in records | 2.4 |
| 2.6 | Kind projections | One first layer per token kind, gelu(RMSNorm(W x + b)) to 256 (ship 607 -> 256, global 136 -> 256, field 84 -> 256, zone 109 -> 256), then one shared Linear 256 -> 128 and a shared RMSNorm. No learned type, team, slot or index embedding: kind is carried by which first projection a token passes through; team and type are input one-hots. | `network.py:18-28, 31-107` | yes | R: input encoder 167 per tick, 56 per call. F: Encoder 18.72 M (ship.first 9.32, zone.first 12.18, field.first 9.81, global.first 8.39, shared 3.41) | 2.1-2.4 |
| 2.7 | Map-memory projection | Field and zone tokens pass a further bias-free Linear 128 -> 64 and RMSNorm(64): the map memory is 64 wide. | `network.py:56-57, 101` | yes | F: Encoder.map_memory 0.74 M | 2.3, 2.4, 3.2 |
| 2.8 | Token order and slicing | Query order: own ships (S), enemy ships (N - S), global. Heads slice by position: policy and value read rows [0, S); enemy-action and next-state read rows [S, N) at play and all N rows in the learner; outcome and density read row N. S is taken from the carry's `previous_action` width. | `model.py:1124, 1154-1167, 1271-1295` | yes | n/a | 5.x |
| 2.9 | Nothing indexed by slot | No parameter is indexed by ship slot, ship index, or count. Ships of a team are distinguished only by their features and positions. | `network.py`, `inputs.py` (whole) | n/a | n/a | — |
| 2.10 | Input-column initialisation | Code columns of each first projection are drawn N(0, 1/groups): ship state-code columns with 1/23, position-code columns of global/field/zone with 1/9, arena-code columns with 1. Other columns keep the Linear default (Kaiming uniform). | `model.py:670-701` | yes | n/a | 2.6 |

---

## 3. Trunk

| # | Name | What it computes | Where | On? | Cost | Depends on |
|---|---|---|---|---|---|---|
| 3.1 | Block layout | 2 blocks, width 128. Each block: 1 temporal sublayer, then 2 spatial sublayers (`order = "temporal_first"`). Total 2 temporal + 4 spatial sublayers. | `network.py:113-284`; `model.py:245-249`; `runtime/train.py:217` | yes | R: `trunk (other)` 886 inclusive per tick, 295 per call | 3.2-3.9 |
| 3.2 | Spatial sublayer | Pre-norm residual attention: the Q query tokens attend over themselves and the M + Z map-memory tokens (one softmax over Q + M + Z keys). Queries/keys/values of tokens from one bias-free Linear 128 -> 384; map keys/values from RMSNorm(64) then a bias-free Linear 64 -> 256 owned by each sublayer; output Linear 128 -> 128 without bias. 2 heads of width 64. Then a pre-norm residual gated MLP. | `spatial.py:18-107` | yes | R: 504 per tick, 42 per sublayer call. L: `SpatialBlock.forward` 152 ops (4 calls). F: SpatialBlock 119.26 M (map_norm 79.96, feedforward 63.44, query_key_value 12.98, map_key_value 5.90, output 4.33) | 2.6, 2.7, 3.3 |
| 3.3 | Rotary attention (relative position and heading) | Queries and token keys are rotated by their own token's angle table, map keys by theirs (rows 1.26, 1.27); the first 38 dims of each 64-wide head are rotated as 19 pairs, the other 26 pass through. Values are not rotated. There is no additive relative-position bias. | `spatial.py:90-107`; `layers.py:49-56` | yes | within 3.2 | 1.26, 1.27 |
| 3.4 | Attention masking | In training no key is masked (`key_mask` is None): dead ships, never-seen enemies (placed at the map centre) and absent zones are all keys. Two runtime switches exist (`mask_benched`, `mask_absent_fields`), both False in training. | `model.py:575, 582`; `inputs.py:421-430`; `runtime/train.py:408-409` | no mask in training | n/a | 3.2, 12 |
| 3.5 | Neighbour limits | None. Attention is dense over all Q + M + Z keys of the view; no nearest-k or distance cutoff. | `spatial.py:87-107` | n/a | n/a | 3.2 |
| 3.6 | Fused attention kernel | On CUDA the rotation and attention run as one Triton program per (view, head), with queries padded to max(16, next power of 2 of Q) and keys to max(16, next power of 2 of Q + M + Z), the whole key set in one tile. The plain path (`scaled_dot_product_attention`) computes the same function elsewhere. | `kernels/attention.py:1-25, 255-266, 321-328` | yes (CUDA) | within 3.2 | 3.2 |
| 3.7 | Gated MLP | down(gelu(gate x) * up x), hidden 512, no biases; gate and up share one 128 -> 1024 matrix. Used after every spatial and temporal sublayer. | `layers.py:59-73` | yes | F: GatedMLP 31.72 M (temporal) plus SpatialBlock.feedforward 63.44 M | 3.2, 3.8 |
| 3.8 | Temporal sublayer | Per query token, over time: RMSNorm -> bias-free Linear 128 -> 128 -> causal depthwise convolution (4 taps: the current input and the 3 before) -> RG-LRU -> multiplied by gelu(Linear(normed)) -> bias-free output Linear -> residual; then pre-norm residual gated MLP. No mixing across tokens. | `temporal.py:145-202` | yes | R: 376 per tick, 63 per sublayer call. L: `TemporalBlock.sequence` and `CausalConvolution.sequence` appear only in the bytes table (F file: 309.9 and 317.6 KB per view-tick) | 3.9 |
| 3.9 | RG-LRU | h_t = a_t h_(t-1) + sqrt(1 - a_t^2) (i_t * x_t), with [r_t, i_t] = sigmoid(Linear 128 -> 256 (x_t)) and a_t = exp(8 r_t logsigmoid(Lambda)). Lambda (128 learned logits) starts as linspace(0, 4). The square-root factor is floored at sqrt(1e-6). | `temporal.py:70-105, 21-23` | yes | within 3.8 | 3.8 |
| 3.10 | Trunk carry | Per view: RG-LRU state (2 blocks, Q, 128) and convolution buffer (2 blocks, Q, 3, 128). Zeroed at an environment reset. The segment scan cuts the recurrence and the convolution taps at episode starts inside a segment. | `model.py:372-396, 799-846, 848-889`; `temporal.py:96-105, 126-142` | yes | R: carry reset 70 per call | 3.8 |
| 3.11 | Normalisation | RMSNorm with learned scale everywhere (pre-norm in each sublayer, after each kind projection, after the shared projection, in the map memory, inside each head). No LayerNorm, no BatchNorm. | `layers.py:13-15` | yes | F: RMSNorm 0.49 M | — |
| 3.12 | Residual scheme | Every sublayer adds to the token stream (x + attention, x + MLP, x + recurrent output). There is no final trunk norm before the heads; each head group starts with its own Linear then RMSNorm. | `spatial.py:83-84`; `temporal.py:160-163` | yes | n/a | 5.1 |
| 3.13 | Segment evaluation | The learner recomputes a stored segment with `unroll`: encoder and spatial sublayers over all ticks at once, temporal sublayers as a scan (chunks of 8 steps, or a Triton scan on CUDA). The belief is not replayed: its values come from the stored replay record. | `model.py:1263-1310`; `temporal.py:32-67` | yes | L: forward and loss 4,796 per micro-batch | 4.7 |
| 3.14 | Default layer initialisation | `nn.Linear` layers: Kaiming-uniform weights (a = sqrt 5), bias uniform in +-1/sqrt(fan in); causal convolution uniform +-1/2; every draw keyed by seed and parameter path. | `models/initialise.py:30-73` | yes | n/a | — |

---

## 4. Belief and forecast machinery

| # | Name | What it computes | Where | On? | Cost | Depends on |
|---|---|---|---|---|---|---|
| 4.1 | Belief state | Per view and ship row: valid (bool), age (ticks since seen), 12 means (x, y, velocity x/y, heading, nose rate, angle of attack, recharge wait, shield, energy, gun cooldown, log index) in physical units, 34 spreads (11 sigmas/variances + 2 cooldown residuals + 21 entries of a 6 x 6 covariance of x, y, vx, vy, heading, nose rate), and a 42-way action forecast. Kept in the model's carry. | `belief.py:16-47`; `state_code.py:22-55, 276-291` | yes | — | 4.2-4.6 |
| 4.2 | Assimilating sight | Before each decision: a visible row takes the observed truth with zero spread and age 0; a hidden row that has been seen since its last spawn keeps its stored forecast and its age rises by 1; a row never seen since spawning is invalid (zeros, uniform action). Own rows are always visible. | `belief.py:50-62`; `model.py:1127` | yes | R: belief update from sightings 51 per tick, 17 per call | 4.1 |
| 4.3 | Extrapolation baseline | The forecast's starting point for t + 1, computed without the network from the row's state and its last-tick rates: position along the arc of last tick's turn at mean speed; velocity plus last tick's tangential/normal acceleration carried into the current velocity frame; heading + nose rate x dt + half nose acceleration x dt^2; nose rate unchanged; angle of attack and energy repeat last tick's change; shield, recharge wait and cooldown follow last tick's change or a quiet-tick rule (recharge 24 per s after the wait; countdowns fall by dt); log index + gradient . velocity x dt where the gradient is known. | `delta.py:915-1007` | yes (`delta_baseline = "extrapolation"`) | L: `DeltaCode.baseline` 107 (1 call) in label codes and again in forward and loss | 4.4 |
| 4.4 | Rates and held rates | Per row each tick: acceleration (world), nose acceleration, the 12-quantity change since the previous tick, gradient, shot flag. A visible row reads the observation's acceleration columns; a hidden row reads the rates it held at its last sighting, multiplied by exp(-dt / 0.25 s) per tick (`extrapolation_decay`), and its shield/countdowns take the quiet-tick rule. The carry holds `previous` (N, 12), `held` (N, 16), `held_valid`, and the enemy rows' last `Motion`. | `model.py:904-962, 372-396`; `inputs.py:197-293` | yes | L: `Yemong.rates` 704 (32 calls) in forward and loss, 726 (33 calls) in label codes; `segment_rates` 138 | 4.3 |
| 4.5 | Physics forecast (delta decode) | The physics head's 500 logits (row 5.4) are decoded per component to a mean and variance of the change from the baseline (median for recharge wait, shield, cooldown); position and velocity changes are rotated from the row's velocity frame (heading frame below 1 m/s) to world and added to the baseline. Covariance: F P F^T + Q with position += velocity x dt, heading variance capped at pi^2 / 3 and nose-rate variance at 8^2 / 3. | `delta.py:1084-1124, 1178-1301, 730-753` | yes | R: delta-code decode 1,519 per tick, 506 per call | 4.3, 5.4 |
| 4.6 | Error-head correction (hidden rows) | For rows whose belief is a forecast (age > 0), the error head's 342 logits (row 5.5) are decoded per component to a median m (window of 2 bins either side) and a quantile-based sigma (16th/50th/84th percentiles). The belief mean moves by eta x g x m with eta = 0.1 and g = |m|^2 / (|m|^2 + sigma^2), capped by r tanh(. / r) at the per-quantity one-tick bounds (position 7.99 m, velocity 40.6 m/s, heading 0.311 rad, nose rate 10.2 rad/s, angle of attack 0.163 rad, energy 0.0457, log index 0.232); recharge wait, shield and cooldown move by m itself. The row's spread becomes sigma^2 + (m - step)^2, replacing the physics covariance. Visible rows take the physics forecast unchanged. | `correction.py:96-112, 517-555, 658-700, 741-842`; `model.py:729-776` | yes (`delta_split`) | R: correction-code decode 9,068 per tick (33.6% of the tick), 3,023 per call | 4.5, 5.5 |
| 4.7 | Forecast feedback and what re-enters the inputs | After each decision the forecast (means, spreads) and softmax of the enemy-action logits are stored as the enemy rows' belief. At the next tick, for rows still hidden, these feed back into: the 542-wide state code (blurred by the spreads), the rotary position and heading of the token, local presence, the pending-action inputs, and time since seen. Only enemy rows are fed back (own rows are truth). Non-finite forecast entries are replaced by 0 and counted. | `model.py:1188-1231`; `belief.py:65-81`; `inputs.py:404-417` | yes (`forecast_feedback`) | R: belief forecast 10,626 inclusive per tick, 1,771 per call | 4.1, 2.1 |
| 4.8 | Replay record | Each decision stores what it read from the carry for the enemy rows (valid, age, means, spreads, action forecast as int16, previous own actions, offset), so the learner's `unroll` rebuilds the same tokens without replaying the belief. No gradient flows through the belief from one tick to the next. | `inputs.py:155-170, 346-357`; `model.py:1128-1136` | yes | not in records | 3.13 |
| 4.9 | Resets | At an environment reset: trunk state, convolution buffer, belief, previous actions, held rates are cleared and a new position offset is drawn. A ship's belief is also invalidated when it spawns unseen. | `model.py:848-889`; `belief.py:54` | yes | R: 70 per call | 3.10, 1.18 |

---

## 5. Output heads

All heads are MLPs of the form Linear(128 -> 256) -> RMSNorm -> GELU -> Linear(256 -> outputs) unless noted; heads reading the same token share one first matrix (`HeadGroup`). First layers orthogonal with gain sqrt 2, last layers orthogonal with gain 0.01, biases zero, except where a row says otherwise. R: output heads 18 ops per tick (about 1 per group call); F: HeadGroup 26.04 M per view-tick.

| # | Name | What it predicts and its form | Where | On? | Cost | Depends on |
|---|---|---|---|---|---|---|
| 5.1 | Policy head | Per own ship: 42 logits over the joint action (3 power x 7 stick x 2 shoot). | `model.py:633, 1154, 1285`; `heads.py:20-62` | yes | within HeadGroup | 6.1 |
| 5.2 | Value head (per-ship levels) | Per own ship: 4 levels (damage, kill_death, objective_progress, objective) x 77 bins = 308 logits. Bins are symexp(linspace(-3, 3, 77)): -19.09 to +19.09 in reward units, spacing 0.079 in symlog, one bin at 0. Decoded as the softmax expectation over bin values. Shares its first matrix with 5.1. | `model.py:250-253, 633, 1077-1087`; `critic.py:23-41`; `heads.py:71-93` | yes | R: value decode 18 per tick, 6 per call | 7.5 |
| 5.3 | Outcome head | On the global token: 4 logits (win, tie, loss, unresolved). The outcome level's value is P(win) - P(loss), the same number for every own ship; the learner multiplies it by the outcome reward weight (10) to put it in reward units. | `model.py:635-637, 1070-1075`; `critic.py:9-10, 44-46`; `learn/ppo.py:427-434` | yes | within HeadGroup | 7.6 |
| 5.4 | Next-state (physics) head | Per ship row: 500 logits = 12 components as symlog HL-Gauss codes of the change from the extrapolation baseline: position along/across and velocity along/across at 51 real bins (+4 padding = 55 each), the other eight (heading, nose rate, angle of attack, recharge wait, shield, energy, log index, cooldown) at 31 real bins (35 each). Knees and bounds per component in `delta.EXTRAPOLATION` (e.g. position along knee 2.62e-4 m, bound 0.3 m; velocity along 0.0188, 26 m/s). Last weight starts at zero, bias at a measured marginal. Shares its first matrix with 5.6. | `delta.py:42-43, 109-122, 807-893, 1134-1160`; `model.py:634, 702-704` | yes | within HeadGroup; decode in 4.5 | 4.3, 4.5 |
| 5.5 | Error (correction) head | Per ship row: 342 logits = 14 symlog HL-Gauss codes of truth(t + 1) minus the physics forecast: position on 3 world axes at 0/60/120 degrees (31 real bins each, bound +-26,244 m, knee 11.9 m), velocity on the same 3 axes (23 bins, bound +-2,118 m/s), heading (13), nose rate (15), angle of attack (15), recharge wait (11), shield (13), energy (19), log index (35), cooldown (3); each plus 4 padding bins. Bin count per code = 2 round(2 ln(1 + bound/knee)) + 1. Last weight starts at zero, bias at a measured marginal. | `correction.py:73-131, 427-460, 557-594, 702-727`; `model.py:662-666, 705-707` | yes | within HeadGroup; decode in 4.6 | 4.6 |
| 5.6 | Enemy-action head | Per ship row: 42 logits over that ship's joint action at t. At play only enemy rows are computed; softmax is stored as the action belief. | `model.py:634, 1155, 1287` | yes | within HeadGroup | 4.7, 1.14 |
| 5.7 | Density head | On the global token: 2 x 331 = 662 log-rates, one per cell of a hex grid of 10 rings (331 cells) for allies and for enemies; the grid spans the arena radius (cell size = arena radius / 10). Log-rates are clipped to [-20, 2]. Output bias starts at ln(max(num_ships, 2) / 2 / 331) = -4.19 for the build's 10 ships. Training only (not computed in `step`). | `density.py:13-31, 34-87, 90-100`; `model.py:254, 638-644, 708` | yes | L: `HexGrid.counts` 60 (label codes) | 8.7 |
| 5.8 | Shot-outcome head | Per ship row: 19 logits = [enemy hit, ally hit, 8 enemy near-miss bins, 8 ally near-miss bins, clear] for the bullet the ship would fire this tick. Near-miss bins are log-distance from the collision radius (12 m) over a span of 6 radii, smoothed with sigma 0.75 bins. Training only. | `shots.py:17-60, 145-173`; `model.py:640` | yes | L: `outcome_code` 24 (label codes) | 8.8 |
| 5.9 | Shot-trajectory head | Per ship row and queried bullet age: 154 logits = displacement 8 levels x 9 (3-colour circular code per axis, outer product, period 5,080 m) + direction 5 levels x 4 + speed 41 bins over [0, 5,542 m/s] + log index 21 bins. Form: latent Linear(128 -> 128) computed once per row plus a bias-free Linear of 9 age features (age / 11 and 4 harmonics), then norm, GELU, Linear, norm, GELU, Linear. Ages are whole moves 1..11. Training only. | `heads.py:96-141`; `shots.py:17-74`; `model.py:84, 641-643, 1327-1340` | yes | not itemised in records | 8.8 |
| 5.10 | Geometry head | For pairs (own ship anchor, partner row): 241 logits = range (48 real bins + 4, symlog knee 10 m, bound 64,000 m, one-sided), three angles (antenna train, aspect, heading crossing) at 48 circular bins each, closure rate (41 real bins + 4, knee 10, bound +-2,000 m/s). Form: Linear(128 -> 256) per row split into an anchor half and a partner half, summed per pair, then norm, GELU, Linear, norm, GELU, Linear. Each anchor has K = min(8, N - 1) partners per tick, cycling by 8 rows per tick. Training only. | `geometry.py:42-71, 125-135, 159-196`; `model.py:667, 1312-1325` | yes (`geometry_head`) | L: `geometry.quantities` 27 (label codes), 648 in prepare (24 calls); F bytes: `GeometryHead.query` 1,324.7 KB per view-tick | 8.9 |
| 5.11 | Head evaluation at play | `step` computes only policy, value, outcome on own/global tokens and enemy-action, next-state, correction on enemy rows. Density, shot and geometry heads run only in `unroll` (the learner) or `inspect`. | `model.py:1143-1167` | yes | — | — |

---

## 6. Action space and sampling

| # | Name | What it computes | Where | On? | Cost | Depends on |
|---|---|---|---|---|---|---|
| 6.1 | Action space | Per ship: power (3: coast, boost, reverse) x stick (7: straight, three levels each side) x shoot (2) = 42 joint actions; joint index = power x 14 + stick x 2 + shoot. | `contracts/action.py:9-47` | yes | n/a | 5.1 |
| 6.2 | Distribution | One 42-way categorical per own ship over the joint action (not three independent factor distributions). Ships are sampled independently given their logits; there is no joint distribution across ships. | `models/yemong/policy.py:11-33, 45-49` | yes | R: action sampling 216 per tick, 36 per call | 5.1 |
| 6.3 | Sampling method | Inverse CDF with three keyed uniforms per ship: power from its marginal, stick given power, shoot given both. Uniforms are a function of the decision key (seed, environment, episode, clock, role). | `policy.py:11-33`; `controllers/yemong.py:68-71` | yes | within 6.2 | — |
| 6.4 | Masking | None. No action is masked for any ship state (dead ships still get logits; the loss masks them by `alive`). | `policy.py`; `learn/ppo.py:905` | n/a | n/a | 8.1 |
| 6.5 | Temperature | The learner samples at temperature 1 (the rollout passes none). A per-view temperature argument exists on the controller. | `controllers/yemong.py:53-67`; `runtime/rollout.py:440-442` | off for the learner | n/a | — |
| 6.6 | Action repeat | None found: the controller decides every tick (30 per s) and the previous joint action is an input (1.14). | `runtime/rollout.py:431-460`; grep for repeat/skip found nothing | n/a | n/a | — |

---

## 7. Rewards and returns

| # | Name | What it computes | Where | On? | Cost | Depends on |
|---|---|---|---|---|---|---|
| 7.1 | Reward levels | Five per-ship reward streams, each weighted: damage 2.74, kill_death 2.83, objective_progress 20, objective 20, outcome 10. Stored per own ship per tick as (L, R, S, 5). | `learn/rewards.py:43-76`; `contracts/trajectory.py:13` | yes | R: rewards 159 per tick | 7.2-7.4 |
| 7.2 | Damage level events | Six events in units of 100 shield points: damage taken (negative), damage dealt to enemies, friendly fire (negative), recharge (refund to the target), charge-back (to each attacker pro rata as the target heals), unforced (boundary damage and twice any friendly fire, paid to each ship of the opposing team as that team's total divided by the receiving team's size). A (B, N, N) ledger tracks unhealed damage per attacker-target pair. | `learn/rewards.py:80-150, 152-157, 198-236, 274-360` | yes | within 7.1 | 12 |
| 7.3 | Kill/death level events | Four events: death (-1 to the ship), kill (the dead ship's unit split half by the final tick's raw hits and half by outstanding ledger damage, to enemy shooters), ally kill (negative to the allied shooter), unforced (boundary deaths and twice ally-held shares, paid to the opposing team divided by its size). | `learn/rewards.py:169-195` | yes | within 7.1 | 12 |
| 7.4 | Objective and outcome events | objective_progress and objective: the mode's per-ship signed amounts from the game's events (each summing to zero over the environment). Outcome: on the tick the match ends, +result to every ship of team 0 and -result to team 1 (dead ships included), result = win weight minus loss weight; in team deathmatch the result is one-hot by the sign of the kill lead (`deathmatch_outcome = "kill_lead"`). | `learn/rewards.py:160-166, 322-331`; `contracts/events.py:33-41` | yes | within 7.1 | — |
| 7.5 | Per-level GAE | Advantages and returns per level and per own ship with level-specific discount and trace decay. RL: gamma = (0.982, 0.990, 0.995, 0.999, 0.9997), lambda = (0.81, 0.90, 0.95, 0.97, 0.97) per tick. BC stage: gamma = (0.982, 0.990, 0.995, 0.999, 1.0), lambda = (0.98, 0.98, 0.99, 0.99, 1.0). An episode end (timeouts included) cuts the trace and the bootstrap. | `learn/returns.py:27-54`; `learn/ppo.py:145-146, 436-452`; `learn/bc.py:49-50` | yes | L: `per_level_gae` 1,168 (2 calls) in prepare | 5.2, 5.3 |
| 7.6 | Outcome targets | The outcome head's target is a 4-way distribution built backwards: target_t = gamma ((1 - lambda) p(s_(t+1)) + lambda target_(t+1)) + (1 - gamma) x unresolved, replaced by the final result's distribution where the episode ended; gamma, lambda are the outcome level's. | `critic.py:49-84`; `learn/ppo.py:474-483` | yes | L: `outcome_targets` 451 in prepare | 5.3 |
| 7.7 | Lookahead and bootstrap | Each update plays 4 shards of 128 ticks, then a further 128 lookahead ticks that are never trained on; GAE runs over all 640 ticks and is bootstrapped from the learner's values on the observation after the last one. The trained ticks keep their advantages and returns. | `runtime/train.py:208-210, 576-594`; `learn/ppo.py:1016-1032`; `runtime/rollout.py:755-787` | yes | within prepare | 7.5 |
| 7.8 | Level sum and team mixing | The policy's advantage per ship is the sum over the five levels of that ship's level advantage, plus team_spirit x the mean over living teammates; team_spirit = 0 for every level, so the sum is the ship's own. The (S, S) teammate-weight matrix is still built. | `learn/returns.py:57-93`; `learn/ppo.py:147, 453-457` | sum on; mixing coefficient 0 | within prepare | 7.9 |
| 7.9 | Return scale | One scalar for the whole batch: the gap between exponential moving averages (decay 0.99 per update) of the 5th and 95th percentiles of the per-ship summed return (all levels added) over living own ship-steps of the update, floored at 1.0. The EMAs start at 0 with no bias correction. Carried from BC to RL. | `critic.py:87-110`; `learn/ppo.py:148-149, 1052`; `runs/checkpoint.py:645-657` | yes | L: `Learner.prepare` 15,189 exclusive | 7.8, 8.1 |
| 7.10 | Advantage normalisation | The summed advantage is divided by the return scale (7.9). No mean is subtracted and no per-batch or per-minibatch standard deviation is applied. | `learn/losses.py:28-43` | yes | within forward and loss | 7.9 |
| 7.11 | Critic targets | Per-ship levels: the level's GAE return (advantage + rollout value), in reward units, clamped to the bin range and two-hot coded. Not divided by the return scale; no per-level normalisation or bias table. | `learn/losses.py:98-126`; `critic.py:28-35` | yes | within forward and loss | 5.2 |
| 7.12 | Valued mask | The critic trains on own ships that are alive or waiting to respawn (`respawn_wait > 0`); the policy trains on own ships alive. | `runtime/rollout.py:457-460`; `learn/ppo.py:61-64` | yes | n/a | 8.1, 8.3 |

---

## 8. Losses and their balancing

Total loss = sum over terms of coefficient x term. Each term is a masked sum over its tokens divided by that token count summed over the whole minibatch (so micro-batches add up to the minibatch). Coefficients are fixed numbers in `LearnerConfig`; there is no automatic balancing, uncertainty weighting, gradient normalisation between terms or schedule on any coefficient. `learn/ppo.py:154-170, 634`.

| # | Name | Form | Weight | Where | On? | Cost | Depends on |
|---|---|---|---|---|---|---|---|
| 8.1 | Policy surrogate (RL) | max(-A r, -A clip(r, 0.8, 1.2)) per living own ship, A = advantage / return scale, r = exp(log pi - log pi_old) with the rollout's log-probability. Mean over living own ship-steps. | 27.0 | `learn/losses.py:28-43`; `learn/ppo.py:893-957` | RL stage | within forward and loss | 7.8-7.10 |
| 8.2 | Entropy terms | Mean of three chain-rule parts of the joint entropy: H(power), H(stick | power), H(shoot | power, stick), one coefficient each. | `LearnerConfig` default -0.0081 each; a warm start from a policy file replaces every one not set on the command line by 0.0 (`WARM_START_ENTROPY`), and launch.toml's `rl_set` is empty, so the RL stage runs at 0.0. BC: 0.0. | `models/yemong/policy.py:52-64`; `learn/ppo.py:47-48, 155-157, 207-212`; `cli.py:931-941, 1036-1038` | computed; coefficient 0 in both stages | within forward and loss | 6.2 |
| 8.3 | Value loss (per-ship levels) | Cross-entropy of the 77-bin logits against the two-hot return, summed over the 4 per-ship levels, mean over valued own ship-steps. | 0.007 (one coefficient on the sum of all five levels' losses) | `learn/losses.py:98-126`; `learn/ppo.py:158, 690-702` | yes | within forward and loss | 7.11 |
| 8.4 | Outcome loss | Cross-entropy of the 4 outcome logits against the outcome targets, mean over view-steps. Added into the value term under the same 0.007. | (in 8.3) | `learn/losses.py:123-126` | yes | within forward and loss | 7.6 |
| 8.5 | Next-state (physics) loss | Grouped cross-entropy of the 500 logits against the HL-Gauss code of the true change from the baseline, summed over the 12 groups. Rows: own ships and enemies visible at t with weight 1; hidden enemies with weight exp(-age / 0.25 s) (`physics_hidden_decay`), their target computed from true rates; the segment's first tick excluded for hidden rows. Requires alive at t and t + 1 and a valid belief. Divided by the count of all next-state rows (seen and hidden). | 0.0033 | `learn/ppo.py:161, 262-275, 704-708`; `learn/labels.py:95-171`; `codes.py:486-526` | yes | L: `grouped_cross_entropy` 257 (4 calls, all coded heads) | 5.4, 4.3 |
| 8.6 | Correction (error-head) loss | Grouped cross-entropy of the 342 logits against the code of truth(t + 1) minus the detached physics forecast, on enemies whose belief at t is a forecast. Same denominator as 8.5. The physics head receives no gradient from this term. | 0.0033 | `learn/ppo.py:164, 709-717`; `learn/labels.py:174-200` | yes | L: `DeltaCode.advance` 63 in forward and loss (the detached forecast) | 5.5 |
| 8.7 | Enemy-action loss | -log softmax at the enemy's true joint action, mean over living enemy rows (visible or not: the label is privileged truth). | 0.11 | `learn/ppo.py:165, 718-724`; `learn/labels.py:203-206` | yes | within forward and loss | 5.6 |
| 8.8 | Density loss | Poisson deviance per cell (exp(log rate) - count x log rate, minus its saturated value) summed over the 662 cells, divided by the number of valued own ships at that step (at least 1), mean over view-steps. Targets: each living ship (privileged truth, both sides) spreads one unit over the 3 nearest hex cells by barycentric weight. | 0.021 | `learn/ppo.py:166, 725-730`; `density.py:46-100`; `learn/labels.py:209-218` | yes | L: `HexGrid.counts` 60 | 5.7 |
| 8.9 | Shot-trajectory loss | Grouped cross-entropy of the 154 logits against the sharp code of the re-simulated bullet's displacement, direction, speed and index at 4 randomly drawn ages per shot, mean over the 4 ages, mean over valid shots of own ships and of enemies visible at t. | 0.007 | `learn/ppo.py:167, 731-743`; `learn/losses.py:129-144`; `learn/shots.py:44` | yes | prepare: `learn.shots.tick` 311,664 ops per update (1,032 calls), the largest single region in L | 5.9, 8.11 |
| 8.10 | Shot-outcome loss | Cross-entropy of the 19 logits against the outcome code of the re-simulated shot, same mask and denominator as 8.9. | 0.019 | `learn/ppo.py:168, 744-746` | yes | within forward and loss | 5.8, 8.11 |
| 8.11 | Shot re-simulation labels | Once per update, for every tick and every living ship row, the learner re-simulates the bullet that ship would have fired with the game's own launch, flight and hit code over its 11-move life against the stored ship motion (including ticks after the segment), with a random spread draw. | n/a | `learn/shots.py:106-302`; `learn/ppo.py:501-542, 1056-1100` | yes | L: 311,664 of prepare's 353,330 ops | 8.9, 8.10 |
| 8.12 | Geometry loss | Grouped cross-entropy over the 5 targets (range, 3 angles, closure), mean over the 5 targets, mean over pairs whose anchor is alive and whose partner is alive and visible. | 0.007 | `learn/ppo.py:170, 747-754`; `learn/losses.py:18-25`; `geometry.py:138-156` | yes | L: `wrapped_hl_gauss` 51 (3 calls) | 5.10 |
| 8.13 | Label smoothing | None on the policy, value or outcome targets. The coded auxiliary targets are HL-Gauss (sigma 0.75 bins) for symlog codes and geometry angles, two-hot for the value and for sharp state/shot codes. | `codes.py:550-551, 594-614` | n/a | n/a | — |
| 8.14 | Grouped cross-entropy | One custom autograd function computes, per softmax group of a code vector, lse(z) x sum(target) - sum(target x z); it keeps its gradient factor in bfloat16 for the backward. Used by the next-state, correction, shot-trajectory and geometry terms. | `codes.py:482-526` | yes | L: 257 ops (4 calls) | 8.5, 8.6, 8.9, 8.12 |
| 8.15 | Non-finite guard | A minibatch whose gradient norm is not finite leaves parameters and Adam state unchanged and is counted. | `learn/ppo.py:1289-1327, 1387-1390` | yes | L: `Learner._step` 1,524 | 9.5 |

---

## 9. PPO and optimisation

| # | Name | What it does | Where | On? | Cost | Depends on |
|---|---|---|---|---|---|---|
| 9.1 | Trust region | PPO's symmetric ratio clip at 1 +- 0.2 (`trust_region = "clip"`). | `learn/ppo.py:150-151, 854-868` | yes | n/a | 8.1 |
| 9.2 | Old log-probabilities | The rollout's (`old_log_probabilities = "rollout"`), not recomputed by `unroll`. A step/unroll ratio drift is measured on 48 views as a diagnostic. | `learn/ppo.py:179, 1049-1050, 1683-1706` | yes | n/a | 3.13 |
| 9.3 | Batch per update | 2048 environments x 128 ticks x 4 shards = 1,048,576 environment steps. Learner views: self-play half contributes both seats, league half one seat, so 3,072 views per shard and 12,288 stored views per update, each a 128-tick segment of one view (N = 10 rows). A further 1,024 lane environments run in the same rollout and are not stored for the learner (RL stage). | `runtime/train.py:206-212, 83, 571-574`; `runtime/lanes.py:112-123`; launch.toml | yes | — | 11.1 |
| 9.4 | Epochs and minibatches | RL: 4 epochs x 32 minibatches (384 views each, a fresh random permutation per epoch; a view keeps its whole 128-tick segment). BC: 4 epochs x 256 minibatches (48 views each). | `learn/ppo.py:175-176, 1374-1420`; `learn/bc.py:44, 60` | yes | L: clip and optimizer 2,808 per minibatch | 9.6 |
| 9.5 | Gradient clipping | Global norm clipped to 1.0 per minibatch. | `learn/ppo.py:174, 1387` | yes | within 9.4 | — |
| 9.6 | Micro-batches | A minibatch's gradient is accumulated over micro-batches of `micro_batch_rows // num_ships` views; each term is divided by the minibatch-total denominator. Default 160 rows (16 views); launch.toml's probe lists [640, 320] rows for RL and [2560, 1280] for BC and uses the largest that fits. | `learn/ppo.py:177, 301-316, 961-974, 1248-1287`; launch.toml `[probe]` | yes | L: forward and loss 4,796 + label codes 2,338 + backward 782 per micro-batch | 12 |
| 9.7 | KL early stop | After each epoch but the last, if the epoch's mean approximate KL ((r - 1) - log r over living own ships) exceeds 0.02 the remaining epochs are skipped. RL only (BC: `target_kl = None`). | `learn/ppo.py:178, 915, 1413-1420` | RL stage | n/a | 8.1 |
| 9.8 | Optimiser | Adam (`torch.optim.Adam` subclass), fused on CUDA, with torch's defaults for betas (0.9, 0.999), eps (1e-8) and weight decay (0): only `lr` is passed. One parameter group for the whole model. | `learn/ppo.py:97-109, 383-385` | yes | L: `Adam.step` 1,272 | — |
| 9.9 | Learning-rate schedule | Linear warm-up from 1e-7 to 3e-4 over the first 10,000,000 environment steps of the stage, then constant. No decay. Each stage (BC, RL) starts its own warm-up with a fresh optimiser. | `learn/ppo.py:171-173, 246-249, 1350-1352` | yes | n/a | 10.6 |
| 9.10 | Weight decay | None. | `learn/ppo.py:383-385` | n/a | n/a | — |
| 9.11 | Precision | `precision = "bfloat16"`: the encoder's, trunk's and heads' matrix products run under bf16 autocast; logits are cast back to float32 and every loss reduction, the belief arithmetic, GAE and the optimiser are float32. | `runtime/train.py:243, 328-336`; `model.py:997-998, 1168-1176` | yes | n/a | — |
| 9.12 | Compilation | The micro-batch loss (unroll + loss) is compiled whole with `torch.compile(fullgraph=True, dynamic=False)`, one compile per micro-batch shape. | `runtime/train.py:360-371`; `learn/ppo.py:399-414` | yes | n/a | 9.6 |
| 9.13 | Averaged policies | Two averages of the live weights are folded after every update: an exponential average (decay 0.95 per update) and a uniform average that starts after the warm-up (10M steps; before that it equals the live weights). Both are saved as policy files, are league pool entries, and are rated in the lanes. The learner itself trains only the live weights. | `league/pool.py:125-175`; `runtime/opponents.py:171-179, 208-211` | yes | not in records | 10.6, 11.2 |
| 9.14 | Diagnostics that do not change the update | Gradient decomposition per term, per-level advantage series, head statistics, representation rank: measured on a cadence, eager. | `learn/gradients.py`, `learn/advantages.py`, `learn/head_statistics.py` | yes (measurement only) | L: heads' sums 10,916 (cadence only) | — |

---

## 10. BC stage

| # | Name | What it does | Where | On? | Cost | Depends on |
|---|---|---|---|---|---|---|
| 10.1 | Teacher | Imugi, the scripted controller, at temperature 1.0 with uniform floor 0.0, giving a 42-way distribution per living own ship. It is run over the stored observations once per update in time order with its own carry. | `learn/bc.py:72-103, 162-184, 230-261` | BC stage | not in records (v2 BC records were not among the files) | — |
| 10.2 | BC policy loss | Cross-entropy of the student's 42 logits against the teacher's distribution, mean over living own ship-steps, coefficient 0.025. Replaces the surrogate; no advantage is read. | `learn/bc.py:53-69, 263-315` | BC stage | not in records | 5.1 |
| 10.3 | Everything else in BC | The critic trains on the student's own returns (BC's gammas and lambdas, row 7.5), the return scale is updated, and every auxiliary loss runs with the same coefficients as RL. No entropy term, no KL stop, 256 minibatches. | `learn/bc.py:1-11, 53-69` | BC stage | — | 8.3-8.12 |
| 10.4 | BC opponents | Half the environments self-play (student vs student), half student vs the teacher (`TeacherOpponent`, playing the labels' distribution). Launch sets `lanes.environments = 0` for BC (no rating lanes). | `runtime/cloning.py:28-35`; launch.toml `bc_set` | BC stage | — | — |
| 10.5 | Stop rule | First of: budget (launch: 100,000,000 environment steps); score target (None: off); plateau of the BC KL (5% over 10M steps, held 10M; computed and logged, `plateau_stops = False` so it stops nothing). | `learn/bc.py:381-416, 585-663`; launch.toml | BC stage | — | — |
| 10.6 | Hand-over to RL | RL starts from the BC run's exponential-average weights (decay 0.95 per update) and its return scale, with a fresh Adam, fresh step counters, a new LR warm-up, entropy coefficients 0.0, and every league slot filled with the same weights. Combined budget 150M steps (BC 100M, RL the rest). | `runs/session.py:46`; `runs/checkpoint.py:645-657`; `cli.py:1036-1038, 1246-1265`; launch.toml | yes | — | 9.9, 9.13 |

---

## 11. League and self-play elements touching the learner or model inputs

| # | Name | What it does | Where | On? | Cost | Depends on |
|---|---|---|---|---|---|---|
| 11.1 | Pools and mode mix | Per mode: 50% self-play (both seats the learner), 50% league seat. Mode shares in code: frontline 0.375, deathmatch 0.3125, elimination 0.3125 (768 / 640 / 640 of 2048). The seat (which team the learner plays) is redrawn per environment at every reset. | `runtime/train.py:73-83, 89-118` | yes | — | 9.3 |
| 11.2 | League slots | 4 opponent slots; slot k plays opponent rows k, k + 4, ... so each slot plays every mode. A slot holds a network entry (stacked parameters run with `torch.vmap`), the full-strength Imugi, or one of 6 cheese agents. Every 2 updates 1 slot (round robin) is redrawn. | `runtime/opponents.py:149-196, 249-331` | RL stage | R: OpponentSlots 8,314 inclusive per tick (973 outside the model; Imugi 4,290) | 11.3 |
| 11.3 | Opponent sampling | Entries are drawn with weight (1 - p)^2 on the predicted probability p that the live policy beats the entry (`sampling = "hard"`), floor 0. Pool entries: Imugi, both averages, snapshots of the live weights at 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 130, 160M steps then every 30M, and the cheese agents. | `league/pool.py:60-80`; `runtime/snapshots.py:68-70`; `runtime/opponents.py:149-180` | RL stage | not in records | 9.13 |
| 11.4 | Carry at a cut | When a slot's entry changes mid-game, the opponent's model carry on those rows is kept (`carry_at_cut = "keep"`) when both are networks of one build; the games in progress are not tallied. The learner's ticks on those rows still train. | `runtime/opponents.py:163-164, 382-401` | RL stage | — | — |
| 11.5 | Rating lanes | 1,024 extra environments play 7 matchups (live, exponential average, uniform average, 4 league slots, each vs Imugi); never stored for the learner or counted in losses. | `runtime/lanes.py:86-123` | RL stage | R: LaneSeat 7,459 inclusive per tick | — |
| 11.6 | What reaches the learner | Opponent identity is not an input to the model and is not used by any loss. Enemy actions (from any opponent kind) are labels for the enemy-action head (8.7). | `learn/labels.py:203-206` | yes | — | 8.7 |
| 11.7 | Lineup | launch.toml fixes each team's ship types per slot: 2 Yongma, 3 Jujak, both teams the same. `TrainConfig.lineup_draw` (random types per slot per episode) defaults to True and is replaced when a fixed lineup is given. | launch.toml; `runtime/train.py:254-255, 496-503` | fixed lineup | — | 1.10 |

---

## 12. Size dependence

Classification: **structural** = cannot run at another size without a code or contract change; **numerical** = runs, but the statistics of an input, an internal sum or a target shift with the size; **none** = unaffected. "Training value" is what the champion configuration always shows the model.

### 12a. Number of ships / team size

| # | Place | Dependence | Kind | Where |
|---|---|---|---|---|
| 12.1 | Model weights | No weight shape depends on N, S, M, Z or the arena. | none | `model.py:106-107`; `network.py` |
| 12.2 | Own/enemy split by position | Heads and the belief split rows at S = width of `previous_action`; the observation orders own ships first and teams are equal in size by the game contract (`n = team x ships_per_team + slot`). | structural for unequal row counts per team (handled only through benched slots, 12.3) | `model.py:1124`; `contracts/config.py:118` |
| 12.3 | Unbalanced teams | Represented as equal slot counts with benched slots. `mask_benched` (a runtime switch, off in training) removes benched rows from attention keys and from presence; the `benched` input flag is always 0 in training. Training plays full teams only. | numerical (switch untrained-on) | `model.py:568-575`; `inputs.py:421-426` |
| 12.4 | Hard maxima | `MAX_SHIPS = 255` total (so at most 127 per team); `MAX_FIELDS = 64`. | structural beyond those | `contracts/config.py:6-7, 157-160` |
| 12.5 | Attention softmax | One softmax over Q + M + Z keys: 26 in training; 56 at 20v20; 116 at 50v50 (with M = 10, Z = 5). Each key's share of attention mass and the global token's pooling change with the count. | numerical | `spatial.py:87-107` |
| 12.6 | Dead and unseen rows as keys | Dead ships and never-seen enemies are unmasked keys (the latter all at the map centre with a zero state code); their number scales with fleet size. | numerical | `inputs.py:405-409`; 3.4 |
| 12.7 | Fused attention kernel tile | Pads to max(16, next pow2) queries and keys and holds a view's whole key set in one tile: 16 x 32 in training, 64 x 64 at 20v20, 128 x 128 at 50v50. Its docstring says the single tile is "fine up to a few dozen keys; at hundreds it needs a tiled kernel (or the plain path)"; `applies` does not check the token count. | numerical / performance; whether it runs at 50v50 is not established from code | `kernels/attention.py:21-24, 262-264, 321-328` |
| 12.8 | Local presence | A sum (not a mean) of Gaussian weights over ships within about 150 m, passed through ln(1 + .). Its range grows with local ship density. | numerical | `features.py:21-39` |
| 12.9 | Score inputs | Global token reads symlog(score_target) and scores / target; in team deathmatch target = 5 kills x ships_per_team = 25 in training (100 at 20v20, 250 at 50v50). Training shows one target value per mode. | numerical | `inputs.py:454-467`; `game/modes/phase.py:136-142` |
| 12.10 | Density head rates and bias | Targets are counts per hex cell (331 cells per side): total mass per side = living ships of that side, so rates scale with fleet size; the initial bias is set from the build's 10 ships. Training-only head. | numerical | `density.py:46-87`; `model.py:644` |
| 12.11 | Density loss normaliser | The per-step deviance over all cells is divided by the valued **own**-ship count of that step, though the cells count both sides' ships. | numerical | `learn/ppo.py:727-730` |
| 12.12 | Geometry head coverage | K = min(8, N - 1) partners per anchor per tick: 8 of 9 other rows at 5v5; 8 of 39 at 20v20; 8 of 99 at 50v50 (cycled over ticks). Training-only. | numerical (coverage per tick) | `geometry.py:42, 125-135` |
| 12.13 | Reward: opposing-team shares | "Unforced" damage and kill events pay each ship the opposing team's total divided by the receiving team's size (slots, dead included). | numerical | `learn/rewards.py:152-157` |
| 12.14 | Reward: outcome | Every ship of a team receives the full +-10 x result at the end; not divided by team size. The outcome value P(win) - P(loss) x 10 is added to every own ship's value. | none per ship; summed over a team it scales with S | `learn/rewards.py:160-166`; `heads.py:71-93` |
| 12.15 | Reward: per-ship damage and kills | Dealt/taken damage and kill shares are per ship; a ship's rate of these events depends on how many enemies are in range. | numerical | `learn/rewards.py:169-236` |
| 12.16 | Critic support | Per-ship level values are categorical over +-19.09 reward units; returns outside are clamped. Level magnitudes that grow with fleet size (12.13, 12.15, objectives) meet a fixed support. | numerical | `critic.py:23-35` |
| 12.17 | Return scale | One scalar from the 5th-95th percentile gap of per-ship summed returns over the batch, floor 1.0; frozen into the policy file. Used only in training. | numerical (training only) | `critic.py:87-110` |
| 12.18 | Advantage aggregation | Sum over 5 levels per ship; team mixing coefficient 0, so no mean over teammates enters. | none | `learn/returns.py:76-93` |
| 12.19 | Loss denominators | Every loss is a mean over its own tokens (living own ships, valued ships, view-steps, next-state rows, living enemies, shots, pairs) counted over the minibatch. Per-view-step terms (outcome, density) and per-ship terms therefore change relative weight as N changes. | numerical (relative weighting) | `learn/ppo.py:61-64, 1257-1258` |
| 12.20 | Micro-batch size | Views per micro-batch = micro_batch_rows // num_ships. Memory only. | none (memory) | `learn/ppo.py:961-964` |
| 12.21 | O(N^2) tensors | Reward ledger (B, N, N); local presence (N, N); teammate weights (S, S); attention (Q x (Q + M + Z)). | none (cost) | `learn/rewards.py:80-94`; `features.py:30` |
| 12.22 | Carry and replay shapes | Trunk state (R, 2, N + 1, 128), buffer (R, 2, N + 1, 3, 128), belief (R, N, ...), replay (R, N - S, ...) are sized at carry creation from `ships_per_team` and `num_ships`. | none (allocated per game) | `model.py:799-846` |
| 12.23 | Policy factorisation | Independent 42-way categorical per ship; nothing in the sampling couples ships. | none | `policy.py:11-33` |
| 12.24 | League slots | Opponent rows must divide evenly by 4 slots. | structural on batch layout only | `runtime/opponents.py:270-272` |
| 12.25 | Ship types | The type one-hot has the card's 3 types; shield/energy fractions use per-type maxima stored in the build. A card with another number of types is another build. | structural on type count (not ship count) | `inputs.py:440`; `state_code.py:216-217` |
| 12.26 | Shot labels | One re-simulated shot per (tick, row); cost linear in N; hit tests against all N ships. | none (cost) | `learn/shots.py:163-268` |

### 12b. Zones and fields

| # | Place | Dependence | Kind | Where |
|---|---|---|---|---|
| 12.27 | Zone slot count | The observation contract fixes Z = 5 slots in every mode; the encoder itself handles any Z. | structural in the contract (Z = 5); none in the weights | `contracts/obs.py:131` |
| 12.28 | Zone role one-hot | 5 roles (own spawn, own defense, neutral, enemy defense, enemy spawn); a role index above 4 would give an all-zero one-hot. | structural for more than 5 roles | `inputs.py:41, 515` |
| 12.29 | Absent zones | In team deathmatch and elimination all 5 slots are the one learned absent token at the map centre (5 identical unmasked keys). In frontline all 5 are present. Training never shows a partial zone set. | numerical for 1-4 zones | `network.py:103-106` |
| 12.30 | Field count | Training always shows 10 present fields (`active_fields = None`); `mask_absent_fields` is off, so an absent field slot would be read as a zero-feature field key. Field-count sampling exists as an option (off). | numerical for other counts | `runtime/train.py:253, 408-409`; `inputs.py:427-430` |
| 12.31 | Field scales | radius / 1000 m, band width / 40 m, level / 2: fixed constants of the model. | numerical if field sizes change | `inputs.py:38-40` |
| 12.32 | Zone distances | Offensive/defensive distances (captures to win/loss) enter as symlog; zone radius / 1000 m. | numerical | `inputs.py:514, 520-521` |

### 12c. Arena size

| # | Place | Dependence | Kind | Where |
|---|---|---|---|---|
| 12.33 | Arena radii seen in training | Exactly two: 7800 m (frontline, elimination) and 4200 m (deathmatch); map-size sampling (`arena_scale`) is None. | — (training fact) | `runtime/train.py:252`; presets |
| 12.34 | Arena code range | 21 real bins from 1,950 m to 62,400 m; clamped beyond. Training activates the codes at ln(1) and ln(4200/7800) only. | numerical | `inputs.py:94-112` |
| 12.35 | Raw arena scalar | arena_radius / 2600: 3.0 or 1.62 in training; 24 at 62.4 km. | numerical | `inputs.py:462` |
| 12.36 | Position colour code period | 236,196 m per axis (9 levels down to 36 m); positions are offset by a uniform draw over the full period, so every level sees all phases whatever the arena. Positions wrap at the period. | none below the period; aliasing beyond it | `model.py:101`; `codes.py:31-53` |
| 12.37 | Rotary coarsest band | 339,625 m; a relative phase is unique within +- half a period (169.8 km). Finest band 64 m. | none below that separation | `model.py:94-100` |
| 12.38 | Pairwise distances in training | With radii 4.2-7.8 km the largest pairwise separation in training is about 15.6 km: the two coarsest rotary bands (339.6 km, 99.7 km) and the coarsest colour levels turn through a small part of their period in training. | numerical | derived from 12.33, 12.37 |
| 12.39 | Error-head position bound | The error code's position range is +-26,244 m per projection axis (`state_code.POSITION_PERIOD`, the old default, hard-coded independently of the build's 236,196 m period); larger errors are clamped to the padding bins. | numerical on maps wider than that | `correction.py:433-450` |
| 12.40 | Geometry range bound | 64,000 m. Training-only. | numerical beyond | `geometry.py:66` |
| 12.41 | Density grid | 10 hex rings scaled to the arena radius: cell width = radius / 10 (780 m or 420 m in training; 6.2 km at 62.4 km). Ships outside the grid are dropped from the target. | numerical | `density.py:46-87` |
| 12.42 | Shot codes | Displacement period 5,080 m and top speed 5,542 m/s come from the card's muzzle speed, the velocity reach and bullet life, not from the arena. | none | `shots.py:63-74` |
| 12.43 | Presence radius | 150 m fixed. | none (density-dependent, 12.8) | `model.py:255` |
| 12.44 | Timers | Respawn/lock/boundary timers use a 16 s horizon; `episode_progress` is a fraction. Episode lengths in training: 600 s, 600 s, 180 s. | numerical if clocks change | `features.py:8-9` |

### 12d. Batch composition

| # | Place | Dependence | Kind | Where |
|---|---|---|---|---|
| 12.45 | Return scale over the mix | One percentile gap over all modes' and pools' ship-steps together; changing mode shares changes the scale every mode's advantage is divided by. | numerical (training) | `learn/ppo.py:1052` |
| 12.46 | One size per batch | A mixed batch requires every game to have the same ships_per_team, num_fields and ship set. | structural for training on mixed sizes in one batch | `runtime/train.py:99-103` |
| 12.47 | Minibatch denominators | Token counts are taken per minibatch, so a minibatch's term means depend on which views (modes, dead-ship fractions) it drew. | numerical (training) | `learn/ppo.py:1257-1258` |
| 12.48 | No batch statistics in the model | No BatchNorm or running input statistics; the model's forward is per view. | none | `layers.py:13-15` |

---

## Options present but off

- `acceleration_inputs` (ship token reads symlog-coded accelerations, 106 inputs): off (`model.py:275`). The accelerations are still computed for the extrapolation baseline.
- `spread_inputs = "ellipse"` (covariance ellipse features on the ship token): off; "scalar" is used (`model.py:273`).
- `position_inputs = "fourier"` (Fourier features in place of the colour code): off (`model.py:274`).
- `physics_hidden_fade` (fade a hidden row's physics change by exp(-age / tau)): off (`model.py:283`).
- `gate` (attention output gate, elementwise or headwise): "none" (`model.py:284`).
- `rotary = False` (identity rotation) and `forecast_feedback = False`: both left at True (`model.py:267-268`).
- `order = "spatial_first"` / `"spatial_only"`: not used (`network.py:110`).
- `next_state = "absolute"`, `delta_baseline = "persistence"`, `delta_split = False`: not used (`model.py:271-279`).
- `spread_fit = "v1"`: not used; "coarsest" is (`model.py:270`).
- Single summed value level (`SUMMED_LEVEL`, `outcome_level = None`): not used (`learn/returns.py:13`).
- `team_spirit` nonzero (team mixing of advantages): all zeros (`learn/ppo.py:147`).
- Trust regions `clip_higher` (upper margin 0.28), `dppo_tv`, `dppo_binary_tv` (delta 0.15): not used (`learn/ppo.py:83, 152-153`).
- `old_log_probabilities = "recomputed"`: not used (`learn/ppo.py:179`).
- Entropy bonus -0.0081 per factor: the default, replaced by 0.0 on the warm start (`learn/ppo.py:155-157, 212`).
- `arena_scale` (log-normal map-size sampling, sigma 0.35, range 0.5x-2x): None (`runtime/train.py:252`).
- `active_fields` (field-count sampling) and `mask_absent_fields`: None / False (`runtime/train.py:253`).
- `mask_benched` (uneven teams): False in training (`model.py:575`).
- `lineup_draw` (random ship types per slot per episode): superseded by the fixed lineup (`runtime/train.py:255`).
- `deathmatch_outcome = "kill_share"`: not used; "kill_lead" is (`learn/rewards.py:62`).
- League `sampling = "variance"`, `floor > 0`, `carry_at_cut = "reset"`: not used (`runtime/opponents.py:163-166`).
- BC `plateau_stops`, `score_target`: off (`learn/bc.py:409-411`). Teacher temperature other than 1 and floor above 0: not used.
- Inference-time `Ablations` (rotary off, forecast off, absent zones masked) and sampling temperature: not used in training (`model.py:399-421`).
- `compile_split`, `combat_collapse_stop`, `host_store` (launch sets `host_store = false`): off (`runtime/train.py:244-258`).
- Models `relational` and `nearest`: not selected (`runtime/train.py:84`).
- Retired: `rotary_readout` (model and learner coefficient) (`model.py:243`; `learn/ppo.py:182`).

---

## v1 equivalents

v1 is `origin/main`, package `src/boost_and_broadside/`. Read from code there (`profiles/rl.py`, `config/defaults.py`, `config/core.py`, `models/yemong/*.py`, `train/rl/features.py`, `train/rl/ppo.py`); its docs were not read.

**Inputs and encodings.** v1 tokens were ships and fields in one list (N + M), plus bullets as key/value-only tokens through a narrow encoder (hidden 64) read by cross-attention in the first spatial sublayer of each block. Features went through a `FeatureCoordinator` of continuous transforms: positions as Fourier features with 4 frequencies at base-2 periods of the world width/height, heading as a 4-frequency Fourier of 2 pi, velocity as direction x symlog(speed), angular velocity as symlog, health/power/cooldown as unit-circle encodings scaled by their maxima, local log index and gradient as identity. There were no categorical input codes, no belief state, no global or zone tokens, no random position offset, and no rotary bands: attention had no explicit relative-position term (plain scaled dot-product over tokens carrying absolute Fourier features of a 1024 x 1024 world, which the config treats as toroidal). The RL profile ran two policy passes per step (`paradigm = "ego_pass"`: the raw observation and a team-flipped one). Profile sizes: `num_ships = 8`, `num_fields = 4`.

**Trunk.** Width 128, 4 heads, 2 blocks, each block 2 spatial sublayers then 1 temporal sublayer (spatial first). Spatial sublayer: pre-RMSNorm multi-head self-attention over all ship and field tokens with dead entities masked out of keys, optional bullet cross-attention with its own key/value projection, then the same gated MLP (4x). Temporal sublayer: causal convolution plus RG-LRU, recurrent for ship tokens only. Encoder: one shared Linear -> RMSNorm -> GELU -> Linear -> RMSNorm over all tokens by default (`encoder_split` optional).

**Heads.** Action head: 12 logits as three independent categoricals (power, turn, shoot), log-probability and entropy summed over factors. Value: a scalar regression per ship with K outputs, one per reward component (the gamma table names 25 components), in a normalised space; the win/loss components came from a separate head on a per-team attention pooling (`TeamPMA`, two learned seeds). Next-state head: one MLP predicting 10 continuous deltas (phase deltas, symlog velocity delta, etc.) with regression losses (`next_state_coef` 0.2, `windowed_loss_coef` 0.1). No outcome classifier, density, shot, geometry, enemy-action or error heads.

**Value and returns.** Per-component GAE with per-component gammas (0.999 win, 0.995 kill/death, 0.991 damage, 0.975 shaping, per 60 Hz tick, raised to the action repeat of 2) and lambdas (0.80-0.97). Returns normalised per component by a `ReturnScaler`: EMA (alpha 0.005) of the 5th/95th percentiles in symlog-reward space with a minimum span of 1e-3. Value loss: Huber (delta 1.0) on normalised targets, mean over alive ships and components. Advantage: components aggregated to one number per ship (with a precomputed lambda/team aggregation), divided by the whole-buffer RMS of the aggregated advantage (floored at 0.1). Rewards included shaping terms (facing 0.09, closing speed 0.08) tapered by a schedule from 1.0 to 0.05 between 100M and 400M steps.

**Loss balancing.** Scheduled coefficients as keypoint tables: policy 1.0, entropy 0.005 (bonus on), value 1.0, BC 2.0 (a BC term could run alongside PPO), plus tier scales per reward family. v2 has fixed coefficients (policy 27, value 0.007, auxiliaries 0.0033-0.11) and no schedules.

**PPO settings.** Clip 0.15; 4 epochs; 32 minibatches; rollout 128 decision steps at action repeat 2 (60 Hz physics, 30 Hz decisions); target KL 0.1 (0.02 above an 80% win rate); max grad norm 1.0; Adam with eps 1e-5; learning rate 1e-7 -> 4.5e-4 linearly over 5M steps, held to 100M, exponential decay to 1.5e-4 at 500M; bf16 autocast; budget 500M steps; league of 20 with 4 slots sampled by Elo proximity (temperature 200), league fraction 0.5. Cost record for v1 (`v1_cost_L32.txt`): forward 56.3 M FLOPs per env-tick, forward + backward 168.7 M (v2: 60.6 M and 213.6 M per view-tick).

---

## Uncertain

- **What launch.py adds.** Only `launch.toml` was read (as instructed); `launch.py` and `arms.toml` were not. How the lineup, the probed `micro_batch_rows`, and any per-arm `--set` lines reach `bnb train` is inferred from `launch.toml`'s comments and `cli.py`, not traced. The micro-batch size actually used (640 or 320 rows for RL; 2560 or 1280 for BC) depends on the GPU probe.
- **launch.toml comments disagree with the code in four places.** The inventory records the code. (a) Mode mix: the comment says frontline 0.5 / deathmatch 0.3125 / elimination 0.1875 (1024 / 640 / 384); `TRAINING_MIXES` gives 0.375 / 0.3125 / 0.3125 (768 / 640 / 640). (b) Rotary ladder: the comment says 48 m to 254.7 km; the code's `ROTARY_FINEST` is 64 m, coarsest 339,625 m. (c) Hidden physics decay: the comment says 0.5; `TrainConfig.yemong` sets `physics_hidden_decay = 0.25` s. (d) `YemongConfig`'s docstring gives the coarsest rotary period as 339,695 m; 64 x 3.405^7 evaluates to 339,625.1 m.
- **RL-stage entropy coefficient.** 0.0 follows from `cli.warm_start_entropy` plus the empty `rl_set`; if an arm sets `learner.entropy_*` it would differ.
- **Which build the records traced.** The records files do not state their model settings; `state_code.ellipse` appears in the prepare phase (13,824 ops), which in the champion build is reachable only through diagnostics. Costs are quoted as recorded.
- **Per-head cost.** The records give head groups in aggregate (R: 18 ops per tick; F: HeadGroup 26.04 M FLOPs) and do not separate policy, value, next-state, correction, density, shot or geometry heads. The shot-trajectory query and geometry query have no FLOP line.
- **BC-stage cost.** No v2 BC record was among the allowed files; the teacher's cost per update is not in the records (Imugi as an opponent is 4,290 ops per rollout tick).
- **Objective reward split.** `objective_progress` and `objective` arrive from the game's mode rules as per-ship signed amounts summing to zero; how a zone's progress or capture is divided among a team's ships (and so how it scales with team size) lives in `game/modes/` and was not traced.
- **Attention kernel at large N.** Whether the single-tile Triton kernel runs, and at what speed, with 116 keys (50v50) is not determinable from code; its `applies` test has no token-count bound and its docstring names "hundreds" of keys as needing another kernel.
- **Dead own ships in presence and attention.** `local_presence` sums over rows with a valid belief and does not test `alive`; what x, y an own dead ship's observation row carries (and therefore where it sits as a key and in presence) was not traced into the game's observation code.
- **Adam hyperparameters.** betas (0.9, 0.999), eps 1e-8 and weight decay 0 are torch 2.15's defaults, inferred from the constructor call passing only `lr` and `fused`; no line in the repository states them.
- **Evaluation-time settings.** Which runtime switches (`mask_benched`, `mask_absent_fields`, temperature) the 20v20 / 50v50 evaluation turns on is in `league/matches.py` and the evaluation settings, outside this inventory's scope; only their existence and their training-time state are recorded.
- **Marginal bias tables.** The next-state and error heads' starting biases are measured tables in `delta.py` and `correction.py`; which table the extrapolation build uses (`EXTRAPOLATION_MARGINAL`) was confirmed by name, its values were not checked.
