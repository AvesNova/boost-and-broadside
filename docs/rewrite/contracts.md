# Contracts (section 0)

Status: **draft, first review done** 2026-10-31 (see the review log at the end).
- Each item is marked **carried** (decided before, or the race spec), **new** (a choice
  made here), or **open**.
- Plan and design rules: [scaffold.md](scaffold.md).
- Game math: the race spec (`physics-race/SPEC.md`, branch `v3-general`), until the
  physics section rewrites it here.

## Conventions

| | Rule | Status |
|---|---|---|
| Shapes | Envs `B`, teams `T` (2 for now, equal sizes), ships per team `S`, ships `N = T·S`, fields `M`, bullet slots per ship `K`, zone slots `Z`, spawn candidates `C`, a segment's steps `L`, views `R`; the reward levels are written `levels` (cleanup pass) | carried |
| Ship order | Team-major: ship `n = team·S + slot`. Ship tensors are stored `(B, N, …)` and viewed `(B, T, S, …)` for free | new |
| Floats | float32 everywhere in game and runtime. float64 only in `reference/`. bf16 only inside models | carried |
| Ints | int32 counters and categoricals; bool flags; int64 only where torch requires it (indexing), converted at the use site | carried |
| Ship indices | At most 255 ships (`MAX_SHIPS`, author: no plans beyond 256). Stored ship indices (killer, first shooter) are int32 with −1 = none; index tensors are int64 where torch requires it | new |
| Units | m, s, kg, J, W, rad. Angles counter-clockwise, y up | carried |
| Containers | frozen `@dataclass`es whose fields are tensors, registered as pytrees (`torch.utils._pytree`) so the runtime can map `copy_`, `clone`, `to(device)` over them. No dicts at contract boundaries | new |
| Versions | `Obs`, `Action` and `Events` each carry a `SCHEMA_VERSION` class constant. A model checkpoint and an exported ONNX model record the versions they were trained on; loading a mismatch fails | new |

## 1. Config

Static values are Python numbers and shapes, so compiled graphs specialize on them.
Tunable values are tensors, so changing them does not recompile.

```python
@dataclass(frozen=True)
class GameConfig:            # static
    num_teams: int           # T = 2
    ships_per_team: int      # S
    num_fields: int          # M, 1..64
    arena_radius: float      # R_a (m)
    bullet_slots: int        # K
    tick: float              # h = 1/30 s (decision tick = physics tick)
    episode_s: float         # time-native rules in seconds; each must be an exact
    respawn_s: float         #   multiple of `tick`, or construction fails
    team_types: tuple[int, ...]   # ship type per slot, the same for every team (length S)
    mode: ModeConfig         # rule components (section 3 of the plan: modes section)

@dataclass(frozen=True)
class ShipTypes:             # tunable: one row per type, tensors of shape (num_types,)
    cl_max: Tensor; a_max: Tensor; cd0: Tensor; power: Tensor; p_min: Tensor
    f_static: Tensor; e_max: Tensor; omega_c: Tensor; shield_max: Tensor
    hit_radius: Tensor; damage: Tensor; muzzle_speed: Tensor; bullet_cost: Tensor
    fire_interval: Tensor; spawn_energy: Tensor   # (race SPEC 1.1; the physics spec owns the list)
```

- **carried:** ship types as data (race v2). Shared flight constants (g, k, α_s, band, …)
  are Python numbers in a `FlightConstants` dataclass owned by the physics section.
- **new:** the slot→type layout is the same for every team (`team_types`), so teams are
  symmetric by construction.

## 2. Game state

Opaque outside `game/`. The runtime copies it; the play shell reads it only through a pure
`snapshot(state)`. Split into four dataclasses so each section owns its part:

| Dataclass | Shape | Fields | Status |
|---|---|---|---|
| `Ships` | `(B, N)` | `x y c s w alpha ca sa r E logn gx gy shield shield_wait` (float); `alive fired spawned` (bool); `respawn gun_wait cursor ship_type` (int) | carried (race INTERFACE) |
| `Bullets` | `(B, N, K)` | `x y dx dy w logn gx gy` (float); `ttl` (int) | carried |
| `Fields` | `(B, M)` | `x y radius width` (float); `level` (int) | carried |
| `Match` | `(B,)` and `(B, T)` | `t` (int); `seat_of_team (B, T)`: which map side each team got this episode; `last_seen (B, T, N)` ticks since team T last saw ship n | **new** |

- **new, `seat_of_team`:**
  - Map generation draws the map in "side" terms (side 0, side 1: spawn sides, and later
    Frontline's clockwise and counter-clockwise roles).
  - A per-episode random permutation assigns sides to teams.
  - Nothing in the observation depends on the team's sim index (§4).
- **new, `last_seen`:** this feeds the decided "time since last seen" observation
  feature. It is counted in ticks and reset to 0 when the ship is visible or spawns.
- The race's `axis_x, axis_y` (spawn axis) become part of `Match` as the side geometry.

## 3. Action

```python
@dataclass(frozen=True)
class Action:                # (B, N) each, int32
    power: Tensor            # 0 coast, 1 boost, 2 reverse
    stick: Tensor            # 0..6: straight, three levels each side
    shoot: Tensor            # 0, 1
    SCHEMA_VERSION = 1
```

- **open → recommended: factored components** in the contract. The 42 joint actions are
  `power·14 + stick·2 + shoot`; helpers `to_joint` and `from_joint` live in
  `contracts/action.py`.
- A model can still use a 42-way joint head internally (Yemong does) and convert.
- Why factored:
  - humans, scripted agents and the ONNX boundary produce components naturally;
  - a new component (a weapon switch, say) adds one field instead of multiplying the joint
    size;
  - events and diagnostics read components directly.
- Shape `(B, N)`, not per team: the runtime fills each ship's slot from its controller
  (§7).

## 4. Observation

One `Obs` per viewing team, built by `observe(state, cfg) -> Obs` with a leading view axis
`(B, T, …)`: view `v` is what team `v` sees. Everything is in world coordinates, with no
rotation or mirroring (decided).

**Team identity (decided 2026-10-31).** Agents never see the sim's team index:
- every entity carries `team_rel = (team − viewer) mod T`;
- **entities are ordered by `team_rel`** (own ships first). This is a reordering, not
  geometry, and costs nothing:
  - with team-major storage it is a roll of the team axis, `(B, T, S, F)`, fused into the
    write the observe stage does anyway;
  - a permutation-invariant model (Yemong: no token-index embedding) does not care;
  - a model that reads token position would otherwise learn its seat. The skeleton's MLP
    over the flattened view is one;
  - own ships sit at a fixed slice `[:S]`, so the per-ship action heads and enemy-action
    heads slice instead of gathering.
- The sim index lives in `Privileged`.

```python
@dataclass(frozen=True)
class Obs:
    ships:   Tensor   # (B, T_view, N, F_ship)    float32, ordered by team_rel then slot
    bullets: Tensor   # (B, T_view, N·K, F_bullet) ordered by owner's team_rel, slot, k
    fields:  Tensor   # (B, T_view, M, F_field)  same for every view (repeated or expanded)
    globals: Tensor   # (B, T_view, F_global)
    SCHEMA_VERSION = 1
```

Ship columns: race SPEC 10.2 with the changes marked.

| Column | Hidden ship | Status |
|---|---|---|
| `x y c s v_x v_y w alpha r E shield gun_wait` | 0 | carried |
| `alive`, `visible`, `respawn` | kept | carried |
| `type` (ship type, categorical) | kept | carried |
| `team_rel` (categorical) | kept | **new**, replaces `ally` |
| `since_seen` (s, from `Match.last_seen`) | kept | **new**, decided feature (decision log, hidden ships) |
| `spawned` (lifecycle flag) | kept | **new**, decided ("resets and respawns arrive as lifecycle flags") |

- **Bullets:** `x y v_x v_y visible team_rel`; zeros when not visible. **new:** `team_rel`.
- **Fields:** `x y radius level`. carried.
- **Globals:**
  - `t / T_ep`, `R_a`;
  - **new:** `fog` flag;
  - `reset` lifecycle flag (this env reset this tick);
  - mode block: mode id, scores by `team_rel`, time left, respawn wait. Defined by the
    modes section; the skeleton carries team deathmatch scores.
- Hidden-entity rule (carried): masked values are zeros plus the visible flag; no
  last-seen values; every ship is visible on the tick it spawns.
- Fog off (carried, mixed fog): every ship and bullet is visible; the flag says so.

## 5. Events

Per tick, from `step`. These are what rewards and diagnostics are built from.

```python
@dataclass(frozen=True)
class Events:                # (B, N) unless noted
    damage: Tensor           # (B, N, N) float: hit damage from shooter to target, this tick
    boundary_damage: Tensor  # float
    died: Tensor; fired: Tensor; spawned: Tensor    # bool
    killer: Tensor           # int: ship index of the killing shooter, −1 if none
    done: Tensor             # (B,) bool: the env reset after this tick
    SCHEMA_VERSION = 1
```

- **carried (revised 2026-10-31):** the pairwise `damage (B, N, N)` stays.
  - Today's rewards need pairs: kill credit is split among attackers, half by the killing
    tick's hits and half by a running per-pair ledger of damage not yet recharged.
    Shield recharge charges attackers back pro rata from that ledger (`rewards.py`
    `KillDeathReward`, `DamageReward`).
  - The matrix is a per-tick transient: the reward stage consumes it and the trajectory
    never stores it. Only per-ship rewards are stored (§9). At N = 100 and B = 1024 it is
    40 MB, plus a ledger of the same size.
  - The ledger `(B, N, N)` is **mode state**: it lives in the runtime carry beside the
    game state, owned by `modes/` and cleared for a slot on respawn. It is not game state.
- Indices in `Events` are sim indices: events are learner-side and privileged in spirit.
  A reward function maps them to per-ship rewards. Agents never receive `Events`.

## 6. Privileged

```python
@dataclass(frozen=True)
class Privileged:
    team: Tensor             # (B, N) int: sim team index
    view_order: Tensor       # (B, T, N) int: sim ship index of each observation row
    ships: Tensor            # (B, T_view, N, F_ship): the observation with fog off
    launches: Tensor         # (B, N, F_launch): per-tick launch state for shot re-simulation
```

- **carried:** a separate type, reachable only from learner batches.
- **new, enforcement by structure:**
  - `Privileged` is produced by the perceive stage and written only into the rollout
    buffer.
  - The act stage's signature has no slot for it.
  - A contract test checks that no `Controller.step` parameter is annotated with it.
- The exact `launches` fields come from the learner-labels section; the skeleton leaves
  them out.

## 7. Controllers

```python
class Controller(Protocol):
    def init_carry(self, batch: int) -> Carry: ...
    def step(self, params: Params, carry: Carry, obs: Obs, gen: torch.Generator
             ) -> tuple[Action, Carry, Outputs]: ...
    def unroll(self, params: Params, carry0: Carry, obs_seq: Obs) -> Outputs: ...

@dataclass(frozen=True)
class Outputs:
    logp: Tensor | None      # (…, S) per controlled ship; None for non-learning controllers
    entropy: Tensor | None
    value: Tensor | None
    extra: dict[str, Tensor] # model-owned (aux heads, inspect); opaque to the runtime
```

- **carried:**
  - an opaque carry;
  - `unroll` equals step-by-step (conformance test);
  - feature encoding belongs to the model;
  - PPO needs only logp, entropy and value;
  - `aux_loss` sits on the model.
- **new (author, 2026-10-31):** a controller plays **one team**.
  - Input: everything that team sees, i.e. its view of `Obs` (own ships, visible
    enemies, bullets, fields, globals), shaped `(B_pool, …)`.
  - Output: `Action` for that team's `S` ships only. Only these are executed or trained
    as actions.
  - A model may also predict the enemy team's actions (an enemy-action head, shared or
    separate). That prediction goes in `Outputs.extra` as an auxiliary output; it is
    never executed.
- **new:** `gen` is the controller's named random stream. The runtime registers it with
  the CUDA graph.
- **carried:** the human controller is a pure function of an input array that the shell
  fills each frame.
- Client play: `step` for a model is exported to ONNX. Its inputs and outputs are the
  `Obs` view tensors and the carry tensors, flattened by the pytree.

## 8. Pools (lineups)

```python
@dataclass(frozen=True)
class Pool:
    name: str                # "self_play", "league_0", …, "scripted"
    share: float             # fraction of B; share · B must be an integer, or construction fails
    seats: tuple[str, ...]   # controller per team, e.g. ("learner", "league_0")
    fog_share: float         # fraction of this pool's envs with fog on
```

- **decided:** fixed ratios, for example X% self-play, X% per league slot, X% scripted.
- **new:**
  - Each pool is a contiguous slice of the B envs.
  - The learner's seat is randomized **per env per episode** within a pool, so it plays
    both sides equally.
  - League slot parameters are swapped in with `copy_` between rollouts.
- Training views: every view whose controller is the learner (both views in self-play).
  Decided in review 4a.

## 9. Trajectory

The rollout buffer, preallocated `(L, B_learn, …)` with L = 128 steps. It holds the
learner's views only.

| Field | Source | Notes |
|---|---|---|
| `obs` | perceive | learner views |
| `action`, `logp`, `value` | act | |
| `reward` | modes(events) | per learner ship |
| `done`, `spawned` | events | episode and life boundaries for GAE and the carry |
| `carry0` | act | the carry at rollout start, for `unroll` |
| `privileged` | perceive | labels only |
| `extra` | model | stored only if the model asks (e.g. last forecast mean and σ) |

- Delayed labels (shot re-simulation, next-state targets) are computed from this buffer
  by `learn/labels`, after the rollout. Carried (decided).
- Observation chunks streamed to host RAM while the next chunk collects (decided, 4d)
  are a runtime option. The contract does not change.

## 10. Functions

```python
reset(cfg, types, gen) -> State
step(cfg, types, state, action, gen) -> tuple[State, Events]
observe(cfg, state) -> tuple[Obs, Privileged]
reward(mode, mode_state, events, state) -> tuple[Tensor, ModeState]  # modes section
snapshot(state) -> RenderSnapshot            # play section
```

- **new:** `observe` is separate from `step`, so the runtime can cut stages between them.
  The race tested `step` returning the observation; the arm split them internally anyway.
- **new:** randomness enters through `gen`. The correctness surface for tests also accepts
  explicit draws, as in the race: spread normals and reset uniforms.

## Review log

- 2026-10-31 (author):
  1. Factored action components: **agreed**.
  2. Ordering by `team_rel`: asked about cost. Answer above (free); kept unless the
     author objects.
  3. Rewards must keep their information: the pairwise matrix stays as a per-tick
     transient, and the ledger becomes mode state.
  4. `last_seen` and `seat_of_team`: **agreed**.
  5. A controller gets the whole team view and acts for its own ships only. An
     enemy-action prediction is an auxiliary output: **agreed, revised as above**.
  6. Learner seat randomized per env per episode: **agreed** (a cheap guard; maps are
     already seat-symmetric).

## Current schema (2026-11-01, after the game section)

The source of truth is the code: `src/bnb/contracts/`. The tables above are the first
cut; where they differ, this section and the code win.

- **Ship columns (29 since 2026-11-13: benched; 28 since 2026-11-08; 24 before),** in two
  groups:
  - zeroed for a ship the team cannot see: x, y, heading_cos, heading_sin, velocity_x,
    velocity_y, proper_speed, angle_of_attack, nose_rate, energy, shield,
    shield_recharge_wait, gun_cooldown, log_refractive_index, log_index_gradient_x,
    log_index_gradient_y, and (2026-11-08, Obs schema still 3: policy files record their
    columns) acceleration_x, acceleration_y, nose_acceleration, acceleration_valid, the
    world-frame change over the last tick, valid only when the team saw the ship alive at
    both ticks and it did not spawn, past a game's first tick;
  - always shown: alive, visible, fired, respawn_wait, ship_type, team_relative,
    time_since_seen, spawned, and (2026-11-13) benched: 1 for a slot out of an uneven
    match (a team's last S - active slots), shown to both teams under fog.

  Times are in seconds.
- **Field columns:** x, y, radius, band_width, level.
- **Global columns:** episode_progress, arena_radius, fogged, reset, then the mode's
  columns (see the modes section below).
- **Fog of war:**
  - per environment, `Match.fogged`, set at reset;
  - `Pool.fog_share` (default 1.0) chooses the fogged share of each pool, as a fixed
    contiguous split;
  - without fog, every living ship is visible.
- **Ship card:** `bnb.game.ship_types.ShipTypes`, per-type tensors shaped `(types,)` or
  `(B, types)` (per-environment cards for sweeps). The game functions that read it take it
  as an argument.
  - **Section 1b (merged 2026-11-02):**
    - the formerly shared parameters (`SCALED`) are per-type scales of their shared
      value in `Rules` or `FlightConstants`. The default card (every scale 1) is the
      shared game, bit for bit;
    - the rate of fire is `fire_rate` in shots per second. `ticks_between_shots` converts
      it exactly (`Fraction`), and `reset` refuses a rate that is not a whole number of
      ticks at the run's tick rate;
    - `bleed_exponent` `e` in the induced drag; at `e = 2` the term is bit-exact with
      the old one;
    - readers of per-type values use the card: Yemong's coding ranges take the card's
      largest stall angle, recharge delay and ticks between shots;
    - the play shell's snapshot still reads the shared `stall_angle` (an open request).
  - **Designer handles:** `bnb.game.handling`. `measure` flies the game's own code;
    `solve` finds the card that meets a set of handles (Levenberg–Marquardt in float64,
    the cruise fix included). `bnb ship-card` is its command.
- **Game API:** `bnb.game.api`: `rules_of`, `ship_types`, `reset`, `move`, `resolve`,
  `step`, `perceive`, `spread`, `uniforms`, `start_clocks`, `sweep_like`. Randomness is
  explicit (spec section 2). For a runtime holding the state in persistent buffers (the
  cleanup pass): `perceived`, `episode_ticks`, `launch_capture`, `ship_team`, `view_order`.
  No other package reads a field of the state (`tests/contract/test_layering.py` checks
  the runtime).
  - **Display functions (section 10, 2026-11-01), for the play shell:**
    - `log_index`: the medium at chosen points;
    - `outer_radius` (2026-11-13): the outer band edge the sight test uses, beside
      `core_radius`;
    - `core_radius`: the cores the sight test uses (a second copy, held to
      `visibility` by a test until the cleanup pass makes it the one definition);
    - `zone_sides`: Frontline's zone owners and attackers, from the game's own role
      rules;
    - `visible`: which ships each team's observation shows, in sim order.

    The api also re-exports `vision_range`, `per_ship`, `TEAMS`, `LN_SQRT2`,
    `MAP_SCALE_RADIUS` and `PRESETS`, which moves to `bnb.runtime` in the cleanup pass.
  - **`bnb.play` imports only** `bnb.game.api`, `bnb.runtime`, `bnb.controllers` and
    `bnb.contracts` (a test parses its imports).
- **Teams:** two, a constant (`rules.TEAMS`).
- **Names:** full words (scaffold rule 10).
- **Modes and rewards (sections 3 and 3b, 2026-11-01):**
  - **`GameConfig`** gains `mode` (`"team_deathmatch"` default, `"frontline"`,
    `"sandbox"`), `spawn_groups` and `kills_per_ship`. `episode_seconds` and `respawn_seconds` default
    to the mode's (None).
  - **`Events` (schema 2)** gains:
    - `shield_lost` and `shield_restored` (B, N);
    - `objective_progress` and `objective` (B, N), each zero-sum per environment;
    - `result` (B,), for team 0 on done environments.
  - **`Obs` (schema 2)** gains a `zones` part (`ZONE_COLUMNS`: x, y, radius, role,
    progress, direction, offensive_distance, defensive_distance). Its rows are in each
    viewer's own ring order, so the order reveals no seat. (Since section 3c, `Z` is 5 in
    every mode; see below.) `GLOBAL_COLUMNS` gains `mode_team_deathmatch`, `mode_frontline`, `score_own`,
    `score_enemy`, `score_target`, `front` and `front_threshold`. All are team-relative.
  - **`Events` (schema 3, 2026-11-12)** gains `enemy_damage_by_aspect` (B, N, 3) and
    `kill_aspect` (B, N) int32, the hit's aspect class (`ASPECTS`: head-on, flank, tail)
    for the statistics. The game's numbers are unchanged; the fingerprint, which hashes
    the state and the observation, stays.
  - **`Events` (schema 4, 2026-11-12)**: the head-on class split in two, so four classes
    (`ASPECTS`: mutual head-on under 10 degrees, offset head-on to 45, flank to 135,
    tail); `enemy_damage_by_aspect` is (B, N, 4). The game's numbers are unchanged.
  - **`REWARD_LEVELS`** (`bnb.contracts.trajectory`): damage, kill_death,
    objective_progress, objective, outcome. `Trajectory.reward` is
    `(L, R, S, len(REWARD_LEVELS))`, per level and weighted.
  - **Reward state:** `bnb.learn.rewards.Ledger(ships (B, N, N), boundary (B, N))`,
    carried by the runtime. That is the spec's (B, N + 1, N), split for speed.
- **Mixed batches (section 3c, 2026-11-01):**
  - **`Pool`** gains `mode`, `arena_radius`, `spawn_groups`, `episode_seconds` and
    `respawn_seconds` (None: the `GameConfig`'s value, and for the clocks then the mode's default).
    `pool_game(config, pool)` gives a pool's game; `environment_multiple(pools)` the batch
    size multiple a lineup needs (64 for the training mix). The `GameConfig` keeps the
    shared sizes and the pools' defaults.
  - **`Rules.pools`** (a tuple of `PoolRules`: environments, mode, arena radius, episode
    and respawn ticks) replaces `Rules.mode`, `arena_radius`, `episode_ticks` and
    `respawn_ticks`. Adjacent pools of one game merge; without pools, one game covers the
    whole batch at any size.
  - **`bnb.game.layout.Environment`:** the per-environment arena radius, map scale and
    clocks, `(B,)` tensors. Mode-owned steps run on static per-mode slices.
  - **Zones:** `Z = 5` in every mode, a fixed shape like `M`. `ZONE_COLUMNS` gains
    `present` (last). An absent zone has `present` 0 and every other column 0.
    **`Obs` is schema 3.**
  - **Yemong:** an absent zone's map-memory token is one learned vector, selected with
    `where` (no mask), with the global token's rotation.
  - **The training mix** (`bnb.runtime.train.TRAINING_MIX`, preset `mixed`): Frontline
    0.5, team deathmatch squad 0.375, team deathmatch battle map 0.125 (5v5, 10 fields,
    arena 8200 m, groups (2, 2, 1)). *2026-11-06:* the battle map is out for now; Frontline
    0.5, team deathmatch squad 0.5.
- **Learner outputs (section 5, 2026-11-01):**
  - **`Outputs`** gains:
    - `values (R, S, levels)`: the critic per reward level, in raw units;
    - `outcome (R, 4)`: the outcome probabilities (win, tie, loss, unresolved);
    - `replay`: an opaque pytree with leading `(R,)`, what the decision read from the
      controller's carry.
    `value` is the sum of `values` when a controller gives them.
  - **`Trajectory`** gains `values (L, R, S, levels)`, `outcome (L, R, 4)`, `replay (L, R, …)`
    and `initial_carry`, the learner's carry at the start of the segment. The runtime fills
    them for the learner's views; all are `None` for controllers without them.
  - **Rule:** the replay must hold exactly what the step read. Never store a rounded copy
    of a value that a compiled stage also uses unrounded: inductor drops fused precision
    casts by default.
- **The learner (section 6, merged 2026-11-02):**
  - **`Trajectory.action`** holds every row's action, `(L, R, N)` in the view's row order
    (own ships first; the learner's are `[..., :S]`).
  - **`Trajectory`** gains:
    - `result (L, R)` int32: the match result for the view's team (+1 win, −1 loss,
      0 draw), nonzero only where `done`. The win rates read it, never the sign of a
      reward;
    - `win_loss (L, R, 2)` float: the view's team's weights of a win and of a loss where
      `done` (`Events.win_loss`, game v17): the outcome reward and the outcome targets read
      it. Team deathmatch's is its kill share `(share, 1 - share)`; the other modes' the
      result one-hot;
    - the label inputs, filled by the runtime for the learner's views:
      - `alive (L, R, N)` and `valued (L, R, S)`;
      - `truth (L, R, N − S, …)`, the enemy rows of `Privileged.ships`;
      - `final_truth (R, N, …)`, after the last step;
      - `capture`, the shot capture (`bnb.learn.shots.ShotCapture`), each row's launch
        state and move at t.
  - **Randomness in the update:** the learner's own streams (`bnb.learn.ppo.random_stream`,
    named `shot_spread` and `minibatch_order`), never the global generator.
  - **Shot labels:** a shot launched within the longest bullet flight of the segment's
    end is invalid, whatever it hit. Episode ends keep the flight-end rule: the bullets
    really vanish there.
- **Runs and checkpoints (section 11, merged 2026-11-02):**
  - **Layer order:** `bnb.runs` sits between `bnb.runtime` and `bnb.league` (a test
    parses the imports).
  - **Models:** every model class has `NAME`, `VERSION` and a constructor taking its build
    record alone (`YemongBuild`, `RelationalBuild`, `MLPBuild`), kept on the instance as
    `build`. `yemong_build(config, rules, types)` derives what Yemong reads from the game
    and the ship card. `VERSION` rises whenever a change alters what saved weights mean;
    `tests/runs/checkpoints` holds one tiny saved file per accepted version (the
    compatibility set). `bnb.runs.registry` maps names to classes and classes to build
    records.
  - **Policy file** (a directory): `checkpoint.json` (format 1: model name, version and
    build; the schema versions of `Obs`, `Action` and `Events`; the `GameConfig`; the ship
    card; steps; commit and dirty flag; file hashes), `weights.safetensors` and, when
    saved with one, `return_scale.safetensors`. `save_policy` and `load_policy` are the
    interface the other sections use.
  - **Training checkpoint:** a policy file plus the `TrainConfig`, the optimizer state,
    the league slot's weights, the rollout's whole carry (`Rollout.saved_carry`) and
    every generator, the device defaults included. Resume is bitwise on the tested cases.
  - **Settings:** resume uses the checkpoint's settings, and only the fields in
    `bnb.runs.checkpoint.OVERRIDABLE` may change (run length, precision, learning rate and
    warm-up, loss weights, `micro_batch_rows`, reward weights). Fork (`--from`) and warm
    start (`--init-from`) take the code's settings. A commit that differs from the
    previous segment's is a printed warning; `run.json` records each segment's commit,
    dirty flag and overrides.
  - **Run directory:** `runs/<name>/` with `run.json` and `checkpoints/`. The name is
    `<date>-<preset or mode>-<adjective>-<noun>`. Retention keeps the newest 8, one
    checkpoint per doubling band of age by ordinal, and the final one.
  - **Uploads** are explicit (`bnb runs upload`), to the W&B project
    `boost-and-broadside-v2`; nothing leaves the machine by default.
  - **Open (section 11 notes):** there is no game version constant yet (files record the
    `GameConfig` and card); the game's damage sum uses float atomics, so a long GPU run's
    resume can differ by rounding after a tick with two hits on one ship.
- **Hits (fix/seat-symmetry, merged 2026-11-02; the closer pass is the hit scored, author 2026-11-02):** a bullet's contacts are ordered by
  `bnb.game.physics.combat.contact_order`: the hit time τ, then the closer pass (the
  smaller miss distance), and only an exact tie in both by ship index. Kill credit uses
  the same order, then the bullet index. The index-first rule favoured team 1 (team-major
  storage with friendly fire). A mirror test in both modes guards it
  (`tests/sections/test_seat_symmetry.py`).
- **The scripted agent and BC (sections 4 and 7, merged 2026-11-02):**
  - **`TrainConfig`** gains `objective` (`bnb.runtime.train.OBJECTIVES`: PPO or BC) and
    `teacher` (`bnb.learn.bc.TeacherConfig`: the labels' temperature and floor, default
    floor `bnb.learn.bc.FLOOR`, 0.01 over the 42 joint actions). `SCRIPTED` names the
    scripted agent's controller slot.
  - **`BehaviourCloning`** is a `Learner`: it overrides only the policy terms and the
    report. The labels are computed inside the compiled loss from the stored
    observations (`bnb.controllers.scripted.distribution`); the rollout records what the
    teacher reads through `extra_reads`.
  - **The hand-off is a policy file:** `bnb train --objective bc` saves the clone at
    `runs/<run>/policy/`, and RL starts from it with `--init-from`. A BC run refuses
    `--resume`, `--from` and `--init-from`.
  - **Evaluation:** `bnb.runtime.evaluate` (`first_results`, `score`, `shares`); seats are
    drawn at random per game.
  - **The scripted agent** reads per-type characteristics measured from the ship card
    (`bnb.game.handling.characteristics`).
  - **The score stop (fix/bc-stop, merged 2026-11-02):** BC stops when the live score's
    lower bound (the window's score less two of its standard errors) reaches 0.40, scored
    against `bnb.learn.bc.TeacherOpponent`: the scripted agent playing the labels'
    distribution (their temperature and floor). PPO's scripted seat stays the plain agent.
- **Yemong's kernels and bf16 (perf/fast-yemong, merged 2026-11-03):**
  - **Five Triton kernels:** the scan, attention with rotary, the gated MLP, the temporal
    mix (training segments only) and the heads. Each is a `torch.library.triton_op` with
    autograd. Each has v2's PyTorch code as its plain twin behind a switch
    (`bnb.models.yemong.kernels.switches`), with a test that the two agree. The weights'
    names and shapes are unchanged, and so is `VERSION`.
  - **Every kernel is bitwise repeatable** on identical inputs on one machine: no atomics
    and no dependence on launch order. Tiles autotune per shape and GPU. The choice is not
    in checkpoints, so a resume on another machine drifts by rounding (author,
    2026-11-03).
  - **`TrainConfig.precision`** defaults to `"bfloat16"`. The matmuls run in bf16. The
    residual stream, norms, softmax, the recurrent state, codes, belief, critic
    expectations, every loss, the weights and the optimizer stay fp32. The update logs
    the first epoch's ratio drift, the clip fraction, `nonfinite_gradients` and
    `nonfinite_beliefs`.
  - **Comparisons between model types** run both plain or both with kernels, never one
    of each.
- **The segment store's staging (fix/intermittent, merged 2026-11-03):**
  - **The cause:** `SegmentStore.batches` allocated a new micro-batch size's staging
    buffers on its worker thread while the caller computed. Kernels still queued on the
    compute stream could then overwrite a copied batch. It allocates every slot it needs
    on the caller's thread before any copy, and records each slot's release after the
    allocation.
  - **Callers run the generator out:** an abandoned one never releases its last batch.
  - **The guard** is `tests/sections/test_store_staging.py`.
  - **Open:** the 26-ship update is not bitwise repeatable between processes. Losses
    differ by up to 5e-7 relative, while the rollout is identical. The probable cause is
    an atomic add in a compiled backward. The 2v2 exact-resume test is unaffected.

- **The cleanup pass (section/cleanup, 2026-11-03):**
  - **Renames:** `Outputs.logp` and `Trajectory.logp` are `log_probability`;
    `GameConfig` and `Pool` take `episode_seconds` and `respawn_seconds`. Neither
    `Outputs` nor `Trajectory` has a schema version (neither is saved), so none rises.
  - **`reset` and `done`:** `reset` is the environment resetting before an observation
    (`Controller.step(carry, obs, reset)`, the observation's `reset` column, Yemong's
    `step` and `unroll`); `done` is an episode ending after a step (`Events.done`,
    `Trajectory.done`).
  - **`GameConfig.num_teams`** is gone: `TEAMS` (2) is one constant in
    `bnb.contracts.config`, which `bnb.game.rules` imports.
  - **Shapes:** `Z` zone slots and `C` spawn candidates join the letters; the reward levels
    are written `levels` (`L` is a segment's steps).
- **The league (section/league, merged 2026-11-03):**
  - **The seat:** `bnb.runtime.opponents` holds 4 network slots (stacked parameters plus
    a spare set for the drain) and the stationary ladder as one controller, all inside
    the captured rollout. The slots' forward runs each set in turn in one graph, not
    vmap: under vmap, attention skips autocast's casts on the plain bf16 path.
  - **The default lineup** is self-play 0.5 and the league 0.5. The lagged `league_0`
    copy and `league_every` are gone.
  - **A drain** starts only when none is running.
  - **A slot's entry** is drawn from the modes' weights, mixed by their share of the
    opponent rows.
  - **A snapshot** is taken from the weights that played its update, and that update's
    games are credited to it.
  - **The training checkpoint** holds the slots and the league's state, and embeds the
    snapshots' and averages' weights. Resume is exact with the league attached.
  - **Metrics** are under a `league` key in each update's row. `bnb ratings` and
    `bnb head-to-head` play the offline matches (`bnb.league.matches`,
    `bnb.league.tournament`).
  - **The GPU resume test** can differ in the last bits when its processes autotune from
    cold. The attention, temporal-mix and heads kernels have configurations whose bits
    differ, and the first tuning picks one by timing.
- **Model settings in `TrainConfig` (fix/hygiene, 2026-11-03):** `TrainConfig` holds
  `yemong` (`YemongConfig`), `relational` (`RelationalConfig`) and `ship_card` (the card's
  values by field, one per type, every field given; `bnb.game.api.default_card`);
  `relational_hidden` is gone. `model_build(config)` builds from them, and their
  `bfloat16_matmuls` must stay False (`precision` sets it). `Rollout` and
  `first_results` take the card (`types`) as their second argument; the `Trainer` passes
  `config.ship_types(device)`. A resume refuses a changed setting (none is in
  `OVERRIDABLE`); a fork or a warm start checks the file's build record against the
  run's (`check_build`) and, when the command sets no model setting, takes the file's
  (`with_model_of`). A run directory written before this cannot be resumed (its
  `config.json` and checkpoints have `relational_hidden` and no `yemong`).
- **Resume: integrity, not bits (author, 2026-11-03):** a resumed run must continue the
  same run (the same state, generators, league and schedule) within rounding. Bit-for-bit
  equality is not required. On the CPU it costs nothing and its tests stay bitwise. On
  the GPU, last-bit differences from autotuning, atomics or another machine are accepted,
  and the GPU resume tests compare within a float tolerance tight enough that state not
  restored still fails. Exactness is revisited only if a bug needs it.

## Decision: no bullets in the observation (author, 2026-10-31)

`Obs` has `ships`, `fields` and `globals` only. Reasons:
- models do not read bullets;
- a firing ship is always visible, so its bullets would be too;
- shot labels are rebuilt learner-side from stored truth, and they use ship visibility.

This supersedes the decision log's "raw observation includes bullets". The sight test no
longer pairs bullets with observers. The race reference tests lost their bullet
observation checks (`test_bullet_visibility` and the `obs.bullets` comparisons).

## Skeleton implementation notes (2026-10-31, revised by the cleanup pass)

Milestone 0 settled these points of the text above; the rest of its notes were superseded
by the sections and are in "Current schema".

- **Controller signature:** `step(carry, obs, reset) -> (action, carry, outputs)`.
  - Following PyTorch idiom, a model's parameters live in the controller's own module.
    There is no `params` argument; a league slot is a separate controller whose weights
    are `copy_`'d in.
  - `reset` is the view's reset flag (the environment reset before this observation).
  - `reads` lists the Obs parts a controller uses; the rollout stores only the learner's.
- **Perception:** `perceive(rules, state) -> (state, Obs, Privileged)`. "When did this
  team last see this ship" is game state (`Match.ticks_since_seen`, ticks, `(B, T, N)`), so
  perception returns the state with it advanced.
- **Seat:** a runtime array `seat (B,)`, redrawn at every reset. Role r plays team
  `(r + seat) mod T`.
- **Constants:** sizes and the two time-native rules (`episode_seconds`,
  `respawn_seconds`) are in `GameConfig`; the game's other numbers are in `bnb.game.rules`
  (`Rules`, `FlightConstants`) and the ship card.
