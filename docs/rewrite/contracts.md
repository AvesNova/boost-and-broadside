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
| Shapes | Envs `B`, teams `T` (2 for now, equal sizes), ships per team `S`, ships `N = T·S`, fields `M`, bullet slots per ship `K` | carried |
| Ship order | Team-major: ship `n = team·S + slot`. Ship tensors are stored `(B, N, …)` and viewed `(B, T, S, …)` for free | new |
| Floats | float32 everywhere in game and runtime. float64 only in `reference/`. bf16 only inside models | carried |
| Ints | int32 counters and categoricals; bool flags; int64 only where torch requires it (indexing), converted at the use site | carried |
| Ship indices | At most 255 ships (author: no plans beyond 256); stored ship indices (killer, per-bullet target, row maps) may be `uint8` with 255 = none. Torch treats a `uint8` index tensor as a boolean mask, so convert to int64 before indexing | new |
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

## Skeleton implementation notes (2026-10-31)

Milestone 0 is built (`src/bnb`). These are the places where it differs from or settles
the text above:

- **Controller signature:** `step(carry, obs, done) -> (action, carry, outputs)`.
  - Following PyTorch idiom, a model's parameters live in the controller's own module.
    There is no `params` argument; a league slot is a separate controller whose weights
    are `copy_`'d in.
  - `done` is the view's reset flag.
  - Random draws use the default generator for now. Named, graph-registered generators
    come later.
  - `reads` lists the Obs parts a controller uses; the rollout stores only the learner's.
- **Perception:** `perceive(p, state) -> (state, Obs, Privileged)`. "When did this team
  last see this ship" is game state (`last_seen`, ticks, `(B, T, N)`), so perception
  returns the state with it advanced.
- **Seat:** a runtime array `seat (B,)`, redrawn at every reset. Role r plays team
  `(r + seat) mod T`. `Match.seat_of_team` (map sides) is not needed by team deathmatch
  and arrives with the first mode that has side-specific roles.
- **Known seat leak, physics section:**
  - The race spawn rule places ship i at phase `frac(0.618 i + 0.382 t)` using the
    sim index.
  - So team 0's slot 0 always spawns due east of its team's centre in world coordinates,
    and a reset reveals the seat.
  - Fix: phase by slot within the team, measured relative to the spawn axis. That
    changes the race goldens.
  - `tests/sections/test_perception.py::test_spawn_layout_does_not_reveal_the_seat` is a
    strict xfail until then.
- **Physics constants:** sizes and the two time-native rules (`episode_s`, `respawn_s`)
  are in `GameConfig`. Everything else is still the race's `presets.json` until the
  physics section.
- **Events** carry the race's per-tick pairwise damage. The stub reward uses only kills,
  deaths and damage dealt and taken. The attribution ledger arrives with the modes
  section.
