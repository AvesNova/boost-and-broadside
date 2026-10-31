# Contracts (section 0)

Status: **draft for review**, 2026-10-31.
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
- **entities are ordered by `team_rel`**: the viewer's own ships come first. This is a
  reordering, not geometry. It is needed because a fixed sim order would leak the seat
  (team 1 would always find itself in the second block). Ships keep their slot order
  within a team.
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
    damage_dealt: Tensor     # float: hit damage this ship's bullets did
    damage_taken: Tensor     # float: hit damage this ship took
    boundary_damage: Tensor  # float
    died: Tensor; fired: Tensor; spawned: Tensor    # bool
    killer: Tensor           # int: ship index of the killing shooter, −1 if none
    done: Tensor             # (B,) bool: the env reset after this tick
    SCHEMA_VERSION = 1
```

- **new:** the race's pairwise `damage (B, N, N)` matrix is **not** in the contract.
  - At N = 100, B = 1024 it is 40 MB per tick, and 5 GB if a 128-step rollout stored it.
  - Per-ship dealt and taken, plus `killer`, cover every reward and diagnostic we have.
  - A mode that needs pairs (an assist rule, say) adds a field when it arrives.
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
- **new:** a controller acts for **one team view** at a time. It receives `Obs` sliced to
  its view, `(B_pool, …)`, and returns actions for that team's `S` ships. Controllers
  never see other views.
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
reward(mode, events, state) -> Tensor        # modes section
snapshot(state) -> RenderSnapshot            # play section
```

- **new:** `observe` is separate from `step`, so the runtime can cut stages between them.
  The race tested `step` returning the observation; the arm split them internally anyway.
- **new:** randomness enters through `gen`. The correctness surface for tests also accepts
  explicit draws, as in the race: spread normals and reset uniforms.

## Open, for review

1. Action contract: factored components (recommended) vs joint 42.
2. Ordering observation rows by `team_rel` (needed to hide the seat; it is a reordering
   only).
3. Dropping the pairwise damage matrix from `Events`.
4. `Match.last_seen` and `seat_of_team` as new state.
5. A controller acts for one team view (no cross-view controllers).
6. Learner seat randomized per env per episode within a pool.
