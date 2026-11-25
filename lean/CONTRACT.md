# Lean core: interfaces

The shared types and the calls between the packages, as the code has them. The types live in
`lean/contract.py`; each package's notes (`NOTES.md`) say what is behind its calls. PyTorch 2.15,
plain code that also compiles. A change to a name or a shape here is a change to every package.

Sizes: `E` environments, `N` ship slots (both teams together, the largest total dealt), `K = 12`
bullet slots per ship, `F` fields, `T` ticks. Layout: batch first, features last, contiguous.
Units: metres, radians, seconds, ticks where stated.

## `lean/contract.py`

```python
STICKS, POWERS, FIRES = 7, 3, 2  # joint action = stick * 6 + power * 2 + fire; JOINT = 42


def split_joint(joint): ...  # (...,) long -> (..., 3) long [stick, power, fire]
def join_action(action): ...  # (..., 3) long -> (...,) long


NO_PENDING = JOINT  # Obs.pending of a row whose pending action is not observed
LEVEL_NAMES = ("damage", "kill_death", "objective_progress", "objective", "outcome")
LEVELS = 5
SHIP_FEATURES, GLOBAL_FEATURES, FIELD_FEATURES  # 18 scalars + 4 flags + type one-hot; 3; 3
BULLET_SLOTS = 12
```

Also here: `symlog`, `map_fields`, `whole_state`, `compile_dynamic` (the one place duck sizing is
turned off), `random_keywords` (a factory's generator keywords; compiled graphs draw from torch's
default generator).

Stick: 0 straight, then (1, 2) gradual, (3, 4) middle, (5, 6) full, one direction odd and the
other even. Power: 0 coast, 1 dive, 2 climb. Fire: 0 or 1.

**`Constants`**: per-ship constants of one run, read by name (`constants["lift"]` or
`constants.lift`). A value equal for the whole lineup is a Python number; one that differs by
ship type is a row of a table `(E, C, N)` gathered at the deal. `lean.game.core.ship_constants`
builds it.

**`State`**: the whole game state, packed by block so per-field work is one operation per block.
Every quantity named in the `*_COLUMNS` tuples reads by name as a view, `state.x` being `(E, N)`
and `state.bullet_x` `(E, N, K)`; `state.replace(name=...)` replaces one.

| Block | Shape | Holds |
|---|---|---|
| `ships` | `(E, columns, N)` float32 | x, y, heading cos and sin, speed, angle of attack (and cos, sin), nose rate, energy (the store), shield, shield wait, cooldown, the cached medium (log index, gradient), velocity, next bullet slot |
| `flags` | `(E, 3, N)` bool | alive, exists (fixed for an episode), fired (a real bullet last tick) |
| `identity` | `(E, 2, N)` long | team (0 or 1), type (index into the ship card) |
| `bullets`, `bullet_counts` | `(E, columns, N, K)` | position, direction, speed, the medium; moves left (0 is inactive) and the tick fired |
| `bullet_real` | `(E, N, K)` bool | real bullet, or a ghost (no damage; labels the hit head) |

Beside the blocks: `constants`, `tick (E,)`, `fields (E, 5, F)` (x, y, radius, band width, target
ln n), `pending (E, N, 3)` long (the action the next tick applies), and with fog on `seen
(E, SEEN_COLUMNS, N)` (each ship as its enemy team last saw it) and `ticks_since_seen (E, N)`.

**`StepOut`** (one tick): `damage_taken (E, N)`, `died (E, N)`, `done (E,)`, `result (E,)` (0 or 1
the winning team, 2 a draw, -1 not done), `shot_outcome (E, N, K)` (-1 none ended, 0 missed, 1
hit an enemy, 2 hit a team-mate) and `shot_tick`, `hit_damage (E, N, N)` [attacker, target]
(real bullets' damage and collisions', each ship's collision damage credited to the other ship of
the pair), `boundary_damage (E, N)`, `shield_restored (E, N)`.

**`Obs`**: one team's view; under fog an enemy row is its last-seen values.

| Field | Shape | |
|---|---|---|
| `ship` | `(E, N, SHIP_FEATURES)` | scaled scalars and flags (no position) |
| `pos`, `heading` | `(E, N, 2)`, `(E, N)` | metres, arena centre at 0; radians |
| `key_mask`, `own` | `(E, N)` bool | row may be attended (exists, alive); rows this view controls |
| `globals` | `(E, GLOBAL_FEATURES)` | |
| `field`, `field_pos` | `(E, F, FIELD_FEATURES)`, `(E, F, 2)` | F may be 0 |
| `visible`, `since_seen` | `(E, N)` | seen this tick; symlog seconds since seen |
| `pending` | `(E, N)` long | own living ships' joint pending action, else `NO_PENDING` |

## `lean/game`

```python
GameConfig(...)                          # the core game by default (see lean/game/NOTES.md)
reset(config, envs, team_sizes, generator, device) -> State   # team_sizes (E, 2) long, each >= 1
deal(config, envs, slots, team_sizes, generator, device) -> State   # fresh states at a given N
take_deal(state, done, fresh, index=None) -> State   # re-deal the finished environments
draw_noise(config, envs, slots, generator, device) -> dict[str, Tensor]
step(config, state, action, noise) -> tuple[State, StepOut]
    # action (E, N, 3) long chosen this tick; the tick applies state.pending and stores action
    # as the new pending one (the one-tick delay, for every agent)
tick(config, state, action, generator) -> tuple[State, StepOut]   # draw_noise, then step
observe(config, state, team) -> Obs
```

## `lean/scripted`

```python
ScriptedConfig()                         # the yardstick's values; .replace(name=value)
personas(config, envs, slots, generator, device, spread) -> dict | None   # None at spread 0
fresh_memory(envs, slots, generator, device) -> dict[str, Tensor]   # carried by every caller
draw_uniforms(envs, slots, generator, device) -> Tensor              # (E, N, 3), one per head
act(config, state, team, persona, uniforms, memory) -> tuple[Tensor, dict]
    # (E, N, 3) long, rows of other teams 0 (team None: both teams in one call), new memory
decide(...)                              # act, plus the Marginals (the BC teacher's distribution)
chaser_act(...)                          # the memoryless floor
```

## `lean/model`

```python
ModelConfig(); Policy(config)
Policy.initial_carry(envs, slots, device) -> Tensor        # (blocks, E, 1 + N, 4, width)
Policy.step(obs, carry, position_offset=None, reset=None) -> (logits, value_logits, carry)
    # logits (E, N, 42), value_logits (E, N, LEVELS, BINS)
Policy.sequence(obs, carry, reset, position_offset=None)
    # obs fields have a leading T, reset (T, E) bool; returns logits, value_logits, next-state
    # logits (T, E, N, 6, BINS), hit logits (T, E, N, 3) and the final carry
next_state_target(config, state, next_state, step_out, tick_length) -> Tensor   # (E, N, 6)
```

How a network acts (shared by the learner and the evaluator), `lean/model/agent.py`:
`agent_step(forward, bf16, view, carry, offset, generator) -> AgentStep` (joint, logits,
value_logits, carry: the one agent tick of training's live policy, the league's networks and the
evaluation runner), `network(policy) -> Network` (a weightless template plus weights),
`network_step(network, view, carry, offset, generator) -> AgentStep` (agent_step with its
weights), `seat_view`, `sample_joint`, `reset_carry`, `precision`. The model reads these `Obs` fields: `lean.model.policy.OBS_FIELDS`.

## `lean/learn`, `lean/train.py`, `lean/eval`

```python
collect(config, game_config, policy, opponent, state, carry, ticks, generator, *, compiled, store, segment) -> Rollout
update(config, policy, optimiser, data, *, teacher_weight, cloning, step_index, generator, compiled) -> dict[str, float]
Evaluator(Store(path), GameConfig())     # ensure / adaptive / report / elo (lean/eval/NOTES.md)
```

The reward levels are `lean.contract.LEVEL_NAMES`; hit classes 0 missed, 1 enemy, 2 team-mate. The
BC teacher is `lean.scripted.decide`'s `Marginals` (stick, power, fire), the product of the three
over the 42 joint actions.

## Rules

Full-word names; docstrings that state the contract; comments only for a non-obvious reason. No
Python loops over ships, bullets or components. Tests name the failure they catch and run on the
CPU in seconds. Each package keeps a short `NOTES.md`: what it is, its entry points, its key
numbers, what is open.
