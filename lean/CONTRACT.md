# Lean core: interfaces

First cut of the lean core, written on the CPU before the framework gate (G1) closed.
PyTorch 2.15, plain eager code that also compiles. Nothing here has run on a GPU. The
plan is `docs/rewrite/core-brief.md`; the reasons behind the model are in
`docs/rewrite/model-audit.md`, the game in `docs/rewrite/sections/game-audit.md`, v1's
recipe in `docs/rewrite/sections/yardstick.md`.

Four packages are written in parallel against this file. Names and shapes here are fixed;
anything a package needs beyond them goes in that package and is recorded in its
`NOTES.md`.

Sizes: `E` environments, `N` ship slots (both teams together, the maximum for the run),
`K = 11` bullet slots per ship, `T` ticks. House layout: batch first, features last,
contiguous.

## `lean/contract.py` (owned by the game package)

```python
STICKS, POWERS, FIRES = 7, 3, 2          # joint action = stick * 6 + power * 2 + fire, 42 values
JOINT = 42
def split_joint(joint): ...              # (...,) long -> (..., 3) long [stick, power, fire]
def join_action(action): ...             # (..., 3) long -> (...,) long

@dataclass
class State:                             # every field a tensor; shapes (E, N) unless noted
    x, y, heading_cos, heading_sin, speed, angle, angle_cos, angle_sin, nose_rate, energy,
    shield, shield_wait                  # float32; metres, radians, seconds
    cooldown, next_slot                  # long
    alive, exists                        # bool; exists is fixed for an episode
    team                                 # long, 0 or 1, any assignment of slots
    type                                 # long, index into the ship card
    constants                            # dict[str, Tensor], per-ship constants gathered at reset
    bullet_x, bullet_y, bullet_dir_x, bullet_dir_y, bullet_speed   # (E, N, K) float32
    bullet_moves_left                    # (E, N, K) long; 0 is inactive
    tick                                 # (E,) long

@dataclass
class StepOut:
    damage_dealt, damage_taken           # (E, N) float32, this tick, enemy damage only in dealt
    friendly_damage                      # (E, N) float32 dealt to team-mates
    died, fired                          # (E, N) bool
    killer                               # (E, N) long, slot of the killer or -1
    done                                 # (E,) bool
    result                               # (E,) long: 0 or 1 = winning team, 2 = draw, -1 = not done

@dataclass
class Obs:                               # one team's view; full information (no fog yet)
    ship                                 # (E, N, SHIP_FEATURES) float32, scaled scalars and flags
    pos                                  # (E, N, 2) float32 metres, arena centre at 0
    heading                              # (E, N) float32 radians
    key_mask                             # (E, N) bool: this row may be attended (exists and alive)
    own                                  # (E, N) bool: rows this view controls (own team, alive)
    globals                              # (E, GLOBAL_FEATURES) float32
SHIP_FEATURES, GLOBAL_FEATURES           # ints, set by the game package
```

## `lean/game` (game package)

```python
@dataclass
class GameConfig: ...                    # arena radius, clock, spawn rule ("sides" | "mixed"), fields, ship lineup
def reset(config, envs, team_sizes, generator, device) -> State
    # team_sizes: (E, 2) long, ships per team in each environment; unused slots have exists = False
def step(config, state, action, noise) -> tuple[State, StepOut]
    # action (E, N, 3) long; noise: dict of the random draws the tick needs, from draw_noise
def draw_noise(config, envs, slots, generator, device) -> dict[str, Tensor]
def reset_done(config, state, done, team_sizes, generator) -> State   # re-deal finished environments
def observe(config, state, team: int) -> Obs
```

Elimination with no shrink on one fixed map: flight, fire and bullets, the swept hit test,
damage, shields, the boundary, death as a mask, a win check and a clock. No respawn.

## `lean/scripted` (scripted opponent)

```python
@dataclass
class ScriptedConfig: ...
def personas(config, envs, slots, generator, device, spread: float) -> dict[str, Tensor]
    # per-ship parameters drawn at reset; spread 0 gives the one fixed yardstick setting
def act(config, state, team: int, persona, uniforms) -> Tensor   # (E, N, 3) long; rows of other teams are 0
```

## `lean/model` (policy)

```python
@dataclass
class ModelConfig: ...                   # width, blocks, heads, levels 5, bins 64, position_code, ...
class Policy(nn.Module):
    def initial_carry(self, envs, slots, device) -> Tensor   # (blocks, E, 1 + N, 4, width)
    def step(self, obs: Obs, carry, position_offset=None, reset=None) -> tuple[Tensor, Tensor, Tensor]
        # logits (E, N, 42), value_logits (E, N, LEVELS, BINS), new carry
    def sequence(self, obs: Obs, carry, reset, position_offset=None)
        # obs fields carry a leading T; reset (T, E) bool zeroes the carry before that tick
        # returns logits (T, E, N, 42), value_logits (T, E, N, LEVELS, BINS),
        #         next_state (T, E, N, NEXT_STATE), hit_logits (T, E, N, 3) and the final carry
```

The model reads from `Obs`: ship, pos, heading, key_mask, globals, field, field_pos,
field_mask, visible, since_seen (`lean.model.policy.OBS_FIELDS`).

## `lean/learn` (rollout, PPO) and `lean/eval` (crossover)

```python
def collect(config, game_config, policy, opponent, state, carry, ticks, generator) -> Rollout
def update(config, policy, optimiser, rollout) -> dict[str, float]
def crossover(game_config, policy, scripted_config, sizes, games, generator) -> dict
```

Reward levels (index into LEVELS), v2's: 0 damage, 1 kill_death, 2 objective_progress,
3 objective, 4 outcome. Hit classes: 0 missed, 1 enemy, 2 team-mate.

## Rules for this first cut

- Read the plan's sections 3, 4 and 9 before writing: full-word names, docstrings that
  state the contract, comments only for a non-obvious reason, no history or opinion.
- No Python loops over ships, bullets or components. No per-row branches computed and
  then selected where a mask or a slice does it.
- Tests are few and name the failure they catch: invariants and one or two behaviours.
  They run on the CPU in seconds.
- Each package keeps a `NOTES.md`: what is implemented, what is untested, every choice
  made that the plan or the audits did not settle, and what to check first on the GPU.
