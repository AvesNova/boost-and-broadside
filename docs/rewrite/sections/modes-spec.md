# Modes spec (section 3)

Status: spec, 2026-11-01. Decisions: `modes.md`. Fixtures: `tests/reference/rewards/`
(the Frontline part; README there).

Modes are game rules in `bnb.game`, pure like the rest of it. Where a rule is v1's, this
text states it and the fixtures check it. **(v2)** marks a change; the change is decided.
**(proposal)** marks a value to confirm with the author: implement it and list it in your
notes.

## 1. Configuration

- **`GameConfig` gains:**
  - `mode`: `"team_deathmatch"`, `"frontline"` or `"sandbox"`. Sandbox is today's game:
    no score and no win (only the clock), with game spec 8's formation spawns. The race
    reference tests and the ship-card tools (section 1b) run in it;
  - `spawn_groups`: a tuple of ships per team in each group; empty means uniform spawns.
    For example `(2, 2, 1)` at 5v5 is three groups (a 2v2, a 2v2 and a 1v1);
  - `kills_per_ship`: 5.
- **`Rules` carries the mode's rules** (static, as everything in `Rules` is).
- **`episode_s`** default is 300 s.
- **`respawn_s`** per mode: team deathmatch 5 s **(proposal**, Halo-like), Frontline 8 s
  (v1).
- **Training presets** (all 5v5):

| preset | mode | fields | arena radius | groups |
|---|---|---:|---:|---|
| `deathmatch` | team deathmatch | 10 | 2600 m | — |
| `deathmatch_wide` | team deathmatch | 40 | 8200 m | (2, 2, 1) |
| `frontline` | Frontline | 10 | 2600 m | — |

- **The training mix:** `deathmatch_wide` takes 25% of team deathmatch training
  **(proposal)**.
- **One game is one mode at one size,** compiled for its shapes. Training on several games
  at once (different shapes) is the learner section's concern. Here each game must run on
  its own, through the runtime as it is.

## 2. The tick (v2 additions to game spec 2)

A **mode phase** runs after damage, death and shields (phase 5) and before the clocks
(phase 6). It updates the mode state and decides whether the match is won.

- **Episode end:** `done = clock ∨ won`. The respawn and reset of phase 8 then run as now,
  with the mode's spawn placement (section 4).
- **The result**, for team 0 on done environments: +1 a win, −1 a loss, 0 a draw. It is 0
  where not done.
- **Tick uniforms grow** by the mode's spawn draws (section 4). Spec section 2's layout is
  extended at the end, and the counts are exact functions of `N`, `K` and the groups.

## 3. Team deathmatch

- **Score:** every death of a ship scores one point for the other team, whatever the
  cause: an enemy kill, friendly fire or the boundary. `score (B, 2)` lives in the mode
  state.
- **Win:** first to `kills_per_ship × S` (25 at 5v5). Both teams reaching it on the same
  tick is a draw. At the clock the higher score wins, and equal scores draw.
- **No objective:** `objective_progress` and `objective` are zero.

## 4. Spawns (v2, mode-owned)

Spawning keeps game spec 8's spawn state: nose, speed, energy, shield, waits and the
medium. Only the position and the heading come from the mode.

### Team deathmatch, uniform (no groups)

- **Candidates:** each spawning ship draws `K = 8` points, uniform over the arena disc
  (`r = R_a √u₁`, angle `2π u₂`), and one heading `2π u₃`.
- **Respawn:** take the candidate whose distance to the nearest enemy alive after phase 5
  is largest; ties go to the lower candidate. With no living enemy, take candidate 0.
- **Reset (all ships at once):** candidate 0 of every ship is its provisional point. Each
  ship then takes the candidate farthest from the nearest *enemy provisional point*.
- **One pass, symmetric:** nothing depends on the team index or the sim index.

### Team deathmatch, grouped (`spawn_groups`)

- **Group centres at reset:**
  - centre 0 is uniform in the disc of radius `0.85 R_a`;
  - each later centre `g` is the best of `K` uniform draws in that disc, by distance to
    the nearest earlier centre.
- **Members:** each team's ships are assigned to groups by formation rank (game spec 8's
  keys), `spawn_groups[g]` ships of each team to group `g`.
- **Positions at reset:** best-of-`K` points in the disc of radius 300 m around the
  group's centre, by distance to the nearest enemy provisional point (as in the uniform
  case, restricted to the group).
- **Respawn:**
  - choose an anchor: a uniformly random living ship other than the spawning one, of
    either team;
  - candidates are `anchor + d (cos θ, sin θ)`, with `d` uniform in [250 m, 600 m] and `θ`
    uniform;
  - take the best of `K` by distance to the nearest living enemy;
  - with no living ship, fall back to the uniform rule.
- **Heading:** random, as in the uniform case.

### Frontline

- **Position:** a uniform point in the disc of radius `0.35 × zone radius` around the
  team's *current* spawn zone, read when the wait ends.
- **Heading:** toward the zone the team attacks (the enemy's defense zone).
- **Protection:** a hit on a ship inside its own team's spawn zone deals no damage. The
  bullet is still spent. Boundary damage is unaffected.

## 5. Frontline (v1's rules; v2 geometry)

### Map (v2)

- **Ring:** five zones at distance `0.615 R_a` from the centre, radius `330 s_a` m, at
  angles `ψ + h · 2πk/5` for `k = 0…4`.
  - The rotation `ψ` is uniform.
  - The handedness `h = ±1` is drawn with probability ½ each.
- **Roles at front 0,** by `k`: neutral, team-0 spawn, team-0 defense, team-1 defense,
  team-1 spawn.
- **Mirror symmetry:** the map is mirror-symmetric about the axis through the centre and
  zone 0, the neutral zone.
  - Field `P + i` is field `i` reflected about that axis.
  - With an odd `M`, the last field lies on the axis, at a signed distance along it of
    `0.85 R_a (2u − 1)`, with its radius and level drawn as usual.
- **No spawn axis:** Frontline has none. The spawn axis uniform is unused.

### Rules (v1, exact; fixtures `frontline_*`)

- **Membership:** a living ship is in a zone when its distance to the centre is at most
  the radius.
- **Roles:** zone `z`'s role is the base pattern rotated by the unwrapped front `f`:
  `role(z) = base[(z − f) mod 5]`.
- **Meters:** only the two defense zones have meters.
  - `net = (team-0 alive inside) − (team-1 alive inside)`; the meter moves only on a
    strict majority.
  - Pressure is `H(|net|)`, where `H` is the harmonic number (`H₀ = 0`).
  - The meter moves `sign · H · h / capture_s` toward the attacker (the side not holding
    the zone), clipped to `[0, 1]`. With `capture_s = 10`, a lead of one captures in 10 s.
  - Non-defense zones read 0.
- **Attempt ledgers** `(B, Z, N)`, per zone and ship:
  - each tick, a rise `δ > 0` credits every present attacker `δ / n_present` and debits
    every absent defender `δ / n_absent`. Dead defenders count as absent; if no defender
    is absent, every defender is charged;
  - a fall scales both ledgers by `new / old` progress;
  - a completion zeroes all ledgers, as does an empty meter.
- **Completion:**
  - a meter reaching 1 captures;
  - each side's share is half its presence on the completing tick and half its ledger
    over the attempt: gain for the attackers, loss for the defenders, each side summing
    to 1;
  - the front moves +1 for a team-0 capture and −1 for a team-1 capture, so both on one
    tick cancel;
  - every meter zeroes and the roles rotate.
- **Win:** `front ≥ 3` (team 0) or `front ≤ −3` (team 1). At the clock the sign of the
  front decides, and 0 draws.
- **Spawn protection:** section 4.

### Objective events (v2 interface; v1 attribution)

Per ship per tick, each summing to zero over the environment:
- `objective_progress`: the meter movement attribution (v1's zone-progress events, payout
  ratio 1). The side the movement favours splits `|δ|` among its ships inside the zone,
  and the other side's ships outside it are charged `|δ|` (all of them, if none is
  outside). Units are meter fractions. A completing tick pays nothing here.
- `objective`: the capture shares, `gain − loss`, in captures.

## 6. Events and state (v2)

- **`Events` gains:**
  - `shield_lost (B, N)`: the shield actually lost this tick, `S⁰ − S` after damage, for
    ships alive at tick start;
  - `shield_restored (B, N)`: the recharge applied this tick;
  - `objective_progress (B, N)` and `objective (B, N)`;
  - `result (B,)` int: +1, −1 or 0 for team 0, nonzero only where `done`.

  `damage` stays raw.
- **Mode state** is part of the game's opaque `State`:
  - the score;
  - the front, zone positions and radii, roles, meters and attempt ledgers;
  - the handedness and the ring rotation.
- **Shapes:** a team deathmatch state carries zero zones.

## 7. Observation (v2)

Everything is team-relative, like the rest of the observation.

- **`Obs` gains `zones (…, Z, len(ZONE_COLUMNS))`, with columns:**
  - `x`, `y`, `radius`;
  - `role`: 0 own spawn, 1 own defense, 2 neutral, 3 enemy defense, 4 enemy spawn, for
    the viewer;
  - `progress`;
  - `direction`: +1 when the meter is moving toward the viewer's capture, −1 toward the
    enemy's, 0 still;
  - `offensive_distance` and `defensive_distance`: captures to the viewer's win and to its
    loss, from v1's terminal distances.

  Team deathmatch has `Z = 0`.
- **`GLOBAL_COLUMNS` gains** `mode_team_deathmatch`, `mode_frontline` (a one-hot),
  `score_own`, `score_enemy`, `score_target`, `front` (positive favours the viewer) and
  `front_threshold`. Absent quantities are 0.
- **Yemong:**
  - zones become a token kind, key/value memory like the fields. The inputs are the
    position code, `radius / 1000 m`, the role one-hot (5), progress, direction, and the
    two distances through `symlog`;
  - the new global columns go on the global token, the scores divided by the target;
  - extend the encoder by a kind. The equivalence tests are kind-generic and stay as they
    are.

## 8. Acceptance

1. `tests/reference/rewards` (the `frontline_*` and `levels_*` cases the modes own) pass
   through `rewards_arm.py`.
2. **Team deathmatch:**
   - win, draw and clock cases;
   - a death of any cause scores for the other side;
   - spawns: candidates are drawn uniformly, the chosen one is the farthest, the reset
     pass is symmetric, and the seat is not revealed (extend
     `test_spawn_layout_does_not_reveal_the_seat` to every mode);
   - grouped spawns: centres spread out, the group compositions are right, and respawns
     land 250–600 m from an anchor.
3. **Frontline:**
   - mirror symmetry of the map (fields, zones, roles at front 0) about the neutral axis;
   - both handednesses occur;
   - spawn protection;
   - a scripted capture end to end through the real game, with the front, roles, win and
     timeout.
4. **Objective events:** zero-sum on every tick, and `objective` sums over an episode to
   the number of captures.
5. **Shield events:** `shield_lost` matches the shield change on damage ticks, and
   `shield_restored` on recharge ticks, through deaths, respawns and resets.
6. **The tick still works** eager and compiled, captured, at any batch size, with fresh
   randomness. The race reference tests still pass, through their adapter, in sandbox
   mode, unchanged.
7. **Speed:** the benchmark cells with each mode. Report the mode phase's cost.
