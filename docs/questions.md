# Questions for the author

Open questions from the lean-core work on `core`. Each has the date it was raised and what
it blocks. Answered questions move to the bottom with the answer and its date.

## Open

9. **Boundary ramp (2026-11-23).** The boundary ramp is the distance beyond the arena edge
   over which boundary damage rises from zero to its maximum. v2 scales it with the map
   (100 m times the map scale: 161.5 m at a 4,200 m arena); lean fixes it at 300 m (a
   builder's assumption). Proposal: v2's rule. Blocks: nothing urgent.

## Answered

1. **Fields in `lean/game` (2026-11-22).** The author says fields are the default, but
   `GameConfig.fields` defaults to none and no field map is defined yet. The game also
   supports only one fixed set of fields for the whole run (a constant of the run), where
   v1 drew from 512 generated field maps per episode. Which should stage 1 use: (a) v2's
   elimination map's fields, fixed; (b) fields drawn per episode, as v1 did; (c) none for
   stage 1, fields with the other modes? (b) needs per-environment fields in the state
   (E, F, 5) and makes the medium a per-environment evaluation. Blocks: the default
   `GameConfig`, the step-4 comparison of fields against v2, and the field-path
   operation count.

   **Answered 2026-11-22:** fields are randomised for every new environment, as (b). How
   to do it efficiently is in `lean/game/NOTES.md`, agreed fix 11.

2. **Fog: when is a ship visible to the enemy? (author, 2026-11-22)** A ship is visible to
   the other team when it is in an enemy's sight, on every tick it fires, on every tick
   it is hit, and when it dies. Every ship is always visible to its own team. The
   observation should be built towards fog from the start, not as full information to be
   replaced later. The scripted opponent reads its team's view, never the full state.

3. **Field tokens (author, 2026-11-22).** Fields must be tokens in the observation; their
   absence from `lean` is a bug. v2 has them (checked: `src/bnb/models/yemong/network.py`,
   and `inventory.md` rows 2.3 and 3.2) and so did v1. The layout to keep is v2's: ships
   and the game token are query tokens (full self-attention, updated, with recurrent
   state); fields and zones are memory tokens (keys and values only, never queries, not
   updated), in the same softmax of the same attention operation. Each kind has its own
   first encoder layer and all share the second. Bullets are not tokens.

4. **Tell, do not infer (author, 2026-11-22).** The mode, the ship type and anything else
   known at reset are given as inputs and stored, not inferred. An environment's mode is
   fixed for the whole run.

5. **Fog: what else reveals a ship, and for how long (2026-11-22).** Does boundary damage
   reveal a ship? Is a reveal for that tick only?
   **Answered 2026-11-22:** boundary damage does not reveal a ship. A reveal lasts only
   for the tick the ship fires or is hit by a bullet. A dead ship that will respawn is
   visible; a ship dead for good (elimination) is masked out.

6. **Both sides training in snapshot games (2026-11-22).** Withdrawn: PPO trains on the
   current policy's actions only, so a snapshot's side never trains. The real gap was the
   missing self-play; see 8.

7. **Reward scaling with team size (2026-11-22).** **Answered 2026-11-22:** keep the
   current scaling; not a concern for now.

8. **The league (author, 2026-11-22).** Half the environments are self-play (the current
   policy on both sides, both sides train) and half are league games (the current policy
   against an opponent from the league, the live side trains). The scripted agent is one
   member of the league, not a fixed half. `lean/learn` today has a fixed half against
   the scripted agent, a half against snapshots, and no self-play: a departure to fix.

