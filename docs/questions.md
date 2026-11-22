# Questions for the author

Open questions from the lean-core work on `core`. Each has the date it was raised and what
it blocks. Answered questions move to the bottom with the answer and its date.

## Open

(none)

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
