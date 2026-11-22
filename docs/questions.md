# Questions for the author

Open questions from the lean-core work on `core`. Each has the date it was raised and what
it blocks. Answered questions move to the bottom with the answer and its date.

## Open

1. **Fields in `lean/game` (2026-11-22).** The author says fields are the default, but
   `GameConfig.fields` defaults to none and no field map is defined yet. The game also
   supports only one fixed set of fields for the whole run (a constant of the run), where
   v1 drew from 512 generated field maps per episode. Which should stage 1 use: (a) v2's
   elimination map's fields, fixed; (b) fields drawn per episode, as v1 did; (c) none for
   stage 1, fields with the other modes? (b) needs per-environment fields in the state
   (E, F, 5) and makes the medium a per-environment evaluation. Blocks: the default
   `GameConfig`, the step-4 comparison of fields against v2, and the field-path
   operation count.
