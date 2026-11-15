# Final freeze check: progress

Branch `check/freeze-final` (from `feat/imugi-boundary` a8919b27). Notes in order.

## 1. vizia out of the real-scale set (2026-11-15)

- **Tried:** removed `vizia` from `order.toml` and deleted `vizia.toml`; updated the tests,
  two bench defaults and the docs that listed it.
- **Worked:** Yongma's and Jujak's files hold absolute handles; their notes keep the values
  taken from vizia ("vizia's 162.7 x 1.0292"), so nothing reads `vizia.toml` to build or
  solve them. The derivation's table stays in `archetypes.md` section 4; the file itself is
  in git before this commit. The game-version reference still matches (vizia was type 2;
  the default lineup only flies types 0 and 1), so **no version bump** for this step.
- **Found and fixed:** Imugi chose its stored doctrines by the card's type count alone. With
  vizia gone the real-scale set has two types, like the standard set, so a standard card
  would have flown Yongma's and Jujak's doctrines (`test_imugi_memory` caught it). The
  lookup now matches the type count and the scale (`real_scale_card` of the card's gravity):
  `imugi_doctrine.for_card`, `rows_by_kind`, `ImugiStep.table` (a `torch.where` on the
  gravity, as the step is exported). Real-scale Imugi flies the same doctrines as before.
- **Note:** the real-scale card now has 2 types instead of 3, so anything sized by the type
  count (Yemong's card input) differs from a model saved before this change.
- Two quick-tier failures exist on `feat/imugi-boundary` before this change and are not
  touched here: `test_ship_sets.py::test_training_takes_its_games_set_card` (a set without
  drawn lineup types) and `test_play_render.py::test_an_edited_colour_draws_on_the_next_frame`.
