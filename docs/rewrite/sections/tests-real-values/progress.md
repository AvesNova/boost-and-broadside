# Tests read the code's values: progress

Branch `tests-real-values` (worktree `../v2-tests`, from `core` at 2b3e8828). The rule and its
classes: `lean/README.md`, "Tests read the code's values".

## 2026-11-25

- Tried: two perturbations of `core`, the quick tier on each (CPU, scratch worktrees, never
  committed). A: vision 1,500 -> 1,700 m, the P-51D's shaft power 2,200 -> 2,300 hp, gun range
  686 -> 640 m. B: arena 4,200 -> 4,800 m, ramp 300 -> 400 m, warm-up 2,000 -> 1,000, league
  floor 0.125 -> 0.1, league decay 0.99 -> 0.98, self-play share 0.5 -> 0.25, MLP ratio 4 -> 3.
  Core: A breaks 3 tests (one the known scripted failure), B breaks 19 (one the known).
- Worked: a helper `lean/game/tests/values.py` (`game_value`: any of core.ship_constants'
  numbers); game, tool, model, learn tests read the card, tuning tables, GameConfig, contract
  sizes, ModelConfig and LearnConfig. The learn tests pin their seat layout
  (`self_play_share=0.5` in `setting.SMALL` and the store tests' setting): a scenario input, so
  the default share can move.
- Found: the model's `rotary_reach` (16,800 m) is a copy of 4 x the arena radius; the rotary test
  used a hard-coded 4,200 m arena, so it passed under a 4,800 m arena while the coarsest band no
  longer covers the map. It now reads GameConfig and fails there, as it should.
- Not edited (another agent's branch): `lean/scripted/tests/`; its cases are listed in the
  final report.
- Measured on this branch (quick tier, CPU, 2026-11-25 15:05-15:12): unperturbed 1 failure (the
  known scripted one); A 1 (the same); B 2 (the same, and test_rotary_bands, which now catches the
  rotary reach no longer covering a 4,800 m arena: a real coupling, `ModelConfig.rotary_reach`
  should follow the arena, an author decision). Core: A 3, B 19.
