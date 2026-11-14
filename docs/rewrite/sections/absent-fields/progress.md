# Absent fields: progress

Branch feat/absent-fields from origin/feat/seed-isolation. CPU only.

## Plan (00:55 KST)
- Game: `Fields.present` (B, M) in the state; an episode's active field count drawn from
  one extra tick uniform (keyed by seed, environment, episode in the rollout) where the
  game's `ActiveFields` range is set; absent fields inert in the medium and the sight test;
  the observation's field rows gain `present` (absent rows zero).
- Model: Yemong masks absent field tokens as keys (the benched-ship key mask path); the
  nearest model skips them.
- Plumbing: `GameConfig.active_fields` / `Pool.active_fields`, `TrainConfig.active_fields`,
  the ratings' `--active-fields`.
- ETA game step 02:00, model 03:00, plumbing + docs 04:00 (worst 05:00).

## Log
### 01:50 game, model, plumbing, tests
- Tried: absent geometry made inert in the medium (band 1 m, radius minus the band: the
  profile is exactly 0, the complement exactly 1) rather than a mask over (B, P, M), so the
  hot path pays two (B, M) selects. The sight test's core and outer radii are 0 when absent.
- Worked: tests/sections/test_absent_fields.py, 16 tests: an absent field laid over the
  ships flies, shoots and observes as the same game without that pair (within rounding);
  present, it changes the game. Sight, rows, the draw (uniform over the symmetric counts),
  keyed by (seed, environment, episode) in the rollout, no other draw moves, the settings,
  Yemong step and unroll read nothing of an absent row with the mask on (and do with it
  off), the nearest model skips it.
- Didn't: nothing failed after two test fixes (the "can tell" checks needed the field on a
  ship, and an unspawned ship for the sight test).
- Next: the related suites, the game fingerprint (new leaf + column: version 20), docs.

### 03:05 suites, fingerprint, docs
- Worked: the related quick suites pass (game section, map sampling, modes, mixed, Yemong
  inputs, crossover, nearest, ship card, training lineup, contracts, keyed draws, race
  rules, perception, play snapshot, learner shots, play replay, figures, docs staleness):
  300+ tests. Fixed fixtures that built 5-column field rows (nearest, learner shots,
  bench/yemong_synthetic.py).
- Found: the base branch's game fingerprint was already stale (7f35d79ccfe509be against
  the recorded e16ea10c244756ab; the keyed draws moved it). Without the new leaf and column
  this branch computes exactly the base's numbers; VERSION 20, FINGERPRINT 477cbfae2f748b1e.
  Tournament records of version 19 are therefore not reused.
- Didn't: tests/docs/test_docstrings.py fails on bench/attention_shares.py (base, not this
  branch). No GPU check of the fused attention kernel with the longer key mask (CPU only;
  the plain path is tested, and the kernel already takes masks over any first K keys).
- Next: none here; the arm sets `TrainConfig.active_fields`.
