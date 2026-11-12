# Bug sweep progress (branch fix/bug-sweep, from integrate/next-runs 9f690a7b)

CPU only; quick tier on the touched files. Worktree runs need `PYTHONPATH=<worktree>/src`
(the shared venv's editable install points at the main checkout).

| # | Failure | Cause | Fix |
|---|---|---|---|
| 1 | test_deathmatch_respawn_takes_the_candidate_farthest... | Stale test: since game v4 the heading is edge-safe (`spawns._heading`), the test expected the plain uniform angle. Code right. | Test computes the spec's heading from the chosen point. |
| 2 | test_shield_events_match_the_shield_changes[team_deathmatch, grouped] | Stale sample size: since game v5 a recharge needs the delay's length of unbroken quiet flight after a hit; only the calm ships give it and few survive unhit. Batch 16 saw none. Events code right (batch 64-128 restores shield). | batch=128. |
| 3 | test_arms_1_and_2_... (golden) | Golden obs had 11 global columns; `mode_elimination` was inserted at column 6. Only the layout: with the globals migrated by name the recorded outputs reproduce to float64 rounding, so the model's outputs did not change; no re-record. | `golden_obs` migrates globals; merged test/tolerances (float64 tolerance goldens). |
| 4 | test_fog_darkens_what_the_team_cannot_see | Stale test: the own-ship probe sat 3 ring radii (983 m at real scale) from the ship, on the soft edge of its 1024 m sight. Renderer right. | Probe at min(3 ring, half the vision range). |
| 5 | tests/docs reference checks (8) | physics.md `w_ref`, `f_0` and runbook `search.jsonl` are not names in the code; five generated sensitivity reports had no `covers`; bench/game_loop lacked docstrings. | `{.plain}`, front matter (and the generator writes it), docstrings. |
| 6 | test_ship_sets new match maps | Passes on this tip. | none |
| 7 | mixed batch past 60 ticks; test/tolerances | See below: float rounding of the acceleration quotient, no divergence; test/tolerances merged, its goldens need no re-record. | acceleration atol; activity guards |

## 7: the mixed-batch drift

The quick variant stops at 60 ticks; the slow one (260 ticks, each game alone resynced to the
mixed batch's previous state, so errors are one tick's and cannot grow chaotically) failed at
`obs`/`truth` ship column `acceleration_x/y`, 2 elements of ~440 thousand, by 7e-5 and 1e-4 on
values of ~40. An acceleration is the velocity's change over a tick divided by the tick length
(30 Hz); a float32 velocity of hundreds of m/s has an ulp of 3e-5, which the quotient turns
into ~1e-3. This is rounding of a difference quotient, not a divergence: every other leaf
(states, events, rewards, ledgers) stays within 1e-6. The test now gives the two acceleration
columns an absolute 1e-3.

The two bit-for-bit slow tests (aligned, 300 and 240 ticks) still agree bit for bit on every
tensor; they failed only on their coverage guards (652 deaths against more than 800, 434
against more than 500), because random play kills less since the edge-safe spawn headings
(game v4). The guards are now 500 and 300; they check that the play has deaths, not physics.

`origin/test/tolerances` (one commit, d10ceb38, 110 commits behind) is merged. Its float64
goldens reproduce on this tip, after `golden_obs` places the 11 recorded global columns in
today's 12 (mode_elimination at column 6, zero in the golden's team deathmatch), so the
recorded outputs of arms 1 and 2, C-fixed and the v1 spread fit are unchanged by the
elimination and position-code changes: no re-record.

Quick-tier results (CPU): tests/sections test_modes, test_mixed, test_play_render,
test_ship_sets, test_game_version, test_yemong_error_head (125 passed, about 10 min),
test_yemong_spread_fit, _accelerations, _codes, test_diagnostics, test_handling,
test_learner_trust_region, test_time_scale; tests/docs. test_mixed's two slow-tier tests
were run too and pass with the new guards.
