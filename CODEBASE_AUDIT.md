# Codebase Audit — Boost and Broadside

_Status: First full pass complete. 18 open findings, 7 fixed, 0 won't-fix
(AUDIT-001, AUDIT-005, AUDIT-009, AUDIT-014 resolved in a later batch)._
_Audit started and completed: 2026-07-23._

This document is the persistent, cross-session record of a repository-wide Python code
quality audit. Findings are never deleted — mark them `✅ Done (<commit>)` or
`⏭ Won't fix: <reason>` in place rather than removing them. Do not renumber existing
`AUDIT-NNN` identifiers; new findings always get the next unused number, regardless of
where they land in the sorted presentation below.

## 1. Executive Summary

**Scope.** Every one of the **82 tracked Python files** in the repository was read in
full and reviewed against `STYLE_GUIDE.md`. No file was skipped, sampled, or only
partially evaluated — see "Files that could not be fully evaluated" below.

**Findings.** **25 findings** total.

By severity:

| Severity | Count |
| --- | ---: |
| Critical | 1 |
| High | 2 |
| Medium | 8 |
| Low | 13 |
| Note (no numeric column in §2) | 1 |

By category (each finding is counted once, under its primary category; several findings
legitimately span two, noted inline):

| Category | Count | Findings |
| --- | ---: | --- |
| Duplication / repeated logic | 8 | AUDIT-002, 007, 012, 015, 016, 018, 019, 024 |
| Dead or unused code | 4 | AUDIT-013, 017, 021, 022 |
| Coupling / architecture | 4 | AUDIT-006, 014, 018, 022 |
| Documentation / stale comments | 4 | AUDIT-004, 017, 020, 025 |
| Typing gaps | 3 | AUDIT-003, 008, 020 |
| Maintainability | 2 | AUDIT-011, 014 |
| Correctness / reliability | 1 | AUDIT-001 |
| Missing validation | 1 | AUDIT-005 |
| Test coverage | 1 | AUDIT-009 |
| Excessive complexity | 1 | AUDIT-019 |
| Non-idiomatic / deprecated API | 1 | AUDIT-023 |
| Hidden mutable state (Note) | 1 | AUDIT-010 |

**Highest-priority repository-wide concerns, in order:**

1. **AUDIT-001 (Critical)** — `--mode bc_warmstart --smoke` silently ignores `--smoke`
   and launches a full-cost, multi-day training run instead of a crash test. Cheap,
   unambiguous fix; highest-value item to land first.
2. **AUDIT-014 (High)** — `ppo.py`'s `_GROUP` and `_LOCAL_COMPONENTS` are two
   hand-maintained registries that happen to agree today but have no structural guarantee
   of staying in sync. If they ever drift, the failure mode is silent — wrong lambda
   aggregation (team-shared vs. self-only credit) for whichever reward component is
   affected, with no crash and no existing test able to catch it, since the current
   regression tests pin *today's* components, not a future one. This is the single
   riskiest spot in the repository for a silent RL-correctness bug.
3. **AUDIT-019 (High)** — `ar_report.py`'s `_generate_report` is a ~540-line, 14-closure
   function doing seven distinct jobs — the strongest complexity outlier in an otherwise
   consistently well-scoped codebase, and the best return on refactoring effort for
   readability.
4. **A recurring "fragile checkpoint/prediction introspection" pattern** (AUDIT-018,
   AUDIT-022 and its addendum) — three independent tools reconstruct the critic width `K`
   from a hardcoded state-dict key, and three independent places hand-decode next-state
   predictions back to raw observations (one of them, `renderer.py`, via bare magic
   indices next to a docstring that's already measurably stale — `AUX_PRED_DIM=10` versus
   the real value, 9). Neither is a live bug today, but both are exactly the kind of
   architectural gap most likely to produce one the next time the model's head structure
   changes.
5. **AUDIT-024 (Medium)** — six of eight scripted-agent files duplicate the same ~15-line
   angle-to-turn-action block verbatim. High-visibility, zero-risk, mechanical cleanup —
   the fastest way to make `agents/` read as deliberately designed rather than
   copy-pasted, which matters for a portfolio audience skimming the codebase.

**Areas of particular strength** (a portfolio audit should note what's already good, not
just what to fix):

- **Physics and reward code** (`env/physics.py`, `env/rewards.py`, `env/obstacle_physics.py`)
  is exceptionally careful: branchless GPU kernels with comments that explain *why*, not
  just what; correct toroidal-wrap handling throughout; and a test suite that verifies
  physical invariants rather than fragile exact values — precisely the philosophy
  STYLE_GUIDE §6.9 asks for.
- **`train/rl/ppo.py`, `checkpoint.py`, `elo_eval.py`, `bradley_terry.py`** show real
  engineering maturity: the non-finite-gradient-scrubbing logic documents the exact NaN
  cascade it prevents; `clone_to_cpu` documents a genuine CUDA async-copy race and carries
  a dedicated regression test that provokes it; the Bradley-Terry fit explains *why*
  bisection was chosen over Newton (a real observed divergence, "~1e144", that this
  avoids); `MAX_ANCHORS`'s comment explicitly documents what silently degrades if the
  constant is ever raised without generalizing its two call sites.
- **Test suite** (256 tests, all passing): consistently one-behavior-per-test, no physics
  mocking, and several tests are explicit regressions for previously-real bugs (the CUDA
  copy race, fp32 accumulator drift, win-component lambda semantics, a position-decode
  W/H mix-up) — evidence of a team that turns incidents into permanent coverage rather
  than one-off fixes.
- **Configuration system** (`config/core.py`, `config/training.py`, `runs/*.py`): frozen
  dataclasses, `__post_init__` validation, and disciplined avoidance of silent defaults
  for active hyperparameters, consistently followed across every run profile.
- **Zero debugging artifacts**: no `TODO`/`FIXME`/`HACK`/`XXX` markers and no
  commented-out code anywhere in the tracked source, and `ruff check` / `ruff format
  --check` surface only two trivial line-length errors and five formatting-only diffs
  across the entire repository (see §4).

**Files that could not be fully evaluated:** None. All 82 tracked Python files were read
in full and reviewed.

## 2. File Inventory

Every tracked Python file in the repository. "Reviewed — no issues" means the file was
read in full and no findings met the evidence bar in the Review Rules; it does not mean
the file is beyond improvement. Severity counts are attributed to each finding's primary
**File:** only — a finding whose evidence spans multiple files is not double-counted
against each one; secondary files carry a "(see AUDIT-NNN)" pointer instead.
`Note`-severity findings have no column in this schema and are called out in Main
Categories instead.

| File | Status | Critical | High | Medium | Low | Main Categories |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `main.py` | Reviewed | 1 | 0 | 1 | 0 | Correctness, duplication |
| `runs/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `runs/bc.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `runs/bc_warmstart.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `runs/rl.py` | Reviewed | 0 | 0 | 0 | 1 | Documentation |
| `runs/rl_obstacles.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `runs/shared.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/agents/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/agents/abreast.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | see AUDIT-024 |
| `src/boost_and_broadside/agents/boom_zoom.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | see AUDIT-024 |
| `src/boost_and_broadside/agents/jinking.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/agents/jouster.py` | Reviewed | 0 | 0 | 1 | 0 | Duplication (representative; see AUDIT-024) |
| `src/boost_and_broadside/agents/reverse_turret.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | see AUDIT-024 |
| `src/boost_and_broadside/agents/run_away.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | see AUDIT-024 |
| `src/boost_and_broadside/agents/scripted_utils.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/agents/spiral_evader.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | see AUDIT-024 |
| `src/boost_and_broadside/agents/stochastic_config.py` | Reviewed | 0 | 0 | 0 | 1 | Documentation |
| `src/boost_and_broadside/agents/stochastic_scripted.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/agents/team_jouster.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | see AUDIT-024 |
| `src/boost_and_broadside/config/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/config/core.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/config/schedule.py` | Reviewed | 0 | 0 | 0 | 1 | Typing |
| `src/boost_and_broadside/config/training.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/constants.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/env/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/env/env.py` | Reviewed | 0 | 0 | 0 | 1 | Validation |
| `src/boost_and_broadside/env/observation.py` | Reviewed | 0 | 0 | 0 | 1 | Typing |
| `src/boost_and_broadside/env/obstacle_cache.py` | Reviewed | 0 | 0 | 1 | 0 | Coupling |
| `src/boost_and_broadside/env/obstacle_physics.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | see AUDIT-006 |
| `src/boost_and_broadside/env/physics.py` | Reviewed | 0 | 0 | 0 | 1 | Duplication |
| `src/boost_and_broadside/env/rewards.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/env/state.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/env/wrapper.py` | Reviewed | 0 | 0 | 0 | 0 | Hidden mutable state (Note, AUDIT-010) |
| `src/boost_and_broadside/models/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/models/mvp/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/models/mvp/attention.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/models/mvp/encoder.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/models/mvp/griffin.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/models/mvp/policy.py` | Reviewed | 0 | 0 | 0 | 1 | Maintainability |
| `src/boost_and_broadside/modes/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/modes/agent_factory.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | see AUDIT-018, AUDIT-022 |
| `src/boost_and_broadside/modes/ar_report.py` | Reviewed | 0 | 1 | 1 | 1 | Complexity, typing, dead code |
| `src/boost_and_broadside/modes/collect.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/modes/elo_calibrate.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | see AUDIT-018; ruff, see §4 |
| `src/boost_and_broadside/modes/elo_calibrate_plots.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | ruff, see §4 |
| `src/boost_and_broadside/modes/elo_stats.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | see AUDIT-018 |
| `src/boost_and_broadside/modes/feature_stats.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/modes/interactive.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/modes/noise_calibration.py` | Reviewed | 0 | 0 | 0 | 1 | Deprecated API |
| `src/boost_and_broadside/train/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/train/rl/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/train/rl/bradley_terry.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/train/rl/buffer.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/train/rl/checkpoint.py` | Reviewed | 0 | 0 | 1 | 2 | Duplication, coupling, misleading comment |
| `src/boost_and_broadside/train/rl/elo_eval.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/train/rl/features.py` | Reviewed | 0 | 0 | 2 | 1 | Duplication, dead code, architecture |
| `src/boost_and_broadside/train/rl/logging.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/train/rl/opponents.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/train/rl/ppo.py` | Reviewed | 0 | 1 | 0 | 1 | Maintainability, duplication |
| `src/boost_and_broadside/train/rl/roster.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | see AUDIT-017 |
| `src/boost_and_broadside/train/rl/sigreg.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/ui/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `src/boost_and_broadside/ui/renderer.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | see AUDIT-022 addendum |
| `tests/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/conftest.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/env/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/env/test_env.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/env/test_physics.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/env/test_rewards.py` | Reviewed | 0 | 0 | 1 | 0 | Test coverage |
| `tests/models/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/models/test_encoder.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/modes/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/modes/test_agent_factory.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/modes/test_elo_calibrate.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | ruff, see §4 |
| `tests/train/__init__.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/train/test_bradley_terry.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/train/test_buffer.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/train/test_checkpoint.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/train/test_elo_eval.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/train/test_ppo.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |
| `tests/train/test_roster.py` | Reviewed — no issues | 0 | 0 | 0 | 0 | — |

## 3. Detailed Findings

Sorted by severity, then by primary file path. IDs are assigned in discovery order and
are stable across re-sorts — do not renumber on future passes.

### AUDIT-001 — `--smoke` is silently ignored for `bc_warmstart`, launching full-cost training

- **File:** `main.py`
- **Location:** `main()`, `case "bc_warmstart":` block
- **Severity:** Critical
- **Category:** Correctness / reliability
- **Status:** ✅ Done (d4e4d23)
- **Confidence:** High

**Problem**

The top-level `--smoke` flag is documented as a blanket switch — "tiny batch (4 envs), no
W&B, no compile, exits after a few updates" — and every other training mode (`bc`, `rl`,
`rl_obstacles`) honors it by running the config through `_apply_smoke()` before building
the trainer, with `_make_trainer` deriving `use_wandb=not args.smoke` and disabling
`compile_mode` from the same flag. The `bc_warmstart` arm instead builds
`warmstart_args = argparse.Namespace(**vars(args), smoke=False)` unconditionally, then
constructs both stages' trainers directly from the full-size `BC_WARMSTART_PRETRAIN_CONFIG`
/ `BC_WARMSTART_RL_CONFIG` — neither is ever passed through `_apply_smoke`. Running
`--mode bc_warmstart --smoke` therefore silently launches the real 20,000,000-step BC
phase followed by the real 1,000,000,000-step RL phase, fully compiled, with W&B logging
on — the opposite of what the flag promises — with no warning that it was dropped.

**Evidence**

```python
case "bc_warmstart":
    warmstart_args = argparse.Namespace(**vars(args), smoke=False)
    ...
    pretrain_trainer = _make_trainer(BC_WARMSTART_PRETRAIN_CONFIG, warmstart_args)
```

`smoke=False` is forced regardless of the CLI value, and neither `BC_WARMSTART_PRETRAIN_CONFIG`
nor `BC_WARMSTART_RL_CONFIG` is ever wrapped in `_apply_smoke(...)` in this branch.

**Recommended change**

Either honor `--smoke` for `bc_warmstart` (run both stage configs through `_apply_smoke`
and thread the real `args.smoke` through instead of forcing `False`), or fail fast — raise
a clear error when `--mode bc_warmstart --smoke` is requested — if smoke-testing a
two-stage pipeline is genuinely unsupported. Per STYLE_GUIDE §1 ("Fail Fast... should
cause an immediate crash rather than silent defaults"), silently dropping the flag is the
one option that shouldn't stay on the table.

**Risks or tradeoffs**

None for fixing — this only touches the `--smoke` code path. If supporting it, shrink
`total_timesteps` for both stages so the combined smoke run still exits after a few
updates rather than just one.

**Validation**

`uv run --no-sync main.py --mode bc_warmstart --smoke` should exit in seconds with no W&B
run created. Worth a regression test asserting `args.smoke=True` reaches both stage
configs.

---

### AUDIT-019 — `_generate_report` is a ~540-line function with 7+ responsibilities and 14 nested closures

- **File:** `src/boost_and_broadside/modes/ar_report.py`
- **Location:** `_generate_report`
- **Severity:** High
- **Category:** Excessive complexity / duplication
- **Status:** Open
- **Confidence:** High

**Problem**

`_generate_report` (roughly lines 214-754 of the file) extracts feature arrays, unwraps
toroidal positions, computes a toroidal center of mass, renders four kinds of plots (full
map, centered map, velocity map, per-feature/error line charts), computes four error
metrics, and writes the markdown report — all in one function body, via 14 locally-defined
closures (`extract_feat`, `get_ship_feat`, `clamp_alive_prob`, `unwrap_1d`, `unwrap_pos`,
`toroidal_center_of_mass`, `center_pos_uw`, `plot_trajectory_on_ax`,
`plot_centered_trajectory`, `plot_vel_trajectory`, `calc_toroidal_euclidean`,
`calc_euclidean`, `calc_4d_euclidean`, `calc_mae`). Unlike the vectorized GPU kernels
elsewhere in the codebase that the STYLE_GUIDE's length exception covers, this is
matplotlib report-generation code with no performance constraint forcing it together.
Three of the closures — `plot_trajectory_on_ax`, `plot_centered_trajectory`, and
`plot_vel_trajectory` — duplicate the same per-ship loop (death-marker, dots every 10
steps, connecting lines) with only minor differences in coordinate wrapping.

**Evidence**

`ar_report.py`, `_generate_report`: single function spanning ~540 lines with 14 nested
`def`s; `plot_trajectory_on_ax` (lines ~334-379), `plot_centered_trajectory` (~418-459),
and `plot_vel_trajectory` (~500-541) share the same dot/line/death-marker plotting
structure almost verbatim.

**Recommended change**

Promote the nested closures to module-level functions grouped by concern: a
`_extract_histories(...)` step, a `_compute_error_metrics(...)` step, a shared
`_plot_ship_trajectories(ax, positions, alive, ...)` helper parametrized by whether
positions need toroidal-wrap line-skipping (unifying the three near-duplicate plotters),
and a `_write_markdown_report(...)` step, called in sequence from a slim
`_generate_report`.

**Risks or tradeoffs**

Output-producing code (PNGs + markdown) — verify byte-for-byte-equivalent (or at least
visually identical) output on a fixed input before/after the refactor, since this mode has
no automated test coverage today (see AUDIT-021).

**Validation**

Run `--mode ar_report` before and after against the same checkpoint and compare the
generated PNGs/markdown.

---

### AUDIT-014 — `_GROUP` and `_LOCAL_COMPONENTS` are two independently-maintained, structurally-redundant registries

- **File:** `src/boost_and_broadside/train/rl/ppo.py`
- **Location:** module level, `_GROUP`, `_LOCAL_COMPONENTS`; consumed in `PPOTrainer._refresh_training_schedule` and `PPOTrainer._precompute_lambda_aggregates`
- **Severity:** High
- **Category:** Maintainability / architecture
- **Status:** ✅ Done (d4e4d23)
- **Confidence:** High

**Problem**

`_LOCAL_COMPONENTS` is, today, exactly `{name for name, group in _GROUP.items() if group == "local_scale"}` — 15 names hand-copied into a second `frozenset` literal instead of being derived from `_GROUP`. They currently agree (verified by inspection and by `TestWinComponentLambdaMatrix`/`test_group_scales_applied_by_trainer` in `test_ppo.py`, which regression-test the two known historical bugs in this area). But nothing enforces that they *stay* in agreement, and the two failure modes if they drift are asymmetric:

- Omit a component from `_GROUP` → `_refresh_training_schedule`'s `_GROUP[component.name]` raises `KeyError` immediately on the first update. Loud, caught instantly.
- Omit a component from `_LOCAL_COMPONENTS` (or add one that shouldn't be there) → `_precompute_lambda_aggregates`'s `torch.where(self.local_k, local_lambda, global_lambda)` silently gives that component team-wide lambda aggregation instead of self-only diagonal lambda (or vice versa). No crash, no test failure for a *new* component — just quietly wrong credit assignment / training dynamics for whatever reward was just added.

`rewards.py`'s own "Adding a new reward" docstring (a 4-step guide: subclass `RewardComponent`, register in `REWARD_COMPONENT_NAMES`, add a `RewardConfig` field, add to `build_reward_components()`) doesn't mention either `_GROUP` or `_LOCAL_COMPONENTS` in `ppo.py`, so a contributor following the documented process has no signal that a fifth, silent-failure-mode step exists in a different file.

**Evidence**

`ppo.py`: `_GROUP: dict[str, str] = {...}` (21 entries) and
`_LOCAL_COMPONENTS: frozenset[str] = frozenset({...})` (15 entries) — the 15 are exactly
the `_GROUP` entries mapped to `"local_scale"`. `_precompute_lambda_aggregates`:
`torch.where(self.local_k, local_lambda, global_lambda)` where `self.local_k` comes from
`_make_local_k()`, built from `_LOCAL_COMPONENTS`.

**Recommended change**

Minimal fix: derive `_LOCAL_COMPONENTS` from `_GROUP` at module scope —
`_LOCAL_COMPONENTS = frozenset(name for name, group in _GROUP.items() if group == "local_scale")`
— so they're structurally unable to diverge. More thorough fix: move both classifications
onto `RewardComponent` itself in `rewards.py` (e.g. a `group: Literal["true_reward", "global", "local"]`
class attribute declared next to each component's `name = "..."`), so step 1 of the
documented "Adding a new reward" process — writing the subclass — is where the
classification lives, and `ppo.py` reads it off the component instead of maintaining a
parallel name-keyed registry two files away. Also worth a line in the rewards.py
docstring either way, pointing at wherever the classification ends up.

**Risks or tradeoffs**

The minimal fix is a one-line, zero-risk change (identical resulting set today). The
thorough fix touches every `RewardComponent` subclass and `_make_local_k`/`_GROUP`
consumers — bigger diff, but removes the cross-file coordination requirement entirely.

**Validation**

`pytest tests/env/test_rewards.py tests/train/test_ppo.py`, in particular
`TestWinComponentLambdaMatrix` and `test_group_scales_applied_by_trainer`, which already
pin the current ally_win/enemy_win classification and would catch a regression in the
derived set.

---

### AUDIT-002 — Duplicated resume/pretrain dispatch for `rl` and `rl_obstacles`

- **File:** `main.py`
- **Location:** `main()`, `case "rl":` and `case "rl_obstacles":` blocks
- **Severity:** Medium
- **Category:** Duplication / maintainability
- **Status:** Open
- **Confidence:** High

**Problem**

The `rl` and `rl_obstacles` match arms are near-identical 15-line bodies: both call
`_apply_smoke`, resolve `_find_resume_checkpoint`, build a trainer via `_make_trainer`,
and conditionally call `load_checkpoint` or `load_pretrained_weights`. Only the config
constant differs. A future change to resume/pretrain handling (a new flag, an added error
case) has to be applied in both places, and it is easy to update one and forget the other.

**Evidence**

`main.py` — the `case "rl":` body and the `case "rl_obstacles":` body are identical apart
from `RL_TRAIN_CONFIG` vs `RL_OBSTACLES_TRAIN_CONFIG`.

**Recommended change**

Extract a helper, e.g. `_run_training_mode(base_config: TrainConfig, args: argparse.Namespace) -> None`,
that performs the smoke-apply / resume-resolve / trainer-build / load / run sequence once,
and call it from both arms.

**Risks or tradeoffs**

Pure refactor, no behavior change expected.

**Validation**

`pytest`; manually confirm `--mode rl --resume` and `--mode rl_obstacles --resume` still
resolve checkpoints correctly after the extraction.

---

### AUDIT-024 — Angle-to-turn-action decision logic is duplicated verbatim across six scripted agents

- **File:** `src/boost_and_broadside/agents/jouster.py` (representative; see Evidence for the full list)
- **Location:** `JousterAgent.get_actions`, `TeamJousterAgent.get_actions`, `AbreastAgent.get_actions`, `ReverseTurretAgent.get_actions`, `RunAwayAgent.get_actions`, `SpiralEvaderAgent.get_actions`
- **Severity:** Medium
- **Category:** Duplication
- **Status:** Open
- **Confidence:** High

**Problem**

Six of the eight scripted-agent files contain the identical ~15-line block that converts
a signed relative angle into a `TurnActions` value via the same two hardcoded thresholds:

```python
turn = torch.full((B, N), TurnActions.GO_STRAIGHT, dtype=torch.int32, device=device)
normal = (abs_angle >= _5DEG) & (abs_angle < _15DEG)
sharp = abs_angle >= _15DEG
turn = torch.where(normal & (rel_angle > 0), torch.tensor(TurnActions.TURN_RIGHT, device=device), turn)
turn = torch.where(normal & (rel_angle < 0), torch.tensor(TurnActions.TURN_LEFT, device=device), turn)
turn = torch.where(sharp & (rel_angle > 0), torch.tensor(TurnActions.SHARP_RIGHT, device=device), turn)
turn = torch.where(sharp & (rel_angle < 0), torch.tensor(TurnActions.SHARP_LEFT, device=device), turn)
```

present nearly verbatim (variables renamed `rel_angle`/`abs_angle` → `target_rel`/`abs_target`
in two of the six) in `jouster.py`, `team_jouster.py`, `abreast.py`, `reverse_turret.py`,
`run_away.py`, and `spiral_evader.py`. The module-level constants it depends on,
`_5DEG = math.radians(5)` and `_15DEG = math.radians(15)` (plus `_1DEG` in three of them),
are separately redeclared in each of those files rather than imported from one place. A
smaller "boost when power_ratio >= 0.5 else coast" snippet is separately duplicated across
`jouster.py`, `team_jouster.py`, and `boom_zoom.py`.

**Evidence**

Byte-for-byte (modulo variable names) identical blocks in the six files named above;
`_5DEG`/`_15DEG`/`_1DEG` module constants redeclared identically in 4-5 files instead of
living in `scripted_utils.py` alongside the other shared agent geometry helpers
(`compute_obstacle_repulsion`, `predict_interception`, `select_targets`,
`compute_team_target_bearings`).

**Recommended change**

Add `TURN_NORMAL_ANGLE = math.radians(5)`, `TURN_SHARP_ANGLE = math.radians(15)`, and a
`turn_toward(rel_angle: torch.Tensor) -> torch.Tensor` helper to `scripted_utils.py`
(mirroring the module's existing geometry-helper pattern), and have all six agents call
it instead of reimplementing it.

**Risks or tradeoffs**

Purely mechanical extraction — each site's `torch.where` chain is behavior-identical, so
this is a pure refactor. Worth a quick numeric diff of each agent's output on a fixed
input before/after, since none of these scripted agents currently have dedicated unit
tests (only exercised indirectly via other modes' integration tests).

**Validation**

`pytest` (existing suite exercises these agents indirectly via `collect_stats`/`elo_stats`
integration paths); compare sampled actions for a fixed `TensorState` before and after the
extraction.

---

### AUDIT-006 — `step_obstacles_harmonic` requires a full `TensorState`, forcing dummy-tensor plumbing in `ObstacleCache`

- **File:** `src/boost_and_broadside/env/obstacle_cache.py`
- **Location:** `ObstacleCache.generate` / `_make_obstacle_state`, and `step_obstacles_harmonic` in `src/boost_and_broadside/env/obstacle_physics.py`
- **Severity:** Medium
- **Category:** Module boundaries / coupling
- **Status:** Open
- **Confidence:** High

**Problem**

`step_obstacles_harmonic(state: TensorState, config, enable_pbd)` only reads and writes `state.obstacle_pos`, `state.obstacle_vel`, `state.obstacle_gcenter`, and `state.obstacle_radius` — confirmed by its body, which never touches any ship or bullet field. Every other function in the same module (`init_obstacles_orbital`, `_pbd_separation`, `check_convergence`) correctly takes the individual tensors it needs rather than a `TensorState`. Because `step_obstacles_harmonic` is the odd one out, `ObstacleCache.generate()` — which only ever has obstacle tensors, no ships — has to fabricate a throwaway `TensorState` via `_make_obstacle_state()`, a ~25-line helper that builds 15 unrelated dummy `(B, 1, ...)` ship/bullet tensors purely to satisfy the dataclass constructor.

**Evidence**

`obstacle_physics.py`: `def step_obstacles_harmonic(state: TensorState, config: ShipConfig, enable_pbd: bool) -> TensorState:` reads only the four `state.obstacle_*` fields. `obstacle_cache.py`: `_make_obstacle_state()` constructs `ship_pos`, `ship_vel`, `ship_attitude`, `bullet_pos`, `bullet_vel`, `damage_matrix`, etc. — none of which `step_obstacles_harmonic` (or anything it calls) ever reads.

**Recommended change**

Change `step_obstacles_harmonic` to take `(pos, vel, radius, gcenter, config, enable_pbd)` and return the updated `(pos, vel)` tuple, matching its sibling functions in the same file. Update its one other call site (`env.py`'s `step()`, `self.state = step_obstacles_harmonic(self.state, ...)`) to unpack into `self.state.obstacle_pos` / `self.state.obstacle_vel`. This deletes `_make_obstacle_state` entirely.

**Risks or tradeoffs**

Touches the main per-step physics call site (`env.py`), so it should be covered by the existing env/physics test suite before and after; behavior should be identical since the underlying computation doesn't change, only how the tensors are threaded through.

**Validation**

`pytest tests/env/` (physics + env integration tests already exercise obstacle stepping and cache generation indirectly); confirm `uv run --no-sync main.py --mode rl_obstacles --smoke` still runs.

---

### AUDIT-020 — `ar_report.py` has no module docstring and two functions with no type hints

- **File:** `src/boost_and_broadside/modes/ar_report.py`
- **Location:** module level; `_run_ar`; `_generate_report`
- **Severity:** Medium
- **Category:** Typing / documentation
- **Status:** Open
- **Confidence:** High

**Problem**

Every other reviewed module in this codebase opens with a docstring explaining its
purpose; `ar_report.py` has none. STYLE_GUIDE §4 requires all function signatures to be
typed ("All arguments and return values must be typed"), and the codebase is otherwise
consistently well-typed — but `_run_ar(agent0, agent1, init_obs, init_hidden0, init_hidden1, num_steps, N, ship_config, forced_actions, is_closed_loop)`
and `_generate_report(history_sim, history_closed, history_open, ship_config, num_steps, out_dir)`
have zero parameter annotations, and `run_ar_report_mode` (which does have parameter
types) is missing its `-> None` return annotation.

**Evidence**

`ar_report.py`: no module-level docstring; `def _run_ar(agent0, agent1, init_obs, ...):`
and `def _generate_report(history_sim, history_closed, history_open, ship_config, num_steps, out_dir):`
have no type annotations anywhere in their signatures.

**Recommended change**

Add a module docstring describing the mode's purpose (mirroring `noise_calibration.py`'s,
which documents this well for a similar diagnostic mode). Type `_run_ar` and
`_generate_report`'s parameters (`agent0: ResolvedAgent`, `init_obs: MVPObservation`,
`init_hidden0: torch.Tensor | None`, `num_steps: int`, `N: int`,
`ship_config: ShipConfig`, `forced_actions: list[torch.Tensor] | None`,
`is_closed_loop: bool`, etc.) and add `-> None` to `run_ar_report_mode`.

**Risks or tradeoffs**

None — annotations and docstrings only.

**Validation**

None beyond review; re-run any type checker the project adopts.

---

### AUDIT-018 — Critic width (K) is never saved explicitly; three call sites reverse-engineer it from a hardcoded state-dict key

- **File:** `src/boost_and_broadside/train/rl/checkpoint.py`
- **Location:** `CheckpointMixin.checkpoint_payload` / `_save_ladder_snapshot`; consumed at `src/boost_and_broadside/modes/agent_factory.py:163`, `src/boost_and_broadside/modes/elo_stats.py:75`, `src/boost_and_broadside/modes/elo_calibrate.py:191`
- **Severity:** Medium
- **Category:** Coupling / duplication
- **Status:** ✅ Done (240f85f)
- **Confidence:** High

**Problem**

`MVPPolicy.__init__` requires `num_value_components` (K) as an explicit constructor
argument, but no checkpoint payload (`checkpoint_payload`, `_checkpoint_payload_lightweight`,
`_save_ladder_snapshot`) ever records it — unlike `team_pma_k`, which every payload does
save specifically so it can be recovered on load. Every place that needs to reconstruct a
policy from a checkpoint without already knowing K independently instead reads it off the
saved state dict by a hardcoded parameter name:
`ckpt["policy_state_dict"]["value_head_local.3.weight"].shape[0]` — identically duplicated
in `agent_factory.py`, `elo_stats.py`, and `elo_calibrate.py`. This couples three unrelated
analysis tools to the exact internal structure of `MVPPolicy.value_head_local`
(an `nn.Sequential` where index 3 happens to be the final `Linear`). Renaming that
attribute breaks all three loudly (`KeyError`); restructuring the `Sequential` (e.g.
inserting a layer) while keeping index 3 populated could silently recover the wrong K.

**Evidence**

`checkpoint.py`: `checkpoint_payload()` saves `policy_state_dict`, `team_pma_k`, and many
training-state fields, but no `num_value_components` / `K`. `agent_factory.py:163`,
`elo_stats.py:75`, `elo_calibrate.py:191`: identical
`ckpt["policy_state_dict"]["value_head_local.3.weight"].shape[0]` expressions.

**Recommended change**

Add `"num_value_components": self.wrapper.num_active_components` to
`checkpoint_payload()`, `_checkpoint_payload_lightweight()`, and `_save_ladder_snapshot()`
(mirroring how `team_pma_k` is already handled), then update the three consumers to read
`ckpt["num_value_components"]` with a fallback to the current shape-introspection for
checkpoints saved before the field existed.

**Risks or tradeoffs**

Additive checkpoint field — old checkpoints keep loading via the fallback. Low risk;
worth doing before the value head's structure changes for any other reason.

**Validation**

`pytest tests/train/test_checkpoint.py tests/modes/`; save a checkpoint and confirm all
three loaders (`agent_factory`, `elo_stats`, `elo_calibrate`) still resolve the correct K
against both the new field and (with it stripped from the payload) the fallback path.

---

### AUDIT-012 — `FeatureCoordinator` rebuilds per-feature dimensions from a dummy observation in five separate methods

- **File:** `src/boost_and_broadside/train/rl/features.py`
- **Location:** `FeatureCoordinator.target_slices`, `label_scale_vector`, `compute_labels`, `apply_all_predictions`, `get_feature_names`
- **Severity:** Medium
- **Category:** Duplication / repeated logic
- **Status:** ✅ Done (240f85f)
- **Confidence:** High

**Problem**

`_init_dims()` (called once in `__init__`) already walks every feature and computes its
target/prediction dimensions — it even builds `self._windowed_loss_specs` to cache
`(offset, dim, WindowedLoss)` for windowed-loss features specifically. But five other
methods each independently re-derive the same per-feature `t_dim`/`p_dim` information by
calling `self._dummy_obs()` and looping `for f in self.features: ... f.get_target(dummy).shape[-1]`
from scratch: `target_slices`, `label_scale_vector`, `compute_labels`, `apply_all_predictions`,
and `get_feature_names`. `label_scale_vector` additionally reallocates a fresh
`torch.tensor(scales, ...)` from a Python list on every call. `compute_labels` (and
transitively `label_scale_vector`) runs at least once per PPO update via the aux
next-state-prediction loss, so this repeated dummy-observation/shape-recomputation dance
runs for the lifetime of every training run — the absolute cost is small (the dummy obs is
`(1, 1, ...)`-shaped) but the same offset-computation logic is now duplicated five ways,
any of which could drift if a new `Transform`/`Predictor` type changes how dimensions are
derived.

**Evidence**

`features.py`: five near-identical blocks of the form
`for f in self.features: if not f.predictor: continue; t_dim = f.get_target(dummy).shape[-1]; ...`,
each preceded by its own `dummy = self._dummy_obs()`.

**Recommended change**

In `_init_dims()`, build one cached list, e.g.
`self._predictor_specs: list[tuple[Feature, int, int, int]]` (feature, t_dim, p_dim,
p_offset), and have all five methods iterate that instead of recomputing via a fresh dummy
observation. `label_scale_vector` can additionally cache its output tensor per
`(device, dtype)` the way `Normalize` already does for its own scale tensor.

**Risks or tradeoffs**

Pure refactor if the cached values are computed identically to today's per-call
recomputation — worth a quick before/after diff of `compute_labels()` output on a fixed
input to confirm no behavior change.

**Validation**

`pytest tests/models/test_encoder.py tests/train/` (features feed both the encoder input
path and the PPO aux-loss path); compare `get_feature_names()`, `target_slices()`, and
`label_scale_vector()` output before/after the refactor on `build_standard_coordinator()`.

---

### AUDIT-022 — `Predictor.decode()` is a required abstract method with zero callers

- **File:** `src/boost_and_broadside/train/rl/features.py`
- **Location:** `Predictor.decode` (abstract) and its three implementations (`AbsolutePredictor.decode`, `AdditivePredictor.decode`, `UnitCirclePredictor.decode`); compare `src/boost_and_broadside/modes/agent_factory.py`'s `_decode_targets_to_obs`
- **Severity:** Medium
- **Category:** Dead code / architecture
- **Status:** ✅ Done (240f85f)
- **Confidence:** High

**Problem**

`Predictor` declares `decode()` as `@abstractmethod` ("Invert target encoding back toward
raw physical space"), forcing every predictor subclass to implement it — but nothing in
the codebase calls `.decode(` anywhere (confirmed by search). The actual consumer that
needs this exact capability, `agent_factory.py`'s `_decode_targets_to_obs` (used by watch
mode's imagined-trajectory rendering and by `ar_report.py`/`noise_calibration.py`'s
autoregressive rollouts), hand-rolls the inverse transform per feature inline
(`atan2`-based angle recovery, `expm1` for symlog inversion, etc.) instead of calling
`decode()`. This makes sense in one respect — `Predictor.decode()` only undoes the
phase/absolute/additive *prediction* semantics, landing at a target-space value, not the
raw physical one, while `_decode_targets_to_obs` also needs to invert the `Transform`'s
domain scaling (e.g. `UnitCircle`'s `scales`) to get back to raw health/power/cooldown —
a capability `Transform` doesn't expose at all. Either way, `decode()` as currently
factored is unused and its docstring overpromises what it actually does.

**Evidence**

`grep -rn "\.decode(" --include="*.py" .` matches only the four `def decode` definitions
themselves, no call sites.

**Recommended change**

Either remove `Predictor.decode()` (and its three implementations) per STYLE_GUIDE §6.8
("Remove, Don't Deprecate") if it's genuinely superseded, or — better — finish the
abstraction: add a matching inverse to `Transform` (or extend `Predictor.decode()` to take
the `Transform` and compose both inversions) and rewrite `_decode_targets_to_obs` to call
it, so the "decode a prediction back to a raw observation" logic lives once instead of
being reimplemented by hand next to the one real caller.

**Risks or tradeoffs**

If unifying rather than deleting: `_decode_targets_to_obs` has a dedicated regression test
(`test_agent_factory.py::test_position_decodes_each_axis_with_its_own_world_extent`, for a
previously real W/H mix-up bug) — any refactor must keep that passing, since this exact
function has broken silently before.

**Validation**

`pytest tests/modes/test_agent_factory.py`; if deleting `decode()`, grep once more to
confirm no caller was missed.

**Addendum — a third, more fragile instance**

`src/boost_and_broadside/ui/renderer.py`'s `_draw_ghost_ships` hand-decodes the same kind
of phase-shift predictions a third time, for watch-mode's imagined-trajectory ghosts —
but unlike `_decode_targets_to_obs` (which resolves dimensions via
`coordinator.target_slices()`), it indexes the raw prediction tensor with bare integer
literals: `pn[n, 0]` / `pn[n, 1]` for position phase deltas, `pn[n, 4]` for the attitude
phase delta, justified only by a docstring comment ("`[0]` Δφ_x and `[1]` Δφ_y... `[4]`
Δφ_att"). That same docstring also states `AUX_PRED_DIM=10`, but
`coordinator.total_prediction_dimension` is 9 (matches the README's "Prediction layout (9
dims total)" and `ppo.py`'s `_NS_FEAT_NAMES`, a 9-tuple) — a live example of exactly the
kind of drift this hand-rolled-decode pattern invites: the dimension count in a comment
went stale while the three indices it also documents happened to still be correct. This
is a second concrete call site for whatever unification `decode()` ends up getting, and
the fastest independent fix is to correct "`AUX_PRED_DIM=10`" to 9 in `_draw_ghost_ships`'s
docstring.

---

### AUDIT-009 — 7 of 21 reward components have no direct unit test

- **File:** `tests/env/test_rewards.py`
- **Location:** module-level (missing `Test*` classes)
- **Severity:** Medium
- **Category:** Test coverage
- **Status:** ✅ Done (d4e4d23)
- **Confidence:** High

**Problem**

`test_rewards.py` has a dedicated `Test*Reward` class with value-level assertions for 14 of
the 21 registered components (`ally_damage`, `enemy_damage`, `ally_death`, `enemy_death`,
`ally_win`, `enemy_win`, `facing`, `closing_speed`, `kill_shot`, `kill_assist`,
`damage_taken`, `damage_dealt_enemy`, `damage_dealt_ally`, `death`). Seven have no direct
test anywhere in the repo: `shoot_quality`, `obstacle_death`, `obstacle_proximity`,
`obstacle_closing_speed`, `obstacle_tti`, `shooting_penalty`, `speed`. The only other place
these run is `tests/env/test_env.py`'s wrapper-level tests, which check output *shape*, not
per-component values, and don't enable obstacles or exercise shooting. This leaves the
entire obstacle-avoidance reward family (the sole training signal for `--mode rl_obstacles`)
and `ShootQualityReward`'s two-term quality formula without any regression coverage — both
non-trivial enough (`ObstacleTTIReward`'s quadratic time-to-intersection solve in
particular) that a sign or index error could pass silently.

**Evidence**

`rewards.py` classes `ShootQualityReward`, `ObstacleDeathReward`, `ObstacleProximityReward`,
`ObstacleClosingSpeedReward`, `ObstacleTTIReward`, `ShootingPenaltyReward`, `SpeedReward`
have no corresponding `Test*` class in `test_rewards.py`, and no other test file references
them by name.

**Recommended change**

Add a `Test*Reward` class per missing component following the existing pattern (controlled
2-4 ship state, one behavior asserted per test) — e.g. for `ObstacleTTIReward`: a ship on a
direct collision course gets a larger penalty than one on a near-miss course; for
`ShootQualityReward`: a close, well-aimed shot scores positive and a far/unaimed one scores
negative, mirroring the docstring's worked example.

**Risks or tradeoffs**

None — additive test-only change. Worth doing before relying on `rl_obstacles` results for
anything presented externally.

**Validation**

New tests should fail against a deliberately-broken version of each component (e.g. flip a
sign) to confirm they actually constrain the behavior, then pass against current code.

---

### AUDIT-004 — Unexplained divisor in `_MICROBATCH_TOKENS`

- **File:** `runs/rl.py`
- **Location:** module level, `_MICROBATCH_TOKENS = _MAX_TOKENS // _NUM_MINIBATCHES // 5`
- **Severity:** Low
- **Category:** Documentation / clarity
- **Status:** Open
- **Confidence:** Medium

**Problem**

Every other derived constant in this file is explained (the phase-structure docstring,
inline comments on the schedule steps), but the `// 5` here has no explanation. A future
contributor retuning this memory-only knob for a different GPU (per
`TrainConfig.microbatch_tokens`'s own docstring) can't tell whether 5 is a VRAM headroom
margin, an empirically-tuned value, or a leftover from an earlier experiment.

**Evidence**

`runs/rl.py`: `_MICROBATCH_TOKENS = _MAX_TOKENS // _NUM_MINIBATCHES // 5`.

**Recommended change**

Add a one-line comment stating why 5 (e.g. the VRAM headroom target it was tuned against),
consistent with the rest of the file's documentation density.

**Risks or tradeoffs**

None — comment-only change.

**Validation**

None needed beyond review.

---

### AUDIT-025 — `StochasticAgentConfig`'s vector-interface docstrings say "24-element" but the vector is 28

- **File:** `src/boost_and_broadside/agents/stochastic_config.py`
- **Location:** `StochasticAgentConfig.from_vector`, `StochasticAgentConfig.default_vector`
- **Severity:** Low
- **Category:** Documentation
- **Status:** Open
- **Confidence:** High

**Problem**

Both docstrings say "24-element flat vector," but `PARAM_BOUNDS` has 14 entries and the
code computes `expected = 2 * len(cls.PARAM_BOUNDS)` = 28 — matching the class's own
inline comment two lines below `PARAM_BOUNDS`'s closing bracket, `# Vector length =
2 * len(PARAM_BOUNDS) = 28`. This is exactly the kind of hardcoded, driftable count
STYLE_GUIDE §5 warns against ("Never hardcode counts in prose... Point at the source of
truth instead") — the two ramp fields added for team-target blending
(`team_target_distance_ramp`/`_prob`) appear to have grown `PARAM_BOUNDS` from 12 to 14
entries without the docstrings being updated from 24 to 28.

**Evidence**

`stochastic_config.py`: `"""Construct a StochasticAgentConfig from a 24-element flat
vector.` and `"""Returns the 24-element [0, 1]-normalized vector...` vs.
`expected = 2 * len(cls.PARAM_BOUNDS)` (= 28) and the correct `# Vector length = ... = 28`
comment.

**Recommended change**

Update both docstrings to say "28-element" (or better, reference `2 * len(PARAM_BOUNDS)`
by name instead of a bare number, so it can't drift again).

**Risks or tradeoffs**

None — `from_vector` already raises `ValueError` on a length mismatch, so this is a
documentation-only fix, not a latent bug.

**Validation**

None beyond review.

---

### AUDIT-003 — Schedule primitive return types widen to `Any`

- **File:** `src/boost_and_broadside/config/schedule.py`
- **Location:** `linear`, `exponential`, `cosine_anneal` (declared return type `Schedule`); module-level `Schedule = Callable[[int], Any]`
- **Severity:** Low
- **Category:** Typing
- **Status:** Open
- **Confidence:** High

**Problem**

`linear()`, `exponential()`, and `cosine_anneal()` each define an inner
`_schedule(step: int) -> float` and are provably float-returning, but their outer
signatures are annotated `-> Schedule`, i.e. `Callable[[int], Any]`, discarding that
precision. Call sites already rely on the real type (e.g.
`TrainingSchedule.learning_rate: Callable[[int], float]`); only the constructors'
declared types are looser than what they actually produce, so a type checker cannot catch
a mismatched assignment at the construction site.

**Evidence**

`def linear(*keypoints: tuple[int, float]) -> Schedule:` wraps an inner
`def _schedule(step: int) -> float:` (schedule.py); the same pattern appears in
`exponential` and `cosine_anneal`.

**Recommended change**

Give the three float-only primitives a precise `Callable[[int], float]` return annotation
instead of the generic `Schedule` alias. `constant`, `stepped`, and `join` are genuinely
polymorphic (used with `float`, `int`, `bool`, and `frozenset[str]` across
`TrainingSchedule`) — leave `Schedule` for those, or promote it to a `TypeVar`-parameterized
alias (PEP 695 `type Schedule[T] = Callable[[int], T]`) if the extra precision is worth
the churn everywhere it is used.

**Risks or tradeoffs**

Type-only change; no runtime behavior difference.

**Validation**

None beyond review; re-run any type checker the project adopts.

---

### AUDIT-005 — `TensorEnv.reset_envs` does not validate `team_sizes`

- **File:** `src/boost_and_broadside/env/env.py`
- **Location:** `TensorEnv.reset_envs`
- **Severity:** Low
- **Category:** Missing validation
- **Status:** ✅ Done (d4e4d23)
- **Confidence:** Medium

**Problem**

`reset_envs` accepts `options={"team_sizes": (n_team0, n_team1)}` and builds `new_alive[:, :n_team0+n_team1] = True` plus a team-id permutation, with no check that `n_team0 + n_team1 <= num_ships` or that both counts are non-negative. If a future caller ever passed `n_team0=0` (a team with no ships), `_check_game_over`'s `team0_exists = (state.ship_team_id == 0).any(dim=1)` would still read `True` — because the unused filler slot(s) beyond `n_team0 + n_team1` default to `team_id=0` — making `team0_exists & (team0_alive == 0)` true from the very first step, i.e. the episode would be marked done before any combat occurred. All current call sites (`elo_stats.py`, `collect.py`, `elo_calibrate.py`, tests) always pass positive, `N`-summing pairs, so this isn't observed in practice today — it's a latent gap, not an active bug.

**Evidence**

`env.py`: `n_team0, n_team1 = options["team_sizes"]` is used directly with no bounds check, versus `TrainConfig.__post_init__` and `ModelConfig.__post_init__` elsewhere in the codebase which do validate their inputs immediately (per STYLE_GUIDE §1 "Fail Fast").

**Recommended change**

Add a guard at the top of `reset_envs`: raise `ValueError` if `n_team0 < 0 or n_team1 < 0 or n_team0 + n_team1 > self.env_config.num_ships`.

**Risks or tradeoffs**

None — purely additive validation on an already-internal option dict.

**Validation**

Add a test that passing an invalid `team_sizes` (e.g. summing past `num_ships`, or `(0, N)`) raises immediately rather than producing a silently-wrong episode.

---

### AUDIT-008 — `MVPObservation` has several untyped or loosely-typed members

- **File:** `src/boost_and_broadside/env/observation.py`
- **Location:** `MVPObservation.data`, `__getitem__`, `__contains__`, `items`, `slice_envs`
- **Severity:** Low
- **Category:** Typing
- **Status:** Open
- **Confidence:** High

**Problem**

STYLE_GUIDE §4 requires all arguments and return values to be typed. `MVPObservation.data`
is declared as a bare `dict` rather than `dict[ObsKey, torch.Tensor]`; `__getitem__(self, key)`,
`__contains__(self, key)`, and `slice_envs(self, idx)` have no parameter annotations at
all; `items(self)` has no return annotation. The class otherwise fully types its property
accessors (`pos`, `vel`, `att`, ... all return `-> torch.Tensor`), so this is a gap in an
otherwise well-typed class rather than a systemic issue.

**Evidence**

`observation.py`: `data: dict`; `def __getitem__(self, key) -> torch.Tensor:`;
`def __contains__(self, key) -> bool:`; `def items(self):`; `def slice_envs(self, idx) -> "MVPObservation":`.

**Recommended change**

`data: dict[ObsKey, torch.Tensor]`; type `key: ObsKey | str` on `__getitem__`/`__contains__`;
`items(self) -> ItemsView[ObsKey, torch.Tensor]`; type `idx` on `slice_envs` from its actual
call sites (likely `slice | torch.Tensor`).

**Risks or tradeoffs**

None — annotation-only change.

**Validation**

None beyond review; re-run any type checker the project adopts.

---

### AUDIT-007 — Identical threshold recomputed under two names in `_update_kinematics`

- **File:** `src/boost_and_broadside/env/physics.py`
- **Location:** `_update_kinematics`
- **Severity:** Low
- **Category:** Duplication / readability
- **Status:** Open
- **Confidence:** High

**Problem**

`stalled = speed < config.min_speed` (used to zero turn/lift) and, a few lines later,
`stopped = speed < config.min_speed` (used to hold attitude instead of aligning to a
near-zero velocity direction) are the exact same expression under two different names.
They currently can't drift apart because they're computed back-to-back in one function,
but nothing signals that the two names denote the same condition — a future edit to one
(e.g. giving "stopped" its own tunable threshold) could silently desync from "stalled"
without it being obvious that they were ever meant to move together, and it's wasted
element-wise work in a per-step hot-path kernel.

**Evidence**

`physics.py`, `_update_kinematics`: two identical `speed < config.min_speed` comparisons,
~15 lines apart, bound to `stalled` and `stopped` respectively.

**Recommended change**

Compute once (e.g. `below_min_speed = speed < config.min_speed`) and reuse for both the
turn/lift zeroing and the attitude-hold branch, with a short comment noting it gates two
different effects.

**Risks or tradeoffs**

None — behavior-preserving, since both sites use the identical condition today.

**Validation**

`pytest tests/env/test_physics.py` (thrust/turning tests already cover both code paths).

---

### AUDIT-011 — Orthogonal-init relies on positional indices into `nn.Sequential` heads

- **File:** `src/boost_and_broadside/models/mvp/policy.py`
- **Location:** `MVPPolicy.__init__`
- **Severity:** Low
- **Category:** Maintainability
- **Status:** Open
- **Confidence:** Medium

**Problem**

The orthogonal-init block reaches into each head by hardcoded position —
`head[0].weight` / `head[3].weight` for `action_head` and `value_head_local`
(`nn.Sequential(Linear, RMSNorm, GELU, Linear)`), and the same `net[0]` / `net[3]`
pattern for `next_state_head` and `value_head_win`. This couples the init code to the
exact layer count and ordering of each `Sequential`, ~20-100 lines away from where
they're defined. Adding, removing, or reordering a layer in any of these heads (e.g. a
`Dropout`) would not silently corrupt anything — `orthogonal_` requires a 2D weight, so
indexing the wrong module raises immediately — but it would require remembering to update
every index in this block too, with no compiler/type-checker link between the two.

**Evidence**

`policy.py`: `nn.init.orthogonal_(head[0].weight, ...)` / `nn.init.orthogonal_(head[3].weight, ...)`
repeated for four different `Sequential` heads.

**Recommended change**

Iterate each head and initialize by type instead of position, e.g.
`linears = [m for m in head if isinstance(m, nn.Linear)]` then apply the first/last-layer
gains to `linears[0]` / `linears[-1]`. Behavior-preserving, removes the positional coupling.

**Risks or tradeoffs**

None — the resulting initialization is identical, just derived by type instead of index.

**Validation**

`pytest tests/models/test_encoder.py` (policy construction/shape tests); a quick check
that per-parameter init statistics (mean/std) are unchanged before/after.

---

### AUDIT-021 — Dead no-op `if` block in `_generate_report`

- **File:** `src/boost_and_broadside/modes/ar_report.py`
- **Location:** `_generate_report`, near the "Full-world 2D Game Map" comment
- **Severity:** Low
- **Category:** Dead code
- **Status:** Open
- **Confidence:** High

**Problem**

```python
# --- 1. Full-world 2D Game Map (ALL Ships) --- only for 2v2
if plot_N > 1 or num_ships > 2:
    # 2v2 case: always draw the full-world map
    pass

# Always draw full-world map for non-1v1
def plot_trajectory_on_ax(...):
```

This `if` block's body is only `pass` — it has no effect. The actual map-drawing is a
separate `if num_ships > 2:` block later in the function. The two comments around it
("only for 2v2" vs. "Always draw... for non-1v1") also describe inconsistent conditions,
suggesting this is a leftover fragment from an earlier version of the function.

**Evidence**

`ar_report.py`, `_generate_report`: the quoted `if plot_N > 1 or num_ships > 2: pass`
block.

**Recommended change**

Delete the dead block and its comments; the real logic already lives in the later
`if num_ships > 2:` block.

**Risks or tradeoffs**

None — removing a no-op statement.

**Validation**

`--mode ar_report` output unchanged (the block does nothing today).

---

### AUDIT-023 — `datetime.utcnow()` is deprecated

- **File:** `src/boost_and_broadside/modes/noise_calibration.py`
- **Location:** `_build_output`
- **Severity:** Low
- **Category:** Non-idiomatic / deprecated API
- **Status:** Open
- **Confidence:** High

**Problem**

`datetime.datetime.utcnow()` has been deprecated since Python 3.12 (it returns a
naive datetime that's easy to misinterpret as local time) in favor of
`datetime.datetime.now(datetime.UTC)`. The project targets Python 3.13+, so this call
emits a `DeprecationWarning` under the project's own minimum supported version.

**Evidence**

`noise_calibration.py`, `_build_output`: `datetime.datetime.utcnow().isoformat() + "Z"`.

**Recommended change**

`datetime.datetime.now(datetime.UTC).isoformat()` (already carries the `+00:00` offset,
so the manual `+ "Z"` suffix can be dropped or kept for the existing string format,
whichever the downstream JSON consumers expect).

**Risks or tradeoffs**

Output format changes from a bare `...Z` suffix to an explicit `+00:00` offset unless the
suffix is preserved manually — check nothing parses `noise_params.json`'s timestamp with a
format expecting exactly `Z`.

**Validation**

`--mode noise_calibration` (or a unit test around `_build_output`) confirms
`metadata.timestamp` still parses as expected.

---

### AUDIT-016 — Duplicated "is a save already in flight" guard in `_save_checkpoint` and `_save_best_checkpoint`

- **File:** `src/boost_and_broadside/train/rl/checkpoint.py`
- **Location:** `CheckpointMixin._save_checkpoint`, `CheckpointMixin._save_best_checkpoint`
- **Severity:** Low
- **Category:** Duplication
- **Status:** Open
- **Confidence:** High

**Problem**

Both methods open with the same ~10-line pattern: check a `self._active_*_thread`
attribute exists, is not `None`, and `.is_alive()`; if so, print a warning and return
early. The only differences are which attribute is checked and the text of the warning.
Both then close with the same `threading.Thread(target=_async_save, daemon=True); .start()`
pair.

**Evidence**

`checkpoint.py`: the `hasattr(self, "_active_save_thread") and ... .is_alive()` block in
`_save_checkpoint`, and the near-identical `_active_best_thread` block in
`_save_best_checkpoint`.

**Recommended change**

Extract a small helper, e.g. `_run_async_save(self, thread_attr: str, label: str, target: Callable[[], None]) -> None`,
that does the busy-check-and-warn, spawns the thread, stores it back onto
`getattr(self, thread_attr)`, and starts it — called from both sites with their
respective attribute name, label, and `_async_save` closure.

**Risks or tradeoffs**

Pure refactor. Keep both call sites' distinct warning text (they name different save
kinds) so log output doesn't get vaguer.

**Validation**

`pytest tests/train/test_checkpoint.py`; manually confirm both a regular and a best-model
checkpoint still skip-and-warn when triggered twice in quick succession.

---

### AUDIT-017 — Checkpoint pruning's roster protection is a no-op given current filename conventions

- **File:** `src/boost_and_broadside/train/rl/checkpoint.py`
- **Location:** `CheckpointMixin._save_checkpoint` (the `_async_save` prune block); `EloRoster.kept_paths` in `src/boost_and_broadside/train/rl/roster.py`
- **Severity:** Low
- **Category:** Misleading comment / dead code path
- **Status:** Open
- **Confidence:** Medium

**Problem**

`_save_checkpoint`'s prune step reads: "Prune: keep only the latest checkpoint + all
roster-referenced files," implemented as
`kept = self.roster.kept_paths(); kept.add(str(path)); for old_path in ckpt_dir.glob("step_*.pt"): if str(old_path) not in kept: unlink`.
`roster.kept_paths()` only ever returns paths for `kind == "checkpoint"` roster entries,
and the only production call site that creates one (`_maybe_advance_ladder` →
`roster.add_checkpoint(str(self._save_ladder_snapshot()), ...)`) always writes to
`ladder_step_{step}.pt`. The prune loop's glob is `"step_*.pt"`, which does not match
filenames starting with `"ladder_step_"`. So `roster.kept_paths()` can never contain a
path this glob would ever enumerate — the set it contributes to `kept` is always
irrelevant here. In practice this is harmless: every `step_*.pt` except the one just
written gets pruned every save (full resumable checkpoints are deliberately kept to
"latest only"), and every `ladder_step_*.pt` survives regardless (per
`_save_ladder_snapshot`'s own docstring: "never pruned") — simply because the two file
families never share a glob, not because `kept_paths()` did anything. `test_roster.py`'s
`test_kept_paths_returns_checkpoint_paths_only` verifies `kept_paths()` in isolation with
arbitrary paths, so nothing exercises the actual interaction with the `step_*.pt` glob.

**Evidence**

`checkpoint.py`: `path = ckpt_dir / f"ladder_step_{self._global_step:012d}.pt"` (`_save_ladder_snapshot`)
vs. `path = ckpt_dir / f"step_{self._global_step:012d}.pt"` (`_save_checkpoint`) vs.
`ckpt_dir.glob("step_*.pt")` in the prune loop.

**Recommended change**

Either fix the comment to describe what actually happens ("keep only the latest full
checkpoint; ladder snapshots use a separate prefix and are never touched by this glob"),
or — if the intent really was for `kept_paths()` to guard `step_*.pt` files generally
(e.g. in case a future code path ever calls `roster.add_checkpoint()` with a `step_*.pt`
path) — broaden the glob to whatever pattern `kept_paths()` is meant to protect, and add a
test that actually exercises `_save_checkpoint`'s prune loop against a populated roster.

**Risks or tradeoffs**

None currently — this is a documentation/clarity fix, not a behavior change, since the
current outcome (keep latest full checkpoint, keep the whole ladder) already matches the
stated goal.

**Validation**

None required for the comment fix. If broadening protection, add an integration test that
populates a roster with a `step_*.pt`-referencing entry and asserts `_save_checkpoint`
does not prune it.

---

### AUDIT-013 — `Feature.weight` is stored but never read

- **File:** `src/boost_and_broadside/train/rl/features.py`
- **Location:** `Feature.__init__` / `Feature.weight`
- **Severity:** Low
- **Category:** Dead code
- **Status:** Open
- **Confidence:** High

**Problem**

`Feature.__init__` accepts and stores `weight: float | tuple[float, ...] = 1.0`, but no
method on `FeatureCoordinator` reads `f.weight` anywhere — `get_loss_weights()` returns a
uniform all-ones tensor and its docstring says so explicitly: "Importance weighting is
deferred; relative scaling is handled by label_scale in compute_labels." No call to
`Feature(...)` in `build_standard_coordinator()` passes `weight=` either, so the field is
inert in both directions: never set to anything but its default, and never consulted.

**Evidence**

`features.py`: `self.weight = weight` in `Feature.__init__` is the only reference to the
attribute in the entire codebase (confirmed by search); `get_loss_weights()` ignores it.

**Recommended change**

Either remove the `weight` parameter until per-feature loss weighting is actually
implemented (STYLE_GUIDE §6.8: remove rather than leave half-wired), or wire it into
`get_loss_weights()` now if the intent is close to being acted on.

**Risks or tradeoffs**

None if removed — no current caller passes a non-default value.

**Validation**

`pytest tests/models/test_encoder.py`; grep for `weight=` on any `Feature(` call site to
confirm none rely on it before removing.

---

### AUDIT-015 — Per-metric accumulator-to-output mapping in `_update_epochs` is hand-written 25 times instead of table-driven

- **File:** `src/boost_and_broadside/train/rl/ppo.py`
- **Location:** `PPOTrainer._update_epochs`
- **Severity:** Low
- **Category:** Duplication
- **Status:** Open
- **Confidence:** Medium

**Problem**

Inside the minibatch loop, `_additive` (a tuple of `(short_key, diag_key)` pairs) already
drives a small loop that fills `scalar_accum_step` from `diag`, avoiding ~20 lines of
one-off assignments. Immediately after, the corresponding step of copying
`scalar_accum_step[...]` into the per-update `accum_scalar[...]` lists is *not*
table-driven — it's ~25 individual `accum_scalar["loss/total"].append(scalar_accum_step["loss"])`-style
lines. Adding a new tracked metric means editing three places (the `_additive` tuple or an
equivalent, the dict pre-declaration, and this block) instead of one.

**Evidence**

`ppo.py`, `_update_epochs`: the block from `accum_scalar["loss/total"].append(...)` through
`accum_scalar["train/nonfinite_grad_fraction"].append(...)` — about 25 lines, each a
1:1 key rename from `scalar_accum_step` to `accum_scalar`.

**Recommended change**

Extend the existing `(short_key, diag_key)` pattern with the namespaced output key, e.g.
a single tuple of `(namespaced_key, short_key)` pairs, and replace the hand-written block
with one loop: `for out_key, short_key in _metric_map: accum_scalar[out_key].append(scalar_accum_step[short_key])`.

**Risks or tradeoffs**

Pure refactor. Low priority — this is bookkeeping, not behavior, and the current form is
at least mechanically easy to get right (each line is a simple rename).

**Validation**

`pytest tests/train/test_ppo.py`; diff the metric dict keys/values before and after on one
`_update_epochs()` call to confirm no key was dropped or renamed.

---

### AUDIT-010 — `_make_prev_state_proxy`'s snapshot guarantee is implicit and undocumented as an invariant

- **File:** `src/boost_and_broadside/env/wrapper.py`
- **Location:** `_make_prev_state_proxy`
- **Severity:** Note
- **Category:** Hidden mutable state
- **Status:** Open
- **Confidence:** Medium

**Problem**

`_make_prev_state_proxy` builds `prev_state` via `dataclasses.replace(state, ship_health=prev_health, ship_alive=prev_alive)`,
which only deep-copies `ship_health`/`ship_alive`; every other field is shared by reference
with `self.env.state` at the moment of the call. This is safe today only because every
reward component that reads `prev_state` reads exclusively `prev_state.ship_health` or
`prev_state.ship_alive` (verified across all `RewardComponent` subclasses in `rewards.py`),
and because the physics engine's convention is to advance fields via reassignment
(`state.field = new_value`) rather than true in-place mutation — with one exception,
`state.ship_hit_obstacle.zero_()` in `env.py`'s `step()`, which happens not to be read from
`prev_state` by anything today. Both of these are conventions, not enforced invariants: a
future reward component that reads e.g. `prev_state.ship_pos` expecting a genuine
pre-physics value, or a future physics change that mutates a field in place instead of
reassigning it, would silently break the "previous state" snapshot for that field with no
error — just a quietly wrong reward on the affected component.

**Evidence**

`wrapper.py`: `_make_prev_state_proxy` docstring says "shares all tensors but swaps in
pre-damage health/alive" — accurate, but doesn't flag that this is a fragile assumption
downstream code must not violate.

**Recommended change**

Add a code comment on `_make_prev_state_proxy` (or on `TensorState`/the physics functions)
stating the invariant explicitly: physics functions must advance state via reassignment,
never in-place mutation, or the `prev_state` snapshot silently breaks for that field. No
functional change needed today.

**Risks or tradeoffs**

None — documentation only. Flagged as a Note rather than a defect because current behavior
is correct; this is a preventive suggestion for future maintainers, not an active bug.

**Validation**

None required today; revisit if a future reward component starts reading a `prev_state`
field other than `ship_health`/`ship_alive`.

<!-- FINDINGS_END -->

## 4. Static Analysis & Test Baseline

Recorded 2026-07-23, before any fixes were applied.

**`uv run --no-sync ruff check .`** — 2 errors, both `E501` (line too long):
- `src/boost_and_broadside/modes/elo_calibrate.py:233` (103 > 100 chars)
- `tests/train/test_bradley_terry.py:160` (101 > 100 chars)

**`uv run --no-sync ruff format --check .`** — 5 files would be reformatted:
`src/boost_and_broadside/modes/elo_calibrate.py`, `src/boost_and_broadside/modes/elo_calibrate_plots.py`,
`tests/env/test_rewards.py`, `tests/modes/test_elo_calibrate.py`, `tests/train/test_bradley_terry.py`.
77 files already formatted. These are pure formatter-noise and are not repeated as
individual findings above (per audit instructions) — run `ruff format .` to fix, and
`ruff check --fix .`/manual line-wrapping for the two `E501`s, as a single mechanical
cleanup commit.

**`uv run --no-sync pytest -q`** — **256 passed**, 0 failed, in 90.9s. Full suite is green
at the start of this audit.
