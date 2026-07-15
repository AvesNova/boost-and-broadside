# Cleanup Session Prompt

> **How to use (human):** Start a fresh thread and either paste this whole file, or just say:
> *"Read docs/cleanup_prompt.md — you are Session N."* Then fill in / answer anything listed
> under **Assignment**. Sessions must run in order; each assumes the previous ones are committed.

---

## Assignment — EDIT THIS SECTION

You are running **Session ___** from the session map below.

All four project-wide decisions are already resolved — see *Resolved decisions* at the bottom of
this file. You do not need to ask about them; just follow them.

Extra instructions for this session: *(none)*

---

## Context

- Goal: make this repo portfolio-ready. The full issue list — with file/line locations, required
  actions, and importance ratings — lives in **docs/cleanup_audit.md**. That document is the
  single source of truth; read it in full before touching code.
- Progress is tracked *inside* the audit doc (see Rules). Check its ✅ marks to confirm the
  previous sessions actually completed before you start.
- Test baseline: **156/156 tests passing** as of 2026-07-15, ~72 s wall-clock.

## Session map

| Session | Model tier | Audit items (from docs/cleanup_audit.md) | Verification |
|---|---|---|---|
| **1 — Hygiene & tooling** | Sonnet | All of **§6** (pyproject description, prune unused deps, untrack logs/.env/src/__init__.py, `.vscode` → `git rm --cached .vscode/*`, .gitignore fixes) · **§5.2** (add `[tool.ruff]`: line-length 100, `select = ["E","F","I","UP"]`; `ruff check --fix`; `ruff format` the repo; move `_EPS` to constants.py) · **§7** test-checkpoint pollution (point test checkpoint dirs at `tmp_path`) | pytest (smoke optional — format is behavior-neutral) |
| **2 — Dead code** | Sonnet | All of **§2** (obs_spec.py, relational_features_head.py, rl_hpc.py, Directional/VelocityPredictor, base_rewards fixture, commented ScaleConfig, deprecated obs_config param, legacy obs keys, tombstone comment) · **SIGReg — keep**: add a README/docstring note and keep the disabled-path gating intact (verified ≈ one `if`); do *not* delete · **§1.3** (dead schedule/config fields + roster.update_elo, across all run profiles and tests) | pytest + smoke |
| **3 — Correctness fixes** | **Fable** | **§1.1** together with **§4.3** (plain `weight` attr on RewardComponent base, delete 21 property boilerplates, share duplicate computes, toroidal helper) + regression test that scheduled group scales reach effective weights · **§1.2 — decision (a) restore zero-sum**: remove win components from `_LOCAL_COMPONENTS`, add `enemy_win` to `enemy_neg_lambda_components` in shared.py, reconcile all configs + lambda-matrix regression test (note: live reward-signal change) · **§1.4** (y-axis decode) · **§7** coverage gaps (roster.py + schedule.py tests) | pytest + smoke |
| **4 — ppo.py decomposition** | **Fable** | **§4.1** (extract elo_eval.py, opponents.py, checkpoint.py, logging.py; break up `train()`; unify stream/CPU branches) · **§4.4** (TensorState slice/replace helpers) · **§4.5** (MicroBatch NamedTuple) · **§4.7** (private-access cleanup, ELO-formula dedup) · **§5.3** ppo.py magic numbers (S_eval, target_kl override → schedule, BC sigmoid constants, wire or delete elo_eval_games) · **§1.5** shape-comment sweep | pytest + smoke; **strictly behavior-identical** — no logic changes mixed into moves |
| **5 — Remaining structure & style** | Sonnet | **§4.2** (main.py dedup via `_make_trainer`) · **§4.6** (`_obs_from_state` → env/) · **§5.1** (modern typing + missing annotations) · **§5.4 — decision (a) keep defaults**: do NOT strip config defaults; only fix the docstrings that wrongly claim "No defaults" · **§5.5** (comment hygiene) · **§5.6** (naming: `_MAX_TOKENS`, bc.py arithmetic) | pytest + smoke |
| **6 — Docs & style guide** | Fable for README (§3.1); rest Sonnet | **§3.1** README rewrite (verify every claim against current code — counts, modes, specs, obs pipeline, aux dims) · **§3.2** module docstrings · **§3.3** ROADMAP rewrite · **§3.4** archive/delete proposal doc · **§3.5** game_design note · **§8** STYLE_GUIDE amendments · final pass over cleanup_audit.md (every item ✅ or ⏭) | pytest (docs only) |

## Rules

1. **Scope:** execute only your assigned items. Do not fix things belonging to other sessions,
   even if you notice them — see Rule 8.
2. **Behavior preservation:** all changes must be behavior-preserving **except** items in audit
   §1, which are explicit bug fixes and require regression tests.
3. **Commits:** one commit per finding or small finding-group, so failures can be bisected.
   Reference the audit item in the message, e.g. `cleanup: delete dead config/obs_spec.py (audit §2)`.
4. **Verification before every commit:** `uv run --no-sync pytest -q` must pass. Run the smoke
   test — `uv run --no-sync main.py --mode rl --smoke` — when your session's verification column
   requires it: at minimum once at the end of the session, and after any individually risky commit.
   Always use `uv run --no-sync`, never bare `python`/`pytest`.
5. **Progress tracking:** after finishing an item, edit **docs/cleanup_audit.md** in place —
   append `— ✅ done (<short-hash>)` to that finding's heading or table row. For items
   deliberately skipped: `— ⏭ won't fix: <one-line reason>`. **Never delete findings.**
   Commit audit-doc updates together with (or immediately after) the code change they record.
6. **Decisions:** the four project-wide decisions are already settled in *Resolved decisions*
   below — follow them, don't re-litigate. If you hit a *new* decision they don't cover, stop and
   ask the user rather than guessing.
7. **Partial completion:** if you can't finish (context/time), commit what's done, mark those
   items ✅, annotate the unfinished item in the audit doc with exactly what remains, and say so
   clearly in your final summary. This matters most for Session 4.
8. **New discoveries:** don't fix them. Append them to a `## Discovered during cleanup` section
   at the end of docs/cleanup_audit.md with the same format (location, action, importance).
9. **Artifacts:** never commit `checkpoints/`, `wandb/`, or stray run dirs. Until Session 1's
   tmp_path fix lands, test runs will create `checkpoints/<timestamp>/` dirs — leave them alone.
10. **End-of-session report:** items completed with commit hashes, decisions made, test + smoke
    results, anything deferred or discovered.

## Resolved decisions (settled 2026-07-15 — do not re-litigate)

| Decision | Needed by | **Resolution** |
|---|---|---|
| **§1.2 enemy_win semantics** | Session 3 | **(a) restore zero-sum.** Remove `ally_win`/`enemy_win` from `_LOCAL_COMPONENTS` in ppo.py; add `enemy_win` to `enemy_neg_lambda_components` in `runs/shared.py`; reconcile every config to agree on the win-component lambda set. This is a live reward-signal change — ship it with the lambda-matrix regression test and a smoke run. |
| **SIGReg** | Session 2 | **Keep** as a config-gated feature. Add a README/docstring note that it exists and is off by default. Preserve the disabled-path gating (verified ≈ one `if`: skips compute and skips returning `z`; no optimizer params). Do not delete; optionally make `self.sigreg` init lazy. |
| **.vscode tracking** | Session 1 | **`git rm --cached .vscode/*`** — stop tracking, keep the `.gitignore` rule. |
| **§8.1 defaults carve-out** | Sessions 5 & 6 | **(a) amend the guide.** Add "disabled-value defaults (0.0 / None) are allowed; active hyperparameters must be explicit" to STYLE_GUIDE §6.3. **Keep** the existing `TrainConfig`/`RewardConfig` defaults — do not strip them. Fix only the docstrings that contradict this. |
