# Portfolio Cleanup Audit

Systematic review of the codebase against STYLE_GUIDE.md and general portfolio-readiness,
performed 2026-07-15. **Nothing has been changed** — this document only records what to do.

**Importance ratings**

| Rating | Meaning |
|---|---|
| **Critical** | Behavioral bug or something that makes the project look broken to a reviewer |
| **High** | First things a reviewer will notice: stale README, dead modules, 2800-line files |
| **Medium** | Clear style-guide violations and drift worth fixing before publishing |
| **Low** | Polish; fix opportunistically |

---

## 1. Correctness risks (verify before any refactor)

### 1.1 Schedule group-scales silently don't apply to 18 of 21 reward components — **Critical**
[ppo.py:1535-1543](src/boost_and_broadside/train/rl/ppo.py#L1535-L1543) does
`setattr(comp, f"{comp.name}_weight", raw * group_scale)`, but most components expose
`weight` as a property reading `self._weight` (e.g. [rewards.py:91-96](src/boost_and_broadside/env/rewards.py#L91-L96)).
The setattr just creates an unused attribute; only `facing`, `closing_speed`, `shoot_quality`
happen to have matching attribute names. `wrapper.refresh_component_weights()` and the
lambda-aggregation `comp_weights` in `_update_epochs` therefore never see scheduled scale changes.
*Currently latent* (all runs use `constant(1.0)` scales) but the whole mechanism is broken.
**Action:** give `RewardComponent` a plain mutable `weight` attribute (delete the 21 property
boilerplates) and set it directly; add a test that scheduled scales actually change effective weights.

### 1.2 `enemy_win` zero-sum lambda is documented everywhere but never applied — **High**
`_LOCAL_COMPONENTS` in [ppo.py:167-187](src/boost_and_broadside/train/rl/ppo.py#L167-L187)
includes `ally_win`/`enemy_win`, and local (diagonal) lambda takes precedence in
`_precompute_lambda_aggregates`. So `enemy_win` is self-only, making it an exact duplicate signal
of `ally_win` — the "lambda=-1 lets the critic distinguish win/draw/loss" design described in
[README.md](README.md), [rewards.py:324-332](src/boost_and_broadside/env/rewards.py#L324-L332) and
[core.py](src/boost_and_broadside/config/core.py) is not what the code does. `runs/shared.py`
also omits `enemy_win` from `enemy_neg_lambda_components` while tests and `rl_obstacles.py` include it.
**Action:** decide the intended semantics, then fix either the code or every doc/config that
describes it. Add a regression test for the lambda matrix of win components.
**Decision (2026-07-15): (a) restore the zero-sum design** — the intended semantics stand.
Remove the win components (`ally_win`, `enemy_win`) from `_LOCAL_COMPONENTS` in ppo.py so they
use the team-based lambda path, and add `enemy_win` to `enemy_neg_lambda_components` in
`runs/shared.py` so it gets lambda=−1 for enemies (matching tests and `rl_obstacles.py`). This is
a **live reward-signal change** — pair it with the win-component lambda-matrix regression test and
a smoke run to confirm stability. Reconcile all configs so they agree on the win-component lambda set.

### 1.3 Dead schedule/config fields imply features that don't exist — **High**
- `allow_scripted_in_roster` and `elo_eval_games` are resolved every update
  ([ppo.py:210-233](src/boost_and_broadside/train/rl/ppo.py#L210-L233)) and set in every run
  profile, but never read.
- `scripted_roster_min_steps` (TrainConfig) is set in all 5 run profiles and referenced in
  comments/docstrings ("scripted entry is added lazily after…") but never used — the scripted
  roster entry is **never added**, contradicting [roster.py:44-47](src/boost_and_broadside/train/rl/roster.py#L44-L47).
- `EloRoster.update_elo()` ([roster.py:201-222](src/boost_and_broadside/train/rl/roster.py#L201-L222))
  is never called — ELO updates moved on-GPU into `train()`.

**Action:** delete the dead fields/method from configs, schedule, run profiles and tests,
or re-implement the scripted-roster feature if it's still wanted.

### 1.4 `_decode_targets_to_obs` uses world width for the y-axis — **Medium**
[agent_factory.py:269-273](src/boost_and_broadside/modes/agent_factory.py#L269-L273):
`W = ship_config.world_size[0]` is used to decode both `pos_x` and `pos_y`.
Latent (world is square) but wrong. **Action:** use `world_size[1]` for y.

### 1.5 Stale docstring shape comments — **Low**
E.g. `store_initial_hidden` says `(1, B*N, D)` but state is `(n_layers, B*num_tokens, CONV_KERNEL*D)`
([buffer.py:303-309](src/boost_and_broadside/train/rl/buffer.py#L303-L309));
[ppo.py:2481](src/boost_and_broadside/train/rl/ppo.py#L2481) says `# (16,) cpu` for a 9-dim tensor.
**Action:** sweep shape comments during the ppo.py refactor.

---

## 2. Dead code

| What | Where | Action | Importance |
|---|---|---|---|
| `config/obs_spec.py` — 347-line `ObsConfig` system, only re-exported, used by nothing (yet documented as *the* observation pipeline in README) | [obs_spec.py](src/boost_and_broadside/config/obs_spec.py), [config/__init__.py](src/boost_and_broadside/config/__init__.py) | Delete module + re-exports; fix `ModelConfig` docstring that references it | **High** |
| `relational_features_head.py` — orphaned scratch script at repo root, never imported, has unused imports/vars | [relational_features_head.py](relational_features_head.py) | Delete (or move to a `sketches/` dir if you must keep it) | **High** |
| `runs/rl_hpc.py` — `RL_HPC_TRAIN_CONFIG` imported nowhere; no `rl_hpc` mode in main.py; README lists it | [rl_hpc.py](runs/rl_hpc.py) | Delete, or wire up a mode; fix README either way | **High** |
| `Directional` transform + `VelocityPredictor` — unused classes | [features.py:185-196](src/boost_and_broadside/train/rl/features.py#L185-L196), [features.py:315-344](src/boost_and_broadside/train/rl/features.py#L315-L344) | Delete | Medium |
| `base_rewards` pytest fixture — constructs `RewardConfig` with ~14 fields that no longer exist; used by zero tests (would `TypeError` if used) | [tests/conftest.py](tests/conftest.py) | Delete | Medium |
| Commented-out `ScaleConfig` block | [runs/rl.py:68-71](runs/rl.py#L68-L71) | Delete | Medium |
| Deprecated `obs_config=None  # deprecated, ignored` parameter (style guide §6.8 forbids this) | [ppo.py:326](src/boost_and_broadside/train/rl/ppo.py#L326) | Delete parameter | Medium |
| Legacy obs-key compatibility: `_LEGACY_KEY_MAP`, `from_dict` legacy branch; features.py still uses legacy string names `"prev_power"` etc. | [observation.py:21-25](src/boost_and_broadside/env/observation.py#L21-L25), [features.py:756-771](src/boost_and_broadside/train/rl/features.py#L756-L771) | Make features.py use `Accessor(ObsKey.PREVIOUS_ACTION, [i])` names only, then delete the legacy map (§6.8) | Medium |
| Leftover tombstone comment `# _run_matchup was removed…` while the module docstring still says it exports `_run_matchup` | [collect.py:4,27](src/boost_and_broadside/modes/collect.py#L27) | Delete comment, fix docstring | Low |
| `SIGReg` machinery always instantiated though `sigreg_coef=0.0` in every profile | [ppo.py:429](src/boost_and_broadside/train/rl/ppo.py#L429), [sigreg.py](src/boost_and_broadside/train/rl/sigreg.py) | **Decision (2026-07-15): keep** as a config-gated feature. (1) Add a short README/docstring note that it exists and is off by default. (2) Verify the disabled-path cost stays ≈ one `if`: confirmed 2026-07-15 that `need_sigreg=False` skips the compute *and* stops `z` being returned from `evaluate_actions`, and that `self.sigreg` holds no optimizer params — leave that gating intact, don't regress it in the ppo.py split. The only always-on remnant is `self.sigreg` init (tiny buffers, negligible); optionally make it lazy. | Low |

---

## 3. Documentation drift

### 3.1 README.md — multiple sections describe a previous generation of the code — **High**
- **Observations section is wrong end-to-end**: documents `ObsConfig` in `runs/shared.py`,
  `raw_dim = 105`, transforms `FourierAngle`, `QuarterWaveFourier`, `VecMag`, `Bucketize`,
  `Clamp`, `AsFloat`, and `Fourier(10, …)` — none exist. Live system is
  `build_standard_coordinator()` in [features.py](src/boost_and_broadside/train/rl/features.py)
  with `Fourier(4)`, `UnitCircle`, `SymlogVelocity`, per-feature `label_scale`.
- **NextStateHead**: README says `AUX_PRED_DIM=10` incl. an alive-logit BCE; actual prediction
  dim is 9, no alive logit.
- **Rewards**: says "19 components / K=19 critic heads"; there are 21 registered components
  (K = active subset). Global/local split disagrees with `_GROUP`/`_LOCAL_COMPONENTS`
  (facing/closing_speed/shoot_quality are local in code, global in README).
- **Modes table** missing `feature_stats`, `ar_report`, `noise_calibration`; project structure
  lists `feature_stats.py` nowhere, lists `rl_hpc.py` (dead), `obs_spec.py` (dead).
- **Agent specs** list 5 specs; [agent_factory.py](src/boost_and_broadside/modes/agent_factory.py)
  supports 13 (`jouster`, `boom_zoom`, `scripted_team`, …). Same gap in main.py's docstring.
- **Commands**: `uv run main.py` / `uv run pytest` contradict STYLE_GUIDE §6.7 (`uv run --no-sync`).
- **"140 tests"** — suite currently collects 156.
- **`bc_warmstart`** described as "BC for 50M steps" here, in main.py and in
  [bc_warmstart.py](runs/bc_warmstart.py) docstring — config says `total_timesteps=20M`.

**Action:** rewrite the stale sections from the current code (this is the highest-value doc fix
for a portfolio — the README is the first and often only thing reviewers read).

### 3.2 Stale module docstrings — **High**
- [ppo.py:1-7](src/boost_and_broadside/train/rl/ppo.py#L1-L7): "Zero Mamba, zero auxiliary
  losses. One clean loop" — the file contains BC loss, next-state aux loss, windowed loss,
  SIGReg, avg-model, league play, in-rollout ELO eval.
- [rewards.py:1-20](src/boost_and_broadside/env/rewards.py#L1): "9-component decomposed critic"
  (docstring), "Construct the 9 reward components" / "List of all 11 RewardComponent instances"
  ([rewards.py:1029-1039](src/boost_and_broadside/env/rewards.py#L1029-L1039)) — actual: 21.
  Same "11-component" claim in [core.py:118](src/boost_and_broadside/config/core.py#L118).
- [wrapper.py:6](src/boost_and_broadside/env/wrapper.py#L6) says it computes "zero-sum rewards";
  step() explicitly returns non-zero-sum component rewards.
- [agent_factory.py:1-9](src/boost_and_broadside/modes/agent_factory.py#L1-L9) and
  [interactive.py:1-12](src/boost_and_broadside/modes/interactive.py#L1-L12) list only 5 agent specs.

**Action:** fix during the corresponding code cleanups; grep for hardcoded component counts.

### 3.3 ROADMAP.md — **High**
Raw brain-dump with typos ("attension", "spacial", "perterbations"), items completed long ago
("Change world to 1024x1024"), and items referencing deleted systems (BC data collection).
**Action:** rewrite as a short, honest "Future work" list (or delete). For a portfolio, a
polished 10-line roadmap beats this.

### 3.4 proposals/data_pipeline_refactor.md — **Medium**
Analyzes `src/modes/collect.py`, `src/data_collector.py`, pickle checkpoints — architecture that
no longer exists. **Action:** delete, or move to a clearly-labeled `docs/archive/`.

### 3.5 docs/game_design.md vs actual config — **Low**
Documents bullet energy cost `3.0` and the head-on damage-reduction mechanic, but
[runs/shared.py:9](runs/shared.py#L9) overrides `bullet_energy_cost=2` and
`bullet_min_damage_frac=1.0` (mechanic disabled in training).
**Action:** add a note that training config overrides these defaults.

---

## 4. Structure: oversized files & duplication

### 4.1 Split `ppo.py` (2839 lines; `train()` alone is ~880 lines) — **High**
Violates §6.2 (40-50-line functions, ≤3 nesting levels) and §6.6 (300-400-line files) about as
hard as possible. Natural seams, in rough order of extraction value:
1. **In-training ELO evaluation** (eval env setup at [ppo.py:844-899](src/boost_and_broadside/train/rl/ppo.py#L844-L899),
   the `_step % 4` block at 1320-1452, the flush at 1454-1485, `_compute_optimal_eval_ratio*`) →
   `train/rl/elo_eval.py` class. Removes ~350 lines and 2 nesting levels from `train()`.
2. **Opponent management** (league sampling, avg-model update/accumulation, opponent overrides,
   `_opp_team_flag`) → `train/rl/opponents.py`.
3. **Checkpointing** (`_checkpoint_payload*`, `_save_*`, `load_*` — ~250 lines) → `train/rl/checkpoint.py`.
4. **Metrics assembly** (the ~200-line block building `metrics` after the update, overview
   remapping, `_log_worker`/W&B init) → `train/rl/logging.py`.
5. **Rollout step** (the body of the `for _step` loop) → method(s) of ~40 lines each.
Also fixes: duplicated CUDA-stream/CPU-fallback branches at
[ppo.py:1114-1184](src/boost_and_broadside/train/rl/ppo.py#L1114-L1184) (identical logic twice).

### 4.2 De-duplicate `main.py` mode dispatch — **Medium**
The `bc` / `rl` / `rl_obstacles` cases are ~90% identical ([main.py:226-284](main.py#L226-L284));
`bc_warmstart` repeats it twice more. Extract a `_make_trainer(config, args) -> PPOTrainer` helper.
Also: 7 occurrences of the mangled one-liner `ship_config=SHIP_CONFIG,                device=device,`
(e.g. [main.py:235](main.py#L235)) — a formatter artifact that looks bad; running `ruff format` fixes it.

### 4.3 Reward component boilerplate — **Medium**
21 components each repeat `__init__(self, weight)` + `@property weight`
([rewards.py](src/boost_and_broadside/env/rewards.py)); several `compute()` bodies are identical
(AllyDamage=EnemyDamage=DamageTaken, AllyDeath=EnemyDeath=LocalDeath), and the toroidal
diff/dist/valid-mask block is copy-pasted in ~6 components.
**Action:** put `weight` on the base class (also fixes finding 1.1), share the identical computes,
extract a `_toroidal_offsets(pos, world_size)` helper.

### 4.4 `TensorState` field-list duplication — **Medium**
The 24-field constructor is written out three times: `_slice_state`
([ppo.py:96-123](src/boost_and_broadside/train/rl/ppo.py#L96-L123)),
`_make_prev_state_proxy` ([wrapper.py:401-436](src/boost_and_broadside/env/wrapper.py#L401-L436)),
and `make_state` in tests. Any new state field must be added in 4 places.
**Action:** `dataclasses.replace(state, ship_health=…, ship_alive=…)` for the proxy; a generic
`dataclasses.fields`-driven slice helper (or a `TensorState.slice_envs()` method) for slicing.

### 4.5 Positional 14-tuple minibatches — **Medium**
`RolloutBuffer.get_minibatch_iterator` yields 14-element tuples; ppo.py indexes them as
`chunk[6]`, `chunk[8]`… with comments explaining which index is which
([ppo.py:1760-1761](src/boost_and_broadside/train/rl/ppo.py#L1760-L1761)).
**Action:** a `NamedTuple`/dataclass `MicroBatch` — self-documenting and type-checkable.

### 4.6 `_obs_from_state` duplicates `MVPEnvWrapper._get_obs` — **Medium**
[collect.py](src/boost_and_broadside/modes/collect.py) reimplements observation building
("Mirrors _get_obs() exactly") and ppo.py imports it *from a mode module* into the training path.
**Action:** move to `env/` (e.g. a `TensorState → MVPObservation` function next to the wrapper)
and have both call sites share it.

### 4.7 Misc structure smells — **Low**
- Private cross-module access: `entry._policy` (ppo), `wrapper._all_components`,
  `wrapper._active_components`, `scaler._p5/_p95`, `adv_scaler._rms`,
  `coordinator._label_scale_vector` (agent_factory), `_infer_team_pma_k` imported by elo_stats,
  `trainer._shutdown()/_run_name/_checkpoint_payload()` from main.py. Promote to public API or move logic inward.
- Duplicated ELO expected-score formula in ≥4 places (`_compute_optimal_eval_ratio`, its tensor
  twin, two inline blocks in `train()`, `roster.update_elo`). One helper.
- The 15-dim target-vector layout is hardcoded in three places
  ([agent_factory.py:260-262](src/boost_and_broadside/modes/agent_factory.py#L260-L262),
  noise_calibration docstring, decode logic) — derive from the coordinator instead.

---

## 5. Style-guide violations (mechanical)

### 5.1 Old-style typing — **Medium** (§4)
`Tuple[...]` / `Union[...]` / `typing.Generator` etc. in:
[stochastic_config.py](src/boost_and_broadside/agents/stochastic_config.py) (13×`Tuple`),
[physics.py:9,334](src/boost_and_broadside/env/physics.py#L9),
[buffer.py:16](src/boost_and_broadside/train/rl/buffer.py#L16),
[obs_spec.py](src/boost_and_broadside/config/obs_spec.py) (dies with the module).
Missing annotations on several public functions (`load_policy(model_config, coordinator, device…)`,
`_apply_smoke(config)`, `get_actions(..., device)`, `build_standard_coordinator(ship_config)`).
**Action:** modernize to `tuple[float, float]` etc.; add the missing hints. `ruff --select UP` automates most of it.

### 5.2 ruff violations: 41 errors — **Medium** — ✅ done (bdadebc, ecceaf6, 9365ab4)
18 unused imports, 14 module-import-not-at-top (the `_EPS = …` wedged between imports pattern in
[wrapper.py:14](src/boost_and_broadside/env/wrapper.py#L14),
[physics.py:13](src/boost_and_broadside/env/physics.py#L13),
[collect.py:12](src/boost_and_broadside/modes/collect.py#L12)), 4 unused variables,
4 empty f-strings, 1 multi-statement line. 22 are auto-fixable.
**Action:** move `_EPS` to `constants.py` (it's duplicated 4×), then `ruff check --fix` + manual pass.

**Done:** `_EPS` consolidated into `constants.EPS` (obs_spec.py's copy left — dead code, Session 2
deletes it). `ruff check --fix` (95 autofixes) + `ruff format` (49 files) + a manual pass for
F841/E501 that ruff couldn't autofix. Remaining ruff state: 2 errors total (1 F841 + 1 UP007),
both in `relational_features_head.py`/`obs_spec.py` (Session 2 deletes both files), plus 26 E501
long-line violations deliberately left in files owned by other sessions — see the note added to
this doc's §5.2 action and the per-file list in the `style: manual ruff cleanup pass` commit
message. 156/156 tests pass throughout.

### 5.3 Magic numbers — **Medium** (§6.3)
Worst offenders, all in [ppo.py](src/boost_and_broadside/train/rl/ppo.py):
- `S_eval = 512` eval envs (×3), `K_eval = 4.0`, eval every `_step % 4` — not in any config;
  ignores `--smoke` (a smoke run still allocates 1536 eval envs). The unused `elo_eval_games`
  schedule field (finding 1.3) looks like it was *meant* to control this.
- The ELO-gated target-kl override `0.02 if elo_norm >= 900.0` duplicated at
  [ppo.py:1550](src/boost_and_broadside/train/rl/ppo.py#L1550) and
  [ppo.py:2253](src/boost_and_broadside/train/rl/ppo.py#L2253) — belongs in the schedule config.
- BC decay sigmoid constants `950.0`, `200.0` (line 1530-1532); scripted anchor ELO `1000` in 4 places;
  deque `maxlen=100`; histogram cadence `update % 10`.
Elsewhere: `Normalize(40.0)` and the `label_scale` calibration constants in
[features.py](src/boost_and_broadside/train/rl/features.py) (at minimum name them);
watch-mode `EnvConfig(num_ships=8, …)` and eval-mode `1024*4` envs inline in [main.py](main.py).
**Action:** promote to `TrainConfig`/schedule/constants with names; wire `elo_eval_games` up or delete it.

### 5.4 Config defaults contradict the guide *and* their own docstrings — **Medium** (§6.3)
`TrainConfig` docstring says "No defaults — all values required" directly above 7 defaulted
fields ([training.py:111-134](src/boost_and_broadside/config/training.py#L111-L134));
`RewardConfig` likewise defaults 9 weights ([core.py:188-198](src/boost_and_broadside/config/core.py#L188-L198));
`ReturnScaler`/`AdvantageScaler` default `ema_alpha`/`min_rms` in code.
**Action (per §8.1 decision — amend the guide, keep the defaults):** do **not** strip the
defaults. Fix only the contradicting docstrings — `TrainConfig`/`RewardConfig` must stop claiming
"No defaults — all values required" — and add the §6.3 carve-out to STYLE_GUIDE. `ReturnScaler`/
`AdvantageScaler` code defaults are fine to keep. Fix the docstrings either way.

### 5.5 Comment hygiene — **Low** (§7)
Mostly good, but: narrating comments that restate code ("# Build and copy checkpoints to CPU
synchronously on the main thread (very fast, ~5-10ms)"), the contradictory `_LOCAL_COMPONENTS`
comment ("These must match the local_scale entries above" — `ally_win`/`enemy_win` don't),
`import math  # needed for math.ceil in generate()` in obstacle_cache.py, and stray double
blank lines / trailing whitespace in ppo.py (285-288, 616-618, 899-901) and main.py (394).

### 5.6 Naming — **Low** (§3)
Generally strong. Exceptions: `runs/bc.py` `num_envs=3 * _MAX_TOKENS // 3 // 8` (obfuscated
arithmetic); `_MAX_TOKENS` means "tokens per update" in rl.py but effectively "env count basis"
in bc.py — same name, different unit; single letters `sc`/`w`/`d`/`a` in ppo.py flush loop;
`ns`/`bc`/`sc` abbreviation soup in `_compute_minibatch_loss` (acceptable but dense).

---

## 6. Repo hygiene & packaging

| Item | Detail | Action | Importance |
|---|---|---|---|
| `pyproject.toml` placeholder description | `description = "Add your description here"` | Write a real one-liner | **High** — ✅ done (e894c6a) |
| Unused dependencies | `h5py`, `scikit-learn`, `scipy` imported nowhere; `ruff` listed as a *runtime* dep; `pytest` itself missing from dev group | Prune deps; move ruff to `[dependency-groups].dev`; add pytest | **High** — ✅ done (e894c6a) |
| Tracked junk files | `error.log` (traceback referencing long-deleted `src/modes/train.py`), `mdpdf.log` | `git rm`, add `*.log` to .gitignore | **High** — ✅ done (eeae409) |
| No ruff/formatter config | STYLE_GUIDE mandates 100-char lines & import order, but there's no `[tool.ruff]` section, so ruff runs with defaults and nothing is enforced | Add `[tool.ruff]` (line-length 100, `select = ["E","F","I","UP"]`), run `ruff format`, consider pre-commit | **High** — ✅ done (e894c6a config; format/fix pass tracked separately below) |
| `.env` tracked | Contains `PYTHONPATH=src` — redundant with `pytest.ini` `pythonpath` and the installed package; env files shouldn't be committed | Delete + gitignore `.env` | Medium — ✅ done (eeae409) |
| `.vscode/` tracked but gitignored | `.gitignore` lists `.vscode/` yet `launch.json`/`settings.json` are tracked (added earlier) | **Decision (2026-07-15):** `git rm --cached .vscode/*` (stop tracking, keep the ignore rule) | Medium — ✅ done (eeae409) |
| `src/__init__.py` tracked | Makes `src` itself a package — wrong with `packages.find where=["src"]` layout | Delete | Medium — ✅ done (eeae409) |
| `.claude/settings.local.json` tracked | "local" settings are per-machine by convention | Untrack + gitignore | Low — ✅ done (eeae409) |
| Stray empty `models/` dir at root | Gitignored but confusing next to `src/.../models/` | Delete locally | Low — ✅ done (local only, not a git change) |
| `checkpoints/` + `wandb/` bulk | Properly gitignored (good), but ~300 run dirs locally; fresh clones look nothing like your working copy | No repo action; consider a `runs/README` note on artifacts | Low |

---

## 7. Tests

- **Suite health: all 156 tests pass** (72 s wall-clock, verified 2026-07-15).
  README's "140 tests" needs updating.
- **Tests pollute the working tree:** running the suite writes real checkpoints into the
  repo's `checkpoints/` directory (e.g. `checkpoints/20260715-110917/best_training.pt` from
  `test_ppo.py`). Point checkpoint dirs at pytest's `tmp_path` fixture. — **Medium** —
  ✅ done (e81696a): all 8 `PPOTrainer`/`_make_trainer` call sites in `test_ppo.py` now pass
  `checkpoint_dir=str(tmp_path)`; verified `checkpoints/` dir count unchanged across a full run.
- **Dead fixture** `base_rewards` with nonexistent fields (finding 2, table).
- **Coverage gaps worth closing for the portfolio story:**
  - No test that scheduled group scales reach effective component weights (would have caught 1.1).
  - No test of the lambda matrix for win components (would have caught 1.2).
  - `roster.py` (sampling, eviction, persistence) and `schedule.py` primitives (`join`,
    `exponential`) have no dedicated tests.
- Style guide §6.9 ("one logical assertion", parametrize) is mostly followed — keep it.

---

## 8. STYLE_GUIDE.md updates (make the guide match reality where reality is right)

1. **§6.3 "No Defaults for Hyperparameters"** — **Decision (2026-07-15): (a) amend the guide.**
   Add: *"Fields that disable an optional feature may default to the disabled value (0.0 / None).
   All active hyperparameters must be explicit."* This legitimizes `RewardConfig`/`TrainConfig`
   obstacle & aux-loss defaults instead of fighting them. Keep the existing defaults; do **not**
   strip them (that was option (b), rejected). Still fix the contradicting docstrings in §5.4 so
   they no longer claim "No defaults — all values required" above defaulted fields.
2. **§2 tooling** — the guide suggests isort/ruff as a "tip"; make it binding and reference the
   (new) `[tool.ruff]` config so consistency is enforced, not aspirational.
3. **§6.7 entry point** — README and guide disagree (`uv run` vs `uv run --no-sync`); pick one
   form and use it in both.
4. **§6.4 shape comments** — add the rule that shape comments must be updated when shapes change
   (several stale ones found), and name the extra letters actually in use (`M` = obstacles,
   `T+1` obs convention).
5. **§5 docstrings** — add: *"Never hardcode counts in prose (component counts, test counts,
   step totals) — they rot."* This single rule covers findings 3.1/3.2.
6. Optionally document the observation/feature pipeline (`FeatureCoordinator`) as the canonical
   encoding layer, since the guide predates it.

---

## Suggested execution order

1. **Hygiene sweep** (§6 + ruff autofixes) — one small PR, zero risk, immediate visual payoff.
2. **Dead code removal** (§2 + dead fields from 1.3) — shrinks the surface before refactoring.
3. **Correctness fixes** (1.1, 1.2, 1.4) with new regression tests.
4. **ppo.py split + main.py dedup** (§4) — the big one; do it after 1–3 so you're not moving
   dead or broken code.
5. **Docs rewrite** (README, ROADMAP, docstrings) — last, so docs describe the final state.
6. **STYLE_GUIDE amendments** (§8) alongside step 5.

---

*Baseline verified before any cleanup: `uv run --no-sync pytest` → 156 passed, 0 failed
(2026-07-15). Re-run after each step above.*
