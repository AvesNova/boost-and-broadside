# Diagnostics audit

Every metric the trainer logs today (`train/rl/*.py`), grouped by prefix, with a first
verdict for the port. Status: **draft, to go through group by group with the author.**
Templated names (`{name}`, `{group}`, ...) expand per reward level, code group,
population or ladder entry, so the real series count is several hundred.

Verdicts: **keep**, **drop** (dead by a decision, constant, or duplicate), **rename**,
**replace** (the thing it measured changed), **check** (needs the author's call).

| Group | Series | What it is | First verdict |
|---|---|---|---|
| `overview/*` | 16 | Curated copies of other series for one dashboard | keep the idea, rebuild from surviving series: drop `score_vs_*`, `bc_coef`, `loss_proxy_bc`, `win_rate_vs_scripted`; add live rating and pool shares |
| `loss/*` | 16 | Unweighted loss per term | keep; drop `sigreg`; drop `next_state_cont` (equal to `next_state`) |
| `loss_proxy/*` | 10 | Weighted loss per term (coefficient x loss) | rename to `loss_weighted/*`; drop `sigreg` |
| `policy/*` | 8 | KL, clip fraction, ratio mean/max, entropy per factor (power, turn, shoot) | keep; add joint 42-way entropy; factor entropies become marginals of the joint |
| `critic/*`, `value/{level}`, `returns/*`, `return_normalizer/*` | ~20 | Explained variance per level, value loss, return/value means, advantage std, percentile scale | keep; add HL-Gauss support use per level (share of target mass in padding bins or clipped at the edge) |
| `outcome/*` | 5 + bins | Predicted vs realised win, Brier, calibration bins | keep |
| `next_state*`, `next_state_ce/{pop}/{group}`, `next_state_baseline_ce/...`, `next_state_population/...` | many | Next-state CE per population and code group, vs the "nothing changes" baseline | keep, consolidate under one prefix; per-population averaging is decided (3c) |
| `belief/{prefix}/{name}` | many | Physical error of the decoded belief (visible / hidden splits) | keep, check naming |
| `enemy_action/*` | 10 | Realised probability, entropy, Brier, accuracy, persistence baseline, uniform baseline | keep; drop the four `uniform_*` series (constants: ln 42 etc., document instead) |
| `global_density/*` + `loss/global_density_*` | 6 | Poisson deviance, ally and enemy | replace: shape KL, count deviance, deviance explained per channel, enemy split by hidden count (3e) |
| `shot/*` | 10 | Shot head realised vs predicted rates, trajectory CE per channel | keep (heads ported as is; item 15) |
| `episode/*`, `episode/event/{name}`, `episode/reward_{name}` | 12 + per level | Win/draw rates, lengths, lifespans, rewards, event counts | keep; `team0_win_rate` / `team1_win_rate` become per-view |
| `fog/*` | 10 | Visibility, occlusion, hidden age, reacquisitions | keep (game analytics) |
| `physics/*`, `frontline/*`, `map_scale/*` | 12 | Speed, power, damage and deaths by source, front advances, respawns, captures, map scale | keep (game analytics) |
| `reward/event_share/*`, `reward/zero_sum_residual/*` | per level | Share of reward by event; zero-sum residual | keep; `zero_sum_residual` depends on tabled item 13 |
| `seat/*` | 2 | `ego_pass` pending-action validity per seat | replace with per-view checks, or drop |
| `schedule/*` | 12 | Current schedule values | keep `learning_rate`, `entropy_coef`, `offensive_bias`, tier scales; drop `bc_cutoff_streak`, `bc_decay_factor`, `behavior_cloning_coef`, `scripted_win_rate`, `policy_gradient_coef`, `target_kl` (all gone or fixed) |
| `train/*` | 11 | Throughput, gradient norms, actor share, clip fire rate, epochs completed, non-finite fraction | keep; `nonfinite_grad_fraction` -> skipped-step count (4d); add time breakdown (rollout, learner, host transfer stall) |
| `counters/*` | 5 | Env steps, tokens, updates, hours | keep |
| `grad_norm/`, `grad_share/`, `grad_cos/`, `grad_coherent_*`, `grad_halves_cos/`, `grad_diag/*` | many | Per-term gradient decomposition at the diagnostic cadence | keep; consolidate under one `grad/` prefix |
| `elo_diag/*`, `eval/*`, `ladder/*`, `two_stage/*`, `live_elo/*`, `matches/*` | ~45 | The dedicated evaluator and its ladder | replace with the in-run joint fit: `rating/live`, `rating/avg`, `rating/stderr`, `pool/share/{kind}`, `matches/{entry}/score` from league games |
| `hist/logprob`, `hist/returns/{level}` | 3 + per level | Histograms every 10 updates | check: likely drop `hist/logprob`; returns histograms either kept at low cadence or replaced by the support-use scalars above |

**Candidates to add** (from decisions so far): skipped-step count and dump path (4d);
carry-staleness check (post-implementation TODO); per-view win rate and actor
share; pool composition and per-entry game counts (4b); HL-Gauss support use for
every code; density deviance explained (3e); timing breakdown including stream
stalls (4d); league drain progress.
