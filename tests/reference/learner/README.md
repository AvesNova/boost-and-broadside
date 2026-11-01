# Learner equivalence fixtures (section 6)

Goldens computed in float64 on the CPU from v1's own code at tag `v1-torch-final`:
per-level GAE, team mixing with the level sum, the clipped policy and entropy terms, the
critic losses and the shot cross-entropies. The rest of the learner (masks and
denominators, micro-batch accumulation, labels, streaming) is tested by the section's own
tests against independent computations.

- **Run:** `uv run --no-sync pytest tests/reference/learner`. The tests skip until
  `learner_arm.make_arm()` returns an implementation.
- **Fixed:** the tests, helpers, conftest and goldens. Only `learner_arm.py` is written
  by the section.
- **Validated:** the tests pass on v1, and every deliberate mutant fails by a value
  comparison.
- **Tolerance:** float64, 1e-9 relative to each array's own scale.

## Conventions

- **Shapes:** time-major, `(T, B, N, ...)` per ship, with the `K` reward levels last,
  in v2's order (damage, kill_death, objective_progress, objective, outcome).
- **`terminated[t]`:** the episode ended on step `t`, so the trace is cut after `t` and
  nothing bootstraps across it (a timeout is a real end).

## Interface

- `per_level_gae(rewards, values, terminated, next_value, next_done, gamma, lam)`:
  `(advantages, returns)`. Here `gamma` and `lam` are `(K,)`, and `next_value (B, N, K)`
  with `next_done (B,)` is the bootstrap after the last step. Each level is plain GAE
  with its own discounts; `returns = advantages + values`.
- `team_aggregate(team, alive, advantages, returns, team_spirit)`: the advantage
  `Σ_k (A_ik + s_k · mean_{living teammates j ≠ i} A_jk)`, and the same of the returns.
  Both are `(T, B, N)`.
- `policy_loss(logp, old_logp, advantage, return_scale, mask, clip)`: the masked mean of
  `max(−Â r, −Â clip(r, 1 ± clip))`, with `Â = advantage / return_scale` and
  `r = exp(logp − old_logp)`.
- `entropy_term(entropy, mask)`: minus the masked mean entropy.
- `critic_losses(local_logits, outcome_logits, local_returns, outcome_targets, valued,
  bins, local_levels, outcome_level)`: `(K,)`, one loss per level.
  - Per-ship levels: the two-hot cross-entropy of the realised return, averaged over
    valued ships (alive, or waiting to respawn).
  - The outcome: the cross-entropy against its categorical targets, averaged over
    environment-steps.
- `shot_cross_entropy(trajectory_logits, trajectory_target, outcome_logits,
  outcome_target, runs)`: the trajectory cross-entropy per part `(..., parts)`, each
  summed over its softmax groups, and the outcome cross-entropy `(...)`.
