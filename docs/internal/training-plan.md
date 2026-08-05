# Training-efficiency plan

Working plan for the sample-efficiency and wall-clock work started on
`fix/league-unification-scaler-floors`. Internal: it records reasoning and
measurements, not reader-facing documentation. Reader-facing material lives in
[training.md](../training.md).

Written so a fresh session can pick this up without re-deriving anything.

---

## Where the time goes

Measured on the 8 GB dev card ([memory-optimization.md](../engineering/memory-optimization.md)),
and consistent with a first-principles token count:

| phase | share |
|---|---:|
| rollout (env + opponent forwards + Elo battery) | **15%** |
| PPO update (epochs × minibatches × micro-batches) | **85%** |

Consequences that drive everything below:

- Wall-clock work belongs in the update phase. The Elo battery, the scripted
  agent and the stream overlap all live inside the 15%.
- Sample efficiency and wall clock are the same lever: anything that extracts
  more learning per token also shortens the run at fixed final Elo.

---

## Done

Seven commits, oldest first. Each entry records *why*, because the reasoning is
harder to recover than the diff.

### `8f8e612` — advantage scaler floor

`AdvantageScaler.min_rms` defaulted to 0.1 with no config path, against true
per-component advantage RMS of 0.0075 (win) to 0.27 (damage). The floor bound
permanently on **six of eleven** components, so `normalize()` divided them by the
guard rather than their own statistics — exactly what the scaler exists to
prevent. Win components were taking ~22% of the policy gradient against the ~62%
their weights ask for.

Dropped to a 1e-4 epsilon, plumbed as `TrainConfig.advantage_min_rms`. No
loss-magnitude risk: the aggregated advantage is renormalized to unit RMS *after*
lambda aggregation, so this changes the mix, not the scale. Verified against
`--smoke`: total loss unchanged at 8.2.

**`return_min_span` was deliberately left at 1.0.** Lowering it is not the same
kind of fix — `ReturnScaler` divides the whole return distribution by a robust
p5–p95 half-span, so sparse components with heavy tails produce very large
normalized targets and the squared value loss follows. Measured **~11× on
`loss/value` at production spans and ~400× in `--smoke`**, which against
`max_grad_norm=1.0` (grad norm 0.65) would make clipping bind every step and
silently cut the effective learning rate. **Block C resolves this** — a two-hot
critic bins in a fixed support and never divides by a span at all.

### `c6610b2` — unified opponent league

Three opponent env groups were each sized at their peak scheduled fraction and
then only ever switched fully on or off, so `scripted_fraction` stepping 0.5 →
0.3 at 50M **never actually stepped down**. The reference run trained against
scripted in 50% of envs for its whole length with self-play at 10% rather than
30% — and by then the scripted matchup sat at a 100% win rate, so half the batch
produced no outcome signal.

Replaced with self-play plus one league whose width is a single `league_fraction`,
split into `league_slots` slots that each draw by Elo proximity every rollout.
The scripted agent became an ordinary roster entry.

Two more bugs fixed here: the `avg` entry's rating was set once at creation and
**never updated**, and a bullet-reading league entry in a bullet-free run *raised
mid-rollout* instead of being retired.

### `11a00ea` — GAE truncation and conv-buffer boundaries

**C1:** GAE keyed its non-terminal mask on physics `dones`, but the wrapper
auto-resets before returning the observation, so a *truncated* episode
bootstrapped off the freshly spawned one. At γ=0.999 that leaked the length of
the trace. Now keyed on `terminated`.

**C2:** `reset_hidden_for_envs` zeroes the whole packed hidden — RG-LRU state
*and* conv buffer — but `forward_sequence` applied `done_mask` only inside the
RG-LRU, so re-evaluation read the previous episode's inputs for `CONV_KERNEL-1`
steps after every boundary and the PPO ratio was wrong there. The boundary mask
depends on the output step **and** the tap's lag, so no input-side mask expresses
it; the conv is now an explicit sum of K shifted taps with a per-(step, lag)
validity mask. **Verified by reverting the fix with the new test in place.**

**M2:** removed the blocking `.cpu()` once per optimizer step.

### `c9e2c41` — generalized measurement ladder

`MAX_ANCHORS = 2` was hardwired into a binary `torch.where` and a single-threshold
draw, with a comment warning that raising it fails *silently*. Now a multinomial
draw over information weights and a gather.

Anchors split into **stationary references** (permanent prefix) and **checkpoint
anchors** (rotating). Stationary references cost no forward pass: every
semi-random rung is a Bernoulli blend of the same two action tensors, so a ladder
of any length is one scripted call plus one random call.

### `33dbef4` — 20 Hz decisions

Physics ran at 1/60 and the policy chose a new action every tick, far finer than
the plant responds to. `EnvConfig.action_repeat=3` holds each action three
physics ticks. Physics, collisions and projectile integration are untouched.

| timescale | seconds | decisions @ 20 Hz |
|---|---:|---:|
| firing cooldown | 0.10 | 2 |
| bullet flight to ~200 px | 0.40 | 8 |
| full 360° turn | 1.3–2.3 | 26–46 |
| mean episode | ~4.7 | ~93 |
| `num_steps=128` rollout | 6.4 | 128 |

At 60 Hz five of six shoot decisions were no-ops against the cooldown, and a
128-step rollout spanned 2.1 s against a ~4.7 s episode — **BPTT never covered a
whole episode**. It now does, and a token buys three times the game time.

- Rewards **summed** across held ticks (scale-preserving over fixed game time).
- Every γ and λ re-derived as `value ** (60/20)`; horizons in seconds are now
  written down in `runs/shared.py`.
- `max_episode_steps` needed **no** change — it counts physics ticks.
- Spawn health/power/cooldown randomised, balanced in expectation across teams.
- `total_timesteps` → 333M: same game time as the 1e9-step reference run.

### `6f8e61e` — reference ladder and scripted-anchored gauge

With only random and scripted as fixed references the live policy saturates both
for the whole early climb, so its rating is barely identified exactly when
opponent sampling and the milestone grid depend on it. Nine semi-random rungs are
now fixed roster entries and anchor-pool members.

**Ratings are environment-specific, and the recompute proved how much:**

| | old 60 Hz | `rl` @20 Hz | `rl_fields` @20 Hz |
|---|---:|---:|---:|
| random | −550.7 | **−350.8** | **+172.7** |
| 0.5 rung | 193.9 | 299.7 | 578.0 |

The tick rate moved random 200 Elo; **fields move it 520** (they compress the
skill scale). Each profile carries its own ladder. Re-run
`--mode semi_random --profile <name>` whenever tick rate, field count, ship
config or fleet size moves.

Gauge is now **scripted pinned at 1000**, matching post-hoc calibration, so
normalized Elo collapses to the rating itself and the milestone grid is absolute.
Slot 2 updates the live policy rather than scripted.

### `a32e178` — scripted controller ignores fields

Its stay-on-your-side steering measured **net-negative**: 2.24 interface
crossings per thousand ship-steps against uniform-random's 1.60, and mean log
index +0.159 against +0.108. Both occupancy artifacts (a ship at index *n* takes
*n*× as long to cross the same ground), not decisions — the bias capped at 35% of
turn intent and only acted within ~120 px of an interface, so pursuit overwhelmed
it. Since crossing costs health that `field_damage_taken` penalises, BC was
imprinting a habit RL had to unlearn.

Two replacements were tried and both failed: `grad(n)` is zero throughout a flat
core so it never reaches a ship in slow medium; the index-difference version
measured worse still. **The metric is also confounded** — time-weighted index is
mechanically biased toward slow media — so a *path-weighted* measure is needed
before attempting this again.

Field representation never depended on it: the auxiliary next-state head predicts
`local_log_index` directly (prediction dim 10 of 10), which cannot be done
without locating the ship against every field, and unlike the BC weight that
pressure never decays.

### `38a5c22` — win-rate KL gate, derived aux names, tournament defaults

- `high_elo_threshold` → `high_winrate_threshold` at 0.8. Both gates now read the
  raw scripted win rate, so the trust region needs no re-deriving when the gauge
  moves. 0.8 reproduces the original firing point.
- `_NS_FEAT_NAMES` hand-listed 9 names against a computed width of 10, silently
  dropping `local_log_index`. Names now come from `coordinator.get_feature_names()`.
- `--mode semi_random` defaulted to a hardcoded run name; `--profile rl` now.

### `55eeaed` — vectorized field maps, refreshed per rollout

Maps came from a bank of 512 built once by a CPU loop with an `.item()` per draw;
a full run drew each thousands of times. Generation now loops over **fields**,
not maps or retries: every field proposes `max_generation_attempts` placements
for all maps at once and takes the first that fits. No data-dependent control
flow, so no host sync. **4 ms per refresh**, called at the top of every rollout —
roughly one distinct map per episode.

`validate_field_layout` costs **eight** device-to-host syncs and raises, so it
cannot run on the hot path. Startup still validates (an unplaceable config is an
error); refreshes keep the previous row for any map that exhausts its budget and
report `physics/field_map_generation_failures`.

Verified: 25 consecutive refreshes × 512 maps all pass the strict validator;
three refreshes yield 1536 distinct layouts.

---

## Remaining blocks

Hard ordering constraints: **C before D** (the categorical critic changes what
every value target means; don't debug latent collapse and a rescaled critic in
one run). A and B are independent of each other and of C.

Roughly one run each. Budget is tight, so each block is a coherent theme that can
be judged as one thing.

### Block A — action distribution

**Joint 2×3×7 = 42-way action head.** Product distributions are a strict subset:
a factored policy *cannot* represent "boost-straight or coast-turn, never
boost-turn", and real dogfighting policies are full of that. Cost is 30 extra
logits.

- `flat_action_sampling=True` **already exists** on `StochasticScriptedAgent` and
  produces the exact joint as the outer product of the three independent
  marginals, so the BC target is exact and already implemented.
- `RolloutBuffer.expert_probs` widens 12 → 42; `POWER_SLICE`/`TURN_SLICE`/
  `SHOOT_SLICE` become a joint layout.

**Per-factor entropy control.** Measured collapse by 10% of the run: power 14% of
max, turn 19%, **shoot 9%**. One coefficient cannot hold three distributions with
maxima `ln 3`, `ln 7`, `ln 2` at sensible floors.

With a joint head, compute marginals per state by reshaping 42 → (3,7,2) and
summing the other two axes, then take entropy of each. Two notes:

- A single `H(joint)` bonus is **insufficient** — max is `ln 42 = 3.74`, and the
  optimizer can satisfy any target on it while shoot stays pinned.
- Marginals are **coupled** under a joint head (moving shoot mass perturbs turn
  and power), so three dual controllers can oscillate. **Start with fixed
  per-factor coefficients**, not duals. `H(marginal_i) ≤ H(joint) ≤ Σ H(marginal_i)`,
  so per-factor floors imply a loose joint floor.
- Regularizing marginals does *not* push toward independence (uniform marginals
  are compatible with strong coupling), so the correlation modelling survives.

**M3** folds in here: the per-head softmaxes stop being throwaway diagnostics.

**Watch:** per-factor marginal entropies at their floors; `policy/kl` rising off
~0.005 (it was suppressed *by* the collapse — a near-deterministic categorical
has small KL under logit perturbation); `policy/clip_fraction` near 0.08.

This block is what makes the win-rate KL gate meaningful — neither KL threshold
has ever bound.

### Block B — token allocation

**`shared_pass`.** In `ego_pass` only team-0 ships carry actor gradient. That is
*correct* in league envs (team 1 is the opponent) but wastes half the ships in
self-play envs. Under `shared_pass` a self-play env yields `N` actor tokens
against `N/2`:

| | league envs | self-play envs | actor tokens @ 75/25 |
|---|---|---|---:|
| `ego_pass` | N/2 | N/2 | **0.5N** |
| `shared_pass` | N/2 | N | **0.875N** |

**1.75×**, plus the rollout forward halves (2B → B). `_opp_team_flag` plumbing
already exists and is tested.

**`league_fraction` → 0.25.** Coupled to `shared_pass`: only under it does a
self-play env yield 2× the actor tokens of a league env. Under `ego_pass` they
are equal and the tilt has no efficiency basis.

**Caveat:** both teams in a self-play env come from *one* episode and are
anticorrelated on outcome, so the win-component gain is well under 1.75×. If Elo
regresses, `shared_pass` is the suspect (it forfeits the canonical team-0 prior).

**Not doing:** hiding enemy pending actions. The leak is 16.7 ms against a ~0.4 s
bullet flight, symmetric, and it would forfeit `shared_pass`. Block D's
next-action head gets the opponent-modeling benefit without paying that.

### Block C — value head

**Two-hot / HL-Gauss critic over all K**, binned in *symlog* space with a fixed
wide support.

| group | gain |
|---|---|
| win, death, kill_shot, kill_assist | **large** — sparse γ^Δt returns are the worst MSE fit |
| damage dealt/taken | **moderate** — heavy tails, bounded per-sample CE |
| facing, closing_speed, shoot_quality | ~none — bounded and well-scaled already |

Do all K rather than a subset: two value paths and two loss types is more
complexity than uniform treatment.

**This resolves the blocked half of `8f8e612`.** Binning in a fixed support means
never dividing by a span, so `return_min_span` and the `ReturnScaler` critic path
are deleted. `AdvantageScaler` stays (it serves the actor).

**Drop TeamPMA** in the same block — `team_pma_k=()` is already a supported path,
and both changes rewrite the value head, so it is one re-tune instead of two.
**M4** disappears with it. Expected impact small and sign-uncertain in isolation;
the trunk already runs full spatial attention every block, so a ship's embedding
is already team-aggregated.

**Watch:** `loss/value` against `max_grad_norm=1.0`; per-component EV, especially
the win heads; activation memory — `(T,B,N,K,bins)` is ~15% up on the 8 GB card
at 51 bins, drop to 31 if tight. GAE reads `E[v] = Σ pᵢ·binᵢ`.

### Block D — representation

The one-step next-state prediction is a **deterministic function of the
observation** (up to `bullet_spread`), not merely short-horizon: `previous_action`
holds the action *about to be applied* and is visible for every ship, so via
spatial attention each ship sees every pending action. It is nearly trivially
solvable, which is why it is a weak representation signal.

**Iterated action-conditioned k-step latent prediction.** One learned step
function applied k times — *not* k independent heads, which each learn a shortcut
and never form a coherent dynamics model. Condition on the ship's own actions
from the buffer; what remains unpredictable is other agents' behaviour, which is
the signal wanted.

- All ship tokens, both teams. Fields excluded (static within an episode, so
  trivially predictable).
- Target `z[t+k].detach()`, masked by `alive` and by episode boundaries — reuse
  the `max_pool1d` window-validity trick from `_triangle_conv_loss`.
- Cosine loss on L2-normalized latents, substantially more collapse-resistant
  than raw MSE.
- **SIGReg on** at a small coefficient as the collapse guard. Already
  implemented, sitting at `sigreg_coef=0`.
- Keep the 1-step explicit state prediction: grounds the latent in physics,
  nearly free.

**Cost: ~zero extra trunk compute.** `z_0 … z_{T-1}` already exists from the
single sequence forward; this adds one small MLP applied k times and a detached
slice. Best compute-to-signal ratio on the list.

**Enemy next-action prediction head.** Predict the *simultaneous* action `a_t`,
genuinely unknown, with `buffer.actions[t]` as the label. Opponent modeling with
**no observation change**, so it survives `shared_pass`. The degeneracy worry —
that ships learn to act predictably to satisfy the head — is unfounded: a sampled
discrete action index is a constant w.r.t. parameters, so no gradient path rewards
being predictable.

**Not doing: dreaming.** Dreamer's premise is that the env is expensive. Here the
env is 15% of wall clock and the update is 85% — a learned world model would be
*slower* than the real simulator.

---

## Deferred, with reasons

| | why |
|---|---|
| **Team-size / token-split randomization** | "trained only on 4v4" is the stronger claim. Revisit behind an env-fraction gate; note `comp_rewards /= _n_ships` uses the static config count and would need to be per-env. The lambda aggregation is already alive-aware and needs no change. |
| **Variable per-agent step size** | Batching kills the payoff — all ships share one forward, so a subset cannot be skipped. Zero throughput gain, and a ragged buffer breaks the dense `(T,B,N)` layout GAE, the lambda einsum and the scan all assume. |
| **Hiding enemy pending actions** | 16.7 ms, symmetric, and it forfeits `shared_pass`. |
| **Scripted field interaction** | Two attempts measured worse than nothing. Needs a *path-weighted* index metric before retrying; the time-weighted one is confounded. |
| **M1 (Hillis–Steele scan)** | Highest ceiling, most work. **Profile before writing anything** — a `torch.profiler` trace of one update settles whether the bandwidth estimate is right. O(T log T) with ~40 full passes per temporal sublayer; a Blelloch or chunked scan is O(T). |
| **M5** | Check `TORCH_LOGS=graph_breaks` first; Inductor probably already fuses it. |
| **M6** | `torch._foreach_nan_to_num_` does not exist in this build. |

---

## Invariants a fresh session must not break

- **Reference ladders are environment-specific.** Any change to tick rate, field
  count, ship config, fleet size *or the scripted agent* invalidates them. Re-run
  `--mode semi_random --profile <name>` and update `reference_ladder` /
  `random_elo` in the profile.
- **The Elo gauge is absolute**, scripted pinned at 1000. Normalized Elo is the
  rating itself. Milestone grid and any rating threshold are absolute.
- **Both "is it strong yet" gates read the scripted win rate**, not Elo. Keep it
  that way — an Elo threshold needs re-deriving every time the gauge moves.
- **`validate_field_layout` must never run on the hot path** — eight syncs and it
  raises. Generated maps are laminar by construction.
- **Scaler floors are epsilons, not scales.** `scaler/floor_bound_rms/*` binding
  on an active component is a bug. `return_min_span` is the deliberate exception
  until Block C.
- **Discounts encode horizons in seconds.** Changing the tick rate means
  `γ ** (rate_old/rate_new)`, not reusing the number.
- **`max_episode_steps` counts physics ticks**, not decisions.

## Definition of done

`uv run --no-sync pytest`, `--mode rl --smoke`, `--mode rl_fields --smoke`, ruff
check and format on changed files. Add tests, update docs, commit at logical
milestones.
