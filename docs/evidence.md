# Documentation evidence map

This file is the working source-of-truth for documentation claims and assets. It is not
intended to be a reader-facing explanation of the project. It records what may safely be
said, where the evidence lives, and what still needs qualification or follow-up.

The completed cross-document check is recorded in the [verification report](verification.md).

- Audit date: 2026-07-28
- Code baseline: `2592ebdce35269c908c191a70725743886f96b93`
- Landmark run: `resilient-resonance-682` (`chpl40cj`)

## Status and evidence rules

- **Verified** means the claim was checked against executable code, stored configuration,
  raw result data, or a reproducible test.
- **Derived** means the value follows directly from verified data, but is not stored in
  that form. The derivation must be stated.
- **Interpretation** means the evidence is real but the explanation is a proposed reading,
  not a measured causal conclusion.
- **Gap** means the repository does not currently preserve enough provenance to make the
  claim without qualification.
- **Stale** means existing prose conflicts with current code or the landmark-run artifact
  and must not be carried into the new documentation.

For the landmark run, its exported configuration and saved checkpoint take precedence
over today's run profiles. Current source code is authoritative for current behavior, but
some training settings and checkpoint conventions have changed since the run was made.
Plots are treated as presentations of their underlying JSON or W&B exports, not as
independent evidence.

## Narrative-ready claims

| Topic | Safe wording | Status and evidence |
|---|---|---|
| Project | Boost and Broadside is a tensorized, GPU-oriented 2D team dogfighting environment and recurrent PPO training system with scripted opponents, self-play, league play, and post-hoc rating tools. | **Verified.** [environment](../src/boost_and_broadside/env/env.py), [trainer](../src/boost_and_broadside/train/rl/ppo.py), [opponents](../src/boost_and_broadside/train/rl/opponents.py), and [calibration](../src/boost_and_broadside/modes/elo_calibrate.py). |
| Controller semantics | One centralized recurrent policy jointly controls every ship on the learned team. A single network evaluation sees all entity tokens and emits one factored action per ship. It is not one neural-network instance per ship. | **Verified.** [policy action/value pass](../src/boost_and_broadside/models/yemong/policy.py) and [rollout policy pass](../src/boost_and_broadside/train/rl/ppo.py). |
| Native training scale | The landmark policy trained with eight ships total, split evenly into two teams: 4-vs-4. | **Verified.** [exported run config](../checkpoints/resilient-resonance-682/wandb_export/config.json) records `num_ships=8`; [environment reset](../src/boost_and_broadside/env/env.py) performs the team split. Avoid calling this an “8-vs-8 training run.” |
| Zero-shot scaling | The same saved weights can be evaluated at different team sizes because ships and obstacles are entity tokens and no learned parameter is sized by ship count. The recorded crossover sweep evaluates one checkpoint with learned-team sizes from 1 through 64 without retraining. | **Verified.** [policy](../src/boost_and_broadside/models/yemong/policy.py), [capture mode](../src/boost_and_broadside/modes/capture.py), [crossover evaluator](../src/boost_and_broadside/modes/crossover.py), and [crossover data](crossover/crossover.json). Prefer “zero-shot team-size transfer” over the ambiguous “8-vs-11 agents.” |
| 8-vs-11 result | With eight policy-controlled ships against 11 scripted-controlled ships, the stored crossover sweep reports a 69.5% learned-team win rate. At 12 scripted ships it falls to 42.2%. | **Verified.** [crossover data](crossover/crossover.json), row `trained=8`. The JSON stores the rates and a run-level maximum of 256 games; current evaluator logic implies 256 games for these 19/20-ship matchups, but the per-row game count is not stored. Do not print an exact `178/256` count until provenance is added. |
| Broad crossover | In the stored sweep, 16 policy-controlled ships remain above 50% through 24 scripted ships, 32 through 47, and 64 through 87. | **Verified.** [crossover data](crossover/crossover.json). These are empirical boundary estimates, not proof of a monotonic scaling law. |
| Final rating | The final live policy is estimated at 2052.9 ELO and the scripted agent at 1240.0 on the post-hoc half-win scale anchored so random reads zero: a difference of about 813 ELO. | **Verified/derived.** [calibration result](../checkpoints/resilient-resonance-682/elo_calibrated.json). The live estimate has 18.4 ELO conditional standard error; scripted has 6.2. The shared zero-point shift carries another 32.8 ELO of uncertainty but cancels in their difference. |
| Training run | The landmark run targeted one billion environment steps and logged 999,424,000. It completed in about 7.50 hours at a final logged 37,000 environment steps/s and 296,002 ship tokens/s on one RTX 5090. | **Verified.** [run config](../checkpoints/resilient-resonance-682/wandb_export/config.json), [summary](../checkpoints/resilient-resonance-682/wandb_export/summary.json), and [hardware metadata](../checkpoints/resilient-resonance-682/wandb_export/files/wandb-metadata.json). These are run-specific measurements, not general performance guarantees. |
| Scripted benchmark | Sampled training history first reaches 95% win rate against the scripted opponent at 127.9M steps and 99% at 221.9M; the final sampled point is 100%. | **Verified.** Derived from 999 sampled points in [W&B history](../checkpoints/resilient-resonance-682/wandb_export/history.jsonl). It is not monotonically saturated: the minimum sampled value after 200M is 89%, so “stays near 100% after 200M” is too strong. |

Recommended thesis sentence:

> A single recurrent team policy, trained only in 4-vs-4 combat, transfers without
> retraining to much larger and asymmetric battles—jointly controlling every ship on the
> learned team and defeating larger scripted teams across the recorded crossover sweep.

This is supported as an empirical result. Claims that the experiment proves a general
law of scale invariance or identifies why transfer occurs would be interpretations.

## Environment and physics

| Claim or topic | Evidence | Status / documentation action |
|---|---|---|
| Parallel simulation state and physics are tensors; hot physics has no Python loop over environments or ships. | [state](../src/boost_and_broadside/env/state.py), [environment](../src/boost_and_broadside/env/env.py), [physics](../src/boost_and_broadside/env/physics.py) | **Verified.** Say “tensorized” or “GPU-oriented,” not that every auxiliary path is loop-free. |
| The world is continuous and toroidal; state advances velocity before position using semi-implicit Euler. | [physics](../src/boost_and_broadside/env/physics.py), [ship config](../src/boost_and_broadside/config/core.py) | **Verified.** |
| The action is factored into power (3), turn (7), and shoot (2), producing 12 logits per ship rather than 42 joint logits. | [constants](../src/boost_and_broadside/constants.py), [policy](../src/boost_and_broadside/models/yemong/policy.py) | **Verified.** Log probabilities are summed across the three categorical factors. |
| A projectile inherits the firing ship's velocity in addition to muzzle velocity. | [shooting physics](../src/boost_and_broadside/env/physics.py) | **Verified.** The pre-rewrite `docs/game_design.md` said bullet speed was relative to the world; that stale statement was removed in [environment.md](environment.md). |
| Projectile-to-ship collision, obstacle collision, and friendly fire are implemented. | [physics](../src/boost_and_broadside/env/physics.py), [obstacle physics](../src/boost_and_broadside/env/obstacle_physics.py), [physics tests](../tests/env/test_physics.py) | **Verified.** |
| Ship-to-ship collision is not implemented in the core physics step. | [physics](../src/boost_and_broadside/env/physics.py) | **Verified absence.** The pre-rewrite wording about a minimal elastic collision was **stale/ambiguous** and was removed in [environment.md](environment.md). |
| A match terminates when an existing team has no surviving ship; reaching the horizon truncates the episode and can produce a draw. | [environment](../src/boost_and_broadside/env/env.py), [capture winner logic](../src/boost_and_broadside/modes/capture.py) | **Verified.** |
| Default angular hit-damage scaling exists, but it was disabled for the landmark combat run. | [physics](../src/boost_and_broadside/env/physics.py), [run config](../checkpoints/resilient-resonance-682/wandb_export/config.json) | **Verified.** The run sets `bullet_min_damage_frac=1.0`; distinguish engine capability from experiment configuration. |

## Observation and model architecture

| Claim or topic | Evidence | Status / documentation action |
|---|---|---|
| Each policy call receives global tokens for all ships and optional obstacles, including kinematics, resources, team/alive state, radius, and previous action. | [observation builder](../src/boost_and_broadside/env/observation.py), [feature coordinator](../src/boost_and_broadside/train/rl/features.py) | **Verified.** “Centralized team policy” is accurate. |
| The canonical encoder uses Fourier phase features for toroidal position/attitude, symlog transforms for velocity/angular velocity, circular resource encodings, and one-hot team/action features. | [feature coordinator](../src/boost_and_broadside/train/rl/features.py) | **Verified.** Keep the exact feature table in the architecture page, not the root README. |
| A Yemong block combines spatial self-attention over entity tokens with per-entity Griffin/RG-LRU temporal recurrence. | [attention](../src/boost_and_broadside/models/yemong/attention.py), [Griffin block](../src/boost_and_broadside/models/yemong/griffin.py), [policy](../src/boost_and_broadside/models/yemong/policy.py) | **Verified.** Obstacles participate in both processing stages; only ship tokens reach the heads. |
| The landmark model uses `d_model=128`, four attention heads, and two Yemong blocks. | [run config](../checkpoints/resilient-resonance-682/wandb_export/config.json) | **Verified run-specific configuration.** |
| The policy has action, decomposed value, and auxiliary next-state prediction heads. Win/loss values use team pooling over alive ships. | [policy](../src/boost_and_broadside/models/yemong/policy.py) | **Verified.** |
| The auxiliary head predicts nine target channels and receives per-step MSE plus a 32-step triangle-window drift loss for position/velocity. | [features](../src/boost_and_broadside/train/rl/features.py), [PPO loss](../src/boost_and_broadside/train/rl/ppo.py) | **Verified current implementation.** Keep changing counts out of general prose where possible. |
| Falling next-state error demonstrates that the auxiliary head learns predictable dynamics channels. | [history](../checkpoints/resilient-resonance-682/wandb_export/history.jsonl), [render script](../scripts/render_charts.py) | **Verified measurement.** Position-x falls from 1.492 to 0.00148; velocity-x from 3.042 to 0.0192; power from 1.465 to 0.0186. Health changes only 0.570 to 0.487. Explaining health's behavior as caused by sparse damage is an **interpretation**, not a causal measurement. |

## Training, rewards, and league system

| Claim or topic | Evidence | Status / documentation action |
|---|---|---|
| Training uses recurrent clipped PPO, per-component GAE/value losses, entropy regularization, win-rate-gated behavior-cloning loss, and auxiliary next-state losses. | [trainer](../src/boost_and_broadside/train/rl/ppo.py), [buffer](../src/boost_and_broadside/train/rl/buffer.py) | **Verified.** SIGReg exists but is disabled for the landmark run; do not present it as an active method. |
| The landmark run uses 7,808 parallel environments, 128 rollout steps, 32 minibatches, and `ego_pass`; its native matchup is 4-vs-4. | [run config](../checkpoints/resilient-resonance-682/wandb_export/config.json) | **Verified run-specific configuration.** |
| `ego_pass` evaluates raw and team-flipped observations in one batched pass. The same weights act for both teams in self-play, while the learning mask assigns policy/BC gradients to the ego side. | [trainer rollout](../src/boost_and_broadside/train/rl/ppo.py), [opponent action merge](../src/boost_and_broadside/train/rl/opponents.py) | **Verified.** Explain carefully; “one policy controls both teams” is true in self-play but misleading as a general project description. |
| The landmark run has 11 active reward components: two outcome terms, three dense combat-shaping terms, two kill-credit terms, three damage-accounting terms, and death. | [run config](../checkpoints/resilient-resonance-682/wandb_export/config.json), [reward registry](../src/boost_and_broadside/env/rewards.py) | **Verified for this run.** General docs should link to the registry/config rather than imply every available component is active. |
| Each active component gets its own critic target and can override gamma/lambda; rewards are divided by total ship count before aggregation for team-size normalization. | [wrapper](../src/boost_and_broadside/env/wrapper.py), [trainer](../src/boost_and_broadside/train/rl/ppo.py), [run reward tables](../runs/shared.py) | **Verified current implementation.** |
| `kill_shot` assigns fatal-step credit in proportion to damage dealt on that step; `kill_assist` uses cumulative episode damage. | [rewards](../src/boost_and_broadside/env/rewards.py) | **Verified.** A comment in `runs/shared.py` describes kill-shot credit as winner-take-all and is **stale**. |
| The current opponent schedule begins with 50% scripted games, then introduces average-policy and league opponents while retaining self-play. | [run profile](../runs/rl.py), [opponents](../src/boost_and_broadside/train/rl/opponents.py) | **Verified for the current profile.** The W&B export does not preserve callable schedule definitions, so do not assert that this exact current schedule produced the landmark results without checking its saved checkpoint. |
| The average policy is a uniform running mean of post-cutoff snapshots. Historical checkpoint entries remain in the roster; `league_size` limits the GPU-resident policy cache, not the number of roster entries. | [opponents](../src/boost_and_broadside/train/rl/opponents.py), [roster](../src/boost_and_broadside/train/rl/roster.py) | **Verified current behavior.** The pre-rewrite README's eviction claim was **stale** and is not carried forward. |
| The random anchor and historical/average policies live in the roster. The scripted agent is used directly as a scheduled opponent and evaluator, not currently sampled as a roster entry. | [roster](../src/boost_and_broadside/train/rl/roster.py), [opponents](../src/boost_and_broadside/train/rl/opponents.py) | **Verified current behavior.** The pre-rewrite README wording was **stale** and is not carried forward. |
| Current continuous evaluation uses five matchup slots: live-vs-anchor, live-vs-floating, live-vs-scripted, live-vs-average, and floating-vs-anchor. | [ELO evaluator](../src/boost_and_broadside/train/rl/elo_eval.py) | **Verified current behavior.** The pre-rewrite README's three-matchup description was **stale** and is not carried forward. |
| Frozen opponents are sampled by ELO proximity, proportional to `exp(-abs(delta)/temperature)`, excluding the random anchor. | [roster](../src/boost_and_broadside/train/rl/roster.py) | **Verified.** |
| Post-hoc calibration plays a stationary tournament, fits Bradley-Terry ratings under two tie conventions, then refits each historical live record against calibrated opponents. | [calibration mode](../src/boost_and_broadside/modes/elo_calibrate.py), [Bradley-Terry fit](../src/boost_and_broadside/train/rl/bradley_terry.py) | **Verified.** Half-win is the primary reported convention; decisive-only is a diagnostic. |

## Result interpretation and uncertainty

### Crossover sweep

The central zero-shot result is supported by [the raw crossover JSON](crossover/crossover.json)
and [the evaluator](../src/boost_and_broadside/modes/crossover.py). Ties count against the
learned team. The search records the largest scripted team with at least 50% learned-team
wins and the first tested adjacent count below 50%.

The current chart line is drawn halfway between those two integer counts. That is a useful
visual convention, not a fitted continuous threshold. The current JSON does not store
per-matchup game counts, seeds, checkpoint hash, source commit, or confidence intervals.
The run-level `num_envs=256` is only a maximum because large battles reduce the batch to
fit the collision-memory budget.

Safe headline examples:

- “Eight policy-controlled ships beat 11 scripted ships in 69.5% of the recorded games.”
- “Across learned-team sizes 1–64, one checkpoint transfers without retraining and the
  empirical crossover stays above numerical parity.”

Avoid “eight agents” unless immediately defined, “scale invariant” as a proven property,
or a smooth/precise threshold implied by the interpolated chart line.

### ELO calibration

[The calibration result](../checkpoints/resilient-resonance-682/elo_calibrated.json) is
converged to its configured target for stationary players. The final live-policy point is
not itself a stationary tournament player: it is reconstructed from that update's 627-game
record and has a wider conditional standard error.

Important distinctions:

- final live estimate at 999,424,000 steps: 2052.95 ±18.41 conditional SE;
- last frozen ladder checkpoint at 876,494,848 steps: 2056.79 ±9.50 conditional SE;
- scripted agent: 1240.03 ±6.20 conditional SE over 7,895 tournament games;
- all absolute ratings share a ±32.81 zero-point uncertainty after shifting random to zero;
- the in-training dashboard's final live rating (1547.28) is a drifting online estimate,
  not the post-hoc calibrated result.

The README may round the final live policy to “about 2053 ELO” and the lead over scripted
to “about 813 ELO,” provided it links to the evaluation methodology. Do not mix the live
curve endpoint, last frozen-checkpoint rating, and online training rating.

### Interpretations to label as interpretations

- The continued ELO rise after the scripted win-rate curve becomes a weak discriminator is
  consistent with self-play/league learning beyond what the scripted baseline measures.
  It does not isolate which training mechanism caused the improvement.
- Stronger crossover ratios at some mid-sized battles may reflect coordinated control,
  but the existing evaluation does not separate coordination from per-ship tactical skill
  or scripted-controller weaknesses.
- The next-state head's health error is plausibly dominated by sparse, hard-to-forecast
  damage events, but no ablation establishes that cause.

## Replay evidence

### Selected hero: 8 policy-controlled vs 11 scripted-controlled ships

Asset: [vs_scripted_8v11_seed03.gif](results/replays/vs_scripted_8v11_seed03.gif)

- **What is verified:** capture code assigns the learned policy to blue/team 0 and the
  scripted controller to red/team 1; one policy instance produces actions for every blue
  ship. The GIF is 480×480, 6.29 seconds, and about 1.5 MiB. Its terminal frame contains
  three surviving blue ships and no red ships, so this replay is a learned-team win.
- **What it demonstrates:** the controller-semantics clarification and asymmetric
  zero-shot story in one compact visual. It is the preferred README hero.
- **Provenance gap:** the GIF has no sidecar or embedded metadata naming the source
  checkpoint/run, source commit, capture arguments, or machine-readable outcome. The
  filename alone is not sufficient evidence of those details. Until regenerated or
  matched byte-for-byte to a provenance-bearing source, caption it as a replay from the
  included learned policy without asserting an exact checkpoint step.
- **Recommended caption:** “One recurrent policy jointly controls all eight blue ships;
  the red team's 11 ships use the scripted controller. This seeded replay ends with three
  blue ships surviving.”

[Capture mode](../src/boost_and_broadside/modes/capture.py) seeds each game, uses the final
`step_*.pt` checkpoint of the selected run, records the terminal winner, and holds the
terminal state for readability. Future curated clips should preserve that printed metadata
in a JSON sidecar.

### Supporting replay choices

| Asset | Role | Status / caution |
|---|---|---|
| [64-vs-80 scripted](results/replays/vs_scripted_64v80_seed01.gif) | Deep evaluation/replay page: large-scale qualitative example. | 480×480, 6.08 s, about 5.3 MiB. Use a poster or click-through rather than auto-embedding beside the hero. Same provenance gap as above. |
| [32-vs-44 scripted](results/replays/vs_scripted_32v44_seed04.gif) | Supporting asymmetric example at an intermediate scale. | About 3.7 MiB. Useful only if it adds a distinct behavior; do not repeat the same visual claim three times. |
| [4-vs-5 scripted seeds 00/04](results/replays/vs_scripted_4v5_seed00.gif) | Candidate pair for qualitative variability at near-native scale. | Outcomes and behavior labels need explicit review before use. |
| [self-play clips](results/replays/self_8v8_seed01.gif) | Replay gallery: show the same weights controlling both perspectives under the team flip. | Do not present self-play footage as independent benchmark evidence. |

## Visual asset plan and provenance

| Asset | Intended location | What it supports | Required action |
|---|---|---|---|
| [8-vs-11 replay](results/replays/vs_scripted_8v11_seed03.gif) | README hero; replays page | One-policy-per-team semantics, asymmetric zero-shot transfer, qualitative result | Usable now with the qualified caption above. Add metadata sidecar when regenerated. |
| [crossover phase plot](results/crossover_phase.png) | README results; evaluation page | Central zero-shot thesis across team sizes | **Relabel before final publication if possible:** “trained agents” → “policy-controlled ships” and “scripted agents” → “scripted-controlled ships.” Until then, the surrounding caption must define the unit. Add uncertainty in a later revision. |
| [crossover ratio plot](results/crossover_ratio.png) | Evaluation page only | Compact view of numerical advantage | Same terminology correction. Treat the claimed mid-size peak/easing as descriptive, not a fitted trend. |
| [calibrated ELO curve](results/elo_curve.png) | README secondary result; evaluation page | Improvement over training and scripted landmark | Underlying values are verified. The simplified README chart omits uncertainty; replace later with a compact uncertainty-aware render or explicitly link to methodology. |
| [scripted win-rate curve](results/win_rate_vs_scripted.png) | Evaluation page; optional small README panel | Scripted baseline becomes saturated before training ends | Usable. Caption should avoid implying monotonic saturation. |
| [policy diagram](policy_architecture.png) | Architecture page; optional compact README overview | Spatial-attention + temporal-recurrence structure and three heads | Existing raster is tall and manually maintained. Regenerate as a wide diagram later and avoid hard-coded reward-component counts. Not blocking. |
| [next-state error](results/next_state_error.png) | Architecture/evaluation deep section | Auxiliary dynamics learning | Usable with factual channel descriptions; label causal explanations as interpretations. |
| [training health](results/training_health.png) | Training/evaluation appendix | Optimization diagnostics | Keep out of the concise README unless space remains. |
| [autoregressive reports](ar_report/) and [noise calibration](noise_calibration/) | Deep analysis links | Prediction behavior and calibration diagnostics | Preserve as technical artifacts; summarize rather than duplicate them. |

## Deferred asset and analysis ledger

No deferred item blocks the documentation rewrite. The crossover terminology is the one
publication-facing issue that should be fixed first, but a precise caption is an acceptable
temporary mitigation.

| Priority | Needed item | Narrative gap filled | Planned location | Blocking? |
|---|---|---|---|---|
| High | Relabeled crossover plots with “policy-controlled ships” / “scripted-controlled ships” | Prevents the one-network-per-ship misreading | README and evaluation | No; caption around current plot for now |
| High | Preserve `games`, seed policy, checkpoint SHA/hash, source commit, and uncertainty per crossover matchup | Makes the central quantitative result independently auditable | `crossover.json` schema and evaluation | No; qualify current numbers |
| High | Replay metadata sidecars and poster frames | Makes outcome/provenance machine-readable and keeps large media lightweight | Replays page and README | No |
| Medium | Wide, count-agnostic system diagram | Explains simulation → observation → team policy → training/evaluation at a glance | README/architecture | No |
| Medium | Boundary or failure replay near an empirical crossover | Shows limitations and prevents cherry-picked qualitative storytelling | Replays/evaluation | No |
| Medium | Compact uncertainty-aware ELO figure | Communicates conditional SE and the shared anchor offset | Evaluation; optional README | No |
| Low | Confidence intervals or repeated-seed analysis for crossover boundary | Distinguishes sampling noise from a robust scale trend | Evaluation | No for initial docs; needed for stronger analytical claims |

## Documentation corrections applied

The Phase 4 rewrite applied these evidence-audit requirements:

- Ambiguous “multi-agent” language near the top was replaced with an immediate definition
  of policy-controlled ships and one centralized team controller.
- The landmark configuration is described as eight ships total, 4-vs-4—not 8-vs-8.
- The README claim that weak roster members are evicted beyond `league_size` was removed.
- The old three-matchup ELO description was replaced with the verified current five-slot
  evaluator, while keeping the landmark-run methodology separate.
- The scripted controller is not listed as a current roster entry.
- `recent_avg.pt` is no longer presented as a general current checkpoint convention. It exists in the
  landmark artifact, but current saves use scheduled `avg_step_*.pt` files.
- Bullet velocity and the absence of ship-to-ship collision are corrected in
  [environment.md](environment.md), which replaces `game_design.md`.
- Inactive reward families are not described as though they trained the landmark policy.
- Kill-shot credit is not called winner-take-all.
- Current source settings remain distinct from the landmark artifact where they differ. For
  example, the landmark export records a 100-ELO milestone gap while today's primary run
  profile uses 200; newer rollout-sharding options are also absent from the export.
- `capture`, `crossover`, and `elo_calibrate` are documented in context in the setup and
  evaluation pages rather than in another large root table.

## Setup, commands, and engineering checks

| Check | Result | Evidence / limitation |
|---|---|---|
| Supported Python | Python 3.13+ | [package metadata](../pyproject.toml); audit ran on Python 3.13.11. |
| CLI help | Pass | `main.py --help` imports and lists all current modes. The sandbox required a temporary uv cache because the normal home cache is read-only. |
| Test suite | Pass | 354 passed, six hardware-specific tests skipped, 72 expected CUDA-autocast warnings on a CPU-visible test process; 14.06 s. |
| Lint | Pass | `ruff check .` reports “All checks passed.” |
| Fresh install | Not exercised | `uv sync` may require package/network access and would be better checked in a clean environment or CI. |
| Training smoke test | Not exercised in this audit | It mutates checkpoint/log output and is more expensive than documentation inspection. Run before publishing setup guarantees. |
| Interactive watch/capture regeneration | Not exercised | Requires display/GPU/ffmpeg conditions beyond the unit suite. Existing `ffmpeg` can inspect the curated GIFs. |
| Git LFS requirement | Configured | [`.gitattributes`](../.gitattributes) tracks `*.pt`; a clean-clone/LFS pull was not performed. |

Repository-level gaps that affect contribution/open-source wording:

- no `LICENSE` file;
- no contribution guide;
- no CI workflow under `.github/`;
- no replay or crossover provenance manifest.

Until those are resolved, avoid licensing badges, “contributions welcome” promises, or
claims that every setup path is continuously verified.

## Asset decisions used by the rewrite

The initial documentation stack uses the current assets without making new production a
prerequisite:

1. use the 8-vs-11 GIF as the single README hero;
2. use crossover as the central quantitative result, with explicit controller semantics;
3. use calibrated ELO as secondary evidence of learning progression;
4. move training-health and prediction diagnostics to supporting pages;
5. record new asset requests in the deferred ledger and integrate them later without
   restructuring the narrative.
