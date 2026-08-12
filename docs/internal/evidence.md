# Documentation evidence map (internal)

This file is the maintainers' source-of-truth for documentation claims. It is not a
reader-facing explanation of the project. It records what may safely be said, where the
evidence lives, and what still needs qualification or follow-up.

- Reference run: `resilient-resonance-682` (`chpl40cj`)

## Status and evidence rules

- **Verified** means the claim was checked against executable code, stored configuration,
  raw result data, or a reproducible test.
- **Derived** means the value follows directly from verified data, but is not stored in
  that form. The derivation must be stated.
- **Interpretation** means the evidence is real but the explanation is a proposed reading,
  not a measured causal conclusion.
- **Gap** means the repository does not currently preserve enough provenance to make the
  claim without qualification.

For the reference run, its exported configuration and saved checkpoints take precedence
over today's run profiles. Current source code is authoritative for current behavior, but
some training settings and checkpoint conventions have changed since the run was made.
Plots are treated as presentations of their underlying JSON or W&B exports, not as
independent evidence.

## Terminology canon

The reader-facing docs consistently use:

- **policy/controller:** one learned recurrent network acting for a whole team;
- **policy-controlled ship:** one ship receiving one action emitted by that network;
- **scripted-controlled ship / scripted controller:** the hand-built stochastic opponent
  (not "scripted agent");
- **reference run:** `resilient-resonance-682`, the run behind every headline figure
  (not "landmark");
- **4-vs-4 training:** eight total ships in the reference environment (never "8-vs-8");
- **zero-shot team-size transfer:** evaluation at unseen team sizes with unchanged weights;
- **live Elo:** the sequential in-training estimate, on the defined gauge that pins random
  at 0 and scripted at 1000 (not "online Elo");
- **calibrated Elo:** the post-hoc Bradley-Terry reconstruction used for results, reported
  with the scripted controller fixed at 1000 (same convention as the fleet-scale view);
- **Elo**, never "ELO": it is a surname, not an acronym.

## Narrative-ready claims

| Topic | Safe wording | Status and evidence |
|---|---|---|
| Project | Boost and Broadside is a tensorized, GPU-oriented 2D team dogfighting environment and recurrent PPO training system with scripted opponents, self-play, league play, and post-hoc rating tools. | **Verified.** [environment](../../src/boost_and_broadside/env/env.py), [trainer](../../src/boost_and_broadside/train/rl/ppo.py), [opponents](../../src/boost_and_broadside/train/rl/opponents.py), and [calibration](../../src/boost_and_broadside/modes/elo_calibrate.py). |
| Controller semantics | One centralized recurrent policy jointly controls every ship on the learned team. A single network evaluation sees all entity tokens and emits one factored action per ship. It is not one neural-network instance per ship. | **Verified.** [policy action/value pass](../../src/boost_and_broadside/models/yemong/policy.py) and [rollout policy pass](../../src/boost_and_broadside/train/rl/ppo.py). |
| Native training scale | The reference policy trained with eight ships total, split evenly into two teams: 4-vs-4. | **Verified.** [exported run config](../../checkpoints/resilient-resonance-682/wandb_export/config.json) records `num_ships=8`; [environment reset](../../src/boost_and_broadside/env/env.py) performs the team split. |
| Zero-shot scaling | The same saved weights can be evaluated at different team sizes because ships and optional fields are entity tokens and no learned parameter is sized by ship count. The recorded crossover sweep evaluates one checkpoint with learned-team sizes from 1 through 64 without retraining. | **Verified.** [policy](../../src/boost_and_broadside/models/yemong/policy.py), [capture mode](../../src/boost_and_broadside/modes/capture.py), [crossover evaluator](../../src/boost_and_broadside/modes/crossover.py), and [crossover data](../crossover/crossover.json). |
| 8-vs-11 result | With eight policy-controlled ships against 11 scripted-controlled ships, the stored crossover sweep reports a 72.7% learned-team win rate (186 wins, 70 losses, 0 ties in 256 games). At 12 scripted ships it falls to 48.4%. | **Verified.** [crossover data](../crossover/crossover.json), row `trained=8`. Every cell now stores its own win/loss/tie counts and mean episode length, so exact counts may be printed. |
| Broad crossover | In the stored sweep, 16 policy-controlled ships remain above 50% through 24 scripted ships, 32 through 48, and 64 through 88. | **Verified.** [crossover data](../crossover/crossover.json). Empirical boundary estimates, not proof of a monotonic scaling law. The largest matchups run fewer than 256 games per cell (64-vs-88 rests on 173), so each boundary carries several points of binomial noise. |
| Final rating | On the calibrated scale with the scripted controller fixed at 1000, the final checkpoint rates 1772.2 (±7.3 conditional SE, 9,347 tournament games): a lead over scripted of about 772 Elo. The live-policy refit at the same step reads 1771.7 (±18.3, 627 games) and the last frozen ladder checkpoint 1765.5 (±7.2). Random reads −335.3 (±10.5). | **Verified.** The elo-calibration artifact named in [the provenance index](../results/provenance.md), whose tournament fields random, nine semi-random rungs, scripted, the ladder, and the final checkpoint; converged in 10 batches over 163,840 games. Scripted is both anchor and fit reference, so it has no conditional error of its own and every other SE is already relative to it. Cross-check: the independent fleet-scale 4-vs-4 tournament rates the same checkpoint 1787 ± 11. Do not mix in the live training rating (1547.3, a different estimator on a different scale), and do not cite the pre-rung random position (−240 ± 33), which rested on saturated matchups. |
| Fleet-scale ratings | Replayed symmetric tournaments rate the final 4-vs-4 checkpoint 1787 ± 11 at its native scale, rising to 2128 ± 17 at 16-vs-16 and 2220 ± 37 at 32-vs-32; 64-vs-64 falls back to 1958 ± 59. | **Verified.** The elo-scale artifact joined with the semi-random-ladder artifact, both named in [the provenance index](../results/provenance.md). Sizes 16, 32, and 64 exhausted the twelve-batch budget without reaching the ±10 Elo target (`converged: false`), so quote them with their error bars, and do not claim a peak fleet size: 16, 32, and 64 overlap within uncertainty. |
| Training run | The reference run targeted one billion environment steps and logged 999,424,000. It completed in about 7.50 hours at a final logged ~296,000 ship-tokens/s (37,000 env steps/s at 8 ships/env) on one RTX 5090. | **Verified.** [run config](../../checkpoints/resilient-resonance-682/wandb_export/config.json), [summary](../../checkpoints/resilient-resonance-682/wandb_export/summary.json), [hardware metadata](../../checkpoints/resilient-resonance-682/wandb_export/files/wandb-metadata.json). Run-specific measurements, not general performance guarantees. |
| Scripted benchmark | Sampled training history first reaches 95% win rate against the scripted controller at 127.9M steps and 99% at 221.9M; the final sampled point is 100%. | **Verified.** Derived from 999 sampled points in [W&B history](../../checkpoints/resilient-resonance-682/wandb_export/history.jsonl). Not monotonically saturated: the minimum sampled value after 200M is 89%. |

Recommended thesis sentence:

> Trained only in 4-vs-4 combat, the reference policy transfers zero-shot to much larger
> and asymmetric battles, defeating larger scripted fleets across the recorded crossover
> sweep.

This is supported as an empirical result. Claims that the experiment proves a general
law of scale invariance or identifies why transfer occurs would be interpretations.

## Interpretations to label as interpretations

- The continued Elo rise after the scripted win-rate curve becomes a weak discriminator is
  consistent with self-play/league learning beyond what the scripted baseline measures.
  It does not isolate which training mechanism caused the improvement.
- Stronger crossover ratios at some mid-sized battles may reflect coordinated control,
  but the existing evaluation does not separate coordination from per-ship tactical skill
  or scripted-controller weaknesses.
- The next-state head's health error is plausibly dominated by sparse, hard-to-forecast
  damage events, but no ablation establishes that cause.

## Replay evidence

Hero asset: [vs_scripted_8v11_seed03.gif](../results/replays/vs_scripted_8v11_seed03.gif).
Capture code assigns the learned policy to blue/team 0 and the scripted controller to
red/team 1; the terminal frame contains three surviving blue ships and no red ships, so
the replay is a learned-team win. The GIF carries no sidecar or embedded metadata naming
the source checkpoint, commit, capture arguments, or machine-readable outcome; caption
it as a replay of the included learned policy without asserting an exact checkpoint step
until a provenance-bearing capture replaces it. Self-play footage must not be presented
as independent benchmark evidence.

## Deferred asset and analysis ledger

No deferred item blocks the current documentation.

| Priority | Needed item | Narrative gap filled | Planned location | Blocking? |
|---|---|---|---|---|
| High | Rerun/migrate the historical crossover artifact with count records; add seed policy, checkpoint SHA/hash, source commit, and uncertainty | Makes the central quantitative result independently auditable | `crossover.json` schema and evaluation | No; new runs already preserve wins/losses/ties/games |
| High | Replay metadata sidecars and poster frames | Makes outcome/provenance machine-readable and keeps large media lightweight | Replays page and README | No |
| Medium | Wide, count-agnostic system diagram | Explains simulation → observation → team policy → training/evaluation at a glance | README/architecture | No |
| Medium | Boundary or failure replay near an empirical crossover | Shows limitations and prevents cherry-picked qualitative storytelling | Replays/evaluation | No |
| Medium | Compact uncertainty-aware training-progression Elo figure | Adds uncertainty to the historical curve already used in the README | Evaluation; optional README | No |
| Low | Confidence intervals or repeated-seed analysis for crossover boundary | Distinguishes sampling noise from a robust scale trend | Evaluation | No for initial docs; needed for stronger analytical claims |
| Low | Contribution guide and CI workflow | Enables contribution promises and continuously verified setup claims | Repository root | No |
