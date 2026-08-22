# Roadmap

Ideas under consideration, roughly ordered by interest. None are commitments.

## Gameplay

- **Kill rewards in-game**: grant energy on kills (paired with a lower base
  regeneration rate and higher energy cap) to reward aggression mechanically
  rather than only through reward shaping.
- **Respawning**: continuous battles instead of elimination rounds.
- **Head-on damage reduction**: the `bullet_min_damage_frac` mechanic exists in
  the physics engine but is disabled in the training config (see
  [docs/environment.md](docs/environment.md)); evaluate enabling it once the
  meta stabilizes.

## Training & performance

- **Reduced precision**: parts of the pipeline already run under bf16 autocast;
  evaluate an end-to-end low-precision training pass for throughput.
- **Hyperparameter search**: successive halving over short runs, gated at 50M and
  250M steps. `--seed` now covers every RNG the trainer draws from, including the
  one that orders minibatches, so two arms differ only where they were meant to.
- **Forking a run mid-training**: `bnb train --from <run> --at <step>` resumes from
  any checkpoint a run kept, but retention discards all but the newest few, so in
  practice only recent steps are reachable.

## Results and presentation

- **Recapture the replays against `good-leaf-719`**: all fifteen clips, including the
  README hero, show `resilient-resonance-682`, which trained under different physics and
  a different decision rate than the run every published figure now measures.
- **Replay provenance**: the clips carry no sidecar and no poster frame, so nothing but the
  filename says which checkpoint, seed, and commit produced one.
- **A boundary or failure replay** near a measured crossover, so the qualitative page shows
  a loss as well as wins.
- **A count-agnostic system diagram**: simulation to observation to team policy to
  training and evaluation, at a glance.
- **Contribution guide and CI**: the setup and test claims in the README are currently
  verified by hand.

## Interpretability

- **Latent-space visualization**: UMAP projections of ship token embeddings
  (ally vs enemy), ideally as an interactive dashboard with selectable labels
  and subsets.
