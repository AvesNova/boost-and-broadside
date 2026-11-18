# Roadmap

Ideas under consideration, roughly ordered by interest. None are commitments.

## Gameplay

- **Kill rewards in-game**: grant energy on kills (paired with a lower base
  regeneration rate and higher energy cap) to reward aggression mechanically
  rather than only through reward shaping.
- **Respawning**: continuous battles instead of elimination rounds.

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

- **Recapture the remaining replays against `wild-universe-731`**: the README hero
  (4 vs 7, seed 2001, game 406) and the 32 vs 50 large battle on the replays page are
  `wild-universe-731` clips, but the fifteen in `docs/results/replays/` (self-play and the
  other asymmetric examples) still show `resilient-resonance-682`, which trained under
  different physics and a different decision rate.
- **Replay provenance**: the clips carry no sidecar and no poster frame. The two
  `wild-universe-731` clips are recorded in `scripts/studies/README.md`; for the others
  nothing but the filename says which checkpoint, seed, and commit produced one.
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
