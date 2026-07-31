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

## Interpretability

- **Latent-space visualization**: UMAP projections of ship token embeddings
  (ally vs enemy), ideally as an interactive dashboard with selectable labels
  and subsets.
