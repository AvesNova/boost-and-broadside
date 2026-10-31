# Library mockups (JAX library choice, tabled item 6)

The Yemong Griffin temporal block (`models/yemong/griffin.py`: RMSNorm, linear, causal
depthwise conv, RG-LRU, GELU gate, gated MLP, two residuals) written four ways so the
NN-library choice can be compared on real model code. Not production code.

- `common.py`: pure math shared by all four (conv with episode-start masking, RG-LRU
  associative scan, norms). This part is library-independent in every option.
- `plain_jax.py`, `linen.py`, `nnx.py`, `equinox_.py`: the block in each paradigm, plus
  `to_plain` / `from_plain` converters to the contract's plain-dict param layout.
- `check.py`: with the same plain-dict weights, all four give identical outputs; the
  one-step rollout equals the sequence pass (episode starts included); 4 stacked league
  slots run in one vmapped call; grad plus an optax step; `jax.export` round trip.

Checked 2026-10-21 on CPU with jax 0.12.0, flax 0.12.10, equinox 0.13.8, optax 0.2.8,
with DeprecationWarnings raised as errors. Not compared numerically against the PyTorch
original.
