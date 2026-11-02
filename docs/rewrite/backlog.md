# Backlog: deferred until v2 trains (2026-11-03)

The author's priority is a minimum viable v2: a BC agent trained, then an RL agent trained
to about 1500 Elo (scripted = 1000), a play interface that makes sense and looks good, and
the analysis tooling up to spec. Once we know things are good, we decide what else is
worth pursuing. These items wait until then. Each says where it came from.

## Speed

- **A per-component profile of the training update:** forward and backward time for the
  encoder, each spatial and temporal layer, each head, the losses, the label codes and the
  optimizer, in bf16 with the kernels. This comes first, before any item below.
  (Update profiling phase 2; branch `perf/update` holds earlier, pre-kernel work.)
- **The shot-trajectory head in bf16:** it runs in fp32 even when training is in bf16,
  because it is called outside the model's autocast.
- **Fewer shot ages:** stratified sampling (an age at or near the end, one at or near the
  hit, two at random in the middle), weighted so the loss stays unbiased. Test it against
  all ages over three seeds. An earlier uniform 3-age test scored 0.62 against 0.97 on one
  seed.
- **A Triton kernel for the shot-trajectory head.**

## Portability

- **Portable batches** (decisions in `sections/portable-batch.md`): the micro-batch size
  and host storage chosen automatically from the memory probe. Needed before the rented-GPU
  publication run, not for the MVP on the laptop.

## Later, by agreement

- The 128-versus-256 width comparison (the tooling comes with the cleanup pass).
- An autoregressive-predictor diagnostic.
- Capsule fields and a new mode.
- Native goldens; power-profile profiling; results and charts; client packaging.
