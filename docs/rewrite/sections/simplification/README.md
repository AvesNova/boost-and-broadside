# Simplification audit: scripts

The scripts behind the numbers in `docs/rewrite/simplification.md`. They count operations,
not time, so they run on the CPU.

- `regions.py`: the counter. Every operation PyTorch dispatches in eager mode, attributed to
  named regions; view-only operations kept apart; `fused` stands in for a GPU kernel and
  counts as one operation.
- `v1_ops.py`: one v1 rollout step (the `rl` profile) by section.
- `v2_ops.py`: one v2 rollout tick (the `launch.toml` configuration) by stage and section,
  with v2's Triton kernels counted as on the GPU.
- `v2_corr.py`: the correction decode by function.
- `v2_fn.py`: every function in `bnb.game` and Imugi, exclusive counts.
- `v2_learn.py`, `v1_learn.py`: the learner by phase (prepare; one micro-batch's label codes,
  forward and loss, backward; clip and optimizer), every model and learner function a region.
  `regions.fused_autograd` counts a Triton kernel as one op forward and one backward.
- `records/`: the full-size outputs (v2 at 2,048 + 1,024 environments, v1 at 2,592).

## Running

v1 needs Python 3.13, v2 Python 3.14 (its `requires-python`). A CPU build of torch 2.15 plus
numpy, safetensors, tyro, platformdirs, wandb, matplotlib and triton (import only) is enough.
Run from this directory, passing the checkout to measure:

```bash
python v1_ops.py <v1 checkout> [num_envs] [elo_envs_per_matchup]   # num_envs must divide 7,776
python v2_ops.py <v2 checkout> [ppo|bc] [environments] [lane_environments]
python v2_corr.py <v2 checkout> ppo 256 128
python v2_fn.py <v2 checkout>
python v2_learn.py <v2 checkout> [ppo|bc] [segment length]
python v1_learn.py <v1 checkout> [num_steps]
```

A small size (v1 288, v2 256 + 128) gives the same counts as the full one in a few minutes.
