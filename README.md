# Boost and Broadside (v2, in progress)

A from-scratch PyTorch rewrite of the game and its training stack, designed as a whole for
fast batched GPU training and for maintainability.

- Plan and scaffold: [docs/rewrite/scaffold.md](docs/rewrite/scaffold.md)
- Contracts between sections: [docs/rewrite/contracts.md](docs/rewrite/contracts.md)
- Decision log (written while JAX was the candidate; the framework is now PyTorch, see the
  scaffold): [docs/rewrite/decisions.md](docs/rewrite/decisions.md)
- The previous code is at tag `v1-torch-final`.

## Status

Milestone 0, the walking skeleton, runs end to end. Section 1 (the game: v2 physics,
perception, spawns, maps, fog of war) is merged. Then: the contract observation, team deathmatch rewards, scripted controllers, a small MLP and plain PPO,
with compiled stages and CUDA-graph rollouts. Every section is a stub to be replaced.

## Running

```
uv sync
uv run --no-sync pytest                     # all tests (the learning test needs a GPU)
uv run --no-sync pytest -m "not slow"       # the fast ones
uv run --no-sync bnb --ships-per-team 2 --num-fields 4 --arena-radius 1200 --num-envs 512
uv run --no-sync python -m bench.run        # skeleton benchmark, about five minutes
```

## Layout

| Path | What |
|---|---|
| `src/bnb/contracts` | types shared between sections, no logic |
| `src/bnb/game` | physics, perception, modes (pure functions) |
| `src/bnb/controllers`, `src/bnb/models` | scripted controllers; models and their heads |
| `src/bnb/learn` | learners (PPO) |
| `src/bnb/runtime` | the only stateful code: rollout, stages, CUDA graphs, training loop |
| `bench/` | benchmark harness (lock, random sizes, cells) |
| `tests/reference` | the physics race's float64 goldens, run through an adapter |
