# The JAX version: progress

The record of where the JAX work stands. The plan is in [plan.md](plan.md). If you are a
new agent, read both, then the latest log entry, then carry on from "Next".

**Update this file at every milestone, then commit and push straight away.** Newest log
entries go at the bottom.

## Now

- **Phase:** not started; waiting on the round-2 winner.
- **Next:** the steps in plan.md, "Before the first task".
- **Blocked on:** the author's choice of the round-2 winner.

## Freeze and baseline

| Item | Value |
|---|---|
| v2 freeze tag | *(not yet)* |
| Round-2 winner | *(not yet)* |
| Baseline machine | RTX 4070 Laptop |
| v2 steps per second | *(not yet measured)* |
| v2 compile time, cold | *(not yet measured; the author reports 20-30 min)* |
| v2 compile time, warm cache | *(not yet measured)* |
| JAX and library versions pinned | *(not yet)* |

## Tasks

Status values: not started, brief written, brief signed off, oracle committed,
implementing, in review, fixes, merged.

| # | Task | Status | Branch | Notes | Speed / compile vs v2 |
|---|---|---|---|---|---|
| 0 | Skeleton | not started | | | |
| 1a | Physics (race) | not started | | | |
| 1b | Perception and observation | not started | | | |
| 1c | Spawns, maps, modes, rewards | not started | | | |
| 1d | Scripted agent | not started | | | |
| 2a | Yemong network (race) | not started | | | |
| 2b | Yemong heads, codes, belief, controller | not started | | | |
| 3a | PPO, losses, labels | not started | | | |
| 3b | Rollout buffer, behaviour cloning | not started | | | |
| 4 | League, runs, figures, play shell | not started | | | |
| — | Final training gate | not started | | | |

## Tolerances loosened (author sign-off required)

| Date | Test set | Old | New | Why | Signed off |
|---|---|---|---|---|---|

## Log

### 2026-11-08: plan agreed

- The author decided to build a JAX version of v2: v2's compile times are 20-30 min, it
  is still launch-bound, and JAX was about 100x faster to compile in the physics race.
- Agreed: freeze v2 first; port only the round-2 winner; no bitwise or float64
  exactness (float32, statistical equivalence where chaotic); the goldens pass at float32;
  faster than v2 in both steps per second and compile time; Equinox with our own layers;
  no kernels at first; Linux only; no ONNX or PyTorch-checkpoint loading for now.
- Agreed process: coordinator, oracle agent, implementer (sees v2), reviewer (blind to
  v2), one-place benchmarking; races with a judge on the physics tick and the Yemong
  trunk; tasks of about 1-3k lines; skeleton, then game, model, learner, the rest.
- The arms digests (`tests/sections/arm_goldens.py`) are not ported.
- Written by the planning session (cloud). Implementation agents will run on the author's
  laptop.
