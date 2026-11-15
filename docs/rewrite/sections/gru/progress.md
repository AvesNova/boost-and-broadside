# GRU temporal block (arm A16): progress

## RESUME NOTE (2026-11-15 22:02, paused for the author)
- Done: the option, tests (13 CPU) and ONNX case, docs (see step 1 below); merged
  origin/feat/imugi-boundary (2da26701, no conflicts); `time_blocks.sh` now takes the block
  as its argument and logs to the ignored `runs/gru-timing/`.
- KEY FINDING: the CUDA-graph capture failure is NOT the GRU's. Griffin (the default)
  fails identically on this tree (21:43-22:00, power "performance"): `rollout.py:431 _act`
  during `_capture`, "Cannot copy between CPU and CUDA tensors during CUDA graph capture
  unless the CPU tensor is pinned". So it comes from the merged imugi-boundary tree
  (split critic or later) with this preset (real_scale_all_modes, league lineup,
  host_store=true), not from the GRU block. A small repro (tests' SHORT_SQUAD,
  learner vs chaser, 16 envs, float32, no host_store) captures fine with the GRU.
  Log: `runs/gru-timing/time-griffin.log`.
- Timing: no numbers yet.
- Note: the run sits at the 8G MemoryMax and swaps (~2 GB) while compiling: ~17 min to
  the capture.
- Resume: (1) find the CPU tensor fed in `_act` under capture with the full preset: rerun
  griffin with `TORCH_LOGS=output_code` and look for `empty_strided_cpu` / a CPU
  argument; suspects: the league `OpponentSlots` path, the split critic, host_store.
  Tell the coordinator (it affects every arm, not just A16). (2) after the fix:
  `systemd-run --user --unit=gru-timing -p MemoryMax=8G --wait --collect -E PATH=$PATH
  /bin/bash /home/vizia/avesnova/bnb-gru/docs/rewrite/sections/gru/time_blocks.sh griffin`,
  then `... gru`; (3) `sublayer_bench.py` (8 and 40 views); report GRU/Griffin multipliers.

Branch `feat/gru-temporal` (from `feat/imugi-boundary`).

## 2026-11-15 step 1: the option
- Tried/worked: `YemongConfig.temporal_block` ("griffin" default, "gru"); `temporal.GRUBlock`:
  pre-norm, a GRU (torch's `nn.GRUCell` maths, its parameters as two `nn.Linear`s, so keyed
  initialisation is `nn.GRUCell`'s), an output projection into the residual, then the same
  gated MLP. The carry's convolution buffer has `history` inputs per block: 3 for Griffin,
  0 for the GRU (an empty tensor, passed through), so the carry keeps its fields and
  every consumer of it is unchanged.
- The sequence path computes every step's input gates in one product, then loops the
  hidden-gate product over time, the state cleared at every later start (plain PyTorch,
  no kernel; the loop unrolls under `torch.compile`).
- Parameters at D = 128: Griffin's sublayer 279 808 (17 D^2 + 10 D), the GRU's 312 320
  (19 D^2 + 8 D), +11.6%; equal width, as A16 compares.
- Tests: `tests/sections/test_yemong_gru.py`, 13 passed on the CPU (default unchanged and
  C-fixed's goldens; torch's GRUCell; sequence = loop of steps with starts inside;
  a start forgets; model unroll = step loop in both orders; trains; bf16 autocast;
  policy file round trip; the --set line).
- Next: the ONNX export test for the GRU, then GPU timing.

## 2026-11-16 04:16 step 2: merge
- Merged origin/feat/imugi-boundary (graph-capture fix 71629552, HL-Gauss codes, head
  switches, Yemong v9) as a7a295e6: two conflicts, both keep-both (the config fields and
  checks; the ONNX cases gain "gru" next to the A13 ones). `.venv` untracked. 130 passed
  (quick tier: gru, export, model, head switches, codes, training checkpoints); ruff clean.
- Next: the timing runs (griffin then gru) are queued on the shared GPU lock.

## 2026-11-16 05:10 step 2: timing (in progress)
- Didn't work: the full preset at 256 environments OOM-kills at the 8G MemoryMax twice
  (04:23, at the default compile workers; 04:51, with 2 workers): the main process alone
  holds ~6.6 GB, mostly the pinned host store, which cannot swap (system swap was full
  too). `time_blocks.sh` now takes the environments as a second argument; both blocks run
  at 128 (same settings for both), so the per-update times are half the reference's
  131k-step updates. Also dropped the removed `--update-policies` flag.
- Worked: `sublayer_bench.py` (power "performance", 05:05), segment forward + backward
  at L = 128, D = 128, bf16, median ms:

  | views | Griffin kernels | Griffin plain | GRU | GRU / Griffin kernels |
  |---|---|---|---|---|
  | 8 (the learner's micro-batch) | 1.88 | 2.97 | 16.17 | 8.6x |
  | 40 | 11.55 | 16.13 | 58.72 | 5.1x |

  The GRU's hidden-gate product is a serial loop over 128 steps (no scan), so it is
  latency-bound at small row counts; its compile is ~106 s against ~18 s.
- Didn't work (05:09-05:18): 128 environments OOM-kill at 8G too (peak 8G + 1.4G swap,
  in the compile, before the first update), so the store is not the cause: the merged
  tree's compile of the full preset needs more than 8G here (before the merge it reached
  the capture at 8G). A 14G MemoryMax was refused by the session's permission check, so
  the whole-run Griffin/GRU timing is NOT measured. GRU run not attempted (stopped).
- Estimate from the sublayer: the learner runs 1536 views / 8 = 192 micro-batches x 4
  epochs x 2 temporal blocks per 131k-step update; at +14.3 ms each, about +22 s on the
  3080's ~67 s learn, about 1.3x per step (rollout: the GRU step is two small products,
  assumed ~unchanged). Cost line in arm-list / arm-review: 3.7 h -> about 4.8 h ($2.40),
  provisional.
- Next: the whole-run timing needs a larger MemoryMax (the author's call) or the rented
  GPU box; `time_blocks.sh griffin` then `gru`, steady = updates 2-4.
