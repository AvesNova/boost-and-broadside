# GRU temporal block (arm A16): progress

## RESUME NOTE (2026-11-15 20:53, laptop shutdown)
- Done: the option (`YemongConfig.temporal_block` "griffin" default / "gru"), tests
  (`tests/sections/test_yemong_gru.py`, 13 passed CPU), ONNX export test with a "gru" case
  (passed with "absolute"), docs (`docs/explanation/yemong.md`), docs tests 609 passed.
  `.venv` was committed by mistake in 1f9af644 and untracked in a later commit.
- Not done: timing. No numbers yet. Power mode was "performance".
  - griffin run failed at 20:35: the shared .venv was being rebuilt (libdevice parse error).
  - gru run failed at ~20:51 in the rollout's CUDA-graph capture: "Cannot copy between CPU
    and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned"
    (`buf.copy_(arg391_1)` in the compiled step). UNDIAGNOSED: may be the GRU path or the
    half-rebuilt venv; rerun griffin first, then gru, to tell. Logs were in the scratchpad
    (lost on reboot).
- Next: rerun `time_blocks.sh` (here; edit S= to a scratch dir) as a systemd unit, one
  arm at a time; if gru repeats the capture error, find which CPU tensor the GRU step
  feeds (the empty (R, blocks, Q, 0, D) buffer is the prime suspect: a zero-size tensor
  may stay on the CPU or lose its device somewhere in the rollout's carry handling).
  Then `sublayer_bench.py` (one sublayer, compiled, 8 and 40 views) for the multiplier;
  merge origin/feat/imugi-boundary (17690eb7 ignores .venv); final report.

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
