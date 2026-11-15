# GRU temporal block (arm A16): progress

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
