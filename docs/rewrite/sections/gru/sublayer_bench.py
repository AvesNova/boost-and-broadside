"""One temporal sublayer's segment forward + backward, compiled as the learner compiles, at
L = 128 and D = 128, for several row counts (views x 11 query tokens); bf16 autocast.
Prints ms per call (median of 30 after 5 warm-up)."""

import json
import sys

import torch

from bnb.models.yemong import kernels
from bnb.models.yemong.temporal import GRUBlock, TemporalBlock

torch.manual_seed(0)
device = torch.device("cuda")
results = {}
for views in (8, 40):
    rows = views * 11
    for name, cls, kernel in (
        ("griffin_kernels", TemporalBlock, True),
        ("griffin_plain", TemporalBlock, False),
        ("gru", GRUBlock, False),
    ):
        kernels.switch_all(kernel)
        torch._dynamo.reset()
        block = cls(128).to(device)
        tokens = torch.randn(128, rows, 128, device=device, requires_grad=True)
        state = torch.zeros(rows, 128, device=device)
        buffer = torch.zeros(rows, cls.history, 128, device=device)
        starts = torch.rand(128, rows, device=device) < 0.01

        def run(tokens, state, buffer, starts, block=block):
            with torch.autocast("cuda", torch.bfloat16):
                out, _, _ = block.sequence(tokens, state, buffer, starts)
            return out.float().square().mean()

        compiled = torch.compile(
            run, fullgraph=True, dynamic=False, options={"comprehensive_padding": False}
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        import time

        began = time.perf_counter()
        compiled(tokens, state, buffer, starts).backward()
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - began
        for _ in range(5):
            compiled(tokens, state, buffer, starts).backward()
        times = []
        for _ in range(30):
            start.record()
            compiled(tokens, state, buffer, starts).backward()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        results[f"{name}@{views}views"] = dict(ms=times[15], compile_s=round(compile_seconds, 1))
        print(f"{name} views={views}: {times[15]:.2f} ms, compile {compile_seconds:.0f} s",
              file=sys.stderr, flush=True)
print(json.dumps(results))
