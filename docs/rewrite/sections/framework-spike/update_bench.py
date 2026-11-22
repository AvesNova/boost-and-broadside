"""The PPO update (unit C) in PyTorch, made comparable with JAX's.

torch_bench.py's update has two differences from jax_bench.py's that are not the framework:
  1. the RG-LRU recurrence over time is a Python loop of T steps (JAX: one lax.scan), which
     torch.compile unrolls into T small steps per block, forward and backward;
  2. only the loss is compiled (forward and, through AOTAutograd, backward); gradient clipping
     and Adam stay eager (JAX: one jit over the loss, the gradient, clipping and Adam).

This script measures the update with each difference removed, in fresh processes with an empty
inductor cache (so the first call is the compile):
  --scan loop      torch_bench's Python loop (unchanged)
  --scan doubling  a log-step scan: the recurrence h_t = a_t h_{t-1} + b_t composed in
                   ceil(log2 T) levels, (A, B) <- (A_t A_{t-d}, B_t + A_t B_{t-d}); exact up to
                   rounding, O(T log T) work, a fixed number of operations for any T
  variants: eager | compiled (loss only, as torch_bench) | compiled_step (loss, clip and Adam)

    python update_bench.py --device cuda --out results/laptop/torch_gpu_update_fair.json
    python update_bench.py --device cpu --check        # doubling scan against the loop
"""

import os
import sys
import time

PROCESS_START = time.perf_counter()

import argparse  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

import spec  # noqa: E402
import torch_bench as tb  # noqa: E402

IMPORT_SECONDS = time.perf_counter() - PROCESS_START
HERE = os.path.abspath(__file__)
LOOP_SCAN = tb.RGLRU.scan


def doubling_scan(self, x, h):
    """tb.RGLRU.scan computed by recursive doubling instead of a loop over time."""
    a, b = self.gates(x)                                   # [env, time, token, width], float32
    b = torch.cat([a[:, :1] * h[:, None] + b[:, :1], b[:, 1:]], 1)
    steps, d = x.shape[1], 1
    while d < steps:
        b = torch.cat([b[:, :d], b[:, d:] + a[:, d:] * b[:, :-d]], 1)
        if 2 * d < steps:
            a = torch.cat([a[:, :d], a[:, d:] * a[:, :-d]], 1)
        d *= 2
    return x + b, b[:, -1]


def use_scan(name):
    tb.RGLRU.scan = {"loop": LOOP_SCAN, "doubling": doubling_scan}[name]


def build(device, variant, bf16, micro, ships):
    policy = tb.load_params(tb.Policy(), spec.init_params()).to(device)
    batch = tb.to_device(spec.make_rollout(micro[0], micro[1], ships), device)
    batch["action"] = batch["action"].long()
    optimizer = tb.make_optimizer(policy)
    loss_fn = tb.ppo_loss if variant == "eager" else torch.compile(tb.ppo_loss)

    def apply():
        nn.utils.clip_grad_norm_(policy.parameters(), spec.MAX_GRAD_NORM)
        optimizer.step()

    if variant == "compiled_step":
        apply = torch.compile(apply)
    state = {}

    def run():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            loss = loss_fn(policy, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        apply()
        state["loss"] = loss

    return run


def child(a, device):
    use_scan(a.scan)
    run = build(device, a.variant, a.precision == "bf16", a.micro, a.ships)
    sync = tb.make_sync(device)
    record = {"suite": "update", "unit": "C", "scan": a.scan, "variant": a.variant, "precision": a.precision,
              "envs": a.micro[0], "time": a.micro[1], "ships": a.ships, "import_s": IMPORT_SECONDS,
              "cache": "cold"}
    record.update(tb.first_calls(run, sync))
    record.update(tb.steady(run, sync, a.repeats, a.budget))
    record["graphs_compiled"] = tb.compiled_graphs()
    spec.emit(record)


def check(device, micro, ships):
    """Loss and every gradient, loop against doubling, same weights and batch, float32 eager."""
    torch.set_float32_matmul_precision("highest")
    results = {}
    for name in ("loop", "doubling"):
        use_scan(name)
        policy = tb.load_params(tb.Policy(), spec.init_params()).to(device)
        batch = tb.to_device(spec.make_rollout(micro[0], micro[1], ships), device)
        batch["action"] = batch["action"].long()
        loss = tb.ppo_loss(policy, batch)
        loss.backward()
        results[name] = (loss.item(), {k: p.grad.clone() for k, p in policy.named_parameters()})
    (loss_loop, grads_loop), (loss_doubling, grads_doubling) = results["loop"], results["doubling"]
    worst = max(((grads_doubling[k] - g).abs().max().item() / g.abs().max().clamp_min(1e-30).item(), k)
                for k, g in grads_loop.items())
    print(f"loss loop {loss_loop:.8g} doubling {loss_doubling:.8g} difference {abs(loss_loop - loss_doubling):.3g}")
    print(f"largest gradient difference relative to that gradient's largest entry: {worst[0]:.3g} ({worst[1]})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--micro", default="16x128")
    parser.add_argument("--ships", type=int, default=10)
    parser.add_argument("--precisions", default="f32,bf16")
    parser.add_argument("--variants", default="eager,compiled,compiled_step")
    parser.add_argument("--scans", default="loop,doubling")
    parser.add_argument("--process-repeats", type=int, default=3, help="fresh processes per combination")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--scan", default="loop")
    parser.add_argument("--variant", default="eager")
    parser.add_argument("--precision", default="f32")
    a = parser.parse_args()
    device = torch.device(a.device)
    a.micro = tuple(int(x) for x in a.micro.split("x"))
    torch.set_float32_matmul_precision("highest")
    if a.check:
        check(device, a.micro, a.ships)
        return
    if a.child:
        child(a, device)
        return
    common = ["--device", a.device, "--micro", f"{a.micro[0]}x{a.micro[1]}", "--ships", a.ships,
              "--repeats", a.repeats, "--budget", a.budget]
    records = []
    for repeat in range(1, a.process_repeats + 1):
        for scan in a.scans.split(","):
            for precision in a.precisions.split(","):
                for variant in a.variants.split(","):
                    cache = tempfile.mkdtemp(prefix="spike-inductor-")
                    extra = ["--child", "--scan", scan, "--variant", variant, "--precision", precision]
                    print(f"[update r{repeat}] {' '.join(extra[1:])}", flush=True)
                    got = spec.run_child(HERE, common + extra, {"TORCHINDUCTOR_CACHE_DIR": cache})
                    shutil.rmtree(cache, ignore_errors=True)
                    for record in got:
                        record["process_repeat"] = repeat
                        print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()
                                      if k not in ("load_before", "load_after", "stderr_tail")}, flush=True)
                    records.extend(got)
    meta = {"framework": "pytorch", "torch": torch.__version__, "device": a.device,
            "float32_matmul_precision": "highest", "micro_batch": f"{a.micro[0]}x{a.micro[1]}",
            "machine": spec.machine()}
    out = a.out or os.path.join(os.path.dirname(HERE), "results", f"torch_{a.device}_update_fair.json")
    spec.write_results(out, meta, records)


if __name__ == "__main__":
    main()
