"""Same model, same stored rollout, learner updates at several micro_batch_rows (CPU)."""

import copy
import sys

import torch

sys.path[:0] = ["tests/sections", "tests", "."]
import learner_cases as lc  # noqa: E402
import yemong_cases as cases  # noqa: E402

from bnb.contracts.series import Mean  # noqa: E402

torch.manual_seed(0)
net = cases.model(bfloat16_matmuls=len(sys.argv) > 2 and sys.argv[2] == "bf16")
print("bf16 matmuls", net.config.bfloat16_matmuls)
runner = lc.rollout(net, environments=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
views = runner.store.views
bootstrap = runner.bootstrap()
MINIBATCHES = 2
SHIPS = runner.rules.num_ships
print("views", views, "ships per view", SHIPS)


def scalar(v):
    if isinstance(v, Mean):
        count = float(v.count)
        return float(v.total) / count if count else float("nan")
    if torch.is_tensor(v):
        return v.float().flatten().tolist() if v.numel() > 1 else float(v)
    return v


def run(micro_views, updates=2):
    torch.manual_seed(1)
    model = copy.deepcopy(net)
    learner = lc.learner(
        model,
        runner,
        minibatches=MINIBATCHES,
        epochs=2,
        warmup_steps=0,
        target_kl=None,
        micro_batch_rows=micro_views * SHIPS,
    )
    chosen = torch.arange(views // MINIBATCHES)
    prepared = learner.prepare(runner.store, bootstrap)
    terms, levels, stats, _, _ = learner.gradient(runner.store, prepared, chosen)
    grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
    learner.optimizer.zero_grad(set_to_none=True)
    series = [
        learner.update(runner.store, bootstrap, environment_steps=1000) for _ in range(updates)
    ]
    params = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
    return dict(
        terms=terms, grad=grad, params=params, series=series, first=learner.drift_views(views)
    )


per_minibatch = views // MINIBATCHES
sizes = sorted({per_minibatch, per_minibatch // 2, max(1, per_minibatch // 5)}, reverse=True)
results = {m: run(m) for m in sizes}
reference = results[per_minibatch]
p0 = torch.cat([p.detach().flatten() for p in net.parameters()])
for m, r in results.items():
    g, g0 = r["grad"], reference["grad"]
    cos = torch.nn.functional.cosine_similarity(g.double(), g0.double(), 0).item()
    rel = ((g.norm() - g0.norm()) / g0.norm()).item()
    diff = ((g - g0).norm() / g0.norm()).item()
    moved = ((r["params"] - reference["params"]).norm() / (reference["params"] - p0).norm()).item()
    tdiff = (r["terms"] - reference["terms"]).abs().max().item()
    print(
        f"micro {m:3d} views (drift sample {r['first']} views): grad 1-cos {1 - cos:.2e}, "
        f"norm rel {rel:+.2e}, |dg|/|g| {diff:.2e}; terms max|d| {tdiff:.2e}; "
        f"params after 2 updates |dp|/|step| {moved:.2e}"
    )
    worst = []
    for u, (s, s0) in enumerate(zip(r["series"], reference["series"], strict=True)):
        for key in s0:
            if key not in s:
                worst.append((1.0, u, key, "missing", None))
                continue
            a, b = scalar(s[key]), scalar(s0[key])
            if isinstance(b, list):
                ta, tb = torch.tensor(a), torch.tensor(b)
                d = (
                    ((ta - tb).abs().max() / tb.abs().max().clamp_min(1e-12)).item()
                    if tb.numel()
                    else 0.0
                )
                a = b = None
            elif isinstance(b, float):
                d = 0.0 if b != b else abs(a - b) / max(abs(b), 1e-12)
            else:
                d = 0.0 if a == b else 1.0
            worst.append((d, u, key, a, b))
    worst.sort(key=lambda t: -t[0])
    print("    worst series (rel diff, update, key, value, reference):")
    for w in worst[:8]:
        print("     ", f"{w[0]:.2e}", w[1], w[2], w[3], w[4])
    for key in (
        "policy/ratio_drift/max",
        "policy/ratio_drift/mean",
        "policy/kl_first",
        "policy/clip_fraction",
    ):
        print("    ", key, [round(scalar(s[key]), 7) for s in r["series"]])
