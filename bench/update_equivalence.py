"""Compare one learner update before and after a change, from the same weights and store.

    uv run --no-sync python -m bench.update_equivalence capture INPUTS '{"preset": ...}'
    uv run --no-sync python -m bench.update_equivalence run INPUTS OUTPUTS ['{"compile": false}']
    uv run --no-sync python -m bench.update_equivalence compare OUTPUTS_A OUTPUTS_B

capture: build the trainer of a cell (as `bench.update_profile`), run one update so the
weights are trained, collect a fresh store and save the cell, the weights, the optimizer,
the store's leaves and the bootstraps.
run: rebuild the trainer from the cell (with the optional overrides: "compile", "learner"
fields), load the saved state into it, and save the outputs: the gradient of the first
minibatch (its views from a fixed generator) at the saved weights, then one update's
reported metrics and the parameter deltas it made.
compare: the largest difference of each output, relative to the output's scale (the
largest magnitude of that tensor, or of that metric).
GPU runs take the benchmark lock.
"""

import json
import os
import sys

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import torch  # noqa: E402

from bench.lock import bench_lock  # noqa: E402
from bench.update_profile import build  # noqa: E402

FIRST_MINIBATCH_SEED = 12345


def capture(path: str, cell: dict) -> None:
    trainer, _ = build(cell)
    store, bootstraps = trainer.collect()
    store.synchronize()
    torch.save(dict(cell=cell, model=trainer.model.state_dict(),
                    optimizer=trainer.ppo.optimizer.state_dict(),
                    return_scale=dict(vars(trainer.ppo.return_scale)),
                    env_steps=trainer.ppo.env_steps,
                    store={name: [leaf.cpu() for leaf in leaves]
                           for name, leaves in store.data.items()},
                    bootstraps=[torch.utils._pytree.tree_map_only(torch.Tensor, torch.Tensor.cpu,
                                                                  b) for b in bootstraps]),
               path)


def _restore(saved: dict, overrides: dict):
    cell = dict(saved["cell"], **{k: v for k, v in overrides.items() if k != "compile"})
    if "learner" in overrides:
        cell["learner"] = dict(saved["cell"].get("learner", {}), **overrides["learner"])
    cell["compile"] = overrides.get("compile", True)
    trainer, _ = build(cell)
    device = torch.device("cuda")
    trainer.model.load_state_dict(saved["model"])
    trainer.ppo.optimizer.load_state_dict(saved["optimizer"])
    for name, value in saved["return_scale"].items():
        setattr(trainer.ppo.return_scale, name,
                value.to(device) if isinstance(value, torch.Tensor) else value)
    trainer.ppo.env_steps = saved["env_steps"]
    store = trainer.rollout.store
    for name, leaves in saved["store"].items():
        if name not in store.data:
            store.data[name] = [torch.empty_like(leaf, device=store.data[store.names[0]][0].device,
                                                 pin_memory=store.pinned) for leaf in leaves]
        for target, source in zip(store.data[name], leaves, strict=True):
            target.copy_(source)
    torch.cuda.synchronize()
    bootstraps = [torch.utils._pytree.tree_map_only(torch.Tensor, lambda x: x.to(device), b)
                  for b in saved["bootstraps"]]
    return trainer, store, bootstraps


def run(inputs: str, outputs: str, overrides: dict) -> None:
    saved = torch.load(inputs, weights_only=False)
    trainer, store, bootstraps = _restore(saved, overrides)
    ppo = trainer.ppo
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
    # The first minibatch's gradient at the saved weights (prepare draws the learner's shot
    # spread, so the update below uses a fresh learner state: rebuild it).
    prepared = ppo.prepare(store, bootstraps)
    generator = torch.Generator().manual_seed(FIRST_MINIBATCH_SEED)
    chosen = torch.randperm(store.views, generator=generator)[:store.views
                                                               // ppo.config.minibatches]
    terms, levels, statistics = ppo.gradient(store, prepared, chosen)
    gradient = [parameter.grad.detach().clone() for parameter in trainer.model.parameters()]
    first = dict(terms=terms.cpu(), levels=levels.cpu(), statistics=statistics.cpu())
    del trainer, ppo, prepared
    torch.cuda.empty_cache()
    trainer, store, bootstraps = _restore(saved, overrides)
    stats = trainer.ppo.update(store, bootstraps)
    deltas = [(after.detach() - start).cpu()
              for after, start in zip(trainer.model.parameters(), before, strict=True)]
    torch.save(dict(first=first, gradient=[g.cpu() for g in gradient], stats=stats,
                    deltas=deltas, names=[name for name, _ in trainer.model.named_parameters()]),
               outputs)


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    scale = max(a.abs().max().item(), b.abs().max().item(), 1e-30)
    return (a - b).abs().max().item() / scale


def compare(path_a: str, path_b: str) -> dict:
    a, b = torch.load(path_a, weights_only=False), torch.load(path_b, weights_only=False)
    out = {f"first/{name}": _relative(a["first"][name], b["first"][name])
           for name in a["first"]}
    gradient = [_relative(x, y) for x, y in zip(a["gradient"], b["gradient"], strict=True)]
    deltas = [_relative(x, y) for x, y in zip(a["deltas"], b["deltas"], strict=True)]
    worst = max(range(len(deltas)), key=deltas.__getitem__)
    out.update(gradient_max=max(gradient),
               gradient_median=sorted(gradient)[len(gradient) // 2],
               delta_max=deltas[worst], delta_max_parameter=a["names"][worst],
               delta_median=sorted(deltas)[len(deltas) // 2])
    metrics = {key: abs(a["stats"][key] - b["stats"][key]) / max(abs(a["stats"][key]), 1e-30)
               for key, value in a["stats"].items() if isinstance(value, float)}
    out["metrics_max"] = max(metrics.values())
    out["metrics_max_name"] = max(metrics, key=metrics.get)
    out["minibatch_steps"] = (a["stats"]["minibatch_steps"], b["stats"]["minibatch_steps"])
    return out


def main() -> None:
    mode = sys.argv[1]
    if mode == "compare":
        print("RESULT " + json.dumps(compare(sys.argv[2], sys.argv[3])), flush=True)
        return
    with bench_lock():
        if mode == "capture":
            capture(sys.argv[2], json.loads(sys.argv[3]))
        else:
            overrides = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
            run(sys.argv[2], sys.argv[3], overrides)
    print("RESULT done", flush=True)


if __name__ == "__main__":
    main()
