"""The bf16 rounding floor: one minibatch's gradient, bf16 matmuls against float32, same
weights and stored rollout, at each micro-batch size; against the split's own difference."""

import copy
import dataclasses
import sys

import torch

sys.path[:0] = ["tests/sections", "tests", "."]
import learner_cases as lc  # noqa: E402
import yemong_cases as cases  # noqa: E402

torch.manual_seed(0)
net = cases.model(bfloat16_matmuls=True)
runner = lc.rollout(net, environments=16)
views = runner.store.views
bootstrap = runner.bootstrap()
SHIPS = runner.rules.num_ships


def gradient(bf16: bool, micro_views: int):
    model = copy.deepcopy(net)
    model.config = dataclasses.replace(model.config, bfloat16_matmuls=bf16)
    learner = lc.learner(model, runner, minibatches=2, micro_batch_rows=micro_views * SHIPS)
    prepared = learner.prepare(runner.store, bootstrap)
    learner.gradient(runner.store, prepared, torch.arange(views // 2))
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def compare(name, g, g0):
    cos = torch.nn.functional.cosine_similarity(g.double(), g0.double(), 0).item()
    print(f"{name}: 1-cos {1 - cos:.2e}, |dg|/|g| {((g - g0).norm() / g0.norm()).item():.2e}")


full = views // 2
reference32 = gradient(False, full)
reference16 = gradient(True, full)
compare(f"bf16 against float32, micro {full}", reference16, reference32)
for micro in (full // 2, 2):
    compare(f"float32 micro {micro} against {full}", gradient(False, micro), reference32)
    compare(f"bf16 micro {micro} against {full}", gradient(True, micro), reference16)
