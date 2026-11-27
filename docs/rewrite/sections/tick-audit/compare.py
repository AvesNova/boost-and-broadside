"""Largest difference per tensor between two trajectory.py files (floats: max abs; others:
count of unequal entries), and whether they are bitwise equal."""

import sys

import torch

a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
worst = {}
for t, (x, y) in enumerate(zip(a, b)):
    for k in x:
        if k not in y:
            continue
        u, v = x[k], y[k]
        if u.dtype.is_floating_point:
            d = (u - v).abs().max().item() if u.numel() else 0.0
        else:
            d = float((u != v).sum())
        worst[k] = max(worst.get(k, 0.0), d)
print({k: v for k, v in worst.items()})
print("bitwise" if all(v == 0 for v in worst.values()) else "differs")
