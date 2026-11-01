"""Loading goldens and comparing against them."""

import os

import numpy as np
import torch

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

# float64 against float64: only summation order may differ.
TIGHT = dict(relative=1e-9, absolute=1e-9)


class Case(dict):
    """golden/<name>.npz as {"config": {...}, "inputs": {...}, "weights": {...},
    "outputs": {...}}; tensors, with config scalars as Python numbers."""

    def __getattr__(self, name):
        return self.get(name, {})


def case(name: str) -> Case:
    out = Case()
    with np.load(os.path.join(GOLDEN, f"{name}.npz")) as data:
        for key in data.files:
            group, item = key.split("/", 1)
            value = data[key]
            if group == "config" and value.ndim == 0:
                value = value.item()
            else:
                value = torch.from_numpy(np.array(value))
            out.setdefault(group, {})[item] = value
    return out


def check(actual, expected, name="", relative=1e-9, absolute=1e-9):
    """Elementwise |actual - expected| <= absolute * max(1, max|expected|) + relative * |expected|.

    The absolute part scales with the array's own magnitude, so one tolerance serves
    probabilities and pre-norm activations alike."""
    actual = torch.as_tensor(actual)
    assert tuple(actual.shape) == tuple(expected.shape), (
        f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    if not expected.is_floating_point():
        assert torch.equal(actual.to(expected.dtype), expected), f"{name}: values differ"
        return
    actual = actual.double()
    expected = expected.double()
    scale = max(1.0, expected.abs().max().item()) if expected.numel() else 1.0
    error = (actual - expected).abs()
    bound = absolute * scale + relative * expected.abs()
    assert torch.isfinite(actual).all(), f"{name}: non-finite values"
    worst = (error - bound).argmax()
    assert (error <= bound).all(), (
        f"{name}: max error {error.max().item():.3e} (at flat index {worst.item()}: "
        f"{actual.flatten()[worst].item()} vs {expected.flatten()[worst].item()}), scale {scale:.3e}"
    )
