"""The linear recurrence h_t = a_t h_{t-1} + b_t over a leading time axis."""

import torch


def sequential_scan(decay, contribution, initial):
    """decay, contribution (T, ..., D); initial (..., D) -> h (T, ..., D)."""
    hidden = initial
    outputs = []
    for t in range(decay.shape[0]):
        hidden = decay[t] * hidden + contribution[t]
        outputs.append(hidden)
    return torch.stack(outputs)


def linear_scan(decay, contribution, initial):
    """h_t = decay_t h_{t-1} + contribution_t, h_{-1} = initial. Time-major (T, ..., D)."""
    return sequential_scan(decay, contribution, initial)
