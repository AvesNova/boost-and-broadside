import numpy as np
import torch
import pytest
from boost_and_broadside.utils.dataset_stats import compute_class_weights, apply_turn_exceptions, normalize_weights
from boost_and_broadside.core.constants import TurnActions, NUM_TURN_ACTIONS

def test_compute_class_weights_basic():
    # counts = [1, 9] (total 10) -> freqs = [0.1, 0.9]
    # w = min(cap, 1 / sqrt(x))
    # w0 = 1 / sqrt(0.1) = 1 / 0.3162 = 3.162
    # w1 = 1 / sqrt(0.9) = 1 / 0.9486 = 1.054
    counts = np.array([1, 9])
    weights = compute_class_weights(counts, cap=10.0)
    assert torch.isclose(weights[0], torch.tensor(3.162), atol=1e-3)
    assert torch.isclose(weights[1], torch.tensor(1.054), atol=1e-3)

def test_compute_class_weights_zero():
    # If a class is 0 freq, weight should be cap.
    counts = np.array([0, 10]) # freqs 0.0, 1.0
    weights = compute_class_weights(counts, cap=10.0)
    # 1/sqrt(0) -> inf -> cap
    assert torch.isclose(weights[0], torch.tensor(10.0), atol=1e-3)
    # 1/sqrt(1) -> 1
    assert torch.isclose(weights[1], torch.tensor(1.0), atol=1e-3)

def test_turn_exceptions():
    # 7 actions
    # Now a pass-through since capping is handled by the global config in the trainer
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0])
    out = apply_turn_exceptions(weights)
    assert torch.allclose(out, weights)

def test_normalize_weights():
    # Counts [100, 100], Weights [1.0, 1.0] -> Expected Sum = 0.5*1 + 0.5*1 = 1.0. Already normalized.
    counts = np.array([100, 100])
    weights = torch.tensor([1.0, 1.0])
    norm = normalize_weights(weights, counts)
    assert torch.allclose(norm, weights)
    
    # Counts [90, 10], Weights [1.0, 10.0] -> Sum = 0.9*1 + 0.1*10 = 0.9 + 1 = 1.9.
    # Expected Norm Factors = 1/1.9 = 0.5263
    counts = np.array([90, 10]) # freq 0.9, 0.1
    weights = torch.tensor([1.0, 10.0])
    norm = normalize_weights(weights, counts)
    expected = weights / 1.9
    assert torch.allclose(norm, expected, atol=1e-4)
