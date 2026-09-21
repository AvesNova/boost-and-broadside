"""Focused pure CPU checks for the bounded RL batch screen."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from benchmarks.rl_batch_screen import _measure, load_candidate


def test_load_candidate_requires_module_callable_syntax():
    with pytest.raises(ValueError, match="module:callable"):
        load_candidate("not-a-reference")


def test_load_candidate_resolves_explicit_callable(monkeypatch):
    module = types.ModuleType("screen_candidate_fixture")
    module.tick = lambda env, action: (env, action)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    assert load_candidate("screen_candidate_fixture:tick") is module.tick


def test_measure_reports_latency_samples_and_cpu_memory_semantics():
    calls = 0

    def operation():
        nonlocal calls
        calls += 1

    result = _measure(operation, device=torch.device("cpu"), warmup=2, steps=3)

    assert calls == 5
    assert result["iterations"] == 3
    assert len(result["samples_ms"]) == 3
    assert result["mean_ms"] >= 0
    assert result["peak_allocated_bytes"] is None
    assert result["peak_reserved_bytes"] is None
