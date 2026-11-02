"""The exact resume (runs acceptance 3): K updates, a training checkpoint, a resume in a
fresh process and K more equal 2K updates straight through, parameters and metrics. With
K = 2 and the league copied every 3 updates, the league slot at the checkpoint lags the
model, so a league slot restored wrong shows; so does a return scale not restored."""

import json
import os
import pathlib
import subprocess
import sys

import pytest
import safetensors.torch
import torch

WORKER = pathlib.Path(__file__).with_name("resume_worker.py")
UPDATES = 2


def _run(mode: str, directory: pathlib.Path, device: str, model: str, *damage: str) -> None:
    environment = dict(os.environ, PYTHONPATH=str(WORKER.parent))
    done = subprocess.run([sys.executable, str(WORKER), mode, str(directory), device, model,
                           str(UPDATES), *damage], capture_output=True, text=True,
                          env=environment)
    assert done.returncode == 0, done.stderr[-4000:]


def _result(directory: pathlib.Path, mode: str):
    metrics = json.loads((directory / mode / "metrics.json").read_text())
    weights = safetensors.torch.load_file(directory / mode / "weights.safetensors")
    return metrics, weights


def _resumes_exactly(tmp_path: pathlib.Path, device: str, model: str) -> None:
    for mode in ("straight", "first", "second"):
        _run(mode, tmp_path, device, model)
    straight, straight_weights = _result(tmp_path, "straight")
    first, _ = _result(tmp_path, "first")
    second, second_weights = _result(tmp_path, "second")
    assert first + second == straight  # every logged metric, exactly (floats by repr)
    assert straight_weights.keys() == second_weights.keys()
    for name, tensor in straight_weights.items():
        assert torch.equal(second_weights[name], tensor), name


@pytest.mark.slow
@pytest.mark.parametrize("damage", ["league", "return_scale"])
def test_the_exact_resume_test_sees_a_damaged_league_slot_or_return_scale(tmp_path, damage):
    for mode in ("straight", "first"):
        _run(mode, tmp_path, "cpu", "relational")
    _run("second", tmp_path, "cpu", "relational", damage)
    straight, _ = _result(tmp_path, "straight")
    first, _ = _result(tmp_path, "first")
    second, _ = _result(tmp_path, "second")
    assert first + second != straight


@pytest.mark.slow
@pytest.mark.parametrize("model", ["relational", "yemong"])
def test_a_resumed_run_equals_one_straight_through_on_the_cpu(tmp_path, model):
    _resumes_exactly(tmp_path, "cpu", model)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("model", ["relational", "yemong"])
def test_a_resumed_run_equals_one_straight_through_on_the_gpu(tmp_path, model):
    """Compiled stages and the graph-captured rollout."""
    _resumes_exactly(tmp_path, "cuda", model)
