"""The resume (runs acceptance 3, league acceptance 6): K updates, a training checkpoint, a
resume in a fresh process and K more equal 2K updates straight through, parameters and
metrics, the league's included (its draws, gauges and pool). Opponent slots restored wrong
show, and so do the league's draws restarted, a gauge's record lost or a return scale not
restored.

On the CPU the two are bitwise equal. On the GPU they are equal within rounding
(contracts, "Resume: integrity, not bits"): the processes autotune their kernels from
cold, and a configuration that wins by timing can sum in another order. The integer and
categorical fields (counters, games, results) stay exact there too, and the floats agree
to `GPU_TOLERANCE`."""

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
# Relative, with the same floor in absolute terms: about 840 times float32's epsilon. A
# generator or the optimizer state left out of the checkpoint puts the largest weight
# difference at 2.5e-3 to 4e-2 and 17 to 36 logged floats past the bound in the two
# updates after the resume (both models, RTX 4060 Laptop, hygiene notes). Rounding is not
# always this small: Adam scales a near-zero gradient's rounding noise up to a full step,
# so a last-bit difference in a kernel can grow past the bound in one update.
GPU_TOLERANCE = 1e-4
# Float metrics that count things: games, wins, draws and losses.
COUNTS = ("results", "games/")


def _run(mode: str, directory: pathlib.Path, device: str, model: str, *damage: str) -> None:
    environment = dict(os.environ, PYTHONPATH=str(WORKER.parent))
    done = subprocess.run(
        [sys.executable, str(WORKER), mode, str(directory), device, model, str(UPDATES), *damage],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert done.returncode == 0, done.stderr[-4000:]


def _result(directory: pathlib.Path, mode: str):
    metrics = json.loads((directory / mode / "metrics.json").read_text())
    weights = safetensors.torch.load_file(directory / mode / "weights.safetensors")
    return metrics, weights


def _differences(got, want, tolerance: float, path: str = "") -> list[str]:
    """The paths where two metric records differ: exactly for integers, strings, flags and
    counts, beyond `tolerance` (relative, with the same absolute floor) for other floats."""
    if isinstance(want, dict):
        if got.keys() != want.keys():
            return [f"{path}: keys {sorted(got)} != {sorted(want)}"]
        return [
            difference
            for key in want
            for difference in _differences(got[key], want[key], tolerance, f"{path}.{key}")
        ]
    if isinstance(want, list):
        if len(got) != len(want):
            return [f"{path}: length {len(got)} != {len(want)}"]
        return [
            difference
            for index, (item, expected) in enumerate(zip(got, want))
            for difference in _differences(item, expected, tolerance, f"{path}[{index}]")
        ]
    rounded = isinstance(want, float) and not any(count in path for count in COUNTS)
    if rounded and type(got) is float:
        close = abs(got - want) <= tolerance * max(abs(want), 1.0)
    else:
        close = type(got) is type(want) and got == want
    return [] if close else [f"{path}: {got!r} != {want!r}"]


def _resumes(tmp_path: pathlib.Path, device: str, model: str, tolerance: float = 0.0) -> None:
    """The resumed run equals the straight one: bitwise at tolerance 0, else every float
    within it (and every count, integer and categorical field exactly)."""
    for mode in ("straight", "first", "second"):
        _run(mode, tmp_path, device, model)
    straight, straight_weights = _result(tmp_path, "straight")
    first, _ = _result(tmp_path, "first")
    second, second_weights = _result(tmp_path, "second")
    if tolerance == 0.0:
        assert first + second == straight  # every logged metric, exactly (floats by repr)
    else:
        differences = _differences(first + second, straight, tolerance, "rows")
        assert not differences, "\n".join(differences[:20])
    assert straight_weights.keys() == second_weights.keys()
    for name, tensor in straight_weights.items():
        if tolerance == 0.0 or not tensor.is_floating_point():
            assert torch.equal(second_weights[name], tensor), name
        else:
            torch.testing.assert_close(
                second_weights[name], tensor, rtol=tolerance, atol=tolerance, msg=name
            )


@pytest.mark.slow
@pytest.mark.parametrize("damage", ["league", "draws", "gauges", "return_scale"])
def test_the_exact_resume_test_sees_damaged_slots_draws_or_return_scale(tmp_path, damage):
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
    _resumes(tmp_path, "cpu", model)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("model", ["relational", "yemong"])
def test_a_resumed_run_equals_one_straight_through_on_the_gpu(tmp_path, model):
    """Compiled stages and the graph-captured rollout, within rounding."""
    _resumes(tmp_path, "cuda", model, GPU_TOLERANCE)
