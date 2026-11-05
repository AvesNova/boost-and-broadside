"""The resume (runs acceptance 3, league acceptance 6): K updates, a training checkpoint, a
resume in a fresh process and K more continue the run that 2K updates straight through make.
Opponent slots restored wrong show, and so do the league's draws restarted, a gauge's ratings
or the ratings' games lost, or a return scale not restored. The runs play two shards and a
lookahead per update, beside rating lanes.

On the CPU the two are bitwise equal, end to end. On the GPU the contract is integrity, not
bits (contracts, "Resume: integrity, not bits"), checked in two parts:

1. The restore is exact. Right after loading, before any compute, the resumed process holds
   what the first held when it saved: parameters, optimizer, generators, carry, league,
   counters and the learning-rate schedule's position, compared bit for bit. The fields
   come from the checkpoint's own enumeration (`checkpoint.training_state`), so a new one
   is covered at once.
2. Training continues as the same run. The processes autotune their kernels from cold, and
   a configuration that wins by timing can sum in another order; Adam turns such rounding
   in near-zero gradients into full steps within one update. So elementwise floats are not
   compared: the logged floats agree within `GPU_TOLERANCE` (relative), the weights within
   `WEIGHT_TOLERANCE` (relative L2 distance), and integers and categoricals (counters,
   games, results, the opponents drawn) stay exact over the first update after the resume.
"""

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
# Chosen from measurements on the RTX 4060 Laptop (both models, the two updates after the
# resume, `resume_worker.py`'s "ulp" and "order_seed" damages against the straight run):
# (a) every weight moved one ulp at the resume, the rounding another kernel configuration
# leaves, must pass; (b) the minibatch order drawn from another seed, a different run, must
# fail. Largest relative difference of the checked floats:
#                      relational   yemong
#   (a) one ulp          4e-3        4.5e-2   (value losses; entropy 8e-4, KL 1e-3 and 3.6e-2)
#   (b) other seed       1.5e-1      2.6e-1   (approximate KL; the value losses 1e-3 and 2e-2)
# so 0.1 sits between. The final weights' relative L2 distance is 1.8e-4 and 1.9e-4 for (a),
# 5.3e-2 for (b) on the relational model but 4.2e-4 on Yemong: that one cannot tell the two
# apart, and 3e-3 only stops gross drift. Yemong is chaotic at one ulp: the sampled actions
# change, so do the rollouts (a pool's reward rate moves by 44%), and the losses on mean-zero
# or tiny quantities (the policy loss, clip fraction, explained variances, ratio drift,
# reward rates) move as much as under a different seed. Those are not compared. Elementwise
# weights are not either: Adam turns rounding in near-zero gradients into full steps (1e-4
# fails (a)).
GPU_TOLERANCE = 0.1
WEIGHT_TOLERANCE = 3e-3
FLOOR = 1e-6  # the smallest magnitude a relative difference is judged at
# The series the tolerance judges: an exact name, or a group when it ends in "/". Each must be
# in an objective's records (`continuation_differences` says so when one is not), so a rename
# cannot leave the tolerance judging nothing. PPO's and BC's KL differ.
CHECKED_BY_OBJECTIVE = {
    "ppo": ("policy/entropy/joint", "policy/kl", "critic/value_loss/"),
    "bc": ("policy/entropy/joint", "bc/kl", "critic/value_loss/"),
}
CHECKED = tuple(dict.fromkeys(c for each in CHECKED_BY_OBJECTIVE.values() for c in each))


def _matches(name: str, checked: str) -> bool:
    return name == checked or (checked.endswith("/") and name.startswith(checked))


def _checked(name: str) -> bool:
    return any(_matches(name, checked) for checked in CHECKED)


# Float metrics that count things: games, wins, draws and losses.
COUNTS = ("results", "games/")


def _run(
    mode: str,
    directory: pathlib.Path,
    device: str,
    model: str,
    *damage: str,
    objective: str = "ppo",
) -> None:
    environment = dict(os.environ, PYTHONPATH=str(WORKER.parent))
    arguments = [mode, str(directory), device, model, objective, str(UPDATES), *damage]
    done = subprocess.run(
        [sys.executable, str(WORKER), *arguments],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert done.returncode == 0, done.stderr[-4000:]


def _result(directory: pathlib.Path, mode: str):
    metrics = json.loads((directory / mode / "metrics.json").read_text())
    weights = safetensors.torch.load_file(directory / mode / "weights.safetensors")
    return metrics, weights


def _differences(got, want, tolerance: float, path: str = "", exact: bool = True) -> list[str]:
    """The paths where two metric records differ: beyond `tolerance` (relative; the floats in
    CHECKED only, when there is one) for floats, and, when `exact`, at all for integers,
    strings, flags and counts."""
    if isinstance(want, dict):
        if got.keys() != want.keys():
            return [f"{path}: keys {sorted(got)} != {sorted(want)}"]
        return [
            difference
            for key in want
            for difference in _differences(got[key], want[key], tolerance, f"{path}.{key}", exact)
        ]
    if isinstance(want, list):
        if len(got) != len(want):
            return [f"{path}: length {len(got)} != {len(want)}"]
        return [
            difference
            for index, (item, expected) in enumerate(zip(got, want))
            for difference in _differences(item, expected, tolerance, f"{path}[{index}]", exact)
        ]
    rounded = isinstance(want, float) and not any(count in path for count in COUNTS)
    if rounded and tolerance and not _checked(path.rsplit(".", 1)[-1].split("[", 1)[0]):
        close = True  # a float the tolerance does not judge (see CHECKED)
    elif rounded and type(got) is float:
        close = abs(got - want) <= tolerance * max(abs(want), FLOOR)
    elif exact:
        close = type(got) is type(want) and got == want
    else:
        close = True
    return [] if close else [f"{path}: {got!r} != {want!r}"]


def _bits(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(-1).view(torch.uint8).numpy().tobytes()


def restore_differences(directory: pathlib.Path) -> list[str]:
    """The fields where the resumed process's state, dumped right after its restore, is not
    the first process's at its save: bit for bit, every tensor the checkpoint enumerates and
    its record's counters, optimizer groups and league record."""
    first, second = (directory / mode for mode in ("first", "second"))
    out = []
    saved, restored = (json.loads((d / "state.json").read_text()) for d in (first, second))
    for key in sorted(saved.keys() | restored.keys()):
        if saved.get(key) != restored.get(key):
            out.append(f"record {key}: {restored.get(key)!r} != {saved.get(key)!r}")
    saved, restored = (torch.load(d / "state.pt") for d in (first, second))
    for name in sorted(saved.keys() | restored.keys()):
        if name not in saved or name not in restored:
            out.append(
                f"tensor {name}: missing from the {'resumed' if name in saved else 'first'} state"
            )
        elif saved[name].shape != restored[name].shape or saved[name].dtype != restored[name].dtype:
            out.append(f"tensor {name}: shape or dtype differs")
        elif _bits(saved[name]) != _bits(restored[name]):
            out.append(f"tensor {name}: not bitwise equal")
    return out


def continuation_differences(
    got: list, want: list, tolerance: float, exact_through: int, objective: str = "ppo"
) -> list[str]:
    """Where the resumed run's rows (first process then second) do not continue the straight
    run's: floats beyond `tolerance` (relative) in every row, and integers, strings,
    flags and counts at all in the rows up to `exact_through` (the first update after the
    resume; later, a different game may legitimately follow a rounding difference)."""
    if len(got) != len(want):
        return [f"rows: {len(got)} != {len(want)}"]
    out = [
        f"checked series {checked!r} is in no record"
        for checked in CHECKED_BY_OBJECTIVE[objective]
        if tolerance and not any(_matches(name, checked) for row in want for name in row)
    ]
    for index, (row, expected) in enumerate(zip(got, want)):
        out += _differences(
            row, expected, tolerance, f"rows[{index}]", exact=index <= exact_through
        )
    return out


def weight_distance(got: dict, want: dict) -> float:
    """The relative L2 distance between two weight dictionaries' floating-point tensors."""
    assert got.keys() == want.keys()
    names = [name for name, tensor in want.items() if tensor.is_floating_point()]
    for name in want:
        if name not in names:
            assert torch.equal(got[name], want[name]), name
    squared = sum(((got[n].double() - want[n].double()) ** 2).sum() for n in names)
    norm = sum((want[n].double() ** 2).sum() for n in names)
    return float((squared / norm).sqrt())


def _resumes(
    tmp_path: pathlib.Path, device: str, model: str, *, bitwise: bool, objective: str = "ppo"
) -> None:
    for mode in ("straight", "first", "second"):
        _run(mode, tmp_path, device, model, objective=objective)
    assert not (problems := restore_differences(tmp_path)), "\n".join(problems)
    straight, straight_weights = _result(tmp_path, "straight")
    first, _ = _result(tmp_path, "first")
    second, second_weights = _result(tmp_path, "second")
    if bitwise:
        assert first + second == straight  # every logged metric, exactly (floats by repr)
        for name, tensor in straight_weights.items():
            assert torch.equal(second_weights[name], tensor), name
        return
    differences = continuation_differences(
        first + second, straight, GPU_TOLERANCE, UPDATES, objective
    )
    assert not differences, "\n".join(differences[:20])
    distance = weight_distance(second_weights, straight_weights)
    assert distance <= WEIGHT_TOLERANCE, f"weights drifted by {distance:.3g} (relative L2)"


@pytest.mark.slow
@pytest.mark.parametrize("damage", ["league", "draws", "gauges", "lanes", "return_scale"])
def test_the_exact_resume_test_sees_damaged_slots_draws_or_return_scale(tmp_path, damage):
    for mode in ("straight", "first"):
        _run(mode, tmp_path, "cpu", "relational")
    _run("second", tmp_path, "cpu", "relational", damage)
    straight, _ = _result(tmp_path, "straight")
    first, _ = _result(tmp_path, "first")
    second, _ = _result(tmp_path, "second")
    assert first + second != straight


def _perturbed(tmp_path: pathlib.Path, device: str, model: str, damage: str):
    """(the rows and weights of a resume that began with `damage`, the straight run's)."""
    for mode in ("straight", "first"):
        _run(mode, tmp_path, device, model)
    _run("second", tmp_path, device, model, damage)
    straight, straight_weights = _result(tmp_path, "straight")
    first, _ = _result(tmp_path, "first")
    second, weights = _result(tmp_path, "second")
    return first + second, straight, weights, straight_weights


def _tolerance_separates(tmp_path: pathlib.Path, device: str, model: str) -> None:
    """The tolerances pass a resume whose every weight moved one ulp (the rounding a
    different kernel configuration leaves) and fail one whose minibatch order was drawn
    from another seed (a different run)."""
    rows, straight, weights, straight_weights = _perturbed(tmp_path / "ulp", device, model, "ulp")
    assert not continuation_differences(rows, straight, GPU_TOLERANCE, -1)
    assert weight_distance(weights, straight_weights) <= WEIGHT_TOLERANCE
    rows, straight, weights, straight_weights = _perturbed(
        tmp_path / "seed", device, model, "order_seed"
    )
    assert (
        continuation_differences(rows, straight, GPU_TOLERANCE, -1)
        or weight_distance(weights, straight_weights) > WEIGHT_TOLERANCE
    )


@pytest.mark.slow
def test_the_tolerances_pass_one_ulp_and_fail_another_minibatch_order_on_the_cpu(tmp_path):
    _tolerance_separates(tmp_path, "cpu", "relational")


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("model", ["relational", "yemong"])
def test_the_tolerances_pass_one_ulp_and_fail_another_minibatch_order_on_the_gpu(tmp_path, model):
    _tolerance_separates(tmp_path, "cuda", model)


@pytest.mark.slow
@pytest.mark.parametrize("model", ["relational", "yemong"])
def test_a_resumed_run_equals_one_straight_through_on_the_cpu(tmp_path, model):
    _resumes(tmp_path, "cpu", model, bitwise=True)


@pytest.mark.slow
def test_a_resumed_bc_run_equals_one_straight_through_on_the_cpu(tmp_path):
    """Behaviour cloning resumes the same way, its stop rule's window included."""
    _resumes(tmp_path, "cpu", "relational", bitwise=True, objective="bc")
    first, _ = _result(tmp_path, "first")
    assert "bc/stop/window_score" in first[-1]  # the window was full at the save


@pytest.mark.slow
@pytest.mark.parametrize("damage", ["stop_rule", "averages"])
def test_the_exact_resume_test_sees_a_lost_stop_rule_window_or_damaged_averages(tmp_path, damage):
    for mode in ("straight", "first"):
        _run(mode, tmp_path, "cpu", "relational", objective="bc")
    _run("second", tmp_path, "cpu", "relational", damage, objective="bc")
    straight, _ = _result(tmp_path, "straight")
    first, _ = _result(tmp_path, "first")
    second, _ = _result(tmp_path, "second")
    assert first + second != straight


@pytest.mark.gpu
@pytest.mark.slow
def test_a_resumed_bc_run_restores_exactly_and_continues_as_one_run_on_the_gpu(tmp_path):
    _resumes(tmp_path, "cuda", "relational", bitwise=False, objective="bc")


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("model", ["relational", "yemong"])
def test_a_resumed_run_restores_exactly_and_continues_as_one_run_on_the_gpu(tmp_path, model):
    """Compiled stages and the graph-captured rollout: the restore bit for bit, the
    continuation within rounding."""
    _resumes(tmp_path, "cuda", model, bitwise=False)
