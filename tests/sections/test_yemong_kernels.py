"""Yemong's Triton kernels against their plain PyTorch twins: each kernel alone, forward and
backward, and the whole model with every kernel on against it with every kernel off, in
float32 and bf16, both measured against the plain path in float64. Every kernel gives
bitwise-identical results on identical inputs (the exact-resume tests depend on it)."""

import copy
import dataclasses
import pathlib
import sys

import pytest
import torch
import yemong_cases as cases
from torch.utils import _pytree

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # `bench` is a top-level package of the repository
    sys.path.insert(0, str(ROOT))

from bench.yemong_synthetic import random_segment, time_slice  # noqa: E402
from bnb.models.yemong import kernels  # noqa: E402
from bnb.models.yemong.kernels import scan  # noqa: E402
from bnb.models.yemong.temporal import parallel_scan  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels need CUDA")
CUDA = torch.device("cuda")
OWN, SHIPS, FIELDS, ZONES = 5, 10, 10, 5
OUTPUTS = ("policy_logits", "enemy_action_logits", "next_state_logits", "value_logits",
           "outcome_logits", "density_log_rates", "shot_outcome_logits")


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device=CUDA).manual_seed(seed)


def _weighted_sum(outputs, seed: int = 1) -> torch.Tensor:
    """A loss whose gradient reaches every element with its own weight."""
    generator = _generator(seed)
    return sum((output.double() * torch.randn(output.shape, generator=generator, device=CUDA,
                                              dtype=torch.float64)).sum() for output in outputs)


def _gradients(function, inputs, seed: int = 1):
    """(outputs, gradients of the inputs) of a weighted sum of function's outputs."""
    inputs = [x.detach().requires_grad_() for x in inputs]
    outputs = function(*inputs)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    gradients = torch.autograd.grad(_weighted_sum(outputs, seed), inputs)
    return [x.detach() for x in outputs], list(gradients)


def _relative_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """rms(actual - expected) / rms(expected): a systematic error shows here, where rounding
    averages out."""
    actual, expected = actual.double(), expected.double()
    return ((actual - expected).square().mean().sqrt()
            / expected.square().mean().sqrt().clamp_min(1e-30)).item()


def _assert_as_close(got, slow, want, floor: float = 0.0, rms_floor: float | None = None,
                     label: str = ""):
    """The kernel's result `got` is as close to the float64 reference `want` as the plain
    path's `slow`: its largest error within twice the plain path's (or the floor), and its
    rms error within 1.5 times the plain path's (or the rms floor, by default the floor)."""
    allowed = max(2.0 * cases.scale_error(slow.double(), want), floor)
    assert cases.scale_error(got.double(), want) <= allowed, label
    allowed = max(1.5 * _relative_rms(slow, want), floor if rms_floor is None else rms_floor)
    assert _relative_rms(got, want) <= allowed, label


def _bf16(function):
    def run(*inputs):
        with torch.autocast("cuda", torch.bfloat16):
            return function(*inputs)
    return run


def _assert_identical(first, second):
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


# ------------------------------------------------------------------ scan


def _scan_inputs(seed: int = 0, length: int = 37, shape=(3, 5, 16)):
    generator = _generator(seed)
    decay = torch.rand(length, *shape, generator=generator, device=CUDA)
    decay[11, 1] = 0.0  # an episode starts here: the recurrence is cut
    contribution = torch.randn(length, *shape, generator=generator, device=CUDA)
    initial = torch.randn(shape, generator=generator, device=CUDA)
    return decay, contribution, initial


def test_scan_equals_the_parallel_scan():
    inputs = _scan_inputs()
    expected = _gradients(lambda *x: parallel_scan(*x)[0], [x.double() for x in inputs])
    actual = _gradients(scan.linear_scan, inputs)
    for got, want in zip(actual[0] + actual[1], expected[0] + expected[1], strict=True):
        torch.testing.assert_close(got.double(), want, rtol=1e-5, atol=1e-5)


def test_scan_is_bitwise_repeatable():
    inputs = _scan_inputs(seed=2)
    first, second = (_gradients(scan.linear_scan, inputs) for _ in range(2))
    _assert_identical(first[0] + first[1], second[0] + second[1])


# ------------------------------------------------------------------ the whole model


@pytest.fixture(scope="module")
def segment():
    """A float64 model on CUDA, a synthetic segment with a reset mid-way, a nonzero carry and
    the replay records `step` stored over the segment."""
    net = cases.double(cases.model()).to(CUDA)
    steps, views = 24, 6
    obs, reset = random_segment(steps, views, SHIPS, FIELDS, ZONES, device=CUDA,
                                dtype=torch.float64, seed=3, reset_steps=(9,))
    carry = net.initial_carry(views, OWN, SHIPS, device=CUDA)
    generator = _generator(4)
    carry = dataclasses.replace(
        carry, state=torch.randn(carry.state.shape, generator=generator, device=CUDA,
                                 dtype=torch.float64),
        convolution_buffer=torch.randn(carry.convolution_buffer.shape, generator=generator,
                                       device=CUDA, dtype=torch.float64))
    replays, stepped = [], carry
    with torch.no_grad():
        for t in range(steps):
            decision = net.step(stepped, time_slice(obs, t), reset[t])
            stepped = decision.carry
            replays.append(decision.replay)
    replay = _pytree.tree_map(lambda *x: torch.stack(x), *replays)
    ages = torch.randint(1, 13, (steps, views, SHIPS, 4), generator=generator, device=CUDA)
    return net, carry, obs, replay, reset, ages


def _model(net, precision: str):
    """A copy of the float64 net in the precision's dtype, in training mode."""
    model = copy.deepcopy(net).to(torch.float64 if precision == "float64" else torch.float32)
    model.config = dataclasses.replace(model.config,
                                       bfloat16_matmuls=precision == "bfloat16")
    return model.train()


def _model_run(segment, precision: str):
    """The outputs and the parameter gradients of a weighted sum of them."""
    net, carry, obs, replay, reset, ages = segment
    model = _model(net, precision)
    dtype = torch.float64 if precision == "float64" else torch.float32
    carry, obs, replay = (cases.cast(tree, dtype) for tree in (carry, obs, replay))
    unrolled = model.unroll(carry, obs, replay, reset)
    outputs = [getattr(unrolled, name) for name in OUTPUTS]
    outputs.append(model.shot_trajectory(unrolled.shot_latents, ages.to(dtype)))
    _weighted_sum(outputs).backward()
    gradients = [parameter.grad for _, parameter in model.named_parameters()]
    names = [name for name, _ in model.named_parameters()]
    return [output.detach() for output in outputs], gradients, names


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_kernels_are_as_close_to_float64_as_the_plain_path(segment, precision):
    """Every output and every parameter gradient, with the kernels on, is within twice the
    plain path's error of the float64 reference (`_assert_as_close`, with a floor at the
    precision's rounding)."""
    reference, reference_gradients, names = _model_run(segment, "float64")
    with kernels.plain():
        plain, plain_gradients, _ = _model_run(segment, precision)
    fast, fast_gradients, _ = _model_run(segment, precision)
    floor, rms_floor = (1e-5, 1e-5) if precision == "float32" else (1e-2, 1e-3)
    labels = list(OUTPUTS) + ["shot_trajectory"] + names
    for label, want, slow, got in zip(labels, reference + reference_gradients,
                                      plain + plain_gradients, fast + fast_gradients,
                                      strict=True):
        if want is None:
            assert slow is None and got is None, label
            continue
        _assert_as_close(got, slow, want, floor, rms_floor, label)


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_the_model_with_kernels_is_bitwise_repeatable(segment, precision):
    first, first_gradients, _ = _model_run(segment, precision)
    second, second_gradients, _ = _model_run(segment, precision)
    _assert_identical(first + [g for g in first_gradients if g is not None],
                      second + [g for g in second_gradients if g is not None])
