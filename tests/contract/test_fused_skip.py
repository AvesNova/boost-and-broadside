"""The learner's guarded step (`Learner._step`) on the fused CUDA Adam: a finite step equals
the plain fused step bit for bit, and a non-finite one leaves the parameters and the
optimizer's state as they were (a first step's state at zero)."""

import pytest
import torch

from bnb.learn.ppo import Adam, Learner

pytestmark = pytest.mark.gpu


class _Stub:
    """Only what `Learner._step` reads."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    _step = Learner._step


def _model(seed: int) -> torch.nn.Module:
    with torch.random.fork_rng(devices=[]):  # the global generator stays as it was
        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(7, 5), torch.nn.GELU(), torch.nn.Linear(5, 3))
    return model.cuda()


def _backward(model: torch.nn.Module, seed: int, poison: bool = False) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(11, 7, device="cuda", generator=generator)
    model.zero_grad(set_to_none=True)
    model(x).square().sum().backward()
    if poison:
        next(model.parameters()).grad[0, 0] = float("nan")


def _snapshot(model, optimizer):
    parameters = [p.detach().clone() for p in model.parameters()]
    state = [
        value.detach().clone()
        for entry in optimizer.state.values()
        for value in entry.values()
        if isinstance(value, torch.Tensor)
    ]
    return parameters, state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_finite_steps_equal_the_plain_fused_step():
    guarded, plain = _model(0), _model(0)
    guarded_optimizer = Adam(guarded.parameters(), lr=1e-2, fused=True)
    plain_optimizer = Adam(plain.parameters(), lr=1e-2, fused=True)
    stub = _Stub(guarded_optimizer)
    for step in range(5):
        _backward(guarded, step)
        _backward(plain, step)
        stub._step(torch.tensor(True, device="cuda"))
        plain_optimizer.step()
    for a, b in zip(_snapshot(guarded, guarded_optimizer), _snapshot(plain, plain_optimizer)):
        for x, y in zip(a, b, strict=True):
            assert torch.equal(x, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_nonfinite_step_changes_nothing():
    model = _model(1)
    optimizer = Adam(model.parameters(), lr=1e-2, fused=True)
    stub = _Stub(optimizer)
    # a first step skipped: the parameters as they were, the created state zero
    before = [p.detach().clone() for p in model.parameters()]
    _backward(model, 0, poison=True)
    stub._step(torch.tensor(False, device="cuda"))
    for p, b in zip(model.parameters(), before, strict=True):
        assert torch.equal(p, b)
    for entry in optimizer.state.values():
        for value in entry.values():
            assert torch.count_nonzero(value) == 0
    # later steps: a skipped one between two finite ones
    _backward(model, 1)
    stub._step(torch.tensor(True, device="cuda"))
    parameters, state = _snapshot(model, optimizer)
    _backward(model, 2, poison=True)
    stub._step(torch.tensor(False, device="cuda"))
    after_parameters, after_state = _snapshot(model, optimizer)
    for x, y in zip(parameters + state, after_parameters + after_state, strict=True):
        assert torch.equal(x, y)
    assert not hasattr(optimizer, "found_inf")
