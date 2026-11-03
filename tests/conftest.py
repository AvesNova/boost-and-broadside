"""Suite-wide fixtures: the benchmark lock for GPU tests, the CUDA and random-state guards,
and clean skips on a machine without CUDA (code style, Tests)."""

import os
import sys

import pytest
import torch

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # Triton 3.9's default, made explicit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.lock import bench_lock  # noqa: E402

# The play shell's gpu tests need an OpenGL context, not CUDA, and skip on their own.
OPENGL_MODULES = ("test_play_",)


def pytest_collection_modifyitems(config, items):
    """Without CUDA, every CUDA test skips (a `gpu` test never falls back to the CPU)."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs CUDA")
    for item in items:
        if "gpu" in item.keywords and not item.module.__name__.split(".")[-1].startswith(
            OPENGL_MODULES
        ):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _gpu_tests_hold_the_benchmark_lock(request):
    """A `gpu` test holds the machine's benchmark lock, so another process on the GPU cannot
    make it run out of memory or invalidate a graph capture."""
    if "gpu" in request.keywords:
        with bench_lock():
            yield
    else:
        yield


def _no_cuda() -> int:
    raise RuntimeError("a test without the gpu marker sees no CUDA device")


@pytest.fixture(autouse=True)
def _only_gpu_tests_touch_cuda(request, monkeypatch):
    """A test without the `gpu` marker must not initialise CUDA, so `-m "not gpu"` never
    does. It runs as on a machine without CUDA, since `torch.compile` initialises CUDA even
    for a graph on the CPU: it saves the CUDA generator's state around every compile when
    `torch.cuda.is_available()`, and its cache key reads the current device's name (once per
    process; without CUDA, the read raises and the key leaves the device out). After a gpu
    test has initialised CUDA, a non-gpu test sees it as it is."""
    gpu, before = "gpu" in request.keywords, torch.cuda.is_initialized()
    if not gpu and not before:  # once a gpu test has initialised CUDA, there is nothing to keep
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "current_device", _no_cuda)
    yield
    if not gpu and not before and torch.cuda.is_initialized():
        pytest.fail("initialised CUDA without the gpu marker", pytrace=False)


@pytest.fixture(autouse=True)
def _the_global_generator_is_unchanged():
    """A test leaves torch's global CPU generator as it found it, so no test's inputs depend
    on which tests ran before it: draw from a local `torch.Generator`, or use the
    `global_generator` fixture where the code under test draws from the default generator."""
    before = torch.get_rng_state()
    yield
    if not torch.equal(before, torch.get_rng_state()):
        pytest.fail(
            "changed torch's global CPU generator: draw from a local torch.Generator, or use "
            "the global_generator fixture",
            pytrace=False,
        )


@pytest.fixture
def global_generator():
    """For code that draws from torch's default generators by design (a model's
    initialisation, the rollout's draws, the scripted opponents): the test runs with them
    seeded at 0 and restores them afterwards."""
    with torch.random.fork_rng(
        devices=range(torch.cuda.device_count()) if torch.cuda.is_initialized() else []
    ):
        torch.manual_seed(0)
        yield


@pytest.fixture
def fresh_compiler():
    """Dynamo's caches emptied first, so a test that compiles does not depend on what the
    tests before it compiled (its recompile limit counts every earlier shape)."""
    torch._dynamo.reset()
    yield
