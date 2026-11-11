"""The realtime match (`bnb.runtime.realtime`): its own generator, its events, keyframes and
the background compile."""

import pytest
import torch
from torch.utils import _pytree

from bnb import cache
from bnb.contracts import GameConfig
from bnb.runtime import built as built_module
from bnb.runtime.realtime import Realtime

CONFIG = GameConfig(2, 4, 900.0, episode_seconds=2.0)


def _actions(generator: torch.Generator, ships: int) -> torch.Tensor:
    power = torch.randint(0, 3, (ships,), generator=generator)
    stick = torch.randint(0, 7, (ships,), generator=generator)
    shoot = torch.randint(0, 2, (ships,), generator=generator)
    return torch.stack([power, stick, shoot], -1).to(torch.int32)


def _run(realtime: Realtime, ticks: int, seed: int = 3) -> list[list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    states = []
    for _ in range(ticks):
        realtime.step(_actions(generator, CONFIG.num_ships))
        states.append([leaf.clone() for leaf in _pytree.tree_leaves(realtime.state)])
    return states


def _equal(a, b) -> bool:
    """Same states: integers and flags exactly, floats to float32's tolerance (no test asks
    for bitwise floats)."""
    return all(_same(x, y) for x, y in zip(a, b, strict=True))


def _same(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.dtype.is_floating_point:
        return torch.allclose(x, y, rtol=1e-5, atol=1e-5)
    return torch.equal(x, y)


def test_the_matchs_generator_decides_every_state():
    """Two matches with equally seeded generators agree, without touching torch's global
    generator; another seed differs."""
    first = Realtime(CONFIG, compiled=False, generator=torch.Generator().manual_seed(5))
    second = Realtime(CONFIG, compiled=False, generator=torch.Generator().manual_seed(5))
    other = Realtime(CONFIG, compiled=False, generator=torch.Generator().manual_seed(6))
    a, b, c = _run(first, 70), _run(second, 70), _run(other, 70)
    assert all(_equal(x, y) for x, y in zip(a, b))
    assert not all(_equal(x, y) for x, y in zip(a, c))


def test_a_restored_keyframe_continues_the_same():
    realtime = Realtime(CONFIG, compiled=False)
    _run(realtime, 25, seed=1)
    keyframe = realtime.keyframe()
    want = _run(realtime, 50, seed=2)
    realtime.restore(keyframe)
    got = _run(realtime, 50, seed=2)
    assert all(_equal(x, y) for x, y in zip(got, want))


def test_the_tick_keeps_its_events():
    realtime = Realtime(CONFIG, compiled=False)
    assert realtime.events is None
    done = []
    generator = torch.Generator().manual_seed(0)
    for _ in range(61):
        realtime.step(_actions(generator, CONFIG.num_ships))
        done.append(bool(realtime.events.done[0]))
    assert realtime.events.died.shape == (1, CONFIG.num_ships)
    assert done.index(True) == 59  # the 2 s episode ends on its 60th tick


@pytest.mark.slow
@pytest.mark.gpu  # compiling, even for the CPU, initialises CUDA
@pytest.mark.usefixtures("fresh_compiler")
def test_the_background_compile_swaps_in_and_keeps_the_states():
    """The match runs the uncompiled tick while the compile runs, then the compiled one;
    the states match an all-uncompiled match to float tolerance, flags exactly."""
    background = Realtime(CONFIG, compiled=True, background=True)
    eager = Realtime(CONFIG, compiled=False)
    assert background.compiling
    assert not background.uses_compiled
    background.wait_compiled()
    generator = torch.Generator().manual_seed(9)
    for _ in range(10):
        actions = _actions(generator, CONFIG.num_ships)
        background.step(actions)
        eager.step(actions)
    assert background.uses_compiled
    for got, want in zip(
        _pytree.tree_leaves(background.state), _pytree.tree_leaves(eager.state), strict=True
    ):
        if want.dtype.is_floating_point:
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-3)
        else:
            assert torch.equal(got, want)


@pytest.mark.slow
def test_the_built_tick_is_the_uncompiled_tick_and_is_kept_on_disk(tmp_path, monkeypatch):
    """The tick built in a child process gives the uncompiled tick's states, floats to
    fusion rounding and flags exactly; it is written once, and a later process (this
    one's loaded ticks forgotten) loads it from disk instead of building it again."""
    monkeypatch.setattr(cache, "ROOT", tmp_path)
    monkeypatch.setattr(built_module, "_LOADED", {})
    built, eager = Realtime(CONFIG, compiled=True), Realtime(CONFIG, compiled=False)
    libraries = list(tmp_path.glob("ticks/*.pt2"))
    assert len(libraries) == 1
    generator = torch.Generator().manual_seed(5)
    for _ in range(10):
        actions = _actions(generator, CONFIG.num_ships)
        built.step(actions)
        eager.step(actions)
    assert built.uses_compiled
    for got, want in zip(
        _pytree.tree_leaves((built.state, built.obs)),
        _pytree.tree_leaves((eager.state, eager.obs)),
        strict=True,
    ):
        if want.dtype.is_floating_point:
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-3)
        else:
            assert torch.equal(got, want)

    monkeypatch.setattr(built_module, "_LOADED", {})
    stamp = libraries[0].stat().st_mtime_ns
    again = Realtime(CONFIG, compiled=True)
    again.step(torch.zeros(CONFIG.num_ships, 3, dtype=torch.int32))
    assert again.uses_compiled
    assert list(tmp_path.glob("ticks/*.pt2")) == libraries
    assert libraries[0].stat().st_mtime_ns == stamp


def test_a_tick_state_shares_no_tensor_between_its_fields():
    """A state whose fields share a tensor fails a compiled tick's guards (traced on
    unshared states) and recompiles on the window's thread."""
    realtime = Realtime(CONFIG, compiled=False)
    realtime.step(torch.zeros(CONFIG.num_ships, 3, dtype=torch.int32))
    leaves = _pytree.tree_leaves(realtime.state)
    assert len({id(leaf) for leaf in leaves}) == len(leaves)
