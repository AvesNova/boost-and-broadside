"""Functions built ahead of time for the play shell: AOTInductor's C++, built in a child
process, kept on disk and loaded in milliseconds (the realtime tick, the scripted agent's
step).

A function here is a picklable callable on a tree of tensors (a dataclass of them, a
tuple, ...) returning another. `library` builds it, for the shapes of example arguments, in
a child process: building it in the calling process would trace and compile on the
interpreter lock for tens of seconds and freeze the window's thread however the work was
threaded; a child has its own, and the calling thread waits for it without holding the
lock. The package is kept at a path the caller chooses in `bnb.cache` (keyed by the code it
holds and whatever else fixes the function), written aside and moved in, so a reader never
meets half a file; once loaded it is shared by the process (`loaded`).

A random op in a built function draws from torch's default generator, as the uncompiled
function does (inductor's `fallback_random`): a caller that swaps its own generator in for
the default one gets the same draws from either. The built function agrees with the
uncompiled one to float32 fusion rounding.
"""

import multiprocessing
import os
import pathlib
import threading
from collections.abc import Callable

import torch
from torch.utils import _pytree

INDUCTOR_SETTINGS = {"fallback_random": True}  # random ops call torch's own, on its generator
_LOADED: dict[pathlib.Path, Library] = {}  # this process's loaded libraries, by path
_GETTING: dict[pathlib.Path, threading.Lock] = {}  # held while a path is built or loaded
_LOCK = threading.Lock()


def clone(tree):
    """A deep copy of a tensor tree (a state, an observation, a carry)."""
    return _pytree.tree_map(torch.clone, tree)


def _tensor_positions(leaves: list) -> list[int]:
    return [index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]


class _Flat(torch.nn.Module):
    """A function on its arguments' tensor leaves, returning its outputs' tensor leaves (what
    export packages: the trees themselves are rebuilt by `Library`)."""

    def __init__(self, function: Callable, arguments: tuple) -> None:
        super().__init__()
        self.function = function
        self.leaves, self.spec = _pytree.tree_flatten(arguments)
        self.positions = _tensor_positions(self.leaves)

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        leaves = list(self.leaves)
        for position, tensor in zip(self.positions, tensors, strict=True):
            leaves[position] = tensor
        out = self.function(*_pytree.tree_unflatten(leaves, self.spec))
        return tuple(leaf for leaf in _pytree.tree_leaves(out) if isinstance(leaf, torch.Tensor))


def _build(path: pathlib.Path, function: Callable, arguments: tuple) -> None:
    """In a child process: export the function for these arguments' shapes and package it
    at `path` (written aside, then moved in)."""
    torch.set_num_threads(1)
    module = _Flat(function, arguments)
    tensors = tuple(module.leaves[position] for position in module.positions)
    with torch.no_grad():
        program = torch.export.export(module, tensors, strict=True)
    partial = path.with_name(f"{path.stem}.{os.getpid()}.partial.pt2")
    torch._inductor.aoti_compile_and_package(
        program, package_path=str(partial), inductor_configs=INDUCTOR_SETTINGS
    )
    partial.replace(path)


class Library:
    """A built function, loaded: called like the uncompiled function, on trees of the shapes
    of `arguments`, returning trees of the structure of `outputs` (an uncompiled call's)."""

    def __init__(self, path: pathlib.Path, arguments: tuple, outputs) -> None:
        self._run = torch._inductor.aoti_load_package(str(path))
        self._positions = _tensor_positions(_pytree.tree_leaves(arguments))
        self._out_leaves, self._out_spec = _pytree.tree_flatten(outputs)
        self._out_positions = _tensor_positions(self._out_leaves)

    def __call__(self, *arguments):
        leaves = _pytree.tree_leaves(arguments)
        tensors = self._run(*[leaves[position] for position in self._positions])
        out = list(self._out_leaves)
        for position, tensor in zip(self._out_positions, tensors, strict=True):
            out[position] = tensor
        return _pytree.tree_unflatten(out, self._out_spec)


def loaded(path: pathlib.Path) -> Library | None:
    """The library at this path if this process has loaded it."""
    return _LOADED.get(path)


def library(
    path: pathlib.Path, function: Callable, arguments: tuple, outputs, name: str
) -> Library:
    """The process's library at `path`: loaded if this process has it, read from disk if it
    was built before, else built first in a child process named `name` (the calling thread
    waits without holding the interpreter lock). `function` must pickle (the child builds
    it); `arguments` give the shapes it is built for and `outputs`, what the uncompiled
    function returned on them, the structure of its results. Callers on several threads
    share one build."""
    with _LOCK:
        getting = _GETTING.setdefault(path, threading.Lock())
    with getting:  # a second caller for the same path waits for the first one's library
        if path in _LOADED:
            return _LOADED[path]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            child = multiprocessing.get_context("spawn").Process(
                target=_build, args=(path, function, arguments), name=name, daemon=True
            )
            child.start()
            child.join()
            if child.exitcode != 0 or not path.exists():
                raise RuntimeError(f"building {name} failed (exit code {child.exitcode})")
        _LOADED[path] = Library(path, arguments, outputs)
        return _LOADED[path]
