"""The learner's hot paths under torch.compile: the two halves of a rollout tick, a league
snapshot's move, the bootstrap value, the critic's target pass and the minibatch loss
(forward and backward). Adam and the gradient clipping stay eager (the framework spike
found nothing to gain there).

Importing this module turns duck sizing off for the process
(`torch.fx.experimental._config.use_duck_shape = False`): with it on, two sizes that happen
to be equal at the first call (ship slots and field slots at 5v5) are tied, and a new team
size recompiles. With it off and `dynamic=True` each graph compiles once and every team
size, environment count and minibatch size after costs nothing (docs/rewrite/scoreboard.md).
"""

import functools
import importlib

import torch
import torch.fx.experimental._config as fx_config

# By module path: lean.learn's `collect` and `update` names are the functions.
_collect = importlib.import_module("lean.learn.collect")
_update = importlib.import_module("lean.learn.update")

fx_config.use_duck_shape = False


def _without_none_generator(factory):
    """`factory` that leaves out `generator` when it is None. Workaround (PyTorch 2.15): a
    random factory called with an explicit `generator=None` and a symbolic size fails to
    trace ("SymIntArrayRef expected to contain only concrete integers"); without the
    keyword it traces. The compiled tick passes None (a graph cannot take a generator), and
    lean.game and lean.scripted pass their generator argument through as a keyword."""

    @functools.wraps(factory)
    def call(*args, generator=None, **kwargs):
        if generator is None:
            return factory(*args, **kwargs)
        return factory(*args, generator=generator, **kwargs)

    return call


for _name in ("rand", "randn", "randint"):
    if not hasattr(getattr(torch, _name), "__wrapped__"):
        setattr(torch, _name, _without_none_generator(getattr(torch, _name)))

# Workaround until lean.game marks it itself: lean.game.core.derived_characteristics (a
# functools.cache of per-type numbers solved once, ending in `.tolist()` of float tensors)
# is called by every deal. Dynamo ignores the cache, traces the solve and breaks the graph
# at the float `.tolist()`, and the break inside reset_done's `_deal` then sends the whole
# deal back to eager (a dynamo failure resuming the `uniform` closure after the break).
# Marked as a constant result, it is called once at trace time and its numbers baked in.
_core = importlib.import_module("lean.game.core")
if not getattr(_core.derived_characteristics, "_lean_constant", False):
    _core.derived_characteristics = torch.compiler.assume_constant_result(_core.derived_characteristics)
    _core.derived_characteristics._lean_constant = True


def _compile(function):
    return torch.compile(function, dynamic=True)


act = _compile(_collect.act)
advance = _compile(_collect.advance)
bootstrap = _compile(_collect.bootstrap)
snapshot_move = _compile(_collect.snapshot_move)
minibatch_values = _compile(_update.minibatch_values)
minibatch_loss = _compile(_update.minibatch_loss)
