"""The learner's hot paths under torch.compile: the two halves of a rollout tick, the league
snapshots' moves, the critic's target pass and the minibatch loss
(forward and backward). Adam and the gradient clipping stay eager (the framework spike
found nothing to gain there).

Importing this module turns duck sizing off for the process
(`torch.fx.experimental._config.use_duck_shape = False`): with it on, two sizes that happen
to be equal at the first call (ship slots and field slots at 5v5) are tied, and a new team
size recompiles. With it off and `dynamic=True` each graph compiles once and every team
size, environment count and minibatch size after costs nothing (docs/rewrite/scoreboard.md).
"""

import importlib

import torch
import torch.fx.experimental._config as fx_config

# By module path: lean.learn's `collect` and `update` names are the functions.
_collect = importlib.import_module("lean.learn.collect")
_update = importlib.import_module("lean.learn.update")

fx_config.use_duck_shape = False


def _compile(function):
    return torch.compile(function, dynamic=True)


act = _compile(_collect.act)
advance = _compile(_collect.advance)
snapshot_actions = _compile(_collect.snapshot_actions)
minibatch_values = _compile(_update.minibatch_values)
minibatch_loss = _compile(_update.minibatch_loss)
