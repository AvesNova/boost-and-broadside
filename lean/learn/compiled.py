"""The learner's hot paths under torch.compile: the two halves of a rollout tick, the league
networks' moves, the critic's target pass and the minibatch loss (forward and backward).
Adam and the gradient clipping stay eager (the framework spike found nothing to gain there).

`graphs()` builds them once, on the first compiled collect or update, and turns duck sizing
off for the process then (`torch.fx.experimental._config.use_duck_shape = False`): with it
on, two sizes that happen to be equal at the first call (ship slots and field slots at 5v5)
are tied, and a new team size recompiles. With it off and `dynamic=True` each graph compiles
once and every team size, environment count and minibatch size after costs nothing
(docs/rewrite/scoreboard.md). Importing this module changes nothing.
"""

import functools
from types import SimpleNamespace

import torch
import torch.fx.experimental._config as fx_config

from lean.learn import rollout, training


@functools.cache
def graphs() -> SimpleNamespace:
    """The compiled functions: act, advance, network_actions (lean.learn.rollout),
    minibatch_values and minibatch_loss (lean.learn.training)."""
    fx_config.use_duck_shape = False

    def compiled(function):
        return torch.compile(function, dynamic=True)

    return SimpleNamespace(
        act=compiled(rollout.act),
        advance=compiled(rollout.advance),
        network_actions=compiled(rollout.network_actions),
        minibatch_values=compiled(training.minibatch_values),
        minibatch_loss=compiled(training.minibatch_loss),
    )
