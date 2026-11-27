"""The learner's hot paths under torch.compile: the two halves of a rollout tick, the league
networks' moves, the critic's target pass and the minibatch loss (forward and
backward). Adam and the gradient clipping stay eager (the framework spike found nothing to gain
there).

`graphs(shapes)` builds them once per `shapes` (LearnConfig.compile_shapes), on the first
compiled collect or update, through lean.contract.compile_graph: "static" (a training run:
every size fixed, a recompile is an error) or "dynamic" (duck sizing off: each graph compiles
once for every team size, environment count and minibatch size). Static, every size is fixed,
the league routing's too (its table width W and slot count K are fixed for the run,
lean.learn.pool). Importing this module changes nothing.
"""

import functools
from types import SimpleNamespace

from lean.contract import compile_graph
from lean.learn import rollout, training


@functools.cache
def graphs(shapes: str = "static") -> SimpleNamespace:
    """The compiled functions: act, advance, league_networks (lean.learn.rollout),
    minibatch_values and minibatch_loss (lean.learn.training)."""
    return SimpleNamespace(
        act=compile_graph(rollout.act, shapes),
        advance=compile_graph(rollout.advance, shapes),
        league_networks=compile_graph(rollout.league_networks, shapes),
        minibatch_values=compile_graph(training.minibatch_values, shapes),
        minibatch_loss=compile_graph(training.minibatch_loss, shapes),
    )
