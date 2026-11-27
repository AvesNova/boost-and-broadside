"""The learner's hot paths under torch.compile: the two halves of a rollout tick, the league
networks' moves, the critic's target pass and the minibatch loss (forward and backward).
Adam and the gradient clipping stay eager (the framework spike found nothing to gain there).

`graphs(shapes)` builds them once per `shapes` (LearnConfig.compile_shapes), on the first
compiled collect or update, through lean.contract.compile_graph: "static" (a training run:
every size fixed, a recompile is an error) or "dynamic" (duck sizing off: each graph compiles
once for every team size, environment count and minibatch size). Static, the league routing
(slot_actions) keeps one axis symbolic: its table width W (`positions`, League.prepare)
follows the league's capacity, which changes as members join; every other size is fixed.
Importing this module changes nothing.
"""

import functools
from types import SimpleNamespace

import torch

from lean.contract import compile_graph
from lean.learn import rollout, training


@functools.cache
def graphs(shapes: str = "static") -> SimpleNamespace:
    """The compiled functions: act, advance, slot_actions (lean.learn.rollout),
    minibatch_values and minibatch_loss (lean.learn.training)."""
    return SimpleNamespace(
        act=compile_graph(rollout.act, shapes),
        advance=compile_graph(rollout.advance, shapes),
        slot_actions=_routing(shapes),
        minibatch_values=compile_graph(training.minibatch_values, shapes),
        minibatch_loss=compile_graph(training.minibatch_loss, shapes),
    )


def _routing(shapes: str):
    """slot_actions compiled with `shapes`; static, with the table width W (`positions`' only
    axis) symbolic, so a new W does not compile again."""
    graph = compile_graph(rollout.slot_actions, shapes)
    if shapes != "static":
        return graph

    def routed(network_, slot, positions, *rest):
        torch._dynamo.maybe_mark_dynamic(positions, 0)
        return graph(network_, slot, positions, *rest)

    return routed
