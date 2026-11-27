"""The learner's hot paths under torch.compile: the two halves of a rollout tick, the scripted
agent's call, the league networks' moves and their merge, the critic's target pass and the
minibatch loss (forward and backward). Adam and the gradient clipping stay eager (the framework
spike found nothing to gain there).

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


class PerEnvironments:
    """`function` (its second argument a game state) compiled once per environment count:
    the scripted agent's call runs on the league environments and, while the teacher is on,
    the self-play ones (rollout.scripted_moves); at self_play_share 0.5 both are one graph,
    otherwise each count has its own, both compiled on the first tick."""

    def __init__(self, function, shapes: str):
        self.function, self.shapes, self.compiled = function, shapes, {}

    def __call__(self, first, state, *arguments):
        envs = state.exists.shape[0]
        if envs not in self.compiled:
            self.compiled[envs] = compile_graph(self.function, self.shapes)
        return self.compiled[envs](first, state, *arguments)


@functools.cache
def graphs(shapes: str = "static") -> SimpleNamespace:
    """The compiled functions: act, advance, scripted_actions, slot_actions, merge_league
    (lean.learn.rollout), minibatch_values and minibatch_loss (lean.learn.training)."""
    return SimpleNamespace(
        act=compile_graph(rollout.act, shapes),
        advance=compile_graph(rollout.advance, shapes),
        scripted_actions=PerEnvironments(rollout.scripted_actions, shapes),
        slot_actions=compile_graph(rollout.slot_actions, shapes),
        merge_league=compile_graph(rollout.merge_league, shapes),
        minibatch_values=compile_graph(training.minibatch_values, shapes),
        minibatch_loss=compile_graph(training.minibatch_loss, shapes),
    )
