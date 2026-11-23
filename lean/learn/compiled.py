"""The learner's hot paths under torch.compile: the two halves of a rollout tick, the league
networks' moves, the critic's target pass and the minibatch loss (forward and backward).
Adam and the gradient clipping stay eager (the framework spike found nothing to gain there).

`graphs()` builds them once, on the first compiled collect or update, through
lean.contract.compile_dynamic (dynamic sizes, duck sizing off for the process: each graph
compiles once for every team size, environment count and minibatch size). Importing this
module changes nothing.
"""

import functools
from types import SimpleNamespace

from lean.contract import compile_dynamic
from lean.learn import rollout, training


@functools.cache
def graphs() -> SimpleNamespace:
    """The compiled functions: act, advance, slot_actions (lean.learn.rollout),
    minibatch_values and minibatch_loss (lean.learn.training)."""
    return SimpleNamespace(
        act=compile_dynamic(rollout.act),
        advance=compile_dynamic(rollout.advance),
        slot_actions=compile_dynamic(rollout.slot_actions),
        minibatch_values=compile_dynamic(training.minibatch_values),
        minibatch_loss=compile_dynamic(training.minibatch_loss),
    )
