"""The learner's hot paths under torch.compile: the two halves of a rollout tick, a league
snapshot's move, the critic's target pass and the minibatch loss
(forward and backward). Adam and the gradient clipping stay eager (the framework spike
found nothing to gain there).

Importing this module turns duck sizing off for the process
(`torch.fx.experimental._config.use_duck_shape = False`): with it on, two sizes that happen
to be equal at the first call (ship slots and field slots at 5v5) are tied, and a new team
size recompiles. With it off and `dynamic=True` each graph compiles once and every team
size, environment count and minibatch size after costs nothing (docs/rewrite/scoreboard.md).
"""

import dataclasses
import functools
import importlib
import types

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

_core = importlib.import_module("lean.game.core")

# Workaround until lean.game keeps the card out of the trace: every deal calls
# lean.game.core.ship_card, a functools.cache whose builder ends in `.item()`. Dynamo looks
# through the cache, traces the builder and breaks the graph there, and the break inside
# reset_done's `_deal` then sends the whole deal back to eager (dynamo fails to resume the
# `uniform` closure after the break). `prepare` builds the run's card eagerly; the
# replacement below then only looks it up, so a trace reads a built card.
_CARDS: dict = {}
if not getattr(_core._ship_card, "_lean_lookup", False):
    _build_card = _core._ship_card

    def _card_lookup(*key):
        if key not in _CARDS:
            _CARDS[key] = _build_card(*key)
        return _CARDS[key]

    _card_lookup._lean_lookup = True
    _core._ship_card = _card_lookup


def prepare(game_config) -> None:
    """Build what the traced tick must find ready (the ship card of `game_config`)."""
    _core.ship_card(game_config)


# Workaround until lean.game reads the fields of the instance: reset_done iterates
# `dataclasses.fields(State)` (the class), which dynamo cannot trace (gb0180, "dataclass
# fields failure" on a class) and so breaks the graph at every deal. lean.game.core's
# `dataclasses` is swapped for a copy whose `fields` returns State's fields, taken once
# here, for the class and defers to dataclasses.fields for anything else.
if not getattr(_core.dataclasses, "_lean_fields", False):
    _STATE_FIELDS = dataclasses.fields(_core.State)

    def _fields(item):
        if item is _core.State:
            return _STATE_FIELDS
        return dataclasses.fields(item)

    _core.dataclasses = types.SimpleNamespace(
        **{name: getattr(dataclasses, name) for name in dataclasses.__all__ if name != "fields"}, fields=_fields,
        _lean_fields=True,
    )


def _compile(function):
    return torch.compile(function, dynamic=True)


act = _compile(_collect.act)
advance = _compile(_collect.advance)
_snapshot_move = _compile(_collect.snapshot_move)


def snapshot_move(config, policy, views, environment, team, carry, position_offset, generator):
    """collect.snapshot_move, compiled for two or more environments. A member playing a
    single environment runs eagerly: dynamic shapes specialise a size of 1, so it would
    otherwise be a second compile of the whole policy step."""
    move = _snapshot_move if environment.shape[0] >= 2 else _collect.snapshot_move
    return move(config, policy, views, environment, team, carry, position_offset, generator)


minibatch_values = _compile(_update.minibatch_values)
minibatch_loss = _compile(_update.minibatch_loss)
