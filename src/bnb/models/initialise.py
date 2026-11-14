"""Keyed initialisation: every parameter's initial values drawn on the CPU from a generator of
its own, keyed by the seed, the path of the module that draws them and the parameter's name
(`bnb.streams.stream_generator`). A model option that adds a module, or changes another
module's shape, leaves every other parameter's initial values exactly as they were.

`initialise` walks the modules children first, so a module's own initialisation runs after
its children's defaults and may override them. A module initialises:
- `nn.Linear` as torch does (Kaiming-uniform weight, uniform bias in +-1/sqrt(fan in));
- a norm with its own (deterministic) `reset_parameters`;
- any other module that holds parameters of its own, or overrides its children's, with an
  `initialise_parameters(generator)` method, where `generator(name)` is the generator for
  the draw `name` (a parameter's path below the module, with a bracketed part for one of
  several draws on it: "first.weight[2]").
A module with parameters of its own and none of these is refused, so no parameter is left
to the global generator's draws. Built on the CPU; move the model to its device after.
"""

import math
from collections.abc import Callable

import torch
from torch import nn

from bnb.streams import stream_generator

Generators = Callable[[str], torch.Generator]
NORMS = (nn.RMSNorm, nn.LayerNorm)


def initialise(model: nn.Module, seed: int) -> None:
    """Initialise every parameter of `model` (on the CPU) from the seed, by parameter path."""
    with torch.no_grad():
        for path, module in reversed(list(model.named_modules())):

            def generator(name: str, path: str = path) -> torch.Generator:
                return stream_generator(seed, f"initialise/{path}:{name}")

            own = getattr(module, "initialise_parameters", None)
            if isinstance(module, nn.Linear):
                linear_default(module, generator)
            elif isinstance(module, NORMS):
                module.reset_parameters()
            elif own is None and any(True for _ in module.parameters(recurse=False)):
                raise TypeError(
                    f"{path or 'the model'} ({type(module).__name__}) holds parameters "
                    "but has no keyed initialisation (`initialise_parameters`)"
                )
            if own is not None:
                own(generator)


def linear_default(layer: nn.Linear, generator: Generators) -> None:
    """torch's default for a linear layer, from keyed generators."""
    kaiming_default(layer.weight, generator("weight"))
    if layer.bias is not None:
        uniform_by_fan_in(layer.bias, layer.weight.shape[1], generator("bias"))


def kaiming_default(weight: torch.Tensor, generator: torch.Generator) -> None:
    """A weight (outputs, inputs) as `nn.Linear` starts it: Kaiming-uniform with a = sqrt 5."""
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)


def uniform_by_fan_in(tensor: torch.Tensor, fan_in: int, generator: torch.Generator) -> None:
    """Uniform in +-1/sqrt(fan_in), as torch starts a bias (0 for no inputs)."""
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
    nn.init.uniform_(tensor, -bound, bound, generator=generator)


def orthogonal(weight: torch.Tensor, gain: float, generator: torch.Generator) -> None:
    """An orthogonal weight with this gain."""
    nn.init.orthogonal_(weight, gain, generator=generator)
