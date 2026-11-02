"""Model names to classes (runs spec 4). Loading goes by the registered name, never by a
Python class path, so renaming a class or a module leaves old checkpoints loadable.

Each registered class has NAME and VERSION (an int that rises whenever a change alters what
saved weights compute), and a constructor that takes its build record alone (BUILDS names
the record's dataclass); an instance keeps its record as `build`.
"""

import torch

from bnb.models.mlp import MLP, MLPBuild
from bnb.models.relational import Relational, RelationalBuild
from bnb.models.yemong.model import Yemong, YemongBuild

BUILDS: dict[type, type] = {Yemong: YemongBuild, Relational: RelationalBuild, MLP: MLPBuild}
MODELS: dict[str, type] = {cls.NAME: cls for cls in BUILDS}


def model_class(name: str) -> type:
    """The class registered as `name`."""
    if name not in MODELS:
        raise ValueError(f"no model is registered as {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name]


def name_of(model: torch.nn.Module) -> str:
    """The registered name of a model instance."""
    cls = type(model)
    if cls not in BUILDS:
        raise ValueError(f"{cls.__name__} is not a registered model; registered: "
                         f"{sorted(MODELS)}")
    return cls.NAME
