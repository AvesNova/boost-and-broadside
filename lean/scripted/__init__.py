"""The scripted opponent (the look-ahead agent, the yardstick and its persona population) and
the chaser baseline."""

from .chaser import ChaserConfig, chaser_act
from .opponent import Choice, act, decide, draw_uniforms, explain, fresh_memory, sample
from .parameters import PARAMETERS, Parameter, ScriptedConfig, personas

__all__ = [
    "PARAMETERS",
    "Parameter",
    "ChaserConfig",
    "Choice",
    "ScriptedConfig",
    "act",
    "chaser_act",
    "decide",
    "draw_uniforms",
    "explain",
    "fresh_memory",
    "personas",
    "sample",
]
