"""The scripted opponent (the yardstick and its persona population) and the chaser baseline."""

from .chaser import ChaserConfig, chaser_act
from .opponent import (
    GATES,
    Marginals,
    ScriptedConfig,
    act,
    decide,
    draw_uniforms,
    fresh_memory,
    marginals,
    personas,
    renew_memory,
    sample,
)

__all__ = [
    "GATES",
    "ChaserConfig",
    "Marginals",
    "ScriptedConfig",
    "act",
    "chaser_act",
    "decide",
    "draw_uniforms",
    "fresh_memory",
    "marginals",
    "personas",
    "renew_memory",
    "sample",
]
