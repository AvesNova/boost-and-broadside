"""Every registered renderer.

Importing this package registers the complete set, so the figure set can be
validated against what actually exists rather than against a list maintained by
hand somewhere else.
"""

from boost_and_broadside.charts.renderers import (  # noqa: F401
    ar_report,
    crossover,
    elo_calibration,
    elo_scale,
    noise_calibration,
    semi_random,
    training,
)

__all__ = [
    "ar_report",
    "crossover",
    "elo_calibration",
    "elo_scale",
    "noise_calibration",
    "semi_random",
    "training",
]
