"""Staged schedules: every `base` M until `first_stage_end` M, then the interval doubles each
time the steps double (k members per doubling, k = first_stage_end / (2 base))."""
import numpy as np
from rules import show


def doubling(base, first_stage_end, end=500):
    points, step, stage_end = [], base, first_stage_end
    position = base
    while position <= end:
        points.append(position)
        if position >= stage_end:
            step, stage_end = step * 2, stage_end * 2
        position += step
    return points


for base, first in ((5, 50), (5, 40), (5, 30), (5, 60)):
    points = doubling(base, first)
    show(f"every {base} to {first}, then doubling ({first // (2 * base)}/doubling)", points)
    print("    to 500M:", points, len(points))
