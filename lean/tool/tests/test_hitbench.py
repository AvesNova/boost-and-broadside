"""The hit-model workbench (lean.tool.hitbench) reads the step: its effective radius is the
graze's closed form on the pushed distance, from the code's own numbers."""

import math

import torch

from lean.game import card, core
from lean.tool import hitbench

torch.set_num_threads(1)


def test_effective_radius_from_astern_is_the_graze_closed_form_minus_d():
    """A smoothstep from the core to the hit radius integrates to the core plus half the band;
    the deflector's D (side value, astern) takes D off it; head-on the frontal preset and the
    nose cone do nothing."""
    astern = hitbench.ASPECTS_DEG.index(180)
    for name in ("off", "symmetric", "frontal"):
        bench = hitbench.measure(name)
        radius, core_radius = bench.inputs["hit_radius"], bench.inputs["core_radius"]
        expected = 0.5 * (radius + core_radius) - card.DEFLECTOR_PRESETS[name]["deflector_side"]
        measured = hitbench.effective_radius(bench)[astern]
        assert torch.allclose(measured, torch.full_like(measured, expected), atol=0.02), name
        assert float(bench.damage[hitbench.ASPECTS_DEG.index(0)].abs().max()) == 0.0, name


def test_a_set_input_reaches_the_step():
    bench = hitbench.measure("symmetric", {"hit_radius": 16.0, "deflector_side": 1.0})
    astern = hitbench.ASPECTS_DEG.index(180)
    expected = 0.5 * (16.0 + bench.inputs["core_radius"]) - 1.0
    assert abs(float(hitbench.effective_radius(bench)[astern, 0]) - expected) < 0.02
    assert hitbench.parse_set(["deflector_inner_angle_deg=20"]) == {
        "deflector_inner_angle": math.radians(20.0)
    }
    assert {entry.name for entry in core.HIT_MODEL} >= set(bench.inputs)
