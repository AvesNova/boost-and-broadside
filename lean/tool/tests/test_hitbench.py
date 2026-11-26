"""The hit-model workbench (lean.tool.hitbench) reads the step: with no turn its effective
radius is the graze's closed form, and with the egg its hit width is the game's hit widths
(core.hit_widths), from the code's own numbers."""

import math

import torch

from lean.game import core
from lean.tool import hitbench

torch.set_num_threads(1)


def test_no_turn_is_the_graze_closed_form():
    """A smoothstep from the core to the hit radius integrates to the core plus half the band,
    from astern and from the side, with the deflector's turn off (within 0.3 m: a pass near
    the band's edge is hit one tick later, its bullet slowed by one more tick of drag)."""
    bench = hitbench.measure("off")
    radius, core_radius = bench.inputs["hit_radius"], bench.inputs["core_radius"]
    for aspect in (90, 180):
        measured = hitbench.effective_radius(bench)[hitbench.ASPECTS_DEG.index(aspect)]
        expected = torch.full_like(measured, 0.5 * (radius + core_radius))
        assert torch.allclose(measured, expected, atol=0.3), aspect


def test_the_egg_hit_width_is_the_games_table():
    """With the egg the band of damaging offsets equals core.hit_widths at that aspect (a still
    ship there; the target's move at cruise shifts it by a few tenths of a metre)."""
    bench = hitbench.measure("egg")
    widths = core.hit_widths(torch.tensor([bench.inputs["hit_radius"]]), bench.inputs)[0]
    for aspect in (0, 90, 180):
        table = float(core.hit_width(widths, torch.tensor(math.cos(math.radians(aspect)))))
        measured = hitbench.hit_width(bench)[hitbench.ASPECTS_DEG.index(aspect)]
        assert bool(((measured - table).abs() < 0.6).all()), (aspect, measured, table)


def test_a_set_input_reaches_the_step():
    bench = hitbench.measure("off", {"hit_radius": 16.0})
    astern = hitbench.ASPECTS_DEG.index(180)
    expected = 0.5 * (16.0 + bench.inputs["core_radius"])
    assert abs(float(hitbench.effective_radius(bench)[astern, 0]) - expected) < 0.05
    assert hitbench.parse_set(["deflector_turn_deg=30"]) == {"deflector_turn": math.radians(30.0)}
    assert {entry.name for entry in core.HIT_MODEL} >= set(bench.inputs)
