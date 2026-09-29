"""CPU tests for the kernel-width sweep's two closed forms.

Both are quoted on the figures as the summary of how wide one ship is, so a
wrong constant would be believed.
"""

import math

import pytest
import torch

from benchmarks.density_kernel_widths import half_max_radius, lit_fraction


class TestHalfMaxRadius:
    def test_it_solves_the_log1p_field_not_the_bare_gaussian(self):
        """The stored field is ``log1p`` of the Gaussian, so its half-max is not
        the Gaussian's own (``r * sqrt(2 ln 2)`` = 1.177r). Conflating the two
        would understate the kernel's reach by about 13%."""
        radius = 500.0
        assert half_max_radius(radius) == pytest.approx(1.3277 * radius, rel=1e-3)
        assert half_max_radius(radius) > radius * math.sqrt(2.0 * math.log(2.0))

    def test_the_field_really_is_half_its_peak_there(self):
        """Checked against the field's own definition rather than the algebra."""
        radius = 500.0
        distance = half_max_radius(radius)
        peak = math.log1p(1.0)
        value = math.log1p(math.exp(-(distance**2) / (2.0 * radius**2)))
        assert value == pytest.approx(peak / 2.0, rel=1e-9)

    def test_it_scales_with_the_radius(self):
        assert half_max_radius(1000.0) == pytest.approx(2.0 * half_max_radius(500.0))


class TestLitFraction:
    def test_it_counts_cells_at_or_above_the_threshold(self):
        values = torch.tensor([1.0, 0.5, 0.11, 0.09, 0.0])
        assert lit_fraction(values, 0.1) == pytest.approx(3 / 5)

    def test_an_empty_field_lights_nothing_rather_than_dividing_by_zero(self):
        assert lit_fraction(torch.zeros(10)) == 0.0

    def test_it_is_scale_free(self):
        """The threshold is a fraction of the peak, so multiplying the field by
        a constant cannot change the answer."""
        values = torch.rand(200)
        assert lit_fraction(values * 7.0) == pytest.approx(lit_fraction(values))
