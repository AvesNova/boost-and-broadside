"""CPU tests for the density audit's arithmetic and grid bookkeeping.

The measurement itself needs a checkpoint and a GPU. What is testable here is
the least-squares baseline the head is scored against, and the ring index the
per-ring plot is drawn against -- reconstructed from ring sizes, so it has to be
checked against the cell geometry it claims to describe rather than assumed.
"""

import pytest
import torch

from benchmarks.density_baselines import _fit_linear, _mse
from benchmarks.density_plots import ring_index
from boost_and_broadside.train.rl.hex_density import (
    HEX_DENSITY_RINGS,
    hex_cell_count,
    hex_cell_offsets,
)


class TestLinearBaseline:
    def test_it_recovers_a_map_that_is_exactly_linear(self):
        torch.manual_seed(0)
        features = torch.randn(400, 3, dtype=torch.float64)
        truth = torch.randn(3, 5, dtype=torch.float64)
        fitted = _fit_linear(features, features @ truth, ridge=1e-12)
        assert torch.allclose(fitted, truth, atol=1e-6)

    def test_a_duplicated_feature_does_not_make_the_solve_singular(self):
        """An intercept beside a constant feature is exactly this case, and it
        is what the ridge term is for."""
        base = torch.randn(200, 2, dtype=torch.float64)
        features = torch.cat([base, base[:, :1]], dim=1)  # column 2 duplicates column 0
        fitted = _fit_linear(features, torch.randn(200, 4, dtype=torch.float64), ridge=1e-3)
        assert torch.isfinite(fitted).all()

    def test_predicting_the_column_mean_scores_the_variance(self):
        """The bar the ladder's R2 is quoted against: a constant predictor's
        error is the target's variance, which is the honest denominator."""
        values = torch.randn(500, 7, dtype=torch.float64)
        constant = values.mean(0, keepdim=True).expand_as(values)
        assert _mse(constant, values) == pytest.approx(float(values.var(0, unbiased=False).mean()))


class TestRingIndex:
    def test_it_covers_every_cell_of_the_grid(self):
        index = ring_index(HEX_DENSITY_RINGS)
        assert index.shape[0] == hex_cell_count(HEX_DENSITY_RINGS)

    def test_each_ring_holds_the_cells_that_ring_should(self):
        index = ring_index(HEX_DENSITY_RINGS)
        assert int((index == 0).sum()) == 1
        for ring in range(1, HEX_DENSITY_RINGS + 1):
            assert int((index == ring).sum()) == 6 * ring

    def test_the_index_agrees_with_where_the_cells_actually_are(self):
        """Reconstructing from ring sizes and reading the geometry must give the
        same answer; if the grid's ordering ever changed, this is what would say
        so, and the plot's x axis would otherwise be quietly wrong."""
        index = ring_index(HEX_DENSITY_RINGS)
        radius = hex_cell_offsets(HEX_DENSITY_RINGS).norm(dim=-1)
        # A ring's cells all sit between its inner-edge and corner distances,
        # and rings are strictly ordered outward by their minimum radius.
        previous = -1.0
        for ring in range(HEX_DENSITY_RINGS + 1):
            here = radius[index == ring]
            assert float(here.min()) > previous
            previous = float(here.min())
        # The outermost ring's corners sit exactly on the playable boundary.
        assert float(radius[index == HEX_DENSITY_RINGS].max()) == pytest.approx(1.0, rel=1e-6)
