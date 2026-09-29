"""CPU tests for the density audit's arithmetic and grid bookkeeping.

The measurement itself needs a checkpoint and a GPU. What is testable here is
the least-squares baseline the head is scored against, and the ring index the
per-ring plot is drawn against -- reconstructed from ring sizes, so it has to be
checked against the cell geometry it claims to describe rather than assumed.
"""

import pytest
import torch

from benchmarks.density_baselines import _deviance, _fit_linear
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


class TestPoissonDeviance:
    """The ladder's scale. A Poisson likelihood has no natural zero, so the
    deviance -- the likelihood ratio against a predictor that matched every
    count exactly -- is what makes two predictors comparable at all."""

    def test_a_perfect_rate_scores_zero(self):
        counts = torch.randint(0, 4, (200, 9)).double()
        assert _deviance(counts, counts) == pytest.approx(0.0, abs=1e-9)

    def test_it_is_non_negative_for_any_rate(self):
        counts = torch.randint(0, 4, (200, 9)).double()
        for rate in (0.01, 0.5, 3.0):
            assert _deviance(torch.full_like(counts, rate), counts) >= -1e-9

    def test_an_empty_cell_still_charges_a_claimed_rate(self):
        """``y = 0`` contributes the rate itself, so a head cannot hedge by
        spreading mass over cells that hold nothing."""
        counts = torch.zeros(50, 4, dtype=torch.float64)
        quiet = _deviance(torch.full_like(counts, 1e-6), counts)
        assert _deviance(torch.full_like(counts, 1.0), counts) > quiet

    def test_being_wrong_either_way_costs_something(self):
        counts = torch.full((50, 4), 2.0, dtype=torch.float64)
        assert _deviance(torch.full_like(counts, 1.0), counts) > 0.0
        assert _deviance(torch.full_like(counts, 4.0), counts) > 0.0


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
